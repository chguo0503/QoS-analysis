"""Verify that exact batched SSD execution loses no timing precision."""

from copy import deepcopy
import random
import unittest

from backends.asu_ssd.simulator import SSDSimulator, load_ssd_config
from experiments.compare_fcfs_cir import TOKEN_CONFIG_FILE, WRR_CONFIG_FILE
from qos_ssd_simulator import run_joint_simulation


def build_backend_config(
    execution_mode,
    batch_commands=32,
    collect_diagnostics=True,
):
    """Function: build an isolated SSD configuration for one execution mode.

    Purpose: make the detailed and exact-batched runs differ only in how the
    same six-stage timing equations are evaluated.

    Input: execution-mode name, maximum local 4 KiB batch length and whether
    high-frequency diagnostic records are collected.

    Output: a new backend configuration dictionary.
    """
    config = deepcopy(load_ssd_config())
    config["execution_mode"] = execution_mode
    config["exact_batch_max_commands"] = batch_commands
    config["collect_stage_peak_statistics"] = collect_diagnostics
    config["collect_nand_service_events"] = collect_diagnostics
    return config


def run_nonblocking_stream(
    execution_mode,
    requests,
    batch_commands=32,
    collect_diagnostics=True,
):
    """Function: submit a deterministic variable-size stream to one SSD.

    Purpose: exercise FCP capacity, all downstream capacities, odd-chunk
    padding, staggered arrivals and non-blocking retries without involving QoS.

    Input: execution mode, ``(request, release_us)`` pairs, batch length and
    diagnostic-collection flag.

    Output: accepted timestamps and the finalized SSD result.
    """
    simulator = SSDSimulator(
        build_backend_config(
            execution_mode,
            batch_commands,
            collect_diagnostics,
        ),
        storage_target_id="SSD0",
    )
    accepted_times_us = []
    current_time_us = 0.0

    for request, release_time_us in requests:
        current_time_us = max(current_time_us, release_time_us)
        while (
            simulator.next_event_time_us() is not None
            and simulator.next_event_time_us() < current_time_us
        ):
            simulator.run_next_event()

        # A rejected probe is not needed to discover the next FCP capacity
        # boundary: both modes expose availability through the same read-only
        # interface. This keeps retry diagnostics independent of event granularity.
        while not simulator.can_accept_at_us(current_time_us):
            current_time_us = simulator.run_next_event()

        input_result = simulator.try_input_at_us(
            request,
            requested_time_us=current_time_us,
        )
        if not input_result["accepted"]:
            raise AssertionError("SSD rejected after reporting available input")
        accepted_times_us.append(input_result["accepted_time_us"])

    result = simulator.end()
    result.pop("execution_mode")
    return accepted_times_us, result


def build_stress_requests():
    """Function: generate a fixed stream covering important size boundaries.

    Purpose: exceed the NAND, BCP and FCP capacities while mixing exact and
    padded 4 KiB/8 KiB requests, so equality is not limited to an idle SSD.

    Input: none; randomness uses the fixed seed 6103.

    Output: one hundred ``(request, release_us)`` pairs.
    """
    random_generator = random.Random(6103)
    release_time_us = 0
    sizes = [
        1,
        4096,
        4097,
        8192,
        12288,
        144 * 1024,
        288 * 1024,
        511 * 4096,
    ]
    requests = []
    for request_index in range(100):
        release_time_us += random_generator.choice([0, 0, 0, 1, 3, 7])
        size_bytes = random_generator.choice(sizes)
        request = {
            "request_id": f"request_{request_index}",
            "queue_id": f"q{request_index % 64:03d}",
            "size_bytes": size_bytes,
            "dispatch_time_us": release_time_us,
        }
        requests.append((request, release_time_us))
    return requests


def normalize_joint_result(result):
    """Function: remove fields that describe execution machinery, not hardware.

    Purpose: compare every workload, QoS, NAND and layer timestamp exactly while
    allowing the optimized mode to process fewer Python event-loop callbacks.

    Input: one complete joint-simulation result.

    Output: a deep-copied result without mode name and callback count.
    """
    normalized = deepcopy(result)
    normalized["event_loop"].pop("processed_event_count")
    for path_result in normalized["storage_paths"].values():
        path_result["ssd"].pop("execution_mode")
    return normalized


def run_small_joint_simulation(execution_mode, policy):
    """Function: run a two-layer multi-GPU/multi-SSD integration case.

    Purpose: prove exact equality of QoS dispatch classes/times, SSD acceptance
    and completion times, GPU layer barriers and final summaries under both
    baseline and CIR-first scheduling.

    Input: backend execution mode and DPU rate-control policy.

    Output: the complete joint-simulation result.
    """
    return run_joint_simulation(
        binding_strategy="balanced_exclusive",
        rate_control_strategy=policy,
        simulation_config_override={
            "start_time_us": 0,
            "topology": {
                "gpu_count": 6,
                "storage_path_count": 2,
            },
            "workload_generation": {
                "inference_count_per_gpu": 1,
                "random_seed": 6103,
                "input_tokens_range": [10_000, 11_000],
                "prefill_layer_hit_ratio_range": [0.50, 0.90],
                "unique_across_gpus": True,
                "inter_inference_gap_us": 0,
            },
        },
        workload_defaults_override={
            "first_layer_index": 0,
            "last_layer_index": 1,
            "arrival_time_us": 0,
            "batch_size": 1,
            "placement": {
                "strategy": "balanced_round_robin",
                "allowed_storage_targets": "all",
                "random_seed": 6103,
            },
        },
        backend_config_override={
            "execution_mode": execution_mode,
            "exact_batch_max_commands": 32,
        },
        token_config_file=TOKEN_CONFIG_FILE,
        scheduler_config_file=WRR_CONFIG_FILE,
    )


class BatchedExactBackendTests(unittest.TestCase):
    """Compare exact batched calculations with the detailed event reference."""

    def test_variable_size_stream_is_bit_exact(self):
        """Function: run the same capacity-stressing stream in both modes.

        Purpose: require exact Python equality for acceptance, NAND starts,
        request completions, bytes, waiting statistics and six-stage peaks.

        Input: the fixed stress stream generated by this module.

        Output: unittest equality assertions.
        """
        requests = build_stress_requests()
        detailed = run_nonblocking_stream("detailed", requests)
        batched = run_nonblocking_stream("batched_exact", requests)
        self.assertEqual(detailed, batched)

    def test_batch_length_does_not_change_any_result(self):
        """Function: repeat one stream with several local batch boundaries.

        Purpose: prove that 32 is only a loop grouping limit and cannot change
        timestamps, FIFO order, backpressure or completion statistics.

        Input: the first 24 requests of the fixed stress stream.

        Output: unittest equality assertions for batch lengths 1, 7, 32 and 64.
        """
        requests = build_stress_requests()[:24]
        reference = run_nonblocking_stream("detailed", requests)
        for batch_commands in (1, 7, 32, 64):
            with self.subTest(batch_commands=batch_commands):
                candidate = run_nonblocking_stream(
                    "batched_exact",
                    requests,
                    batch_commands=batch_commands,
                )
                self.assertEqual(reference, candidate)

    def test_joint_qos_and_layer_results_are_bit_exact(self):
        """Function: compare complete joint results for both DPU policies.

        Purpose: ensure reducing SSD internal callbacks cannot alter QoS WRR,
        CIR/EXCESS classification, Block placement, GPU barriers or completion.

        Input: baseline and demand-aware FCFS CIR integration runs.

        Output: exact equality plus a strict reduction in Python event count.
        """
        for policy in ("baseline", "demand_aware_fcfs_cir"):
            with self.subTest(policy=policy):
                detailed = run_small_joint_simulation("detailed", policy)
                batched = run_small_joint_simulation("batched_exact", policy)
                self.assertEqual(
                    normalize_joint_result(detailed),
                    normalize_joint_result(batched),
                )
                self.assertLess(
                    batched["event_loop"]["processed_event_count"],
                    detailed["event_loop"]["processed_event_count"],
                )

    def test_disabling_diagnostics_does_not_change_physical_results(self):
        """Function: run exact batching with diagnostics enabled and disabled.

        Purpose: prove the formal scan's summary-only switches affect only NAND
        plotting records and occupancy peaks, never acceptance or completion.

        Input: the first 24 requests of the fixed stress stream.

        Output: exact equality after removing the two intentionally disabled
        diagnostic fields.
        """
        requests = build_stress_requests()[:24]
        with_diagnostics = run_nonblocking_stream(
            "batched_exact",
            requests,
            collect_diagnostics=True,
        )
        without_diagnostics = run_nonblocking_stream(
            "batched_exact",
            requests,
            collect_diagnostics=False,
        )
        for _, result in (with_diagnostics, without_diagnostics):
            result.pop("nand_service_events")
            result.pop("stage_statistics")
        self.assertEqual(with_diagnostics, without_diagnostics)


if __name__ == "__main__":
    unittest.main()
