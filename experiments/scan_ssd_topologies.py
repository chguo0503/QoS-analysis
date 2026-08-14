#!/usr/bin/env python3
"""Run the configurable multi-GPU, multi-SSD QoS topology scan."""

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from backends.asu_ssd.time_utils import time_to_us  # noqa: E402
from experiments.compare_fcfs_cir import (  # noqa: E402
    TOKEN_CONFIG_FILE,
    WRR_CONFIG_FILE,
)
from llm_workload.layer_request import GPU_ESTIMATE  # noqa: E402
from qos_ssd_simulator import JointSimulation  # noqa: E402


DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR
    / "experiments"
    / "results"
    / "batch1_2x_flops_topology_scan"
    / "summary.json"
)
POLICIES = ("baseline", "demand_aware_fcfs_cir")


class CountOnlyAppendLog:
    """Count appended records without retaining their per-request dictionaries."""

    def __init__(self):
        """Function: create an empty count-only record sink.

        Purpose: prevent completed-request and NAND-event diagnostics from
        accumulating millions of dictionaries during a summary-only scan.

        Input: none.

        Output: none; initializes the record count to zero.
        """
        self.count = 0

    def append(self, record):
        """Function: consume one diagnostic record.

        Purpose: preserve the exact append call and its ordering cost while
        retaining only the aggregate count required for conservation checks.

        Input: one completed-request or NAND-service record.

        Output: none; increments the count and discards the dictionary.
        """
        self.count += 1

    def __len__(self):
        """Function: return how many records were appended.

        Purpose: expose the same count operation as a normal Python list.

        Input: none.

        Output: integer number of consumed records.
        """
        return self.count


class DispatchAggregateLog:
    """Keep QoS dispatch aggregates while discarding individual request records."""

    def __init__(self):
        """Function: create an empty dispatch aggregate.

        Purpose: retain request/byte and CIR/EXCESS totals needed by the scan,
        without keeping 7.3 million mutable request dictionaries in memory.

        Input: none.

        Output: none; initializes all counters to zero.
        """
        self.count = 0
        self.byte_count = 0
        self.cir_count = 0
        self.excess_count = 0

    def append(self, request):
        """Function: aggregate one request after successful SSD submission.

        Purpose: replace only result retention; the request has already passed
        the normal WRR, token, FCP-capacity and backend submission path.

        Input: the ordinary dispatched QoS request dictionary.

        Output: none; updates count, bytes and rate-class counters.
        """
        self.count += 1
        self.byte_count += request["size_bytes"]
        if request["qos_rate_class"] == "CIR":
            self.cir_count += 1
        else:
            self.excess_count += 1

    def __len__(self):
        """Function: return the exact successful dispatch count.

        Purpose: keep `dispatch_index = len(log) + 1` unchanged in QoS.

        Input: none.

        Output: integer number of successfully dispatched requests.
        """
        return self.count


def parse_arguments():
    """Function: parse topology-scan command-line options.

    Purpose: provide the requested 64/1/4/2,3,4 defaults while allowing a
    small pilot to reuse the identical execution path.

    Input: process command line.

    Output: parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Scan exact-batched ASU SSD counts for two QoS policies.",
    )
    parser.add_argument("--gpu-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer-count", type=int, default=4)
    parser.add_argument("--ssd-counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--input-token-min", type=int, default=100_000)
    parser.add_argument("--input-token-max", type=int, default=200_000)
    parser.add_argument("--hit-ratio-min", type=float, default=0.50)
    parser.add_argument("--hit-ratio-max", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=6103)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--merge-inputs",
        type=Path,
        nargs="+",
        default=None,
        help="merge completed per-topology summaries without running simulations",
    )
    return parser.parse_args()


def nearest_rank_p95(values):
    """Function: calculate an interpolation-free nearest-rank P95.

    Purpose: keep topology summaries comparable with the existing FCFS-CIR
    experiment and avoid library-specific percentile interpolation.

    Input: one non-empty sequence of numeric values.

    Output: the value at ``ceil(0.95 * N) - 1`` after sorting.
    """
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def gpu_utilization_percent(inference):
    """Function: calculate one GPU's compute utilization during inference.

    Purpose: measure the fraction of TTFT spent performing modeled GPU
    computation rather than waiting at a layer barrier for SSD reads.

    Input: one completed inference containing compute-only TTFT and actual TTFT.

    Output: GPU compute utilization as a percentage in the model.
    """
    return (
        inference["compute_only_ttft_us"]
        / inference["ttft_us"]
        * 100
    )


def build_simulation(arguments, ssd_count, policy):
    """Function: construct one requested topology and policy from empty state.

    Purpose: ensure every point uses the same seed, workload range, LLM batch,
    four-layer plan, stable-random placement, queue binding and exact
    32-command backend.

    Input: parsed arguments, SSD count and rate-control policy name.

    Output: a not-yet-run ``JointSimulation`` instance.
    """
    return JointSimulation(
        binding_strategy_name="balanced_exclusive",
        rate_control_strategy_name=policy,
        simulation_config_override={
            "start_time_us": 0,
            "topology": {
                "gpu_count": arguments.gpu_count,
                "storage_path_count": ssd_count,
            },
            "workload_generation": {
                "inference_count_per_gpu": 1,
                "random_seed": arguments.seed,
                "input_tokens_range": [
                    arguments.input_token_min,
                    arguments.input_token_max,
                ],
                "prefill_layer_hit_ratio_range": [
                    arguments.hit_ratio_min,
                    arguments.hit_ratio_max,
                ],
                "unique_across_gpus": True,
                "inter_inference_gap_us": 0,
            },
        },
        workload_defaults_override={
            "first_layer_index": 0,
            "last_layer_index": arguments.layer_count - 1,
            "arrival_time_us": 0,
            "batch_size": arguments.batch_size,
            "placement": {
                # 每个Block根据固定seed和request_id稳定随机选盘；
                # 两种QoS策略因此仍使用完全相同的Block→SSD映射。
                "strategy": "random",
                "allowed_storage_targets": "all",
                "random_seed": arguments.seed,
            },
        },
        backend_config_override={
            "execution_mode": "batched_exact",
            "exact_batch_max_commands": 32,
            # 正式扫描只输出层、GPU和SSD完成指标；关闭
            # 这两项高频诊断不会跳过任何六级时序计算。
            "collect_stage_peak_statistics": False,
            "collect_nand_service_events": False,
        },
        token_config_file=TOKEN_CONFIG_FILE,
        scheduler_config_file=WRR_CONFIG_FILE,
    )


def install_summary_only_logs(simulation):
    """Function: replace only high-volume result-retention containers.

    Purpose: keep the real request objects through Placement, DPU, QoS, FCP and
    completion callbacks, then release them after use instead of retaining a
    second diagnostic copy for the whole experiment.

    Input: a newly constructed joint simulation before its first event runs.

    Output: mapping from SSD ID to its dispatch aggregate log.
    """
    dispatch_logs = {}
    for storage_target_id, storage_path in simulation.storage_paths.items():
        dispatch_log = DispatchAggregateLog()
        storage_path.qos.dispatched_requests = dispatch_log
        storage_path.ssd.backend.completed_requests = CountOnlyAppendLog()
        storage_path.ssd.backend.nand_service_events = CountOnlyAppendLog()
        dispatch_logs[storage_target_id] = dispatch_log
    return dispatch_logs


def collect_inference_results(simulation):
    """Function: flatten completed per-GPU inference results in GPU order.

    Purpose: derive layer timing and TTFT summaries without invoking the normal
    result builder that intentionally includes per-request QoS/SSD diagnostics.

    Input: a joint simulation whose GPUs have all completed.

    Output: list containing exactly one completed inference per GPU.
    """
    return [
        simulation.completed_inference_results[gpu_id][0]
        for gpu_id in simulation.gpu_workload_sequences
    ]


def summarize_run(simulation, dispatch_logs, wall_time_seconds):
    """Function: build compact physical and performance metrics for one run.

    Purpose: report layer deadlines, TTFT, SSD tails, event count and strict
    request/byte conservation without materializing detailed result lists.

    Input: completed simulation, per-SSD dispatch logs and wall-clock duration.

    Output: policy summary including average and minimum GPU utilization.
    """
    inferences = collect_inference_results(simulation)
    signed_deltas_us = []
    actual_reads_us = []
    for inference in inferences:
        for layer in inference["layers"]:
            layer_start_us = layer["layer_start_time_us"]
            actual_read_us = layer["io_completion_time_us"] - layer_start_us
            target_window_us = layer["compute_done_time_us"] - layer_start_us
            actual_reads_us.append(actual_read_us)
            signed_deltas_us.append(actual_read_us - target_window_us)

    expected_request_count = sum(
        inference["request_count"] for inference in inferences
    )
    expected_completed_count = sum(
        inference["completed_request_count"] for inference in inferences
    )
    qos_request_count = sum(log.count for log in dispatch_logs.values())
    qos_byte_count = sum(log.byte_count for log in dispatch_logs.values())
    ssd_request_count = sum(
        len(path.ssd.backend.completed_requests)
        for path in simulation.storage_paths.values()
    )
    ssd_byte_count = sum(
        path.ssd.backend.completed_bytes()
        for path in simulation.storage_paths.values()
    )
    expected_byte_count = sum(
        inference["request_count"] * inference["block_size_bytes"]
        for inference in inferences
    )
    conservation = {
        expected_request_count,
        expected_completed_count,
        qos_request_count,
        ssd_request_count,
    }
    if len(conservation) != 1:
        raise RuntimeError(f"request conservation failed: {conservation}")
    if len({expected_byte_count, qos_byte_count, ssd_byte_count}) != 1:
        raise RuntimeError("byte conservation failed")

    last_completion_by_ssd_us = {
        storage_target_id: time_to_us(path.ssd.backend.last_completion_time)
        for storage_target_id, path in simulation.storage_paths.items()
    }
    ttft_values_us = [inference["ttft_us"] for inference in inferences]
    gpu_utilizations_percent = [
        gpu_utilization_percent(inference) for inference in inferences
    ]
    rate_control = simulation.dpu.statistics()["rate_control"]
    if rate_control is not None and rate_control["active_demand_count"] != 0:
        raise RuntimeError("demand-aware run ended with active Queue demands")

    summary = {
        "late_gpu_layer_count": sum(
            delta_us > 0 for delta_us in signed_deltas_us
        ),
        "p95_delta_us": nearest_rank_p95(signed_deltas_us),
        "worst_delta_us": max(signed_deltas_us),
        "mean_actual_read_us": sum(actual_reads_us) / len(actual_reads_us),
        "mean_ttft_us": sum(ttft_values_us) / len(ttft_values_us),
        "p95_ttft_us": nearest_rank_p95(ttft_values_us),
        "max_ttft_us": max(ttft_values_us),
        "mean_gpu_utilization_percent": (
            sum(gpu_utilizations_percent)
            / len(gpu_utilizations_percent)
        ),
        "min_gpu_utilization_percent": min(gpu_utilizations_percent),
        "last_completion_by_ssd_us": last_completion_by_ssd_us,
        "overall_last_completion_us": max(last_completion_by_ssd_us.values()),
        "request_count": expected_request_count,
        "byte_count": expected_byte_count,
        "cir_dispatch_count": sum(log.cir_count for log in dispatch_logs.values()),
        "excess_dispatch_count": sum(
            log.excess_count for log in dispatch_logs.values()
        ),
        "processed_event_count": simulation.event_loop.processed_event_count,
        "wall_time_seconds": wall_time_seconds,
    }
    return summary


def run_one(arguments, ssd_count, policy):
    """Function: execute and summarize one topology-policy point.

    Purpose: isolate state and peak memory between points while printing enough
    progress for a multi-hour command to remain observable.

    Input: parsed arguments, SSD count and policy name.

    Output: compact policy summary including GPU utilization.
    """
    print(
        f"START ssd_count={ssd_count} policy={policy}",
        flush=True,
    )
    simulation = build_simulation(arguments, ssd_count, policy)
    dispatch_logs = install_summary_only_logs(simulation)
    started_at = perf_counter()
    simulation.event_loop.run_until(simulation._all_gpus_complete)
    wall_time_seconds = perf_counter() - started_at
    summary = summarize_run(
        simulation,
        dispatch_logs,
        wall_time_seconds,
    )
    print(
        f"DONE ssd_count={ssd_count} policy={policy} "
        f"wall={wall_time_seconds:.3f}s mean_ttft={summary['mean_ttft_us']:.3f}us",
        flush=True,
    )
    return summary


def summarize_pair(baseline_summary, demand_aware_summary):
    """Function: compare mean GPU utilization between the two policies.

    Purpose: express Demand-aware's average utilization change in percentage
    points, avoiding the magnitude-blind improved/worsened GPU count.

    Input: compact Baseline and Demand-aware policy summaries.

    Output: Demand-aware minus Baseline mean utilization percentage points.
    """
    return {
        "mean_gpu_utilization_gain_percentage_points": (
            demand_aware_summary["mean_gpu_utilization_percent"]
            - baseline_summary["mean_gpu_utilization_percent"]
        ),
    }


def write_summary(summary, output_file):
    """Function: atomically replace the compact JSON checkpoint.

    Purpose: preserve all completed topology points if a later long-running
    point is interrupted, while never creating per-request CSV or manifests.

    Input: current summary dictionary and destination path.

    Output: none; writes UTF-8 JSON through a sibling temporary file.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_file.replace(output_file)


def run_scan(arguments):
    """Function: run all requested SSD counts and policies sequentially.

    Purpose: bound peak memory on a 22 GiB host and checkpoint after each
    completed policy rather than holding multiple detailed simulations.

    Input: parsed scan arguments.

    Output: final compact experiment summary dictionary.
    """
    summary = {
        "experiment": {
            "gpu_count": arguments.gpu_count,
            "batch_size": arguments.batch_size,
            # 从LLM配置读取实际建模算力，保证结果能够明确区分
            # 本次100-TFLOPS实验与此前50-TFLOPS历史结果。
            "effective_compute_tflops": GPU_ESTIMATE[
                "effective_compute_tflops"
            ],
            "layer_count": arguments.layer_count,
            "ssd_counts": arguments.ssd_counts,
            "policies": arguments.policies,
            "seed": arguments.seed,
            "input_tokens_range": [
                arguments.input_token_min,
                arguments.input_token_max,
            ],
            "hit_ratio_range": [
                arguments.hit_ratio_min,
                arguments.hit_ratio_max,
            ],
            "placement_strategy": "random",
            "queue_binding_strategy": "balanced_exclusive",
            "backend_execution_mode": "batched_exact",
            "backend_batch_commands": 32,
        },
        "topologies": {},
    }
    for ssd_count in arguments.ssd_counts:
        topology_key = f"{ssd_count}_ssd"
        topology_summary = {}
        summary["topologies"][topology_key] = topology_summary
        for policy in arguments.policies:
            policy_summary = run_one(
                arguments,
                ssd_count,
                policy,
            )
            topology_summary[policy] = policy_summary
            write_summary(summary, arguments.output)
        if all(policy in topology_summary for policy in POLICIES):
            topology_summary["paired"] = summarize_pair(
                topology_summary["baseline"],
                topology_summary["demand_aware_fcfs_cir"],
            )
            write_summary(summary, arguments.output)
    return summary


def merge_summaries(input_files):
    """Function: merge independently completed SSD-count checkpoints.

    Purpose: allow 2/3/4-SSD scans to run in separate processes, then create the
    same single final summary without copying or recomputing any metric.

    Input: paths to compact JSON files produced by this script.

    Output: one summary containing all topology entries in numeric SSD order.
    """
    partial_summaries = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in input_files
    ]
    merged = {
        "experiment": dict(partial_summaries[0]["experiment"]),
        "topologies": {},
    }
    for partial in partial_summaries:
        merged["topologies"].update(partial["topologies"])
    ordered_keys = sorted(
        merged["topologies"],
        key=lambda key: int(key.split("_", 1)[0]),
    )
    merged["topologies"] = {
        key: merged["topologies"][key] for key in ordered_keys
    }
    merged["experiment"]["ssd_counts"] = [
        int(key.split("_", 1)[0]) for key in ordered_keys
    ]
    return merged


def main():
    """Function: execute the topology scan and print its final JSON.

    Purpose: provide one reproducible command for the requested experiment and
    make terminal output identical to the persisted final summary.

    Input: process command line.

    Output: none; writes and prints the compact summary.
    """
    arguments = parse_arguments()
    summary = (
        merge_summaries(arguments.merge_inputs)
        if arguments.merge_inputs is not None
        else run_scan(arguments)
    )
    write_summary(summary, arguments.output)
    serialized = json.dumps(summary, indent=2, ensure_ascii=False)
    print(serialized)


if __name__ == "__main__":
    main()
