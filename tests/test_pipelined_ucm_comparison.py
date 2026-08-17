"""Focused regression tests for the bounded N+1 UCM pipeline."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from analysis_tools.run_pipelined_ucm_comparison import (
    ContinuousPipelinedUcmSimulation,
    PipelinedUcmSimulation,
    _clamp_duration_delta_ns,
)
from qos import build_queue_layout
from simulation_common.config_utils import load_yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _batch_retrieve(cid):
    """Build one legal BatchRetrieve SQE containing one 147456-byte entry."""

    raw = bytearray(64 + 36)
    header = [0] * 16
    header[0] = (cid << 16) | (3 << 14) | 0x46
    header[1] = 100
    header[8] = 36
    header[9] = 1 << 24
    header[10] = 1
    struct.pack_into("<16I", raw, 0, *header)

    entry = [0] * 9
    entry[5] = 0x1000 * cid
    entry[7] = (0x12 << 24) | 147_456
    entry[8] = (0x40 << 24) | 0x3456
    struct.pack_into("<9I", raw, 64, *entry)
    raw[68:76] = f"key{cid:05d}".encode("ascii")
    return bytes(raw)


def _write_trace_bundle(
    directory,
    *,
    gpu_count,
    arrival_times_ns,
    compute_times_ns,
    layer_count,
):
    """Write the minimal multi-GPU, single-ASU bundle used by these tests."""

    directory = Path(directory)
    metadata = {
        "schema_version": "ucm-sqe-simulation/v1",
        "ucm_helper": {"endianness": "little"},
        "config": {"model": {"hidden_layers": 78}},
        "output_counts": {
            "retrieve_sqe_count": gpu_count * layer_count,
            "retrieve_entry_count": gpu_count * layer_count,
        },
    }
    workload = {
        "schema_version": "ucm-sqe-simulation/v1",
        "gpu_count": gpu_count,
        "asu_count": 1,
        "trace_layer_count": layer_count,
        "estimated_retrieve_entry_count": gpu_count * layer_count,
        "estimated_retrieve_payload_bytes": (
            gpu_count * layer_count * 147_456
        ),
        "requests": [
            {
                "gpu_id": gpu_id,
                "source_request_id": f"gpu-{gpu_id:04d}-prefix-0000",
                "arrival_time_ns": arrival_times_ns[gpu_id],
                "input_tokens": 100_000,
                "sampled_cached_prefix_ratio": 0.6,
                "cached_token_count": 60_000,
                "single_layer_compute_ns": compute_times_ns[gpu_id],
            }
            for gpu_id in range(gpu_count)
        ],
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (directory / "workload_summary.json").write_text(
        json.dumps(workload),
        encoding="utf-8",
    )

    raw_parts = [bytes(64)]
    records = [{
        "sqe_uid": "exist",
        "phase": "prefix_query",
        "opcode": "Exist",
        "timestamp_ns": 100_000,
        "gpu_id": 0,
        "source_request_id": "gpu-0000-prefix-0000",
        "layer_id": None,
        "target_asu_id": 0,
        "batch_number": 1,
        "descriptor_bytes": 64,
        "payload_bytes": 0,
        "record_index": 0,
        "raw_offset": 0,
        "raw_length": 64,
    }]
    raw_offset = 64
    record_index = 1
    original_times_ns = [
        1 if layer_id == 0 else layer_id * 999_999_999
        for layer_id in range(layer_count)
    ]
    for layer_id, original_time_ns in enumerate(original_times_ns):
        for gpu_id in range(gpu_count):
            raw = _batch_retrieve(record_index)
            records.append({
                "sqe_uid": (
                    f"retrieve-gpu-{gpu_id:04d}-layer-{layer_id:02d}"
                ),
                "phase": "layer_retrieve",
                "opcode": "BatchRetrieve",
                "timestamp_ns": original_time_ns,
                "gpu_id": gpu_id,
                "source_request_id": f"gpu-{gpu_id:04d}-prefix-0000",
                "layer_id": layer_id,
                "target_asu_id": 0,
                "batch_number": 1,
                "descriptor_bytes": len(raw),
                "payload_bytes": 147_456,
                "record_index": record_index,
                "raw_offset": raw_offset,
                "raw_length": len(raw),
            })
            raw_parts.append(raw)
            raw_offset += len(raw)
            record_index += 1

    (directory / "raw_sqe.bin").write_bytes(b"".join(raw_parts))
    (directory / "sqe_manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class PipelinedUcmSimulationTests(unittest.TestCase):
    def test_busy_window_clamps_only_rounding_noise(self):
        duration_ns = 1_000_000_000.0
        self.assertEqual(
            _clamp_duration_delta_ns(duration_ns + 1e-7, duration_ns),
            duration_ns,
        )
        self.assertEqual(_clamp_duration_delta_ns(-1e-7, duration_ns), 0.0)
        with self.assertRaises(RuntimeError):
            _clamp_duration_delta_ns(duration_ns + 1.0, duration_ns)
        with self.assertRaises(RuntimeError):
            _clamp_duration_delta_ns(-1.0, duration_ns)

    def _write_eight_template_bundle(self, directory):
        self._write_distinct_template_bundle(
            directory,
            gpu_count=8,
            layer_count=2,
        )

    def _write_distinct_template_bundle(
        self,
        directory,
        *,
        gpu_count,
        layer_count,
    ):
        _write_trace_bundle(
            directory,
            gpu_count=gpu_count,
            arrival_times_ns=[gpu_id * 1_000 for gpu_id in range(gpu_count)],
            compute_times_ns=[
                20_000 + gpu_id * 1_000 for gpu_id in range(gpu_count)
            ],
            layer_count=layer_count,
        )
        workload_path = Path(directory) / "workload_summary.json"
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        for gpu_id, row in enumerate(workload["requests"]):
            row["input_tokens"] = 100_000 + gpu_id
            row["sampled_cached_prefix_ratio"] = 0.50 + gpu_id / 100
            row["cached_token_count"] = 50_000 + gpu_id
        workload_path.write_text(
            json.dumps(workload),
            encoding="utf-8",
        )

    def test_finite_64_gpu_three_inferences_measures_only_index_one(self):
        """Exact requested finite shape keeps the drain out of metrics."""

        with tempfile.TemporaryDirectory() as directory:
            self._write_distinct_template_bundle(
                directory,
                gpu_count=64,
                layer_count=4,
            )
            base_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            simulation = PipelinedUcmSimulation(
                bundle_dir=directory,
                policy="baseline",
                simulation_config=deepcopy(base_config),
                inference_count=3,
                measured_indices=(1,),
                rotation_stride=29,
                expected_layer_count=4,
                # The CLI unit is microseconds: 50 ms == 50,000 us.
                initial_arrival_jitter_max_us=50_000,
            )

            layout = build_queue_layout(
                simulation.config["qos"]["queue_layout"]
            )
            self.assertEqual(layout.group_count, 64)
            for state in simulation.gpu_states.values():
                slot0 = simulation.dpu.binding.slot_bindings[
                    (state.p_node_id, "SSD0", 0)
                ]
                slot1 = simulation.dpu.binding.slot_bindings[
                    (state.p_node_id, "SSD0", 1)
                ]
                self.assertNotEqual(slot0, slot1)
                self.assertEqual(
                    layout.queue_to_group[slot0],
                    layout.queue_to_group[slot1],
                )

            summary = simulation.run(max_events=2_000_000)

        self.assertEqual(summary["gpu_count"], 64)
        self.assertEqual(summary["layer_count"], 4)
        self.assertEqual(summary["inference_count_per_gpu"], 3)
        self.assertEqual(summary["measured_inference_indices"], [1])
        self.assertEqual(summary["completed_inference_count"], 64 * 3)
        self.assertTrue(summary["all_gpus_complete"])
        self.assertTrue(summary["request_conservation"]["passed"])
        self.assertEqual(
            summary["request_conservation"]["expected"]["layers"],
            64 * 3 * 4,
        )
        self.assertEqual(
            summary["initial_arrival"][
                "configured_effective_jitter_max_ns"
            ],
            50_000_000,
        )
        self.assertLessEqual(
            summary["initial_arrival"]["sample_arrival_span_ns"],
            50_000_000,
        )

        manual_utilization = []
        for gpu in summary["gpus"].values():
            self.assertEqual(gpu["completed_inference_order"], [0, 1, 2])
            self.assertEqual(gpu["measured_inference_indices"], [1])
            self.assertEqual(len(gpu["inferences"]), 3)
            self.assertEqual(len(set(gpu["input_token_sequence"])), 3)
            self.assertEqual(len(set(gpu["cache_ratio_sequence"])), 3)
            inference0, inference1, inference2 = gpu["inferences"]
            self.assertEqual(len(inference1["layers"]), 4)
            self.assertEqual(
                inference1["logical_arrival_time_ns"],
                inference0["completion_time_ns"],
            )
            self.assertEqual(
                inference2["logical_arrival_time_ns"],
                inference1["completion_time_ns"],
            )
            self.assertLess(
                inference2["prefetch_issue_time_ns"],
                inference2["logical_arrival_time_ns"],
            )
            # Only index-2 Layer 0 may overlap the measured inference.  Its
            # remaining layers cannot be submitted until index 1 completes.
            self.assertLess(
                inference2["layers"][0]["effective_issue_time_ns"],
                inference1["completion_time_ns"],
            )
            for layer in inference2["layers"][1:]:
                self.assertGreaterEqual(
                    layer["effective_issue_time_ns"],
                    inference1["completion_time_ns"],
                )
            manual_utilization.append(
                inference1["compute_time_ns"]
                / inference1["ttft_ns"]
                * 100
            )

        self.assertAlmostEqual(
            summary["mean_gpu_utilization_percent"],
            sum(manual_utilization) / 64,
        )
        self.assertEqual(
            summary["layer0_prefetch"]["eligible_inference_count"],
            64,
        )
        self.assertEqual(
            summary["metric_scope"]["layer0_prefetch"],
            "measured_inference_indices_only",
        )

    def test_finite_128_gpu_uses_exactly_two_queues_and_drains(self):
        """128 GPUs is the tight two-Queue-per-Group layout boundary."""

        with tempfile.TemporaryDirectory() as directory:
            self._write_distinct_template_bundle(
                directory,
                gpu_count=128,
                layer_count=4,
            )
            base_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            for policy in ("baseline", "cir_only"):
                with self.subTest(policy=policy):
                    simulation = PipelinedUcmSimulation(
                        bundle_dir=directory,
                        policy=policy,
                        simulation_config=deepcopy(base_config),
                        inference_count=3,
                        measured_indices=(1,),
                        rotation_stride=29,
                        expected_layer_count=4,
                        initial_arrival_jitter_max_us=50_000,
                    )
                    layout = build_queue_layout(
                        simulation.config["qos"]["queue_layout"]
                    )
                    self.assertEqual(layout.group_count, 128)
                    self.assertEqual(layout.queues_per_group, 2)
                    self.assertEqual(
                        simulation.dpu.binding.queues_per_group,
                        2,
                    )
                    for state in simulation.gpu_states.values():
                        slot0 = simulation.dpu.binding.slot_bindings[
                            (state.p_node_id, "SSD0", 0)
                        ]
                        slot1 = simulation.dpu.binding.slot_bindings[
                            (state.p_node_id, "SSD0", 1)
                        ]
                        self.assertEqual(
                            {slot0, slot1},
                            set(layout.group_queues[
                                layout.queue_to_group[slot0]
                            ]),
                        )
                    summary = simulation.run(max_events=4_000_000)

                    self.assertEqual(summary["gpu_count"], 128)
                    self.assertEqual(summary["completed_inference_count"], 384)
                    self.assertEqual(
                        summary["request_conservation"]["expected"]["layers"],
                        128 * 3 * 4,
                    )
                    self.assertTrue(summary["request_conservation"]["passed"])
                    self.assertEqual(summary["event_loop"]["pending_event_count"], 0)
                    self.assertEqual(
                        summary["layer0_prefetch"]["eligible_inference_count"],
                        128,
                    )
                    for gpu in summary["gpus"].values():
                        self.assertEqual(
                            gpu["completed_inference_order"],
                            [0, 1, 2],
                        )
                        self.assertEqual(
                            gpu["measured_inference_indices"],
                            [1],
                        )
                        self.assertEqual(
                            len(set(gpu["input_token_sequence"])),
                            3,
                        )
                        self.assertEqual(
                            len(set(gpu["cache_ratio_sequence"])),
                            3,
                        )
                    if policy == "baseline":
                        self.assertIsNone(summary["rate_control"])
                    else:
                        self.assertEqual(
                            summary["rate_control"]["active_demand_count"],
                            0,
                        )

    def test_template_validation_requires_each_field_to_change(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_eight_template_bundle(directory)
            workload_path = Path(directory) / "workload_summary.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            for row in workload["requests"]:
                row["input_tokens"] = 100_000
            workload_path.write_text(json.dumps(workload), encoding="utf-8")
            base_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            with self.assertRaisesRegex(ValueError, "distinct input lengths"):
                PipelinedUcmSimulation(
                    bundle_dir=directory,
                    policy="baseline",
                    simulation_config=deepcopy(base_config),
                    inference_count=3,
                    measured_indices=(1,),
                    rotation_stride=3,
                    expected_layer_count=2,
                )

    def test_baseline_and_cir_use_two_contexts_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_eight_template_bundle(directory)
            base_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]

            for policy in ("baseline", "cir_only"):
                with self.subTest(policy=policy):
                    simulation = PipelinedUcmSimulation(
                        bundle_dir=directory,
                        policy=policy,
                        simulation_config=deepcopy(base_config),
                        inference_count=5,
                        measured_indices=(1, 2, 3),
                        rotation_stride=3,
                        expected_layer_count=2,
                    )

                    # Both fixed slots are different queues inside the same
                    # per-GPU group on every SSD.
                    layout = build_queue_layout(
                        simulation.config["qos"]["queue_layout"]
                    )
                    for state in simulation.gpu_states.values():
                        for storage_target_id in simulation.storage_target_ids:
                            slot0 = simulation.dpu.binding.slot_bindings[
                                (state.p_node_id, storage_target_id, 0)
                            ]
                            slot1 = simulation.dpu.binding.slot_bindings[
                                (state.p_node_id, storage_target_id, 1)
                            ]
                            self.assertNotEqual(slot0, slot1)
                            self.assertEqual(
                                layout.queue_to_group[slot0],
                                layout.queue_to_group[slot1],
                            )

                    summary = simulation.run(max_events=1_000_000)
                    self.assertTrue(summary["all_gpus_complete"])
                    self.assertTrue(
                        summary["request_conservation"]["passed"]
                    )
                    self.assertEqual(
                        summary["completed_inference_count"],
                        8 * 5,
                    )
                    if policy == "baseline":
                        self.assertIsNone(summary["rate_control"])
                    else:
                        self.assertEqual(
                            summary["rate_control"]["active_demand_count"],
                            0,
                        )

                    for gpu in summary["gpus"].values():
                        self.assertEqual(
                            gpu["completed_inference_order"],
                            [0, 1, 2, 3, 4],
                        )
                        self.assertEqual(
                            [
                                item["queue_slot"]
                                for item in gpu["inferences"]
                            ],
                            [0, 1, 0, 1, 0],
                        )
                        self.assertEqual(
                            len(set(gpu["template_gpu_sequence"])),
                            5,
                        )
                        self.assertEqual(
                            len(set(zip(
                                gpu["input_token_sequence"],
                                gpu["cache_ratio_sequence"],
                            ))),
                            5,
                        )

                        previous_compute_done = None
                        for inference_index, inference in enumerate(
                            gpu["inferences"]
                        ):
                            if inference_index > 0:
                                previous = gpu["inferences"][
                                    inference_index - 1
                                ]
                                self.assertEqual(
                                    inference["activation_time_ns"],
                                    previous["completion_time_ns"],
                                )
                                self.assertEqual(
                                    inference["logical_arrival_time_ns"],
                                    previous["completion_time_ns"],
                                )
                                self.assertLess(
                                    inference["prefetch_issue_time_ns"],
                                    inference["activation_time_ns"],
                                )
                            for layer in inference["layers"]:
                                if previous_compute_done is not None:
                                    self.assertGreaterEqual(
                                        layer["compute_start_time_ns"],
                                        previous_compute_done,
                                    )
                                self.assertGreaterEqual(
                                    layer["compute_start_time_ns"],
                                    layer["load_completion_time_ns"],
                                )
                                previous_compute_done = layer[
                                    "compute_done_time_ns"
                                ]

    def test_continuous_mode_uses_one_common_window_then_drains(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_eight_template_bundle(directory)
            base_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            simulation = ContinuousPipelinedUcmSimulation(
                bundle_dir=directory,
                policy="baseline",
                simulation_config=deepcopy(base_config),
                steady_window_us=300,
                settling_guard_us=50,
                rotation_stride=3,
                expected_layer_count=2,
                initial_arrival_jitter_max_us=50,
            )
            summary = simulation.run(max_events=1_000_000)

            self.assertEqual(
                summary["initial_arrival"][
                    "configured_effective_jitter_max_ns"
                ],
                50_000,
            )
            self.assertLessEqual(
                summary["initial_arrival"]["sample_arrival_span_ns"],
                50_000,
            )

            self.assertEqual(
                summary["measurement_end_time_ns"]
                - summary["measurement_start_time_ns"],
                300_000,
            )
            self.assertTrue(summary["source_continued_through_measurement"])
            self.assertTrue(summary["new_lookahead_admissions_stopped_at_t1"])
            self.assertTrue(summary["drain_complete"])
            self.assertTrue(summary["request_conservation"]["passed"])
            self.assertEqual(summary["event_loop"]["pending_event_count"], 0)
            paired = summary["paired_post_warmup_inference"]
            self.assertEqual(paired["sample_count"], 8)
            self.assertTrue(paired["all_templates_once"])
            self.assertTrue(paired["all_completed_before_measurement_end"])
            self.assertEqual(
                paired["layer0_ready_before_activation_percent"],
                100,
            )

            snapshots = summary["measurement_snapshots"]
            completion_delta = 0
            for gpu_id, start in snapshots["start"]["gpus"].items():
                end = snapshots["end"]["gpus"][gpu_id]
                self.assertGreaterEqual(start["completed_inference_count"], 1)
                self.assertGreater(
                    end["completed_inference_count"],
                    start["completed_inference_count"],
                )
                completion_delta += (
                    end["completed_inference_count"]
                    - start["completed_inference_count"]
                )
                busy_delta = end["compute_busy_ns"] - start["compute_busy_ns"]
                self.assertEqual(
                    summary["gpus"][gpu_id]["measurement_compute_busy_ns"],
                    busy_delta,
                )

            self.assertEqual(
                summary["completed_inference_count_in_window"],
                completion_delta,
            )
            self.assertAlmostEqual(
                summary[
                    "mean_inference_throughput_per_gpu_per_second"
                ],
                completion_delta / 8 / 0.0003,
            )

            self.assertGreater(
                sum(
                    row["completed_bytes"]
                    for row in snapshots["end"]["storage_paths"].values()
                ),
                sum(
                    row["completed_bytes"]
                    for row in snapshots["start"]["storage_paths"].values()
                ),
            )
            self.assertGreater(summary["completed_inference_count"], 8 * 5)
            for gpu in summary["gpus"].values():
                sequence = gpu["template_gpu_sequence"]
                self.assertEqual(len(set(sequence[:5])), 5)
                self.assertEqual(sequence[5], sequence[0])
                for inference in gpu["inferences"]:
                    self.assertGreaterEqual(
                        inference["activation_time_ns"],
                        summary["measurement_start_time_ns"],
                    )
                    self.assertLess(
                        inference["completion_time_ns"],
                        summary["measurement_end_time_ns"],
                    )


if __name__ == "__main__":
    unittest.main()
