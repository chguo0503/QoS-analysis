"""UCM trace 到 DPU/QoS/SSD 的小型 Layerwise 闭环测试。"""

from concurrent.futures import Future
from copy import deepcopy
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from backends.asu_ssd.time_utils import TIME_UNITS_PER_US
from simulation_common.config_utils import load_yaml
from qos import build_queue_layout
from ucm_trace_qos_simulator import (
    UcmTraceQosSimulation,
    _configure_one_group_per_gpu_qos,
    run_configured_experiment,
    run_configured_steady_experiment,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _batch_retrieve(cid):
    """构造一条只含一个 147456-byte Entry 的合法 SQE。"""

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


def _write_two_layer_bundle(
    directory,
    gpu_count=1,
    arrival_times_ns=None,
    compute_times_ns=None,
):
    """写入少量 GPU、一块 ASU、两层的最小 trace bundle。"""

    directory = Path(directory)
    metadata = {
        "schema_version": "ucm-sqe-simulation/v1",
        "ucm_helper": {"endianness": "little"},
        # 真实模型仍是78层，这份小 trace 只回放前2层。
        "config": {"model": {"hidden_layers": 78}},
        "output_counts": {
            "retrieve_sqe_count": gpu_count * 2,
            "retrieve_entry_count": gpu_count * 2,
        },
    }
    workload = {
        "schema_version": "ucm-sqe-simulation/v1",
        "gpu_count": gpu_count,
        "asu_count": 1,
        "trace_layer_count": 2,
        "estimated_retrieve_entry_count": gpu_count * 2,
        "estimated_retrieve_payload_bytes": gpu_count * 2 * 147_456,
        "requests": [
            {
                "gpu_id": gpu_id,
                "source_request_id": f"gpu-{gpu_id:04d}-prefix-0000",
                "arrival_time_ns": (
                    arrival_times_ns[gpu_id]
                    if arrival_times_ns is not None
                    else 100_000 + gpu_id * 10_000
                ),
                "input_tokens": 100_000,
                "sampled_cached_prefix_ratio": 0.6,
                "cached_token_count": 60_000,
                "single_layer_compute_ns": (
                    compute_times_ns[gpu_id]
                    if compute_times_ns is not None
                    else 20_000
                ),
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
    # 按层排列各 GPU，保持模板时间单调。
    record_index = 1
    for layer_id, original_time_ns in enumerate((1, 999_999_999)):
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


class UcmTraceQosSimulationTests(unittest.TestCase):
    def test_steady_qos_layout_has_one_group_per_gpu(self):
        """64张GPU对应64个Group，每组四条Queue。"""

        simulation = deepcopy(load_yaml(
            PROJECT_DIR / "config" / "simulation_config.yaml"
        )["simulation"])
        qos = simulation["qos"]
        _configure_one_group_per_gpu_qos(qos, gpu_count=64)

        self.assertEqual(qos["queue_layout"]["queue_count"], 256)
        self.assertEqual(qos["queue_layout"]["group_count"], 64)
        self.assertEqual(qos["queue_layout"]["queues_per_group"], 4)
        self.assertEqual(len(qos["token_bucket"]["group_rates"]), 64)
        self.assertEqual(
            qos["token_bucket"]["queue_cir_weight_bitmap"],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            qos["scheduler"]["group_weight_bitmap"],
            [1] * 64,
        )
        self.assertEqual(
            set(qos["scheduler"]["queue_weight_bitmaps"]),
            {f"g{index}" for index in range(64)},
        )
        self.assertTrue(all(
            weights == [1, 1, 1, 1]
            for weights in qos["scheduler"][
                "queue_weight_bitmaps"
            ].values()
        ))
        layout = build_queue_layout(qos["queue_layout"])
        for gpu_index in range(64):
            queue_id = f"q{gpu_index * 4:03d}"
            self.assertEqual(
                layout.queue_to_group[queue_id],
                f"g{gpu_index}",
            )

    def test_default_config_selects_four_layer_ssd_sweep(self):
        """默认入口只运行 Baseline 和 PIR 不封顶 CIR-only。"""

        simulation = load_yaml(
            PROJECT_DIR / "config" / "simulation_config.yaml"
        )["simulation"]
        trace = simulation["ucm_trace"]
        self.assertEqual(
            trace["trace_bundle_root"],
            "../ucm-sqe-simulator/outputs/glm51_128npu_4layer_asu_1_10",
        )
        self.assertEqual(
            trace["trace_bundle_pattern"],
            "gpu_128_asu_{ssd_count}",
        )
        self.assertEqual(trace["policies"], ["baseline", "cir_only"])
        self.assertEqual(simulation["topology"]["ssd_counts"], list(range(1, 11)))
        steady = simulation["ucm_trace_steady"]
        self.assertEqual(steady["ssd_counts"], [1, 2, 3, 4, 5])
        self.assertEqual(steady["inference_count_per_gpu"], 5)
        self.assertEqual(steady["warmup_inference_count"], 0)
        self.assertEqual(
            steady["stop_mode"],
            "first_gpu_reaches_limit",
        )
        self.assertEqual(steady["parallel_workers"], 15)
        self.assertEqual(steady["queue_binding_strategy"], "one_group_per_gpu")
        self.assertEqual(
            steady["policies"],
            ["baseline", "cir_only", "utility_edf_integer_l750"],
        )

    def test_two_layer_closed_loop_and_effective_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_two_layer_bundle(directory)
            simulation_config = deepcopy(load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"])
            effective_path = Path(directory) / "effective_sqe_manifest.jsonl"
            simulation = UcmTraceQosSimulation(
                bundle_dir=directory,
                policy="baseline",
                simulation_config=simulation_config,
                effective_manifest_path=effective_path,
            )

            submitted = []
            original_submit_batch = simulation.dpu.submit_batch

            def record_submit(requests, arrival_time_us):
                submitted.append({
                    "arrival_time_us": arrival_time_us,
                    "deadline_us": requests[0]["demand_bw"]["deadline_us"],
                    "request_count": len(requests),
                })
                return original_submit_batch(requests, arrival_time_us)

            simulation.dpu.submit_batch = record_submit
            result = simulation.run()
            effective_rows = [
                json.loads(line)
                for line in effective_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        # 每层只调用一次 submit_batch，Exist 不进入 QoS。
        self.assertEqual(len(submitted), 2)
        self.assertEqual([row["request_count"] for row in submitted], [1, 1])
        self.assertEqual(result["submitted_entry_count"], 2)
        self.assertEqual(len(effective_rows), 2)
        self.assertEqual(
            [row["effective_issue_sequence"] for row in effective_rows],
            [0, 1],
        )

        gpu = result["gpus"]["0"]
        layer0, layer1 = gpu["layers"]
        self.assertAlmostEqual(submitted[0]["arrival_time_us"], 100.0)
        self.assertAlmostEqual(
            layer0["compute_start_time_ns"],
            layer0["load_completion_time_ns"],
        )
        self.assertAlmostEqual(
            layer1["effective_issue_time_ns"],
            layer0["compute_start_time_ns"],
        )
        self.assertAlmostEqual(
            layer1["compute_start_time_ns"],
            max(
                layer0["compute_done_time_ns"],
                layer1["load_completion_time_ns"],
            ),
        )
        self.assertAlmostEqual(
            gpu["first_token_time_ns"],
            layer1["compute_done_time_ns"],
        )

        # EDF 看到的 deadline 使用闭环 issue，不使用原 manifest 时间。
        for row in submitted:
            self.assertAlmostEqual(
                row["deadline_us"],
                row["arrival_time_us"] + 20.0,
            )
        self.assertLess(
            effective_rows[1]["effective_issue_time_ns"],
            effective_rows[1]["original_timestamp_ns"],
        )
        self.assertEqual(
            result["request_conservation"],
            {
                "expected_trace_layers": 2,
                "expected_trace_sqes": 2,
                "expected_trace_entries": 2,
                "expected_trace_bytes": 294_912,
                "trace_entries": 2,
                "qos_dispatched": 2,
                "ssd_completed": 2,
                "trace_bytes": 294_912,
                "qos_dispatched_bytes": 294_912,
                "ssd_completed_bytes": 294_912,
            },
        )

    def test_required_policies_can_replay_the_same_trace(self):
        policies = (
            "baseline",
            "cir_only",
        )
        simulation_config = load_yaml(
            PROJECT_DIR / "config" / "simulation_config.yaml"
        )["simulation"]
        for policy in policies:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                _write_two_layer_bundle(directory)
                result = UcmTraceQosSimulation(
                    bundle_dir=directory,
                    policy=policy,
                    simulation_config=simulation_config,
                    effective_manifest_path=(
                        Path(directory) / "effective_sqe_manifest.jsonl"
                    ),
                ).run()

                self.assertEqual(result["policy"], policy)
                self.assertEqual(result["effective_retrieve_sqe_count"], 2)
                self.assertEqual(result["submitted_entry_count"], 2)
                if policy == "cir_only":
                    self.assertEqual(
                        result["rate_control"]["strategy"],
                        "demand_aware_fcfs_cir",
                    )

    def test_two_gpu_two_inference_steady_replay(self):
        """同一 GPU 的下一次推理紧接上一次最终计算。"""

        with tempfile.TemporaryDirectory() as directory:
            _write_two_layer_bundle(directory, gpu_count=2)
            effective_path = Path(directory) / "effective_sqe_manifest.jsonl"
            simulation_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            simulation = UcmTraceQosSimulation(
                bundle_dir=directory,
                policy="utility_edf_integer_l750",
                simulation_config=simulation_config,
                effective_manifest_path=effective_path,
                inference_count_per_gpu=2,
                warmup_inference_count=1,
                queue_binding_strategy="one_group_per_gpu",
            )

            request_ids = []
            demand_group_ids = []
            original_submit_batch = simulation.dpu.submit_batch

            def record_submit(requests, arrival_time_us):
                request_ids.extend(
                    request["basic"]["request_id"] for request in requests
                )
                demand_group_ids.append(
                    requests[0]["demand_bw"]["demand_group_id"]
                )
                return original_submit_batch(requests, arrival_time_us)

            simulation.dpu.submit_batch = record_submit
            live_layout = simulation.storage_paths["SSD0"].qos.queue_layout
            self.assertEqual(live_layout.group_count, 2)
            self.assertEqual(live_layout.queues_per_group, 128)
            result = simulation.run()
            effective_rows = [
                json.loads(line)
                for line in effective_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(result["submitted_layer_count"], 8)
        self.assertEqual(
            result["request_conservation"]["expected_trace_layers"],
            8,
        )
        self.assertEqual(result["effective_retrieve_sqe_count"], 8)
        self.assertEqual(result["submitted_entry_count"], 8)
        self.assertEqual(result["submitted_bytes"], 8 * 147_456)
        self.assertEqual(result["rate_control"]["active_demand_count"], 0)
        self.assertEqual(result["rate_control"]["active_coflow_count"], 0)
        self.assertEqual(result["rate_control"]["completed_layer_count"], 8)
        self.assertFalse(
            result["rate_control"]["restore_after_final_layer"]
        )
        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertEqual(len(demand_group_ids), len(set(demand_group_ids)))
        self.assertEqual(len({row["sqe_uid"] for row in effective_rows}), 8)
        self.assertEqual(
            sorted(row["inference_index"] for row in effective_rows),
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        for gpu in result["gpus"].values():
            first, second = gpu["inferences"]
            self.assertTrue(first["is_warmup"])
            self.assertFalse(second["is_warmup"])
            self.assertEqual(
                second["arrival_time_ns"],
                first["first_token_time_ns"],
            )
            self.assertEqual(gpu["measured_inference_count"], 1)
            self.assertAlmostEqual(
                gpu["gpu_utilization_percent"],
                second["gpu_utilization_percent"],
            )

    def test_first_gpu_fifth_inference_stops_async_replay(self):
        """第一张GPU完成第五次时，不等其他GPU排空。"""

        with tempfile.TemporaryDirectory() as directory:
            _write_two_layer_bundle(
                directory,
                gpu_count=2,
                arrival_times_ns=[100_000, 110_000],
                compute_times_ns=[20_000, 5_000_000],
            )
            simulation_config = load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"]
            simulation = UcmTraceQosSimulation(
                bundle_dir=directory,
                policy="utility_edf_integer_l750",
                simulation_config=simulation_config,
                effective_manifest_path=(
                    Path(directory) / "effective_sqe_manifest.jsonl"
                ),
                inference_count_per_gpu=5,
                warmup_inference_count=0,
                queue_binding_strategy="one_group_per_gpu",
                stop_mode="first_gpu_reaches_limit",
            )
            result = simulation.run()
            gpu_states = {
                state.gpu_id: state
                for state in simulation.gpu_states.values()
            }

        completed_counts = result["completed_inference_count_by_gpu"]
        self.assertEqual(completed_counts["0"], 5)
        self.assertLess(completed_counts["1"], 5)
        self.assertEqual(result["termination"]["stopping_gpu_ids"], [0])
        self.assertFalse(result["termination"]["full_trace_completed"])
        self.assertEqual(result["termination"]["completed_gpu_count"], 1)
        self.assertGreater(result["termination"]["pending_event_count"], 0)
        self.assertTrue(result["gpus"]["1"]["has_inflight_inference"])
        self.assertGreaterEqual(result["termination"]["inflight_gpu_count"], 1)
        self.assertEqual(
            result["rate_control"]["completed_layer_count"]
            + result["rate_control"]["active_coflow_count"],
            result["submitted_layer_count"],
        )

        winner_completion_ns = result["gpus"]["0"]["inferences"][-1][
            "first_token_time_ns"
        ]
        self.assertAlmostEqual(
            result["event_loop"]["completion_time_us"] * 1_000,
            winner_completion_ns,
        )

        stop_time_ns = result["event_loop"]["completion_time_us"] * 1_000
        slow_compute = gpu_states[1].layer_results[0]
        self.assertLess(slow_compute["compute_start_time_ns"], stop_time_ns)
        self.assertGreater(slow_compute["compute_done_time_ns"], stop_time_ns)
        self.assertEqual(
            result["gpus"]["0"]["observation_compute_busy_ns"],
            5 * 2 * 20_000,
        )
        self.assertIs(
            gpu_states[0].layer_results,
            gpu_states[0].inference_results[-1].layers,
        )
        self.assertAlmostEqual(
            result["gpus"]["1"]["observation_compute_busy_ns"],
            stop_time_ns - slow_compute["compute_start_time_ns"],
        )
        self.assertIsNone(result["gpus"]["1"]["gpu_utilization_percent"])
        self.assertGreater(
            result["gpus"]["1"][
                "observation_window_gpu_utilization_percent"
            ],
            0,
        )
        self.assertEqual(result["observation_window_gpu_count"], 2)
        observation_values = [
            gpu["observation_window_gpu_utilization_percent"]
            for gpu in result["gpus"].values()
        ]
        self.assertAlmostEqual(
            result["mean_observation_window_gpu_utilization_percent"],
            sum(observation_values) / len(observation_values),
        )

        snapshot = result["request_conservation"]
        self.assertEqual(snapshot["status"], "partial_snapshot")
        self.assertEqual(
            result["submitted_sqe_count"],
            snapshot["submitted"]["sqes"],
        )
        self.assertEqual(
            result["completed_layer_sqe_count"],
            snapshot["effective_completed_sqes"],
        )
        self.assertEqual(
            snapshot["outstanding"]["total_entries"],
            snapshot["submitted"]["entries"]
            - snapshot["ssd_completed"]["entries"],
        )
        self.assertEqual(
            snapshot["outstanding"]["owned_entry_count"],
            snapshot["outstanding"]["total_entries"],
        )
        self.assertEqual(
            snapshot["outstanding"][
                "active_layer_pending_entry_count"
            ],
            snapshot["outstanding"]["total_entries"],
        )
        measured_utilization = [
            gpu["gpu_utilization_percent"]
            for gpu in result["gpus"].values()
            if gpu["gpu_utilization_percent"] is not None
        ]
        self.assertEqual(
            result["utilization_gpu_count"],
            len(measured_utilization),
        )
        self.assertAlmostEqual(
            result["mean_gpu_utilization_percent"],
            sum(measured_utilization) / len(measured_utilization),
        )

    def test_partial_storage_snapshot_before_first_completion(self):
        """已接收IO但还没SSD completion时也能输出吞吐快照。"""

        with tempfile.TemporaryDirectory() as directory:
            _write_two_layer_bundle(directory)
            simulation = UcmTraceQosSimulation(
                bundle_dir=directory,
                policy="baseline",
                simulation_config=load_yaml(
                    PROJECT_DIR / "config" / "simulation_config.yaml"
                )["simulation"],
                effective_manifest_path=(
                    Path(directory) / "effective_sqe_manifest.jsonl"
                ),
                stop_mode="first_gpu_reaches_limit",
            )
            try:
                path = simulation.storage_paths["SSD0"]
                log = simulation.dispatch_logs["SSD0"]
                log.count = 1
                log.byte_count = 147_456
                path.ssd.backend.first_submit_time = 0
                path.ssd.backend.last_completion_time = None
                simulation.event_loop.current_time = 100 * TIME_UNITS_PER_US

                storage = simulation._storage_results()["SSD0"]
            finally:
                simulation.effective_manifest.close()
                simulation.catalog.close()

        self.assertEqual(storage["completed_request_count"], 0)
        self.assertEqual(storage["completed_bytes"], 0)
        self.assertEqual(storage["active_time_us"], 100)
        self.assertAlmostEqual(
            storage["qos_dispatched_bandwidth_gb_s"],
            1.47456,
        )
        self.assertEqual(storage["ssd_completed_bandwidth_gb_s"], 0)

    def test_configured_sweep_writes_ten_rows_and_two_curves(self):
        """扫描器为每个 SSD 数量选择独立 bundle。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            simulation_config = deepcopy(load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"])
            simulation_config["topology"]["ssd_counts"] = list(range(1, 11))
            simulation_config["ucm_trace"] = {
                "trace_bundle_root": str(root / "bundles"),
                "trace_bundle_pattern": "gpu_128_asu_{ssd_count}",
                "output_dir": str(result_dir),
                "policies": ["baseline", "cir_only"],
            }
            config_path = root / "simulation.json"
            config_path.write_text(
                json.dumps({"simulation": simulation_config}),
                encoding="utf-8",
            )

            calls = []

            def fake_run(bundle_dir, policy, simulation_config, output_dir):
                ssd_count = int(Path(bundle_dir).name.rsplit("_", 1)[1])
                calls.append((ssd_count, policy, Path(output_dir)))
                base = 10 if policy == "baseline" else 20
                return {
                    "policy": policy,
                    "storage_path_count": ssd_count,
                    "mean_ttft_us": 1_000 - ssd_count,
                    "mean_gpu_utilization_percent": base + ssd_count,
                }

            with patch(
                "ucm_trace_qos_simulator.run_trace_policy",
                side_effect=fake_run,
            ):
                experiment = run_configured_experiment(config_path)

            csv_path = result_dir / "gpu_utilization_vs_ssd_count.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(len(calls), 20)
            self.assertEqual(
                calls[:2],
                [
                    (1, "baseline", result_dir / "1_ssd"),
                    (1, "cir_only", result_dir / "1_ssd"),
                ],
            )
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["ssd_count"], "1")
            self.assertEqual(
                rows[-1]["cir_only_mean_gpu_utilization_percent"],
                "30",
            )
            self.assertEqual(
                experiment["policy_labels"]["cir_only"],
                "CIR-only (PIR uncapped)",
            )
            self.assertEqual(
                experiment["topologies"]["10_ssd"]["trace_bundle_dir"],
                str(root / "bundles" / "gpu_128_asu_10"),
            )
            self.assertTrue((result_dir / "summary.json").is_file())
            self.assertGreater(
                (result_dir / "gpu_utilization_vs_ssd_count.png").stat().st_size,
                0,
            )
            self.assertGreater(
                (result_dir / "gpu_utilization_vs_ssd_count.svg").stat().st_size,
                0,
            )

    def test_configured_steady_sweep_writes_five_rows_and_three_curves(self):
        """稳态扫描为三种策略传入5次和首GPU停止模式。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "steady-results"
            simulation_config = deepcopy(load_yaml(
                PROJECT_DIR / "config" / "simulation_config.yaml"
            )["simulation"])
            simulation_config["ucm_trace_steady"] = {
                "trace_bundle_root": str(root / "bundles"),
                "trace_bundle_pattern": "gpu_64_asu_{ssd_count}",
                "output_dir": str(result_dir),
                "ssd_counts": [1, 2, 3, 4, 5],
                "inference_count_per_gpu": 5,
                "warmup_inference_count": 0,
                "stop_mode": "first_gpu_reaches_limit",
                "parallel_workers": 5,
                "queue_binding_strategy": "one_group_per_gpu",
                "policies": [
                    "baseline",
                    "cir_only",
                    "utility_edf_integer_l750",
                ],
            }
            config_path = root / "simulation.json"
            config_path.write_text(
                json.dumps({"simulation": simulation_config}),
                encoding="utf-8",
            )

            calls = []
            executor_workers = []

            class InlineExecutor:
                """在当前进程执行，便于测试记录15个点。"""

                def __init__(self, max_workers):
                    executor_workers.append(max_workers)

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    return False

                def submit(self, function, *arguments):
                    future = Future()
                    try:
                        future.set_result(function(*arguments))
                    except Exception as error:
                        future.set_exception(error)
                    return future

            def fake_run(
                bundle_dir,
                policy,
                simulation_config,
                output_dir,
                **steady,
            ):
                ssd_count = int(Path(bundle_dir).name.rsplit("_", 1)[1])
                calls.append((ssd_count, policy, steady))
                policy_index = (
                    simulation_config["ucm_trace_steady"]["policies"].index(
                        policy
                    )
                )
                return {
                    "policy": policy,
                    "mean_ttft_us": 1_000 - ssd_count,
                    "mean_gpu_utilization_percent": (
                        10 * (policy_index + 1) + ssd_count
                    ),
                    "mean_observation_window_gpu_utilization_percent": (
                        40 + 5 * policy_index + ssd_count
                    ),
                }

            with patch(
                "ucm_trace_qos_simulator.run_trace_policy",
                side_effect=fake_run,
            ), patch(
                "ucm_trace_qos_simulator.ProcessPoolExecutor",
                InlineExecutor,
            ):
                experiment = run_configured_steady_experiment(config_path)

            csv_path = (
                result_dir / "steady_gpu_utilization_vs_ssd_count.csv"
            )
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            completed_csv_path = (
                result_dir
                / "steady_completed_only_gpu_utilization_vs_ssd_count.csv"
            )
            with completed_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                completed_rows = list(csv.DictReader(stream))

            self.assertEqual(len(calls), 15)
            self.assertEqual(executor_workers, [5])
            self.assertEqual(calls[0][2], {
                "inference_count_per_gpu": 5,
                "warmup_inference_count": 0,
                "queue_binding_strategy": "one_group_per_gpu",
                "stop_mode": "first_gpu_reaches_limit",
            })
            self.assertEqual(len(rows), 5)
            self.assertEqual(
                rows[-1][
                    "utility_edf_integer_l750_"
                    "mean_observation_window_gpu_utilization_percent"
                ],
                "55",
            )
            self.assertEqual(
                completed_rows[-1][
                    "utility_edf_integer_l750_mean_gpu_utilization_percent"
                ],
                "35",
            )
            self.assertEqual(
                experiment["policy_labels"]["utility_edf_integer_l750"],
                "Utility+EDF",
            )
            self.assertEqual(
                experiment["stop_mode"],
                "first_gpu_reaches_limit",
            )
            self.assertEqual(
                experiment["maximum_measured_inference_count_per_gpu"],
                5,
            )
            self.assertEqual(
                experiment["artifacts"]["completed_only_utilization_csv"],
                str(completed_csv_path),
            )
            self.assertGreater(
                (
                    result_dir
                    / "steady_gpu_utilization_vs_ssd_count.png"
                ).stat().st_size,
                0,
            )
            self.assertGreater(
                (
                    result_dir
                    / "steady_gpu_utilization_vs_ssd_count.svg"
                ).stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()
