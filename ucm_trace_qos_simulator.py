#!/usr/bin/env python3
"""重放 UCM raw SQE trace，并计算 Layerwise Prefix TTFT。

这个入口不再生成 LLM 请求，也不重新做 KV Placement。UCM trace
已经决定了每个 Entry 属于哪张 GPU、哪一层和哪块 ASU/SSD。
仿真只在一层所有 Entry 完成后启动该层计算，并在同一时刻
下发下一层 SQE。
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
import json
import math
from pathlib import Path
import re
from time import perf_counter

from DPU import (
    DPURequestGateway,
    DemandAwareFCFSCIRController,
    UtilityEDFController,
    build_queue_binding_strategy,
)
from DPU.ucm_trace import UcmTraceBundle, parse_batch_retrieve
from backends.asu_ssd import SSDSimulator
from backends.asu_ssd.time_utils import (
    TIME_UNITS_PER_US,
    time_to_us,
)
from discrete_simulation import EventLoop
from qos import build_qos_simulator, build_queue_layout
from simulation_common.aggregate_logs import (
    CountOnlyAppendLog,
    DispatchAggregateLog,
)
from simulation_common.config_utils import load_yaml
from simulation_common.storage_path import StoragePath


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
GPU_COMPLETION_PRIORITY = 5
GPU_COMPUTE_START_PRIORITY = 10
TIME_UNITS_PER_NS = TIME_UNITS_PER_US // 1_000
POLICY_LABELS = {
    "baseline": "Baseline",
    "cir_only": "CIR-only (PIR uncapped)",
    "utility_edf_integer_l750": "Utility+EDF",
}
UTILIZATION_PLOT_STEM = "gpu_utilization_vs_ssd_count"
STEADY_UTILIZATION_PLOT_STEM = "steady_gpu_utilization_vs_ssd_count"
STEADY_COMPLETED_UTILIZATION_CSV_STEM = (
    "steady_completed_only_gpu_utilization_vs_ssd_count"
)
COMPLETED_UTILIZATION_METRIC = "mean_gpu_utilization_percent"
OBSERVATION_UTILIZATION_METRIC = (
    "mean_observation_window_gpu_utilization_percent"
)


def _configure_one_group_per_gpu_qos(qos_config, gpu_count):
    """把256条Queue重排为每张GPU一个Group。"""

    layout = qos_config["queue_layout"]
    queues_per_group = layout["queue_count"] // gpu_count
    layout["group_count"] = gpu_count
    layout["queues_per_group"] = queues_per_group

    group_ids = [
        f"{layout['group_id_prefix']}{gpu_index}"
        for gpu_index in range(gpu_count)
    ]
    queue_weights = [1] * queues_per_group
    token_bucket = qos_config["token_bucket"]
    token_bucket["group_rates"] = [
        {"group_id": group_id, "cir_gb_s": 0}
        for group_id in group_ids
    ]
    token_bucket["queue_cir_weight_bitmap"] = list(queue_weights)

    scheduler = qos_config["scheduler"]
    scheduler["group_weight_bitmap"] = [1] * gpu_count
    scheduler["queue_weight_bitmaps"] = {
        group_id: list(queue_weights)
        for group_id in group_ids
    }


def _time_to_ns(time_value):
    """把仿真内部时间转为 ns。

    SSD 流水线可以产生小数 ns，因此返回值可以是小数。
    """

    nanoseconds = time_value / TIME_UNITS_PER_NS
    if time_value % TIME_UNITS_PER_NS == 0:
        return int(nanoseconds)
    return round(nanoseconds, 6)


def _nearest_rank_p95(values):
    """按 nearest-rank 定义计算 P95。"""

    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


class UcmLayerCatalog:
    """只索引每层在 manifest 中的位置，不常驻数百万 Entry。"""

    def __init__(self, bundle):
        self.bundle = bundle
        self.layer_ranges = {}
        self.layer_ids_by_source = {}
        self.storage_target_ids = set()
        self._build_index()
        self.manifest_stream = self.bundle.manifest_path.open("rb")
        self.raw_stream = self.bundle.raw_path.open("rb")

    def _build_index(self):
        """每个逻辑层只保存 manifest 的起止字节位置。"""

        current_key = None
        finished_keys = set()
        with self.bundle.manifest_path.open("rb") as stream:
            while True:
                line_start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                if (
                    record.get("opcode") == "Exist"
                    or record.get("phase") == "prefix_query"
                ):
                    continue

                key = (
                    record["source_request_id"],
                    int(record["layer_id"]),
                )
                if key != current_key:
                    if current_key is not None:
                        finished_keys.add(current_key)
                    if key in finished_keys:
                        raise ValueError(
                            f"layer {key!r} is not contiguous in manifest"
                        )
                    self.layer_ranges[key] = [line_start, stream.tell()]
                    current_key = key
                else:
                    self.layer_ranges[key][1] = stream.tell()
                self.storage_target_ids.add(
                    f"SSD{int(record['target_asu_id'])}"
                )

        for source_request_id, layer_id in self.layer_ranges:
            self.layer_ids_by_source.setdefault(
                source_request_id,
                [],
            ).append(layer_id)
        for source_request_id, layer_ids in self.layer_ids_by_source.items():
            self.layer_ids_by_source[source_request_id] = tuple(
                sorted(layer_ids)
            )

    def load_layer(self, source_request_id, layer_id):
        """在层真正发出时才解析它的 raw SQE Entry。"""

        start, end = self.layer_ranges[(source_request_id, layer_id)]
        self.manifest_stream.seek(start)
        records = []
        while self.manifest_stream.tell() < end:
            record = json.loads(self.manifest_stream.readline())
            raw_sqe = self.bundle._read_raw_record(
                self.raw_stream,
                record,
            )
            parsed = parse_batch_retrieve(raw_sqe)
            if parsed["batch_number"] != int(record["batch_number"]):
                raise ValueError(
                    f"{record['sqe_uid']}: manifest/raw batch_number mismatch"
                )
            if parsed["payload_bytes"] != int(record["payload_bytes"]):
                raise ValueError(
                    f"{record['sqe_uid']}: manifest/raw payload mismatch"
                )
            records.append((record, parsed))

        submission = self.bundle._build_layer_submission(records)
        return submission, records

    def close(self):
        self.manifest_stream.close()
        self.raw_stream.close()


@dataclass
class ActiveLayer:
    """一张 GPU 当前正在等待的一层 SQE。"""

    inference_index: int
    runtime_source_request_id: str
    layer_id: int
    issue_time: int
    submission: object
    pending_entry_count: int
    sqes: dict
    completion_by_storage_target: dict = field(default_factory=dict)


@dataclass
class CompletedInference:
    """一次已完成的四层推理。"""

    inference_index: int
    arrival_time: int
    completion_time: int
    layers: list


@dataclass
class GpuTraceState:
    """一张 GPU 的 Layerwise 计算与 IO 状态。"""

    source_request_id: str
    gpu_id: int
    arrival_time: int
    single_layer_compute_time: int
    layer_ids: tuple
    workload: dict
    active_layer: ActiveLayer | None = None
    inference_index: int = 0
    inference_arrival_time: int | None = None
    previous_compute_done_time: int | None = None
    completed: bool = False
    first_token_time: int | None = None
    layer_results: list = field(default_factory=list)
    inference_results: list = field(default_factory=list)


class UcmTraceQosSimulation:
    """用现有 DPU/QoS/SSD 数据面回放一份 UCM trace。"""

    def __init__(
        self,
        bundle_dir,
        policy,
        simulation_config,
        effective_manifest_path,
        inference_count_per_gpu=1,
        warmup_inference_count=0,
        queue_binding_strategy="balanced_exclusive",
        stop_mode="all_gpus_complete",
    ):
        self.policy = policy
        self.config = deepcopy(simulation_config)
        self.inference_count_per_gpu = int(inference_count_per_gpu)
        self.warmup_inference_count = int(warmup_inference_count)
        self.queue_binding_strategy = queue_binding_strategy
        self.stop_mode = stop_mode
        self.bundle = UcmTraceBundle(bundle_dir)
        self.catalog = UcmLayerCatalog(self.bundle)
        self.effective_manifest_path = Path(effective_manifest_path)
        self.effective_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.effective_manifest = self.effective_manifest_path.open(
            "w",
            encoding="utf-8",
        )
        self.effective_record_index = 0
        self.effective_issue_sequence = 0

        self.start_time_us = self.config["start_time_us"]
        self.event_loop = EventLoop(
            start_time=int(self.start_time_us * TIME_UNITS_PER_US)
        )
        self.request_owner = {}
        self.completed_source_requests = set()
        self.submitted_layer_count = 0
        self.submitted_sqe_count = 0
        self.submitted_entry_count = 0
        self.submitted_bytes = 0

        self.gpu_states = self._build_gpu_states()
        if self.queue_binding_strategy == "one_group_per_gpu":
            _configure_one_group_per_gpu_qos(
                self.config["qos"],
                len(self.gpu_states),
            )
        self.storage_target_ids = [
            f"SSD{asu_id}"
            for asu_id in range(int(self.bundle.workload_summary["asu_count"]))
        ]
        if not self.catalog.storage_target_ids.issubset(
            self.storage_target_ids
        ):
            raise ValueError("manifest target ASU is outside workload topology")
        self._build_data_path()
        self._schedule_initial_layers()

    def _build_gpu_states(self):
        """直接使用 workload_summary 中的到达时间和每层计算时间。"""

        states = {}
        for workload in self.bundle.workload_summary["requests"]:
            source_request_id = workload["source_request_id"]
            layer_ids = self.catalog.layer_ids_by_source[source_request_id]
            states[source_request_id] = GpuTraceState(
                source_request_id=source_request_id,
                gpu_id=int(workload["gpu_id"]),
                arrival_time=(
                    int(workload["arrival_time_ns"]) * TIME_UNITS_PER_NS
                ),
                single_layer_compute_time=(
                    int(workload["single_layer_compute_ns"])
                    * TIME_UNITS_PER_NS
                ),
                layer_ids=layer_ids,
                workload=workload,
                inference_arrival_time=(
                    int(workload["arrival_time_ns"]) * TIME_UNITS_PER_NS
                ),
            )
        expected_layers = next(iter(states.values())).layer_ids
        for state in states.values():
            if state.layer_ids != expected_layers:
                raise ValueError("all GPUs must contain the same trace layers")
        return states

    def _build_rate_controller(self, capacity_by_storage_target):
        """根据策略名创建 DPU 速率控制器。Baseline 不创建控制器。"""

        if self.policy == "baseline":
            return None
        if self.policy in {"cir_only", "demand_aware_fcfs_cir"}:
            return DemandAwareFCFSCIRController(
                capacity_by_storage_target
            )

        match = re.fullmatch(
            r"utility_edf_(integer|power)_l([1-9][0-9]*)",
            self.policy,
        )
        if match is None:
            raise ValueError(f"unsupported UCM trace policy {self.policy!r}")
        layer_count = len(next(iter(self.gpu_states.values())).layer_ids)
        return UtilityEDFController(
            capacity_bytes_per_second_by_storage_target=(
                capacity_by_storage_target
            ),
            score_mode=match.group(1),
            deadline_allowance_us=int(match.group(2)),
            compute_layer_count=layer_count,
            # 连续推理期间不恢复 uncapped。这样下一次
            # Layer 0 会等新的控制周期，不会偷走 EXCESS。
            restore_after_final_layer=(
                self.inference_count_per_gpu == 1
            ),
        )

    def _build_data_path(self):
        """每个 ASU 创建一套独立 QoS+SSD，然后统一接入 DPU。"""

        qos_config = self.config["qos"]
        backend_config = self.config["ssd"]["backend"]
        queue_layout = build_queue_layout(qos_config["queue_layout"])

        self.storage_paths = {}
        self.dispatch_logs = {}
        for storage_target_id in self.storage_target_ids:
            qos = build_qos_simulator(
                qos_config=qos_config,
                start_time_us=self.start_time_us,
                queue_layout=queue_layout,
            )
            ssd = SSDSimulator(
                backend_config=deepcopy(backend_config),
                completion_sink=self._on_storage_complete,
                storage_target_id=storage_target_id,
            )
            path = StoragePath(
                storage_target_id=storage_target_id,
                qos=qos,
                ssd=ssd,
                event_loop=self.event_loop,
            )

            # 数百万条请求只保存计数和字节，不保存逐条日志。
            dispatch_log = DispatchAggregateLog()
            qos.dispatched_requests = dispatch_log
            ssd.backend.completed_requests = CountOnlyAppendLog()
            ssd.backend.nand_service_events = CountOnlyAppendLog()
            self.dispatch_logs[storage_target_id] = dispatch_log
            self.storage_paths[storage_target_id] = path

        queue_ids_by_storage_target = {
            storage_target_id: queue_layout.queue_order
            for storage_target_id in self.storage_target_ids
        }
        p_node_ids = [
            f"P{state.gpu_id}"
            for state in sorted(
                self.gpu_states.values(),
                key=lambda state: state.gpu_id,
            )
        ]
        binding = build_queue_binding_strategy(
            strategy_name=self.queue_binding_strategy,
            p_node_ids=p_node_ids,
            queue_ids_by_storage_target=queue_ids_by_storage_target,
        )
        capacity_by_storage_target = {
            storage_target_id: backend_config["nand"][
                "read_bandwidth_bytes_per_second"
            ]
            for storage_target_id in self.storage_target_ids
        }
        rate_controller = self._build_rate_controller(
            capacity_by_storage_target
        )
        self.dpu = DPURequestGateway(
            queue_ids_by_storage_target=queue_ids_by_storage_target,
            queue_binding_strategy=binding,
            request_sink=self._route_qos_request,
            qos_interfaces_by_storage_target={
                storage_target_id: path.qos
                for storage_target_id, path in self.storage_paths.items()
            },
            rate_controller=rate_controller,
        )
        for path in self.storage_paths.values():
            path.start()

    def _route_qos_request(self, request):
        return self.storage_paths[request["storage_target_id"]].input(request)

    def _schedule_initial_layers(self):
        """每张 GPU 都在自己的随机到达时刻发出 Layer 0。"""

        for state in self.gpu_states.values():
            first_layer_id = state.layer_ids[0]
            self.event_loop.schedule_at(
                event_time=state.arrival_time,
                priority=GPU_COMPUTE_START_PRIORITY,
                event_name=f"trace-layer-issue:{state.gpu_id}:{first_layer_id}",
                callback=partial(
                    self._issue_layer,
                    state.source_request_id,
                    first_layer_id,
                ),
            )

    def _issue_layer(self, source_request_id, layer_id, event_time):
        """解析一个完整逻辑层，然后只调用一次 submit_batch。"""

        state = self.gpu_states[source_request_id]
        inference_index = state.inference_index
        runtime_source_request_id = source_request_id
        if self.inference_count_per_gpu > 1:
            runtime_source_request_id = (
                f"{source_request_id}:inference:{inference_index:04d}"
            )
        submission, records = self.catalog.load_layer(
            source_request_id,
            layer_id,
        )
        issue_time_us = time_to_us(event_time)
        inference_arrival_time_us = time_to_us(
            state.inference_arrival_time
        )

        # manifest 原时间只用于对照。EDF deadline 必须从本次
        # 闭环真实发出时刻重新计算，否则后续层会全部过期。
        for request in submission.requests:
            basic = request["basic"]
            template_request_id = basic["request_id"]
            template_sqe_uid, entry_index = template_request_id.rsplit(
                ":entry:",
                1,
            )
            runtime_sqe_uid = template_sqe_uid
            if self.inference_count_per_gpu > 1:
                runtime_sqe_uid = (
                    f"{template_sqe_uid}:inference:{inference_index:04d}"
                )
            basic["request_id"] = (
                f"{runtime_sqe_uid}:entry:{entry_index}"
            )
            demand = request["demand_bw"]
            demand["demand_group_id"] = (
                f"{runtime_source_request_id}:layer:{layer_id}"
            )
            demand["inference_arrival_time_us"] = (
                inference_arrival_time_us
            )
            demand["deadline_us"] = (
                issue_time_us + demand["service_window_us"]
            )

        sqes = {}
        for record, parsed in records:
            runtime_sqe_uid = record["sqe_uid"]
            if self.inference_count_per_gpu > 1:
                runtime_sqe_uid = (
                    f"{record['sqe_uid']}:inference:{inference_index:04d}"
                )
            sqes[runtime_sqe_uid] = {
                "record": record,
                "runtime_sqe_uid": runtime_sqe_uid,
                "issue_sequence": self.effective_issue_sequence,
                "pending_entry_count": parsed["batch_number"],
                "completion_time": None,
            }
            self.effective_issue_sequence += 1
        active_layer = ActiveLayer(
            inference_index=inference_index,
            runtime_source_request_id=runtime_source_request_id,
            layer_id=layer_id,
            issue_time=event_time,
            submission=submission,
            pending_entry_count=len(submission.requests),
            sqes=sqes,
        )
        state.active_layer = active_layer

        for request in submission.requests:
            request_id = request["basic"]["request_id"]
            sqe_uid = request_id.rsplit(":entry:", 1)[0]
            self.request_owner[request_id] = (
                source_request_id,
                sqe_uid,
            )

        self.submitted_layer_count += 1
        self.submitted_sqe_count += len(records)
        self.submitted_entry_count += len(submission.requests)
        self.submitted_bytes += submission.batch_total_bytes
        self.dpu.submit_batch(
            requests=submission.requests,
            arrival_time_us=issue_time_us,
        )

    def _on_storage_complete(self, completion):
        """只保存正在读取层的 Entry 归属，完成后立即删除。"""

        request_id = completion["request_id"]
        source_request_id, sqe_uid = self.request_owner.pop(request_id)
        state = self.gpu_states[source_request_id]
        active_layer = state.active_layer
        completion_time = self.event_loop.current_time

        active_layer.pending_entry_count -= 1
        storage_target_id = completion["storage_target_id"]
        active_layer.completion_by_storage_target[storage_target_id] = max(
            completion_time,
            active_layer.completion_by_storage_target.get(
                storage_target_id,
                completion_time,
            ),
        )
        sqe = active_layer.sqes[sqe_uid]
        sqe["pending_entry_count"] -= 1
        sqe["completion_time"] = completion_time

        if active_layer.pending_entry_count == 0:
            self._finish_layer_read(state, completion_time)

    def _write_effective_layer(self, state, active_layer):
        """一层到齐后按原 record_index 输出每条 SQE 的真实时间。"""

        for sqe in sorted(
            active_layer.sqes.values(),
            key=lambda value: int(value["record"]["record_index"]),
        ):
            record = sqe["record"]
            row = {
                "effective_record_index": self.effective_record_index,
                "effective_issue_sequence": sqe["issue_sequence"],
                "inference_index": active_layer.inference_index,
                "original_record_index": int(record["record_index"]),
                "sqe_uid": sqe["runtime_sqe_uid"],
                "template_sqe_uid": record["sqe_uid"],
                "source_request_id": active_layer.runtime_source_request_id,
                "template_source_request_id": state.source_request_id,
                "gpu_id": state.gpu_id,
                "layer_id": active_layer.layer_id,
                "target_asu_id": int(record["target_asu_id"]),
                "batch_number": int(record["batch_number"]),
                "payload_bytes": int(record["payload_bytes"]),
                "raw_offset": int(record["raw_offset"]),
                "raw_length": int(record["raw_length"]),
                "original_timestamp_ns": int(record["timestamp_ns"]),
                "effective_issue_time_ns": _time_to_ns(
                    active_layer.issue_time
                ),
                "effective_completion_time_ns": _time_to_ns(
                    sqe["completion_time"]
                ),
            }
            self.effective_manifest.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            self.effective_record_index += 1

    def _finish_layer_read(self, state, completion_time):
        """层内最后一个 Entry 完成后，建立 GPU 计算屏障。"""

        active_layer = state.active_layer
        if any(
            sqe["pending_entry_count"] != 0
            for sqe in active_layer.sqes.values()
        ):
            raise RuntimeError("layer completed before all SQEs completed")
        self._write_effective_layer(state, active_layer)
        previous_compute_done = state.previous_compute_done_time
        compute_start_time = (
            completion_time
            if previous_compute_done is None
            else max(previous_compute_done, completion_time)
        )
        state.layer_results.append({
            "inference_index": active_layer.inference_index,
            "layer_id": active_layer.layer_id,
            "original_issue_time_ns": active_layer.submission.timestamp_ns,
            "effective_issue_time_ns": _time_to_ns(
                active_layer.issue_time
            ),
            "load_completion_time_ns": _time_to_ns(completion_time),
            "load_latency_ns": _time_to_ns(
                completion_time - active_layer.issue_time
            ),
            "compute_start_time_ns": _time_to_ns(compute_start_time),
            "compute_done_time_ns": _time_to_ns(
                compute_start_time + state.single_layer_compute_time
            ),
            "entry_count": len(active_layer.submission.requests),
            "bytes": active_layer.submission.batch_total_bytes,
            "path_bytes_by_storage_target": (
                active_layer.submission.path_bytes_by_storage_target
            ),
            "completion_time_ns_by_storage_target": {
                storage_target_id: _time_to_ns(target_completion_time)
                for storage_target_id, target_completion_time in sorted(
                    active_layer.completion_by_storage_target.items()
                )
            },
        })
        state.active_layer = None

        # priority 10 让同一时刻的所有 SSD completion(priority 0)
        # 先处理完，再统一发下一层。
        self.event_loop.schedule_at(
            event_time=compute_start_time,
            priority=GPU_COMPUTE_START_PRIORITY,
            event_name=(
                f"compute-start:{state.gpu_id}:"
                f"{active_layer.inference_index}:{active_layer.layer_id}"
            ),
            callback=partial(
                self._process_compute_start,
                state.source_request_id,
                active_layer.layer_id,
            ),
        )

    def _process_compute_start(
        self,
        source_request_id,
        layer_id,
        event_time,
    ):
        """开始本层计算，并在同一时刻发出下一层 SQE。"""

        state = self.gpu_states[source_request_id]
        compute_done_time = event_time + state.single_layer_compute_time
        state.previous_compute_done_time = compute_done_time
        layer_position = state.layer_ids.index(layer_id)
        if layer_position + 1 < len(state.layer_ids):
            next_layer_id = state.layer_ids[layer_position + 1]
            self._issue_layer(
                source_request_id,
                next_layer_id,
                event_time,
            )
            return

        self.event_loop.schedule_at(
            event_time=compute_done_time,
            priority=GPU_COMPLETION_PRIORITY,
            event_name=(
                f"gpu-complete:{state.gpu_id}:{state.inference_index}"
            ),
            callback=partial(
                self._process_gpu_completion,
                source_request_id,
            ),
        )

    def _process_gpu_completion(self, source_request_id, event_time):
        """记录当前推理，并立即启动同 GPU 的下一次。"""

        state = self.gpu_states[source_request_id]
        state.inference_results.append(CompletedInference(
            inference_index=state.inference_index,
            arrival_time=state.inference_arrival_time,
            completion_time=event_time,
            layers=state.layer_results,
        ))
        if state.inference_index + 1 < self.inference_count_per_gpu:
            state.inference_index += 1
            state.inference_arrival_time = event_time
            state.previous_compute_done_time = None
            state.layer_results = []
            self._issue_layer(
                source_request_id,
                state.layer_ids[0],
                event_time,
            )
            return

        state.completed = True
        state.first_token_time = event_time
        self.completed_source_requests.add(source_request_id)

    def _all_gpus_complete(self):
        return len(self.completed_source_requests) == len(self.gpu_states)

    def _stop_condition(self):
        """按配置决定等全部GPU，还是等第一张GPU达标。"""

        if self.stop_mode == "first_gpu_reaches_limit":
            return any(
                len(state.inference_results)
                >= self.inference_count_per_gpu
                for state in self.gpu_states.values()
            )
        return self._all_gpus_complete()

    def _inference_output(self, state, completed):
        """将一次完成记录转成可持久化指标。"""

        duration = completed.completion_time - completed.arrival_time
        compute_time = (
            len(state.layer_ids) * state.single_layer_compute_time
        )
        return {
            "inference_index": completed.inference_index,
            "is_warmup": (
                completed.inference_index < self.warmup_inference_count
            ),
            "source_request_id": (
                state.source_request_id
                if self.inference_count_per_gpu == 1
                else (
                    f"{state.source_request_id}:inference:"
                    f"{completed.inference_index:04d}"
                )
            ),
            "arrival_time_ns": _time_to_ns(completed.arrival_time),
            "first_token_time_ns": _time_to_ns(completed.completion_time),
            "ttft_ns": _time_to_ns(duration),
            "ttft_us": time_to_us(duration),
            "compute_only_ttft_ns": _time_to_ns(compute_time),
            "storage_stall_ns": _time_to_ns(duration - compute_time),
            "gpu_utilization_percent": compute_time / duration * 100,
            "layers": completed.layers,
        }

    def _observation_window_metrics(self, state):
        """统计该GPU从首次arrival到全局stop的实际计算占用。"""

        stop_time_ns = _time_to_ns(self.event_loop.current_time)
        arrival_time_ns = int(state.workload["arrival_time_ns"])
        window_ns = max(0, stop_time_ns - arrival_time_ns)
        layers = [
            layer
            for completed in state.inference_results
            for layer in completed.layers
        ]
        # 达标GPU的current list就是最后一次已完成推理，不再重复计数。
        if not state.completed:
            layers.extend(state.layer_results)

        busy_ns = 0
        for layer in layers:
            compute_start_ns = max(
                arrival_time_ns,
                layer["compute_start_time_ns"],
            )
            compute_done_ns = min(
                stop_time_ns,
                layer["compute_done_time_ns"],
            )
            busy_ns += max(0, compute_done_ns - compute_start_ns)

        utilization = None
        if window_ns > 0:
            utilization = busy_ns / window_ns * 100
        return busy_ns, window_ns, utilization

    def _gpu_results(self):
        results = {}
        for state in sorted(
            self.gpu_states.values(),
            key=lambda item: item.gpu_id,
        ):
            inference_outputs = [
                self._inference_output(state, completed)
                for completed in state.inference_results
            ]
            measured = state.inference_results[
                self.warmup_inference_count:
            ]
            measured_duration = sum(
                item.completion_time - item.arrival_time
                for item in measured
            )
            measured_compute = (
                len(measured)
                * len(state.layer_ids)
                * state.single_layer_compute_time
            )
            utilization = None
            if measured:
                utilization = measured_compute / measured_duration * 100
            has_inflight_inference = (
                not state.completed
                and (
                    state.active_layer is not None
                    or bool(state.layer_results)
                    or state.inference_index > 0
                )
            )
            (
                observation_compute_busy_ns,
                observation_window_ns,
                observation_utilization,
            ) = self._observation_window_metrics(state)
            result = {
                "gpu_id": state.gpu_id,
                "template_source_request_id": state.source_request_id,
                "arrival_time_ns": int(state.workload["arrival_time_ns"]),
                "input_tokens": state.workload.get("input_tokens"),
                "cached_prefix_ratio": state.workload.get(
                    "sampled_cached_prefix_ratio"
                ),
                "cached_token_count": state.workload.get(
                    "cached_token_count"
                ),
                "single_layer_compute_ns": int(
                    state.workload["single_layer_compute_ns"]
                ),
                "layer_count": len(state.layer_ids),
                "inference_count": len(inference_outputs),
                "warmup_inference_count": self.warmup_inference_count,
                "measured_inference_count": len(measured),
                "measured_compute_time_ns": _time_to_ns(
                    measured_compute
                ),
                "measured_inference_duration_ns": _time_to_ns(
                    measured_duration
                ),
                "gpu_utilization_percent": utilization,
                "observation_compute_busy_ns": (
                    observation_compute_busy_ns
                ),
                "observation_window_ns": observation_window_ns,
                "observation_window_gpu_utilization_percent": (
                    observation_utilization
                ),
                "has_inflight_inference": has_inflight_inference,
                "inflight_inference_index": (
                    state.inference_index
                    if has_inflight_inference
                    else None
                ),
                "completed_layer_count_in_inflight_inference": (
                    len(state.layer_results)
                    if has_inflight_inference
                    else 0
                ),
                "active_layer_id": (
                    state.active_layer.layer_id
                    if state.active_layer is not None
                    else None
                ),
                "inferences": inference_outputs,
            }
            # 单次模式保留原有 GPU 摘要字段。
            if self.inference_count_per_gpu == 1 and inference_outputs:
                only = inference_outputs[0]
                result.update({
                    "source_request_id": only["source_request_id"],
                    "first_token_time_ns": only["first_token_time_ns"],
                    "ttft_ns": only["ttft_ns"],
                    "ttft_us": only["ttft_us"],
                    "compute_only_ttft_ns": only[
                        "compute_only_ttft_ns"
                    ],
                    "storage_stall_ns": only["storage_stall_ns"],
                    "layers": only["layers"],
                })
            results[str(state.gpu_id)] = result
        return results

    def _storage_results(self):
        results = {}
        full_trace_completed = self._all_gpus_complete()
        stop_time_us = time_to_us(self.event_loop.current_time)
        for storage_target_id, path in self.storage_paths.items():
            backend = path.ssd.backend
            log = self.dispatch_logs[storage_target_id]
            completed_request_count = backend.completed_requests.count
            completed_bytes = backend.completed_bytes()
            first_submit_us = (
                None
                if backend.first_submit_time is None
                else time_to_us(backend.first_submit_time)
            )
            last_completion_us = (
                None
                if backend.last_completion_time is None
                else time_to_us(backend.last_completion_time)
            )
            activity_end_us = (
                last_completion_us
                if full_trace_completed
                else stop_time_us
            )
            active_time_us = (
                0
                if first_submit_us is None
                else activity_end_us - first_submit_us
            )
            results[storage_target_id] = {
                "request_count": log.count,
                "bytes": log.byte_count,
                "completed_request_count": completed_request_count,
                "completed_bytes": completed_bytes,
                "cir_dispatch_count": log.cir_count,
                "excess_dispatch_count": log.excess_count,
                "first_submit_time_us": first_submit_us,
                "last_completion_time_us": last_completion_us,
                "active_time_us": active_time_us,
                "average_bandwidth_gb_s": (
                    0
                    if active_time_us == 0
                    else log.byte_count / active_time_us / 1_000
                ),
                "qos_dispatched_bandwidth_gb_s": (
                    0
                    if active_time_us == 0
                    else log.byte_count / active_time_us / 1_000
                ),
                "ssd_completed_bandwidth_gb_s": (
                    0
                    if active_time_us == 0
                    else completed_bytes / active_time_us / 1_000
                ),
                "ssd_blocked_attempt_count": (
                    path.ssd.blocked_attempt_count
                ),
            }
        return results

    def _rate_control_summary(self):
        statistics = self.dpu.statistics()
        control = statistics["rate_control"]
        if control is None:
            return None
        keys = (
            "strategy",
            "active_demand_count",
            "completed_demand_count_by_storage_target",
            "completed_coflow_count",
            "active_coflow_count",
            "completed_layer_count",
            "completed_layer_count_by_p_node",
            "current_inference_completed_layer_count_by_p_node",
            "restore_after_final_layer",
            "peak_assigned_cir_bytes_per_second",
            "decision_count",
            "initial_decision_count",
            "prefetch_decision_count",
            "feasibility_conflict_count",
            "selection_change_count",
            "score_mode",
            "deadline_allowance_us",
            "compute_layer_count",
        )
        return {key: control[key] for key in keys if key in control}

    def _build_summary(self, wall_time_seconds):
        gpu_results = self._gpu_results()
        storage_results = self._storage_results()
        output_counts = self.bundle.metadata["output_counts"]
        expected_entry_count = (
            int(output_counts["retrieve_entry_count"])
            * self.inference_count_per_gpu
        )
        expected_sqe_count = (
            int(output_counts["retrieve_sqe_count"])
            * self.inference_count_per_gpu
        )
        trace_layer_count = int(
            self.bundle.workload_summary["trace_layer_count"]
        )
        expected_layer_count = (
            len(gpu_results)
            * trace_layer_count
            * self.inference_count_per_gpu
        )
        workload_entry_count = (
            int(self.bundle.workload_summary[
                "estimated_retrieve_entry_count"
            ])
            * self.inference_count_per_gpu
        )
        expected_payload_bytes = (
            int(self.bundle.workload_summary[
                "estimated_retrieve_payload_bytes"
            ])
            * self.inference_count_per_gpu
        )
        measured_inferences = [
            inference
            for result in gpu_results.values()
            for inference in result["inferences"]
            if not inference["is_warmup"]
        ]
        ttft_values = [
            inference["ttft_us"] for inference in measured_inferences
        ]
        utilization_values = [
            result["gpu_utilization_percent"]
            for result in gpu_results.values()
            if result["gpu_utilization_percent"] is not None
        ]
        observation_utilization_values = [
            result["observation_window_gpu_utilization_percent"]
            for result in gpu_results.values()
            if result["observation_window_gpu_utilization_percent"]
            is not None
        ]
        qos_request_count = sum(
            result["request_count"] for result in storage_results.values()
        )
        qos_bytes = sum(
            result["bytes"] for result in storage_results.values()
        )
        ssd_request_count = sum(
            path.ssd.backend.completed_requests.count
            for path in self.storage_paths.values()
        )
        ssd_bytes = sum(
            path.ssd.backend.completed_bytes()
            for path in self.storage_paths.values()
        )
        active_layer_pending_entry_count = sum(
            state.active_layer.pending_entry_count
            for state in self.gpu_states.values()
            if state.active_layer is not None
        )
        full_trace_completed = self._all_gpus_complete()
        if workload_entry_count != expected_entry_count:
            raise RuntimeError("UCM trace workload Entry count is inconsistent")
        if full_trace_completed:
            if len({
                self.submitted_entry_count,
                qos_request_count,
                ssd_request_count,
            }) != 1:
                raise RuntimeError("UCM trace request conservation failed")
            if len({self.submitted_bytes, qos_bytes, ssd_bytes}) != 1:
                raise RuntimeError("UCM trace byte conservation failed")
            if self.submitted_entry_count != expected_entry_count:
                raise RuntimeError("UCM trace Entry count is incomplete")
            if self.submitted_sqe_count != expected_sqe_count:
                raise RuntimeError("UCM trace submitted SQE count is incomplete")
            if self.effective_record_index != expected_sqe_count:
                raise RuntimeError("UCM trace SQE count is incomplete")
            if self.submitted_layer_count != expected_layer_count:
                raise RuntimeError("UCM trace layer count is incomplete")
            if self.submitted_bytes != expected_payload_bytes:
                raise RuntimeError(
                    "UCM trace payload byte count is incomplete"
                )
            if self.request_owner:
                raise RuntimeError(
                    "UCM trace ended with active Entry ownership"
                )
        else:
            if not (
                0
                <= ssd_request_count
                <= qos_request_count
                <= self.submitted_entry_count
            ):
                raise RuntimeError("partial request snapshot is inconsistent")
            if not (
                0 <= ssd_bytes <= qos_bytes <= self.submitted_bytes
            ):
                raise RuntimeError("partial byte snapshot is inconsistent")
            outstanding_entry_count = (
                self.submitted_entry_count - ssd_request_count
            )
            if len(self.request_owner) != outstanding_entry_count:
                raise RuntimeError("partial request ownership is inconsistent")
            if active_layer_pending_entry_count != outstanding_entry_count:
                raise RuntimeError("partial active layer count is inconsistent")

        rate_control = self._rate_control_summary()
        if rate_control is not None and full_trace_completed:
            if rate_control["active_demand_count"] != 0:
                raise RuntimeError("rate controller ended with active demand")
            if rate_control.get("active_coflow_count", 0) != 0:
                raise RuntimeError("rate controller ended with active coflow")
            if (
                "completed_layer_count" in rate_control
                and rate_control["completed_layer_count"]
                != expected_layer_count
            ):
                raise RuntimeError("rate controller layer count is incomplete")
        if (
            rate_control is not None
            and not full_trace_completed
            and "completed_layer_count" in rate_control
            and (
                rate_control["completed_layer_count"]
                + rate_control.get("active_coflow_count", 0)
                != self.submitted_layer_count
            )
        ):
            raise RuntimeError("partial controller layer count is inconsistent")

        completed_inference_counts = {
            gpu_id: result["inference_count"]
            for gpu_id, result in gpu_results.items()
        }
        stopping_gpu_ids = [
            int(gpu_id)
            for gpu_id, count in completed_inference_counts.items()
            if count >= self.inference_count_per_gpu
        ]
        request_conservation = {
            "expected_trace_layers": expected_layer_count,
            "expected_trace_sqes": expected_sqe_count,
            "expected_trace_entries": expected_entry_count,
            "expected_trace_bytes": expected_payload_bytes,
            "trace_entries": self.submitted_entry_count,
            "qos_dispatched": qos_request_count,
            "ssd_completed": ssd_request_count,
            "trace_bytes": self.submitted_bytes,
            "qos_dispatched_bytes": qos_bytes,
            "ssd_completed_bytes": ssd_bytes,
        }
        if not full_trace_completed:
            request_conservation = {
                "status": "partial_snapshot",
                "target": {
                    "layers": expected_layer_count,
                    "sqes": expected_sqe_count,
                    "entries": expected_entry_count,
                    "bytes": expected_payload_bytes,
                },
                "submitted": {
                    "layers": self.submitted_layer_count,
                    "sqes": self.submitted_sqe_count,
                    "entries": self.submitted_entry_count,
                    "bytes": self.submitted_bytes,
                },
                "effective_completed_sqes": self.effective_record_index,
                "qos_dispatched": {
                    "entries": qos_request_count,
                    "bytes": qos_bytes,
                },
                "ssd_completed": {
                    "entries": ssd_request_count,
                    "bytes": ssd_bytes,
                },
                "outstanding": {
                    "before_qos_entries": (
                        self.submitted_entry_count - qos_request_count
                    ),
                    "before_qos_bytes": self.submitted_bytes - qos_bytes,
                    "in_ssd_entries": (
                        qos_request_count - ssd_request_count
                    ),
                    "in_ssd_bytes": qos_bytes - ssd_bytes,
                    "total_entries": (
                        self.submitted_entry_count - ssd_request_count
                    ),
                    "total_bytes": self.submitted_bytes - ssd_bytes,
                    "owned_entry_count": len(self.request_owner),
                    "active_layer_pending_entry_count": (
                        active_layer_pending_entry_count
                    ),
                },
            }

        return {
            "policy": self.policy,
            "trace_bundle_dir": str(self.bundle.bundle_dir),
            "effective_manifest": str(self.effective_manifest_path),
            "gpu_count": len(gpu_results),
            "storage_path_count": len(storage_results),
            "layer_count_per_gpu": len(
                next(iter(self.gpu_states.values())).layer_ids
            ),
            "submitted_layer_count": self.submitted_layer_count,
            "inference_count_per_gpu": self.inference_count_per_gpu,
            "warmup_inference_count": self.warmup_inference_count,
            "measured_inference_count_per_gpu": (
                self.inference_count_per_gpu - self.warmup_inference_count
                if full_trace_completed
                else None
            ),
            "completed_inference_count": sum(
                completed_inference_counts.values()
            ),
            "completed_inference_count_by_gpu": (
                completed_inference_counts
            ),
            "queue_binding_strategy": self.queue_binding_strategy,
            "stop_mode": self.stop_mode,
            "submitted_sqe_count": self.submitted_sqe_count,
            "completed_layer_sqe_count": self.effective_record_index,
            "effective_retrieve_sqe_count": self.effective_record_index,
            "submitted_entry_count": self.submitted_entry_count,
            "submitted_bytes": self.submitted_bytes,
            "mean_ttft_us": sum(ttft_values) / len(ttft_values),
            "p95_ttft_us": _nearest_rank_p95(ttft_values),
            "max_ttft_us": max(ttft_values),
            "mean_gpu_utilization_percent": (
                sum(utilization_values) / len(utilization_values)
            ),
            "min_gpu_utilization_percent": min(utilization_values),
            "utilization_gpu_count": len(utilization_values),
            "mean_observation_window_gpu_utilization_percent": (
                sum(observation_utilization_values)
                / len(observation_utilization_values)
            ),
            "observation_window_gpu_count": len(
                observation_utilization_values
            ),
            "gpus": gpu_results,
            "storage_paths": storage_results,
            "request_conservation": request_conservation,
            "rate_control": rate_control,
            "termination": {
                "stop_mode": self.stop_mode,
                "full_trace_completed": full_trace_completed,
                "stopping_gpu_ids": stopping_gpu_ids,
                "completed_gpu_count": len(
                    self.completed_source_requests
                ),
                "inflight_gpu_count": sum(
                    result["has_inflight_inference"]
                    for result in gpu_results.values()
                ),
                "pending_event_count": len(self.event_loop.events),
            },
            "event_loop": {
                "completion_time_us": time_to_us(
                    self.event_loop.current_time
                ),
                "processed_event_count": (
                    self.event_loop.processed_event_count
                ),
            },
            "wall_time_seconds": wall_time_seconds,
        }

    def run(self):
        started_at = perf_counter()
        try:
            self.event_loop.run_until(self._stop_condition)
            summary = self._build_summary(perf_counter() - started_at)
        finally:
            self.effective_manifest.close()
            self.catalog.close()
        return summary


def run_trace_policy(
    bundle_dir,
    policy,
    simulation_config,
    output_dir,
    inference_count_per_gpu=1,
    warmup_inference_count=0,
    queue_binding_strategy="balanced_exclusive",
    stop_mode="all_gpus_complete",
):
    """运行一种策略，并将有效 SQE manifest 放在该策略目录。"""

    policy_dir = Path(output_dir) / policy
    policy_dir.mkdir(parents=True, exist_ok=True)
    simulation = UcmTraceQosSimulation(
        bundle_dir=bundle_dir,
        policy=policy,
        simulation_config=simulation_config,
        effective_manifest_path=(
            policy_dir / "effective_sqe_manifest.jsonl"
        ),
        inference_count_per_gpu=inference_count_per_gpu,
        warmup_inference_count=warmup_inference_count,
        queue_binding_strategy=queue_binding_strategy,
        stop_mode=stop_mode,
    )
    summary = simulation.run()
    (policy_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_steady_point(
    ssd_count,
    bundle_dir,
    policy,
    simulation_config,
    topology_dir,
    inference_count,
    warmup_count,
    queue_binding,
    stop_mode,
):
    """在独立进程中运行一个SSD数量和策略组合。"""

    summary = run_trace_policy(
        bundle_dir=bundle_dir,
        policy=policy,
        simulation_config=simulation_config,
        output_dir=topology_dir,
        inference_count_per_gpu=inference_count,
        warmup_inference_count=warmup_count,
        queue_binding_strategy=queue_binding,
        stop_mode=stop_mode,
    )
    return ssd_count, policy, summary


def _write_json(path, value):
    """写入一份可直接查看的仿真摘要。"""

    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_utilization_csv(
    experiment,
    output_dir,
    plot_stem=UTILIZATION_PLOT_STEM,
    metric_name=COMPLETED_UTILIZATION_METRIC,
):
    """将 GPU 利用率曲线写成精确数值。"""

    csv_path = Path(output_dir) / f"{plot_stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "ssd_count",
            *[
                f"{policy}_{metric_name}"
                for policy in experiment["policies"]
            ],
        ])
        for ssd_count in experiment["ssd_counts"]:
            topology = experiment["topologies"][f"{ssd_count}_ssd"]
            writer.writerow([
                ssd_count,
                *[
                    topology[policy][metric_name]
                    for policy in experiment["policies"]
                ],
            ])
    return csv_path


def _write_utilization_plot(
    experiment,
    output_dir,
    plot_stem=UTILIZATION_PLOT_STEM,
    metric_name=COMPLETED_UTILIZATION_METRIC,
):
    """绘制 Baseline 和 PIR 不封顶 CIR-only 的利用率曲线。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ssd_counts = experiment["ssd_counts"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy in experiment["policies"]:
        values = [
            experiment["topologies"][f"{ssd_count}_ssd"][policy][
                metric_name
            ]
            for ssd_count in ssd_counts
        ]
        axis.plot(
            ssd_counts,
            values,
            marker="o",
            linewidth=2,
            label=POLICY_LABELS[policy],
        )

    axis.set_xlabel("SSD count")
    axis.set_ylabel("Mean GPU utilization (%)")
    axis.set_xticks(ssd_counts)
    axis.set_ylim(0, 100)
    if experiment.get("stop_mode") == "first_gpu_reaches_limit":
        axis.set_ylabel("Mean observation-window GPU utilization (%)")
        axis.set_title(
            "Stop when the first GPU completes "
            f"{experiment['inference_count_per_gpu']} inferences"
        )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_dir = Path(output_dir)
    png_path = output_dir / f"{plot_stem}.png"
    svg_path = output_dir / f"{plot_stem}.svg"
    figure.savefig(png_path, dpi=200)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


def run_configured_experiment(config_file=DEFAULT_CONFIG_FILE):
    """逐个回放 1～10 SSD trace，并输出配对利用率曲线。"""

    config_file = Path(config_file)
    simulation_config = load_yaml(config_file)["simulation"]
    trace_config = simulation_config["ucm_trace"]

    bundle_root = Path(trace_config["trace_bundle_root"])
    output_dir = Path(trace_config["output_dir"])
    if not bundle_root.is_absolute():
        bundle_root = (PROJECT_DIR / bundle_root).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ssd_counts = list(simulation_config["topology"]["ssd_counts"])
    policies = list(trace_config["policies"])
    experiment = {
        "trace_bundle_root": str(bundle_root),
        "trace_bundle_pattern": trace_config["trace_bundle_pattern"],
        "ssd_counts": ssd_counts,
        "policies": policies,
        "policy_labels": {
            policy: POLICY_LABELS[policy] for policy in policies
        },
        "topologies": {},
    }
    for ssd_count in ssd_counts:
        topology_key = f"{ssd_count}_ssd"
        bundle_dir = bundle_root / trace_config[
            "trace_bundle_pattern"
        ].format(ssd_count=ssd_count)
        topology_dir = output_dir / topology_key
        topology = {"trace_bundle_dir": str(bundle_dir)}
        experiment["topologies"][topology_key] = topology

        for policy in policies:
            print(
                f"START UCM trace ssd_count={ssd_count} policy={policy}",
                flush=True,
            )
            topology[policy] = run_trace_policy(
                bundle_dir=bundle_dir,
                policy=policy,
                simulation_config=simulation_config,
                output_dir=topology_dir,
            )
            _write_json(output_dir / "summary.json", experiment)
            print(
                f"DONE UCM trace ssd_count={ssd_count} policy={policy} "
                f"mean_ttft_us={topology[policy]['mean_ttft_us']:.3f}",
                flush=True,
            )

    csv_path = _write_utilization_csv(experiment, output_dir)
    png_path, svg_path = _write_utilization_plot(experiment, output_dir)
    experiment["artifacts"] = {
        "summary": str(output_dir / "summary.json"),
        "utilization_csv": str(csv_path),
        "utilization_png": str(png_path),
        "utilization_svg": str(svg_path),
    }
    _write_json(output_dir / "summary.json", experiment)
    return experiment


def run_configured_steady_experiment(config_file=DEFAULT_CONFIG_FILE):
    """回放稳态 trace，并按配置的GPU达标条件停止。"""

    config_file = Path(config_file)
    simulation_config = load_yaml(config_file)["simulation"]
    trace_config = simulation_config["ucm_trace_steady"]

    bundle_root = Path(trace_config["trace_bundle_root"])
    output_dir = Path(trace_config["output_dir"])
    if not bundle_root.is_absolute():
        bundle_root = (PROJECT_DIR / bundle_root).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ssd_counts = list(trace_config["ssd_counts"])
    policies = list(trace_config["policies"])
    inference_count = int(trace_config["inference_count_per_gpu"])
    warmup_count = int(trace_config["warmup_inference_count"])
    parallel_workers = int(trace_config["parallel_workers"])
    queue_binding = trace_config["queue_binding_strategy"]
    stop_mode = trace_config["stop_mode"]
    experiment = {
        "trace_bundle_root": str(bundle_root),
        "trace_bundle_pattern": trace_config["trace_bundle_pattern"],
        "ssd_counts": ssd_counts,
        "policies": policies,
        "policy_labels": {
            policy: POLICY_LABELS[policy] for policy in policies
        },
        "inference_count_per_gpu": inference_count,
        "warmup_inference_count": warmup_count,
        "maximum_measured_inference_count_per_gpu": (
            inference_count - warmup_count
        ),
        "parallel_workers": parallel_workers,
        "queue_binding_strategy": queue_binding,
        "stop_mode": stop_mode,
        "topologies": {},
    }

    points = []
    for ssd_count in ssd_counts:
        topology_key = f"{ssd_count}_ssd"
        bundle_dir = bundle_root / trace_config[
            "trace_bundle_pattern"
        ].format(ssd_count=ssd_count)
        topology_dir = output_dir / topology_key
        topology = {"trace_bundle_dir": str(bundle_dir)}
        experiment["topologies"][topology_key] = topology

        for policy in policies:
            points.append((
                ssd_count,
                bundle_dir,
                policy,
                simulation_config,
                topology_dir,
                inference_count,
                warmup_count,
                queue_binding,
                stop_mode,
            ))
            print(
                f"START steady UCM trace ssd_count={ssd_count} "
                f"policy={policy}",
                flush=True,
            )

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [
            executor.submit(_run_steady_point, *point)
            for point in points
        ]
        for future in as_completed(futures):
            ssd_count, policy, policy_summary = future.result()
            topology = experiment["topologies"][f"{ssd_count}_ssd"]
            topology[policy] = policy_summary
            _write_json(output_dir / "summary.json", experiment)
            print(
                f"DONE steady UCM trace ssd_count={ssd_count} "
                f"policy={policy} mean_ttft_us="
                f"{policy_summary['mean_ttft_us']:.3f} "
                f"completed_inferences="
                f"{policy_summary.get('completed_inference_count', '-')}",
                flush=True,
            )

    csv_path = _write_utilization_csv(
        experiment,
        output_dir,
        STEADY_UTILIZATION_PLOT_STEM,
        OBSERVATION_UTILIZATION_METRIC,
    )
    completed_only_csv_path = _write_utilization_csv(
        experiment,
        output_dir,
        STEADY_COMPLETED_UTILIZATION_CSV_STEM,
        COMPLETED_UTILIZATION_METRIC,
    )
    png_path, svg_path = _write_utilization_plot(
        experiment,
        output_dir,
        STEADY_UTILIZATION_PLOT_STEM,
        OBSERVATION_UTILIZATION_METRIC,
    )
    experiment["artifacts"] = {
        "summary": str(output_dir / "summary.json"),
        "utilization_csv": str(csv_path),
        "completed_only_utilization_csv": str(
            completed_only_csv_path
        ),
        "utilization_png": str(png_path),
        "utilization_svg": str(svg_path),
    }
    _write_json(output_dir / "summary.json", experiment)
    return experiment
