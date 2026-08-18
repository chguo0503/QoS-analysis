#!/usr/bin/env python3
"""Compare Baseline and demand-aware CIR with one-inference SSD lookahead.

This experiment intentionally lives outside the original one-active-layer
replay.  Each runtime GPU owns two fixed queues in one QoS group:

* slot ``N % 2`` carries inference N;
* while layer 0 of inference N is computing, layer 0 of N+1 is submitted;
* layers 1..3 of N+1 keep the original layerwise rule and are submitted only
  when the preceding layer of N+1 starts computing;
* GPU computation is strictly ordered by inference index.

Template rounds use a deterministic rotation over the bundle GPUs.  A round
is a permutation, so aggregate bytes/demand are identical in every round,
while each runtime GPU sees a different input length and cache ratio.  The
selected ``measured_indices`` alone define utilization/TTFT/prefetch metrics;
all other rounds only warm up or drain the bounded pipeline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from DPU import (  # noqa: E402
    DPURequestGateway,
    DemandAwareFCFSCIRController,
    build_queue_binding_strategy,
)
from DPU.ucm_trace import UcmTraceBundle  # noqa: E402
from backends.asu_ssd import SSDSimulator  # noqa: E402
from backends.asu_ssd.time_utils import (  # noqa: E402
    TIME_UNITS_PER_US,
    time_to_us,
)
from discrete_simulation import EventLoop  # noqa: E402
from qos import build_qos_simulator, build_queue_layout  # noqa: E402
from simulation_common.aggregate_logs import (  # noqa: E402
    CountOnlyAppendLog,
    DispatchAggregateLog,
)
from simulation_common.config_utils import load_yaml  # noqa: E402
from simulation_common.storage_path import StoragePath  # noqa: E402
from ucm_trace_qos_simulator import (  # noqa: E402
    GPU_COMPLETION_PRIORITY,
    GPU_COMPUTE_START_PRIORITY,
    TIME_UNITS_PER_NS,
    UcmLayerCatalog,
    _configure_one_group_per_gpu_qos,
    _time_to_ns,
)


DEFAULT_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
DEFAULT_INFERENCE_COUNT = 5
DEFAULT_MEASURED_INDICES = (1, 2, 3)
DEFAULT_ROTATION_STRIDE = 29
DEFAULT_STEADY_WINDOW_US = 1_000_000
STEADY_TEMPLATE_CYCLE_LENGTH = 5
TIME_ROUNDING_TOLERANCE_NS = 1e-3
# Start after all ordinary same-timestamp work has settled; stop before any
# t1 device/GPU event.  Their snapshots therefore describe one exact
# half-open interval [t0, t1).
MEASUREMENT_START_PRIORITY = 100
MEASUREMENT_STOP_PRIORITY = -1
DEFAULT_QUEUE_BINDING = "one_group_per_gpu_slots"
DEFAULT_CIR_ORDERING = "shortest"
SUPPORTED_POLICIES = ("baseline", "cir_only")
PLOT_STEM = "pipelined_gpu_utilization_vs_ssd_count"


def _nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile) - 1]


def _mean(values):
    return None if not values else sum(values) / len(values)


def _clamp_duration_delta_ns(value, duration_ns):
    """Clamp only sub-nanosecond conversion noise at a window boundary."""
    if value < -TIME_ROUNDING_TOLERANCE_NS:
        raise RuntimeError("GPU busy time is outside steady window")
    if value > duration_ns + TIME_ROUNDING_TOLERANCE_NS:
        raise RuntimeError("GPU busy time is outside steady window")
    return min(max(value, 0.0), duration_ns)


@dataclass
class PipelineLayer:
    layer_id: int
    issue_time: int
    original_issue_time_ns: int
    queue_slot: int
    entry_count: int
    sqe_count: int
    byte_count: int
    path_bytes_by_storage_target: dict
    pending_entry_count: int
    completion_by_storage_target: dict = field(default_factory=dict)
    load_completion_time: int | None = None
    compute_start_time: int | None = None
    compute_done_time: int | None = None
    compute_scheduled: bool = False


@dataclass
class PipelineInference:
    inference_index: int
    template_source_request_id: str
    template_gpu_id: int
    workload: dict
    layer_ids: tuple
    queue_slot: int
    prefetch_issue_time: int | None = None
    activation_time: int | None = None
    completion_time: int | None = None
    next_compute_position: int = 0
    layers: dict = field(default_factory=dict)

    @property
    def compute_time_per_layer(self):
        return int(self.workload["single_layer_compute_ns"]) * TIME_UNITS_PER_NS


@dataclass
class PipelineGpuState:
    runtime_position: int
    gpu_id: int
    p_node_id: str
    current_inference_index: int = 0
    gpu_available_time: int = 0
    contexts: dict = field(default_factory=dict)
    completed_inferences: list = field(default_factory=list)
    completed: bool = False


@dataclass
class ClientPathSubmission:
    """One logical demand path retained by the upper-client orchestrator."""

    storage_target_id: str
    requests: tuple
    chunk_count: int
    next_offset: int = 0
    next_chunk_index: int = 0
    outstanding_request_ids: set = field(default_factory=set)


class ClientTrafficOrchestrator:
    """Submit at most N IOs per SSD path, replenishing after completion ACKs."""

    def __init__(self, gateway, max_io_per_chunk=None):
        if max_io_per_chunk is not None and (
            not isinstance(max_io_per_chunk, int)
            or isinstance(max_io_per_chunk, bool)
            or max_io_per_chunk <= 0
        ):
            raise ValueError("max_io_per_chunk must be a positive integer")
        self.gateway = gateway
        self.max_io_per_chunk = max_io_per_chunk
        self.pending_by_request_id = {}
        self.demand_path_count = 0
        self.chunk_count = 0
        self.intermediate_replenishment_count = 0
        self.submitted_io_count = 0
        self.max_submitted_chunk_io_count = 0

    def submit_batch(self, requests, arrival_time_us):
        """Split one logical layer by SSD and submit the first path chunks."""
        requests = tuple(requests)
        if self.max_io_per_chunk is None:
            self.submitted_io_count += len(requests)
            return self.gateway.submit_batch(requests, arrival_time_us)

        requests_by_storage_target = {}
        for request in requests:
            storage_target_id = request["basic"]["storage_target_id"]
            requests_by_storage_target.setdefault(
                storage_target_id,
                [],
            ).append(request)

        submitted = []
        for storage_target_id, path_requests in (
            requests_by_storage_target.items()
        ):
            chunk_count = math.ceil(
                len(path_requests) / self.max_io_per_chunk
            )
            state = ClientPathSubmission(
                storage_target_id=storage_target_id,
                requests=tuple(path_requests),
                chunk_count=chunk_count,
            )
            self.demand_path_count += 1
            submitted.extend(self._submit_next_chunk(
                state,
                arrival_time_us,
            ))
        return submitted

    def _submit_next_chunk(self, state, arrival_time_us):
        start = state.next_offset
        stop = min(
            start + self.max_io_per_chunk,
            len(state.requests),
        )
        chunk = state.requests[start:stop]
        chunk_index = state.next_chunk_index
        submission_complete = chunk_index + 1 == state.chunk_count
        for request in chunk:
            demand = request.setdefault("demand_bw", {})
            demand["submission_chunk_index"] = chunk_index
            demand["submission_chunk_count"] = state.chunk_count
            demand["submission_complete"] = submission_complete

        qos_requests = self.gateway.submit_batch(chunk, arrival_time_us)
        queue_ids = {request["queue_id"] for request in qos_requests}
        if len(queue_ids) != 1:
            raise RuntimeError(
                "one client demand path must bind to exactly one Queue"
            )
        queue_id = next(iter(queue_ids))
        state.next_offset = stop
        state.next_chunk_index += 1
        self.chunk_count += 1
        self.submitted_io_count += len(chunk)
        self.max_submitted_chunk_io_count = max(
            self.max_submitted_chunk_io_count,
            len(chunk),
        )
        if not submission_complete:
            state.outstanding_request_ids = {
                request["request_id"] for request in qos_requests
            }
            if len(state.outstanding_request_ids) != len(qos_requests):
                raise RuntimeError("client chunk request IDs must be unique")
            for request_id in state.outstanding_request_ids:
                if request_id in self.pending_by_request_id:
                    raise RuntimeError(
                        "one IO completion cannot replenish two client paths"
                    )
                self.pending_by_request_id[request_id] = state
        return qos_requests

    def on_io_complete(self, request_id, completion_time_us):
        """Replenish one path only after every IO in its prior chunk completes."""
        state = self.pending_by_request_id.pop(request_id, None)
        if state is None:
            return
        state.outstanding_request_ids.remove(request_id)
        if not state.outstanding_request_ids:
            self.intermediate_replenishment_count += 1
            self._submit_next_chunk(state, completion_time_us)

    def statistics(self):
        return {
            "enabled": self.max_io_per_chunk is not None,
            "max_io_per_chunk_per_ssd": self.max_io_per_chunk,
            "demand_path_count": self.demand_path_count,
            "submitted_chunk_count": self.chunk_count,
            "intermediate_replenishment_count": (
                self.intermediate_replenishment_count
            ),
            "submitted_io_count": self.submitted_io_count,
            "max_submitted_chunk_io_count": (
                self.max_submitted_chunk_io_count
            ),
            "pending_path_count": len({
                id(state)
                for state in self.pending_by_request_id.values()
            }),
            "pending_io_completion_count": len(
                self.pending_by_request_id
            ),
        }


class PipelinedUcmSimulation:
    """Replay a bounded two-context pipeline on the existing data path."""

    def __init__(
        self,
        *,
        bundle_dir,
        policy,
        simulation_config,
        inference_count=DEFAULT_INFERENCE_COUNT,
        measured_indices=DEFAULT_MEASURED_INDICES,
        rotation_stride=DEFAULT_ROTATION_STRIDE,
        expected_layer_count=4,
        initial_arrival_jitter_max_us=None,
        queue_binding_strategy=DEFAULT_QUEUE_BINDING,
        client_io_chunk_size=None,
        cir_ordering=DEFAULT_CIR_ORDERING,
    ):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"policy must be one of {SUPPORTED_POLICIES}, got {policy!r}"
            )
        self.policy = policy
        self.config = deepcopy(simulation_config)
        self.inference_count = int(inference_count)
        self.measured_indices = tuple(int(value) for value in measured_indices)
        self.rotation_stride = int(rotation_stride)
        self.expected_layer_count = int(expected_layer_count)
        self.initial_arrival_jitter_max_us = (
            None
            if initial_arrival_jitter_max_us is None
            else int(initial_arrival_jitter_max_us)
        )
        if (
            self.initial_arrival_jitter_max_us is not None
            and self.initial_arrival_jitter_max_us < 0
        ):
            raise ValueError("initial_arrival_jitter_max_us must be non-negative")
        self.queue_binding_strategy = queue_binding_strategy
        self.client_io_chunk_size = client_io_chunk_size
        self.cir_ordering = cir_ordering
        if self.inference_count < 2:
            raise ValueError("pipelined replay requires at least two inferences")
        if not self.measured_indices:
            raise ValueError("measured_indices cannot be empty")
        if len(set(self.measured_indices)) != len(self.measured_indices):
            raise ValueError("measured_indices must be unique")
        if any(
            index < 0 or index >= self.inference_count
            for index in self.measured_indices
        ):
            raise ValueError("measured_indices are outside the inference range")

        self.bundle = UcmTraceBundle(bundle_dir)
        self.catalog = UcmLayerCatalog(self.bundle)
        self.start_time_us = self.config["start_time_us"]
        self.event_loop = EventLoop(
            start_time=int(self.start_time_us * TIME_UNITS_PER_US)
        )
        self.request_owner = {}
        self.completed_gpu_ids = set()
        self.global_completion_order = []
        self.submitted_layer_count = 0
        self.submitted_sqe_count = 0
        self.submitted_entry_count = 0
        self.submitted_bytes = 0

        self.template_workloads = sorted(
            self.bundle.workload_summary["requests"],
            key=lambda row: int(row["gpu_id"]),
        )
        try:
            self._validate_templates()
        except Exception:
            # Validation happens after the lazy catalog opens its two bundle
            # streams.  Do not leak descriptors when rejecting a workload.
            self.catalog.close()
            raise
        self._build_initial_arrival_times()
        self.gpu_states = self._build_gpu_states()
        _configure_one_group_per_gpu_qos(
            self.config["qos"],
            len(self.gpu_states),
        )
        self.storage_target_ids = [
            f"SSD{asu_id}"
            for asu_id in range(
                int(self.bundle.workload_summary["asu_count"])
            )
        ]
        if not self.catalog.storage_target_ids.issubset(
            self.storage_target_ids
        ):
            raise ValueError("manifest target ASU is outside workload topology")
        self._build_data_path()
        self._schedule_initial_inferences()

    def _validate_templates(self):
        gpu_count = len(self.template_workloads)
        if gpu_count > 128:
            raise ValueError(
                "two fixed queue slots require at most 128 GPUs with 256 queues"
            )
        if 256 % gpu_count != 0:
            raise ValueError("GPU count must divide the fixed 256-queue layout")
        if 256 // gpu_count < 2:
            raise ValueError("each GPU group must contain at least two queues")
        if self.inference_count > gpu_count:
            raise ValueError(
                "inference_count cannot exceed template GPU count when every "
                "inference must use a distinct template"
            )
        if math.gcd(self.rotation_stride, gpu_count) != 1:
            raise ValueError("rotation_stride must be coprime to GPU count")

        gpu_ids = [int(row["gpu_id"]) for row in self.template_workloads]
        if len(set(gpu_ids)) != gpu_count:
            raise ValueError("template GPU IDs must be unique")
        layer_ids = None
        for row in self.template_workloads:
            source_request_id = row["source_request_id"]
            current_ids = self.catalog.layer_ids_by_source.get(
                source_request_id
            )
            if current_ids is None:
                raise ValueError(
                    f"workload template {source_request_id!r} has no layers"
                )
            if layer_ids is None:
                layer_ids = current_ids
            elif current_ids != layer_ids:
                raise ValueError("all templates must have identical layer IDs")
        if len(layer_ids) != self.expected_layer_count:
            raise ValueError(
                f"bundle has {len(layer_ids)} layers; expected "
                f"{self.expected_layer_count}"
            )

        # Each round must remain a full permutation.  Each runtime GPU must
        # also see distinct input lengths *and* distinct cache ratios.  A
        # uniqueness check on only the pair would incorrectly accept a
        # repeated input when just the ratio changed (or vice versa).
        all_positions = set(range(gpu_count))
        for inference_index in range(self.inference_count):
            round_positions = {
                (runtime_position + inference_index * self.rotation_stride)
                % gpu_count
                for runtime_position in range(gpu_count)
            }
            if round_positions != all_positions:
                raise RuntimeError("template rotation round is not a permutation")
        for runtime_position in range(gpu_count):
            input_tokens = []
            cache_ratios = []
            for inference_index in range(self.inference_count):
                row = self._template_for(runtime_position, inference_index)
                input_tokens.append(int(row["input_tokens"]))
                cache_ratios.append(row.get("sampled_cached_prefix_ratio"))
            if len(set(input_tokens)) != len(input_tokens):
                raise ValueError(
                    "template rotation does not provide distinct input "
                    f"lengths for runtime GPU position {runtime_position}"
                )
            if len(set(cache_ratios)) != len(cache_ratios):
                raise ValueError(
                    "template rotation does not provide distinct cache "
                    f"ratios for runtime GPU position {runtime_position}"
                )

    def _template_for(self, runtime_position, inference_index):
        template_position = (
            runtime_position + inference_index * self.rotation_stride
        ) % len(self.template_workloads)
        return self.template_workloads[template_position]

    def _build_initial_arrival_times(self):
        arrival = self.bundle.workload_summary.get("arrival", {})
        base_ns = int(arrival.get("base_time_ns", 0))
        source_min_ns = int(arrival.get("jitter_min_ns", 0))
        source_max_ns = int(arrival.get("jitter_max_ns", 0))
        if source_min_ns < 0 or source_max_ns < source_min_ns:
            raise ValueError("bundle arrival jitter range is invalid")
        sampled_jitters_ns = [
            int(workload["arrival_time_ns"]) - base_ns
            for workload in self.template_workloads
        ]
        source_range_origin = "bundle_metadata"
        if (
            min(sampled_jitters_ns) < source_min_ns
            or max(sampled_jitters_ns) > source_max_ns
        ):
            source_min_ns = min(sampled_jitters_ns)
            source_max_ns = max(sampled_jitters_ns)
            source_range_origin = "sample_fallback"
        source_span_ns = source_max_ns - source_min_ns
        target_max_ns = (
            source_max_ns
            if self.initial_arrival_jitter_max_us is None
            else self.initial_arrival_jitter_max_us * 1_000
        )

        times = {}
        for workload in self.template_workloads:
            gpu_id = int(workload["gpu_id"])
            source_arrival_ns = int(workload["arrival_time_ns"])
            source_jitter_ns = source_arrival_ns - base_ns
            if not source_min_ns <= source_jitter_ns <= source_max_ns:
                raise ValueError("workload arrival is outside bundle jitter range")
            if self.initial_arrival_jitter_max_us is None:
                effective_jitter_ns = source_jitter_ns
            elif source_span_ns == 0:
                effective_jitter_ns = 0
            else:
                effective_jitter_ns = (
                    (source_jitter_ns - source_min_ns) * target_max_ns
                    // source_span_ns
                )
            times[gpu_id] = base_ns + effective_jitter_ns

        effective_values = list(times.values())
        self.initial_arrival_time_ns_by_gpu = times
        self.initial_arrival_metadata = {
            "mapping": (
                "bundle_values"
                if self.initial_arrival_jitter_max_us is None
                else "linear_integer_rescale_preserving_sample_order"
            ),
            "source_range_origin": source_range_origin,
            "base_time_ns": base_ns,
            "source_jitter_min_ns": source_min_ns,
            "source_jitter_max_ns": source_max_ns,
            "configured_effective_jitter_min_ns": 0,
            "configured_effective_jitter_max_ns": target_max_ns,
            "sample_first_arrival_time_ns": min(effective_values),
            "sample_last_arrival_time_ns": max(effective_values),
            "sample_arrival_span_ns": max(effective_values) - min(effective_values),
        }

    def _build_gpu_states(self):
        states = {}
        for runtime_position, workload in enumerate(self.template_workloads):
            gpu_id = int(workload["gpu_id"])
            states[gpu_id] = PipelineGpuState(
                runtime_position=runtime_position,
                gpu_id=gpu_id,
                p_node_id=f"P{gpu_id}",
                gpu_available_time=self.event_loop.current_time,
            )
        return states

    def _new_context(self, state, inference_index):
        workload = self._template_for(
            state.runtime_position,
            inference_index,
        )
        source_request_id = workload["source_request_id"]
        return PipelineInference(
            inference_index=inference_index,
            template_source_request_id=source_request_id,
            template_gpu_id=int(workload["gpu_id"]),
            workload=workload,
            layer_ids=self.catalog.layer_ids_by_source[source_request_id],
            queue_slot=inference_index % 2,
        )

    def _build_rate_controller(self, capacity_by_storage_target):
        if self.policy == "baseline":
            return None
        return DemandAwareFCFSCIRController(
            capacity_by_storage_target,
            ordering=self.cir_ordering,
        )

    def _build_data_path(self):
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
            dispatch_log = DispatchAggregateLog()
            qos.dispatched_requests = dispatch_log
            ssd.backend.completed_requests = CountOnlyAppendLog()
            ssd.backend.nand_service_events = CountOnlyAppendLog()
            self.storage_paths[storage_target_id] = path
            self.dispatch_logs[storage_target_id] = dispatch_log

        queue_ids_by_storage_target = {
            storage_target_id: queue_layout.queue_order
            for storage_target_id in self.storage_target_ids
        }
        p_node_ids = [
            state.p_node_id
            for state in sorted(
                self.gpu_states.values(),
                key=lambda state: state.runtime_position,
            )
        ]
        binding = build_queue_binding_strategy(
            strategy_name=self.queue_binding_strategy,
            p_node_ids=p_node_ids,
            queue_ids_by_storage_target=queue_ids_by_storage_target,
        )
        capacity = int(
            backend_config["nand"]["read_bandwidth_bytes_per_second"]
        )
        capacity_by_storage_target = {
            storage_target_id: capacity
            for storage_target_id in self.storage_target_ids
        }
        self.dpu = DPURequestGateway(
            queue_ids_by_storage_target=queue_ids_by_storage_target,
            queue_binding_strategy=binding,
            request_sink=self._route_qos_request,
            qos_interfaces_by_storage_target={
                storage_target_id: path.qos
                for storage_target_id, path in self.storage_paths.items()
            },
            rate_controller=self._build_rate_controller(
                capacity_by_storage_target
            ),
        )
        self.client = ClientTrafficOrchestrator(
            self.dpu,
            max_io_per_chunk=self.client_io_chunk_size,
        )
        for path in self.storage_paths.values():
            path.start()

    def _route_qos_request(self, request):
        return self.storage_paths[request["storage_target_id"]].input(
            request
        )

    def _schedule_initial_inferences(self):
        for state in self.gpu_states.values():
            context = self._new_context(state, 0)
            state.contexts[0] = context
            arrival_time = (
                self.initial_arrival_time_ns_by_gpu[state.gpu_id]
                * TIME_UNITS_PER_NS
            )
            context.activation_time = arrival_time
            self.event_loop.schedule_at(
                event_time=arrival_time,
                priority=GPU_COMPUTE_START_PRIORITY,
                event_name=f"pipeline-initial-layer:{state.gpu_id}",
                callback=lambda event_time, gpu_id=state.gpu_id: (
                    self._issue_layer(
                        gpu_id,
                        0,
                        self.gpu_states[gpu_id].contexts[0].layer_ids[0],
                        event_time,
                    )
                ),
            )

    def _issue_layer(
        self,
        gpu_id,
        inference_index,
        layer_id,
        event_time,
    ):
        state = self.gpu_states[gpu_id]
        context = state.contexts[inference_index]
        if layer_id in context.layers:
            raise RuntimeError("one pipeline layer was issued twice")
        submission, records = self.catalog.load_layer(
            context.template_source_request_id,
            layer_id,
        )
        if submission.gpu_id != context.template_gpu_id:
            raise RuntimeError("template GPU metadata changed while loading")
        issue_time_us = time_to_us(event_time)
        inference_reference_time = (
            event_time
            if context.activation_time is None
            else context.activation_time
        )
        runtime_prefix = (
            f"runtime-gpu-{gpu_id:04d}:inference-{inference_index:04d}:"
            f"layer-{layer_id:02d}"
        )
        for request in submission.requests:
            basic = request["basic"]
            template_request_id = basic["request_id"]
            basic["request_id"] = f"{runtime_prefix}:{template_request_id}"
            basic["p_node_id"] = state.p_node_id
            basic["queue_slot"] = context.queue_slot
            demand = request["demand_bw"]
            demand["demand_group_id"] = runtime_prefix
            demand["inference_arrival_time_us"] = time_to_us(
                inference_reference_time
            )
            demand["deadline_us"] = (
                issue_time_us + demand["service_window_us"]
            )

        layer = PipelineLayer(
            layer_id=layer_id,
            issue_time=event_time,
            original_issue_time_ns=submission.timestamp_ns,
            queue_slot=context.queue_slot,
            entry_count=len(submission.requests),
            sqe_count=len(records),
            byte_count=submission.batch_total_bytes,
            path_bytes_by_storage_target=dict(
                submission.path_bytes_by_storage_target
            ),
            pending_entry_count=len(submission.requests),
        )
        context.layers[layer_id] = layer
        if layer_id == context.layer_ids[0]:
            if context.prefetch_issue_time is not None:
                raise RuntimeError("inference layer 0 was prefetched twice")
            context.prefetch_issue_time = event_time

        for request in submission.requests:
            request_id = request["basic"]["request_id"]
            if request_id in self.request_owner:
                raise RuntimeError(f"duplicate runtime request ID {request_id!r}")
            self.request_owner[request_id] = (
                gpu_id,
                inference_index,
                layer_id,
            )

        self.submitted_layer_count += 1
        self.submitted_sqe_count += len(records)
        self.submitted_entry_count += len(submission.requests)
        self.submitted_bytes += submission.batch_total_bytes
        self.client.submit_batch(
            requests=submission.requests,
            arrival_time_us=issue_time_us,
        )

    def _on_storage_complete(self, completion):
        request_id = completion["request_id"]
        try:
            gpu_id, inference_index, layer_id = self.request_owner.pop(
                request_id
            )
        except KeyError as error:
            raise RuntimeError(
                f"completion has no pipeline owner: {request_id!r}"
            ) from error
        state = self.gpu_states[gpu_id]
        context = state.contexts[inference_index]
        layer = context.layers[layer_id]
        completion_time = self.event_loop.current_time
        self.client.on_io_complete(
            request_id,
            completion["completion_time_us"],
        )
        layer.pending_entry_count -= 1
        if layer.pending_entry_count < 0:
            raise RuntimeError("pipeline layer completion underflow")
        storage_target_id = completion["storage_target_id"]
        layer.completion_by_storage_target[storage_target_id] = max(
            completion_time,
            layer.completion_by_storage_target.get(
                storage_target_id,
                completion_time,
            ),
        )
        if layer.pending_entry_count == 0:
            layer.load_completion_time = completion_time
            if inference_index == state.current_inference_index:
                expected_layer_id = context.layer_ids[
                    context.next_compute_position
                ]
                if layer_id == expected_layer_id:
                    self._schedule_compute(state, context, layer)

    def _schedule_compute(self, state, context, layer):
        if layer.compute_scheduled or layer.compute_start_time is not None:
            return
        if context.activation_time is None:
            return
        compute_start_time = max(
            layer.load_completion_time,
            state.gpu_available_time,
            context.activation_time,
        )
        layer.compute_scheduled = True
        self.event_loop.schedule_at(
            event_time=compute_start_time,
            priority=GPU_COMPUTE_START_PRIORITY,
            event_name=(
                f"pipeline-compute:{state.gpu_id}:"
                f"{context.inference_index}:{layer.layer_id}"
            ),
            callback=lambda event_time, gpu_id=state.gpu_id,
            inference_index=context.inference_index,
            layer_id=layer.layer_id: self._process_compute_start(
                gpu_id,
                inference_index,
                layer_id,
                event_time,
            ),
        )

    def _process_compute_start(
        self,
        gpu_id,
        inference_index,
        layer_id,
        event_time,
    ):
        state = self.gpu_states[gpu_id]
        if inference_index != state.current_inference_index:
            raise RuntimeError("GPU attempted out-of-order inference compute")
        context = state.contexts[inference_index]
        expected_layer_id = context.layer_ids[context.next_compute_position]
        if layer_id != expected_layer_id:
            raise RuntimeError("GPU attempted out-of-order layer compute")
        layer = context.layers[layer_id]
        if layer.load_completion_time is None:
            raise RuntimeError("GPU compute started before layer IO completed")
        if event_time < state.gpu_available_time:
            raise RuntimeError("two GPU compute intervals overlapped")

        compute_done_time = event_time + context.compute_time_per_layer
        layer.compute_start_time = event_time
        layer.compute_done_time = compute_done_time
        state.gpu_available_time = compute_done_time
        layer_position = context.next_compute_position
        context.next_compute_position += 1

        # Preserve the existing layerwise prefetch rule for the current
        # inference.  Give it deterministic FCFS registration precedence over
        # the N+1 Layer-0 lookahead at the same timestamp.
        if layer_position + 1 < len(context.layer_ids):
            self._issue_layer(
                gpu_id,
                inference_index,
                context.layer_ids[layer_position + 1],
                event_time,
            )

        # The only cross-inference lookahead is N+1 Layer 0.  It occupies the
        # other fixed queue slot and may finish while N continues computing.
        if layer_position == 0:
            self._admit_lookahead(state, context, event_time)

        if layer_position + 1 == len(context.layer_ids):
            self.event_loop.schedule_at(
                event_time=compute_done_time,
                priority=GPU_COMPLETION_PRIORITY,
                event_name=(
                    f"pipeline-inference-complete:{gpu_id}:"
                    f"{inference_index}"
                ),
                callback=lambda completion_time, gpu_id=gpu_id,
                inference_index=inference_index: (
                    self._process_inference_completion(
                        gpu_id,
                        inference_index,
                        completion_time,
                    )
                ),
            )

    def _admit_lookahead(self, state, current_context, event_time):
        next_index = current_context.inference_index + 1
        if next_index >= self.inference_count:
            return
        if next_index in state.contexts:
            raise RuntimeError("pipeline admitted N+1 more than once")
        if len(state.contexts) != 1:
            raise RuntimeError("pipeline exceeded two-context admission logic")
        context = self._new_context(state, next_index)
        state.contexts[next_index] = context
        self._issue_layer(
            state.gpu_id,
            next_index,
            context.layer_ids[0],
            event_time,
        )

    def _process_inference_completion(
        self,
        gpu_id,
        inference_index,
        event_time,
    ):
        state = self.gpu_states[gpu_id]
        if inference_index != state.current_inference_index:
            raise RuntimeError("GPU inference completion order changed")
        context = state.contexts[inference_index]
        if context.next_compute_position != len(context.layer_ids):
            raise RuntimeError("inference completed before all layer compute")
        context.completion_time = event_time
        state.completed_inferences.append(context)
        self.global_completion_order.append({
            "sequence": len(self.global_completion_order),
            "gpu_id": gpu_id,
            "inference_index": inference_index,
            "completion_time_ns": _time_to_ns(event_time),
        })
        del state.contexts[inference_index]

        next_index = inference_index + 1
        if next_index >= self.inference_count:
            if state.contexts:
                raise RuntimeError("final inference left an admitted successor")
            state.completed = True
            self.completed_gpu_ids.add(gpu_id)
            return

        if set(state.contexts) != {next_index}:
            raise RuntimeError("N+1 context is missing at GPU promotion")
        next_context = state.contexts[next_index]
        next_context.activation_time = event_time
        state.current_inference_index = next_index
        first_layer = next_context.layers[next_context.layer_ids[0]]
        if first_layer.load_completion_time is not None:
            self._schedule_compute(state, next_context, first_layer)

    def _all_gpus_complete(self):
        return len(self.completed_gpu_ids) == len(self.gpu_states)

    def _layer_output(self, layer):
        return {
            "layer_id": layer.layer_id,
            "queue_slot": layer.queue_slot,
            "original_issue_time_ns": layer.original_issue_time_ns,
            "effective_issue_time_ns": _time_to_ns(layer.issue_time),
            "load_completion_time_ns": _time_to_ns(
                layer.load_completion_time
            ),
            "load_latency_ns": _time_to_ns(
                layer.load_completion_time - layer.issue_time
            ),
            "compute_start_time_ns": _time_to_ns(
                layer.compute_start_time
            ),
            "compute_done_time_ns": _time_to_ns(layer.compute_done_time),
            "entry_count": layer.entry_count,
            "sqe_count": layer.sqe_count,
            "bytes": layer.byte_count,
            "path_bytes_by_storage_target": (
                layer.path_bytes_by_storage_target
            ),
            "completion_time_ns_by_storage_target": {
                target: _time_to_ns(completion_time)
                for target, completion_time in sorted(
                    layer.completion_by_storage_target.items()
                )
            },
        }

    def _inference_output(self, context):
        duration = context.completion_time - context.activation_time
        compute_time = (
            len(context.layer_ids) * context.compute_time_per_layer
        )
        first_layer = context.layers[context.layer_ids[0]]
        return {
            "inference_index": context.inference_index,
            "measurement_role": (
                "measured"
                if context.inference_index in self.measured_indices
                else (
                    "warmup"
                    if context.inference_index < min(self.measured_indices)
                    else "cooldown"
                )
            ),
            "template_source_request_id": (
                context.template_source_request_id
            ),
            "template_gpu_id": context.template_gpu_id,
            "input_tokens": int(context.workload["input_tokens"]),
            "cached_prefix_ratio": context.workload.get(
                "sampled_cached_prefix_ratio"
            ),
            "cached_token_count": context.workload.get(
                "cached_token_count"
            ),
            "single_layer_compute_ns": int(
                context.workload["single_layer_compute_ns"]
            ),
            "queue_slot": context.queue_slot,
            "prefetch_issue_time_ns": _time_to_ns(
                context.prefetch_issue_time
            ),
            # This is the TTFT origin.  For N>0 it is deliberately later
            # than the early Layer-0 SSD submission and equals completion of
            # inference N-1.
            "logical_arrival_time_ns": _time_to_ns(
                context.activation_time
            ),
            "activation_time_ns": _time_to_ns(context.activation_time),
            "completion_time_ns": _time_to_ns(context.completion_time),
            "ttft_ns": _time_to_ns(duration),
            "ttft_us": time_to_us(duration),
            "compute_time_ns": _time_to_ns(compute_time),
            "storage_stall_ns": _time_to_ns(duration - compute_time),
            "gpu_utilization_percent": compute_time / duration * 100,
            "layer0_prefetched_before_activation": (
                context.prefetch_issue_time < context.activation_time
            ),
            "layer0_ready_before_activation": (
                first_layer.load_completion_time <= context.activation_time
            ),
            "layer0_ready_lead_ns": _time_to_ns(
                context.activation_time - first_layer.load_completion_time
            ),
            "layers": [
                self._layer_output(context.layers[layer_id])
                for layer_id in context.layer_ids
            ],
        }

    def _gpu_results(self):
        results = {}
        for state in sorted(
            self.gpu_states.values(),
            key=lambda item: item.runtime_position,
        ):
            if [
                context.inference_index
                for context in state.completed_inferences
            ] != list(range(self.inference_count)):
                raise RuntimeError("GPU did not complete strict inference order")
            outputs = [
                self._inference_output(context)
                for context in state.completed_inferences
            ]
            measured = [
                context
                for context in state.completed_inferences
                if context.inference_index in self.measured_indices
            ]
            measured_duration = sum(
                context.completion_time - context.activation_time
                for context in measured
            )
            measured_compute = sum(
                len(context.layer_ids) * context.compute_time_per_layer
                for context in measured
            )
            results[str(state.gpu_id)] = {
                "gpu_id": state.gpu_id,
                "completed_inference_count": len(outputs),
                "completed_inference_order": [
                    output["inference_index"] for output in outputs
                ],
                "measured_inference_indices": list(
                    self.measured_indices
                ),
                "measured_window_start_ns": outputs[
                    min(self.measured_indices)
                ]["activation_time_ns"],
                "measured_window_end_ns": outputs[
                    max(self.measured_indices)
                ]["completion_time_ns"],
                "measured_compute_time_ns": _time_to_ns(
                    measured_compute
                ),
                "measured_duration_ns": _time_to_ns(measured_duration),
                "gpu_utilization_percent": (
                    measured_compute / measured_duration * 100
                ),
                "template_gpu_sequence": [
                    output["template_gpu_id"] for output in outputs
                ],
                "input_token_sequence": [
                    output["input_tokens"] for output in outputs
                ],
                "cache_ratio_sequence": [
                    output["cached_prefix_ratio"] for output in outputs
                ],
                "inferences": outputs,
            }
        return results

    def _storage_results(self):
        results = {}
        for storage_target_id, path in sorted(self.storage_paths.items()):
            backend = path.ssd.backend
            log = self.dispatch_logs[storage_target_id]
            first_submit_time_us = (
                None
                if backend.first_submit_time is None
                else time_to_us(backend.first_submit_time)
            )
            last_completion_time_us = (
                None
                if backend.last_completion_time is None
                else time_to_us(backend.last_completion_time)
            )
            active_time_us = (
                0
                if first_submit_time_us is None
                else last_completion_time_us - first_submit_time_us
            )
            results[storage_target_id] = {
                "request_count": log.count,
                "bytes": log.byte_count,
                "completed_request_count": (
                    backend.completed_requests.count
                ),
                "completed_bytes": backend.completed_bytes(),
                "cir_dispatch_count": log.cir_count,
                "excess_dispatch_count": log.excess_count,
                "first_submit_time_us": first_submit_time_us,
                "last_completion_time_us": last_completion_time_us,
                "active_time_us": active_time_us,
                "average_bandwidth_gb_s": (
                    0
                    if active_time_us == 0
                    else log.byte_count / active_time_us / 1_000
                ),
                "ssd_blocked_attempt_count": (
                    path.ssd.blocked_attempt_count
                ),
            }
        return results

    def _validate_conservation(self, storage_results):
        expected_entry_count = (
            int(self.bundle.metadata["output_counts"][
                "retrieve_entry_count"
            ])
            * self.inference_count
        )
        expected_sqe_count = (
            int(self.bundle.metadata["output_counts"][
                "retrieve_sqe_count"
            ])
            * self.inference_count
        )
        expected_byte_count = (
            int(self.bundle.workload_summary[
                "estimated_retrieve_payload_bytes"
            ])
            * self.inference_count
        )
        expected_layer_count = (
            len(self.gpu_states)
            * self.expected_layer_count
            * self.inference_count
        )
        qos_entries = sum(
            result["request_count"] for result in storage_results.values()
        )
        qos_bytes = sum(
            result["bytes"] for result in storage_results.values()
        )
        ssd_entries = sum(
            result["completed_request_count"]
            for result in storage_results.values()
        )
        ssd_bytes = sum(
            result["completed_bytes"]
            for result in storage_results.values()
        )
        actual_values = {
            "layers": self.submitted_layer_count,
            "sqes": self.submitted_sqe_count,
            "entries": self.submitted_entry_count,
            "bytes": self.submitted_bytes,
            "qos_entries": qos_entries,
            "qos_bytes": qos_bytes,
            "ssd_entries": ssd_entries,
            "ssd_bytes": ssd_bytes,
        }
        client_statistics = self.client.statistics()
        actual_values["client_submitted_entries"] = client_statistics[
            "submitted_io_count"
        ]
        if self.submitted_layer_count != expected_layer_count:
            raise RuntimeError("pipeline layer count is incomplete")
        if self.submitted_sqe_count != expected_sqe_count:
            raise RuntimeError("pipeline SQE count is incomplete")
        if len({self.submitted_entry_count, qos_entries, ssd_entries}) != 1:
            raise RuntimeError("pipeline request conservation failed")
        if self.submitted_entry_count != expected_entry_count:
            raise RuntimeError("pipeline Entry count is incomplete")
        if len({self.submitted_bytes, qos_bytes, ssd_bytes}) != 1:
            raise RuntimeError("pipeline byte conservation failed")
        if self.submitted_bytes != expected_byte_count:
            raise RuntimeError("pipeline byte count is incomplete")
        if self.request_owner:
            raise RuntimeError("pipeline ended with active request ownership")
        if client_statistics["pending_path_count"] != 0:
            raise RuntimeError("pipeline ended with staged client requests")
        if (
            client_statistics["submitted_io_count"]
            != self.submitted_entry_count
        ):
            raise RuntimeError("client submission conservation failed")

        dpu_statistics = self.dpu.statistics()
        if any(
            depth != 0
            for depths in dpu_statistics[
                "queue_io_counts_by_storage_target"
            ].values()
            for depth in depths.values()
        ):
            raise RuntimeError("pipeline ended with non-empty QoS queues")
        rate_control = dpu_statistics["rate_control"]
        if (
            rate_control is not None
            and rate_control["active_demand_count"] != 0
        ):
            raise RuntimeError("pipeline ended with active CIR demand")
        return {
            "expected": {
                "layers": expected_layer_count,
                "sqes": expected_sqe_count,
                "entries": expected_entry_count,
                "bytes": expected_byte_count,
            },
            "actual": actual_values,
            "passed": True,
        }, dpu_statistics

    def _build_summary(self, wall_time_seconds):
        if not self._all_gpus_complete():
            raise RuntimeError("summary requested before all GPUs completed")
        gpu_results = self._gpu_results()
        storage_results = self._storage_results()
        conservation, dpu_statistics = self._validate_conservation(
            storage_results
        )
        measured_outputs = [
            inference
            for gpu in gpu_results.values()
            for inference in gpu["inferences"]
            if inference["inference_index"] in self.measured_indices
        ]
        utilization_values = [
            gpu["gpu_utilization_percent"]
            for gpu in gpu_results.values()
        ]
        total_measured_compute_ns = sum(
            gpu["measured_compute_time_ns"]
            for gpu in gpu_results.values()
        )
        total_measured_duration_ns = sum(
            gpu["measured_duration_ns"]
            for gpu in gpu_results.values()
        )
        # Keep this auxiliary statistic on exactly the same finite scope as
        # utilization and TTFT.  For a three-inference run measuring index 1,
        # index 2 is a necessary N+1 drain context, not a reported sample.
        prefetched_outputs = [
            inference
            for inference in measured_outputs
            if inference["inference_index"] > 0
        ]
        rate_control = dpu_statistics["rate_control"]
        compact_rate_control = None
        if rate_control is not None:
            compact_rate_control = {
                "strategy": rate_control["strategy"],
                "ordering": rate_control["ordering"],
                "active_demand_count": rate_control[
                    "active_demand_count"
                ],
                "completed_demand_count_by_storage_target": (
                    rate_control[
                        "completed_demand_count_by_storage_target"
                    ]
                ),
                "peak_assigned_cir_bytes_per_second": rate_control[
                    "peak_assigned_cir_bytes_per_second"
                ],
                "rate_control_write_count": dpu_statistics[
                    "rate_control_write_count"
                ],
                "registered_chunk_count": rate_control.get(
                    "registered_chunk_count"
                ),
                "intermediate_empty_count": rate_control.get(
                    "intermediate_empty_count"
                ),
            }
        return {
            "schema_version": "pipelined-ucm-comparison/v1",
            "policy": self.policy,
            "trace_bundle_dir": str(self.bundle.bundle_dir),
            "gpu_count": len(gpu_results),
            "ssd_count": len(storage_results),
            "layer_count": self.expected_layer_count,
            "inference_count_per_gpu": self.inference_count,
            "measured_inference_indices": list(self.measured_indices),
            "metric_scope": {
                "gpu_utilization": "measured_inference_indices_only",
                "ttft": "measured_inference_indices_only",
                "storage_stall": "measured_inference_indices_only",
                "layer0_prefetch": "measured_inference_indices_only",
                "storage_paths": (
                    "entire_bounded_run_including_warmup_and_drain"
                ),
                "completion_order": "entire_bounded_run",
            },
            "initial_arrival": self.initial_arrival_metadata,
            "pipeline_semantics": {
                "gpu_compute_order": "strict_inference_then_layer_order",
                "cross_inference_lookahead_depth": 1,
                "cross_inference_prefetch_scope": "next_inference_layer0_only",
                "lookahead_trigger": "current_inference_layer0_compute_start",
                "logical_arrival_rule": (
                    "inference0_template_jitter_then_previous_completion"
                ),
                "ttft_origin": "logical_arrival_not_prefetch_issue",
                "layers_1_to_3_rule": (
                    "issue_when_previous_layer_compute_starts"
                ),
                "template_rotation_stride": self.rotation_stride,
                "each_round_is_template_permutation": True,
                "queue_binding_strategy": self.queue_binding_strategy,
                "queue_slot_rule": "inference_index_mod_2",
                "same_group_queue_slots": [0, 1],
                "client_submission": (
                    "per_ssd_completion_ack_replenishment"
                    if self.client_io_chunk_size is not None
                    else "whole_layer"
                ),
            },
            "mean_gpu_utilization_percent": _mean(utilization_values),
            "aggregate_gpu_utilization_percent": (
                total_measured_compute_ns
                / total_measured_duration_ns
                * 100
            ),
            "min_gpu_utilization_percent": min(utilization_values),
            "max_gpu_utilization_percent": max(utilization_values),
            "mean_ttft_us": _mean([
                output["ttft_us"] for output in measured_outputs
            ]),
            "p95_ttft_us": _nearest_rank(
                [output["ttft_us"] for output in measured_outputs],
                0.95,
            ),
            "max_ttft_us": max(
                output["ttft_us"] for output in measured_outputs
            ),
            "mean_storage_stall_us": _mean([
                output["storage_stall_ns"] / 1_000
                for output in measured_outputs
            ]),
            "layer0_prefetch": {
                "eligible_inference_count": len(prefetched_outputs),
                "issued_before_activation_count": sum(
                    output["layer0_prefetched_before_activation"]
                    for output in prefetched_outputs
                ),
                "ready_before_activation_count": sum(
                    output["layer0_ready_before_activation"]
                    for output in prefetched_outputs
                ),
                "ready_before_activation_percent": (
                    None
                    if not prefetched_outputs
                    else sum(
                        output["layer0_ready_before_activation"]
                        for output in prefetched_outputs
                    ) / len(prefetched_outputs) * 100
                ),
                "mean_ready_lead_us": _mean([
                    output["layer0_ready_lead_ns"] / 1_000
                    for output in prefetched_outputs
                ]),
            },
            "completed_inference_count": sum(
                gpu["completed_inference_count"]
                for gpu in gpu_results.values()
            ),
            "all_gpus_complete": True,
            "gpus": gpu_results,
            "global_completion_order": self.global_completion_order,
            "storage_paths": storage_results,
            "request_conservation": conservation,
            "client_traffic_orchestration": self.client.statistics(),
            "rate_control": compact_rate_control,
            "event_loop": {
                "completion_time_us": time_to_us(
                    self.event_loop.current_time
                ),
                "processed_event_count": (
                    self.event_loop.processed_event_count
                ),
                "pending_event_count": len(self.event_loop.events),
            },
            "wall_time_seconds": wall_time_seconds,
        }

    def run(self, max_events=None):
        started_at = perf_counter()
        try:
            self.event_loop.run_until(
                self._all_gpus_complete,
                max_events=max_events,
            )
            return self._build_summary(perf_counter() - started_at)
        finally:
            self.catalog.close()


class ContinuousPipelinedUcmSimulation(PipelinedUcmSimulation):
    """Measure the same N+1 pipeline in one common steady-state window.

    The finite five-inference experiment above is kept as the default.  This
    variant repeats the same five-template sequence indefinitely while the
    measurement is active.  At the end marker it disables only *new* N+1
    admissions, then drains the current and already-admitted contexts so the
    normal request/demand conservation checks can still be applied.
    """

    def __init__(
        self,
        *,
        bundle_dir,
        policy,
        simulation_config,
        steady_window_us=DEFAULT_STEADY_WINDOW_US,
        settling_guard_us=0,
        rotation_stride=DEFAULT_ROTATION_STRIDE,
        expected_layer_count=4,
        initial_arrival_jitter_max_us=None,
        queue_binding_strategy=DEFAULT_QUEUE_BINDING,
        client_io_chunk_size=None,
        cir_ordering=DEFAULT_CIR_ORDERING,
    ):
        steady_window_us = int(steady_window_us)
        settling_guard_us = int(settling_guard_us)
        if steady_window_us <= 0:
            raise ValueError("steady_window_us must be positive")
        if settling_guard_us < 0:
            raise ValueError("settling_guard_us must be non-negative")

        self.steady_window_us = steady_window_us
        self.settling_guard_us = settling_guard_us
        self._steady_window_time = steady_window_us * TIME_UNITS_PER_US
        self._settling_guard_time = settling_guard_us * TIME_UNITS_PER_US
        self._measurement_start_scheduled = False
        self._warmup_barrier_time = None
        self._measurement_start_time = None
        self._measurement_end_time = None
        self._measurement_start_snapshot = None
        self._measurement_end_snapshot = None
        self._admissions_stopped = False
        self._drain_complete = False

        # Five is the template-cycle length, not a completion limit in this
        # subclass.  Passing it through lets the base validator prove that the
        # five consecutive templates for every runtime GPU are distinct.
        super().__init__(
            bundle_dir=bundle_dir,
            policy=policy,
            simulation_config=simulation_config,
            inference_count=STEADY_TEMPLATE_CYCLE_LENGTH,
            measured_indices=tuple(range(STEADY_TEMPLATE_CYCLE_LENGTH)),
            rotation_stride=rotation_stride,
            expected_layer_count=expected_layer_count,
            initial_arrival_jitter_max_us=initial_arrival_jitter_max_us,
            queue_binding_strategy=queue_binding_strategy,
            client_io_chunk_size=client_io_chunk_size,
            cir_ordering=cir_ordering,
        )

    def _template_for(self, runtime_position, inference_index):
        """Repeat five distinct templates while preserving round permutations."""

        cycle_position = inference_index % STEADY_TEMPLATE_CYCLE_LENGTH
        template_position = (
            runtime_position + cycle_position * self.rotation_stride
        ) % len(self.template_workloads)
        return self.template_workloads[template_position]

    def _admit_lookahead(self, state, current_context, event_time):
        """Continue the source until t1, then suppress only new contexts."""

        if self._admissions_stopped:
            return
        next_index = current_context.inference_index + 1
        if next_index in state.contexts:
            raise RuntimeError("pipeline admitted N+1 more than once")
        if len(state.contexts) != 1:
            raise RuntimeError("pipeline exceeded two-context admission logic")
        context = self._new_context(state, next_index)
        state.contexts[next_index] = context
        self._issue_layer(
            state.gpu_id,
            next_index,
            context.layer_ids[0],
            event_time,
        )

    def _process_inference_completion(
        self,
        gpu_id,
        inference_index,
        event_time,
    ):
        """Promote the queued successor, or finish this GPU during drain."""

        state = self.gpu_states[gpu_id]
        if inference_index != state.current_inference_index:
            raise RuntimeError("GPU inference completion order changed")
        context = state.contexts[inference_index]
        if context.next_compute_position != len(context.layer_ids):
            raise RuntimeError("inference completed before all layer compute")
        context.completion_time = event_time
        state.completed_inferences.append(context)
        self.global_completion_order.append({
            "sequence": len(self.global_completion_order),
            "gpu_id": gpu_id,
            "inference_index": inference_index,
            "completion_time_ns": _time_to_ns(event_time),
        })
        del state.contexts[inference_index]

        next_index = inference_index + 1
        if next_index in state.contexts:
            if set(state.contexts) != {next_index}:
                raise RuntimeError("more than one successor remained at promotion")
            next_context = state.contexts[next_index]
            next_context.activation_time = event_time
            state.current_inference_index = next_index
            first_layer = next_context.layers[next_context.layer_ids[0]]
            if first_layer.load_completion_time is not None:
                self._schedule_compute(state, next_context, first_layer)
        elif self._admissions_stopped:
            state.completed = True
            self.completed_gpu_ids.add(gpu_id)
            self._drain_complete = self._all_gpus_complete()
        else:
            raise RuntimeError("N+1 context is missing before measurement stop")

        self._maybe_arm_measurement(event_time)

    def _maybe_arm_measurement(self, event_time):
        """Arm one t0 only after every GPU has completed its warmup."""

        if self._measurement_start_scheduled or self._admissions_stopped:
            return
        if not all(
            len(state.completed_inferences) >= 1
            for state in self.gpu_states.values()
        ):
            return
        self._measurement_start_scheduled = True
        self._warmup_barrier_time = event_time
        self.event_loop.schedule_at(
            event_time=event_time + self._settling_guard_time,
            priority=MEASUREMENT_START_PRIORITY,
            event_name="pipeline-steady-measurement-start",
            callback=self._begin_measurement,
        )

    def _contexts_known_at_snapshot(self, state):
        yield from state.completed_inferences
        yield from state.contexts.values()

    def _cumulative_compute_busy(self, state, snapshot_time):
        """Count actual compute intervals clipped at a global timestamp."""

        busy = 0
        for context in self._contexts_known_at_snapshot(state):
            for layer in context.layers.values():
                if layer.compute_start_time is None:
                    continue
                interval_end = min(snapshot_time, layer.compute_done_time)
                busy += max(0, interval_end - layer.compute_start_time)
        return busy

    def _take_measurement_snapshot(self, event_time):
        return {
            "time_ns": _time_to_ns(event_time),
            "time_us": time_to_us(event_time),
            "processed_event_count": self.event_loop.processed_event_count,
            "gpus": {
                str(state.gpu_id): {
                    "compute_busy_ns": _time_to_ns(
                        self._cumulative_compute_busy(state, event_time)
                    ),
                    "completed_inference_count": len(
                        state.completed_inferences
                    ),
                }
                for state in sorted(
                    self.gpu_states.values(),
                    key=lambda item: item.runtime_position,
                )
            },
            "storage_paths": {
                storage_target_id: {
                    "completed_bytes": path.ssd.backend.completed_bytes(),
                    "completed_request_count": (
                        path.ssd.backend.completed_requests.count
                    ),
                }
                for storage_target_id, path in sorted(
                    self.storage_paths.items()
                )
            },
        }

    def _begin_measurement(self, event_time):
        if self._measurement_start_time is not None:
            raise RuntimeError("measurement start was scheduled twice")
        if not all(
            len(state.completed_inferences) >= 1
            for state in self.gpu_states.values()
        ):
            raise RuntimeError("measurement started before every GPU warmed up")
        self._measurement_start_time = event_time
        self._measurement_start_snapshot = self._take_measurement_snapshot(
            event_time
        )
        self.event_loop.schedule_at(
            event_time=event_time + self._steady_window_time,
            priority=MEASUREMENT_STOP_PRIORITY,
            event_name="pipeline-steady-measurement-stop",
            callback=self._finish_measurement,
        )

    def _finish_measurement(self, event_time):
        if self._admissions_stopped:
            raise RuntimeError("measurement stop was scheduled twice")
        self._measurement_end_time = event_time
        self._measurement_end_snapshot = self._take_measurement_snapshot(
            event_time
        )
        self._admissions_stopped = True

    def _validate_drained_conservation(self, storage_results):
        qos_entries = sum(
            result["request_count"] for result in storage_results.values()
        )
        qos_bytes = sum(
            result["bytes"] for result in storage_results.values()
        )
        ssd_entries = sum(
            result["completed_request_count"]
            for result in storage_results.values()
        )
        ssd_bytes = sum(
            result["completed_bytes"] for result in storage_results.values()
        )
        if len({self.submitted_entry_count, qos_entries, ssd_entries}) != 1:
            raise RuntimeError("steady pipeline request conservation failed")
        if len({self.submitted_bytes, qos_bytes, ssd_bytes}) != 1:
            raise RuntimeError("steady pipeline byte conservation failed")
        if self.request_owner:
            raise RuntimeError("steady pipeline drain left request ownership")
        dpu_statistics = self.dpu.statistics()
        if any(
            depth != 0
            for depths in dpu_statistics[
                "queue_io_counts_by_storage_target"
            ].values()
            for depth in depths.values()
        ):
            raise RuntimeError("steady pipeline drain left non-empty queues")
        rate_control = dpu_statistics["rate_control"]
        if rate_control is not None and rate_control["active_demand_count"] != 0:
            raise RuntimeError("steady pipeline drain left active CIR demand")
        return {
            "expected": "all dynamically admitted work drains",
            "actual": {
                "layers": self.submitted_layer_count,
                "sqes": self.submitted_sqe_count,
                "entries": self.submitted_entry_count,
                "bytes": self.submitted_bytes,
                "qos_entries": qos_entries,
                "qos_bytes": qos_bytes,
                "ssd_entries": ssd_entries,
                "ssd_bytes": ssd_bytes,
            },
            "passed": True,
        }, dpu_statistics

    def _fully_contained_contexts(self, state):
        return [
            context
            for context in state.completed_inferences
            if (
                context.activation_time >= self._measurement_start_time
                and context.completion_time < self._measurement_end_time
            )
        ]

    def _build_steady_summary(self, wall_time_seconds):
        if not self._drain_complete:
            raise RuntimeError("steady summary requested before drain completed")
        if (
            self._measurement_start_snapshot is None
            or self._measurement_end_snapshot is None
        ):
            raise RuntimeError("steady measurement snapshots are incomplete")

        start = self._measurement_start_snapshot
        end = self._measurement_end_snapshot
        duration_ns = _time_to_ns(self._steady_window_time)
        duration_seconds = duration_ns / 1_000_000_000
        gpu_results = {}
        utilization_values = []
        contained_outputs = []
        completed_inference_count_in_window = 0
        for state in sorted(
            self.gpu_states.values(),
            key=lambda item: item.runtime_position,
        ):
            completed_order = [
                context.inference_index
                for context in state.completed_inferences
            ]
            if completed_order != list(range(len(completed_order))):
                raise RuntimeError("steady GPU inference order changed")
            start_busy = start["gpus"][str(state.gpu_id)]["compute_busy_ns"]
            end_busy = end["gpus"][str(state.gpu_id)]["compute_busy_ns"]
            busy_ns = _clamp_duration_delta_ns(
                end_busy - start_busy,
                duration_ns,
            )
            completed_in_window = (
                end["gpus"][str(state.gpu_id)][
                    "completed_inference_count"
                ]
                - start["gpus"][str(state.gpu_id)][
                    "completed_inference_count"
                ]
            )
            if completed_in_window < 0:
                raise RuntimeError("GPU completion counter moved backwards")
            completed_inference_count_in_window += completed_in_window
            utilization = busy_ns / duration_ns * 100
            utilization_values.append(utilization)
            contained = self._fully_contained_contexts(state)
            outputs = [self._inference_output(context) for context in contained]
            for output in outputs:
                output["measurement_role"] = (
                    "fully_contained_common_steady_window"
                )
            contained_outputs.extend(outputs)
            gpu_results[str(state.gpu_id)] = {
                "gpu_id": state.gpu_id,
                "completed_inference_count": len(state.completed_inferences),
                "completed_inference_order": completed_order,
                "template_gpu_sequence": [
                    context.template_gpu_id
                    for context in state.completed_inferences
                ],
                "first_cycle_template_gpu_sequence": [
                    context.template_gpu_id
                    for context in state.completed_inferences[:5]
                ],
                "measurement_compute_busy_ns": busy_ns,
                "gpu_utilization_percent": utilization,
                "completed_inference_count_in_window": (
                    completed_in_window
                ),
                "fully_contained_ttft_sample_count": len(outputs),
                "mean_ttft_us": _mean([
                    output["ttft_us"] for output in outputs
                ]),
                "inferences": outputs,
            }

        storage_results = self._storage_results()
        conservation, dpu_statistics = self._validate_drained_conservation(
            storage_results
        )
        storage_window = {}
        for target in start["storage_paths"]:
            completed_bytes = (
                end["storage_paths"][target]["completed_bytes"]
                - start["storage_paths"][target]["completed_bytes"]
            )
            completed_requests = (
                end["storage_paths"][target]["completed_request_count"]
                - start["storage_paths"][target]["completed_request_count"]
            )
            storage_window[target] = {
                "completed_bytes": completed_bytes,
                "completed_request_count": completed_requests,
                "completed_bandwidth_gb_s": (
                    completed_bytes / duration_seconds / 1_000_000_000
                ),
            }

        ttft_values = [row["ttft_us"] for row in contained_outputs]
        stall_values = [
            row["storage_stall_ns"] / 1_000 for row in contained_outputs
        ]
        prefetched = [
            row for row in contained_outputs if row["inference_index"] > 0
        ]
        paired_outputs = []
        for state in self.gpu_states.values():
            if len(state.completed_inferences) <= 1:
                raise RuntimeError("GPU did not complete paired inference 1")
            paired_outputs.append(
                self._inference_output(state.completed_inferences[1])
            )
        paired_template_ids = {
            row["template_gpu_id"] for row in paired_outputs
        }
        expected_template_ids = {
            int(row["gpu_id"]) for row in self.template_workloads
        }
        if paired_template_ids != expected_template_ids:
            raise RuntimeError("paired inference is not a template permutation")
        paired_ttft_ns = [row["ttft_ns"] for row in paired_outputs]
        paired_compute_ns = [
            row["compute_time_ns"] for row in paired_outputs
        ]
        paired_stall_us = [
            row["storage_stall_ns"] / 1_000 for row in paired_outputs
        ]
        paired_utilization = [
            row["gpu_utilization_percent"] for row in paired_outputs
        ]
        rate_control = dpu_statistics["rate_control"]
        compact_rate_control = None
        if rate_control is not None:
            compact_rate_control = {
                "strategy": rate_control["strategy"],
                "ordering": rate_control["ordering"],
                "active_demand_count": rate_control["active_demand_count"],
                "completed_demand_count_by_storage_target": rate_control[
                    "completed_demand_count_by_storage_target"
                ],
                "peak_assigned_cir_bytes_per_second": rate_control[
                    "peak_assigned_cir_bytes_per_second"
                ],
                "rate_control_write_count": dpu_statistics[
                    "rate_control_write_count"
                ],
                "registered_chunk_count": rate_control.get(
                    "registered_chunk_count"
                ),
                "intermediate_empty_count": rate_control.get(
                    "intermediate_empty_count"
                ),
            }

        mean_utilization = _mean(utilization_values)
        return {
            "schema_version": "pipelined-ucm-comparison/v2-steady-window",
            "policy": self.policy,
            "trace_bundle_dir": str(self.bundle.bundle_dir),
            "gpu_count": len(gpu_results),
            "ssd_count": len(storage_results),
            "layer_count": self.expected_layer_count,
            "inference_count_per_gpu": None,
            "template_cycle_length": STEADY_TEMPLATE_CYCLE_LENGTH,
            "rotation_stride": self.rotation_stride,
            "initial_arrival": self.initial_arrival_metadata,
            "measurement_mode": "continuous_fixed_common_window_then_drain",
            "warmup_definition": "every GPU completed at least 1 inference",
            "settling_guard_us": self.settling_guard_us,
            "steady_window_us": self.steady_window_us,
            "warmup_barrier_time_ns": _time_to_ns(self._warmup_barrier_time),
            "measurement_start_time_ns": start["time_ns"],
            "measurement_end_time_ns": end["time_ns"],
            "measurement_snapshots": {"start": start, "end": end},
            "paired_post_warmup_inference": {
                "inference_index": 1,
                "warmup_scope": "one_prior_inference_per_gpu_local",
                "sample_count": len(paired_outputs),
                "all_templates_once": (
                    paired_template_ids == expected_template_ids
                ),
                "all_completed_before_measurement_end": all(
                    row["completion_time_ns"] < end["time_ns"]
                    for row in paired_outputs
                ),
                "layer0_ready_before_activation_percent": (
                    100
                    * sum(
                        row["layer0_ready_before_activation"]
                        for row in paired_outputs
                    )
                    / len(paired_outputs)
                ),
                "mean_gpu_utilization_percent": _mean(
                    paired_utilization
                ),
                "aggregate_gpu_utilization_percent": (
                    sum(paired_compute_ns) / sum(paired_ttft_ns) * 100
                ),
                "mean_ttft_us": _mean([
                    value / 1_000 for value in paired_ttft_ns
                ]),
                "p95_ttft_us": _nearest_rank(
                    [value / 1_000 for value in paired_ttft_ns],
                    0.95,
                ),
                "max_ttft_us": max(paired_ttft_ns) / 1_000,
                "mean_storage_stall_us": _mean(paired_stall_us),
            },
            "mean_gpu_utilization_percent": mean_utilization,
            "aggregate_gpu_utilization_percent": mean_utilization,
            "min_gpu_utilization_percent": min(utilization_values),
            "max_gpu_utilization_percent": max(utilization_values),
            "mean_ttft_us": _mean(ttft_values),
            "p95_ttft_us": _nearest_rank(ttft_values, 0.95),
            "max_ttft_us": max(ttft_values) if ttft_values else None,
            "mean_storage_stall_us": _mean(stall_values),
            "ttft_sample_count": len(ttft_values),
            "layer0_prefetch": {
                "eligible_inference_count": len(prefetched),
                "issued_before_activation_count": sum(
                    row["layer0_prefetched_before_activation"]
                    for row in prefetched
                ),
                "ready_before_activation_count": sum(
                    row["layer0_ready_before_activation"] for row in prefetched
                ),
                "ready_before_activation_percent": (
                    None
                    if not prefetched
                    else sum(
                        row["layer0_ready_before_activation"]
                        for row in prefetched
                    ) / len(prefetched) * 100
                ),
            },
            "completed_inference_count": sum(
                row["completed_inference_count"]
                for row in gpu_results.values()
            ),
            "completed_inference_count_in_window": (
                completed_inference_count_in_window
            ),
            "fully_contained_ttft_sample_count": len(contained_outputs),
            "mean_inference_throughput_per_gpu_per_second": (
                completed_inference_count_in_window
                / len(gpu_results)
                / duration_seconds
            ),
            "all_gpus_complete": True,
            "drain_complete": True,
            "new_lookahead_admissions_stopped_at_t1": True,
            "source_continued_through_measurement": True,
            "every_gpu_completed_inference_during_measurement": all(
                end["gpus"][gpu_id]["completed_inference_count"]
                > start["gpus"][gpu_id]["completed_inference_count"]
                for gpu_id in start["gpus"]
            ),
            "gpus": gpu_results,
            "global_completion_order": self.global_completion_order,
            "storage_paths": storage_results,
            "measurement_storage_paths": storage_window,
            "request_conservation": conservation,
            "client_traffic_orchestration": self.client.statistics(),
            "rate_control": compact_rate_control,
            "pipeline_semantics": {
                "gpu_compute_order": "strict_inference_then_layer_order",
                "cross_inference_lookahead_depth": 1,
                "cross_inference_prefetch_scope": "next_inference_layer0_only",
                "template_sequence": "five_template_cycle",
                "queue_slot_rule": "inference_index_mod_2",
                "compute_measurement_interval": "one_common_half_open_[t0,t1)_window",
                "ttft_inclusion": "activation>=t0_and_completion<t1",
                "termination": "stop_new_lookahead_at_t1_then_drain_admitted",
                "client_submission": (
                    "per_ssd_completion_ack_replenishment"
                    if self.client_io_chunk_size is not None
                    else "whole_layer"
                ),
            },
            "event_loop": {
                "completion_time_us": time_to_us(self.event_loop.current_time),
                "processed_event_count": self.event_loop.processed_event_count,
                "pending_event_count": len(self.event_loop.events),
            },
            "wall_time_seconds": wall_time_seconds,
        }

    def run(self, max_events=None):
        started_at = perf_counter()
        try:
            self.event_loop.run_until(
                lambda: self._drain_complete,
                max_events=max_events,
            )
            return self._build_steady_summary(perf_counter() - started_at)
        finally:
            self.catalog.close()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_pipeline_policy(
    *,
    bundle_dir,
    policy,
    simulation_config,
    output_dir,
    inference_count=DEFAULT_INFERENCE_COUNT,
    measured_indices=DEFAULT_MEASURED_INDICES,
    rotation_stride=DEFAULT_ROTATION_STRIDE,
    expected_layer_count=4,
    initial_arrival_jitter_max_us=None,
    cir_client_io_chunk_size=None,
    cir_ordering=DEFAULT_CIR_ORDERING,
    steady_window_us=None,
    settling_guard_us=0,
    max_events=None,
):
    policy_dir = Path(output_dir) / policy
    policy_dir.mkdir(parents=True, exist_ok=True)
    client_io_chunk_size = (
        cir_client_io_chunk_size if policy == "cir_only" else None
    )
    if steady_window_us is None:
        simulation = PipelinedUcmSimulation(
            bundle_dir=bundle_dir,
            policy=policy,
            simulation_config=simulation_config,
            inference_count=inference_count,
            measured_indices=measured_indices,
            rotation_stride=rotation_stride,
            expected_layer_count=expected_layer_count,
            initial_arrival_jitter_max_us=initial_arrival_jitter_max_us,
            client_io_chunk_size=client_io_chunk_size,
            cir_ordering=cir_ordering,
        )
    else:
        simulation = ContinuousPipelinedUcmSimulation(
            bundle_dir=bundle_dir,
            policy=policy,
            simulation_config=simulation_config,
            steady_window_us=steady_window_us,
            settling_guard_us=settling_guard_us,
            rotation_stride=rotation_stride,
            expected_layer_count=expected_layer_count,
            initial_arrival_jitter_max_us=initial_arrival_jitter_max_us,
            client_io_chunk_size=client_io_chunk_size,
            cir_ordering=cir_ordering,
        )
    summary = simulation.run(max_events=max_events)
    summary_path = policy_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary


def _compact_summary(summary, summary_path):
    return {
        "summary_path": str(summary_path),
        "gpu_count": summary["gpu_count"],
        "ssd_count": summary["ssd_count"],
        "mean_gpu_utilization_percent": summary[
            "mean_gpu_utilization_percent"
        ],
        "aggregate_gpu_utilization_percent": summary[
            "aggregate_gpu_utilization_percent"
        ],
        "mean_ttft_us": summary["mean_ttft_us"],
        "p95_ttft_us": summary["p95_ttft_us"],
        "max_ttft_us": summary["max_ttft_us"],
        "mean_storage_stall_us": summary["mean_storage_stall_us"],
        "layer0_ready_before_activation_percent": summary[
            "layer0_prefetch"
        ]["ready_before_activation_percent"],
        "wall_time_seconds": summary["wall_time_seconds"],
        "all_gpus_complete": summary["all_gpus_complete"],
        "request_conservation_passed": summary[
            "request_conservation"
        ]["passed"],
        "client_traffic_orchestration": summary[
            "client_traffic_orchestration"
        ],
        "paired_post_warmup_inference": summary.get(
            "paired_post_warmup_inference"
        ),
    }


def _run_sweep_point(
    ssd_count,
    bundle_dir,
    policy,
    simulation_config,
    topology_dir,
    inference_count,
    measured_indices,
    rotation_stride,
    expected_layer_count,
    initial_arrival_jitter_max_us,
    cir_client_io_chunk_size,
    cir_ordering,
    steady_window_us,
    settling_guard_us,
    max_events,
):
    summary = run_pipeline_policy(
        bundle_dir=bundle_dir,
        policy=policy,
        simulation_config=simulation_config,
        output_dir=topology_dir,
        inference_count=inference_count,
        measured_indices=measured_indices,
        rotation_stride=rotation_stride,
        expected_layer_count=expected_layer_count,
        initial_arrival_jitter_max_us=initial_arrival_jitter_max_us,
        cir_client_io_chunk_size=cir_client_io_chunk_size,
        cir_ordering=cir_ordering,
        steady_window_us=steady_window_us,
        settling_guard_us=settling_guard_us,
        max_events=max_events,
    )
    summary_path = Path(topology_dir) / policy / "summary.json"
    return ssd_count, policy, _compact_summary(summary, summary_path)


def _write_sweep_csv(experiment, output_dir):
    path = Path(output_dir) / f"{PLOT_STEM}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "ssd_count",
            *[
                f"{policy}_mean_gpu_utilization_percent"
                for policy in experiment["policies"]
            ],
        ])
        for ssd_count in experiment["ssd_counts"]:
            writer.writerow([
                ssd_count,
                *[
                    experiment["topologies"][f"{ssd_count}_ssd"][policy][
                        "mean_gpu_utilization_percent"
                    ]
                    for policy in experiment["policies"]
                ],
            ])
    return path


def _write_sweep_plot(experiment, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "baseline": "Baseline",
        "cir_only": (
            "CIR-only "
            f"({experiment.get('cir_ordering', 'fcfs')}, PIR uncapped)"
        ),
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy in experiment["policies"]:
        values = [
            experiment["topologies"][f"{ssd_count}_ssd"][policy][
                "mean_gpu_utilization_percent"
            ]
            for ssd_count in experiment["ssd_counts"]
        ]
        axis.plot(
            experiment["ssd_counts"],
            values,
            marker="o",
            linewidth=2,
            label=labels[policy],
        )
    axis.set_xlabel("SSD count")
    if experiment.get("steady_window_us") is None:
        measured = experiment.get("measured_inference_indices", [])
        if len(measured) == 1:
            scope = f"inference {measured[0]}"
        else:
            scope = "inferences " + ",".join(str(value) for value in measured)
        axis.set_ylabel(f"Mean GPU utilization, {scope} (%)")
    else:
        axis.set_ylabel("Mean GPU utilization, common steady window (%)")
    axis.set_xticks(experiment["ssd_counts"])
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir = Path(output_dir)
    png_path = output_dir / f"{PLOT_STEM}.png"
    svg_path = output_dir / f"{PLOT_STEM}.svg"
    figure.savefig(png_path, dpi=200)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


def run_pipeline_sweep(
    *,
    bundle_root,
    bundle_pattern,
    ssd_counts,
    policies,
    simulation_config,
    output_dir,
    inference_count=DEFAULT_INFERENCE_COUNT,
    measured_indices=DEFAULT_MEASURED_INDICES,
    rotation_stride=DEFAULT_ROTATION_STRIDE,
    expected_layer_count=4,
    initial_arrival_jitter_max_us=None,
    cir_client_io_chunk_size=None,
    cir_ordering=DEFAULT_CIR_ORDERING,
    steady_window_us=None,
    settling_guard_us=0,
    parallel_workers=1,
    max_events=None,
    write_plot=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ssd_counts = tuple(int(value) for value in ssd_counts)
    policies = tuple(policies)
    if not ssd_counts:
        raise ValueError("ssd_counts cannot be empty")
    if not policies or any(policy not in SUPPORTED_POLICIES for policy in policies):
        raise ValueError(f"policies must be selected from {SUPPORTED_POLICIES}")
    experiment = {
        "schema_version": "pipelined-ucm-sweep/v1",
        "bundle_root": str(Path(bundle_root)),
        "bundle_pattern": bundle_pattern,
        "ssd_counts": list(ssd_counts),
        "policies": list(policies),
        "inference_count_per_gpu": (
            int(inference_count) if steady_window_us is None else None
        ),
        "measured_inference_indices": (
            list(measured_indices) if steady_window_us is None else None
        ),
        "template_cycle_length": (
            None
            if steady_window_us is None
            else STEADY_TEMPLATE_CYCLE_LENGTH
        ),
        "rotation_stride": int(rotation_stride),
        "expected_layer_count": int(expected_layer_count),
        "initial_arrival_jitter_max_us": initial_arrival_jitter_max_us,
        "cir_client_io_chunk_size": cir_client_io_chunk_size,
        "cir_ordering": cir_ordering,
        "measurement_mode": (
            "finite_middle_inferences"
            if steady_window_us is None
            else "continuous_fixed_common_window_then_drain"
        ),
        "steady_window_us": steady_window_us,
        "settling_guard_us": int(settling_guard_us),
        "topologies": {
            f"{ssd_count}_ssd": {} for ssd_count in ssd_counts
        },
    }
    jobs = []
    for ssd_count in ssd_counts:
        bundle_dir = Path(bundle_root) / bundle_pattern.format(
            ssd_count=ssd_count
        )
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"missing trace bundle: {bundle_dir}")
        topology_dir = output_dir / f"{ssd_count}_ssd"
        for policy in policies:
            jobs.append((
                ssd_count,
                str(bundle_dir),
                policy,
                simulation_config,
                str(topology_dir),
                inference_count,
                tuple(measured_indices),
                rotation_stride,
                expected_layer_count,
                initial_arrival_jitter_max_us,
                cir_client_io_chunk_size,
                cir_ordering,
                steady_window_us,
                settling_guard_us,
                max_events,
            ))

    started_at = perf_counter()
    if parallel_workers > 1:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_run_sweep_point, *job) for job in jobs]
            for future in as_completed(futures):
                ssd_count, policy, compact = future.result()
                experiment["topologies"][f"{ssd_count}_ssd"][policy] = compact
    else:
        for job in jobs:
            ssd_count, policy, compact = _run_sweep_point(*job)
            experiment["topologies"][f"{ssd_count}_ssd"][policy] = compact
    experiment["wall_time_seconds"] = perf_counter() - started_at
    csv_path = _write_sweep_csv(experiment, output_dir)
    experiment["utilization_csv"] = str(csv_path)
    if write_plot:
        png_path, svg_path = _write_sweep_plot(experiment, output_dir)
        experiment["utilization_plot_png"] = str(png_path)
        experiment["utilization_plot_svg"] = str(svg_path)
    _write_json(output_dir / "summary.json", experiment)
    return experiment


def _parse_integer_list(value):
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _parse_policy_list(value):
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or any(item not in SUPPORTED_POLICIES for item in values):
        raise argparse.ArgumentTypeError(
            f"policies must be comma-separated values from {SUPPORTED_POLICIES}"
        )
    return values


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--bundle-root", type=Path)
    parser.add_argument(
        "--bundle-pattern",
        default="gpu_128_asu_{ssd_count}",
    )
    parser.add_argument("--ssd-counts", type=_parse_integer_list)
    parser.add_argument(
        "--policies",
        type=_parse_policy_list,
        default=SUPPORTED_POLICIES,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-count", type=int, default=5)
    parser.add_argument(
        "--measured-indices",
        type=_parse_integer_list,
        default=DEFAULT_MEASURED_INDICES,
    )
    parser.add_argument(
        "--rotation-stride",
        type=int,
        default=DEFAULT_ROTATION_STRIDE,
    )
    parser.add_argument("--expected-layer-count", type=int, default=4)
    parser.add_argument(
        "--initial-arrival-jitter-max-us",
        type=int,
        help=(
            "linearly rescale bundle inference-0 jitter into [0, value] us "
            "while preserving the sampled GPU order"
        ),
    )
    parser.add_argument(
        "--steady-window-us",
        type=int,
        help=(
            "enable continuous fixed-window mode with this duration; "
            f"1,000,000 us is recommended (finite mode is the default)"
        ),
    )
    parser.add_argument(
        "--settling-guard-us",
        type=int,
        default=0,
        help="extra continuous-load settling time after every GPU warms up",
    )
    parser.add_argument(
        "--cir-client-io-chunk-size",
        type=int,
        help=(
            "enable CIR-only upper-client traffic orchestration and submit "
            "at most this many IOs per SSD path before completion-ACK refill"
        ),
    )
    parser.add_argument(
        "--cir-ordering",
        choices=("fcfs", "shortest"),
        default=DEFAULT_CIR_ORDERING,
        help="CIR demand ordering; shortest uses complete coflow bytes",
    )
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.steady_window_us is not None and args.steady_window_us <= 0:
        raise ValueError("--steady-window-us must be positive")
    if args.settling_guard_us < 0:
        raise ValueError("--settling-guard-us must be non-negative")
    if (
        args.cir_client_io_chunk_size is not None
        and args.cir_client_io_chunk_size <= 0
    ):
        raise ValueError("--cir-client-io-chunk-size must be positive")
    if (
        args.initial_arrival_jitter_max_us is not None
        and args.initial_arrival_jitter_max_us < 0
    ):
        raise ValueError("--initial-arrival-jitter-max-us must be non-negative")
    if args.steady_window_us is None and args.settling_guard_us != 0:
        raise ValueError(
            "--settling-guard-us requires --steady-window-us"
        )
    simulation_config = load_yaml(args.config)["simulation"]
    if args.bundle is not None:
        if args.ssd_counts is not None:
            raise ValueError("--ssd-counts is only valid with --bundle-root")
        results = {}
        for policy in args.policies:
            summary = run_pipeline_policy(
                bundle_dir=args.bundle,
                policy=policy,
                simulation_config=simulation_config,
                output_dir=args.output_dir,
                inference_count=args.inference_count,
                measured_indices=args.measured_indices,
                rotation_stride=args.rotation_stride,
                expected_layer_count=args.expected_layer_count,
                initial_arrival_jitter_max_us=(
                    args.initial_arrival_jitter_max_us
                ),
                cir_client_io_chunk_size=args.cir_client_io_chunk_size,
                cir_ordering=args.cir_ordering,
                steady_window_us=args.steady_window_us,
                settling_guard_us=args.settling_guard_us,
                max_events=args.max_events,
            )
            results[policy] = _compact_summary(
                summary,
                args.output_dir / policy / "summary.json",
            )
        _write_json(args.output_dir / "summary.json", {
            "schema_version": "pipelined-ucm-single-bundle/v1",
            "bundle": str(args.bundle),
            "policies": list(args.policies),
            "initial_arrival_jitter_max_us": (
                args.initial_arrival_jitter_max_us
            ),
            "cir_client_io_chunk_size": args.cir_client_io_chunk_size,
            "cir_ordering": args.cir_ordering,
            "results": results,
        })
        return 0

    if args.ssd_counts is None:
        raise ValueError("--ssd-counts is required with --bundle-root")
    run_pipeline_sweep(
        bundle_root=args.bundle_root,
        bundle_pattern=args.bundle_pattern,
        ssd_counts=args.ssd_counts,
        policies=args.policies,
        simulation_config=simulation_config,
        output_dir=args.output_dir,
        inference_count=args.inference_count,
        measured_indices=args.measured_indices,
        rotation_stride=args.rotation_stride,
        expected_layer_count=args.expected_layer_count,
        initial_arrival_jitter_max_us=args.initial_arrival_jitter_max_us,
        cir_client_io_chunk_size=args.cir_client_io_chunk_size,
        cir_ordering=args.cir_ordering,
        steady_window_us=args.steady_window_us,
        settling_guard_us=args.settling_guard_us,
        parallel_workers=args.parallel_workers,
        max_events=args.max_events,
        write_plot=not args.skip_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
