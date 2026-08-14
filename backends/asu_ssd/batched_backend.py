#!/usr/bin/env python3
"""ASU六级流水线的精确批量时序后端。

该实现不省略FCP、BCP、NFI、NAND、BDP或DAS中任何一级。
它将原来逐4 KiB命令的start/complete堆事件，改写为同等的
整数max-plus时序递推，并且只把FCP入口重新可用和完整IO完成
这两类对上层可观察事件放入堆。
"""

import heapq
from collections import deque

from .time_utils import (
    bandwidth_to_interval,
    rate_to_interval,
    time_to_us,
    us_to_time,
)


COMPLETE_PRIORITY = 0
CAPACITY_PRIORITY = 1


class IndexedTimeWindow:
    """保存时序递推只需要的最近固定数量整数时刻。"""

    def __init__(self, capacity):
        """功能：创建可按全局顺序号读取的环形时刻窗口。

        目的：精确递推最多只依赖一个硬件容量之前的离开时刻，
        因此不必为数百万条4 KiB命令保存完整Python列表。

        输入：需要保留的最大历史项数。

        输出：无；初始化空环形窗口。
        """
        self.capacity = capacity
        self.values = [0] * capacity
        self.count = 0

    def append(self, value):
        """功能：追加一个新的整数时刻。

        目的：用环形覆盖丢弃已经不可能被容量约束引用的旧值。

        输入：ASU内部整数时间单位表示的时刻。

        输出：无；窗口项数加一。
        """
        self.values[self.count % self.capacity] = value
        self.count += 1

    def get(self, global_index):
        """功能：按从0开始的全局顺序号返回历史时刻。

        目的：直接读取如 ``n-512`` 的硬件槽位释放时刻，
        避免deque按下标扫描。

        输入：还在窗口中的全局顺序号。

        输出：对应的ASU整数时刻。
        """
        return self.values[global_index % self.capacity]


class StagePeakTracker:
    """使用单调时刻队列计算阶段等待和占用峰值。"""

    def __init__(self):
        """功能：创建空的阶段峰值跟踪器。

        目的：在不生成逐命令事件对象的前提下，保留原模型
        ``max_waiting`` 和 ``max_occupied`` 统计。

        输入：无。

        输出：无；初始峰值均为0。
        """
        self.pending_start_times = deque()
        self.pending_departure_times = deque()
        self.max_waiting = 0
        self.max_occupied = 0

    def accept(
        self,
        arrival_time,
        start_time,
        departure_time,
        starts_at_same_time_are_earlier=False,
    ):
        """功能：登记一条命令的到达、启动和离开时刻。

        目的：按详细模型的同时刻先后关系移除已启动/已离开
        命令，然后记录本次accept后的瞬时深度。

        输入：到达时刻、启动时刻、离开时刻，以及同时刻start
        是否已先于本次accept执行。

        输出：无；更新两个峰值和待处理时刻队列。
        """
        while self.pending_start_times and (
            self.pending_start_times[0] < arrival_time
            or (
                starts_at_same_time_are_earlier
                and self.pending_start_times[0] == arrival_time
            )
        ):
            self.pending_start_times.popleft()

        while (
            self.pending_departure_times
            and self.pending_departure_times[0] <= arrival_time
        ):
            self.pending_departure_times.popleft()

        self.pending_start_times.append(start_time)
        self.pending_departure_times.append(departure_time)
        self.max_waiting = max(
            self.max_waiting,
            len(self.pending_start_times),
        )
        self.max_occupied = max(
            self.max_occupied,
            len(self.pending_departure_times),
        )


class BatchedExactASUBackend:
    """使用有限容量max-plus递推精确执行ASU六级流水线。"""

    execution_mode = "batched_exact"

    def __init__(self, backend_config):
        """功能：根据同一份ASU YAML创建精确批量后端。

        目的：使用与逐事件模型相同的速率、延迟、槽位容量、
        FIFO和反压语义，但每次最多计算32条连续4 KiB命令。

        输入：包含ASU六级硬件参数和最大批量的backend配置。

        输出：无；初始化空时序窗口、可观察事件堆和统计。
        """
        self.chunk_size_bytes = backend_config["chunk_size_bytes"]
        self.max_batch_commands = backend_config.get(
            "exact_batch_max_commands",
            32,
        )
        # 大规模实验可只关闭高频诊断统计；该开关不参与
        # 任何时序、FIFO、反压、接收或完成方程。
        self.collect_stage_peaks = backend_config.get(
            "collect_stage_peak_statistics",
            True,
        )
        self.collect_nand_service_events = backend_config.get(
            "collect_nand_service_events",
            True,
        )
        self.current_time = 0
        self.events = []
        self.event_sequence = 0
        self.capacity_event_time = None
        self.next_request_accept_time = 0

        self.completed_requests = []
        self.pending_completion_records = deque()
        self.completed_byte_count = 0
        self.first_submit_time = None
        self.last_completion_time = None
        self.nand_service_events = []
        self.nand_read_bandwidth_bytes_per_second = backend_config["nand"][
            "read_bandwidth_bytes_per_second"
        ]

        self._load_hardware_parameters(backend_config)
        self._create_timing_state()

    def _load_hardware_parameters(self, backend_config):
        """功能：将ASU YAML参数转换成精确整数时间和容量。

        目的：批量模式与详细模式共用`time_utils.py`的
        15000 units/us时钟，不引入任何浮点近似。

        输入：完整backend配置字典。

        输出：无；保存各级间隔、延迟和槽位数。
        """
        fcp = backend_config["fcp"]
        bcp = backend_config["bcp"]
        nfi = backend_config["nfi"]
        nand = backend_config["nand"]
        bdp = backend_config["bdp"]
        das = backend_config["das"]
        media_mode = backend_config["media_mode"].lower()

        self.fcp_slots = fcp["atom_slots"]
        self.fcp_interval = rate_to_interval(fcp["atom_rate_per_second"])
        self.fcp_latency = us_to_time(fcp["latency_us"])
        self.bcp_waiting_capacity = bcp["input_queue_depth_commands"]
        self.bcp_inflight = bcp["max_inflight_commands"]
        self.bcp_interval = rate_to_interval(bcp["command_rate_per_second"])
        self.bcp_latency = us_to_time(bcp["latency_us"])
        self.nfi_slots = nfi["command_slots"]
        self.nfi_interval = rate_to_interval(nfi["command_rate_per_second"])
        self.nfi_latency = us_to_time(nfi["latency_us"])
        self.nand_slots = nand["command_slots"]
        self.nand_interval = bandwidth_to_interval(
            self.chunk_size_bytes,
            nand["read_bandwidth_bytes_per_second"],
        )
        self.nand_latency = us_to_time(
            nand[f"{media_mode}_latency_us"]
        )
        self.bdp_slots = bdp["command_slots"]
        self.bdp_interval = rate_to_interval(bdp["command_rate_per_second"])
        self.bdp_latency = us_to_time(bdp["latency_us"])
        self.das_slots = das["atom_slots"]
        self.das_interval = rate_to_interval(das["atom_rate_per_second"])
        self.das_latency = us_to_time(das["latency_us"])

    def _create_timing_state(self):
        """功能：创建各级递推需要的有界时刻窗口。

        目的：只保存最近一个硬件容量的历史，使内存用量
        与请求总数无关。

        输入：无；使用已加载的各级容量。

        输出：无；初始化计数器、下次启动时刻和峰值跟踪器。
        """
        self.atom_count = 0
        self.command_count = 0
        self.next_fcp_start = 0
        self.next_bcp_start = 0
        self.next_nfi_start = 0
        self.next_nand_start = 0
        self.next_bdp_start = 0
        self.next_das_start = 0

        self.fcp_departures = IndexedTimeWindow(self.fcp_slots + 1)
        self.bcp_starts = IndexedTimeWindow(
            self.bcp_waiting_capacity + 2
        )
        self.bcp_departures = IndexedTimeWindow(self.bcp_inflight + 1)
        self.nfi_departures = IndexedTimeWindow(self.nfi_slots + 1)
        self.nand_departures = IndexedTimeWindow(self.nand_slots + 1)
        self.bdp_departures = IndexedTimeWindow(self.bdp_slots + 1)
        self.das_departures = IndexedTimeWindow(self.das_slots + 1)

        self.previous_bcp_departure = 0
        self.previous_nfi_departure = 0
        self.previous_nand_departure = 0
        self.previous_bdp_departure = 0
        self.last_fcp_arrival = 0
        self.last_das_departure = 0

        self.fcp_peak = StagePeakTracker()
        self.bcp_peak = StagePeakTracker()
        self.nfi_peak = StagePeakTracker()
        self.nand_peak = StagePeakTracker()
        self.bdp_peak = StagePeakTracker()
        self.das_peak = StagePeakTracker()

    def synchronize_time(self, current_time):
        """功能：将后端可见时钟同步到全局事件时刻。

        目的：`can_accept_request()`只依据已处理的精确容量时刻
        判断，不在查询函数中隐式推进任何事件。

        输入：ASU内部整数时刻。

        输出：无；更新当前可见时刻。
        """
        self.current_time = current_time

    def can_accept_request(self):
        """功能：判断FCP拆分器当前能否接收新完整IO。

        目的：精确复现“前一请求已全部填入FCP且至少有一个
        原子槽可用”的非阻塞入口条件。

        输入：无；读取当前同步时钟和下次可接收时刻。

        输出：当前时刻可接收时返回True。
        """
        return self.current_time >= self.next_request_accept_time

    def _schedule_observable_event(
        self,
        event_time,
        priority,
        event_type,
        payload=None,
    ):
        """功能：向局部堆加入一个对QoS或LLM可观察的事件。

        目的：只保留FCP容量释放和完整IO完成，避免为每条
        4 KiB命令生成六级start/complete Python对象。

        输入：整数时刻、优先级、事件类型和可选完成记录。

        输出：无；事件按时刻、优先级和稳定序号入堆。
        """
        self.event_sequence += 1
        heapq.heappush(self.events, (
            event_time,
            priority,
            self.event_sequence,
            event_type,
            payload,
        ))

    def _schedule_atom(self, request_id, queue_id, submit_time):
        """功能：精确计算一个8 KiB原子所含两条4 KiB命令的全流程时刻。

        目的：逐级应用与详细事件模型等价的FIFO、启动间隔、
        固定延迟、有限槽位和下游反压max-plus约束。

        输入：原始请求ID、QoS Queue ID和请求被SSD接收的整数时刻。

        输出：该原子最后在DAS完成的整数时刻。
        """
        atom_index = self.atom_count
        first_command_index = self.command_count

        fcp_arrival = submit_time
        if atom_index >= self.fcp_slots:
            fcp_arrival = max(
                fcp_arrival,
                self.fcp_departures.get(atom_index - self.fcp_slots),
            )
        fcp_start = max(fcp_arrival, self.next_fcp_start)
        self.next_fcp_start = fcp_start + self.fcp_interval
        fcp_completion = fcp_start + self.fcp_latency

        required_bcp_start_index = (
            first_command_index + 1 - self.bcp_waiting_capacity
        )
        fcp_departure = fcp_completion
        bcp_capacity_released_first = False
        if required_bcp_start_index >= 0:
            required_bcp_start = self.bcp_starts.get(
                required_bcp_start_index
            )
            if required_bcp_start >= fcp_departure:
                bcp_capacity_released_first = True
            fcp_departure = max(fcp_departure, required_bcp_start)

        self.fcp_departures.append(fcp_departure)
        self.last_fcp_arrival = fcp_arrival
        self.atom_count += 1

        atom_departure = 0
        for chunk_offset in (0, 1):
            command_index = first_command_index + chunk_offset
            atom_departure = self._schedule_command(
                request_id=request_id,
                queue_id=queue_id,
                command_index=command_index,
                atom_index=atom_index,
                bcp_arrival=fcp_departure,
                bcp_capacity_released_first=(
                    bcp_capacity_released_first
                    and chunk_offset == 0
                ),
            )

        self.command_count += 2
        if self.collect_stage_peaks:
            self.fcp_peak.accept(
                arrival_time=fcp_arrival,
                start_time=fcp_start,
                departure_time=fcp_departure,
            )
        self.last_das_departure = atom_departure
        return atom_departure

    def _schedule_command(
        self,
        request_id,
        queue_id,
        command_index,
        atom_index,
        bcp_arrival,
        bcp_capacity_released_first,
    ):
        """功能：计算一条4 KiB命令从BCP到DAS的精确时序。

        目的：用有界历史窗口表达每一级的槽位释放时刻，
        不改变命令顺序、反压时刻或下游可见完成时刻。

        输入：请求/Queue ID、全局4 KiB顺序号、原子顺序号、
        BCP成对到达时刻，以及到达是否由同时刻BCP start释放空间。

        输出：当前AS原子已形成时返回其完成时刻；首个分片返回0。
        """
        bcp_start = max(bcp_arrival, self.next_bcp_start)
        if command_index >= self.bcp_inflight:
            bcp_start = max(
                bcp_start,
                self.bcp_departures.get(
                    command_index - self.bcp_inflight
                ),
            )
        self.next_bcp_start = bcp_start + self.bcp_interval
        bcp_completion = bcp_start + self.bcp_latency

        nfi_arrival = max(bcp_completion, self.previous_bcp_departure)
        if command_index >= self.nfi_slots:
            nfi_arrival = max(
                nfi_arrival,
                self.nfi_departures.get(command_index - self.nfi_slots),
            )
        bcp_departure = nfi_arrival
        self.bcp_starts.append(bcp_start)
        self.bcp_departures.append(bcp_departure)
        self.previous_bcp_departure = bcp_departure

        nfi_start = max(nfi_arrival, self.next_nfi_start)
        self.next_nfi_start = nfi_start + self.nfi_interval
        nfi_completion = nfi_start + self.nfi_latency

        nand_arrival = max(nfi_completion, self.previous_nfi_departure)
        if command_index >= self.nand_slots:
            nand_arrival = max(
                nand_arrival,
                self.nand_departures.get(command_index - self.nand_slots),
            )
        nfi_departure = nand_arrival
        self.nfi_departures.append(nfi_departure)
        self.previous_nfi_departure = nfi_departure

        nand_start = max(nand_arrival, self.next_nand_start)
        self.next_nand_start = nand_start + self.nand_interval
        nand_completion = nand_start + self.nand_latency
        if self.collect_nand_service_events:
            self.nand_service_events.append({
                "start_time_us": time_to_us(nand_start),
                "request_id": request_id,
                "queue_id": queue_id,
                "size_bytes": self.chunk_size_bytes,
            })

        bdp_arrival = max(nand_completion, self.previous_nand_departure)
        if command_index >= self.bdp_slots:
            bdp_arrival = max(
                bdp_arrival,
                self.bdp_departures.get(command_index - self.bdp_slots),
            )
        nand_departure = bdp_arrival
        self.nand_departures.append(nand_departure)
        self.previous_nand_departure = nand_departure

        bdp_start = max(bdp_arrival, self.next_bdp_start)
        self.next_bdp_start = bdp_start + self.bdp_interval
        bdp_completion = bdp_start + self.bdp_latency

        bdp_departure = max(
            bdp_completion,
            self.previous_bdp_departure,
        )
        if command_index % 2 == 1 and atom_index >= self.das_slots:
            bdp_departure = max(
                bdp_departure,
                self.das_departures.get(atom_index - self.das_slots),
            )
        self.bdp_departures.append(bdp_departure)
        self.previous_bdp_departure = bdp_departure

        das_departure = 0
        if command_index % 2 == 1:
            das_start = max(bdp_departure, self.next_das_start)
            self.next_das_start = das_start + self.das_interval
            das_departure = das_start + self.das_latency
            self.das_departures.append(das_departure)
            if self.collect_stage_peaks:
                self.das_peak.accept(
                    arrival_time=bdp_departure,
                    start_time=das_start,
                    departure_time=das_departure,
                )

        if self.collect_stage_peaks:
            self.bcp_peak.accept(
                arrival_time=bcp_arrival,
                start_time=bcp_start,
                departure_time=bcp_departure,
                starts_at_same_time_are_earlier=bcp_capacity_released_first,
            )
            self.nfi_peak.accept(
                arrival_time=nfi_arrival,
                start_time=nfi_start,
                departure_time=nfi_departure,
            )
            self.nand_peak.accept(
                arrival_time=nand_arrival,
                start_time=nand_start,
                departure_time=nand_departure,
            )
            self.bdp_peak.accept(
                arrival_time=bdp_arrival,
                start_time=bdp_start,
                departure_time=bdp_departure,
            )
        return das_departure

    def submit_request(self, request, current_time):
        """功能：批量计算一个完整SSD请求的六级流水时序。

        目的：以最多32条4 KiB命令为一个计算批次，生成与详细
        模式相同的FCP再接收时刻、NAND启动时刻和DAS完成时刻。

        输入：完整IO描述符和SSD接收它的ASU整数时刻。

        输出：无；登记下次容量事件和完整IO完成事件。
        """
        request_id = request["request_id"]
        queue_id = request["queue_id"]
        size_bytes = request["size_bytes"]
        chunk_count = (
            size_bytes + self.chunk_size_bytes - 1
        ) // self.chunk_size_bytes
        padded_chunk_count = chunk_count + chunk_count % 2
        atom_count = padded_chunk_count // 2

        if self.first_submit_time is None:
            self.first_submit_time = current_time

        remaining_atoms = atom_count
        completion_time = current_time
        atoms_per_batch = max(1, self.max_batch_commands // 2)
        while remaining_atoms:
            current_batch_atoms = min(remaining_atoms, atoms_per_batch)
            for _ in range(current_batch_atoms):
                completion_time = self._schedule_atom(
                    request_id,
                    queue_id,
                    current_time,
                )
            remaining_atoms -= current_batch_atoms

        completion_time_us = time_to_us(completion_time)
        submit_time_us = time_to_us(current_time)
        completion_record = {
            "request_id": request_id,
            "queue_id": queue_id,
            "size_bytes": size_bytes,
            "backend_submit_time_us": submit_time_us,
            "backend_completion_time_us": completion_time_us,
            "backend_latency_us": completion_time_us - submit_time_us,
        }
        self._schedule_observable_event(
            completion_time,
            COMPLETE_PRIORITY,
            "complete",
            completion_record,
        )

        if self.atom_count < self.fcp_slots:
            next_accept_time = self.last_fcp_arrival
        else:
            next_accept_time = max(
                self.last_fcp_arrival,
                self.fcp_departures.get(
                    self.atom_count - self.fcp_slots
                ),
            )
        self.next_request_accept_time = next_accept_time
        if next_accept_time > current_time:
            self.capacity_event_time = next_accept_time
            self._schedule_observable_event(
                next_accept_time,
                CAPACITY_PRIORITY,
                "capacity",
            )
        else:
            self.capacity_event_time = None

    def next_event_time(self):
        """功能：返回最近的FCP容量或完整IO完成时刻。

        目的：让联合仿真只在对上层可观察的精确边界唤醒SSD。

        输入：无。

        输出：最近ASU整数时刻；无事件时返回None。
        """
        if not self.events:
            return None
        return self.events[0][0]

    def process_events_at(self, event_time):
        """功能：处理指定时刻的全部可观察SSD事件。

        目的：保持同时刻稳定序号，并且仅在DAS最后一个原子
        完成时向上层发布一次完整IO记录。

        输入：必须等于当前堆顶的ASU整数时刻。

        输出：无；更新当前时钟、容量标记和待发布完成队列。
        """
        self.current_time = event_time
        while self.events and self.events[0][0] == event_time:
            _, _, _, event_type, payload = heapq.heappop(self.events)
            if event_type == "capacity":
                if self.capacity_event_time == event_time:
                    self.capacity_event_time = None
                continue

            self.completed_requests.append(payload)
            self.pending_completion_records.append(payload)
            self.completed_byte_count += payload["size_bytes"]
            self.last_completion_time = event_time

    def process_next_event_time(self):
        """功能：跳到并处理最近可观察事件。

        目的：为SSD独立排空测试复用与联合仿真相同的精确路径。

        输入：无。

        输出：已处理的ASU整数时刻；无事件时返回None。
        """
        event_time = self.next_event_time()
        if event_time is None:
            return None
        self.process_events_at(event_time)
        return event_time

    def drain_completion_records(self):
        """功能：取出尚未发布给LLM的完整IO完成记录。

        目的：确保每条原始IO只在对应DAS原子全部完成后通知一次。

        输入：无。

        输出：按精确DAS完成顺序排列的新记录列表。
        """
        records = list(self.pending_completion_records)
        self.pending_completion_records.clear()
        return records

    def completed_bytes(self):
        """功能：返回已经完成的原始IO字节总数。

        目的：保留与详细后端一致的守恒统计接口。

        输入：无。

        输出：已完成原始IO的整数Byte数。
        """
        return self.completed_byte_count

    def stage_statistics(self):
        """功能：返回六级流水的启动、完成和峰值统计。

        目的：让`batched_exact`的诊断字段与`detailed`保持相同schema，
        便于将统计本身纳入精度对照。

        输入：无。

        输出：按FCP/BCP/NFI/NAND/BDP/DAS保存的统计字典。
        """
        atom_count = self.atom_count
        command_count = self.command_count
        return {
            "FCP": {
                "started": atom_count,
                "completed": atom_count,
                "max_occupied": self.fcp_peak.max_occupied,
            },
            "BCP": self._ordinary_stage_statistics(
                command_count,
                self.bcp_peak,
            ),
            "NFI": self._ordinary_stage_statistics(
                command_count,
                self.nfi_peak,
            ),
            "NAND": self._ordinary_stage_statistics(
                command_count,
                self.nand_peak,
            ),
            "BDP": self._ordinary_stage_statistics(
                command_count,
                self.bdp_peak,
            ),
            "DAS": self._ordinary_stage_statistics(
                atom_count,
                self.das_peak,
            ),
        }

    def _ordinary_stage_statistics(self, item_count, peak_tracker):
        """功能：生成一级普通流水的标准统计字典。

        目的：避免六级结果重复组装字段，同时保持与详细模型一致。

        输入：该级处理项数和对应峰值跟踪器。

        输出：`started/completed/max_waiting/max_occupied`字典。
        """
        return {
            "started": item_count,
            "completed": item_count,
            "max_waiting": peak_tracker.max_waiting,
            "max_occupied": peak_tracker.max_occupied,
        }
