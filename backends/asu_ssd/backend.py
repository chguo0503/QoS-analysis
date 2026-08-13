#!/usr/bin/env python3
"""FCP、BCP、NFI、NAND、BDP和DAS组成的ASU后端离散事件模型。"""

import heapq  # heapq按照时间顺序保存后端启动和完成事件。
from collections import deque  # deque保存尚未向上层发布的完成记录。

from .pipeline_stage import DASPairer, FCPStage, PipelineStage  # 导入三个简单阶段模型。
from .time_utils import bandwidth_to_interval, rate_to_interval, time_to_us, us_to_time  # 导入时间换算。


COMPLETE_PRIORITY = 0  # 同一时刻先处理完成事件并释放下游资源。
START_PRIORITY = 1  # 完成和输出处理后再启动新的流水线命令。


class ASUBackend:
    """管理ASU所有阶段、反压关系、事件日历和原始请求完成状态。

    events保存未来的模块启动和完成事件。
    request_states将4 KiB内部命令和8 KiB原子重新关联到完整SSD请求。
    completed_requests按DAS完成顺序保存独立的完成记录。
    pending_completion_records保存等待上层发布的新完成记录。
    FCP只允许一个活动拆分请求，但分片可以随着槽位释放逐步进入。
    """

    def __init__(self, backend_config):
        """根据YAML基础参数创建全部后端阶段。"""
        self.chunk_size_bytes = backend_config["chunk_size_bytes"]  # 保存后端内部4 KiB命令大小。
        self.current_time = 0  # 后端内部当前时间从0开始。
        self.events = []  # 最小堆保存所有未来启动和完成事件。
        self.event_sequence = 0  # 相同时间和优先级下使用递增序号保持稳定顺序。
        self.request_states = {}  # 保存每个原始请求已经完成多少个DAS原子。
        self.completed_requests = []  # 保存已经在DAS完成的独立记录。
        self.pending_completion_records = deque()  # 保存还没有被模拟器发布的完成记录。
        self.completed_byte_count = 0  # 累计已完成SSD请求的字节数。
        self.first_submit_time = None  # 记录第一个请求进入FCP的时间。
        self.last_completion_time = None  # 记录最近一个请求完成DAS的时间。
        self.nand_read_bandwidth_bytes_per_second = backend_config["nand"]["read_bandwidth_bytes_per_second"]  # 保存NAND配置的物理读带宽上限。
        self.nand_service_events = []  # 按启动顺序保存每个NAND 4 KiB命令的队列归属。
        self._create_stages(backend_config)  # 根据配置创建FCP到DAS的全部模块。

    def _create_stages(self, backend_config):
        """创建六级流水线以及DAS相邻分片配对器。"""
        fcp_config = backend_config["fcp"]  # 读取FCP基础参数。
        self.fcp = FCPStage(  # FCP使用专用的4 KiB流式拆分模型。
            atom_slots=fcp_config["atom_slots"],  # 30个共享8 KiB原子槽。
            start_interval=rate_to_interval(fcp_config["atom_rate_per_second"]),  # 6M原子/s。
            latency=us_to_time(fcp_config["latency_us"]),  # 固定5 us处理延迟。
        )

        bcp_config = backend_config["bcp"]  # 读取BCP基础参数。
        self.bcp = PipelineStage(  # BCP具有独立等待FIFO和20个在途位置。
            name="BCP",  # 保存阶段名称。
            start_interval=rate_to_interval(bcp_config["command_rate_per_second"]),  # 0.1 us间隔。
            latency=us_to_time(bcp_config["latency_us"]),  # 固定2 us延迟。
            max_inflight=bcp_config["max_inflight_commands"],  # 20个处理或阻塞完成位置。
            waiting_capacity=bcp_config["input_queue_depth_commands"],  # 1024项等待FIFO。
            total_slot_mode=False,  # 等待FIFO不计入20个处理位置。
        )

        nfi_config = backend_config["nfi"]  # 读取NFI基础参数。
        self.nfi = PipelineStage(  # NFI的等待、处理和完成阻塞共用10个槽。
            name="NFI",  # 保存阶段名称。
            start_interval=rate_to_interval(nfi_config["command_rate_per_second"]),  # 0.1 us间隔。
            latency=us_to_time(nfi_config["latency_us"]),  # 固定1 us延迟。
            max_inflight=nfi_config["command_slots"],  # 10个共享槽位。
            total_slot_mode=True,  # 所有状态共同占用10个槽。
        )

        nand_config = backend_config["nand"]  # 读取NAND带宽、延迟和容量。
        media_mode = backend_config["media_mode"].lower()  # 将TLC或SLC转换成小写字段前缀。
        nand_latency_us = nand_config[f"{media_mode}_latency_us"]  # 选择本次仿真的介质延迟。
        self.nand = PipelineStage(  # NAND使用聚合带宽和512个共享命令槽。
            name="NAND",  # 保存阶段名称。
            start_interval=bandwidth_to_interval(  # 根据4 KiB和40 GB/s推导0.1024 us。
                self.chunk_size_bytes,  # 每个NAND读取命令固定为4 KiB。
                nand_config["read_bandwidth_bytes_per_second"],  # 整盘聚合读取带宽40 GB/s。
            ),
            latency=us_to_time(nand_latency_us),  # TLC为50 us，SLC为23 us。
            max_inflight=nand_config["command_slots"],  # 512个共享命令槽。
            total_slot_mode=True,  # 等待、读取和完成阻塞共同占用512个槽。
        )

        bdp_config = backend_config["bdp"]  # 读取BDP基础参数。
        self.bdp = PipelineStage(  # BDP使用20个共享命令槽。
            name="BDP",  # 保存阶段名称。
            start_interval=rate_to_interval(bdp_config["command_rate_per_second"]),  # 0.1 us间隔。
            latency=us_to_time(bdp_config["latency_us"]),  # 固定2 us延迟。
            max_inflight=bdp_config["command_slots"],  # 20个共享槽位。
            total_slot_mode=True,  # 等待、处理和完成阻塞共同占用20个槽。
        )

        das_config = backend_config["das"]  # 读取DAS基础参数。
        self.das = PipelineStage(  # DAS处理重新配对完成的8 KiB DMA原子。
            name="DAS",  # 保存阶段名称。
            start_interval=rate_to_interval(das_config["atom_rate_per_second"]),  # 6M原子/s。
            latency=us_to_time(das_config["latency_us"]),  # 固定3 us延迟。
            max_inflight=das_config["atom_slots"],  # 18个共享原子槽。
            total_slot_mode=True,  # 等待、DMA和阻塞完成共同占用18个槽。
        )
        self.das_pairer = DASPairer(self.das)  # 创建按request_id和相邻chunk编号配对的缓冲。
        self.stages = {  # 建立名称到阶段对象的映射，方便事件处理。
            "FCP": self.fcp,  # 保存FCP阶段。
            "BCP": self.bcp,  # 保存BCP阶段。
            "NFI": self.nfi,  # 保存NFI阶段。
            "NAND": self.nand,  # 保存NAND阶段。
            "BDP": self.bdp,  # 保存BDP阶段。
            "DAS": self.das,  # 保存DAS阶段。
        }

    def _schedule_event(self, event_time, priority, event_type, stage_name, command=None):
        """向最小堆加入一个稳定排序的后端事件。"""
        self.event_sequence += 1  # 每加入一个事件就生成新的稳定序号。
        event = (  # 使用普通元组保存事件，避免复杂类结构。
            event_time,  # 第一项是整数时间，最小堆优先处理最早时间。
            priority,  # 第二项确保同一时刻先完成后启动。
            self.event_sequence,  # 第三项确保完全相同时仍保持加入顺序。
            event_type,  # 保存start或complete事件类型。
            stage_name,  # 保存事件属于哪个后端模块。
            command,  # complete事件携带正在完成的命令。
        )
        heapq.heappush(self.events, event)  # 将事件加入未来事件日历。

    def _schedule_stage_start(self, stage, current_time):
        """在阶段有命令可启动时安排唯一的下一次启动事件。"""
        if not stage.can_start():  # 没有等待命令或处理位置时不能启动。
            return  # 等待下一个输入或下游释放事件再次尝试。
        if stage.start_event_pending:  # 已经安排过启动事件时不重复加入。
            return  # 保留现有启动事件即可。
        start_time = max(current_time, stage.next_start_time)  # 同时满足当前时间和启动间隔。
        stage.start_event_pending = True  # 标记这个阶段已有一个待处理启动事件。
        self._schedule_event(start_time, START_PRIORITY, "start", stage.name)  # 加入事件日历。

    def _schedule_all_starts(self, current_time):
        """让所有当前具备条件的阶段安排下一次启动事件。"""
        for stage in self.stages.values():  # 按FCP到DAS顺序检查全部阶段。
            self._schedule_stage_start(stage, current_time)  # 每个阶段最多保留一个启动事件。

    def can_accept_request(self):
        """返回FCP拆分器是否能开始接收新的可变大小描述符。"""
        return self.fcp.can_accept_request()  # 只需一个空原子槽，其余命令后续流式进入。

    def submit_request(self, request, current_time):
        """复制请求的SSD必要信息，并让FCP逐步切成4 KiB分片。

        上层字典只用于读取请求编号、队列和长度。SSD内部另建流水线描述符，
        不再往原请求字典中写入分片或完成字段。
        """
        request_id = request["request_id"]  # 保存上层用于匹配完成的唯一编号。
        queue_id = request["queue_id"]  # 保存请求所属的QoS队列，供完成统计使用。
        size_bytes = request["size_bytes"]  # 保存这个完整SSD请求的实际传输长度。
        chunk_count = (size_bytes + self.chunk_size_bytes - 1) // self.chunk_size_bytes  # 根据请求长度推导4 KiB命令数。
        padded_chunk_count = chunk_count + chunk_count % 2  # 奇数命令补齐后再两个组成8 KiB原子。
        submit_time_us = time_to_us(current_time)  # 记录后端接收描述符时刻。
        pipeline_request = {  # 构造仅在SSD内部流转的拆分描述符。
            "request_id": request_id,  # 内部分片沿用上层请求编号。
            "queue_id": queue_id,  # 内部分片保留原QoS队列。
            "chunk_size_bytes": self.chunk_size_bytes,  # FCP每次产生一个4 KiB分片。
            "backend_chunk_count": padded_chunk_count,  # 奇数分片按DAS的8 KiB原子要求补齐。
        }
        request_state = {  # 保存DAS完成这个完整请求所需的最少状态。
            "request_id": request_id,  # 完成时使用这个编号向上层通知。
            "queue_id": queue_id,  # 完成记录保留来源QoS队列。
            "size_bytes": size_bytes,  # 完成时累计实际传输字节。
            "backend_submit_time_us": submit_time_us,  # 完成时用于计算SSD内部延迟。
            "atom_count": padded_chunk_count // 2,  # 保存本请求最终应完成的8 KiB原子数。
            "completed_atoms": 0,  # 初始时尚未完成任何DAS原子。
        }
        self.request_states[request_id] = request_state  # 使用request_id关联所有内部分片。
        self.fcp.submit_request(pipeline_request)  # FCP立即填入当前能容纳的4 KiB分片。
        if self.first_submit_time is None:  # 只在第一个请求提交时记录开始时间。
            self.first_submit_time = current_time  # 联合仿真使用这个时间计算端到端吞吐。
        self._schedule_stage_start(self.fcp, current_time)  # 新形成的8 KiB原子可以进入FCP流水线。

    def next_event_time(self):
        """返回最近的后端事件时间；没有事件时返回None。"""
        if not self.events:  # 空事件日历表示当前后端没有未来动作。
            return None  # 调用者可以等待QoS或新请求事件。
        return self.events[0][0]  # 最小堆第一项就是最早整数时间。

    def _handle_start(self, stage, current_time):
        """启动阶段FIFO中的一个命令并安排固定延迟后的完成事件。"""
        stage.start_event_pending = False  # 当前唯一启动事件已经开始处理。
        if not stage.can_start():  # 下游反压可能使预定启动暂时失效。
            return  # 等待以后释放槽位时重新安排。
        command = stage.start_next()  # 从阶段等待FIFO取出最早命令。
        if stage.name == "NAND":  # NAND命令开始占用带宽时记录其原始QoS队列。
            self.nand_service_events.append({  # 每条记录对应一个实际启动的4 KiB内部命令。
                "start_time_us": time_to_us(current_time),  # 记录NAND服务开始时刻。
                "request_id": command["request_id"],  # 保留上层请求编号便于校验。
                "queue_id": command["queue_id"],  # 保留该命令所属的QoS队列。
                "size_bytes": command["size_bytes"],  # 记录本次NAND服务的字节数。
            })
        stage.next_start_time = current_time + stage.start_interval  # 计算下一次允许启动的时刻。
        completion_time = current_time + stage.latency  # 计算这个命令固定延迟后的完成时刻。
        self._schedule_event(  # 将命令完成事件加入最小堆。
            completion_time,  # 保存未来完成时刻。
            COMPLETE_PRIORITY,  # 完成事件优先于同一时刻的启动事件。
            "complete",  # 保存事件类型。
            stage.name,  # 保存所属模块名称。
            command,  # 携带完成时需要的命令描述符。
        )
        self._schedule_stage_start(stage, current_time)  # 如果仍有命令则按照间隔安排下一个启动。

    def _complete_original_request(self, atom, current_time):
        """记录一个DAS原子，最后一个原子到达时生成独立完成记录。"""
        request_state = self.request_states[atom["request_id"]]  # 找到原始可变大小SSD请求状态。
        request_state["completed_atoms"] += 1  # 当前请求完成的8 KiB原子数量加1。
        if request_state["completed_atoms"] != request_state["atom_count"]:  # 尚未完成本请求的全部原子。
            return  # 等待这个request_id的其他DAS原子。
        completion_us = time_to_us(current_time)  # 将整数完成时间换算成微秒。
        completion_record = {  # 完成结果与上层原请求字典彻底分离。
            "request_id": request_state["request_id"],  # 上层用这个编号匹配一次完成。
            "queue_id": request_state["queue_id"],  # 保留来源队列供后续统计。
            "size_bytes": request_state["size_bytes"],  # 记录完整请求的传输字节数。
            "backend_submit_time_us": request_state["backend_submit_time_us"],  # 记录进入FCP的时刻。
            "backend_completion_time_us": completion_us,  # 记录最后一个DAS原子完成时刻。
            "backend_latency_us": completion_us - request_state["backend_submit_time_us"],  # 保存SSD内部延迟。
        }
        self.completed_requests.append(completion_record)  # 按DAS完成顺序保存独立统计记录。
        self.pending_completion_records.append(completion_record)  # 等待模拟器在当前事件处理完后发布。
        del self.request_states[atom["request_id"]]  # 请求已经完成，不再保留内部原子计数。
        self.completed_byte_count += request_state["size_bytes"]  # 累计已完成的上层请求字节。
        self.last_completion_time = current_time  # 更新最近完成时间供吞吐统计。

    def drain_completion_records(self):
        """取出当前尚未发布的全部完成记录，并清空待通知队列。"""
        records = list(self.pending_completion_records)  # 保留DAS产生完成的先后顺序。
        self.pending_completion_records.clear()  # 已发布记录不再重复通知上层。
        return records  # 模拟器只会对这批新完成执行一次回调。

    def _handle_complete(self, stage, command, current_time):
        """完成一个阶段命令；普通阶段等待下游，DAS直接完成原始请求。"""
        if stage.name == "DAS":  # 第一版认为DAS输出网络始终可以接收。
            stage.finish_and_release()  # DAS完成后立即释放一个原子槽。
            self._complete_original_request(command, current_time)  # 更新原始SSD请求完成状态。
            return  # DAS没有后续ASU阶段需要传输。
        stage.finish(command)  # 其他阶段完成后先进入阻塞输出区并继续占用槽位。

    def _move_outputs(self, current_time):
        """按照DAS到FCP的反向顺序传输所有当前可移动的完成命令。"""
        moved_any_command = True  # True表示上一轮至少释放了一个上游槽位。

        while moved_any_command:  # 反复传播下游空位，直到整条流水线无法继续移动。
            moved_any_command = False  # 每轮开始先假设没有命令能够移动。

            while self.bdp.completed:  # BDP可能有多个已完成4 KiB等待DAS。
                command = self.bdp.completed[0]  # 只查看最早完成的命令以保持FIFO。
                if not self.das_pairer.can_accept(command):  # 第二个相邻分片需要DAS原子槽。
                    break  # DAS满时保留命令并向NAND逐级反压。
                command = self.bdp.pop_completed()  # DAS可接收后才释放BDP槽位。
                formed_atom = self.das_pairer.accept(command)  # 暂存首片或形成相邻8 KiB原子。
                if formed_atom:  # 只有形成完整原子时DAS才需要启动处理。
                    self._schedule_stage_start(self.das, current_time)  # 安排DAS DMA启动。
                moved_any_command = True  # BDP至少释放了一个槽位。

            while self.nand.completed and self.bdp.can_accept():  # BDP有空槽时接收NAND数据。
                command = self.nand.pop_completed()  # 释放一个NAND命令槽。
                self.bdp.accept(command)  # 将4 KiB数据命令送入BDP。
                self._schedule_stage_start(self.bdp, current_time)  # 安排BDP处理启动。
                moved_any_command = True  # NAND空位可以继续向上游传播。

            while self.nfi.completed and self.nand.can_accept():  # NAND有空槽时接收NFI命令。
                command = self.nfi.pop_completed()  # 释放一个NFI命令槽。
                self.nand.accept(command)  # 将4 KiB读取命令送入NAND。
                self._schedule_stage_start(self.nand, current_time)  # 安排NAND带宽受限启动。
                moved_any_command = True  # NFI空位可以继续向上游传播。

            while self.bcp.completed and self.nfi.can_accept():  # NFI有空槽时接收BCP输出。
                command = self.bcp.pop_completed()  # 释放一个BCP在途位置。
                self.nfi.accept(command)  # 将4 KiB命令送入NFI共享槽。
                self._schedule_stage_start(self.nfi, current_time)  # 安排NFI启动。
                moved_any_command = True  # BCP处理位置得到释放。

            while self.fcp.completed and self.bcp.can_accept(2):  # BCP至少有两个FIFO位置才接收原子。
                atom = self.fcp.pop_completed()  # 释放一个FCP原子槽并继续流式填入4 KiB。
                self.bcp.accept(atom["chunks"][0])  # 原子中的第一个4 KiB进入BCP。
                self.bcp.accept(atom["chunks"][1])  # 原子中的第二个4 KiB原子性进入BCP。
                self._schedule_stage_start(self.bcp, current_time)  # 安排BCP处理两个新命令。
                self._schedule_stage_start(self.fcp, current_time)  # 新填成的FCP原子可以继续启动。
                moved_any_command = True  # FCP输出成功，可能允许拆分器继续前进。

    def process_events_at(self, event_time):
        """处理指定时刻的全部后端事件以及由它们产生的同时间事件。"""
        self.current_time = event_time  # 将后端时钟跳到当前离散事件时刻。

        while self.events and self.events[0][0] == event_time:  # 持续处理同一时间戳事件。
            event = heapq.heappop(self.events)  # 取出时间和优先级最小的事件。
            event_type = event[3]  # 读取start或complete。
            stage = self.stages[event[4]]  # 根据名称找到对应阶段对象。
            command = event[5]  # complete事件携带命令，start事件为None。
            if event_type == "complete":  # 完成事件优先释放下游资源。
                self._handle_complete(stage, command, event_time)  # 更新阶段完成状态。
            else:  # 另一种事件类型是流水线启动。
                self._handle_start(stage, event_time)  # 启动一个等待命令。
            self._move_outputs(event_time)  # 每个事件后立即尝试从下游向上游释放反压。
            self._schedule_all_starts(event_time)  # 为所有新具备条件的阶段安排启动。

        self._move_outputs(event_time)  # 最后再传播一次可能剩余的下游空位。
        self._schedule_all_starts(event_time)  # 确保所有等待阶段都有且只有一个启动事件。

    def process_next_event_time(self):
        """跳到最近后端事件，处理该时刻全部动作并返回整数时间。"""
        event_time = self.next_event_time()  # 读取事件日历中最早的时间。
        if event_time is None:  # 没有事件时后端无法自行推进。
            return None  # 调用者需要提交新请求后再继续。
        self.process_events_at(event_time)  # 处理这个时刻的所有启动、完成和传输。
        return event_time  # 返回处理后的时刻供调用方或QoS继续提交。

    def completed_bytes(self):
        """返回已经在DAS完成的原始请求总Byte数。"""
        return self.completed_byte_count  # 请求大小可变，直接返回累计完成字节。

    def stage_statistics(self):
        """返回各阶段的启动、完成和占用峰值统计。"""
        return {  # 使用普通嵌套字典返回关键统计。
            "FCP": {  # FCP以8 KiB原子计数。
                "started": self.fcp.started_count,  # FCP累计启动原子数。
                "completed": self.fcp.completed_count,  # FCP累计完成原子数。
                "max_occupied": self.fcp.max_occupied_slots,  # FCP最大原子槽占用。
            },
            "BCP": self._ordinary_stage_statistics(self.bcp),  # BCP以4 KiB命令计数。
            "NFI": self._ordinary_stage_statistics(self.nfi),  # NFI以4 KiB命令计数。
            "NAND": self._ordinary_stage_statistics(self.nand),  # NAND以4 KiB命令计数。
            "BDP": self._ordinary_stage_statistics(self.bdp),  # BDP以4 KiB命令计数。
            "DAS": self._ordinary_stage_statistics(self.das),  # DAS以8 KiB原子计数。
        }

    def _ordinary_stage_statistics(self, stage):
        """返回一个普通流水线阶段的精简统计。"""
        return {  # 只保留判断吞吐与反压所需的信息。
            "started": stage.started_count,  # 累计启动命令或原子数量。
            "completed": stage.completed_count,  # 累计完成处理数量。
            "max_waiting": stage.max_waiting_depth,  # 最大等待FIFO深度。
            "max_occupied": stage.max_occupied_slots,  # 最大模块总占用量。
        }
