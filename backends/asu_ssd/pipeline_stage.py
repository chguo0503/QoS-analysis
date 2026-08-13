#!/usr/bin/env python3
"""ASU后端使用的简单流水线阶段和FCP拆分器。"""

from collections import deque  # deque用于实现各阶段的先进先出等待队列。


class PipelineStage:
    """表示处理固定粒度命令的普通流水线阶段。

    waiting保存尚未启动的命令。
    processing_count记录已经启动但尚未完成的命令数。
    completed保存处理完成但因下游反压尚未输出的命令。
    waiting_capacity只限制独立输入FIFO，例如BCP的1024项队列。
    max_inflight限制处理和阻塞完成命令，例如BCP的20个在途位置。
    total_slot_mode为True时，等待、处理和阻塞完成共同占用同一组槽位。
    """

    def __init__(
        self,
        name,
        start_interval,
        latency,
        max_inflight,
        waiting_capacity=None,
        total_slot_mode=False,
    ):
        """保存阶段配置并创建空的运行状态。"""
        self.name = name  # 保存模块名称，例如BCP或NAND。
        self.start_interval = start_interval  # 保存相邻命令的最小启动间隔。
        self.latency = latency  # 保存一个命令从启动到完成的固定延迟。
        self.max_inflight = max_inflight  # 保存处理和阻塞完成命令的容量。
        self.waiting_capacity = waiting_capacity  # None表示没有独立等待FIFO。
        self.total_slot_mode = total_slot_mode  # True表示所有状态共享同一组槽位。
        self.waiting = deque()  # 保存已经进入模块但尚未启动的命令。
        self.completed = deque()  # 保存完成后等待输出的命令。
        self.processing_count = 0  # 初始时没有正在处理的命令。
        self.next_start_time = 0  # 第一个命令允许在仿真开始时启动。
        self.start_event_pending = False  # False表示当前没有重复的启动事件。
        self.started_count = 0  # 记录累计启动命令数。
        self.completed_count = 0  # 记录累计完成处理的命令数。
        self.max_waiting_depth = 0  # 记录运行期间最大的等待FIFO深度。
        self.max_occupied_slots = 0  # 记录运行期间最大的总占用量。

    def inflight_count(self):
        """返回正在处理和完成后阻塞的命令数量。"""
        return self.processing_count + len(self.completed)  # 两种状态都会占用在途位置。

    def occupied_slots(self):
        """返回当前模块内保存的全部命令数量。"""
        return len(self.waiting) + self.inflight_count()  # 等待、处理和阻塞完成全部计入。

    def _update_maximums(self):
        """更新队列深度和槽位占用的历史峰值。"""
        self.max_waiting_depth = max(self.max_waiting_depth, len(self.waiting))  # 更新等待峰值。
        self.max_occupied_slots = max(self.max_occupied_slots, self.occupied_slots())  # 更新占用峰值。

    def can_accept(self, command_count=1):
        """返回当前阶段是否能接收指定数量的新命令。"""
        if self.total_slot_mode:  # NFI、NAND和BDP使用共享总槽位。
            return self.occupied_slots() + command_count <= self.max_inflight  # 所有状态合计不能超限。
        return len(self.waiting) + command_count <= self.waiting_capacity  # BCP只检查独立输入FIFO。

    def accept(self, command):
        """把一个新命令加入阶段等待FIFO。"""
        self.waiting.append(command)  # 新命令排在现有命令后面。
        self._update_maximums()  # 接收后更新压力峰值。

    def can_start(self):
        """返回当前是否有等待命令和可用处理位置。"""
        has_waiting_command = bool(self.waiting)  # 等待FIFO非空时才有命令可启动。
        has_processing_slot = self.inflight_count() < self.max_inflight  # 阻塞完成也占处理位置。
        return has_waiting_command and has_processing_slot  # 两个条件同时满足才能启动。

    def start_next(self):
        """从等待FIFO取出一个命令并标记为正在处理。"""
        command = self.waiting.popleft()  # 启动最早进入模块的命令。
        self.processing_count += 1  # 正在处理的命令数量加1。
        self.started_count += 1  # 累计启动数量加1。
        return command  # 返回命令供后端安排完成事件。

    def finish(self, command):
        """将一个处理完成的命令放入阻塞输出区。"""
        self.processing_count -= 1  # 命令不再处于处理中。
        self.completed.append(command)  # 完成命令在下游可接收前继续占用槽位。
        self.completed_count += 1  # 累计完成处理数量加1。
        self._update_maximums()  # 更新当前占用峰值。

    def finish_and_release(self):
        """完成一个不需要等待下游的命令并立即释放槽位。"""
        self.processing_count -= 1  # DAS输出默认可被网络立即接收。
        self.completed_count += 1  # 累计完成处理数量加1。

    def pop_completed(self):
        """取出最早完成且下游已经能够接收的命令。"""
        return self.completed.popleft()  # 输出后当前阶段对应槽位被释放。


class FCPStage:
    """保存30个原子槽，并把一个活动请求逐步拆成4 KiB分片。

    active_request表示拆分器当前持有的可变大小SSD请求描述符。
    next_chunk_index表示该请求下一个尚未进入FCP的4 KiB编号。
    half_atom保存已经进入一个4 KiB、等待相邻分片的半个8 KiB原子。
    waiting、processing_count和completed共同占用30个原子槽。
    """

    def __init__(self, atom_slots, start_interval, latency):
        """保存FCP容量、启动间隔和固定延迟。"""
        self.name = "FCP"  # 保存模块名称。
        self.atom_slots = atom_slots  # 保存等待、处理和阻塞完成共享的原子槽数量。
        self.active_chunk_count = 0  # 每次接收描述符时保存它的4 KiB命令数量。
        self.start_interval = start_interval  # 保存相邻8 KiB原子的最小启动间隔。
        self.latency = latency  # 保存一个8 KiB原子的固定处理延迟。
        self.active_request = None  # 初始时拆分器没有持有请求。
        self.next_chunk_index = 0  # 初始分片编号从0开始。
        self.half_atom = None  # 初始时没有只填入一个4 KiB的半原子。
        self.waiting = deque()  # 保存已经组成8 KiB但尚未启动的原子。
        self.completed = deque()  # 保存完成后因BCP反压尚未输出的原子。
        self.processing_count = 0  # 初始时没有正在处理的原子。
        self.next_start_time = 0  # 第一个原子允许在仿真开始时启动。
        self.start_event_pending = False  # False表示当前没有重复启动事件。
        self.started_count = 0  # 记录累计启动的8 KiB原子数量。
        self.completed_count = 0  # 记录累计完成处理的8 KiB原子数量。
        self.max_occupied_slots = 0  # 记录FCP原子槽的最大占用量。

    def occupied_slots(self):
        """返回等待、处理、阻塞完成和半原子占用的槽位总数。"""
        half_slot = 1 if self.half_atom is not None else 0  # 半个原子仍然占用一个完整原子槽。
        return len(self.waiting) + self.processing_count + len(self.completed) + half_slot  # 汇总所有状态。

    def _update_maximum(self):
        """更新FCP原子槽历史最大占用量。"""
        self.max_occupied_slots = max(self.max_occupied_slots, self.occupied_slots())  # 保存较大值。

    def can_accept_request(self):
        """返回拆分器是否空闲且FCP至少还有一个原子槽。"""
        splitter_is_idle = self.active_request is None  # 一个拆分器一次只持有一个请求。
        has_atom_space = self.occupied_slots() < self.atom_slots  # 至少有一个槽才能放入首个4 KiB。
        return splitter_is_idle and has_atom_space  # 两个条件同时满足才能接收新描述符。

    def submit_request(self, request):
        """让拆分器接收一个请求，并立即填入当前能够容纳的4 KiB分片。"""
        self.active_request = request  # 拆分器开始持有这个完整请求描述符。
        self.active_chunk_count = request["backend_chunk_count"]  # 读取这个描述符自己的4 KiB命令数。
        self.next_chunk_index = 0  # 新请求从chunk 0开始切分。
        self.fill_available_chunks()  # 有多少空间就先填入多少4 KiB分片。

    def _make_chunk(self, chunk_index):
        """根据活动请求和编号创建一个4 KiB分片描述符。"""
        return {  # 使用普通字典保存后端需要的分片信息。
            "request_id": self.active_request["request_id"],  # 保留原始SSD请求编号。
            "queue_id": self.active_request["queue_id"],  # 保留请求来自哪个QoS队列。
            "chunk_index": chunk_index,  # 记录当前4 KiB在原请求中的位置。
            "size_bytes": self.active_request["chunk_size_bytes"],  # 每个分片固定为4 KiB。
        }

    def fill_available_chunks(self):
        """将活动请求的4 KiB分片逐个填入当前可用的FCP原子槽。"""
        while self.active_request is not None:  # 只要请求还有分片就继续尝试填充。
            if self.half_atom is None:  # 新的偶数编号分片需要占用一个空原子槽。
                if self.occupied_slots() >= self.atom_slots:  # 所有30个原子槽都已占用。
                    break  # 保留活动请求，等待后续输出释放槽位。
                first_chunk = self._make_chunk(self.next_chunk_index)  # 创建相邻分片中的第一个。
                self.half_atom = first_chunk  # 首个4 KiB暂存在半原子中。
                self.next_chunk_index += 1  # 下一个待填分片编号加1。
                self._update_maximum()  # 半原子占槽后更新峰值。
            else:  # 半原子已经有第一个4 KiB，可以填入相邻的第二个。
                second_chunk = self._make_chunk(self.next_chunk_index)  # 创建相邻的第二个4 KiB。
                atom = {  # 两个相邻分片组成一个完整8 KiB原子。
                    "request_id": self.active_request["request_id"],  # 保存原始请求编号。
                    "queue_id": self.active_request["queue_id"],  # 保存原始QoS队列编号。
                    "atom_index": self.half_atom["chunk_index"] // 2,  # 计算原请求中的原子编号。
                    "chunks": [self.half_atom, second_chunk],  # 保存原始相邻的两个4 KiB分片。
                }
                self.half_atom = None  # 两个分片已经配对，清空半原子状态。
                self.waiting.append(atom)  # 完整8 KiB原子进入FCP启动等待队列。
                self.next_chunk_index += 1  # 下一个待填分片编号加1。
                self._update_maximum()  # 完整原子仍然占用同一个槽。

            if self.next_chunk_index == self.active_chunk_count:  # 当前请求的4 KiB命令已经全部进入FCP。
                self.active_request = None  # 拆分器释放完整请求描述符。
                self.active_chunk_count = 0  # 清空上一个请求的命令数。
                self.next_chunk_index = 0  # 为下一个请求恢复初始编号。
                break  # 当前请求填充完成，本次函数结束。

    def can_start(self):
        """返回FCP是否有已经配对完成、可以启动的8 KiB原子。"""
        return bool(self.waiting)  # 总槽位已经限制容量，因此只需检查等待队列。

    def start_next(self):
        """启动最早组成的8 KiB原子。"""
        atom = self.waiting.popleft()  # 按FIFO顺序取出一个完整原子。
        self.processing_count += 1  # 正在处理的原子数量加1。
        self.started_count += 1  # 累计启动数量加1。
        return atom  # 返回原子供后端安排5 us完成事件。

    def finish(self, atom):
        """把一个完成处理的原子放入BCP输出等待区。"""
        self.processing_count -= 1  # 原子不再处于处理中。
        self.completed.append(atom)  # BCP能同时接收两个命令前继续占用FCP槽。
        self.completed_count += 1  # 累计完成处理数量加1。

    def pop_completed(self):
        """向BCP输出一个原子并释放一个FCP原子槽。"""
        atom = self.completed.popleft()  # 取出最早完成的8 KiB原子。
        self.fill_available_chunks()  # 槽位释放后立刻继续填入活动请求的4 KiB分片。
        return atom  # 返回原子中的两个4 KiB命令。


class DASPairer:
    """按request_id和相邻chunk编号重新组合8 KiB DMA原子。"""

    def __init__(self, stage):
        """保存DAS流水线阶段并创建空的待配对字典。"""
        self.stage = stage  # stage负责18个原子槽、启动速率和3 us延迟。
        self.pending_chunks = {}  # 每个相邻分片对最多暂存一个4 KiB命令。

    def _pair_key(self, chunk):
        """返回相邻两个4 KiB分片共同使用的配对键。"""
        atom_index = chunk["chunk_index"] // 2  # chunk 0和1映射到atom 0，以此类推。
        return chunk["request_id"], atom_index  # 不同请求的相同编号不会混合。

    def can_accept(self, chunk):
        """返回DAS是否能接收当前4 KiB命令。"""
        pair_key = self._pair_key(chunk)  # 计算当前分片的相邻配对键。
        if pair_key not in self.pending_chunks:  # 第一个分片只进入pending缓冲。
            return True  # pending缓冲不作为性能队列限速。
        return self.stage.can_accept()  # 第二个分片需要一个完整8 KiB原子槽。

    def accept(self, chunk):
        """暂存第一个分片，或用第二个分片组成一个相邻8 KiB原子。"""
        pair_key = self._pair_key(chunk)  # 计算当前分片对应的原子编号。
        if pair_key not in self.pending_chunks:  # 当前是该相邻分片对中的第一个。
            self.pending_chunks[pair_key] = chunk  # 等待另一个相邻4 KiB到达。
            return False  # False表示本次还没有形成8 KiB原子。
        first_chunk = self.pending_chunks.pop(pair_key)  # 取出此前等待的相邻分片。
        chunks = sorted([first_chunk, chunk], key=lambda item: item["chunk_index"])  # 恢复地址顺序。
        atom = {  # 构造DAS需要处理的8 KiB DMA原子。
            "request_id": chunk["request_id"],  # 保存原始SSD请求编号。
            "queue_id": chunk["queue_id"],  # 保存原始QoS队列编号。
            "atom_index": pair_key[1],  # 保存原请求中的8 KiB原子编号。
            "chunks": chunks,  # 保存相邻的两个4 KiB分片。
        }
        self.stage.accept(atom)  # 完整原子进入DAS的18个槽位。
        return True  # True表示新产生了一个可启动的DAS原子。
