#!/usr/bin/env python3
"""向联合事件引擎提供非阻塞SSD输入、单步推进和完成统计接口。"""

from pathlib import Path

from simulation_common.config_utils import load_yaml

from .backend import ASUBackend
from .batched_backend import BatchedExactASUBackend
from .time_utils import time_to_us, us_to_time


CONFIG_DIR = Path(__file__).parent / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "asu_backend_config.yaml"


def load_ssd_config(config_file=DEFAULT_CONFIG_FILE):
    """功能：读取SSD内部流水线参数。

    目的：把ASU流水线的容量、速率和延迟配置与Python实现分离。

    输入：
        config_file: ASU后端YAML文件路径。

    输出：
        dict: YAML中 ``backend`` 节点对应的配置字典。
    """
    return load_yaml(config_file)["backend"]


class SSDSimulator:
    """封装一个独立SSD的非阻塞入口、内部事件日历和完成回调。"""

    def __init__(
        self,
        backend_config,
        completion_sink=None,
        storage_target_id=None,
    ):
        """功能：创建一块SSD的完整ASU流水线运行状态。

        目的：让每个 ``QoS+SSD`` StoragePath拥有完全独立的容量、时钟和统计，
        同时允许完成通知返回全局可识别的SSD编号。

        输入：
            backend_config: ASUBackend需要的流水线配置字典。
            completion_sink: 完整请求完成时调用的可选回调。
            storage_target_id: 当前SSD在全局拓扑中的唯一ID。

        输出：
            None: 初始化空SSD、非阻塞输入统计和完成状态。
        """
        # 执行模式在SSD封装层切换，保证QoS和上层仿真器
        # 看到的始终是同一个非阻塞后端接口。
        self.execution_mode = backend_config.get("execution_mode", "detailed")
        backend_class = {
            "detailed": ASUBackend,
            "batched_exact": BatchedExactASUBackend,
        }[self.execution_mode]
        self.backend = backend_class(backend_config)
        self.completion_sink = completion_sink
        self.storage_target_id = storage_target_id
        self.current_time = 0
        self.end_result = None

        self.input_request_count = 0
        self.blocked_attempt_count = 0
        self.blocked_request_ids = set()
        self.first_attempt_time_us = {}
        self.total_input_wait_us = 0.0
        self.max_input_wait_us = 0.0

    def _publish_completions(self):
        """功能：发布当前SSD事件新产生的全部完整请求完成通知。

        目的：每个原始IO只在最后一个DAS原子完成时通知一次所属GPU，并在
        多SSD场景中显式携带 ``storage_target_id``。

        输入：
            无；读取ASUBackend的待发布完成队列。

        输出：
            None: 清空待发布记录并调用上层回调。
        """
        for record in self.backend.drain_completion_records():
            if self.completion_sink is None:
                continue
            self.completion_sink({
                "request_id": record["request_id"],
                "storage_target_id": self.storage_target_id,
                "completion_time_us": record["backend_completion_time_us"],
            })

    def _process_next_event(self):
        """功能：处理当前SSD内部最早时间戳的全部事件。

        目的：为独立SSD排空接口和全局事件协调器复用同一条精确推进路径。

        输入：
            无；事件时间来自ASUBackend最小堆。

        输出：
            int | None: 已处理的内部整数时刻；无事件时返回None。
        """
        event_time = self.backend.next_event_time()
        if event_time is None:
            return None
        self.backend.process_events_at(event_time)
        self.current_time = event_time
        self._publish_completions()
        return event_time

    def _synchronize_without_processing(self, requested_time):
        """功能：把空闲接口时钟同步到请求时刻，但绝不处理未来事件。

        目的：强制所有SSD内部事件由全局协调器按时间顺序推进；如果调用者试图
        跨过尚未处理的更早事件，立即报错而不是隐藏因果顺序问题。

        输入：
            requested_time: 请求尝试进入SSD的内部整数仿真时刻。

        输出：
            None: 合法时只更新SSD接口当前时刻。
        """
        if requested_time < self.current_time:
            raise RuntimeError(
                "SSD input cannot move backwards from "
                f"{time_to_us(self.current_time)} us to "
                f"{time_to_us(requested_time)} us"
            )

        next_event_time = self.backend.next_event_time()
        # 等于requested_time的事件可能是本轮QoS刚提交请求后新产生的start，
        # 它可以在QoS下发阶段结束后处理；严格小于才表示全局事件被错误跳过。
        if next_event_time is not None and next_event_time < requested_time:
            raise RuntimeError(
                "SSD has an unprocessed event at "
                f"{time_to_us(next_event_time)} us before input time "
                f"{time_to_us(requested_time)} us"
            )
        self.current_time = requested_time
        # 精确批量后端通过解析时刻判断FCP是否可接收，
        # 因此它必须与封装层的全局时钟保持一致。
        self.backend.synchronize_time(requested_time)

    def next_event_time(self):
        """功能：返回SSD内部最近事件的精确整数时刻。

        目的：让全局事件日历比较多块SSD时不经过浮点微秒转换。

        输入：
            无。

        输出：
            int | None: 最近内部事件时刻；SSD空闲时返回None。
        """
        return self.backend.next_event_time()

    def next_event_time_us(self):
        """功能：返回SSD最近事件的微秒表示。

        目的：保留便于诊断和旧调用方查看的公开接口，事件排序仍使用整数时刻。

        输入：
            无。

        输出：
            float | None: 最近事件的微秒时刻；SSD空闲时返回None。
        """
        event_time = self.next_event_time()
        if event_time is None:
            return None
        return time_to_us(event_time)

    def process_events_at(self, event_time):
        """功能：处理指定整数时刻的全部SSD内部事件。

        目的：由全局事件协调器精确推进某一块SSD，不允许组件自行跳到更晚时刻。

        输入：
            event_time: 必须等于当前SSD最早事件的内部整数时刻。

        输出：
            float: 处理完成后的微秒时刻。
        """
        next_event_time = self.backend.next_event_time()
        if next_event_time != event_time:
            raise RuntimeError(
                f"requested SSD event {event_time}, but next event is "
                f"{next_event_time}"
            )
        self.backend.process_events_at(event_time)
        self.current_time = event_time
        self._publish_completions()
        return time_to_us(self.current_time)

    def run_next_event(self):
        """功能：处理SSD的下一个内部事件时刻。

        目的：支持SSD独立测试和诊断；多设备联合仿真优先调用 ``process_events_at``。

        输入：
            无。

        输出：
            float | None: 处理后的微秒时刻；无事件时返回None。
        """
        event_time = self._process_next_event()
        if event_time is None:
            return None
        return time_to_us(event_time)

    def run_until_idle(self):
        """功能：排空当前SSD已经接收的全部内部事件。

        目的：供SSD独立测试或联合仿真结束阶段使用，不作为运行中反压等待手段。

        输入：
            无。

        输出：
            float: SSD排空后的微秒时刻。
        """
        while self.backend.next_event_time() is not None:
            self._process_next_event()
        return time_to_us(self.current_time)

    def _finalize_result(self):
        """功能：在SSD排空后生成一次最终统计结果。

        目的：集中整理NAND服务事件、完成记录、流水线占用和入口反压信息。

        输入：
            无；要求SSD已不再接收新请求。

        输出：
            dict: 当前SSD的完整最终结果。
        """
        first_submit_time_us = None
        last_completion_time_us = None
        if self.backend.first_submit_time is not None:
            first_submit_time_us = time_to_us(self.backend.first_submit_time)
            last_completion_time_us = time_to_us(
                self.backend.last_completion_time
            )
        self.end_result = {
            "stopped": True,
            "storage_target_id": self.storage_target_id,
            "execution_mode": self.execution_mode,
            "completion_time_us": time_to_us(self.current_time),
            "first_submit_time_us": first_submit_time_us,
            "last_completion_time_us": last_completion_time_us,
            "backend_chunk_size_bytes": self.backend.chunk_size_bytes,
            "nand_read_bandwidth_bytes_per_second": (
                self.backend.nand_read_bandwidth_bytes_per_second
            ),
            "nand_service_events": list(self.backend.nand_service_events),
            "completed_bytes": self.backend.completed_bytes(),
            "completed_requests": list(self.backend.completed_requests),
            "stage_statistics": self.backend.stage_statistics(),
            "input_wait_statistics": self._input_wait_statistics(),
        }
        return self.end_result

    def end(self):
        """功能：排空SSD并返回幂等的最终统计。

        目的：确保顶层重复读取结果时不会再次推进事件或重复发布完成通知。

        输入：
            无。

        输出：
            dict: 第一次调用生成、后续调用复用的SSD最终结果。
        """
        if self.end_result is not None:
            return self.end_result
        self.run_until_idle()
        return self._finalize_result()

    def _record_accepted_input(self, request_id, accepted_time_us):
        """功能：记录一个最终被SSD入口接收的普通IO。

        目的：计算从首次非阻塞尝试到真实接收之间的入口反压等待，并保证
        blocked_request_count按请求去重而不是按重试次数计算。

        输入：
            request_id: 当前完整SSD请求的唯一编号。
            accepted_time_us: 请求真正进入FCP的微秒时刻。

        输出：
            None: 原地更新请求数和等待时间统计。
        """
        first_attempt_time_us = self.first_attempt_time_us.pop(
            request_id,
            accepted_time_us,
        )
        wait_us = accepted_time_us - first_attempt_time_us
        self.input_request_count += 1
        self.total_input_wait_us += wait_us
        self.max_input_wait_us = max(self.max_input_wait_us, wait_us)

    def _input_wait_statistics(self):
        """功能：返回SSD入口非阻塞尝试与等待统计。

        目的：区分请求在QoS中等待SSD入口和请求进入SSD后的流水线服务时间。

        输入：
            无。

        输出：
            dict: 接收请求数、阻塞请求数、重试次数和等待时长。
        """
        blocked_request_count = len(self.blocked_request_ids)
        blocked_ratio = 0.0
        if self.input_request_count:
            blocked_ratio = blocked_request_count / self.input_request_count
        return {
            "request_count": self.input_request_count,
            "blocked_request_count": blocked_request_count,
            "blocked_request_ratio": blocked_ratio,
            "blocked_attempt_count": self.blocked_attempt_count,
            "total_wait_us": self.total_input_wait_us,
            "max_wait_us": self.max_input_wait_us,
        }

    def can_accept_at_us(self, requested_time_us):
        """功能：检查SSD在指定微秒时刻能否接收一个完整描述符。

        目的：让QoS在出队和扣令牌之前先检查后端反压，避免请求被提前移除。

        输入：
            requested_time_us: 全局事件引擎当前的仿真微秒时刻。

        输出：
            bool: FCP拆分器当前可接收时返回True，否则返回False。
        """
        requested_time = us_to_time(requested_time_us)
        self._synchronize_without_processing(requested_time)
        return self.backend.can_accept_request()

    def try_input_at_us(self, io_descriptor, requested_time_us=None):
        """功能：在当前仿真时刻非阻塞尝试接收一个普通IO。

        目的：SSD入口满时立即返回失败，让全局事件循环继续推进其他SSD，而不是
        在本函数内部处理未来事件直到空间出现。

        输入：
            io_descriptor: QoS输出的完整SSD请求描述符。
            requested_time_us: 可选显式尝试时刻；默认读取 ``dispatch_time_us``。

        输出：
            dict: 包含 ``accepted`` 和当前尝试/接收时刻；失败时不会改变流水线。
        """
        if self.end_result is not None:
            raise RuntimeError("cannot submit input after SSD.end()")

        if requested_time_us is None:
            requested_time_us = io_descriptor["dispatch_time_us"]
        requested_time = us_to_time(requested_time_us)
        self._synchronize_without_processing(requested_time)

        request_id = io_descriptor["request_id"]
        self.first_attempt_time_us.setdefault(request_id, requested_time_us)
        if not self.backend.can_accept_request():
            self.blocked_attempt_count += 1
            self.blocked_request_ids.add(request_id)
            return {
                "accepted": False,
                "accepted_time_us": None,
                "attempt_time_us": requested_time_us,
            }

        self.backend.submit_request(io_descriptor, requested_time)
        accepted_time_us = time_to_us(requested_time)
        self._record_accepted_input(request_id, accepted_time_us)
        return {
            "accepted": True,
            "accepted_time_us": accepted_time_us,
            "attempt_time_us": requested_time_us,
        }

    def input(self, io_descriptor):
        """功能：兼容旧调用形式并执行一次非阻塞SSD输入尝试。

        目的：保留公开方法名称，同时明确新语义不再自动等待FCP释放空间。

        输入：
            io_descriptor: 必须包含 ``dispatch_time_us`` 的完整SSD请求。

        输出：
            dict: 与 ``try_input_at_us`` 相同的接受或拒绝结果。
        """
        return self.try_input_at_us(io_descriptor)
