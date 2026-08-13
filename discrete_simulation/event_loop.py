"""提供多GPU、多QoS和多SSD共同使用的稳定全局事件日历。"""

import heapq


class EventLoop:
    """按整数时间、事件优先级和稳定序号推进整个联合仿真。"""

    def __init__(self, start_time=0):
        """功能：创建一个空的全局离散事件日历。

        目的：让所有GPU和StoragePath共享唯一单调时钟，避免组件各自阻塞推进
        导致某块SSD越过另一块SSD的更早事件。

        输入：
            start_time: 全局起始内部整数时刻。

        输出：
            None: 初始化空最小堆、稳定序号和事件统计。
        """
        self.current_time = start_time
        self.events = []
        self.event_sequence = 0
        self.processed_event_count = 0

    def schedule_at(self, event_time, priority, event_name, callback):
        """功能：把一个回调安排到指定全局整数时刻。

        目的：使用 ``(时间, 优先级, 序号)`` 保证跨GPU、QoS和SSD事件具有
        确定性顺序；同一优先级下保持实际登记顺序。

        输入：
            event_time: 不早于当前全局时间的内部整数时刻。
            priority: 同一时间戳使用的整数优先级，数值越小越先处理。
            event_name: 用于异常诊断的人类可读事件名称。
            callback: 接收 ``event_time`` 参数的无返回值函数。

        输出：
            int: 当前事件获得的全局稳定序号。
        """
        if event_time < self.current_time:
            raise RuntimeError(
                f"cannot schedule {event_name!r} at {event_time}; "
                f"global time is already {self.current_time}"
            )
        self.event_sequence += 1
        heapq.heappush(
            self.events,
            (
                event_time,
                priority,
                self.event_sequence,
                event_name,
                callback,
            ),
        )
        return self.event_sequence

    def next_event_time(self):
        """功能：返回全局事件日历中最早的整数时刻。

        目的：支持测试和死锁诊断，不暴露最小堆内部元组结构。

        输入：
            无。

        输出：
            int | None: 最近事件时刻；日历为空时返回None。
        """
        if not self.events:
            return None
        return self.events[0][0]

    def run_next_event(self):
        """功能：处理全局日历中的一个最早事件。

        目的：只推进到真实事件时间，不使用固定时间步长；回调可以安全地在
        当前时刻继续登记由本事件因果产生的新事件。

        输入：
            无。

        输出：
            str | None: 已处理事件名称；日历为空时返回None。
        """
        if not self.events:
            return None
        event_time, _, _, event_name, callback = heapq.heappop(self.events)
        self.current_time = event_time
        callback(event_time)
        self.processed_event_count += 1
        return event_name

    def run_until(self, stop_condition, max_events=None):
        """功能：持续处理事件直到调用方给出的完成条件成立。

        目的：统一检测多GPU全部完成、事件耗尽死锁和可选测试事件上限。

        输入：
            stop_condition: 无参数函数，返回True时终止仿真。
            max_events: 可选最大处理事件数，用于测试发现无限事件循环。

        输出：
            int: 本次调用累计处理的事件数量。
        """
        initial_count = self.processed_event_count
        while not stop_condition():
            if not self.events:
                raise RuntimeError(
                    "global event calendar became empty before simulation completed"
                )
            if (
                max_events is not None
                and self.processed_event_count - initial_count >= max_events
            ):
                raise RuntimeError(
                    f"simulation exceeded max_events={max_events}"
                )
            self.run_next_event()
        return self.processed_event_count - initial_count
