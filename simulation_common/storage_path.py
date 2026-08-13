"""把一套独立QoS和一块SSD连接成可由全局事件日历推进的StoragePath。"""

from backends.asu_ssd.time_utils import time_to_us, us_to_time


SSD_EVENT_PRIORITY = 0
QOS_EVENT_PRIORITY = 20
SSD_EVENT_AFTER_QOS_PRIORITY = 30


class StoragePath:
    """管理一个 ``storage_target_id`` 对应的QoS、SSD和事件唤醒。"""

    def __init__(self, storage_target_id, qos, ssd, event_loop):
        """功能：创建一条独立的QoS到SSD数据路径。

        目的：封装本路径的事件去重、非阻塞反压和局部结果，使增加SSD数量只需
        增加StoragePath实例而无需复制顶层事件循环逻辑。

        输入：
            storage_target_id: 当前路径的唯一SSD编号。
            qos: 当前SSD前独立的QoS模拟器。
            ssd: 当前路径独立的SSD模拟器。
            event_loop: 所有路径共享的全局事件日历。

        输出：
            None: 连接QoS后端并初始化事件去重集合。
        """
        self.storage_target_id = storage_target_id
        self.qos = qos
        self.ssd = ssd
        self.event_loop = event_loop
        self.qos.set_backend(ssd)
        self.scheduled_qos_times = set()
        self.scheduled_ssd_times = set()

    def input(self, request):
        """功能：把一个DPU请求登记到当前SSD对应的QoS。

        目的：使用请求自带到达时间安排一次去重QoS唤醒，使同一时刻的数千个
        Block先全部进入到达堆，再由一次QoS事件统一接收和仲裁。

        输入：
            request: DPU展平且 ``storage_target_id`` 指向当前路径的QoS请求。

        输出：
            dict: QoS登记并返回的原请求对象。
        """
        registered_request = self.qos.input(request)
        self._schedule_qos_at(us_to_time(request["arrival_time_us"]))
        return registered_request

    def _schedule_qos_at(self, event_time):
        """功能：为当前路径安排一次去重的QoS事件。

        目的：避免同一层每个Block都创建重复QoS回调，同时允许SSD释放入口后
        在非令牌时刻立即重新触发调度。

        输入：
            event_time: QoS应被唤醒的全局内部整数时刻。

        输出：
            None: 事件不存在时加入全局日历，已存在时保持原事件。
        """
        event_time = max(event_time, self.event_loop.current_time)
        if event_time in self.scheduled_qos_times:
            return
        self.scheduled_qos_times.add(event_time)
        self.event_loop.schedule_at(
            event_time=event_time,
            priority=QOS_EVENT_PRIORITY,
            event_name=f"qos:{self.storage_target_id}",
            callback=self._process_qos_event,
        )

    def _schedule_next_qos_event(self):
        """功能：根据QoS内部状态安排其下一次自然事件。

        目的：让未来IO到达和令牌补充进入全局日历，而SSD入口释放唤醒由
        ``_process_ssd_event`` 额外负责。

        输入：
            无。

        输出：
            None: 存在下一事件时完成去重登记。
        """
        next_event_time_us = self.qos.next_event_time_us()
        if next_event_time_us is None:
            return
        self._schedule_qos_at(us_to_time(next_event_time_us))

    def _schedule_next_ssd_event(self, priority=SSD_EVENT_PRIORITY):
        """功能：把SSD当前最早内部事件登记到全局日历。

        目的：SSD仍可维护高效的局部流水线事件堆，但所有设备的堆顶事件必须
        通过同一个全局日历比较，才能保持跨SSD因果顺序。

        输入：
            priority: 同时间戳中本次SSD事件使用的全局阶段优先级。

        输出：
            None: 无SSD事件或相同时刻已登记时不执行操作。
        """
        event_time = self.ssd.next_event_time()
        if event_time is None or event_time in self.scheduled_ssd_times:
            return
        self.scheduled_ssd_times.add(event_time)
        self.event_loop.schedule_at(
            event_time=event_time,
            priority=priority,
            event_name=f"ssd:{self.storage_target_id}",
            callback=self._process_ssd_event,
        )

    def _process_qos_event(self, event_time):
        """功能：在全局指定时刻处理当前路径的一次QoS事件。

        目的：完成令牌补充、全部到达接收和非阻塞下发，再把新产生的SSD启动
        事件以及QoS下一自然事件重新登记到全局日历。

        输入：
            event_time: 当前回调对应的全局内部整数时刻。

        输出：
            None: 原地推进当前路径并安排后续事件。
        """
        self.scheduled_qos_times.discard(event_time)
        event_time_us = time_to_us(event_time)

        # 旧的重复唤醒可能在QoS已被同时间其他因果事件推进后到达；此时安全跳过，
        # 不允许组件时钟回退。
        if event_time_us < self.qos.current_time_us:
            return
        self.qos.process_at(event_time_us)
        self._schedule_next_qos_event()

        # QoS本轮成功提交请求后，ASUBackend可能生成当前同一时刻的FCP start。
        # 使用较晚阶段优先级，明确表示该start由本轮QoS下发因果产生。
        self._schedule_next_ssd_event(
            priority=SSD_EVENT_AFTER_QOS_PRIORITY,
        )

    def _process_ssd_event(self, event_time):
        """功能：处理当前路径SSD在指定时刻的全部内部流水线事件。

        目的：发布完成通知、安排下一SSD事件，并在FCP入口重新可用时于同一
        仿真时刻唤醒仍有积压的QoS。

        输入：
            event_time: 当前回调对应的全局内部整数时刻。

        输出：
            None: 原地推进SSD并安排后续事件。
        """
        self.scheduled_ssd_times.discard(event_time)
        actual_next_time = self.ssd.next_event_time()
        if actual_next_time != event_time:
            # 新输入可能让SSD产生一个更早事件，使旧日历项失效。失效事件不处理，
            # 只确保当前真正堆顶重新登记即可。
            self._schedule_next_ssd_event()
            return

        event_time_us = self.ssd.process_events_at(event_time)
        self._schedule_next_ssd_event()

        if (
            self.qos.has_queued_requests()
            and self.ssd.can_accept_at_us(event_time_us)
        ):
            self._schedule_qos_at(event_time)

    def start(self):
        """功能：把当前路径已经存在的首个QoS和SSD事件加入全局日历。

        目的：支持先装配或预登记请求、后统一启动事件循环的构建方式。

        输入：
            无。

        输出：
            None: 安排当前可见的下一事件。
        """
        self._schedule_next_qos_event()
        self._schedule_next_ssd_event()

    def end(self):
        """功能：固化当前StoragePath的QoS和SSD结果。

        目的：以 ``storage_target_id`` 为命名空间返回独立设备统计，避免不同
        SSD中相同 ``queue_id`` 被误认为同一条Queue。

        输入：
            无。

        输出：
            dict: 包含当前路径ID、QoS结果和SSD结果。
        """
        return {
            "storage_target_id": self.storage_target_id,
            "qos": self.qos.end(),
            "ssd": self.ssd.end(),
        }
