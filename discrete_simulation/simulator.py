#!/usr/bin/env python3
"""实现Queue令牌桶、WRR和非阻塞SSD入口之间的QoS离散事件组件。"""

import heapq


class DiscreteEventSimulator:
    """维护一套独立QoS的到达事件、令牌状态、调度器和下游反压。"""

    def __init__(self, token_stage, scheduler, simulation_config):
        """功能：创建一套QoS组件的全部运行状态。

        目的：让同一个类既可独立仿真QoS，也可作为一条 ``QoS+SSD`` 路径
        被全局多设备事件协调器按时间戳单步推进。

        输入：
            token_stage: 保存Queue FIFO及CIR/PIR令牌的阶段对象。
            scheduler: 根据资格函数选择Queue的分层WRR调度器。
            simulation_config: 起始时刻和同时间事件顺序配置。

        输出：
            None: 初始化空到达堆、仿真时钟和统计列表。
        """
        self.token_stage = token_stage
        self.scheduler = scheduler
        self.simulation_config = simulation_config
        self.start_time_us = simulation_config["start_time_us"]
        self.current_time_us = self.start_time_us
        self.completed_refill_periods = 0
        self.backend = None

        # 到达堆元组按(时间、稳定序号、请求)排序。稳定序号避免两个请求同一
        # 时间到达时让Python继续比较普通字典，并保留DPU提交先后顺序。
        self.pending_arrivals = []
        self.arrival_sequence = 0
        # DPU读取的是“已经登记到本QoS、但尚未成功下发到SSD”的逻辑
        # Queue occupancy。它同时覆盖到达堆与FIFO，避免仿真事件分阶段处理
        # 让DPU在同一时刻暂时看不到刚提交的IO。
        self.registered_queue_io_counts = {
            queue_id: 0 for queue_id in self.token_stage.queue_order
        }

        # 同一时刻同一Queue只保留DPU最后一次整数速率设置。
        self.pending_rate_update_keys = []
        self.pending_rate_updates = {}

        # Group WRR权重与Queue速率共用rate_update事件阶段，但独立
        # 保存整张权重：Group只控制调度机会，不是第二级令牌桶。
        self.pending_group_weight_update_keys = []
        self.pending_group_weight_updates = {}

        # observer只在Queue depth变化后唤醒DPU读取快照，
        # 事件本身不携带请求、Demand或SSD完成信息。
        self.queue_state_observer = None
        self.total_request_count = 0
        self.dispatched_requests = []
        self.end_result = None

    def set_backend(self, backend):
        """功能：为当前QoS连接唯一对应的SSD后端。

        目的：一个QoS实例只服务一个StoragePath，使不同SSD的反压彼此隔离。

        输入：
            backend: 提供 ``can_accept_at_us`` 和 ``try_input_at_us`` 的SSD对象；
                传入None表示只仿真QoS。

        输出：
            DiscreteEventSimulator: 返回自身，便于装配代码连续调用。
        """
        self.backend = backend
        return self

    def set_queue_state_observer(self, observer):
        """功能：连接QoS Queue状态变化到DPU唤醒接口。

        目的：QoS在一轮下发改变Queue depth后只发送无负载
        唤醒。DPU若关心状态，必须另行读取 ``queue_io_counts``；
        这避免QoS向DPU泄露逐IO dispatch或Demand信息。

        输入：
            observer: 可调用对象，只接收 ``event_time_us``；
                None表示关闭状态唤醒。

        输出：
            DiscreteEventSimulator: 返回自身，便于顶层完成DPU与QoS直连装配。
        """
        self.queue_state_observer = observer
        return self

    def queue_io_counts(self):
        """功能：读取当前QoS全部Queue的逻辑IO数量。

        目的：一次性把Queue Bank occupancy状态传递给DPU，使多SSD环境中
        每个QoS实例都能独立提供自己的q000到q255状态。

        输入：
            无。

        输出：
            dict: ``queue_id -> 尚未下发IO数量`` 的独立快照。
        """
        return dict(self.registered_queue_io_counts)

    def schedule_queue_rate_update(
        self,
        queue_id,
        cir_fill_bytes_per_tick,
        pir_fill_bytes_per_tick,
        effective_time_us,
    ):
        """功能：登记一条由DPU发出的Queue CIR/PIR动态设置命令。

        目的：将DPU控制面调用转换成有明确生效时刻的QoS事件。同一Queue在
        同一时刻被多次重算时只保留最后一条，降低批量IO提交产生的控制开销。

        输入：
            queue_id: 当前QoS实例中的合法Queue ID。
            cir_fill_bytes_per_tick: 新CIR的整数周期补充Byte数。
            pir_fill_bytes_per_tick: 新PIR的整数周期补充Byte数；
                None表示uncapped。
            effective_time_us: 设置开始生效的仿真微秒时刻。

        输出：
            None: 登记或覆盖同时刻同Queue的控制事件。
        """
        update_key = (effective_time_us, queue_id)
        if update_key not in self.pending_rate_updates:
            heapq.heappush(self.pending_rate_update_keys, update_key)
        self.pending_rate_updates[update_key] = (
            cir_fill_bytes_per_tick,
            pir_fill_bytes_per_tick,
        )

    def schedule_group_weight_update(self, weights, effective_time_us):
        """功能：登记一张DPU动态Group WRR权重。

        目的：使Group机会分配与Queue CIR/PIR具有相同的离散事件
        因果性；同一时刻多次重算只保留DPU最后一张完整权重。

        输入：``group_id -> 整数权重`` 映射和生效微秒时刻。
        输出：无；登记或覆盖该时刻的Group控制事件。
        """
        if effective_time_us not in self.pending_group_weight_updates:
            heapq.heappush(
                self.pending_group_weight_update_keys,
                effective_time_us,
            )
        self.pending_group_weight_updates[effective_time_us] = dict(weights)

    def input(self, request):
        """功能：登记一个未来或当前时刻到达QoS的普通IO。

        目的：允许多个GPU以任意提交顺序登记请求，并通过最小堆保证实际接收
        始终遵守 ``arrival_time_us`` 和稳定提交顺序。

        输入：
            request: 包含 ``arrival_time_us``、``queue_id`` 和请求大小的字典。

        输出：
            dict: 原请求对象，便于上层保存同一份可追踪记录。
        """
        queue_id = request["queue_id"]
        self.arrival_sequence += 1
        heapq.heappush(
            self.pending_arrivals,
            (
                request["arrival_time_us"],
                self.arrival_sequence,
                request,
            ),
        )
        # 这是DPU可见Queue状态的写入点：请求一旦被QoS接受登记，就属于
        # 该Queue，即使同一时间戳的io_arrival阶段尚未把它移入内部FIFO。
        self.registered_queue_io_counts[queue_id] += 1
        self.total_request_count += 1
        return request

    def _apply_control_updates(self):
        """功能：应用不晚于当前QoS时刻的DPU控制设置。

        目的：在旧速率的周期补充已经结算后、IO参加调度前更新Queue令牌桶，
        从而保持动态CIR/PIR的离散时间因果关系。

        输入：
            无；读取控制事件堆和当前QoS时刻。

        输出：
            int: 本阶段实际生效的Queue和Group设置数量。
        """
        applied_count = 0
        while self.pending_rate_update_keys:
            effective_time_us, queue_id = self.pending_rate_update_keys[0]
            if effective_time_us > self.current_time_us:
                break
            heapq.heappop(self.pending_rate_update_keys)
            update_key = (effective_time_us, queue_id)
            cir_fill, pir_fill = self.pending_rate_updates.pop(update_key)
            self.token_stage.set_queue_rate(
                queue_id,
                cir_fill,
                pir_fill,
            )
            applied_count += 1

        while self.pending_group_weight_update_keys:
            effective_time_us = self.pending_group_weight_update_keys[0]
            if effective_time_us > self.current_time_us:
                break
            heapq.heappop(self.pending_group_weight_update_keys)
            weights = self.pending_group_weight_updates.pop(effective_time_us)
            self.scheduler.set_group_weights(weights)
            applied_count += 1
        return applied_count

    def _refill_tokens(self):
        """功能：补齐当前时刻以前所有尚未处理的令牌周期。

        目的：事件时间可能一次跨越多个80 μs周期，必须逐周期执行容量截断，
        才能与连续处理每个补充事件得到相同结果。

        输入：
            无；使用 ``current_time_us`` 和令牌周期。

        输出：
            None: 原地补充全部Queue令牌桶。
        """
        elapsed_us = self.current_time_us - self.start_time_us
        reached_periods = int(elapsed_us // self.token_stage.update_period_us)
        while self.completed_refill_periods < reached_periods:
            self.token_stage.refill()
            self.completed_refill_periods += 1

    def _is_token_refill_time(self):
        """功能：判断当前时刻是否跨过了尚未处理的令牌周期。

        目的：避免在非补充时刻无意义地遍历所有Queue，同时不遗漏跨期补充。

        输入：
            无。

        输出：
            bool: 至少存在一个未处理完整周期时返回True。
        """
        elapsed_us = self.current_time_us - self.start_time_us
        reached_periods = int(elapsed_us // self.token_stage.update_period_us)
        return reached_periods > self.completed_refill_periods

    def _accept_arrivals(self):
        """功能：把所有不晚于当前时刻的IO加入对应Queue FIFO。

        目的：同一时间戳先完整接收全部GPU到达，再执行WRR，避免调用顺序让
        第一张GPU在同时间事件中获得不应有的提前调度优势。

        输入：
            无；读取到达最小堆和当前QoS时刻。

        输出：
            None: 原地移动已到达请求到Queue FIFO。
        """
        while self.pending_arrivals:
            arrival_time_us, _, request = self.pending_arrivals[0]
            if arrival_time_us > self.current_time_us:
                break
            heapq.heappop(self.pending_arrivals)
            self.token_stage.enqueue(request)

    def _backend_can_accept(self):
        """功能：检查当前QoS对应SSD是否能在本时刻接收描述符。

        目的：在WRR移动游标、Queue出队和令牌扣除之前传播真实后端反压。

        输入：
            无；使用当前时刻和已经连接的backend。

        输出：
            bool: 独立QoS模式或SSD入口可用时返回True。
        """
        if self.backend is None:
            return True
        return self.backend.can_accept_at_us(self.current_time_us)

    def _send_to_backend(self, request):
        """功能：把已经从QoS出队的请求非阻塞提交给对应SSD。

        目的：保留DPU来源、目标SSD和QoS Queue等追踪字段，并验证调度前的
        后端可接收检查与实际提交结果一致。

        输入：
            request: 已写入 ``dispatch_time_us`` 的完整QoS请求。

        输出：
            float: 请求真正进入SSD FCP的微秒时刻。
        """
        backend_request = {
            "request_id": request["request_id"],
            "p_node_id": request["p_node_id"],
            "storage_target_id": request["storage_target_id"],
            "queue_id": request["queue_id"],
            "size_bytes": request["size_bytes"],
            "dispatch_time_us": request["dispatch_time_us"],
        }
        input_result = self.backend.try_input_at_us(
            backend_request,
            requested_time_us=self.current_time_us,
        )
        if not input_result["accepted"]:
            # 单线程事件模型中，can_accept与try_input之间没有其他组件修改SSD；
            # 如果仍失败，说明接口契约或事件顺序被破坏，不能静默丢失请求。
            raise RuntimeError(
                "SSD rejected a request after QoS observed available input capacity"
            )
        accepted_time_us = input_result["accepted_time_us"]
        request["backend_accept_time_us"] = accepted_time_us
        request["backend_wait_us"] = (
            accepted_time_us - request["dispatch_time_us"]
        )
        return accepted_time_us

    def _dispatch_scheduler(self):
        """功能：按CIR优先、EXCESS借用规则持续下发当前可接受请求。

        目的：保持原有两轮WRR语义，同时在SSD反压时立即停止本路径，让全局
        事件循环能够继续推进其他SSD，而不是在一次函数调用中等待未来空间。

        输入：
            无；读取Queue资格、WRR游标、令牌和后端入口状态。

        输出：
            int: 本次调用成功下发的完整IO数量。
        """
        dispatched_count = 0
        while True:
            # 后端检查必须发生在select_next_queue之前；否则SSD已满时一次失败
            # 尝试也会移动WRR游标，从而改变下一次真正成功仲裁的公平顺序。
            if not self._backend_can_accept():
                break

            queue_id = self.scheduler.select_next_queue(
                self.token_stage.is_cir_eligible
            )
            rate_class = "CIR"
            if queue_id is None:
                queue_id = self.scheduler.select_next_queue(
                    self.token_stage.is_excess_eligible
                )
                rate_class = "EXCESS"
            if queue_id is None:
                break

            request = self.token_stage.dequeue(
                queue_id,
                self.current_time_us,
                rate_class=rate_class,
            )
            request["dispatch_time_us"] = self.current_time_us
            request["dispatch_index"] = len(self.dispatched_requests) + 1

            if self.backend is not None:
                self._send_to_backend(request)
            self.dispatched_requests.append(request)
            dispatched_count += 1

            self.registered_queue_io_counts[queue_id] -= 1

        if dispatched_count and self.queue_state_observer is not None:
            # 一次非阻塞下发循环结束后只唤醒一次DPU，
            # 不携带刚下发的IO列表；DPU必须主动读取256条Queue状态。
            self.queue_state_observer(event_time_us=self.current_time_us)

        return dispatched_count

    def _process_current_time(self):
        """功能：按照YAML顺序处理当前QoS时刻的全部事件阶段。

        目的：显式固定令牌补充、IO到达和调度的同时间戳语义，使结果可复现。

        输入：
            无；使用simulation配置中的 ``same_timestamp_event_order``。

        输出：
            int: 当前时刻成功下发的请求数量。
        """
        dispatched_count = 0
        for event_name in self.simulation_config["same_timestamp_event_order"]:
            if event_name == "token_refill":
                if self._is_token_refill_time():
                    self._refill_tokens()
            elif event_name == "rate_update":
                self._apply_control_updates()
            elif event_name == "io_arrival":
                self._accept_arrivals()
            elif event_name in ("wrr_dispatch", "scheduler_dispatch"):
                dispatched_count += self._dispatch_scheduler()
            else:
                raise ValueError(f"unknown QoS event name {event_name!r}")
        return dispatched_count

    def _has_queued_requests(self):
        """功能：检查是否存在已经到达但尚未下发的Queue请求。

        目的：读取Queue Bank的硬件式总occupancy状态，决定QoS是否
        需要等待令牌或SSD入口事件，不再顺序扫描全部Queue。

        输入：
            无。

        输出：
            bool: 任意Queue FIFO非空时返回True。
        """
        return self.token_stage.has_queued_requests()

    def has_queued_requests(self):
        """功能：公开返回当前QoS是否仍有FIFO积压。

        目的：让StoragePath在SSD释放入口后只唤醒真正有待发请求的QoS。

        输入：
            无。

        输出：
            bool: 至少一个Queue有请求时返回True。
        """
        return self._has_queued_requests()

    def next_event_time_us(self):
        """功能：返回当前QoS自身下一次到达或令牌事件时刻。

        目的：供独立运行和全局StoragePath安排下一次QoS唤醒；SSD入口释放
        属于外部事件，由StoragePath在释放时额外安排同时间QoS调度。

        输入：
            无。

        输出：
            float | None: 下一QoS事件微秒时刻；完全空闲时返回None。
        """
        if self.end_result is not None:
            return None

        next_arrival_time = None
        if self.pending_arrivals:
            next_arrival_time = self.pending_arrivals[0][0]
            if next_arrival_time <= self.current_time_us:
                return self.current_time_us

        next_rate_update_time = None
        if self.pending_rate_update_keys:
            next_rate_update_time = self.pending_rate_update_keys[0][0]
            if next_rate_update_time <= self.current_time_us:
                return self.current_time_us

        next_group_weight_update_time = None
        if self.pending_group_weight_update_keys:
            next_group_weight_update_time = (
                self.pending_group_weight_update_keys[0]
            )
            if next_group_weight_update_time <= self.current_time_us:
                return self.current_time_us

        if not self._has_queued_requests():
            event_times = [
                event_time
                for event_time in (
                    next_arrival_time,
                    next_rate_update_time,
                    next_group_weight_update_time,
                )
                if event_time is not None
            ]
            return min(event_times) if event_times else None

        next_token_time = (
            self.start_time_us
            + (self.completed_refill_periods + 1)
            * self.token_stage.update_period_us
        )
        event_times = [next_token_time]
        if next_arrival_time is not None:
            event_times.append(next_arrival_time)
        if next_rate_update_time is not None:
            event_times.append(next_rate_update_time)
        if next_group_weight_update_time is not None:
            event_times.append(next_group_weight_update_time)
        return min(event_times)

    def process_at(self, event_time_us):
        """功能：在全局协调器指定的微秒时刻处理一次QoS事件。

        目的：禁止QoS组件自行越过其他SSD的更早事件，并允许SSD入口释放事件
        在非令牌时刻立即重新触发调度。

        输入：
            event_time_us: 不早于当前QoS时钟的全局事件时刻。

        输出：
            int: 该时刻成功下发的请求数量。
        """
        if event_time_us < self.current_time_us:
            raise RuntimeError(
                f"QoS time cannot move backwards from {self.current_time_us} "
                f"to {event_time_us} us"
            )
        self.current_time_us = event_time_us
        return self._process_current_time()

    def process_next_event(self):
        """功能：处理QoS自身的下一个事件时刻。

        目的：保留QoS独立仿真接口；联合多设备模式由StoragePath调用 ``process_at``。

        输入：
            无。

        输出：
            float | None: 处理后的QoS微秒时刻；无事件时返回None。
        """
        next_time_us = self.next_event_time_us()
        if next_time_us is None:
            return None
        self.process_at(next_time_us)
        return self.current_time_us

    def run(self):
        """功能：在没有SSD后端时调度全部已登记QoS请求。

        目的：保留纯QoS实验能力；连接非阻塞SSD后必须由全局事件协调器同时
        推进QoS和SSD，防止只推进QoS造成反压死循环。

        输入：
            无。

        输出：
            list[dict]: 当前已经成功下发的全部请求。
        """
        if self.end_result is not None:
            return self.dispatched_requests
        if self.backend is not None:
            raise RuntimeError(
                "QoS with a non-blocking backend must be run by StoragePath/EventLoop"
            )

        while len(self.dispatched_requests) < self.total_request_count:
            if self.process_next_event() is None:
                break

        # 最后一批IO出队时，DPU observer可能在当前时刻登记
        # “空Queue的CIR/PIR归零、Group权重重算”控制事件。
        # 纯QoS模式没有后续
        # SSD事件来再次唤醒QoS，因此在返回前应用所有当前时刻
        # 的状态设置；未来时刻的外部控制命令不会被提前执行。
        self._apply_control_updates()
        return self.dispatched_requests

    def _queue_statistics(self):
        """功能：按Queue汇总下发数量、字节、速率类别和最后下发时刻。

        目的：为绑定策略比较提供Queue热点、分散度和CIR/EXCESS使用情况。

        输入：
            无；遍历已经下发的请求记录。

        输出：
            dict: ``queue_id -> 统计字典`` 映射。
        """
        statistics = {
            queue_id: {
                "dispatched_requests": 0,
                "dispatched_bytes": 0,
                "cir_dispatched_requests": 0,
                "cir_dispatched_bytes": 0,
                "excess_dispatched_requests": 0,
                "excess_dispatched_bytes": 0,
                "last_dispatch_time_us": None,
            }
            for queue_id in self.token_stage.queue_order
        }

        for request in self.dispatched_requests:
            queue_statistics = statistics[request["queue_id"]]
            queue_statistics["dispatched_requests"] += 1
            queue_statistics["dispatched_bytes"] += request["size_bytes"]
            rate_class_key = request["qos_rate_class"].lower()
            queue_statistics[f"{rate_class_key}_dispatched_requests"] += 1
            queue_statistics[f"{rate_class_key}_dispatched_bytes"] += request[
                "size_bytes"
            ]
            queue_statistics["last_dispatch_time_us"] = request[
                "dispatch_time_us"
            ]
        return statistics

    def _all_queues_empty(self):
        """功能：判断当前QoS是否没有未来到达和FIFO积压。

        目的：最终结果不仅核对下发数量，还验证没有请求遗留在任何内部队列。

        输入：
            无。

        输出：
            bool: 到达堆和全部Queue均为空时返回True。
        """
        return not self._has_queued_requests() and not self.pending_arrivals

    def end(self):
        """功能：固化并返回当前QoS最终结果。

        目的：纯QoS模式先自动运行到完成；联合模式由全局事件循环完成推进后
        只做守恒统计，不越权排空SSD。

        输入：
            无。

        输出：
            dict: QoS完成状态、下发记录和每Queue统计。
        """
        if self.end_result is not None:
            return self.end_result
        if self.backend is None:
            self.run()

        all_dispatched = (
            len(self.dispatched_requests) == self.total_request_count
        )
        all_queues_empty = self._all_queues_empty()
        cir_requests = [
            request
            for request in self.dispatched_requests
            if request["qos_rate_class"] == "CIR"
        ]
        excess_requests = [
            request
            for request in self.dispatched_requests
            if request["qos_rate_class"] == "EXCESS"
        ]

        self.end_result = {
            "completed": all_dispatched and all_queues_empty,
            "start_time_us": self.start_time_us,
            "end_time_us": self.current_time_us,
            "input_request_count": self.total_request_count,
            "dispatched_request_count": len(self.dispatched_requests),
            "dispatched_bytes": sum(
                request["size_bytes"] for request in self.dispatched_requests
            ),
            "cir_dispatched_request_count": len(cir_requests),
            "cir_dispatched_bytes": sum(
                request["size_bytes"] for request in cir_requests
            ),
            "excess_dispatched_request_count": len(excess_requests),
            "excess_dispatched_bytes": sum(
                request["size_bytes"] for request in excess_requests
            ),
            "dispatched_requests": self.dispatched_requests,
            "queue_statistics": self._queue_statistics(),
            # 输出Queue状态和动态设置摘要，方便验证DPU确实连接到了每个SSD
            # 自己的QoS实例，而不是把相同queue_id跨SSD混在一起。
            "queue_io_counts": self.queue_io_counts(),
            # 输出最终Group WRR权重：本次Baseline和FCFS-CIR
            # 始终保持为1，但动态权重接口仍保留给未来策略。
            "group_weight_bitmap": [
                self.scheduler.group_scheduler.weights[group_id]
                for group_id in self.queue_layout.group_order
            ],
        }
        return self.end_result
