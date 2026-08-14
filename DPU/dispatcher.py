"""DPU把KV请求绑定到Queue，并直接读写每块SSD的QoS。"""

from functools import partial


class DPURequestGateway:
    """连接Queue绑定、需求感知控制和多个独立QoS实例。"""

    def __init__(
        self,
        queue_ids_by_storage_target,
        queue_binding_strategy,
        request_sink,
        qos_interfaces_by_storage_target,
        rate_controller=None,
    ):
        """功能：建立DPU数据路径和QoS状态/设置直连关系。

        输入：每SSD Queue列表、绑定策略、请求出口、QoS接口和可选速率控制器。
        输出：无，同时把QoS Queue状态唤醒连回DPU。
        """
        self.queue_ids = {
            target: list(queue_ids)
            for target, queue_ids in queue_ids_by_storage_target.items()
        }
        self.binding = queue_binding_strategy
        self.request_sink = request_sink
        self.qos = dict(qos_interfaces_by_storage_target)
        self.rate_controller = rate_controller
        self.assignment_counts = {}
        self.rate_control_write_count = 0
        self.group_weight_write_count = 0
        if self.rate_controller is not None:
            for storage_target_id, qos in self.qos.items():
                # QoS唤醒不携带IO或Demand信息；SSD ID由初始接线
                # 固定，DPU在回调中主动读取该QoS的Queue depth快照。
                qos.set_queue_state_observer(partial(
                    self.on_qos_queue_state_change,
                    storage_target_id,
                ))

    def queue_io_counts(self, storage_target_id):
        """功能：读取一块SSD前全部Queue的尚未下发IO数。

        输入：SSD ID。输出：``queue_id -> IO数量`` 快照。
        """
        return self.qos[storage_target_id].queue_io_counts()

    def set_queue_rate(
        self,
        storage_target_id,
        queue_id,
        cir_bytes_per_second,
        pir_bytes_per_second,
        effective_time_us,
    ):
        """功能：用整数Byte/s设置目标Queue的CIR/PIR。

        目的：CIR表示需求保证；PIR传入None时表示uncapped，
        允许Queue借用SSD的空闲带宽。

        输入：SSD、Queue、整数CIR、整数PIR或None和生效时刻。
        输出：无。
        """
        qos = self.qos[storage_target_id]
        period_us = qos.token_stage.update_period_us
        # RISC-V IM可用的整数乘除：Byte/s向上换算为每周期Byte。
        cir_fill = (
            cir_bytes_per_second * period_us + 999_999
        ) // 1_000_000
        pir_fill = None
        if pir_bytes_per_second is not None:
            pir_fill = (
                pir_bytes_per_second * period_us + 999_999
            ) // 1_000_000
        qos.schedule_queue_rate_update(
            queue_id,
            cir_fill,
            pir_fill,
            effective_time_us,
        )
        self.rate_control_write_count += 1

    def set_group_weights(
        self,
        storage_target_id,
        weights_bytes_per_second,
        effective_time_us,
    ):
        """功能：用整数带宽需求设置目标SSD的Group WRR权重。

        目的：把每Group活跃Queue诉求之和换算为每80 us的整数
        Byte权重，只调整组间机会，不创建Group CIR/PIR。

        输入：SSD ID、``group_id -> Byte/s`` 映射和生效微秒时刻。
        输出：无；登记一张同时生效的Group权重。
        """
        qos = self.qos[storage_target_id]
        period_us = qos.token_stage.update_period_us
        weights_per_tick = {
            group_id: (rate * period_us + 999_999) // 1_000_000
            for group_id, rate in weights_bytes_per_second.items()
        }
        qos.schedule_group_weight_update(
            weights_per_tick,
            effective_time_us,
        )
        self.group_weight_write_count += len(weights_per_tick)

    def _write_control_updates(
        self,
        storage_target_id,
        updates,
        event_time_us,
    ):
        """功能：把需求感知控制器输出写入目标QoS。

        目的：在同一个仿真时刻更新Queue CIR与Group WRR。
        Queue PIR保持uncapped，由SSD后端保证整盘物理上限。

        输入：SSD ID、Queue/Group控制变化和仿真时刻。
        输出：无；将变化写入QoS的待生效控制事件。
        """
        for queue_id, rate in updates["queue_rates"].items():
            # CIR表示该路径的带宽保证；PIR不封顶，让活跃
            # Queue在其他路径未用满时继续向SSD下发。
            self.set_queue_rate(
                storage_target_id,
                queue_id,
                rate,
                None,
                event_time_us,
            )
        if updates["group_weights"] is not None:
            self.set_group_weights(
                storage_target_id,
                updates["group_weights"],
                event_time_us,
            )

    def _submit(self, request, arrival_time_us):
        """功能：为一个IO绑定Queue并登记到对应QoS。

        输入：KV Placement请求和到达时刻。输出：展平后的QoS请求。
        """
        basic = request["basic"]
        target = basic["storage_target_id"]
        queue_id = self.binding.select_queue(
            request,
            self.queue_ids[target],
        )
        qos_request = {
            "request_id": basic["request_id"],
            "p_node_id": basic["p_node_id"],
            "storage_target_id": target,
            "size_bytes": basic["size_bytes"],
            "queue_id": queue_id,
            "arrival_time_us": arrival_time_us,
        }
        target_counts = self.assignment_counts.setdefault(target, {})
        node_counts = target_counts.setdefault(basic["p_node_id"], {})
        node_counts[queue_id] = node_counts.get(queue_id, 0) + 1
        self.request_sink(qos_request)
        return qos_request

    def submit_batch(self, requests, arrival_time_us):
        """功能：先登记同一层的全部IO，再创建DPU Demand。

        目的：先让QoS的逻辑Queue depth反映整个批次，然后DPU
        才按 ``(SSD, Queue)`` 登记一次KV Placement已聚合的诉求。
        这个顺序避免在批量尚未入队时把空Queue误判为已完成。

        输入：同一层请求列表和到达时刻。
        输出：仅包含普通IO数据面字段的QoS请求列表。
        """
        qos_requests = [
            self._submit(request, arrival_time_us) for request in requests
        ]
        if self.rate_controller is not None:
            demand_rates = {}
            for request, qos_request in zip(requests, qos_requests):
                path_key = (
                    qos_request["storage_target_id"],
                    qos_request["queue_id"],
                )
                # 同一GPU、同一层、同一SSD的Block重复携带
                # KV Placement已经聚合的同一整数诉求，DPU只登记一次。
                demand_rates[path_key] = request["demand_bw"][
                    "aggregate_required_bytes_per_second"
                ]

            affected_targets = set()
            for (target, queue_id), requested_cir in demand_rates.items():
                self.rate_controller.register_demand(
                    storage_target_id=target,
                    queue_id=queue_id,
                    requested_cir_bytes_per_second=requested_cir,
                    arrival_time_us=arrival_time_us,
                )
                affected_targets.add(target)

            for target in sorted(affected_targets):
                updates = self.rate_controller.recalculate(target)
                self._write_control_updates(target, updates, arrival_time_us)
        return qos_requests

    def on_qos_queue_state_change(self, storage_target_id, event_time_us):
        """功能：在QoS状态唤醒后主动读取Queue depth并释放Demand。

        目的：满足硬件接口“QoS→DPU只提供Queue空/非空或
        depth”。唤醒事件本身不携带request_id、Demand或逐IO
        dispatch信息；DPU只用快照中的0检测Demand结束。

        输入：
            storage_target_id: 初始接线已经确定的SSD ID。
            event_time_us: Queue状态变化发生的仿真时刻。

        输出：
            None: 必要时将重新分配的CIR/PIR写回同一QoS。
        """
        if self.rate_controller is None:
            return
        queue_depths = self.queue_io_counts(storage_target_id)
        updates = self.rate_controller.release_empty_demands(
            storage_target_id,
            queue_depths,
        )
        self._write_control_updates(
            storage_target_id,
            updates,
            event_time_us,
        )

    def statistics(self):
        """功能：返回Queue绑定、当前Queue计数和需求控制统计。

        输入：无。输出：可直接放入仿真结果的字典。
        """
        return {
            "strategy": self.binding.strategy_name,
            "assignment_counts": self.assignment_counts,
            "queue_io_counts_by_storage_target": {
                target: self.queue_io_counts(target) for target in self.qos
            },
            "rate_control_write_count": self.rate_control_write_count,
            "group_weight_write_count": self.group_weight_write_count,
            "rate_control": (
                None
                if self.rate_controller is None
                else self.rate_controller.statistics()
            ),
        }
