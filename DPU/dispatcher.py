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
        self.queue_weight_write_count = 0
        self.control_update_tick_aligned_write_count = 0
        self.control_update_non_tick_write_count = 0
        # Queue CIR/PIR是同一条硬件命令的两个字段。该镜像让
        # 未来控制器可以只输出其中一个变化，而不会重置另一个。
        self.queue_rate_settings = {}
        self.queue_weight_settings = {}
        if self.rate_controller is not None:
            for storage_target_id, qos in self.qos.items():
                # QoS唤醒不携带IO或Demand信息；SSD ID由初始接线
                # 固定，DPU在回调中主动读取该QoS的Queue depth快照。
                qos.set_queue_state_observer(partial(
                    self.on_qos_queue_state_change,
                    storage_target_id,
                ))
        if (
            self.rate_controller is not None
            and callable(getattr(
                self.rate_controller,
                "prepark_all_queues",
                None,
            ))
        ):
            queue_owners_by_storage_target = {
                target: {} for target in self.queue_ids
            }
            for binding_key, queue_id in getattr(
                self.binding,
                "bindings",
                {},
            ).items():
                if (
                    not isinstance(binding_key, tuple)
                    or len(binding_key) != 2
                ):
                    continue
                p_node_id, storage_target_id = binding_key
                if storage_target_id not in queue_owners_by_storage_target:
                    continue
                queue_owners_by_storage_target[storage_target_id][
                    queue_id
                ] = p_node_id
            # 有预绑定时，DPU只需要park实际属于p_node的
            # Queue（正式128-GPU拓扑为每盘128条）。只有
            # 通用binding不暴露预映射时，才fallback到该QoS
            # 的全部Queue，保证首个非边界Demand仍不泄漏。
            managed_queue_ids_by_storage_target = {
                storage_target_id: (
                    sorted(owners)
                    if owners
                    else list(self.queue_ids[storage_target_id])
                )
                for storage_target_id, owners in (
                    queue_owners_by_storage_target.items()
                )
            }
            initial_updates = (
                self.rate_controller.prepark_all_queues(
                    managed_queue_ids_by_storage_target,
                    queue_owners_by_storage_target,
                )
            )
            for storage_target_id, updates in initial_updates.items():
                initial_time_us = getattr(
                    self.qos[storage_target_id],
                    "start_time_us",
                    0,
                )
                self._write_control_updates(
                    storage_target_id,
                    updates,
                    initial_time_us,
                )

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
        self.queue_rate_settings[(storage_target_id, queue_id)] = (
            cir_bytes_per_second,
            pir_bytes_per_second,
        )
        self.rate_control_write_count += 1
        self._record_control_write_time(
            storage_target_id,
            effective_time_us,
        )

    def set_queue_weights(
        self,
        storage_target_id,
        weights,
        effective_time_us,
    ):
        """功能：动态写入一组Queue WRR权重。

        目的：coflow策略用0/1权重配合PIR Gate准入Queue，
        同时保持Group WRR权重为静态配置。

        输入：SSD ID、部分 ``queue_id -> 非负整数`` 更新和时刻。
        输出：无。
        """
        if not weights:
            return
        qos = self.qos[storage_target_id]
        normalized_weights = {}
        for queue_id, weight in weights.items():
            if (
                not isinstance(weight, int)
                or isinstance(weight, bool)
                or weight < 0
            ):
                raise ValueError(
                    f"weight for {queue_id!r} must be a non-negative integer"
                )
            normalized_weights[queue_id] = weight
        qos.schedule_queue_weight_update(
            normalized_weights,
            effective_time_us,
        )
        for queue_id, weight in normalized_weights.items():
            self.queue_weight_settings[(storage_target_id, queue_id)] = (
                weight
            )
        self.queue_weight_write_count += len(normalized_weights)
        self._record_control_write_time(
            storage_target_id,
            effective_time_us,
            write_count=len(normalized_weights),
        )

    def _record_control_write_time(
        self,
        storage_target_id,
        effective_time_us,
        write_count=1,
    ):
        """记录控制字段是否严格落在目标QoS的周期边界。"""
        qos = self.qos[storage_target_id]
        period_us = qos.token_stage.update_period_us
        origin_us = getattr(qos, "start_time_us", 0)
        relative_time_us = effective_time_us - origin_us
        quotient = relative_time_us // period_us
        aligned = relative_time_us == quotient * period_us
        if aligned:
            self.control_update_tick_aligned_write_count += write_count
        else:
            self.control_update_non_tick_write_count += write_count

    def _control_effective_time_us(
        self,
        storage_target_id,
        event_time_us,
        after_control_phase=False,
    ):
        """Return the hardware-effective control tick for one update.

        Utility+EDF commands are aligned to the target QoS token period.  A
        GPU arrival is processed before the same-timestamp QoS rate_update
        phase, so a command already on a tick may use that tick.  A Queue-empty
        callback is emitted from scheduler_dispatch after rate_update has
        passed, so it must always use the strictly following tick.
        """
        if not (
            getattr(
                self.rate_controller,
                "requires_tick_aligned_control",
                False,
            )
            or getattr(
                self.rate_controller,
                "strict_control_update_grid",
                False,
            )
        ):
            return event_time_us
        qos = self.qos[storage_target_id]
        period_us = qos.token_stage.update_period_us
        origin_us = getattr(qos, "start_time_us", 0)
        relative_time_us = event_time_us - origin_us
        if relative_time_us < 0:
            raise ValueError("control event cannot precede QoS start time")
        tick_index = int(relative_time_us // period_us)
        is_on_tick = (
            relative_time_us == tick_index * period_us
        )
        if after_control_phase or not is_on_tick:
            tick_index += 1
        return origin_us + tick_index * period_us

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
        self._record_control_write_time(
            storage_target_id,
            effective_time_us,
            write_count=len(weights_per_tick),
        )

    def _write_control_updates(
        self,
        storage_target_id,
        updates,
        event_time_us,
        after_control_phase=False,
    ):
        """功能：把需求感知控制器输出写入目标QoS。

        目的：在同一个仿真时刻更新Queue CIR与Group WRR。
        Queue PIR保持uncapped，由SSD后端保证整盘物理上限。

        输入：SSD ID、Queue/Group控制变化和仿真时刻。
        输出：无；将变化写入QoS的待生效控制事件。
        """
        effective_time_us = self._control_effective_time_us(
            storage_target_id,
            event_time_us,
            after_control_phase=after_control_phase,
        )
        queue_rates = updates.get("queue_rates", {})
        queue_pirs = updates.get("queue_pirs", {})
        for queue_id in sorted(queue_rates.keys() | queue_pirs.keys()):
            old_cir, old_pir = self.queue_rate_settings.get(
                (storage_target_id, queue_id),
                (0, None),
            )
            cir = queue_rates.get(queue_id, old_cir)
            pir = queue_pirs.get(queue_id, old_pir)
            self.set_queue_rate(
                storage_target_id,
                queue_id,
                cir,
                pir,
                effective_time_us,
            )
        queue_weights = updates.get("queue_weights", {})
        if queue_weights:
            self.set_queue_weights(
                storage_target_id,
                queue_weights,
                effective_time_us,
            )
        if updates.get("group_weights") is not None:
            self.set_group_weights(
                storage_target_id,
                updates["group_weights"],
                effective_time_us,
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
            # submit_batch的调用边界是一个完整层批次。先从
            # 普通IO字段推导路径/批次字节，再优先采用
            # KV Placement已显式给出的聚合元数据。
            group_total_bytes = {}
            group_explicit_total_bytes = {}
            path_demands = {}
            for request, qos_request in zip(requests, qos_requests):
                basic = request["basic"]
                demand_bw = request.get("demand_bw", {})
                p_node_id = basic["p_node_id"]
                demand_group_id = demand_bw.get(
                    "demand_group_id",
                    basic["request_id"],
                )
                group_key = (p_node_id, demand_group_id)
                group_total_bytes[group_key] = (
                    group_total_bytes.get(group_key, 0)
                    + basic["size_bytes"]
                )
                explicit_batch_total = next(
                    (
                        demand_bw[field]
                        for field in (
                            "batch_total_bytes",
                            "aggregate_batch_bytes",
                            "total_bytes",
                        )
                        if demand_bw.get(field) is not None
                    ),
                    None,
                )
                if explicit_batch_total is not None:
                    group_explicit_total_bytes[group_key] = int(
                        explicit_batch_total
                    )

                path_key = (
                    qos_request["storage_target_id"],
                    qos_request["queue_id"],
                )
                is_new_path = path_key not in path_demands
                path = path_demands.setdefault(path_key, {
                    "p_node_id": p_node_id,
                    "demand_group_id": demand_group_id,
                    "group_key": group_key,
                    "derived_path_bytes": 0,
                    "derived_path_request_count": 0,
                    "uniform_block_size_bytes": basic["size_bytes"],
                    "explicit_path_bytes": None,
                    "requested_cir": None,
                    "service_window_us": None,
                    "deadline_us": None,
                    "compute_layer_index": demand_bw.get(
                        "compute_layer_index"
                    ),
                    "prefetch_layer_index": demand_bw.get(
                        "prefetch_layer_index"
                    ),
                    "inference_arrival_time_us": demand_bw.get(
                        "inference_arrival_time_us"
                    ),
                })
                if (
                    path["p_node_id"] != p_node_id
                    or path["demand_group_id"] != demand_group_id
                ):
                    raise ValueError(
                        "one Queue cannot carry multiple demands in one batch"
                    )
                if not is_new_path:
                    for field in (
                        "compute_layer_index",
                        "prefetch_layer_index",
                        "inference_arrival_time_us",
                    ):
                        if path[field] != demand_bw.get(field):
                            raise ValueError(
                                f"inconsistent {field} within one path"
                            )
                path["derived_path_bytes"] += basic["size_bytes"]
                path["derived_path_request_count"] += 1
                if (
                    path["uniform_block_size_bytes"] is not None
                    and path["uniform_block_size_bytes"]
                    != basic["size_bytes"]
                ):
                    path["uniform_block_size_bytes"] = None

                explicit_path_bytes = next(
                    (
                        demand_bw[field]
                        for field in (
                            "aggregate_bytes_on_storage_target",
                            "path_bytes",
                            "aggregate_path_bytes",
                        )
                        if demand_bw.get(field) is not None
                    ),
                    None,
                )
                if explicit_path_bytes is not None:
                    path["explicit_path_bytes"] = int(explicit_path_bytes)

                requested_cir = demand_bw.get(
                    "aggregate_required_bytes_per_second"
                )
                if requested_cir is not None:
                    path["requested_cir"] = int(requested_cir)

                service_window_us = next(
                    (
                        demand_bw[field]
                        for field in (
                            "service_window_us",
                            "service_window",
                        )
                        if demand_bw.get(field) is not None
                    ),
                    None,
                )
                if service_window_us is not None:
                    path["service_window_us"] = service_window_us

                deadline_us = next(
                    (
                        demand_bw[field]
                        for field in (
                            "deadline_us",
                            "deadline_time_us",
                            "deadline",
                        )
                        if demand_bw.get(field) is not None
                    ),
                    None,
                )
                if deadline_us is not None:
                    path["deadline_us"] = deadline_us

            affected_targets = set()
            for (target, queue_id), path in path_demands.items():
                path_bytes = (
                    path["explicit_path_bytes"]
                    if path["explicit_path_bytes"] is not None
                    else path["derived_path_bytes"]
                )
                batch_total_bytes = group_explicit_total_bytes.get(
                    path["group_key"],
                    group_total_bytes[path["group_key"]],
                )
                requested_cir = path["requested_cir"]
                service_window_us = path["service_window_us"]
                deadline_us = path["deadline_us"]
                inference_arrival_time_us = path[
                    "inference_arrival_time_us"
                ]
                if inference_arrival_time_us is None:
                    inference_arrival_time_us = arrival_time_us

                if service_window_us is None:
                    if deadline_us is not None:
                        service_window_us = max(
                            0,
                            deadline_us - arrival_time_us,
                        )
                    elif requested_cir is not None and requested_cir > 0:
                        service_window_us = (
                            path_bytes * 1_000_000
                            + requested_cir - 1
                        ) // requested_cir
                    else:
                        service_window_us = 0
                if deadline_us is None:
                    deadline_us = arrival_time_us + service_window_us
                if requested_cir is None:
                    if service_window_us > 0:
                        requested_cir = (
                            path_bytes * 1_000_000
                            + service_window_us - 1
                        ) // service_window_us
                    else:
                        requested_cir = 0

                self.rate_controller.register_demand(
                    storage_target_id=target,
                    queue_id=queue_id,
                    requested_cir_bytes_per_second=requested_cir,
                    arrival_time_us=arrival_time_us,
                    p_node_id=path["p_node_id"],
                    demand_group_id=path["demand_group_id"],
                    batch_total_bytes=batch_total_bytes,
                    path_bytes=path_bytes,
                    path_request_count=path[
                        "derived_path_request_count"
                    ],
                    block_size_bytes=path[
                        "uniform_block_size_bytes"
                    ],
                    service_window_us=service_window_us,
                    deadline_us=deadline_us,
                    compute_layer_index=path["compute_layer_index"],
                    prefetch_layer_index=path["prefetch_layer_index"],
                    inference_arrival_time_us=inference_arrival_time_us,
                )
                affected_targets.add(target)

            if getattr(
                self.rate_controller,
                "coordinates_storage_targets",
                False,
            ):
                targets_to_recalculate = sorted(self.rate_controller.capacity)
            else:
                targets_to_recalculate = sorted(affected_targets)
            for target in targets_to_recalculate:
                if getattr(
                    self.rate_controller,
                    "uses_queue_depths_for_recalculate",
                    False,
                ):
                    updates = self.rate_controller.recalculate(
                        target,
                        event_time_us=arrival_time_us,
                        queue_depths=self.queue_io_counts(target),
                    )
                elif getattr(
                    self.rate_controller,
                    "coordinates_storage_targets",
                    False,
                ):
                    updates = self.rate_controller.recalculate(
                        target,
                        event_time_us=arrival_time_us,
                    )
                else:
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
        coordinates_targets = getattr(
            self.rate_controller,
            "coordinates_storage_targets",
            False,
        )
        uses_event_time_for_release = getattr(
            self.rate_controller,
            "uses_event_time_for_release",
            False,
        )
        if coordinates_targets or uses_event_time_for_release:
            updates = self.rate_controller.release_empty_demands(
                storage_target_id,
                queue_depths,
                event_time_us=event_time_us,
            )
        else:
            updates = self.rate_controller.release_empty_demands(
                storage_target_id,
                queue_depths,
            )
        self._write_control_updates(
            storage_target_id,
            updates,
            event_time_us,
            after_control_phase=True,
        )
        if coordinates_targets and updates.get("coordinates_changed", False):
            # 最后一条路径释放后，全局top-K可能发生变化；
            # 其他SSD上新选中GPU的Queue也必须在同时刻解锁。
            for target in sorted(self.rate_controller.capacity):
                if target == storage_target_id:
                    continue
                if getattr(
                    self.rate_controller,
                    "uses_queue_depths_for_recalculate",
                    False,
                ):
                    target_updates = self.rate_controller.recalculate(
                        target,
                        event_time_us=event_time_us,
                        queue_depths=self.queue_io_counts(target),
                    )
                else:
                    target_updates = self.rate_controller.recalculate(
                        target,
                        event_time_us=event_time_us,
                    )
                self._write_control_updates(
                    target,
                    target_updates,
                    event_time_us,
                    after_control_phase=True,
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
            "queue_weight_write_count": self.queue_weight_write_count,
            "control_update_tick_aligned_write_count": (
                self.control_update_tick_aligned_write_count
            ),
            "control_update_non_tick_write_count": (
                self.control_update_non_tick_write_count
            ),
            "control_update_period_us_by_storage_target": {
                target: qos.token_stage.update_period_us
                for target, qos in self.qos.items()
            },
            "rate_control": (
                None
                if self.rate_controller is None
                else self.rate_controller.statistics()
            ),
        }
