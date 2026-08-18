"""DPU内部的Queue速率与coflow优先级控制器。"""

from fractions import Fraction
from functools import cmp_to_key
import re


class DemandAwareFCFSCIRController:
    """按FCFS或完整coflow大小为每块SSD分配有限CIR容量。"""

    strategy_name = "demand_aware_fcfs_cir"
    supports_chunked_demands = True

    def __init__(
        self,
        capacity_bytes_per_second_by_storage_target,
        ordering="fcfs",
    ):
        """功能：为每块SSD创建独立的FCFS CIR状态。

        目的：DPU只使用整数Byte/s做减法、比较和取最小值，
        不使用浮点比例缩放。每条 ``(SSD, queue_id)`` 只保存
        当前层的requested CIR、assigned CIR和到达顺序。

        输入：
            capacity_bytes_per_second_by_storage_target:
                ``SSD ID -> 整数物理带宽Byte/s`` 映射。

        输出：
            None: 初始化空Demand表和速率镜像。
        """
        if ordering not in ("fcfs", "shortest"):
            raise ValueError("CIR ordering must be 'fcfs' or 'shortest'")
        self.capacity = dict(capacity_bytes_per_second_by_storage_target)
        self.ordering = ordering
        self.strategy_name = f"demand_aware_{ordering}_cir"
        self.demands = {target: {} for target in self.capacity}
        self.queue_rates = {target: {} for target in self.capacity}
        self.arrival_sequence = {target: 0 for target in self.capacity}
        self.completed_demand_count = {target: 0 for target in self.capacity}
        self.peak_assigned_cir = {target: 0 for target in self.capacity}
        self.registered_chunk_count = 0
        self.intermediate_empty_count = 0

    def register_demand(
        self,
        storage_target_id,
        queue_id,
        requested_cir_bytes_per_second,
        arrival_time_us,
        p_node_id=None,
        demand_group_id=None,
        batch_total_bytes=None,
        path_bytes=None,
        path_request_count=None,
        block_size_bytes=None,
        service_window_us=None,
        deadline_us=None,
        compute_layer_index=None,
        prefetch_layer_index=None,
        inference_arrival_time_us=None,
        submission_chunk_index=0,
        submission_chunk_count=1,
        submission_complete=True,
    ):
        """功能：登记一条已经完整批量入队的Queue Demand。

        目的：每个GPU在一块SSD上独占Queue，且层与层串行，
        因此 ``(SSD, queue_id)`` 足以唯一标识当前Demand。
        该函数只在整层所有IO都已向QoS登记后调用，避免把
        批量入队前的空Queue误判为Demand完成。

        输入：
            storage_target_id: Demand对应的SSD ID。
            queue_id: 当前GPU→SSD路径独占的Queue ID。
            requested_cir_bytes_per_second: KV Placement生成的整数Byte/s诉求。
            arrival_time_us: 该层IO批量到达DPU的仿真时刻。

        输出：
            None: 在DPU内部保存新Demand，尚不写QoS。
        """
        if (
            not isinstance(submission_chunk_index, int)
            or isinstance(submission_chunk_index, bool)
            or submission_chunk_index < 0
        ):
            raise ValueError("submission_chunk_index must be non-negative")
        if (
            not isinstance(submission_chunk_count, int)
            or isinstance(submission_chunk_count, bool)
            or submission_chunk_count <= 0
        ):
            raise ValueError("submission_chunk_count must be positive")
        if submission_chunk_index >= submission_chunk_count:
            raise ValueError(
                "submission_chunk_index must be less than chunk count"
            )
        if not isinstance(submission_complete, bool):
            raise TypeError("submission_complete must be bool")
        if submission_complete != (
            submission_chunk_index + 1 == submission_chunk_count
        ):
            raise ValueError(
                "submission_complete must identify the final chunk"
            )
        normalized_batch_total_bytes = (
            path_bytes if batch_total_bytes is None else batch_total_bytes
        )
        if normalized_batch_total_bytes is None:
            normalized_batch_total_bytes = 0
        if (
            not isinstance(normalized_batch_total_bytes, int)
            or isinstance(normalized_batch_total_bytes, bool)
            or normalized_batch_total_bytes < 0
        ):
            raise ValueError("batch_total_bytes must be non-negative")

        demands = self.demands[storage_target_id]
        existing = demands.get(queue_id)
        if existing is not None:
            if existing["submission_chunk_count"] == 1:
                raise ValueError(
                    f"queue {queue_id!r} on {storage_target_id!r} "
                    "already has an active non-chunked demand"
                )
            if existing["demand_group_id"] != demand_group_id:
                raise ValueError(
                    "one Queue cannot interleave two chunked demands"
                )
            if existing["submission_chunk_count"] != submission_chunk_count:
                raise ValueError("chunk count changed within one demand")
            if submission_chunk_index != existing["last_chunk_index"] + 1:
                raise ValueError("demand chunks must arrive in order")
            if existing["submission_complete"]:
                raise ValueError("a completed submission cannot accept chunks")
            if (
                existing["requested_cir"]
                != requested_cir_bytes_per_second
            ):
                raise ValueError("requested CIR changed within one demand")
            if (
                existing["batch_total_bytes"]
                != normalized_batch_total_bytes
            ):
                raise ValueError("batch size changed within one demand")
            existing["last_chunk_index"] = submission_chunk_index
            existing["submission_complete"] = submission_complete
            existing["awaiting_next_chunk"] = False
            self.registered_chunk_count += 1
            return

        if submission_chunk_index != 0:
            raise ValueError("a chunked demand must start at chunk zero")
        self.arrival_sequence[storage_target_id] += 1
        demands[queue_id] = {
            "requested_cir": requested_cir_bytes_per_second,
            "assigned_cir": 0,
            "arrival_time_us": arrival_time_us,
            "arrival_order": self.arrival_sequence[storage_target_id],
            "demand_group_id": demand_group_id,
            "batch_total_bytes": normalized_batch_total_bytes,
            "submission_chunk_count": submission_chunk_count,
            "last_chunk_index": submission_chunk_index,
            "submission_complete": submission_complete,
            "awaiting_next_chunk": False,
        }
        self.registered_chunk_count += 1

    def recalculate(self, storage_target_id, event_time_us=None):
        """功能：按配置的FCFS或shortest顺序重新分配一块SSD的CIR。

        目的：对每个活跃Demand执行
        ``assigned=min(requested, remaining_capacity)``。后到Demand
        可以获得部分CIR或0 CIR，但不会被禁止通过EXCESS下发。

        输入：
            storage_target_id: 需要重算的SSD ID。

        输出：
            dict: 只包含发生变化的 ``queue_id -> CIR Byte/s``，
            以及固定为None的 ``group_weights``。
        """
        remaining_capacity = self.capacity[storage_target_id]
        new_queue_rates = {}

        def priority(item):
            demand = item[1]
            if self.ordering == "shortest":
                return (
                    demand["batch_total_bytes"],
                    demand["arrival_time_us"],
                    demand["arrival_order"],
                )
            return (
                demand["arrival_time_us"],
                demand["arrival_order"],
            )

        ordered_demands = sorted(
            self.demands[storage_target_id].items(),
            key=priority,
        )

        for queue_id, demand in ordered_demands:
            assigned_cir = min(
                demand["requested_cir"],
                remaining_capacity,
            )
            demand["assigned_cir"] = assigned_cir
            new_queue_rates[queue_id] = assigned_cir
            remaining_capacity -= assigned_cir

        old_queue_rates = self.queue_rates[storage_target_id]
        queue_updates = {
            queue_id: new_queue_rates.get(queue_id, 0)
            for queue_id in sorted(old_queue_rates.keys() | new_queue_rates.keys())
            if (
                queue_id not in old_queue_rates
                or queue_id not in new_queue_rates
                or old_queue_rates[queue_id] != new_queue_rates[queue_id]
            )
        }
        self.queue_rates[storage_target_id] = new_queue_rates
        assigned_total = sum(new_queue_rates.values())
        self.peak_assigned_cir[storage_target_id] = max(
            self.peak_assigned_cir[storage_target_id],
            assigned_total,
        )
        return {
            "queue_rates": queue_updates,
            # 本策略严格隔离Queue CIR变量，不写Group WRR。
            # QoS的动态Group权重接口仍保留给未来策略。
            "group_weights": None,
        }

    def release_empty_demands(
        self,
        storage_target_id,
        queue_depths,
        event_time_us=None,
    ):
        """功能：仅根据Queue depth快照释放已经排空的Demand。

        目的：QoS到DPU的状态接口不携带IO、request_id或
        demand_id。DPU只查看自己正在跟踪的 ``(SSD, Queue)``；
        Queue从已登记的非空状态变为0，即表示该Demand的IO
        已全部离开QoS，可以释放CIR并重新分配。

        输入：
            storage_target_id: 发生Queue状态变化的SSD ID。
            queue_depths: QoS读出的 ``queue_id -> 尚未下发IO数`` 快照。

        输出：
            dict: 释放后需要写回QoS的Queue CIR变化。
        """
        demands = self.demands[storage_target_id]
        empty_queue_ids = [
            queue_id
            for queue_id, demand in demands.items()
            if (
                queue_depths[queue_id] == 0
                and demand["submission_complete"]
            )
        ]
        for queue_id, demand in demands.items():
            if (
                queue_depths[queue_id] == 0
                and not demand["submission_complete"]
                and not demand["awaiting_next_chunk"]
            ):
                demand["awaiting_next_chunk"] = True
                self.intermediate_empty_count += 1
        for queue_id in empty_queue_ids:
            del demands[queue_id]
            self.completed_demand_count[storage_target_id] += 1
        return self.recalculate(storage_target_id)

    def statistics(self):
        """功能：返回不暴露单Demand速率的DPU控制面统计。

        目的：供回归测试确认仿真结束时没有残留Demand，并检查
        峰值CIR总和从未超过SSD容量。不输出requested CIR或
        assigned CIR的逐Demand明细。

        输入：无。

        输出：
            dict: 策略名、活跃/完成Demand数和每SSD峰值CIR总和。
        """
        return {
            "strategy": self.strategy_name,
            "ordering": self.ordering,
            "active_demand_count": sum(
                len(demands) for demands in self.demands.values()
            ),
            "completed_demand_count_by_storage_target": dict(
                self.completed_demand_count
            ),
            "peak_assigned_cir_bytes_per_second": dict(
                self.peak_assigned_cir
            ),
            "registered_chunk_count": self.registered_chunk_count,
            "intermediate_empty_count": self.intermediate_empty_count,
        }


class CoflowPriorityController:
    """用跨SSD一致的GPU优先级集中服务少数coflow。

    控制器只保存DPU在请求提交时可见的元数据，运行时只读
    Queue depth和仿真时钟。它不读SSD completion、inflight或NAND状态。
    每个p_node首次出现时固化一个推理级优先级，因此同一GPU
    后续层在所有SSD上使用同一排名。
    """

    strategy_name = "coflow_priority"
    coordinates_storage_targets = True
    _DEFAULT_QUEUE_STATE = (0, None, 1)

    def __init__(
        self,
        capacity_bytes_per_second_by_storage_target,
        ordering="shortest",
        selection_width=1,
        persistent_cohort=False,
        expected_coflow_count=None,
        finite_selected_pir=False,
    ):
        """功能：创建跨SSD共享排名的coflow控制状态。

        输入：
            capacity_bytes_per_second_by_storage_target: ``SSD -> Byte/s``。
            ordering: ``shortest``/``largest``按首批总字节排序，
                ``highest_demand``/``lowest_demand``按首批带宽需求
                降序/升序，``longest_window``按服务窗口降序，
                ``min_window_<正整数微秒>``先选首批窗口不小于
                阈值的GPU，两组内都按首批字节升序，
                ``most_progress``/``remaining_shortest``使用已完成
                coflow数动态排序，``fifo``按到达。
            selection_width: 同时准入的p_node数K，必须为正整数。
            persistent_cohort: True时，选中p_node在层间无Queue
                期间仍保留准入槽，直到完成预期数量的coflow。
            expected_coflow_count: persistent模式下每个p_node完成所需
                的KV读组数。
            finite_selected_pir: True时将选中Queue的PIR设为
                与CIR share相同的有限速率，禁止它通过EXCESS超发。

        输出：None。
        """
        fixed_orderings = (
            "shortest",
            "largest",
            "highest_demand",
            "lowest_demand",
            "longest_window",
            "most_progress",
            "remaining_shortest",
            "fifo",
        )
        min_window_match = (
            re.fullmatch(r"min_window_([1-9][0-9]*)", ordering)
            if isinstance(ordering, str)
            else None
        )
        if ordering not in fixed_orderings and min_window_match is None:
            raise ValueError(
                "ordering must be 'shortest', 'largest', "
                "'highest_demand', 'lowest_demand', "
                "'longest_window', 'most_progress', "
                "'remaining_shortest', 'fifo', or "
                "'min_window_<positive integer microseconds>'"
            )
        min_window_threshold_us = (
            int(min_window_match.group(1))
            if min_window_match is not None
            else None
        )
        if (
            not isinstance(selection_width, int)
            or isinstance(selection_width, bool)
            or selection_width <= 0
        ):
            raise ValueError("selection_width must be a positive integer")
        if not isinstance(persistent_cohort, bool):
            raise TypeError("persistent_cohort must be bool")
        if not isinstance(finite_selected_pir, bool):
            raise TypeError("finite_selected_pir must be bool")
        if expected_coflow_count is not None and (
            not isinstance(expected_coflow_count, int)
            or isinstance(expected_coflow_count, bool)
            or expected_coflow_count <= 0
        ):
            raise ValueError(
                "expected_coflow_count must be a positive integer"
            )
        if persistent_cohort and expected_coflow_count is None:
            raise ValueError(
                "persistent_cohort requires expected_coflow_count"
            )
        if ordering in ("most_progress", "remaining_shortest") and (
            expected_coflow_count is None
        ):
            raise ValueError(
                f"{ordering} requires expected_coflow_count"
            )

        self.capacity = dict(capacity_bytes_per_second_by_storage_target)
        if not self.capacity:
            raise ValueError("at least one storage target is required")
        for target, capacity in self.capacity.items():
            if (
                not isinstance(capacity, int)
                or isinstance(capacity, bool)
                or capacity <= 0
            ):
                raise ValueError(
                    f"capacity for {target!r} must be a positive integer"
                )

        self.ordering = ordering
        self.min_window_threshold_us = min_window_threshold_us
        self.selection_width = selection_width
        self.persistent_cohort = persistent_cohort
        self.expected_coflow_count = expected_coflow_count
        self.finite_selected_pir = finite_selected_pir
        # 首个Queue真正排空前，cohort只是候选快照，
        # 同时到达的更优GPU仍可替换它。首次服务后
        # 才锁定，避免Python提交顺序决定正式cohort。
        self.cohort_started = False
        self.cohort_locked = False
        self.cohort_members = ()
        self.demands = {target: {} for target in self.capacity}
        self.queue_rates = {target: {} for target in self.capacity}
        self.queue_pirs = {target: {} for target in self.capacity}
        self.queue_weights = {target: {} for target in self.capacity}
        self._programmed_queue_states = {
            target: {} for target in self.capacity
        }
        self.p_node_profiles = {}
        self.active_group_paths = {}
        self.arrival_sequence = 0
        self.current_time_us = 0
        self.selected_p_nodes = ()
        self.completed_demand_count = {
            target: 0 for target in self.capacity
        }
        self.completed_coflow_count = 0
        self.peak_assigned_cir = {
            target: 0 for target in self.capacity
        }
        self.total_selection_changes = 0
        self.max_queue_wait_us = 0
        self.total_queue_wait_us = 0
        self.selected_queue_count = 0

    @staticmethod
    def _non_negative_int(value, fallback=0):
        """将可选元数据收敛为非负整数。"""
        if value is None:
            return fallback
        normalized = int(value)
        if normalized < 0:
            raise ValueError("demand metadata must be non-negative")
        return normalized

    @staticmethod
    def _non_negative_time(value, fallback=0):
        """保留仿真时钟精度并拒绝负时间。"""
        if value is None:
            return fallback
        if isinstance(value, bool) or value < 0:
            raise ValueError("demand time metadata must be non-negative")
        return value

    @staticmethod
    def _exact_fraction(value):
        """将整数或十进制时间转成精确有理数。

        浮点输入先转成其十进制文本，不把二进制表示误差
        带入排名。``Fraction``比较时使用整数交叉相乘。
        """
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return Fraction(value, 1)
        return Fraction(str(value))

    def _active_p_nodes(self):
        """返回至少在一块SSD上有活跃Queue的p_node集合。"""
        return {
            demand["p_node_id"]
            for target_demands in self.demands.values()
            for demand in target_demands.values()
        }

    def _profile_priority_key(self, p_node_id):
        """返回首次登记后不再变化的推理级排序键。"""
        profile = self.p_node_profiles[p_node_id]
        if self.ordering == "shortest":
            return (
                profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "largest":
            return (
                -profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "highest_demand":
            priority_demand = profile[
                "priority_demand_bytes_per_second"
            ]
            # service_window=0表示没有可等待窗口，排在所有
            # 有限demand之前；其余按Byte/s降序。
            return (
                0 if priority_demand is None else 1,
                0 if priority_demand is None else -priority_demand,
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "lowest_demand":
            ratio_numerator = profile[
                "priority_demand_ratio_numerator"
            ]
            if ratio_numerator is None:
                # 零窗口无法定义有限的bytes/window比例，
                # lowest_demand明确将它放在所有有限比例之后。
                return (
                    1,
                    Fraction(0, 1),
                    profile["priority_batch_total_bytes"],
                    profile["first_arrival_time_us"],
                    profile["arrival_order"],
                )
            return (
                0,
                Fraction(
                    ratio_numerator,
                    profile["priority_demand_ratio_denominator"],
                ),
                profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "longest_window":
            service_window = Fraction(
                profile["priority_service_window_numerator"],
                profile["priority_service_window_denominator"],
            )
            return (
                -service_window,
                profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.min_window_threshold_us is not None:
            meets_threshold = (
                profile["priority_service_window_numerator"]
                >= self.min_window_threshold_us
                * profile["priority_service_window_denominator"]
            )
            return (
                0 if meets_threshold else 1,
                profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "most_progress":
            completed_count = profile["completed_coflow_count"]
            return (
                -completed_count,
                profile["priority_batch_total_bytes"],
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        if self.ordering == "remaining_shortest":
            completed_count = profile["completed_coflow_count"]
            return (
                profile["priority_batch_total_bytes"]
                * (self.expected_coflow_count - completed_count),
                -completed_count,
                profile["first_arrival_time_us"],
                profile["arrival_order"],
            )
        return (
            profile["first_arrival_time_us"],
            profile["arrival_order"],
        )

    def _ordered_active_p_nodes(self):
        """按全局固定优先级排序当前活跃p_node。"""
        return sorted(
            self._active_p_nodes(),
            key=self._profile_priority_key,
        )

    def _persistent_cohort_selection(self):
        """生成persistent模式的全局cohort成员。

        未锁定时，每次从当前全部活跃GPU重选top-K；
        锁定后，未完成预期coflow数的owner即使暂时无
        Queue也留在槽中，空槽才按固定排名补入等待GPU。
        """
        ordered_active = self._ordered_active_p_nodes()
        if not self.cohort_locked:
            selected = ordered_active[: self.selection_width]
            self.cohort_members = tuple(selected)
            return selected

        retained = [
            p_node_id
            for p_node_id in self.cohort_members
            if self.p_node_profiles[p_node_id][
                "completed_coflow_count"
            ] < self.expected_coflow_count
        ]
        retained_set = set(retained)
        candidates = [
            p_node_id
            for p_node_id in ordered_active
            if (
                p_node_id not in retained_set
                and self.p_node_profiles[p_node_id][
                    "completed_coflow_count"
                ] < self.expected_coflow_count
            )
        ]
        selected = retained + candidates[
            : self.selection_width - len(retained)
        ]
        self.cohort_members = tuple(selected)
        return selected

    def _is_p_node_active(self, p_node_id):
        """检查一个p_node是否仍有任意路径Demand。"""
        return any(
            demand["p_node_id"] == p_node_id
            for target_demands in self.demands.values()
            for demand in target_demands.values()
        )

    def _ensure_profile(
        self,
        p_node_id,
        arrival_time_us,
        batch_total_bytes,
        service_window_us,
    ):
        """首次见到GPU时固化其推理级排名元数据。"""
        if p_node_id in self.p_node_profiles:
            return self.p_node_profiles[p_node_id]
        self.arrival_sequence += 1
        exact_service_window = self._exact_fraction(service_window_us)
        if service_window_us == 0:
            priority_demand = None
            demand_ratio_numerator = None
            demand_ratio_denominator = None
        else:
            exact_demand_ratio = (
                Fraction(batch_total_bytes, 1) / exact_service_window
            )
            demand_per_second = exact_demand_ratio * 1_000_000
            priority_demand = (
                demand_per_second.numerator
                + demand_per_second.denominator
                - 1
            ) // demand_per_second.denominator
            demand_ratio_numerator = exact_demand_ratio.numerator
            demand_ratio_denominator = exact_demand_ratio.denominator
        profile = {
            "first_arrival_time_us": arrival_time_us,
            "arrival_order": self.arrival_sequence,
            "priority_batch_total_bytes": batch_total_bytes,
            "priority_service_window_us": service_window_us,
            "priority_service_window_numerator": (
                exact_service_window.numerator
            ),
            "priority_service_window_denominator": (
                exact_service_window.denominator
            ),
            "priority_demand_bytes_per_second": priority_demand,
            "priority_demand_ratio_numerator": demand_ratio_numerator,
            "priority_demand_ratio_denominator": demand_ratio_denominator,
            "activation_count": 0,
            "selection_count": 0,
            "completed_demand_count": 0,
            "completed_coflow_count": 0,
            "total_wait_us": 0,
            "max_wait_us": 0,
            "total_selected_us": 0,
            "active_since_us": None,
            "waiting_since_us": None,
            "selected_since_us": None,
        }
        self.p_node_profiles[p_node_id] = profile
        return profile

    def register_demand(
        self,
        storage_target_id,
        queue_id,
        requested_cir_bytes_per_second,
        arrival_time_us,
        p_node_id=None,
        demand_group_id=None,
        batch_total_bytes=None,
        path_bytes=None,
        path_request_count=None,
        block_size_bytes=None,
        service_window_us=None,
        deadline_us=None,
        compute_layer_index=None,
        prefetch_layer_index=None,
        inference_arrival_time_us=None,
    ):
        """功能：登记一条coflow的SSD路径Demand。

        ``batch_total_bytes``只在p_node首次出现时用来固化shortest
        排名；后续层即使大小不同也沿用原排名。
        """
        if storage_target_id not in self.demands:
            raise KeyError(f"unknown storage target {storage_target_id!r}")
        if queue_id in self.demands[storage_target_id]:
            raise ValueError(
                f"queue {queue_id!r} on {storage_target_id!r} "
                "already has an active demand"
            )
        if p_node_id is None:
            # 旧调用方未传元数据时仍给出稳定身份，但正式
            # coflow路径由dispatcher始终传入真实p_node_id。
            p_node_id = f"{storage_target_id}:{queue_id}"
        if demand_group_id is None:
            demand_group_id = (
                p_node_id,
                arrival_time_us,
            )

        arrival_time_us = self._non_negative_time(arrival_time_us)
        requested_cir = self._non_negative_int(
            requested_cir_bytes_per_second
        )
        path_bytes = self._non_negative_int(path_bytes)
        if batch_total_bytes is None:
            batch_total_bytes = path_bytes
        batch_total_bytes = self._non_negative_int(
            batch_total_bytes,
            fallback=path_bytes,
        )
        service_window_us = self._non_negative_time(service_window_us)
        if deadline_us is None:
            deadline_us = arrival_time_us + service_window_us
        deadline_us = self._non_negative_time(deadline_us)

        was_active = self._is_p_node_active(p_node_id)
        profile = self._ensure_profile(
            p_node_id,
            arrival_time_us,
            batch_total_bytes,
            service_window_us,
        )
        if not was_active:
            profile["activation_count"] += 1
            profile["active_since_us"] = arrival_time_us
            profile["waiting_since_us"] = arrival_time_us
            if (
                self.persistent_cohort
                and self.cohort_locked
                and p_node_id in self.cohort_members
            ):
                # owner在compute/SSD-drain空档期间始终占有槽位；
                # 新层Queue到达时不应被统计为重新等待准入。
                profile["waiting_since_us"] = None

        group_key = (p_node_id, demand_group_id)
        demand = {
            "requested_cir": requested_cir,
            "assigned_cir": 0,
            "arrival_time_us": arrival_time_us,
            "p_node_id": p_node_id,
            "demand_group_id": demand_group_id,
            "batch_total_bytes": batch_total_bytes,
            "path_bytes": path_bytes,
            "service_window_us": service_window_us,
            "deadline_us": deadline_us,
            "first_selected_time_us": None,
            "selection_count": 0,
        }
        self.demands[storage_target_id][queue_id] = demand
        self.active_group_paths.setdefault(group_key, set()).add(
            (storage_target_id, queue_id)
        )
        self.current_time_us = max(self.current_time_us, arrival_time_us)

    def _record_selection_transition(self, selected_p_nodes, event_time_us):
        """统计GPU在全局准入集中的等待与服务区间。"""
        active_p_nodes = self._active_p_nodes()
        old_selected = set(self.selected_p_nodes)
        new_selected = set(selected_p_nodes)

        for p_node_id in old_selected - new_selected:
            profile = self.p_node_profiles[p_node_id]
            selected_since = profile["selected_since_us"]
            if selected_since is not None:
                profile["total_selected_us"] += max(
                    0,
                    event_time_us - selected_since,
                )
            profile["selected_since_us"] = None
            profile["waiting_since_us"] = (
                event_time_us if p_node_id in active_p_nodes else None
            )

        for p_node_id in new_selected - old_selected:
            profile = self.p_node_profiles[p_node_id]
            waiting_since = profile["waiting_since_us"]
            wait_us = 0
            if waiting_since is not None:
                wait_us = max(0, event_time_us - waiting_since)
            profile["total_wait_us"] += wait_us
            profile["max_wait_us"] = max(
                profile["max_wait_us"],
                wait_us,
            )
            profile["waiting_since_us"] = None
            profile["selected_since_us"] = event_time_us
            profile["selection_count"] += 1

        for p_node_id in active_p_nodes - new_selected:
            profile = self.p_node_profiles[p_node_id]
            if profile["waiting_since_us"] is None:
                profile["waiting_since_us"] = event_time_us

        for p_node_id, profile in self.p_node_profiles.items():
            if p_node_id in active_p_nodes:
                continue
            waiting_since = profile["waiting_since_us"]
            if waiting_since is not None:
                wait_us = max(0, event_time_us - waiting_since)
                profile["total_wait_us"] += wait_us
                profile["max_wait_us"] = max(
                    profile["max_wait_us"],
                    wait_us,
                )
            profile["waiting_since_us"] = None
            profile["active_since_us"] = None

        new_tuple = tuple(selected_p_nodes)
        if new_tuple != self.selected_p_nodes:
            self.total_selection_changes += 1
        self.selected_p_nodes = new_tuple

    def _selected_set(self, event_time_us):
        """计算全局top-K GPU并更新选择统计。"""
        if self.persistent_cohort:
            selected = self._persistent_cohort_selection()
        else:
            ordered = self._ordered_active_p_nodes()
            selected = ordered[: self.selection_width]
        self._record_selection_transition(selected, event_time_us)
        return set(selected)

    def _desired_active_states(self, storage_target_id, selected_p_nodes):
        """对一块SSD生成活跃Queue的CIR/PIR/weight目标。"""
        target_demands = self.demands[storage_target_id]
        selected_queue_ids = sorted(
            (
                queue_id
                for queue_id, demand in target_demands.items()
                if demand["p_node_id"] in selected_p_nodes
            ),
            key=lambda queue_id: (
                self._profile_priority_key(
                    target_demands[queue_id]["p_node_id"]
                ),
                queue_id,
            ),
        )
        selected_count = len(selected_queue_ids)
        shares = {}
        if selected_count:
            base_share, remainder = divmod(
                self.capacity[storage_target_id],
                selected_count,
            )
            shares = {
                queue_id: base_share + (index < remainder)
                for index, queue_id in enumerate(selected_queue_ids)
            }

        desired = {}
        for queue_id, demand in target_demands.items():
            if queue_id in shares:
                selected_pir = (
                    shares[queue_id]
                    if self.finite_selected_pir
                    else None
                )
                desired[queue_id] = (
                    shares[queue_id],
                    selected_pir,
                    1,
                )
                demand["assigned_cir"] = shares[queue_id]
            else:
                desired[queue_id] = (0, 0, 0)
                demand["assigned_cir"] = 0
        return desired

    def recalculate(self, storage_target_id, event_time_us=None):
        """功能：按全局top-K结果重算一块SSD的Queue控制。

        选中Queue平分该SSD容量且PIR不封顶；未选活跃Queue
        设为CIR=0、PIR=0、weight=0。已完成Queue恢复默认
        ``(CIR=0, PIR=uncapped, weight=1)``。
        """
        if storage_target_id not in self.demands:
            raise KeyError(f"unknown storage target {storage_target_id!r}")
        if event_time_us is None:
            event_time_us = self.current_time_us
        event_time_us = self._non_negative_time(event_time_us)
        self.current_time_us = max(self.current_time_us, event_time_us)
        selected_p_nodes = self._selected_set(event_time_us)
        desired_active = self._desired_active_states(
            storage_target_id,
            selected_p_nodes,
        )
        old_states = self._programmed_queue_states[storage_target_id]
        desired_states = dict(desired_active)

        queue_rate_updates = {}
        queue_pir_updates = {}
        queue_weight_updates = {}
        for queue_id in sorted(old_states.keys() | desired_states.keys()):
            old_state = old_states.get(
                queue_id,
                self._DEFAULT_QUEUE_STATE,
            )
            new_state = desired_states.get(
                queue_id,
                self._DEFAULT_QUEUE_STATE,
            )
            if old_state[:2] != new_state[:2]:
                # CIR/PIR属于同一条硬件命令；任一字段变化时
                # 都输出完整新值，避免用None混淆“不更新”和uncapped。
                queue_rate_updates[queue_id] = new_state[0]
                queue_pir_updates[queue_id] = new_state[1]
            if old_state[2] != new_state[2]:
                queue_weight_updates[queue_id] = new_state[2]

        self._programmed_queue_states[storage_target_id] = {
            queue_id: state
            for queue_id, state in desired_states.items()
            if state != self._DEFAULT_QUEUE_STATE
        }
        self.queue_rates[storage_target_id] = {
            queue_id: state[0]
            for queue_id, state in desired_active.items()
        }
        self.queue_pirs[storage_target_id] = {
            queue_id: state[1]
            for queue_id, state in desired_active.items()
        }
        self.queue_weights[storage_target_id] = {
            queue_id: state[2]
            for queue_id, state in desired_active.items()
        }
        assigned_total = sum(self.queue_rates[storage_target_id].values())
        self.peak_assigned_cir[storage_target_id] = max(
            self.peak_assigned_cir[storage_target_id],
            assigned_total,
        )

        for queue_id, demand in self.demands[storage_target_id].items():
            if queue_id not in desired_active:
                continue
            is_selected = desired_active[queue_id][2] > 0
            if is_selected and demand["first_selected_time_us"] is None:
                demand["first_selected_time_us"] = event_time_us
                demand["selection_count"] += 1
                wait_us = max(
                    0,
                    event_time_us - demand["arrival_time_us"],
                )
                self.total_queue_wait_us += wait_us
                self.max_queue_wait_us = max(
                    self.max_queue_wait_us,
                    wait_us,
                )
                self.selected_queue_count += 1

        return {
            "queue_rates": queue_rate_updates,
            "queue_pirs": queue_pir_updates,
            "queue_weights": queue_weight_updates,
            # Group权重由硬件静态配置，coflow策略永不写它。
            "group_weights": None,
        }

    def release_empty_demands(
        self,
        storage_target_id,
        queue_depths,
        event_time_us=None,
    ):
        """功能：只依据Queue depth=0释放路径Demand。"""
        if event_time_us is None:
            event_time_us = self.current_time_us
        event_time_us = self._non_negative_time(event_time_us)
        target_demands = self.demands[storage_target_id]
        empty_queue_ids = [
            queue_id
            for queue_id in target_demands
            if queue_depths[queue_id] == 0
        ]
        if not empty_queue_ids:
            # QoS会在每轮depth变化后唤醒DPU，但绝大多数
            # 唤醒没有任何Queue归零。直接返回可避免每个IO
            # dispatch loop都重新排序128个GPU并扫描其他SSD。
            return {
                "queue_rates": {},
                "queue_pirs": {},
                "queue_weights": {},
                "group_weights": None,
                "coordinates_changed": False,
            }

        old_selected_p_nodes = self.selected_p_nodes
        if self.persistent_cohort and not self.cohort_locked:
            # 第一次Queue排空是DPU可见的“已发生实际服务”
            # 边界。在删除Demand前固化最后一次全局排名结果，
            # 使同时到达但后登记的更优候选能正常替换早到者。
            if not self.selected_p_nodes:
                self._selected_set(event_time_us)
            self.cohort_members = tuple(self.selected_p_nodes)
            self.cohort_started = True
            self.cohort_locked = True
        for queue_id in empty_queue_ids:
            demand = target_demands.pop(queue_id)
            p_node_id = demand["p_node_id"]
            profile = self.p_node_profiles[p_node_id]
            profile["completed_demand_count"] += 1
            self.completed_demand_count[storage_target_id] += 1

            group_key = (p_node_id, demand["demand_group_id"])
            group_paths = self.active_group_paths[group_key]
            group_paths.discard((storage_target_id, queue_id))
            if not group_paths:
                del self.active_group_paths[group_key]
                self.completed_coflow_count += 1
                profile["completed_coflow_count"] += 1

        self.current_time_us = max(self.current_time_us, event_time_us)
        updates = self.recalculate(storage_target_id, event_time_us)
        updates["coordinates_changed"] = (
            self.selected_p_nodes != old_selected_p_nodes
        )
        return updates

    def _profile_statistics(self, event_time_us):
        """生成包含当前未结束区间的p_node统计快照。"""
        result = {}
        active_p_nodes = self._active_p_nodes()
        selected_p_nodes = set(self.selected_p_nodes)
        for p_node_id, profile in self.p_node_profiles.items():
            total_wait_us = profile["total_wait_us"]
            max_wait_us = profile["max_wait_us"]
            waiting_since = profile["waiting_since_us"]
            if p_node_id in active_p_nodes and waiting_since is not None:
                current_wait_us = max(0, event_time_us - waiting_since)
                total_wait_us += current_wait_us
                max_wait_us = max(max_wait_us, current_wait_us)

            total_selected_us = profile["total_selected_us"]
            selected_since = profile["selected_since_us"]
            if p_node_id in selected_p_nodes and selected_since is not None:
                total_selected_us += max(0, event_time_us - selected_since)

            result[p_node_id] = {
                "priority_batch_total_bytes": profile[
                    "priority_batch_total_bytes"
                ],
                "priority_service_window_us": profile[
                    "priority_service_window_us"
                ],
                "priority_demand_bytes_per_second": profile[
                    "priority_demand_bytes_per_second"
                ],
                "first_arrival_time_us": profile["first_arrival_time_us"],
                "arrival_order": profile["arrival_order"],
                "activation_count": profile["activation_count"],
                "selection_count": profile["selection_count"],
                "completed_demand_count": profile[
                    "completed_demand_count"
                ],
                "completed_coflow_count": profile[
                    "completed_coflow_count"
                ],
                "total_wait_us": total_wait_us,
                "max_wait_us": max_wait_us,
                "total_selected_us": total_selected_us,
                "is_active": p_node_id in active_p_nodes,
                "is_selected": p_node_id in selected_p_nodes,
            }
        return result

    def statistics(self):
        """返回等待、选择、完成与容量上界统计。"""
        p_node_statistics = self._profile_statistics(self.current_time_us)
        result = {
            "strategy": self.strategy_name,
            "ordering": self.ordering,
            "selection_width": self.selection_width,
            "active_demand_count": sum(
                len(target_demands)
                for target_demands in self.demands.values()
            ),
            "active_p_node_count": len(self._active_p_nodes()),
            "selected_p_node_ids": list(self.selected_p_nodes),
            "completed_demand_count_by_storage_target": dict(
                self.completed_demand_count
            ),
            "completed_coflow_count": self.completed_coflow_count,
            "peak_assigned_cir_bytes_per_second": dict(
                self.peak_assigned_cir
            ),
            "selection_change_count": self.total_selection_changes,
            "selected_queue_count": self.selected_queue_count,
            "total_queue_wait_us": self.total_queue_wait_us,
            "max_queue_wait_us": self.max_queue_wait_us,
            "p_node_statistics": p_node_statistics,
        }
        if self.persistent_cohort:
            result.update({
                "persistent_cohort": True,
                "expected_coflow_count": self.expected_coflow_count,
                "cohort_started": self.cohort_started,
                "cohort_locked": self.cohort_locked,
                "cohort_member_ids": list(self.cohort_members),
            })
        if self.finite_selected_pir:
            result["finite_selected_pir"] = True
        if self.min_window_threshold_us is not None:
            result["min_window_threshold_us"] = (
                self.min_window_threshold_us
            )
        return result


class UtilityEDFController:
    """以推理价值密度启动stage0，并用EDF保护预取截止期。

    控制器只使用请求到达时元数据、当前时钟与完整
    Queue depth。不读SSD completion、inflight或后端内部状态。
    每次全局只准入一个p_node；选中Queue默认使用
    ``CIR=SSD capacity, PIR=uncapped``，也可选有限PIR；
    其余Queue为``0/0/0``。
    """

    strategy_name = "utility_edf"
    coordinates_storage_targets = True
    uses_queue_depths_for_recalculate = True
    uses_event_time_for_release = True
    requires_tick_aligned_control = True
    strict_control_update_grid = True
    _DEFAULT_QUEUE_STATE = (0, None, 1)
    _PARKED_QUEUE_STATE = (0, 0, 0)

    def __init__(
        self,
        capacity_bytes_per_second_by_storage_target,
        score_mode="integer",
        deadline_allowance_us=1_000,
        compute_layer_count=4,
        finite_selected_pir=False,
        restore_after_final_layer=True,
    ):
        """功能：创建单p_node、work-conserving的Utility+EDF控制器。"""
        if score_mode not in ("integer", "power"):
            raise ValueError("score_mode must be 'integer' or 'power'")
        if (
            not isinstance(deadline_allowance_us, int)
            or isinstance(deadline_allowance_us, bool)
            or deadline_allowance_us <= 0
        ):
            raise ValueError(
                "deadline_allowance_us must be a positive integer"
            )
        if (
            not isinstance(compute_layer_count, int)
            or isinstance(compute_layer_count, bool)
            or compute_layer_count <= 0
        ):
            raise ValueError("compute_layer_count must be a positive integer")
        if not isinstance(finite_selected_pir, bool):
            raise TypeError("finite_selected_pir must be bool")
        if not isinstance(restore_after_final_layer, bool):
            raise TypeError("restore_after_final_layer must be bool")

        self.capacity = dict(
            capacity_bytes_per_second_by_storage_target
        )
        if not self.capacity:
            raise ValueError("at least one storage target is required")
        for target, capacity in self.capacity.items():
            if (
                not isinstance(capacity, int)
                or isinstance(capacity, bool)
                or capacity <= 0
            ):
                raise ValueError(
                    f"capacity for {target!r} must be a positive integer"
                )

        self.score_mode = score_mode
        self.deadline_allowance_us = deadline_allowance_us
        self.compute_layer_count = compute_layer_count
        self.finite_selected_pir = finite_selected_pir
        self.restore_after_final_layer = restore_after_final_layer
        self.demands = {target: {} for target in self.capacity}
        self.queue_depths = {target: {} for target in self.capacity}
        self.queue_rates = {target: {} for target in self.capacity}
        self.queue_pirs = {target: {} for target in self.capacity}
        self.queue_weights = {target: {} for target in self.capacity}
        self._programmed_queue_states = {
            target: {} for target in self.capacity
        }
        # Utility模式在t=0把DPU可管理的Queue全部park。
        # managed_queue_ids保留这个硬件初态；一条路径
        # 完成整次四读组推理前，即使当前没有Demand，
        # 也不会恢复默认uncapped EXCESS。
        self.managed_queue_ids = {
            target: set() for target in self.capacity
        }
        self.queue_owner_by_storage_target = {
            target: {} for target in self.capacity
        }
        self.restored_queue_paths = set()
        self.active_group_paths = {}
        self.seen_p_node_ids = set()
        # 当前推理计数每次Layer 0重置；累计计数不重置。
        self.current_inference_arrival_time_us_by_p_node = {}
        self.current_inference_completed_layer_count_by_p_node = {}
        self.completed_layer_count_by_p_node = {}
        # 保留旧字段名，一个coflow对应一层读组。
        self.completed_coflow_count_by_p_node = (
            self.completed_layer_count_by_p_node
        )
        self.completed_demand_count = {
            target: 0 for target in self.capacity
        }
        self.peak_assigned_cir = {
            target: 0 for target in self.capacity
        }
        self.arrival_sequence = 0
        self.current_time_us = 0
        self.selected_p_node_id = None
        self.owner_locked = False
        self.decision_count = 0
        self.initial_decision_count = 0
        self.prefetch_decision_count = 0
        self.feasibility_conflict_count = 0
        self.selection_change_count = 0
        self.decision_history = []

    def prepark_all_queues(
        self,
        queue_ids_by_storage_target,
        queue_owners_by_storage_target=None,
    ):
        """在首个IO到达前把DPU可管理的Queue全部park。

        初始状态是 ``(CIR=0, PIR=0, weight=0)``，使首个
        非80 us边界到达的Demand在下一控制tick打开前
        不能借用Baseline的uncapped EXCESS。可选owner映射来自
        初始Queue binding，用于第四个读组后恢复该p_node
        的全部已绑定路径。
        """
        if any(self.demands[target] for target in self.demands):
            raise RuntimeError(
                "Utility Queue parking must be initialized before demands"
            )
        owners_by_target = (
            {}
            if queue_owners_by_storage_target is None
            else queue_owners_by_storage_target
        )
        updates_by_target = {}
        for storage_target_id, queue_ids in (
            queue_ids_by_storage_target.items()
        ):
            if storage_target_id not in self.capacity:
                raise KeyError(
                    f"unknown storage target {storage_target_id!r}"
                )
            managed = self.managed_queue_ids[storage_target_id]
            managed.update(queue_ids)
            owner_map = self.queue_owner_by_storage_target[
                storage_target_id
            ]
            for queue_id, p_node_id in owners_by_target.get(
                storage_target_id,
                {},
            ).items():
                if queue_id not in managed:
                    raise ValueError(
                        f"owner supplied for unmanaged queue {queue_id!r}"
                    )
                previous_owner = owner_map.get(queue_id)
                if (
                    previous_owner is not None
                    and previous_owner != p_node_id
                ):
                    raise ValueError(
                        f"queue {queue_id!r} has conflicting owners"
                    )
                owner_map[queue_id] = p_node_id

            desired = {
                queue_id: self._PARKED_QUEUE_STATE
                for queue_id in sorted(managed)
            }
            updates_by_target[storage_target_id] = self._state_updates(
                storage_target_id,
                desired,
            )
        return updates_by_target

    # 保留语义更显式的别名，便于旧的实验接线过渡；
    # 生产gateway使用prepark_all_queues硬件契约名。
    initialize_parked_queues = prepark_all_queues

    @staticmethod
    def _non_negative_integer(value, field_name, fallback=0):
        """将正整数或非负实数四舍五入为整数微秒。"""
        if value is None:
            return fallback
        if isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return int(round(value))

    @staticmethod
    def _positive_integer(value, field_name):
        """校验并返回正整数路径元数据。"""
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{field_name} must be a positive integer")
        return value

    def _repark_p_node_paths(self, p_node_id):
        """让一张GPU的全部固定路径重新受Utility Gate管理。"""
        for storage_target_id, owner_map in (
            self.queue_owner_by_storage_target.items()
        ):
            for queue_id, owner_id in owner_map.items():
                if owner_id == p_node_id:
                    self.restored_queue_paths.discard((
                        storage_target_id,
                        queue_id,
                    ))

    def _reset_inference_policy_state(self, p_node_id):
        """为子类保留新推理状态重置钩子。"""

    def _track_inference(
        self,
        p_node_id,
        inference_arrival_time_us,
        is_layer_zero,
    ):
        """用推理到达时刻识别同一GPU的连续推理。"""
        arrivals = self.current_inference_arrival_time_us_by_p_node
        previous_arrival = arrivals.get(p_node_id)
        if previous_arrival is None:
            arrivals[p_node_id] = inference_arrival_time_us
            self.current_inference_completed_layer_count_by_p_node[
                p_node_id
            ] = 0
            if is_layer_zero:
                self._repark_p_node_paths(p_node_id)
                self._reset_inference_policy_state(p_node_id)
            return
        if previous_arrival == inference_arrival_time_us:
            return
        if not is_layer_zero:
            raise ValueError("a new inference must start with Layer 0")
        completed = self.current_inference_completed_layer_count_by_p_node.get(
            p_node_id,
            0,
        )
        if completed < self.compute_layer_count:
            raise ValueError(
                "a new inference cannot start before the previous one completes"
            )

        arrivals[p_node_id] = inference_arrival_time_us
        self.current_inference_completed_layer_count_by_p_node[p_node_id] = 0
        self._repark_p_node_paths(p_node_id)
        self._reset_inference_policy_state(p_node_id)

    def _record_completed_layer(self, p_node_id):
        """同时更新当前推理和累计完成层数。"""
        current_counts = (
            self.current_inference_completed_layer_count_by_p_node
        )
        current_counts[p_node_id] = current_counts.get(p_node_id, 0) + 1
        total_counts = self.completed_layer_count_by_p_node
        total_counts[p_node_id] = total_counts.get(p_node_id, 0) + 1
        return current_counts[p_node_id]

    def register_demand(
        self,
        storage_target_id,
        queue_id,
        requested_cir_bytes_per_second,
        arrival_time_us,
        p_node_id=None,
        demand_group_id=None,
        batch_total_bytes=None,
        path_bytes=None,
        path_request_count=None,
        block_size_bytes=None,
        service_window_us=None,
        deadline_us=None,
        compute_layer_index=None,
        prefetch_layer_index=None,
        inference_arrival_time_us=None,
    ):
        """登记一条已完整入队的层路径Demand。"""
        if storage_target_id not in self.demands:
            raise KeyError(f"unknown storage target {storage_target_id!r}")
        if queue_id in self.demands[storage_target_id]:
            raise ValueError(
                f"queue {queue_id!r} on {storage_target_id!r} "
                "already has an active demand"
            )
        if p_node_id is None:
            p_node_id = f"{storage_target_id}:{queue_id}"
        if demand_group_id is None:
            demand_group_id = (p_node_id, arrival_time_us)

        arrival_time_us = self._non_negative_integer(
            arrival_time_us,
            "arrival_time_us",
        )
        inference_arrival_time_us = self._non_negative_integer(
            inference_arrival_time_us,
            "inference_arrival_time_us",
            fallback=arrival_time_us,
        )

        self.managed_queue_ids[storage_target_id].add(queue_id)
        owner_map = self.queue_owner_by_storage_target[storage_target_id]
        previous_owner = owner_map.get(queue_id)
        if previous_owner is not None and previous_owner != p_node_id:
            raise ValueError(
                f"queue {queue_id!r} on {storage_target_id!r} "
                "cannot change p_node owner"
            )
        owner_map[queue_id] = p_node_id
        self._track_inference(
            p_node_id,
            inference_arrival_time_us,
            compute_layer_index is None,
        )
        if (
            self.current_inference_completed_layer_count_by_p_node.get(
                p_node_id,
                0,
            )
            < self.compute_layer_count
        ):
            self.restored_queue_paths.discard(
                (storage_target_id, queue_id)
            )

        service_window_us = self._non_negative_integer(
            service_window_us,
            "service_window_us",
        )
        if deadline_us is None:
            deadline_us = arrival_time_us + service_window_us
        deadline_us = self._non_negative_integer(
            deadline_us,
            "deadline_us",
        )
        path_bytes = self._positive_integer(path_bytes, "path_bytes")
        path_request_count = self._positive_integer(
            path_request_count,
            "path_request_count",
        )
        if block_size_bytes is not None:
            block_size_bytes = self._positive_integer(
                block_size_bytes,
                "block_size_bytes",
            )

        self.arrival_sequence += 1
        demand = {
            "p_node_id": p_node_id,
            "demand_group_id": demand_group_id,
            "arrival_time_us": arrival_time_us,
            "arrival_sequence": self.arrival_sequence,
            "inference_arrival_time_us": inference_arrival_time_us,
            "service_window_us": service_window_us,
            "deadline_us": deadline_us,
            "compute_layer_index": compute_layer_index,
            "prefetch_layer_index": prefetch_layer_index,
            "path_bytes": path_bytes,
            "path_request_count": path_request_count,
            "block_size_bytes": block_size_bytes,
        }
        self.demands[storage_target_id][queue_id] = demand
        self.seen_p_node_ids.add(p_node_id)
        self.queue_depths[storage_target_id].setdefault(
            queue_id,
            path_request_count,
        )
        group_key = (p_node_id, demand_group_id)
        self.active_group_paths.setdefault(group_key, set()).add(
            (storage_target_id, queue_id)
        )

    def _remaining_bytes(self, demand, queue_depth):
        """从完整depth和提交时路径元数据求当前剩余Byte。"""
        if queue_depth <= 0:
            return 0
        block_size = demand["block_size_bytes"]
        if block_size is not None:
            return min(
                demand["path_bytes"],
                queue_depth * block_size,
            )
        return (
            demand["path_bytes"] * queue_depth
            + demand["path_request_count"] - 1
        ) // demand["path_request_count"]

    def _remaining_service_us(
        self,
        storage_target_id,
        demand,
        queue_depth,
    ):
        """按整盘容量向上取整剩余服务微秒。"""
        remaining_bytes = self._remaining_bytes(demand, queue_depth)
        if remaining_bytes == 0:
            return 0
        capacity = self.capacity[storage_target_id]
        return (
            remaining_bytes * 1_000_000 + capacity - 1
        ) // capacity

    def _active_candidates(self):
        """将多SSD路径收敛为全局p_node demand group候选。"""
        groups = {}
        for storage_target_id, target_demands in self.demands.items():
            target_depths = self.queue_depths[storage_target_id]
            for queue_id, demand in target_demands.items():
                queue_depth = target_depths.get(
                    queue_id,
                    demand["path_request_count"],
                )
                remaining_service_us = self._remaining_service_us(
                    storage_target_id,
                    demand,
                    queue_depth,
                )
                if remaining_service_us == 0:
                    continue
                group_key = (
                    demand["p_node_id"],
                    demand["demand_group_id"],
                )
                candidate = groups.setdefault(group_key, {
                    "p_node_id": demand["p_node_id"],
                    "demand_group_id": demand["demand_group_id"],
                    "is_initial": demand["compute_layer_index"] is None,
                    "arrival_time_us": demand["arrival_time_us"],
                    "arrival_sequence": demand["arrival_sequence"],
                    "inference_arrival_time_us": demand[
                        "inference_arrival_time_us"
                    ],
                    "service_window_us": demand["service_window_us"],
                    "deadline_us": demand["deadline_us"],
                    "remaining_service_us": 0,
                    "queue_paths": [],
                })
                if candidate["is_initial"] != (
                    demand["compute_layer_index"] is None
                ):
                    raise ValueError(
                        "all paths in one demand group must share stage type"
                    )
                candidate["arrival_time_us"] = min(
                    candidate["arrival_time_us"],
                    demand["arrival_time_us"],
                )
                candidate["arrival_sequence"] = min(
                    candidate["arrival_sequence"],
                    demand["arrival_sequence"],
                )
                candidate["deadline_us"] = min(
                    candidate["deadline_us"],
                    demand["deadline_us"],
                )
                # 多盘路径并行，coflow剩余时间由最慢路径决定。
                candidate["remaining_service_us"] = max(
                    candidate["remaining_service_us"],
                    remaining_service_us,
                )
                candidate["queue_paths"].append(
                    (storage_target_id, queue_id)
                )
        return list(groups.values())

    def _initial_score_f(self, candidate, event_time_us):
        """计算价值密度分母中的整数F。"""
        elapsed_us = max(
            0,
            event_time_us - candidate["inference_arrival_time_us"],
        )
        return (
            elapsed_us
            + candidate["remaining_service_us"]
            + self.compute_layer_count * candidate["service_window_us"]
        )

    def _compare_initial_candidates(self, event_time_us, left, right):
        """用大整数交叉相乘比较两个价值密度。"""
        left_c = left["service_window_us"]
        right_c = right["service_window_us"]
        left_b = left["remaining_service_us"]
        right_b = right["remaining_service_us"]
        left_f = self._initial_score_f(left, event_time_us)
        right_f = self._initial_score_f(right, event_time_us)
        if self.score_mode == "integer":
            left_cross = left_c * right_f * right_b ** 2
            right_cross = right_c * left_f * left_b ** 2
        else:
            left_cross = (
                left_c ** 10 * right_f ** 13 * right_b ** 12
            )
            right_cross = (
                right_c ** 10 * left_f ** 13 * left_b ** 12
            )
        if left_cross != right_cross:
            return -1 if left_cross > right_cross else 1
        left_tie = (
            left_b,
            left["arrival_time_us"],
            str(left["p_node_id"]),
        )
        right_tie = (
            right_b,
            right["arrival_time_us"],
            str(right["p_node_id"]),
        )
        return (left_tie > right_tie) - (left_tie < right_tie)

    def _ordered_initials(self, candidates, event_time_us):
        """按精确价值密度降序返回stage0候选。"""
        initials = [
            candidate for candidate in candidates
            if candidate["is_initial"]
        ]
        return sorted(
            initials,
            key=cmp_to_key(lambda left, right: (
                self._compare_initial_candidates(
                    event_time_us,
                    left,
                    right,
                )
            )),
        )

    def _prefetch_key(self, candidate):
        """返回预取EDF与稳定平局键。"""
        completed_count = (
            self.current_inference_completed_layer_count_by_p_node.get(
                candidate["p_node_id"],
                0,
            )
        )
        return (
            candidate["deadline_us"],
            candidate["remaining_service_us"],
            -completed_count,
            candidate["arrival_sequence"],
            str(candidate["p_node_id"]),
        )

    def _owner_has_started(self, p_node_id):
        """完整depth小于提交数表示当前owner已开始下发。"""
        return any(
            self.queue_depths[storage_target_id].get(
                queue_id,
                demand["path_request_count"],
            )
            < demand["path_request_count"]
            for storage_target_id, target_demands in self.demands.items()
            for queue_id, demand in target_demands.items()
            if demand["p_node_id"] == p_node_id
        )

    def _choose_candidate(self, event_time_us):
        """执行一次stage0插入可行性与EDF决策。"""
        candidates = self._active_candidates()
        current_owner_candidates = [
            candidate
            for candidate in candidates
            if candidate["p_node_id"] == self.selected_p_node_id
        ]
        if current_owner_candidates and (
            self.owner_locked
            or self._owner_has_started(self.selected_p_node_id)
        ):
            self.owner_locked = True
            return current_owner_candidates[0], "owner_locked", False
        if not current_owner_candidates:
            self.owner_locked = False
        initials = self._ordered_initials(candidates, event_time_us)
        prefetches = sorted(
            (
                candidate for candidate in candidates
                if not candidate["is_initial"]
            ),
            key=self._prefetch_key,
        )
        if not initials and not prefetches:
            return None, "idle", False
        if not prefetches:
            return initials[0], "initial_only", False
        if not initials:
            return prefetches[0], "prefetch_only", False

        initial = initials[0]
        cumulative_service_us = initial["remaining_service_us"]
        conflict = False
        for prefetch in prefetches:
            cumulative_service_us += prefetch["remaining_service_us"]
            if (
                event_time_us + cumulative_service_us
                > prefetch["deadline_us"]
                + self.deadline_allowance_us
            ):
                conflict = True
                break
        if conflict:
            return prefetches[0], "edf_conflict", True
        return initial, "initial_feasible", False

    def _record_decision(
        self,
        event_time_us,
        candidate,
        reason,
        conflict,
    ):
        """保存每次控制决策的可审计摘要。"""
        selected_p_node_id = (
            None if candidate is None else candidate["p_node_id"]
        )
        if selected_p_node_id != self.selected_p_node_id:
            self.selection_change_count += 1
            self.owner_locked = False
        self.selected_p_node_id = selected_p_node_id
        self.decision_count += 1
        if candidate is not None:
            if candidate["is_initial"]:
                self.initial_decision_count += 1
            else:
                self.prefetch_decision_count += 1
        if conflict:
            self.feasibility_conflict_count += 1
        self.decision_history.append({
            "event_time_us": event_time_us,
            "selected_p_node_id": selected_p_node_id,
            "selected_demand_group_id": (
                None if candidate is None
                else candidate["demand_group_id"]
            ),
            "selected_stage": (
                None if candidate is None
                else "initial" if candidate["is_initial"] else "prefetch"
            ),
            "remaining_service_us": (
                None if candidate is None
                else candidate["remaining_service_us"]
            ),
            "reason": reason,
        })

    def _desired_states(self, storage_target_id, selected_p_node_id):
        """生成一块SSD上的有限PIR单p_node Gate状态。"""
        # 已经在t=0 park的Queue保持park，直到所属p_node
        # 完成全部compute_layer_count个读组。这里故意包含
        # 当前没有Demand的层间Queue，防止它被恢复成
        # Baseline的uncapped EXCESS后，下一层在控制边界前泄漏。
        desired = {
            queue_id: self._PARKED_QUEUE_STATE
            for queue_id in self.managed_queue_ids[storage_target_id]
            if (
                storage_target_id,
                queue_id,
            ) not in self.restored_queue_paths
        }
        target_depths = self.queue_depths[storage_target_id]
        capacity = self.capacity[storage_target_id]
        for queue_id, demand in self.demands[storage_target_id].items():
            queue_depth = target_depths.get(
                queue_id,
                demand["path_request_count"],
            )
            if queue_depth <= 0:
                continue
            if demand["p_node_id"] == selected_p_node_id:
                desired[queue_id] = (
                    capacity,
                    capacity if self.finite_selected_pir else None,
                    1,
                )
            else:
                desired[queue_id] = self._PARKED_QUEUE_STATE
        return desired

    def _state_updates(self, storage_target_id, desired):
        """只输出与已编程Queue状态不同的字段。"""
        programmed = self._programmed_queue_states[storage_target_id]
        queue_rates = {}
        queue_pirs = {}
        queue_weights = {}
        for queue_id in sorted(programmed.keys() | desired.keys()):
            old_state = programmed.get(
                queue_id,
                self._DEFAULT_QUEUE_STATE,
            )
            new_state = desired.get(
                queue_id,
                self._DEFAULT_QUEUE_STATE,
            )
            if old_state[0] != new_state[0]:
                queue_rates[queue_id] = new_state[0]
            if old_state[1] != new_state[1]:
                queue_pirs[queue_id] = new_state[1]
            if old_state[2] != new_state[2]:
                queue_weights[queue_id] = new_state[2]
            if new_state == self._DEFAULT_QUEUE_STATE:
                programmed.pop(queue_id, None)
            else:
                programmed[queue_id] = new_state

        self.queue_rates[storage_target_id] = {
            queue_id: state[0] for queue_id, state in desired.items()
        }
        self.queue_pirs[storage_target_id] = {
            queue_id: state[1] for queue_id, state in desired.items()
        }
        self.queue_weights[storage_target_id] = {
            queue_id: state[2] for queue_id, state in desired.items()
        }
        assigned_total = sum(
            state[0] for state in desired.values()
        )
        self.peak_assigned_cir[storage_target_id] = max(
            self.peak_assigned_cir[storage_target_id],
            assigned_total,
        )
        return {
            "queue_rates": queue_rates,
            "queue_pirs": queue_pirs,
            "queue_weights": queue_weights,
            "group_weights": None,
        }

    def recalculate(
        self,
        storage_target_id,
        event_time_us=None,
        queue_depths=None,
    ):
        """使用当前完整depth重算全局唯一准入p_node。"""
        if storage_target_id not in self.demands:
            raise KeyError(f"unknown storage target {storage_target_id!r}")
        event_time_us = self._non_negative_integer(
            event_time_us,
            "event_time_us",
            fallback=self.current_time_us,
        )
        self.current_time_us = max(self.current_time_us, event_time_us)
        if queue_depths is not None:
            self.queue_depths[storage_target_id] = {
                queue_id: self._non_negative_integer(
                    depth,
                    "queue_depth",
                )
                for queue_id, depth in queue_depths.items()
            }

        old_selected_p_node_id = self.selected_p_node_id
        candidate, reason, conflict = self._choose_candidate(
            event_time_us
        )
        self._record_decision(
            event_time_us,
            candidate,
            reason,
            conflict,
        )
        desired = self._desired_states(
            storage_target_id,
            self.selected_p_node_id,
        )
        updates = self._state_updates(storage_target_id, desired)
        updates["coordinates_changed"] = (
            old_selected_p_node_id != self.selected_p_node_id
        )
        return updates

    def release_empty_demands(
        self,
        storage_target_id,
        queue_depths,
        event_time_us=None,
    ):
        """仅根据depth=0释放Demand，再执行一次决策。"""
        self.queue_depths[storage_target_id] = dict(queue_depths)
        target_demands = self.demands[storage_target_id]
        empty_queue_ids = [
            queue_id
            for queue_id in target_demands
            if queue_depths.get(queue_id, 0) == 0
        ]
        for queue_id in empty_queue_ids:
            demand = target_demands[queue_id]
            if (
                demand["p_node_id"] == self.selected_p_node_id
                and queue_depths.get(
                    queue_id,
                    demand["path_request_count"],
                ) < demand["path_request_count"]
            ):
                # Queue depth是DPU可见的“已开始下发”边界。
                # 在删除这条已排空路径前持久化owner lock，
                # 否则多SSD中其他路径的中间depth仍可能是旧快照，
                # 全局重算会误以为owner尚未服务而抢占。
                # _choose_candidate只在该owner已无任何候选路径
                # 时解锁，因此最后一条路径消失后仍能正常轮转。
                self.owner_locked = True
            demand = target_demands.pop(queue_id)
            self.completed_demand_count[storage_target_id] += 1
            group_key = (
                demand["p_node_id"],
                demand["demand_group_id"],
            )
            paths = self.active_group_paths[group_key]
            paths.discard((storage_target_id, queue_id))
            if not paths:
                del self.active_group_paths[group_key]
                p_node_id = demand["p_node_id"]
                current_completed = self._record_completed_layer(p_node_id)
                if (
                    self.restore_after_final_layer
                    and current_completed >= self.compute_layer_count
                ):
                    # 第四个读组的最后一条SSD路径排空后，
                    # 才让这个p_node的所有固定Queue恢复
                    # Baseline默认状态。其他p_node/Queue仍保持park。
                    for target_id, target_owners in (
                        self.queue_owner_by_storage_target.items()
                    ):
                        for owned_queue_id, owner_id in (
                            target_owners.items()
                        ):
                            if owner_id == p_node_id:
                                self.restored_queue_paths.add((
                                    target_id,
                                    owned_queue_id,
                                ))
        return self.recalculate(
            storage_target_id,
            event_time_us=event_time_us,
            queue_depths=queue_depths,
        )

    def statistics(self):
        """返回策略参数、决策、完成与容量统计。"""
        return {
            "strategy": self.strategy_name,
            "score_mode": self.score_mode,
            "deadline_allowance_us": self.deadline_allowance_us,
            "compute_layer_count": self.compute_layer_count,
            "active_demand_count": sum(
                len(target_demands)
                for target_demands in self.demands.values()
            ),
            "active_p_node_count": len({
                demand["p_node_id"]
                for target_demands in self.demands.values()
                for demand in target_demands.values()
            }),
            "active_coflow_count": len(self.active_group_paths),
            "selected_p_node_id": self.selected_p_node_id,
            "completed_demand_count_by_storage_target": dict(
                self.completed_demand_count
            ),
            "completed_coflow_count": sum(
                self.completed_coflow_count_by_p_node.values()
            ),
            "completed_coflow_count_by_p_node": dict(
                self.completed_coflow_count_by_p_node
            ),
            "completed_layer_count": sum(
                self.completed_layer_count_by_p_node.values()
            ),
            "completed_layer_count_by_p_node": dict(
                self.completed_layer_count_by_p_node
            ),
            "current_inference_completed_layer_count_by_p_node": dict(
                self.current_inference_completed_layer_count_by_p_node
            ),
            "current_inference_arrival_time_us_by_p_node": dict(
                self.current_inference_arrival_time_us_by_p_node
            ),
            "p_node_statistics": {
                p_node_id: {
                    "completed_coflow_count": (
                        self.completed_coflow_count_by_p_node.get(
                            p_node_id,
                            0,
                        )
                    ),
                    "completed_layer_count": (
                        self.completed_layer_count_by_p_node.get(
                            p_node_id,
                            0,
                        )
                    ),
                    "current_inference_completed_layer_count": (
                        self.current_inference_completed_layer_count_by_p_node.get(
                            p_node_id,
                            0,
                        )
                    ),
                    "current_inference_arrival_time_us": (
                        self.current_inference_arrival_time_us_by_p_node.get(
                            p_node_id
                        )
                    ),
                }
                for p_node_id in sorted(
                    self.seen_p_node_ids,
                    key=str,
                )
            },
            "peak_assigned_cir_bytes_per_second": dict(
                self.peak_assigned_cir
            ),
            "decision_count": self.decision_count,
            "initial_decision_count": self.initial_decision_count,
            "prefetch_decision_count": self.prefetch_decision_count,
            "feasibility_conflict_count": (
                self.feasibility_conflict_count
            ),
            "selection_change_count": self.selection_change_count,
            "owner_locked": self.owner_locked,
            "decision_history": list(self.decision_history),
            "finite_selected_pir": self.finite_selected_pir,
            "restore_after_final_layer": self.restore_after_final_layer,
            "managed_queue_count": sum(
                len(queue_ids)
                for queue_ids in self.managed_queue_ids.values()
            ),
            "restored_queue_count": len(self.restored_queue_paths),
        }


class UtilityEDFAblationController(UtilityEDFController):
    """将跨SSD协调、Stage0 Utility和Prefetch EDF拆成三个正交因子。

    ``c1`` 保留UtilityEDFController的全局p_node owner和跨SSD sticky
    lock；``c0`` 则让每块SSD维护独立owner，只由本地Queue depth
    锁定。``u1`` 使用原Stage0价值密度，``u0`` 使用FCFS。
    ``e1`` 使用原EDF prefix feasibility，``e0`` 在有Stage0时总是
    先选Stage0，仅在没有Stage0时按FCFS选Prefetch。

    所有组合继承相同的t=0 prepark、80 us严格控制网格和
    ``selected PIR=uncapped`` Gate。
    """

    strategy_name = "ablation"

    def __init__(
        self,
        capacity_bytes_per_second_by_storage_target,
        coordination_enabled,
        utility_enabled,
        edf_enabled,
        compute_layer_count=4,
    ):
        """创建一个固定L=750 us、整数Utility的三因子消融控制器。"""
        factors = {
            "coordination_enabled": coordination_enabled,
            "utility_enabled": utility_enabled,
            "edf_enabled": edf_enabled,
        }
        for field_name, value in factors.items():
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be bool")

        super().__init__(
            capacity_bytes_per_second_by_storage_target,
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=compute_layer_count,
            finite_selected_pir=False,
        )
        self.coordination_enabled = coordination_enabled
        self.utility_enabled = utility_enabled
        self.edf_enabled = edf_enabled
        self.coordinates_storage_targets = coordination_enabled
        self.strategy_name = (
            f"ablation_c{int(coordination_enabled)}"
            f"_u{int(utility_enabled)}"
            f"_e{int(edf_enabled)}"
        )

        # c0不能复用基类单个selected/locked字段：两块SSD可以在
        # 同一时刻服务不同p_node，且某一块的depth变化不得锁住另一块。
        self.selected_p_node_id_by_storage_target = {
            target: None for target in self.capacity
        }
        self.owner_locked_by_storage_target = {
            target: False for target in self.capacity
        }
        self.completed_path_group_count_by_storage_target = {
            target: {} for target in self.capacity
        }

    def _reset_inference_policy_state(self, p_node_id):
        """c0的每盘完成层数也按推理重置。"""
        for counts in (
            self.completed_path_group_count_by_storage_target.values()
        ):
            counts[p_node_id] = 0

    @staticmethod
    def _fcfs_key(candidate):
        """用Demand到达与登记序号生成稳定FCFS键。"""
        return (
            candidate["arrival_time_us"],
            candidate["arrival_sequence"],
            str(candidate["p_node_id"]),
            str(candidate["demand_group_id"]),
        )

    def _ordered_initials_for_policy(self, candidates, event_time_us):
        """按u因子排序Stage0。"""
        if self.utility_enabled:
            return self._ordered_initials(candidates, event_time_us)
        return sorted(
            (
                candidate for candidate in candidates
                if candidate["is_initial"]
            ),
            key=self._fcfs_key,
        )

    def _local_prefetch_key(self, storage_target_id, candidate):
        """返回c0的本地EDF键，不读取其他SSD的完成进度。"""
        completed_count = (
            self.completed_path_group_count_by_storage_target[
                storage_target_id
            ].get(candidate["p_node_id"], 0)
        )
        return (
            candidate["deadline_us"],
            candidate["remaining_service_us"],
            -completed_count,
            candidate["arrival_sequence"],
            str(candidate["p_node_id"]),
        )

    def _ordered_prefetches_for_policy(
        self,
        candidates,
        storage_target_id=None,
    ):
        """按e因子排序Prefetch。"""
        prefetches = [
            candidate for candidate in candidates
            if not candidate["is_initial"]
        ]
        if not self.edf_enabled:
            return sorted(prefetches, key=self._fcfs_key)
        if storage_target_id is None:
            return sorted(prefetches, key=self._prefetch_key)
        return sorted(
            prefetches,
            key=lambda candidate: self._local_prefetch_key(
                storage_target_id,
                candidate,
            ),
        )

    def _choose_candidate_for_policy(
        self,
        event_time_us,
        candidates,
        selected_p_node_id,
        owner_locked,
        owner_has_started,
        storage_target_id=None,
    ):
        """在一个全局或本地候选集上执行u/e组合。"""
        current_owner_candidates = [
            candidate
            for candidate in candidates
            if candidate["p_node_id"] == selected_p_node_id
        ]
        if current_owner_candidates and (
            owner_locked or owner_has_started
        ):
            return (
                current_owner_candidates[0],
                "owner_locked",
                False,
                True,
            )
        if not current_owner_candidates:
            owner_locked = False

        initials = self._ordered_initials_for_policy(
            candidates,
            event_time_us,
        )
        prefetches = self._ordered_prefetches_for_policy(
            candidates,
            storage_target_id=storage_target_id,
        )
        if not initials and not prefetches:
            return None, "idle", False, owner_locked
        if not prefetches:
            return initials[0], "initial_only", False, owner_locked
        if not initials:
            return prefetches[0], "prefetch_only", False, owner_locked

        # e0的定义是Stage0绝对优先；Prefetch FCFS只在没有
        # Stage0时才会影响选择。
        if not self.edf_enabled:
            return initials[0], "initial_priority", False, owner_locked

        initial = initials[0]
        cumulative_service_us = initial["remaining_service_us"]
        for prefetch in prefetches:
            cumulative_service_us += prefetch["remaining_service_us"]
            if (
                event_time_us + cumulative_service_us
                > prefetch["deadline_us"] + self.deadline_allowance_us
            ):
                return (
                    prefetches[0],
                    "edf_conflict",
                    True,
                    owner_locked,
                )
        return initial, "initial_feasible", False, owner_locked

    def _choose_candidate(self, event_time_us):
        """在c1下使用全局候选；c1u1e1原样调用基类实现。"""
        if self.coordination_enabled and self.utility_enabled and self.edf_enabled:
            # 这条原路径是“逐决策等价”的核心保证：不复制原
            # Utility/EDF comparator、tie-break或owner-lock状态机。
            return super()._choose_candidate(event_time_us)

        candidates = self._active_candidates()
        candidate, reason, conflict, owner_locked = (
            self._choose_candidate_for_policy(
                event_time_us,
                candidates,
                self.selected_p_node_id,
                self.owner_locked,
                self._owner_has_started(self.selected_p_node_id),
            )
        )
        self.owner_locked = owner_locked
        return candidate, reason, conflict

    def _active_candidates_for_storage_target(self, storage_target_id):
        """把一块SSD上的路径聚合成c0本地候选。"""
        groups = {}
        target_depths = self.queue_depths[storage_target_id]
        for queue_id, demand in self.demands[storage_target_id].items():
            queue_depth = target_depths.get(
                queue_id,
                demand["path_request_count"],
            )
            remaining_service_us = self._remaining_service_us(
                storage_target_id,
                demand,
                queue_depth,
            )
            if remaining_service_us == 0:
                continue
            group_key = (
                demand["p_node_id"],
                demand["demand_group_id"],
            )
            candidate = groups.setdefault(group_key, {
                "p_node_id": demand["p_node_id"],
                "demand_group_id": demand["demand_group_id"],
                "is_initial": demand["compute_layer_index"] is None,
                "arrival_time_us": demand["arrival_time_us"],
                "arrival_sequence": demand["arrival_sequence"],
                "inference_arrival_time_us": demand[
                    "inference_arrival_time_us"
                ],
                "service_window_us": demand["service_window_us"],
                "deadline_us": demand["deadline_us"],
                "remaining_service_us": 0,
                "queue_paths": [],
            })
            if candidate["is_initial"] != (
                demand["compute_layer_index"] is None
            ):
                raise ValueError(
                    "all paths in one demand group must share stage type"
                )
            candidate["arrival_time_us"] = min(
                candidate["arrival_time_us"],
                demand["arrival_time_us"],
            )
            candidate["arrival_sequence"] = min(
                candidate["arrival_sequence"],
                demand["arrival_sequence"],
            )
            candidate["deadline_us"] = min(
                candidate["deadline_us"],
                demand["deadline_us"],
            )
            candidate["remaining_service_us"] = max(
                candidate["remaining_service_us"],
                remaining_service_us,
            )
            candidate["queue_paths"].append(
                (storage_target_id, queue_id)
            )
        return list(groups.values())

    def _local_owner_has_started(self, storage_target_id, p_node_id):
        """只用一块SSD的depth判断c0 owner是否已开始。"""
        return any(
            self.queue_depths[storage_target_id].get(
                queue_id,
                demand["path_request_count"],
            ) < demand["path_request_count"]
            for queue_id, demand in self.demands[
                storage_target_id
            ].items()
            if demand["p_node_id"] == p_node_id
        )

    def _choose_local_candidate(self, storage_target_id, event_time_us):
        """在c0下只对一块SSD做决策。"""
        selected_p_node_id = (
            self.selected_p_node_id_by_storage_target[storage_target_id]
        )
        return self._choose_candidate_for_policy(
            event_time_us,
            self._active_candidates_for_storage_target(storage_target_id),
            selected_p_node_id,
            self.owner_locked_by_storage_target[storage_target_id],
            self._local_owner_has_started(
                storage_target_id,
                selected_p_node_id,
            ),
            storage_target_id=storage_target_id,
        )

    def _record_local_decision(
        self,
        storage_target_id,
        event_time_us,
        candidate,
        reason,
        conflict,
    ):
        """记录c0决策；单SSD时保持与基类相同的history形状。"""
        selected_p_node_id = (
            None if candidate is None else candidate["p_node_id"]
        )
        previous_p_node_id = (
            self.selected_p_node_id_by_storage_target[storage_target_id]
        )
        if selected_p_node_id != previous_p_node_id:
            self.selection_change_count += 1
            self.owner_locked_by_storage_target[storage_target_id] = False
        self.selected_p_node_id_by_storage_target[
            storage_target_id
        ] = selected_p_node_id
        self.decision_count += 1
        if candidate is not None:
            if candidate["is_initial"]:
                self.initial_decision_count += 1
            else:
                self.prefetch_decision_count += 1
        if conflict:
            self.feasibility_conflict_count += 1

        decision = {
            "event_time_us": event_time_us,
            "selected_p_node_id": selected_p_node_id,
            "selected_demand_group_id": (
                None if candidate is None
                else candidate["demand_group_id"]
            ),
            "selected_stage": (
                None if candidate is None
                else "initial" if candidate["is_initial"] else "prefetch"
            ),
            "remaining_service_us": (
                None if candidate is None
                else candidate["remaining_service_us"]
            ),
            "reason": reason,
        }
        if len(self.capacity) > 1:
            decision["storage_target_id"] = storage_target_id
        self.decision_history.append(decision)

        # 单SSD时同步基类兼容字段，使c0和c1的直接观测结果
        # 一致；多SSD时单值无法表达独立owner，因此保持None。
        if len(self.capacity) == 1:
            self.selected_p_node_id = selected_p_node_id
            self.owner_locked = self.owner_locked_by_storage_target[
                storage_target_id
            ]
        else:
            self.selected_p_node_id = None
            self.owner_locked = any(
                self.owner_locked_by_storage_target.values()
            )

    def recalculate(
        self,
        storage_target_id,
        event_time_us=None,
        queue_depths=None,
    ):
        """在c1下保留基类全局重算，c0下只重算目标SSD。"""
        if self.coordination_enabled:
            return super().recalculate(
                storage_target_id,
                event_time_us=event_time_us,
                queue_depths=queue_depths,
            )
        if storage_target_id not in self.demands:
            raise KeyError(f"unknown storage target {storage_target_id!r}")
        event_time_us = self._non_negative_integer(
            event_time_us,
            "event_time_us",
            fallback=self.current_time_us,
        )
        self.current_time_us = max(self.current_time_us, event_time_us)
        if queue_depths is not None:
            self.queue_depths[storage_target_id] = {
                queue_id: self._non_negative_integer(
                    depth,
                    "queue_depth",
                )
                for queue_id, depth in queue_depths.items()
            }

        old_selected_p_node_id = (
            self.selected_p_node_id_by_storage_target[storage_target_id]
        )
        candidate, reason, conflict, owner_locked = (
            self._choose_local_candidate(
                storage_target_id,
                event_time_us,
            )
        )
        self.owner_locked_by_storage_target[
            storage_target_id
        ] = owner_locked
        self._record_local_decision(
            storage_target_id,
            event_time_us,
            candidate,
            reason,
            conflict,
        )
        selected_p_node_id = (
            self.selected_p_node_id_by_storage_target[storage_target_id]
        )
        desired = self._desired_states(
            storage_target_id,
            selected_p_node_id,
        )
        updates = self._state_updates(storage_target_id, desired)
        updates["coordinates_changed"] = (
            old_selected_p_node_id != selected_p_node_id
        )
        return updates

    def release_empty_demands(
        self,
        storage_target_id,
        queue_depths,
        event_time_us=None,
    ):
        """释放空Demand；c0的lock、读组计数与Queue恢复均为本地。"""
        if self.coordination_enabled:
            return super().release_empty_demands(
                storage_target_id,
                queue_depths,
                event_time_us=event_time_us,
            )

        self.queue_depths[storage_target_id] = dict(queue_depths)
        target_demands = self.demands[storage_target_id]
        empty_queue_ids = [
            queue_id
            for queue_id in target_demands
            if queue_depths.get(queue_id, 0) == 0
        ]
        for queue_id in empty_queue_ids:
            demand = target_demands[queue_id]
            if (
                demand["p_node_id"]
                == self.selected_p_node_id_by_storage_target[
                    storage_target_id
                ]
                and queue_depths.get(
                    queue_id,
                    demand["path_request_count"],
                ) < demand["path_request_count"]
            ):
                self.owner_locked_by_storage_target[
                    storage_target_id
                ] = True

            demand = target_demands.pop(queue_id)
            self.completed_demand_count[storage_target_id] += 1
            p_node_id = demand["p_node_id"]
            group_key = (p_node_id, demand["demand_group_id"])
            paths = self.active_group_paths[group_key]
            paths.discard((storage_target_id, queue_id))

            # c0的第G个本地读组结束后就恢复该SSD上这个p_node
            # 的固定Queue，不等其他SSD。多Queue属于同一本地组时
            # 只在最后一条本地路径排空后计数。
            if not any(
                target_id == storage_target_id
                for target_id, _ in paths
            ):
                local_counts = (
                    self.completed_path_group_count_by_storage_target[
                        storage_target_id
                    ]
                )
                local_counts[p_node_id] = (
                    local_counts.get(p_node_id, 0) + 1
                )
                if local_counts[p_node_id] >= self.compute_layer_count:
                    for owned_queue_id, owner_id in (
                        self.queue_owner_by_storage_target[
                            storage_target_id
                        ].items()
                    ):
                        if owner_id == p_node_id:
                            self.restored_queue_paths.add((
                                storage_target_id,
                                owned_queue_id,
                            ))

            # 全局coflow计数仍保留为统计，但不参与c0的owner
            # lock、EDF tie-break或Queue恢复。
            if not paths:
                del self.active_group_paths[group_key]
                self._record_completed_layer(p_node_id)

        return self.recalculate(
            storage_target_id,
            event_time_us=event_time_us,
            queue_depths=queue_depths,
        )

    def statistics(self):
        """在UtilityEDF统计上增加消融因子与c0本地状态。"""
        statistics = super().statistics()
        statistics.update({
            "coordination_enabled": self.coordination_enabled,
            "utility_enabled": self.utility_enabled,
            "edf_enabled": self.edf_enabled,
        })
        if not self.coordination_enabled:
            statistics.update({
                "selected_p_node_id_by_storage_target": dict(
                    self.selected_p_node_id_by_storage_target
                ),
                "owner_locked_by_storage_target": dict(
                    self.owner_locked_by_storage_target
                ),
                "completed_path_group_count_by_storage_target": {
                    target: dict(counts)
                    for target, counts in (
                        self.completed_path_group_count_by_storage_target.items()
                    )
                },
            })
        return statistics
