"""DPU内部的先到先服务Queue CIR分配器。"""


class DemandAwareFCFSCIRController:
    """按每块SSD的Demand到达顺序分配有限CIR容量。"""

    strategy_name = "demand_aware_fcfs_cir"

    def __init__(self, capacity_bytes_per_second_by_storage_target):
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
        self.capacity = dict(capacity_bytes_per_second_by_storage_target)
        self.demands = {target: {} for target in self.capacity}
        self.queue_rates = {target: {} for target in self.capacity}
        self.arrival_sequence = {target: 0 for target in self.capacity}
        self.completed_demand_count = {target: 0 for target in self.capacity}
        self.peak_assigned_cir = {target: 0 for target in self.capacity}

    def register_demand(
        self,
        storage_target_id,
        queue_id,
        requested_cir_bytes_per_second,
        arrival_time_us,
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
        self.arrival_sequence[storage_target_id] += 1
        self.demands[storage_target_id][queue_id] = {
            "requested_cir": requested_cir_bytes_per_second,
            "assigned_cir": 0,
            "arrival_time_us": arrival_time_us,
            "arrival_order": self.arrival_sequence[storage_target_id],
        }

    def recalculate(self, storage_target_id):
        """功能：按Demand到达顺序重新分配一块SSD的CIR。

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
        ordered_demands = sorted(
            self.demands[storage_target_id].items(),
            key=lambda item: (
                item[1]["arrival_time_us"],
                item[1]["arrival_order"],
            ),
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

    def release_empty_demands(self, storage_target_id, queue_depths):
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
            for queue_id in demands
            if queue_depths[queue_id] == 0
        ]
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
            "active_demand_count": sum(
                len(demands) for demands in self.demands.values()
            ),
            "completed_demand_count_by_storage_target": dict(
                self.completed_demand_count
            ),
            "peak_assigned_cir_bytes_per_second": dict(
                self.peak_assigned_cir
            ),
        }
