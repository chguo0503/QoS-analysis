"""DPU根据活跃带宽需求设置Queue速率和Group调度权重。"""

from collections import deque


class DemandAwareRateController:
    """Queue CIR/PIR约束速率，Group WRR权重分配调度机会。"""

    strategy_name = "demand_aware"

    def __init__(
        self,
        capacity_bytes_per_second_by_storage_target,
        queue_to_group_by_storage_target,
    ):
        """功能：为每块SSD建立独立的需求和带宽状态。

        目的：容量足够时完整满足先到需求；容量不足时不缩放
        已配置CIR/PIR，后到需求保留在自己的QoS Queue中等待。

        输入：每SSD的整数Byte/s容量，以及每SSD的Queue到Group映射。
        输出：无；初始化等待、活跃、Queue速率和Group权重。
        """
        self.capacity = dict(capacity_bytes_per_second_by_storage_target)
        self.queue_to_group = {
            target: dict(mapping)
            for target, mapping in queue_to_group_by_storage_target.items()
        }
        self.demands = {}
        self.request_to_demand = {}
        self.waiting = {target: deque() for target in self.capacity}
        self.active = {target: {} for target in self.capacity}
        self.reserved = {target: 0 for target in self.capacity}
        self.queue_rates = {target: {} for target in self.capacity}
        self.group_weights = {
            target: {
                group_id: 0
                for group_id in dict.fromkeys(self.queue_to_group[target].values())
            }
            for target in self.capacity
        }
        self.peak_reserved = dict(self.reserved)
        self.peak_waiting = {target: 0 for target in self.capacity}

    @staticmethod
    def _demand_key(request):
        """功能：生成一份SSD带宽需求的唯一标识。

        目的：同一GPU、同一层、同一SSD上的所有Block只登记一次
        ``aggregate_required_bytes_per_second``，不按IO重复累加。

        输入：DPU展平后的IO请求。
        输出：``(P节点, demand_group, SSD)`` 元组。
        """
        return (
            request["p_node_id"],
            request["demand_group_id"],
            request["storage_target_id"],
        )

    def register(self, request):
        """功能：登记一个已经进入QoS Queue的IO。

        目的：为新需求保存一份聚合速率，同时按Queue记录尚未
        离开QoS的Byte数，供后续速率分配和需求释放使用。

        输入：带Queue、需求ID和整数Byte/s诉求的请求。
        输出：无；更新需求、IO到需求的映射和准入标记。
        """
        key = self._demand_key(request)
        target = request["storage_target_id"]
        demand = self.demands.get(key)
        if demand is None:
            demand = {
                "rate": min(
                    request["aggregate_required_bytes_per_second"],
                    self.capacity[target],
                ),
                "requests": [],
                "remaining_bytes_by_queue": {},
            }
            self.demands[key] = demand
            self.waiting[target].append(key)

        queue_id = request["queue_id"]
        demand["requests"].append(request)
        demand["remaining_bytes_by_queue"][queue_id] = (
            demand["remaining_bytes_by_queue"].get(queue_id, 0)
            + request["size_bytes"]
        )
        request["qos_admitted"] = key in self.active[target]
        self.request_to_demand[request["request_id"]] = key

    def _admit_waiting_demands(self, storage_target_id):
        """功能：按到达顺序准入SSD剩余容量能完整满足的需求。

        目的：不使用浮点比例缩放；队首需求放不下时，后续
        需求不插队，对应硬件中简单的整数加减和比较。

        输入：要重算容量的SSD ID。
        输出：无；原地更新等待队列、活跃集合和预留带宽。
        """
        waiting = self.waiting[storage_target_id]
        free = self.capacity[storage_target_id] - self.reserved[storage_target_id]
        while waiting:
            key = waiting[0]
            demand = self.demands[key]
            if demand["rate"] > free:
                break
            waiting.popleft()
            self.active[storage_target_id][key] = None
            self.reserved[storage_target_id] += demand["rate"]
            free -= demand["rate"]
            for request in demand["requests"]:
                request["qos_admitted"] = True

    def _active_queue_rates(self, storage_target_id):
        """功能：把每份活跃需求的速率分配到它当前使用的Queue。

        目的：互斥固定绑定时一份需求直接对应一个Queue；
        其他绑定策略将IO分散到多Queue时，则按剩余Byte数用整数分配。

        输入：要计算的SSD ID。
        输出：``queue_id -> Byte/s`` 聚合速率映射。
        """
        queue_rates = {}
        for key in self.active[storage_target_id]:
            demand = self.demands[key]
            remaining = demand["remaining_bytes_by_queue"]
            total_bytes = sum(remaining.values())
            queue_ids = sorted(remaining)
            unassigned_rate = demand["rate"]
            for index, queue_id in enumerate(queue_ids):
                if index == len(queue_ids) - 1:
                    rate = unassigned_rate
                else:
                    rate = demand["rate"] * remaining[queue_id] // total_bytes
                    unassigned_rate -= rate
                queue_rates[queue_id] = queue_rates.get(queue_id, 0) + rate
        return queue_rates

    def update(self, storage_target_id):
        """功能：准入可满足需求并重算Queue速率与Group权重。

        目的：Queue CIR/PIR承担速率约束；Group权重等于组内活跃
        Queue速率之和，只影响WRR机会，完全不涉及Group CIR/PIR。

        输入：要更新的SSD ID。
        输出：发生变化的Queue速率和可选的整张Group权重。
        """
        self._admit_waiting_demands(storage_target_id)
        new_queue_rates = self._active_queue_rates(storage_target_id)
        old_queue_rates = self.queue_rates[storage_target_id]
        queue_updates = {
            queue_id: new_queue_rates.get(queue_id, 0)
            for queue_id in sorted(old_queue_rates.keys() | new_queue_rates.keys())
            if old_queue_rates.get(queue_id, 0)
            != new_queue_rates.get(queue_id, 0)
        }

        new_group_weights = {
            group_id: 0
            for group_id in self.group_weights[storage_target_id]
        }
        for queue_id, rate in new_queue_rates.items():
            group_id = self.queue_to_group[storage_target_id][queue_id]
            new_group_weights[group_id] += rate
        group_update = (
            new_group_weights
            if new_group_weights != self.group_weights[storage_target_id]
            else None
        )

        self.queue_rates[storage_target_id] = new_queue_rates
        self.group_weights[storage_target_id] = new_group_weights
        self.peak_reserved[storage_target_id] = max(
            self.peak_reserved[storage_target_id],
            self.reserved[storage_target_id],
        )
        self.peak_waiting[storage_target_id] = max(
            self.peak_waiting[storage_target_id],
            len(self.waiting[storage_target_id]),
        )
        return {
            "queue_rates": queue_updates,
            "group_weights": group_update,
        }

    def dispatched(self, requests):
        """功能：根据QoS下发的IO更新每份需求的剩余量。

        目的：DPU只使用可见的Queue occupancy判断需求是否释放，
        不依赖SSD完成回调或NAND内部状态；Queue排空后可准入后续需求。

        输入：同一块SSD在本轮成功离开QoS Queue的IO列表。
        输出：需要写回QoS的Queue速率和Group权重变化。
        """
        storage_target_id = requests[0]["storage_target_id"]
        for request in requests:
            key = self.request_to_demand.pop(request["request_id"])
            demand = self.demands[key]
            queue_id = request["queue_id"]
            remaining = demand["remaining_bytes_by_queue"]
            remaining[queue_id] -= request["size_bytes"]
            if remaining[queue_id] == 0:
                del remaining[queue_id]
            if not remaining:
                self.reserved[storage_target_id] -= demand["rate"]
                del self.active[storage_target_id][key]
                del self.demands[key]
        return self.update(storage_target_id)

    def statistics(self):
        """功能：返回需求感知控制器的实验统计。

        目的：验证最终无遗留需求，并观察每SSD的峰值预留和等待。

        输入：无。
        输出：策略名、需求数、预留带宽和Group权重字典。
        """
        return {
            "strategy": self.strategy_name,
            "active_demand_count": sum(len(items) for items in self.active.values()),
            "waiting_demand_count": sum(len(items) for items in self.waiting.values()),
            "reserved_bytes_per_second": dict(self.reserved),
            "peak_reserved_bytes_per_second": dict(self.peak_reserved),
            "peak_waiting_demand_count": dict(self.peak_waiting),
            "group_weights_bytes_per_second": {
                target: dict(weights)
                for target, weights in self.group_weights.items()
            },
        }
