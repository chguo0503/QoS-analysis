"""组合组间WRR和组内WRR的顶层QoS调度器。"""

from .weighted_round_robin import (
    SmoothWeightedRoundRobinScheduler,
    WeightedRoundRobinScheduler,
)


class HierarchicalScheduler:
    """先选择可用组，再从获胜组中选择可用业务队列。"""

    def __init__(self, queue_layout, scheduler_config):
        """按队列布局和权重配置创建两级WRR调度器。"""
        self.queue_layout = queue_layout
        # 第一级在8个组之间仲裁，第二级为每个组各建一个队列仲裁器。
        self.group_scheduler = SmoothWeightedRoundRobinScheduler(
            queue_layout.group_order,
            scheduler_config["group_weight_bitmap"],
        )
        self.queue_schedulers = {
            group_id: WeightedRoundRobinScheduler(
                queue_layout.group_queues[group_id],
                scheduler_config["queue_weight_bitmaps"][group_id],
            )
            for group_id in queue_layout.group_order
        }

    def select_next_queue(self, is_eligible):
        """返回组间WRR和组内WRR共同选中的queue_id。"""

        def group_is_eligible(group_id):
            """返回一个组内是否至少有一个队列可以下发。"""
            # 将队列级判定包装成组级回调，通用WRR无需理解组内结构。
            return any(
                is_eligible(queue_id)
                for queue_id in self.queue_layout.group_queues[group_id]
            )

        # 先选出“组内确实有请求可发”的组，再只在该组内选最终队列。
        group_id = self.group_scheduler.select_next(group_is_eligible)
        if group_id is None:
            return None
        return self.queue_schedulers[group_id].select_next(is_eligible)

    def set_group_weights(self, weights):
        """功能：动态设置八个Group的WRR权重。

        目的：把DPU计算的每Group活跃带宽需求传入第一级调度器；
        Group只负责分配调度机会，不创建或修改Group CIR/PIR。

        输入：``group_id -> 非负整数权重`` 映射。
        输出：无；更新组间WRR的运行时权重。
        """
        self.group_scheduler.set_weights(weights)
