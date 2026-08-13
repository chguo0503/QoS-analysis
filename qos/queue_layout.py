"""生成QoS的固定输入队列和分组关系。"""


class QueueLayout:
    """根据紧凑配置生成全部queue_id、group_id及其对应关系。"""

    def __init__(self, layout_config):
        """把YAML中的队列数量和名称规则展开成完整布局。"""
        self.queue_count = layout_config["queue_count"]
        self.group_count = layout_config["group_count"]
        self.queues_per_group = layout_config["queues_per_group"]
        queue_prefix = layout_config["queue_id_prefix"]
        queue_width = layout_config["queue_id_width"]
        group_prefix = layout_config["group_id_prefix"]

        # ``:0{queue_width}d`` 是动态宽度的整数格式：例如宽度3会生成q000、q001。
        self.queue_order = [
            f"{queue_prefix}{queue_index:0{queue_width}d}"
            for queue_index in range(self.queue_count)
        ]
        self.group_order = [
            f"{group_prefix}{group_index}"
            for group_index in range(self.group_count)
        ]
        self.group_queues = {}

        # DPU动态计算Group WRR权重时需要O(1)查询Queue所在组；
        # 这是固定布局连线，不包含Group速率或令牌状态。
        self.queue_to_group = {}

        # 每组用一段连续切片取队列。
        for group_index, group_id in enumerate(self.group_order):
            start = group_index * self.queues_per_group
            end = start + self.queues_per_group
            queue_ids = self.queue_order[start:end]
            self.group_queues[group_id] = queue_ids
            for queue_id in queue_ids:
                self.queue_to_group[queue_id] = group_id
