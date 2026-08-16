"""定义DPU把每条GPU→SSD路径绑定到独占QoS Queue的可扩展策略。"""


class QueueBindingStrategy:
    """Queue绑定策略的最小公共接口。"""

    def prepare_bindings(self, p_node_ids, queue_ids_by_storage_target):
        """功能：在仿真启动前生成全部路径的Queue绑定。

        目的：让需要全局协调的策略在IO到达前完成分配，
        避免运行时遍历顺序改变绑定结果。

        输入：
            p_node_ids: 按GPU索引排列的P节点ID列表。
            queue_ids_by_storage_target: ``SSD ID -> Queue ID列表`` 映射。

        输出：
            None: 基类不保存任何绑定。
        """


class BalancedExclusiveBindingStrategy(QueueBindingStrategy):
    """按GPU索引将每条GPU→SSD路径固定绑定到互斥Queue。"""

    strategy_name = "balanced_exclusive"

    def __init__(self):
        """功能：创建空的路径绑定表。

        目的：在同一SSD命名空间中让一条Queue只属于一张GPU，
        从而使 ``(SSD, queue_id)`` 能唯一代表当前层Demand。

        输入：无。
        输出：无；初始化空的 ``bindings`` 映射。
        """
        self.bindings = {}

    def prepare_bindings(self, p_node_ids, queue_ids_by_storage_target):
        """功能：按固定公式为每个 ``(GPU, SSD)`` 选择Queue。

        目的：对64张GPU构造每SSD恰64条不同活跃Queue，
        并8个Group各含8条。公式为
        ``32 * (gpu_index % 8) + gpu_index // 8``。不同SSD是独立
        namespace，因此可以复用相同queue_id。

        输入：
            p_node_ids: 按GPU0、GPU1…顺序排列的P节点ID。
            queue_ids_by_storage_target: 每块SSD上按q000…q255排列的Queue。

        输出：
            None: 填充 ``(p_node_id, storage_target_id) -> queue_id``。
        """
        for storage_target_id, queue_ids in queue_ids_by_storage_target.items():
            if len(p_node_ids) > len(queue_ids):
                raise ValueError(
                    "balanced_exclusive requires at least one Queue per GPU "
                    f"on {storage_target_id}"
                )
            for gpu_index, p_node_id in enumerate(p_node_ids):
                queue_index = (
                    32 * (gpu_index % 8)
                    + gpu_index // 8
                )
                self.bindings[(p_node_id, storage_target_id)] = queue_ids[
                    queue_index
                ]

    def select_queue(self, request, queue_ids):
        """功能：查询一个IO所属GPU→SSD路径的固定Queue。

        目的：让同一GPU访问同一SSD的所有层和Block始终
        复用实验开始前分配的独占Queue。

        输入：
            request: 包含 ``basic.p_node_id`` 和
                ``basic.storage_target_id`` 的DPU请求。
            queue_ids: 统一策略接口保留的Queue列表，本函数不再重选。

        输出：
            str: 已经固定的Queue ID。
        """
        basic = request["basic"]
        return self.bindings[
            (basic["p_node_id"], basic["storage_target_id"])
        ]


class OneGroupPerGpuBindingStrategy(BalancedExclusiveBindingStrategy):
    """每张GPU独占一段连续Queue，并固定使用段首Queue。"""

    strategy_name = "one_group_per_gpu"
    queue_count = 256

    def prepare_bindings(self, p_node_ids, queue_ids_by_storage_target):
        """将256条Queue等分成N组，GPU i使用第i组首Queue。"""
        gpu_count = len(p_node_ids)
        if gpu_count == 0:
            raise ValueError("one_group_per_gpu requires at least one GPU")
        if len(set(p_node_ids)) != gpu_count:
            raise ValueError("one_group_per_gpu requires unique p_node IDs")
        if self.queue_count % gpu_count != 0:
            raise ValueError(
                "one_group_per_gpu requires the GPU count to divide 256"
            )

        queues_per_group = self.queue_count // gpu_count
        for storage_target_id, queue_ids in queue_ids_by_storage_target.items():
            if len(queue_ids) != self.queue_count:
                raise ValueError(
                    "one_group_per_gpu requires exactly 256 Queues on "
                    f"{storage_target_id}"
                )
            if len(set(queue_ids)) != self.queue_count:
                raise ValueError(
                    "one_group_per_gpu requires unique Queue IDs on "
                    f"{storage_target_id}"
                )
            for gpu_index, p_node_id in enumerate(p_node_ids):
                self.bindings[(p_node_id, storage_target_id)] = queue_ids[
                    gpu_index * queues_per_group
                ]


QUEUE_BINDING_STRATEGIES = {
    BalancedExclusiveBindingStrategy.strategy_name: (
        BalancedExclusiveBindingStrategy
    ),
    OneGroupPerGpuBindingStrategy.strategy_name: (
        OneGroupPerGpuBindingStrategy
    ),
}


def build_queue_binding_strategy(
    strategy_name,
    p_node_ids=(),
    queue_ids_by_storage_target=None,
):
    """功能：根据配置名称创建并准备Queue绑定策略。

    目的：集中保存策略注册与构造逻辑，未来新增绑定算法时
    不需改动DPU请求网关。

    输入：
        strategy_name: ``QUEUE_BINDING_STRATEGIES`` 中的策略名称。
        p_node_ids: 实验中的全部P节点ID。
        queue_ids_by_storage_target: 每块SSD独立的Queue ID列表。

    输出：
        QueueBindingStrategy: 已完成预绑定的新策略实例。
    """
    strategy = QUEUE_BINDING_STRATEGIES[strategy_name]()
    strategy.prepare_bindings(
        list(p_node_ids),
        {} if queue_ids_by_storage_target is None else queue_ids_by_storage_target,
    )
    return strategy
