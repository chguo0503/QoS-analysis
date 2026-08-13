"""定义DPU把普通IO绑定到目标SSD之QoS Queue的可替换策略。"""

import hashlib


def _stable_queue_index(random_seed, identity_parts, queue_count):
    """功能：把请求身份稳定地映射成一个Queue下标。

    目的：提供不依赖Python进程哈希盐、共享随机数状态和事件执行顺序的
    可复现伪随机选择，使不同绑定策略能够在相同工作负载下公平比较。

    输入：
        random_seed: 仿真配置中的整数随机种子。
        identity_parts: 共同决定本次绑定身份的字符串序列。
        queue_count: 目标SSD对应QoS可用Queue的数量。

    输出：
        int: 范围为 ``[0, queue_count)`` 的稳定Queue下标。
    """
    # 使用明确的分隔符连接各字段，避免 ["ab", "c"] 与 ["a", "bc"]
    # 形成同一个输入字节串。blake2b在不同Python进程中输出完全一致。
    identity_text = "\x1f".join(str(part) for part in identity_parts)
    digest = hashlib.blake2b(
        f"{random_seed}\x1e{identity_text}".encode("utf-8"),
        digest_size=8,
    ).digest()
    random_value = int.from_bytes(digest, byteorder="big", signed=False)
    return random_value % queue_count


class QueueBindingStrategy:
    """保存Queue绑定策略共用的随机种子。"""

    def __init__(self, random_seed):
        """功能：保存策略运行所需的可复现随机种子。

        目的：让策略实例拥有独立配置，避免KV Placement和DPU绑定共用随机状态。

        输入：
            random_seed: 用于生成稳定伪随机Queue下标的整数种子。

        输出：
            None: 只初始化策略状态。
        """
        self.random_seed = random_seed

    def prepare_bindings(self, p_node_ids, queue_ids_by_storage_target):
        """功能：在实验启动前预生成可选的Queue绑定。

        目的：为需要全局唯一Queue的策略提供统一初始化入口；
        不需要预绑定的策略保持无状态。

        输入：全部P节点ID和每块SSD的Queue ID列表。
        输出：无；基类默认不做任何处理。
        """


class RandomUniqueStickyBindingStrategy(QueueBindingStrategy):
    """实现“实验前随机、GPU间互斥、运行中固定”的Queue绑定。"""

    strategy_name = "random_unique_sticky"

    def __init__(self, random_seed):
        """功能：创建空的互斥绑定表。

        目的：让Baseline和需求感知策略能使用同一组随机Queue，
        避免两个GPU共用一个Queue干扰速率与Group权重对比。

        输入：可复现的整数随机种子。
        输出：无；初始化空的 ``(P节点, SSD) -> Queue`` 表。
        """
        super().__init__(random_seed)
        self.bindings = {}

    def prepare_bindings(self, p_node_ids, queue_ids_by_storage_target):
        """功能：为每块SSD将不重复Queue随机分配给全部P节点。

        目的：在0 us第一批IO到达以前完成绑定，使运行时只需
        查表，同时保证同一SSD上一个Queue只属于一个GPU。

        输入：P节点ID列表以及 ``SSD -> Queue列表`` 映射。
        输出：无；填充固定绑定表。
        """
        for storage_target_id, queue_ids in queue_ids_by_storage_target.items():
            # 为每个Queue生成稳定伪随机排名，排名后的前N个Queue
            # 与P节点一一配对。这等价于可复现的随机不放回抽样。
            shuffled_queues = sorted(
                queue_ids,
                key=lambda queue_id: _stable_queue_index(
                    self.random_seed,
                    (storage_target_id, queue_id),
                    1 << 63,
                ),
            )
            if len(p_node_ids) > len(shuffled_queues):
                raise ValueError(
                    "random_unique_sticky requires at least one Queue per GPU "
                    f"on {storage_target_id}"
                )
            for p_node_id, queue_id in zip(p_node_ids, shuffled_queues):
                self.bindings[(p_node_id, storage_target_id)] = queue_id

    def select_queue(self, request, queue_ids):
        """功能：查询当前P节点到目标SSD的互斥Queue。

        目的：运行中所有IO稳定复用实验前选定的唯一Queue。

        输入：带P节点和SSD的DPU请求；queue_ids仅为统一策略接口参数。
        输出：实验启动前固定的Queue ID。
        """
        basic = request["basic"]
        return self.bindings[
            (basic["p_node_id"], basic["storage_target_id"])
        ]


class RandomStickyBindingStrategy(QueueBindingStrategy):
    """实现“首次随机、后续固定”的P节点与SSD绑定策略。"""

    strategy_name = "random_sticky"

    def __init__(self, random_seed):
        """功能：创建首次随机绑定策略及其绑定缓存。

        目的：保证同一个 ``(p_node_id, storage_target_id)`` 首次选定Queue后，
        该GPU后续访问同一SSD时始终复用这个Queue。

        输入：
            random_seed: 用于首次稳定伪随机选择的整数种子。

        输出：
            None: 初始化空的P节点/SSD绑定表。
        """
        super().__init__(random_seed)
        self.bindings = {}

    def select_queue(self, request, queue_ids):
        """功能：返回P节点访问目标SSD时固定使用的Queue。

        目的：实现传统的流亲和性，让同一GPU到同一SSD的全部IO保持在一个FIFO。

        输入：
            request: 包含 ``basic.p_node_id`` 和 ``basic.storage_target_id`` 的请求。
            queue_ids: 目标SSD QoS的全部合法Queue ID。
        输出：
            str: 首次稳定随机选中并在后续复用的Queue ID。
        """
        basic = request["basic"]
        binding_key = (
            basic["p_node_id"],
            basic["storage_target_id"],
        )
        if binding_key not in self.bindings:
            queue_index = _stable_queue_index(
                self.random_seed,
                binding_key,
                len(queue_ids),
            )
            self.bindings[binding_key] = queue_ids[queue_index]
        return self.bindings[binding_key]


class RandomPerRequestBindingStrategy(QueueBindingStrategy):
    """实现“每个IO独立随机一次”的Queue绑定策略。"""

    strategy_name = "random_per_request"

    def select_queue(self, request, queue_ids):
        """功能：根据每个请求自己的身份独立选择Queue。

        目的：把同一GPU发往同一SSD的IO分散到多个Queue，用作与固定绑定策略
        比较吞吐、TTFT、Queue占用数量和多GPU公平性的实验基线。

        输入：
            request: 包含唯一 ``request_id`` 和 ``storage_target_id`` 的请求。
            queue_ids: 目标SSD QoS的全部合法Queue ID。
        输出：
            str: 当前请求独立稳定随机选中的Queue ID。
        """
        basic = request["basic"]
        queue_index = _stable_queue_index(
            self.random_seed,
            (
                basic["request_id"],
                basic["storage_target_id"],
            ),
            len(queue_ids),
        )
        return queue_ids[queue_index]


QUEUE_BINDING_STRATEGIES = {
    RandomUniqueStickyBindingStrategy.strategy_name: (
        RandomUniqueStickyBindingStrategy
    ),
    RandomStickyBindingStrategy.strategy_name: RandomStickyBindingStrategy,
    RandomPerRequestBindingStrategy.strategy_name: RandomPerRequestBindingStrategy,
}


def build_queue_binding_strategy(
    strategy_name,
    random_seed,
    p_node_ids=(),
    queue_ids_by_storage_target=None,
):
    """功能：根据YAML名称创建对应的DPU Queue绑定策略。

    目的：把策略选择集中在一个工厂函数中，使未来新增策略无需修改DPU主流程。

    输入：
        strategy_name: 注册表中的策略名称。
        random_seed: 传给策略实例的整数随机种子。
        p_node_ids: 需要在启动前绑定的P节点ID。
        queue_ids_by_storage_target: 每块SSD可用的Queue ID列表。

    输出：
        QueueBindingStrategy: 新创建且状态独立的策略实例。
    """
    strategy_class = QUEUE_BINDING_STRATEGIES[strategy_name]
    strategy = strategy_class(random_seed=random_seed)
    strategy.prepare_bindings(
        list(p_node_ids),
        {} if queue_ids_by_storage_target is None else queue_ids_by_storage_target,
    )
    return strategy
