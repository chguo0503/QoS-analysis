"""定义KV Block选择存储目标时使用的可扩展放置策略。"""

import hashlib


def _stable_target_index(random_seed, identity_parts, target_count):
    """功能：把Block身份稳定地映射成一个SSD下标。

    目的：让随机放置在多GPU事件顺序变化后仍保持可复现，避免策略比较时
    因共享随机数调用顺序不同而改变Block到SSD的基础数据分布。

    输入：
        random_seed: LLM工作负载中配置的放置随机种子。
        identity_parts: 用于唯一描述逻辑Block的字符串序列。
        target_count: 当前策略允许选择的SSD数量。

    输出：
        int: 范围为 ``[0, target_count)`` 的稳定SSD下标。
    """
    identity_text = "\x1f".join(str(part) for part in identity_parts)
    digest = hashlib.blake2b(
        f"{random_seed}\x1e{identity_text}".encode("utf-8"),
        digest_size=8,
    ).digest()
    random_value = int.from_bytes(digest, byteorder="big", signed=False)
    return random_value % target_count


class PlacementStrategy:
    """保存KV Placement策略共用的随机种子。"""

    def __init__(self, random_seed):
        """功能：保存当前放置策略的独立随机种子。

        目的：把KV放置随机性与DPU Queue绑定随机性彻底隔离。

        输入：
            random_seed: 用于可复现目标选择的整数种子。

        输出：
            None: 只初始化策略配置。
        """
        self.random_seed = random_seed

class RandomPlacementStrategy(PlacementStrategy):
    """使用稳定伪随机方式把每个Block放到一个允许访问的SSD。"""

    strategy_name = "random"

    def select_target(self, block, storage_target_ids):
        """功能：为每个Block独立稳定随机选择一个SSD。

        目的：建立当前第一种放置基线，同时保证相同种子和Block身份重复运行
        时获得完全一致的映射。

        输入：
            block: 至少包含全局唯一 ``request_id`` 的Block字典。
            storage_target_ids: 当前GPU允许读取的非空SSD ID列表。
        输出：
            str: 当前Block稳定随机选中的SSD ID。
        """
        target_index = _stable_target_index(
            self.random_seed,
            (block["request_id"],),
            len(storage_target_ids),
        )
        return storage_target_ids[target_index]


PLACEMENT_STRATEGIES = {
    RandomPlacementStrategy.strategy_name: RandomPlacementStrategy,
}


def build_placement_strategy(strategy_name, random_seed):
    """功能：根据LLM YAML中的名称创建KV Placement策略。

    目的：集中完成策略名称校验和实例构造，便于以后注册新的放置算法。

    输入：
        strategy_name: ``PLACEMENT_STRATEGIES`` 注册表中的名称。
        random_seed: 当前GPU独立使用的放置随机种子。

    输出：
        PlacementStrategy: 新创建的放置策略实例。
    """
    strategy_class = PLACEMENT_STRATEGIES[strategy_name]
    return strategy_class(random_seed=random_seed)
