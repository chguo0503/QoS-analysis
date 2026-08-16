"""Queue使用可动态设置的槽位WRR，Group使用平滑WRR。"""

from .round_robin import RoundRobinScheduler


def expand_weight_bitmap(item_order, weight_bitmap):
    """功能：把每个对象的整数权重展开成RR扫描槽位。

    目的：保留Queue级原有的固定权重位图和扫描顺序。
    输入：对象顺序和一一对应的非负整数权重。
    输出：权重N即重复N次的对象槽位列表。
    """
    slots = []
    for item_id, weight in zip(item_order, weight_bitmap):
        slots.extend([item_id] * weight)
    return slots


class WeightedRoundRobinScheduler:
    """使用可动态替换权重槽位的Queue级WRR调度器。"""

    def __init__(self, item_order, weight_bitmap):
        """功能：用初始权重位图创建RR扫描表。

        目的：保持原有槽位展开与RR游标语义，同时允许控制面在运行时
        替换Queue权重；没有控制写入时的Baseline扫描顺序完全不变。
        输入：Queue顺序和固定权重位图。
        输出：无；初始化RR游标。
        """
        self.item_order = list(item_order)
        self.weights = {item_id: 0 for item_id in self.item_order}
        self.rr_scheduler = RoundRobinScheduler([])
        self.set_weights(dict(zip(self.item_order, weight_bitmap)))

    def set_weights(self, weights):
        """功能：在运行时更新本调度器指定Queue的整数权重。

        目的：让DPU可以用0临时门控Queue，并用正整数调整组内服务机会；
        每次命中本组的写入都重置RR游标，避免旧槽位位置泄漏到新权重周期。

        输入：``queue_id -> 非负整数权重`` 映射；未提供的Queue保持原权重。
        输出：无；新槽位在下一次仲裁时生效，全0会创建安全的空扫描表。
        """
        normalized_weights = dict(self.weights)
        has_local_update = False
        for item_id in self.item_order:
            if item_id not in weights:
                continue
            has_local_update = True
            weight = weights.get(item_id, 0)
            if not isinstance(weight, int) or isinstance(weight, bool):
                raise TypeError(
                    f"weight for {item_id!r} must be a non-negative integer"
                )
            if weight < 0:
                raise ValueError(
                    f"weight for {item_id!r} must be a non-negative integer"
                )
            normalized_weights[item_id] = weight

        # HierarchicalScheduler会把一张全局部分更新交给每个组；没有命中
        # 本组Queue时保持RR游标，避免无关控制写入改变该组仲裁顺序。
        if not has_local_update:
            return

        self.weights = normalized_weights
        slots = expand_weight_bitmap(
            self.item_order,
            [self.weights[item_id] for item_id in self.item_order],
        )
        self.rr_scheduler = RoundRobinScheduler(slots)

    def has_eligible(self, is_eligible):
        """返回是否至少有一个正权重Queue当前可以参加仲裁。"""
        return any(
            self.weights[item_id] > 0 and is_eligible(item_id)
            for item_id in self.item_order
        )

    def select_next(self, is_eligible):
        """功能：从当前加权槽位中选择下一个可用对象。

        目的：复用原有RR指针语义调度Queue。
        输入：接收item_id并返回是否可调度的函数。
        输出：选中的item_id；全部不可用或全部权重为0时返回None。
        """
        return self.rr_scheduler.select_next(is_eligible)


class SmoothWeightedRoundRobinScheduler:
    """用整数加减和比较实现可动态设置的Group WRR。"""

    def __init__(self, item_order, weight_bitmap):
        """功能：保存Group顺序和初始权重。

        目的：不把权重展开成重复槽位，因此可直接使用每80 us的
        需求Byte数，无需浮点运算或与权重成比例的内存。

        输入：稳定Group顺序和一一对应的非负整数权重。
        输出：无；初始化加权仲裁状态。
        """
        self.item_order = list(item_order)
        self.weights = dict(zip(self.item_order, weight_bitmap))
        self.current_weights = {item_id: 0 for item_id in self.item_order}

    def set_weights(self, weights):
        """功能：在运行时替换所有Group的整数权重。

        目的：根据每Group活跃带宽需求更新组间机会；重置累计
        值可防止已结束需求留下旧的调度信用。

        输入：``group_id -> 非负整数权重`` 映射。
        输出：无；新权重在下一次仲裁时生效。
        """
        self.weights = {
            item_id: int(weights.get(item_id, 0))
            for item_id in self.item_order
        }
        self.current_weights = {item_id: 0 for item_id in self.item_order}

    def select_next(self, is_eligible):
        """功能：返回本轮平滑WRR选中的可用Group。

        目的：只在当前有可下发Queue的Group之间按权重分配机会，
        空Group或令牌不足的Group不消耗本轮服务份额。

        输入：接收group_id并返回组内是否存在可调度Queue的函数。
        输出：选中的group_id；没有可用正权重Group时返回None。
        """
        eligible_items = [
            item_id
            for item_id in self.item_order
            if self.weights[item_id] > 0 and is_eligible(item_id)
        ]
        if not eligible_items:
            return None

        total_weight = 0
        selected_item = eligible_items[0]
        selected_weight = None
        for item_id in eligible_items:
            weight = self.weights[item_id]
            total_weight += weight
            self.current_weights[item_id] += weight
            if (
                selected_weight is None
                or self.current_weights[item_id] > selected_weight
            ):
                selected_item = item_id
                selected_weight = self.current_weights[item_id]

        self.current_weights[selected_item] -= total_weight
        return selected_item
