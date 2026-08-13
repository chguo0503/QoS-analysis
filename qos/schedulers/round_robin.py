"""RR调度器，只负责按固定顺序循环选择可用对象。"""


class RoundRobinScheduler:
    """保存RR指针，并从上次停止位置继续扫描。"""

    def __init__(self, item_order):
        """保存固定扫描顺序，并让指针从第一项开始。"""
        self.item_order = list(item_order)
        self.next_item_index = 0

    def select_next(self, is_eligible):
        """返回从当前指针开始遇到的第一个可用对象。"""
        for offset in range(len(self.item_order)):
            # 取模让下标走到末尾后回到开头，完成一整圈扫描。
            item_index = (self.next_item_index + offset) % len(self.item_order)
            item_id = self.item_order[item_index]
            if is_eligible(item_id):
                # 只在成功选中后前移指针，下次优先从后一项开始。
                self.next_item_index = (item_index + 1) % len(self.item_order)
                return item_id
        # 全部对象都不可用时保持原指针，等待下一次调度。
        return None
