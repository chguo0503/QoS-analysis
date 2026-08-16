"""动态Queue WRR控制面和仲裁语义的回归测试。"""

from types import SimpleNamespace
import unittest

from discrete_simulation.simulator import DiscreteEventSimulator
from qos.schedulers.hierarchical import HierarchicalScheduler
from qos.schedulers.weighted_round_robin import WeightedRoundRobinScheduler


class StubTokenStage:
    """只实现控制事件测试所需的最小Queue令牌接口。"""

    def __init__(self, queue_order):
        self.queue_order = list(queue_order)
        self.update_period_us = 80
        self.rate_updates = []

    def set_queue_rate(self, queue_id, cir_fill, pir_fill):
        """记录同阶段生效的Queue速率设置。"""
        self.rate_updates.append((queue_id, cir_fill, pir_fill))

    def has_queued_requests(self):
        """本测试不登记数据面请求。"""
        return False


def build_two_group_scheduler():
    """创建每组两条Queue、全部初始权重为1的层级调度器。"""
    layout = SimpleNamespace(
        group_order=["g0", "g1"],
        group_queues={
            "g0": ["q0", "q1"],
            "g1": ["q2", "q3"],
        },
    )
    scheduler = HierarchicalScheduler(
        layout,
        {
            "group_weight_bitmap": [1, 1],
            "queue_weight_bitmaps": {
                "g0": [1, 1],
                "g1": [1, 1],
            },
        },
    )
    return layout, scheduler


class WeightedRoundRobinSchedulerTests(unittest.TestCase):
    """验证动态槽位更新、0权重和原有初始扫描顺序。"""

    def test_initial_bitmap_keeps_original_slot_order(self):
        scheduler = WeightedRoundRobinScheduler(
            ["q0", "q1", "q2"],
            [2, 1, 0],
        )

        selected = [scheduler.select_next(lambda _queue_id: True) for _ in range(6)]

        self.assertEqual(selected, ["q0", "q0", "q1", "q0", "q0", "q1"])

    def test_runtime_zero_weight_excludes_queue_and_all_zero_is_safe(self):
        scheduler = WeightedRoundRobinScheduler(["q0", "q1"], [1, 1])

        scheduler.set_weights({"q1": 3})
        self.assertEqual(scheduler.weights, {"q0": 1, "q1": 3})

        scheduler.set_weights({"q0": 0})
        self.assertEqual(
            [scheduler.select_next(lambda _queue_id: True) for _ in range(3)],
            ["q1", "q1", "q1"],
        )

        scheduler.set_weights({"q0": 0, "q1": 0})
        self.assertIsNone(scheduler.select_next(lambda _queue_id: True))
        self.assertFalse(scheduler.has_eligible(lambda _queue_id: True))

    def test_negative_and_non_integer_weights_are_rejected(self):
        scheduler = WeightedRoundRobinScheduler(["q0"], [1])

        with self.assertRaises(ValueError):
            scheduler.set_weights({"q0": -1})
        with self.assertRaises(TypeError):
            scheduler.set_weights({"q0": 1.5})


class HierarchicalDynamicQueueWeightTests(unittest.TestCase):
    """验证Queue门控不会改写固定Group WRR。"""

    def test_partial_update_preserves_other_queues_and_untouched_group_cursor(self):
        _, scheduler = build_two_group_scheduler()
        untouched_rr = scheduler.queue_schedulers["g1"].rr_scheduler

        scheduler.set_queue_weights({"q0": 0})

        self.assertEqual(
            scheduler.queue_schedulers["g0"].weights,
            {"q0": 0, "q1": 1},
        )
        self.assertEqual(
            scheduler.queue_schedulers["g1"].weights,
            {"q2": 1, "q3": 1},
        )
        self.assertIs(scheduler.queue_schedulers["g1"].rr_scheduler, untouched_rr)

    def test_disabled_group_is_skipped_without_changing_group_weights(self):
        _, scheduler = build_two_group_scheduler()
        initial_group_weights = dict(scheduler.group_scheduler.weights)

        scheduler.set_queue_weights({"q0": 0, "q1": 0, "q2": 4, "q3": 0})

        self.assertEqual(
            scheduler.select_next_queue(lambda _queue_id: True),
            "q2",
        )
        self.assertEqual(scheduler.group_scheduler.weights, initial_group_weights)


class QueueWeightControlEventTests(unittest.TestCase):
    """验证Queue权重与rate_update阶段和事件时钟的连接。"""

    def test_same_time_partial_writes_merge_and_last_value_wins(self):
        layout, scheduler = build_two_group_scheduler()
        token_stage = StubTokenStage(
            queue_id
            for group_id in layout.group_order
            for queue_id in layout.group_queues[group_id]
        )
        qos = DiscreteEventSimulator(
            token_stage=token_stage,
            scheduler=scheduler,
            simulation_config={
                "start_time_us": 0,
                "same_timestamp_event_order": ["rate_update"],
            },
        )

        qos.schedule_queue_rate_update("q2", 80, None, 80)
        qos.schedule_queue_weight_update({"q0": 9, "q1": 0, "q3": 0}, 80)
        qos.schedule_queue_weight_update({"q0": 0, "q2": 5}, 80)

        self.assertEqual(qos.next_event_time_us(), 80)
        qos.process_at(80)

        self.assertEqual(token_stage.rate_updates, [("q2", 80, None)])
        self.assertEqual(scheduler.queue_schedulers["g0"].weights, {"q0": 0, "q1": 0})
        self.assertEqual(scheduler.queue_schedulers["g1"].weights, {"q2": 5, "q3": 0})
        self.assertEqual(scheduler.group_scheduler.weights, {"g0": 1, "g1": 1})
        self.assertEqual(
            scheduler.select_next_queue(lambda _queue_id: True),
            "q2",
        )


if __name__ == "__main__":
    unittest.main()
