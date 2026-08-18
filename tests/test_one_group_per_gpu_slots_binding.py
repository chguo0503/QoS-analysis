"""固定GPU Group内多slot Queue绑定的回归测试。"""

import unittest

from DPU.queue_binding_strategies import build_queue_binding_strategy


def make_request(gpu_index, storage_target_id="SSD0", queue_slot=None):
    """构造只包含Queue绑定所需字段的最小请求。"""
    basic = {
        "p_node_id": f"P{gpu_index}",
        "storage_target_id": storage_target_id,
    }
    if queue_slot is not None:
        basic["queue_slot"] = queue_slot
    return {"basic": basic}


class OneGroupPerGpuSlotsBindingTests(unittest.TestCase):
    """验证128 GPU拓扑中的两个固定Queue slot。"""

    def setUp(self):
        self.p_node_ids = [f"P{index}" for index in range(128)]
        self.queues = [f"q{index:03d}" for index in range(256)]
        self.strategy = build_queue_binding_strategy(
            "one_group_per_gpu_slots",
            self.p_node_ids,
            {"SSD0": self.queues, "SSD1": self.queues},
        )

    def test_default_and_explicit_slots_use_the_two_queues_in_gpu_group(self):
        for gpu_index in (0, 1, 63, 127):
            group_start = gpu_index * 2
            for storage_target_id in ("SSD0", "SSD1"):
                default_queue = self.strategy.select_queue(
                    make_request(gpu_index, storage_target_id),
                    self.queues,
                )
                slot_zero_queue = self.strategy.select_queue(
                    make_request(gpu_index, storage_target_id, 0),
                    self.queues,
                )
                slot_one_queue = self.strategy.select_queue(
                    make_request(gpu_index, storage_target_id, 1),
                    self.queues,
                )

                self.assertEqual(default_queue, f"q{group_start:03d}")
                self.assertEqual(slot_zero_queue, default_queue)
                self.assertEqual(slot_one_queue, f"q{group_start + 1:03d}")
                self.assertEqual(
                    int(slot_zero_queue[1:]) // 2,
                    int(slot_one_queue[1:]) // 2,
                )

    def test_slot_is_stable_across_repeated_requests(self):
        request = make_request(47, "SSD1", 1)
        selected = [
            self.strategy.select_queue(request, list(reversed(self.queues)))
            for _ in range(3)
        ]
        self.assertEqual(selected, ["q095", "q095", "q095"])

    def test_slot_must_be_in_group_range(self):
        for invalid_slot in (-1, 2):
            with self.subTest(queue_slot=invalid_slot):
                with self.assertRaisesRegex(ValueError, "0 <= queue_slot < 2"):
                    self.strategy.select_queue(
                        make_request(0, queue_slot=invalid_slot),
                        self.queues,
                    )

    def test_slot_must_be_an_integer(self):
        for invalid_slot in (True, "1", 1.0):
            with self.subTest(queue_slot=invalid_slot):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    self.strategy.select_queue(
                        make_request(0, queue_slot=invalid_slot),
                        self.queues,
                    )

    def test_original_strategy_still_uses_group_first_queue(self):
        original = build_queue_binding_strategy(
            "one_group_per_gpu",
            self.p_node_ids,
            {"SSD0": self.queues},
        )
        self.assertEqual(
            original.select_queue(make_request(47, queue_slot=1), self.queues),
            "q094",
        )


if __name__ == "__main__":
    unittest.main()
