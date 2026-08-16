"""Utility+EDF多SSD owner lock回归测试。"""

import unittest

from DPU import UtilityEDFController


CAPACITY_BYTES_PER_SECOND = 1_000_000


def register_demand(
    controller,
    storage_target_id,
    queue_id,
    p_node_id,
    demand_group_id,
    *,
    is_initial,
    byte_count,
    request_count,
    deadline_us,
):
    """登记一条整数服务时间的等长Block路径。"""
    controller.register_demand(
        storage_target_id=storage_target_id,
        queue_id=queue_id,
        requested_cir_bytes_per_second=CAPACITY_BYTES_PER_SECOND,
        arrival_time_us=0,
        p_node_id=p_node_id,
        demand_group_id=demand_group_id,
        batch_total_bytes=byte_count,
        path_bytes=byte_count,
        path_request_count=request_count,
        block_size_bytes=byte_count // request_count,
        service_window_us=100,
        deadline_us=deadline_us,
        compute_layer_index=None if is_initial else 0,
        prefetch_layer_index=0 if is_initial else 1,
        inference_arrival_time_us=0,
    )


class UtilityEDFMultiSSDStickyOwnerTests(unittest.TestCase):
    """确保任一路径开始后，owner保留到全部路径排空。"""

    def build_controller(self):
        return UtilityEDFController(
            {
                "SSD0": CAPACITY_BYTES_PER_SECOND,
                "SSD1": CAPACITY_BYTES_PER_SECOND,
            },
            score_mode="integer",
            deadline_allowance_us=2,
        )

    def test_first_empty_path_keeps_owner_with_stale_other_path_depth(self):
        """先排空的路径必须锁住仍在另一SSD服务的owner。"""
        controller = self.build_controller()
        for storage_target_id in ("SSD0", "SSD1"):
            register_demand(
                controller,
                storage_target_id,
                "q0",
                "Powner",
                "owner_group",
                is_initial=True,
                byte_count=100,
                request_count=10,
                deadline_us=100,
            )
        register_demand(
            controller,
            "SSD1",
            "q1",
            "Pprefetch",
            "prefetch_group",
            is_initial=False,
            byte_count=10,
            request_count=1,
            deadline_us=120,
        )

        controller.recalculate("SSD0", 0, {"q0": 10})
        controller.recalculate("SSD1", 0, {"q0": 10, "q1": 1})
        self.assertEqual(controller.selected_p_node_id, "Powner")
        self.assertFalse(controller.owner_locked)

        # SSD1的真实depth已下降到5，但DPU仍只保存初值10。
        # t=50时如果删除SSD0路径后没有sticky lock，EDF
        # 可行性会误判冲突并切到Pprefetch。
        updates = controller.release_empty_demands(
            "SSD0",
            {"q0": 0},
            event_time_us=50,
        )

        self.assertEqual(controller.queue_depths["SSD1"]["q0"], 10)
        self.assertEqual(controller.selected_p_node_id, "Powner")
        self.assertTrue(controller.owner_locked)
        self.assertFalse(updates["coordinates_changed"])
        self.assertEqual(
            controller.decision_history[-1]["reason"],
            "owner_locked",
        )

        # 最后一条owner路径消失后必须解锁并轮到等待者。
        updates = controller.release_empty_demands(
            "SSD1",
            {"q0": 0, "q1": 1},
            event_time_us=60,
        )
        self.assertEqual(controller.selected_p_node_id, "Pprefetch")
        self.assertFalse(controller.owner_locked)
        self.assertTrue(updates["coordinates_changed"])

    def test_nonempty_depth_drop_locks_owner_before_empty_callback(self):
        """完整depth快照中的部分下发也必须立即锁owner。"""
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode="integer",
            deadline_allowance_us=2,
        )
        register_demand(
            controller,
            "SSD0",
            "q0",
            "Powner",
            "owner_group",
            is_initial=True,
            byte_count=100,
            request_count=10,
            deadline_us=100,
        )
        controller.recalculate("SSD0", 0, {"q0": 10})
        register_demand(
            controller,
            "SSD0",
            "q1",
            "Pbetter",
            "better_group",
            is_initial=True,
            byte_count=1,
            request_count=1,
            deadline_us=100,
        )

        updates = controller.release_empty_demands(
            "SSD0",
            {"q0": 5, "q1": 1},
            event_time_us=1,
        )

        self.assertEqual(controller.selected_p_node_id, "Powner")
        self.assertTrue(controller.owner_locked)
        self.assertFalse(updates["coordinates_changed"])
        self.assertEqual(updates["queue_rates"], {})
        # 新arrival必须首次写入等待Gate；已服务owner
        # q0本身不产生任何重配置，因此不会清空其token。
        self.assertEqual(updates["queue_pirs"], {"q1": 0})
        self.assertEqual(updates["queue_weights"], {"q1": 0})
        self.assertEqual(
            controller.decision_history[-1]["reason"],
            "owner_locked",
        )


if __name__ == "__main__":
    unittest.main()
