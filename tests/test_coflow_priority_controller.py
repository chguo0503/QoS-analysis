"""跨SSD coflow优先级控制器回归测试。"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from DPU import (
    CoflowPriorityController,
    DPURequestGateway,
    build_queue_binding_strategy,
)
from qos import build_qos_simulator
from qos_ssd_simulator import JointSimulation
from simulation_common.config_utils import load_yaml


SSD_CAPACITY_BYTES_PER_SECOND = 40_000_000_000
PROJECT_DIR = Path(__file__).resolve().parents[1]
SIMULATION_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"


def register_path(
    controller,
    target,
    queue_id,
    p_node_id,
    demand_group_id,
    batch_total_bytes,
    arrival_time_us,
):
    """用统一元数据登记一条coflow路径。"""
    controller.register_demand(
        storage_target_id=target,
        queue_id=queue_id,
        requested_cir_bytes_per_second=1_000_000_000,
        arrival_time_us=arrival_time_us,
        p_node_id=p_node_id,
        demand_group_id=demand_group_id,
        batch_total_bytes=batch_total_bytes,
        path_bytes=batch_total_bytes // 2,
        service_window_us=1_000,
        deadline_us=arrival_time_us + 1_000,
    )


class CoflowPriorityControllerTests(unittest.TestCase):
    """验证固定GPU排名、跨盘准入和有限工作量无饥饿。"""

    def build_controller(self, ordering="shortest", selection_width=1):
        """创建两块40 GB/s SSD的控制器。"""
        return CoflowPriorityController(
            {
                "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
                "SSD1": SSD_CAPACITY_BYTES_PER_SECOND,
            },
            ordering=ordering,
            selection_width=selection_width,
        )

    def register_two_path_coflow(
        self,
        controller,
        p_node_id,
        queue_id,
        demand_group_id,
        batch_total_bytes,
        arrival_time_us,
    ):
        """在两块SSD上为同一GPU登记同一coflow。"""
        for target in ("SSD0", "SSD1"):
            register_path(
                controller,
                target,
                queue_id,
                p_node_id,
                demand_group_id,
                batch_total_bytes,
                arrival_time_us,
            )

    def test_shortest_preempts_consistently_and_waiter_is_eventually_selected(self):
        """短coflow在两盘上同时抢占，完成后等待者恢复服务。"""
        controller = self.build_controller(selection_width=1)
        self.register_two_path_coflow(
            controller,
            "Pslow",
            "q000",
            "slow_layer_0",
            100_000,
            0,
        )
        for target in ("SSD0", "SSD1"):
            controller.recalculate(target, event_time_us=0)
        self.assertEqual(controller.selected_p_nodes, ("Pslow",))

        self.register_two_path_coflow(
            controller,
            "Pfast",
            "q001",
            "fast_layer_0",
            10_000,
            10,
        )
        updates_by_target = {
            target: controller.recalculate(target, event_time_us=10)
            for target in ("SSD0", "SSD1")
        }
        self.assertEqual(controller.selected_p_nodes, ("Pfast",))
        for updates in updates_by_target.values():
            self.assertEqual(
                updates["queue_rates"],
                {
                    "q000": 0,
                    "q001": SSD_CAPACITY_BYTES_PER_SECOND,
                },
            )
            self.assertEqual(
                updates["queue_pirs"],
                {"q000": 0, "q001": None},
            )
            self.assertEqual(updates["queue_weights"], {"q000": 0})
            self.assertIsNone(updates["group_weights"])

        # 只释放Pfast的第一条路径时，它在另一块SSD上
        # 仍活跃，全局准入集不应提前切换。
        controller.release_empty_demands(
            "SSD0",
            {"q000": 1, "q001": 0},
            event_time_us=20,
        )
        self.assertEqual(controller.selected_p_nodes, ("Pfast",))

        controller.release_empty_demands(
            "SSD1",
            {"q000": 1, "q001": 0},
            event_time_us=30,
        )
        ssd0_updates = controller.recalculate("SSD0", event_time_us=30)
        self.assertEqual(controller.selected_p_nodes, ("Pslow",))
        self.assertEqual(
            ssd0_updates["queue_rates"],
            {"q000": SSD_CAPACITY_BYTES_PER_SECOND},
        )
        self.assertEqual(ssd0_updates["queue_pirs"], {"q000": None})
        self.assertEqual(ssd0_updates["queue_weights"], {"q000": 1})

        statistics = controller.statistics()
        slow_stats = statistics["p_node_statistics"]["Pslow"]
        self.assertEqual(slow_stats["selection_count"], 2)
        self.assertEqual(slow_stats["total_wait_us"], 20)
        self.assertEqual(slow_stats["max_wait_us"], 20)

    def test_later_layer_keeps_first_batch_priority(self):
        """同一GPU的后续层不会因当层大小改变推理级排名。"""
        controller = self.build_controller(selection_width=1)
        self.register_two_path_coflow(
            controller,
            "P0",
            "q000",
            "p0_layer_0",
            10,
            0,
        )
        for target in ("SSD0", "SSD1"):
            controller.recalculate(target, event_time_us=0)
            controller.release_empty_demands(
                target,
                {"q000": 0},
                event_time_us=1,
            )

        self.register_two_path_coflow(
            controller,
            "P1",
            "q001",
            "p1_layer_0",
            100,
            2,
        )
        self.register_two_path_coflow(
            controller,
            "P0",
            "q000",
            "p0_layer_1",
            10_000,
            2,
        )
        for target in ("SSD0", "SSD1"):
            controller.recalculate(target, event_time_us=2)

        self.assertEqual(controller.selected_p_nodes, ("P0",))
        self.assertEqual(
            controller.p_node_profiles["P0"]["priority_batch_total_bytes"],
            10,
        )
        self.assertEqual(controller.queue_pirs["SSD0"]["q001"], 0)
        self.assertEqual(controller.queue_weights["SSD1"]["q001"], 0)

    def test_selection_width_splits_capacity_and_gates_only_non_selected(self):
        """K=2时两条短Queue平分容量，第三条被PIR和权重双重门控。"""
        controller = self.build_controller(selection_width=2)
        for p_node_id, queue_id, byte_count in (
            ("P0", "q000", 30),
            ("P1", "q001", 10),
            ("P2", "q002", 20),
        ):
            register_path(
                controller,
                "SSD0",
                queue_id,
                p_node_id,
                f"{p_node_id}_layer_0",
                byte_count,
                0,
            )

        updates = controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(controller.selected_p_nodes, ("P1", "P2"))
        self.assertEqual(
            controller.queue_rates["SSD0"],
            {
                "q000": 0,
                "q001": 20_000_000_000,
                "q002": 20_000_000_000,
            },
        )
        self.assertEqual(controller.queue_pirs["SSD0"]["q000"], 0)
        self.assertEqual(controller.queue_weights["SSD0"]["q000"], 0)
        self.assertEqual(updates["queue_weights"], {"q000": 0})
        self.assertLessEqual(
            sum(controller.queue_rates["SSD0"].values()),
            SSD_CAPACITY_BYTES_PER_SECOND,
        )

    def test_finite_selected_pir_matches_share_and_keeps_waiter_gated(self):
        """paced模式选中Queue的PIR=CIR share，未选仍为全零。"""
        controller = CoflowPriorityController(
            {"SSD0": SSD_CAPACITY_BYTES_PER_SECOND},
            ordering="shortest",
            selection_width=2,
            finite_selected_pir=True,
        )
        for p_node_id, queue_id, byte_count in (
            ("P0", "q000", 30),
            ("P1", "q001", 10),
            ("P2", "q002", 20),
        ):
            register_path(
                controller,
                "SSD0",
                queue_id,
                p_node_id,
                f"{p_node_id}_group_0",
                byte_count,
                0,
            )

        updates = controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(controller.selected_p_nodes, ("P1", "P2"))
        self.assertEqual(
            controller.queue_rates["SSD0"],
            {
                "q000": 0,
                "q001": 20_000_000_000,
                "q002": 20_000_000_000,
            },
        )
        self.assertEqual(
            controller.queue_pirs["SSD0"],
            {
                "q000": 0,
                "q001": 20_000_000_000,
                "q002": 20_000_000_000,
            },
        )
        self.assertEqual(controller.queue_weights["SSD0"]["q000"], 0)
        self.assertEqual(
            updates["queue_pirs"],
            {
                "q000": 0,
                "q001": 20_000_000_000,
                "q002": 20_000_000_000,
            },
        )
        self.assertIsNone(updates["group_weights"])

    def test_largest_and_highest_demand_orderings(self):
        """largest按字节数，highest_demand按固化的Byte/s选择。"""
        largest = self.build_controller(
            ordering="largest",
            selection_width=1,
        )
        for p_node_id, queue_id, byte_count in (
            ("P0", "q000", 100),
            ("P1", "q001", 50),
            ("P2", "q002", 1),
        ):
            register_path(
                largest,
                "SSD0",
                queue_id,
                p_node_id,
                f"{p_node_id}_layer_0",
                byte_count,
                0,
            )
        largest.recalculate("SSD0", event_time_us=0)
        self.assertEqual(largest.selected_p_nodes, ("P0",))

        highest = self.build_controller(
            ordering="highest_demand",
            selection_width=1,
        )
        for p_node_id, queue_id, byte_count, service_window_us in (
            ("P0", "q000", 100, 1_000),
            ("P1", "q001", 50, 100),
            # 零服务窗口按无限紧迫需求排在有限值之前。
            ("P2", "q002", 1, 0),
        ):
            highest.register_demand(
                storage_target_id="SSD0",
                queue_id=queue_id,
                requested_cir_bytes_per_second=1,
                arrival_time_us=0,
                p_node_id=p_node_id,
                demand_group_id=f"{p_node_id}_layer_0",
                batch_total_bytes=byte_count,
                path_bytes=byte_count,
                service_window_us=service_window_us,
                deadline_us=service_window_us,
            )
        highest.recalculate("SSD0", event_time_us=0)
        self.assertEqual(highest.selected_p_nodes, ("P2",))
        self.assertIsNone(
            highest.p_node_profiles["P2"][
                "priority_demand_bytes_per_second"
            ]
        )
        self.assertEqual(
            highest.p_node_profiles["P1"][
                "priority_demand_bytes_per_second"
            ],
            500_000,
        )

    def test_lowest_demand_uses_exact_ratio_and_places_zero_window_last(self):
        """lowest_demand精确区分2**53以上比例，零窗口最后。"""
        controller = self.build_controller(
            ordering="lowest_demand",
            selection_width=3,
        )
        exact_integer = 2 ** 53
        for p_node_id, queue_id, byte_count, service_window_us in (
            # Pslightly的比例仅比1大1/2**53，浮点键容易将
            # 它与Pexact合并；精确交叉相乘必须选Pexact在前。
            ("Pslightly", "q000", exact_integer + 1, exact_integer),
            ("Pexact", "q001", exact_integer, exact_integer),
            ("Pzero", "q002", 1, 0),
        ):
            controller.register_demand(
                storage_target_id="SSD0",
                queue_id=queue_id,
                requested_cir_bytes_per_second=1,
                arrival_time_us=0,
                p_node_id=p_node_id,
                demand_group_id=f"{p_node_id}_layer_0",
                batch_total_bytes=byte_count,
                path_bytes=byte_count,
                service_window_us=service_window_us,
                deadline_us=service_window_us,
            )

        controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(
            controller.selected_p_nodes,
            ("Pexact", "Pslightly", "Pzero"),
        )
        self.assertIsNone(
            controller.p_node_profiles["Pzero"][
                "priority_demand_ratio_numerator"
            ]
        )

    def test_longest_window_descends_then_uses_batch_bytes_as_tie_break(self):
        """longest_window先选长窗口，同窗口先选小batch。"""
        controller = self.build_controller(
            ordering="longest_window",
            selection_width=3,
        )
        for p_node_id, queue_id, byte_count, service_window_us in (
            ("Pshort", "q000", 1, 100),
            ("Plarge", "q001", 1_000, 200),
            ("Psmall", "q002", 500, 200),
        ):
            controller.register_demand(
                storage_target_id="SSD0",
                queue_id=queue_id,
                requested_cir_bytes_per_second=1,
                arrival_time_us=0,
                p_node_id=p_node_id,
                demand_group_id=f"{p_node_id}_layer_0",
                batch_total_bytes=byte_count,
                path_bytes=byte_count,
                service_window_us=service_window_us,
                deadline_us=service_window_us,
            )

        controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(
            controller.selected_p_nodes,
            ("Psmall", "Plarge", "Pshort"),
        )

    def test_min_window_prioritizes_boundary_then_sorts_each_group_by_bytes(self):
        """窗口等于阈值属于优先组，两组各自执行SPT。"""
        controller = self.build_controller(
            ordering="min_window_50000",
            selection_width=7,
        )
        candidates = (
            # 低于阈值的1 Byte不能越过优先组。
            ("Pbelow_tiny", "q000", 1, 49_999, 0),
            ("Pboundary_large", "q001", 30, 50_000, 0),
            ("Pabove_small", "q002", 10, 60_000, 5),
            ("Pabove_early", "q003", 20, 70_000, 1),
            ("Pabove_late", "q004", 20, 80_000, 2),
            # 同bytes/到达时刻时由登记顺序稳定打破平局。
            ("Pabove_later_order", "q005", 20, 90_000, 2),
            ("Pbelow_mid", "q006", 2, 0, 0),
        )
        for (
            p_node_id,
            queue_id,
            byte_count,
            service_window_us,
            arrival_time_us,
        ) in candidates:
            controller.register_demand(
                storage_target_id="SSD0",
                queue_id=queue_id,
                requested_cir_bytes_per_second=1,
                arrival_time_us=arrival_time_us,
                p_node_id=p_node_id,
                demand_group_id=f"{p_node_id}_layer_0",
                batch_total_bytes=byte_count,
                path_bytes=byte_count,
                service_window_us=service_window_us,
                deadline_us=arrival_time_us + service_window_us,
            )

        controller.recalculate("SSD0", event_time_us=5)
        self.assertEqual(
            controller.selected_p_nodes,
            (
                "Pabove_small",
                "Pabove_early",
                "Pabove_late",
                "Pabove_later_order",
                "Pboundary_large",
                "Pbelow_tiny",
                "Pbelow_mid",
            ),
        )
        self.assertEqual(controller.min_window_threshold_us, 50_000)
        self.assertEqual(
            controller.statistics()["min_window_threshold_us"],
            50_000,
        )

    def test_min_window_requires_canonical_positive_integer_threshold(self):
        """min_window严格拒绝零、符号、小数、前导零和后缀。"""
        for ordering in (
            "min_window_0",
            "min_window_-1",
            "min_window_+1",
            "min_window_1.0",
            "min_window_050000",
            "min_window_50000_extra",
        ):
            with self.subTest(ordering=ordering):
                with self.assertRaises(ValueError):
                    self.build_controller(ordering=ordering)

        unchanged = self.build_controller(ordering="shortest")
        self.assertIsNone(unchanged.min_window_threshold_us)
        self.assertNotIn(
            "min_window_threshold_us",
            unchanged.statistics(),
        )


class PersistentCohortControllerTests(unittest.TestCase):
    """验证cohort锁定、层间持有、轮转和有限任务无饥饿。"""

    def build_controller(self):
        """创建K=1、每GPU四个KV读组的persistent cohort。"""
        return CoflowPriorityController(
            {"SSD0": SSD_CAPACITY_BYTES_PER_SECOND},
            ordering="shortest",
            selection_width=1,
            persistent_cohort=True,
            expected_coflow_count=4,
        )

    def register(
        self,
        controller,
        p_node_id,
        queue_id,
        group_index,
        byte_count,
        arrival_time_us,
    ):
        """登记persistent测试的单SSD coflow。"""
        register_path(
            controller,
            "SSD0",
            queue_id,
            p_node_id,
            f"{p_node_id}_group_{group_index}",
            byte_count,
            arrival_time_us,
        )

    def build_locked_fast_owner(self):
        """构造同时更优候选替换早到者并开始服务的状态。"""
        controller = self.build_controller()
        self.register(controller, "Pslow", "q000", 0, 100, 0)
        controller.recalculate("SSD0", event_time_us=0)
        self.register(controller, "Pfast", "q001", 0, 10, 0)
        controller.recalculate("SSD0", event_time_us=0)
        controller.release_empty_demands(
            "SSD0",
            {"q000": 1, "q001": 0},
            event_time_us=1,
        )
        return controller

    def test_same_time_late_better_candidate_replaces_before_first_service(self):
        """首次Queue排空前，同时后登记的更短GPU可替换候选。"""
        controller = self.build_controller()
        self.register(controller, "Pslow", "q000", 0, 100, 0)
        controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(controller.selected_p_nodes, ("Pslow",))
        self.assertFalse(controller.cohort_locked)

        self.register(controller, "Pfast", "q001", 0, 10, 0)
        updates = controller.recalculate("SSD0", event_time_us=0)
        self.assertEqual(controller.selected_p_nodes, ("Pfast",))
        self.assertEqual(controller.cohort_members, ("Pfast",))
        self.assertEqual(controller.queue_pirs["SSD0"]["q000"], 0)
        self.assertEqual(controller.queue_weights["SSD0"]["q000"], 0)
        self.assertEqual(
            updates["queue_rates"]["q001"],
            SSD_CAPACITY_BYTES_PER_SECOND,
        )
        self.assertFalse(controller.cohort_locked)

        controller.release_empty_demands(
            "SSD0",
            {"q000": 1, "q001": 0},
            event_time_us=1,
        )
        self.assertTrue(controller.cohort_started)
        self.assertTrue(controller.cohort_locked)
        self.assertEqual(controller.cohort_members, ("Pfast",))

    def test_inactive_owner_keeps_slot_across_compute_gap(self):
        """已锁定owner的Queue排空后仍占槽，等待GPU不会被提前灌入。"""
        controller = self.build_locked_fast_owner()

        self.assertEqual(controller._active_p_nodes(), {"Pslow"})
        self.assertEqual(controller.selected_p_nodes, ("Pfast",))
        self.assertEqual(controller.cohort_members, ("Pfast",))
        self.assertEqual(controller.queue_pirs["SSD0"]["q000"], 0)
        self.assertEqual(controller.queue_weights["SSD0"]["q000"], 0)

        # 模拟Pfast完成SSD屏障和compute切换后提交下一层。
        self.register(controller, "Pfast", "q001", 1, 10, 2)
        controller.recalculate("SSD0", event_time_us=2)
        self.assertEqual(controller.selected_p_nodes, ("Pfast",))
        self.assertEqual(
            controller.queue_rates["SSD0"]["q001"],
            SSD_CAPACITY_BYTES_PER_SECOND,
        )
        self.assertIsNone(controller.queue_pirs["SSD0"]["q001"])
        self.assertEqual(controller.queue_weights["SSD0"]["q001"], 1)
        self.assertEqual(controller.queue_pirs["SSD0"]["q000"], 0)

    def test_four_groups_rotate_slot_and_all_finite_work_completes(self):
        """每个owner完成四组后让出槽位，等待GPU最终获得服务。"""
        controller = self.build_locked_fast_owner()

        # Pfast的group0已在fixture中排空，再完成1至3。
        for group_index in range(1, 4):
            event_time_us = group_index * 10
            self.register(
                controller,
                "Pfast",
                "q001",
                group_index,
                10,
                event_time_us,
            )
            controller.recalculate("SSD0", event_time_us=event_time_us)
            controller.release_empty_demands(
                "SSD0",
                {"q000": 1, "q001": 0},
                event_time_us=event_time_us + 1,
            )

        self.assertEqual(
            controller.p_node_profiles["Pfast"]["completed_coflow_count"],
            4,
        )
        self.assertEqual(controller.selected_p_nodes, ("Pslow",))
        self.assertEqual(controller.cohort_members, ("Pslow",))
        self.assertEqual(
            controller.queue_rates["SSD0"]["q000"],
            SSD_CAPACITY_BYTES_PER_SECOND,
        )

        # Pslow的group0一直在Queue内等待；轮到它后排空，
        # 然后持续占槽完成余下三组。
        controller.release_empty_demands(
            "SSD0",
            {"q000": 0},
            event_time_us=40,
        )
        for group_index in range(1, 4):
            event_time_us = 40 + group_index * 10
            self.register(
                controller,
                "Pslow",
                "q000",
                group_index,
                100,
                event_time_us,
            )
            controller.recalculate("SSD0", event_time_us=event_time_us)
            controller.release_empty_demands(
                "SSD0",
                {"q000": 0},
                event_time_us=event_time_us + 1,
            )

        statistics = controller.statistics()
        self.assertEqual(statistics["active_demand_count"], 0)
        self.assertEqual(statistics["selected_p_node_ids"], [])
        self.assertEqual(statistics["cohort_member_ids"], [])
        self.assertEqual(statistics["completed_coflow_count"], 8)
        self.assertEqual(
            {
                p_node_id: profile["completed_coflow_count"]
                for p_node_id, profile in statistics[
                    "p_node_statistics"
                ].items()
            },
            {"Pslow": 4, "Pfast": 4},
        )

    def test_cohort_strategy_parser_passes_uniform_layer_count(self):
        """JointSimulation将cohort名称和四层KV读组数传入控制器。"""
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        simulation = JointSimulation(
            config=config,
            rate_control_strategy_name="cohort_lowest_demand_k2",
        )
        controller = simulation.dpu.rate_controller

        self.assertIsInstance(controller, CoflowPriorityController)
        self.assertEqual(controller.ordering, "lowest_demand")
        self.assertEqual(controller.selection_width, 2)
        self.assertTrue(controller.persistent_cohort)
        self.assertEqual(controller.expected_coflow_count, 4)


class DynamicProgressOrderingTests(unittest.TestCase):
    """验证动态进度排名只由完整coflow Queue排空驱动。"""

    def build_controller(self, ordering, targets=("SSD0",)):
        """创建四个KV读组的非persistent动态控制器。"""
        return CoflowPriorityController(
            {
                target: SSD_CAPACITY_BYTES_PER_SECOND
                for target in targets
            },
            ordering=ordering,
            selection_width=1,
            persistent_cohort=False,
            expected_coflow_count=4,
        )

    def register(
        self,
        controller,
        target,
        queue_id,
        p_node_id,
        group_id,
        batch_total_bytes,
        arrival_time_us,
    ):
        """登记一条指定coflow路径。"""
        register_path(
            controller,
            target,
            queue_id,
            p_node_id,
            group_id,
            batch_total_bytes,
            arrival_time_us,
        )

    def test_multi_ssd_group_increments_progress_only_after_last_path(self):
        """同一demand group的两块SSD路径全部排空才计1次进度。"""
        controller = self.build_controller(
            "most_progress",
            targets=("SSD0", "SSD1"),
        )
        for target in ("SSD0", "SSD1"):
            self.register(
                controller,
                target,
                "q000",
                "P0",
                "P0_group_0",
                100,
                0,
            )
            controller.recalculate(target, event_time_us=0)

        controller.release_empty_demands(
            "SSD0",
            {"q000": 0},
            event_time_us=1,
        )
        self.assertEqual(
            controller.p_node_profiles["P0"]["completed_coflow_count"],
            0,
        )
        self.assertEqual(controller.completed_coflow_count, 0)

        controller.release_empty_demands(
            "SSD1",
            {"q000": 0},
            event_time_us=2,
        )
        self.assertEqual(
            controller.p_node_profiles["P0"]["completed_coflow_count"],
            1,
        )
        self.assertEqual(controller.completed_coflow_count, 1)

    def _complete_first_p0_group(self, controller):
        """让P0单独完成第一组，建立可见的动态进度。"""
        self.register(
            controller,
            "SSD0",
            "q000",
            "P0",
            "P0_group_0",
            100,
            0,
        )
        controller.recalculate("SSD0", event_time_us=0)
        controller.release_empty_demands(
            "SSD0",
            {"q000": 0},
            event_time_us=1,
        )
        self.assertEqual(
            controller.p_node_profiles["P0"]["completed_coflow_count"],
            1,
        )

    def _register_next_p0_and_new_p1(self, controller, p1_bytes):
        """同时登记已有进度的P0与新来P1。"""
        self.register(
            controller,
            "SSD0",
            "q000",
            "P0",
            "P0_group_1",
            100,
            2,
        )
        self.register(
            controller,
            "SSD0",
            "q001",
            "P1",
            "P1_group_0",
            p1_bytes,
            2,
        )
        controller.recalculate("SSD0", event_time_us=2)

    def test_most_progress_reorders_after_completed_group(self):
        """most_progress让已完成1组的较大P0压过新来的小P1。"""
        controller = self.build_controller("most_progress")
        self._complete_first_p0_group(controller)
        self._register_next_p0_and_new_p1(controller, p1_bytes=10)

        self.assertEqual(controller.selected_p_nodes, ("P0",))
        self.assertEqual(controller.queue_pirs["SSD0"]["q001"], 0)
        self.assertEqual(controller.queue_weights["SSD0"]["q001"], 0)

    def test_remaining_shortest_uses_remaining_group_product(self):
        """remaining_shortest比较3×100和4×80，选前者。"""
        controller = self.build_controller("remaining_shortest")
        self._complete_first_p0_group(controller)
        self._register_next_p0_and_new_p1(controller, p1_bytes=80)

        self.assertEqual(controller.selected_p_nodes, ("P0",))
        p0_key = controller._profile_priority_key("P0")
        p1_key = controller._profile_priority_key("P1")
        self.assertEqual(p0_key[:2], (300, -1))
        self.assertEqual(p1_key[:2], (320, 0))

    def test_existing_shortest_ignores_completed_progress(self):
        """相同进度场景中，原shortest仍只按首批batch选P1。"""
        controller = self.build_controller("shortest")
        self._complete_first_p0_group(controller)
        self._register_next_p0_and_new_p1(controller, p1_bytes=10)

        self.assertEqual(controller.selected_p_nodes, ("P1",))

    def test_parser_accepts_both_dynamic_orderings_without_persistence(self):
        """普通coflow策略传入四层数但不启用persistent cohort。"""
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        for strategy_name, ordering, width in (
            ("coflow_most_progress_k1", "most_progress", 1),
            ("coflow_remaining_shortest_k2", "remaining_shortest", 2),
        ):
            with self.subTest(strategy_name=strategy_name):
                simulation = JointSimulation(
                    config=config,
                    rate_control_strategy_name=strategy_name,
                )
                controller = simulation.dpu.rate_controller
                self.assertEqual(controller.ordering, ordering)
                self.assertEqual(controller.selection_width, width)
                self.assertFalse(controller.persistent_cohort)
                self.assertEqual(controller.expected_coflow_count, 4)


class PacedCoflowParserTests(unittest.TestCase):
    """验证paced策略名只启用有限选中PIR。"""

    def test_parser_enables_finite_pir_without_persistent_cohort(self):
        """paced_<ordering>_kK透传ordering/K并保持非persistent。"""
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        simulation = JointSimulation(
            config=config,
            rate_control_strategy_name="paced_remaining_shortest_k2",
        )
        controller = simulation.dpu.rate_controller

        self.assertEqual(controller.ordering, "remaining_shortest")
        self.assertEqual(controller.selection_width, 2)
        self.assertEqual(controller.expected_coflow_count, 4)
        self.assertTrue(controller.finite_selected_pir)
        self.assertFalse(controller.persistent_cohort)


class MinWindowParserTests(unittest.TestCase):
    """验证参数化ordering不被策略名的最后_kK分割破坏。"""

    def test_parser_preserves_full_ordering_and_threshold(self):
        """coflow_min_window_50000_k1传入完整ordering。"""
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        simulation = JointSimulation(
            config=config,
            rate_control_strategy_name="coflow_min_window_50000_k1",
        )
        controller = simulation.dpu.rate_controller

        self.assertEqual(controller.ordering, "min_window_50000")
        self.assertEqual(controller.min_window_threshold_us, 50_000)
        self.assertEqual(controller.selection_width, 1)
        self.assertFalse(controller.persistent_cohort)
        self.assertFalse(controller.finite_selected_pir)


class _FakeBinding:
    """测试用固定 ``(p_node, SSD) -> Queue`` 绑定。"""

    strategy_name = "fake_exclusive"

    def __init__(self, bindings):
        self.bindings = dict(bindings)

    def select_queue(self, request, queue_ids):
        basic = request["basic"]
        queue_id = self.bindings[(
            basic["p_node_id"],
            basic["storage_target_id"],
        )]
        if queue_id not in queue_ids:
            raise KeyError(queue_id)
        return queue_id


class _FakeQoS:
    """只实现DPU硬件边界的记录型QoS。"""

    def __init__(self, queue_ids):
        self.token_stage = SimpleNamespace(update_period_us=80)
        self.depths = {queue_id: 0 for queue_id in queue_ids}
        self.rate_updates = []
        self.queue_weight_updates = []
        self.group_weight_updates = []
        self.observer = None

    def set_queue_state_observer(self, observer):
        self.observer = observer

    def queue_io_counts(self):
        return dict(self.depths)

    def input(self, request):
        self.depths[request["queue_id"]] += 1
        return request

    def schedule_queue_rate_update(self, queue_id, cir, pir, event_time_us):
        self.rate_updates.append((queue_id, cir, pir, event_time_us))

    def schedule_queue_weight_update(self, weights, event_time_us):
        self.queue_weight_updates.append((dict(weights), event_time_us))

    def schedule_group_weight_update(self, weights, event_time_us):
        self.group_weight_updates.append((dict(weights), event_time_us))


def make_request(
    request_id,
    p_node_id,
    target,
    size_bytes,
    demand_group_id,
    requested_cir,
    **demand_fields,
):
    """构造一条带可选聚合元数据的DPU请求。"""
    demand_bw = {
        "demand_group_id": demand_group_id,
        "aggregate_required_bytes_per_second": requested_cir,
    }
    demand_bw.update(demand_fields)
    return {
        "basic": {
            "request_id": request_id,
            "p_node_id": p_node_id,
            "storage_target_id": target,
            "size_bytes": size_bytes,
        },
        "demand_bw": demand_bw,
    }


class DPUCoflowIntegrationTests(unittest.TestCase):
    """验证submit_batch聚合元数据与Queue控制写入。"""

    def test_submit_batch_passes_aggregate_metadata_and_never_writes_groups(self):
        """路径/批次字节正确聚合，Queue Gate写入但Group不动。"""
        queue_ids = ["q000", "q001"]
        qos_by_target = {
            "SSD0": _FakeQoS(queue_ids),
            "SSD1": _FakeQoS(queue_ids),
        }
        controller = CoflowPriorityController(
            {
                "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
                "SSD1": SSD_CAPACITY_BYTES_PER_SECOND,
            },
            ordering="fifo",
            selection_width=1,
        )
        binding = _FakeBinding({
            ("P0", "SSD0"): "q000",
            ("P0", "SSD1"): "q000",
            ("P1", "SSD0"): "q001",
        })
        gateway = DPURequestGateway(
            {target: queue_ids for target in qos_by_target},
            binding,
            lambda request: qos_by_target[
                request["storage_target_id"]
            ].input(request),
            qos_by_target,
            controller,
        )

        requests = [
            make_request(
                "p0_s0_a",
                "P0",
                "SSD0",
                100,
                "p0_layer_1",
                600_000,
                aggregate_bytes_on_storage_target=300,
                service_window_us=500,
            ),
            make_request(
                "p0_s0_b",
                "P0",
                "SSD0",
                200,
                "p0_layer_1",
                600_000,
                aggregate_bytes_on_storage_target=300,
                service_window_us=500,
            ),
            make_request(
                "p0_s1",
                "P0",
                "SSD1",
                300,
                "p0_layer_1",
                600_000,
                aggregate_bytes_on_storage_target=300,
                service_window_us=500,
            ),
            # 第二个GPU缺失service_window，dispatcher应用
            # path_bytes/requested rate推导500 us。
            make_request(
                "p1_s0",
                "P1",
                "SSD0",
                50,
                "p1_layer_1",
                100_000,
            ),
        ]
        submitted = gateway.submit_batch(requests, arrival_time_us=50)

        p0_ssd0 = controller.demands["SSD0"]["q000"]
        p0_ssd1 = controller.demands["SSD1"]["q000"]
        p1_ssd0 = controller.demands["SSD0"]["q001"]
        for demand in (p0_ssd0, p0_ssd1):
            self.assertEqual(demand["batch_total_bytes"], 600)
            self.assertEqual(demand["path_bytes"], 300)
            self.assertEqual(demand["service_window_us"], 500)
            self.assertEqual(demand["deadline_us"], 550)
        self.assertEqual(p1_ssd0["batch_total_bytes"], 50)
        self.assertEqual(p1_ssd0["path_bytes"], 50)
        self.assertEqual(p1_ssd0["service_window_us"], 500)
        self.assertEqual(p1_ssd0["deadline_us"], 550)

        self.assertEqual(controller.selected_p_nodes, ("P0",))
        self.assertEqual(controller.queue_pirs["SSD0"]["q001"], 0)
        self.assertEqual(controller.queue_weights["SSD0"]["q001"], 0)
        self.assertEqual(
            qos_by_target["SSD0"].queue_weight_updates,
            [({"q001": 0}, 50)],
        )
        self.assertFalse(qos_by_target["SSD1"].queue_weight_updates)
        self.assertFalse(qos_by_target["SSD0"].group_weight_updates)
        self.assertFalse(qos_by_target["SSD1"].group_weight_updates)
        self.assertEqual(gateway.group_weight_write_count, 0)
        self.assertEqual(gateway.queue_weight_write_count, 1)
        self.assertEqual(len(submitted), 4)
        self.assertEqual(
            set(submitted[0]),
            {
                "request_id",
                "p_node_id",
                "storage_target_id",
                "size_bytes",
                "queue_id",
                "arrival_time_us",
            },
        )

    def test_depth_change_without_empty_demand_produces_no_control_write(self):
        """非空Queue的高频depth变化不重算也不写任何SSD。"""
        queue_ids = ["q000"]
        qos_by_target = {
            "SSD0": _FakeQoS(queue_ids),
            "SSD1": _FakeQoS(queue_ids),
        }
        controller = CoflowPriorityController({
            "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
            "SSD1": SSD_CAPACITY_BYTES_PER_SECOND,
        })
        gateway = DPURequestGateway(
            {target: queue_ids for target in qos_by_target},
            _FakeBinding({
                ("P0", "SSD0"): "q000",
                ("P0", "SSD1"): "q000",
            }),
            lambda request: qos_by_target[
                request["storage_target_id"]
            ].input(request),
            qos_by_target,
            controller,
        )
        gateway.submit_batch([
            make_request(
                "p0_s0_a",
                "P0",
                "SSD0",
                100,
                "p0_layer_1",
                100_000,
            ),
            make_request(
                "p0_s0_b",
                "P0",
                "SSD0",
                100,
                "p0_layer_1",
                100_000,
            ),
            make_request(
                "p0_s1_a",
                "P0",
                "SSD1",
                100,
                "p0_layer_1",
                100_000,
            ),
        ], arrival_time_us=0)
        initial_rate_writes = gateway.rate_control_write_count
        initial_weight_writes = gateway.queue_weight_write_count

        # 模拟SSD0只下发了一条IO，其Demand对应Queue仍非空。
        qos_by_target["SSD0"].depths["q000"] = 1
        gateway.on_qos_queue_state_change("SSD0", event_time_us=10)

        self.assertEqual(gateway.rate_control_write_count, initial_rate_writes)
        self.assertEqual(
            gateway.queue_weight_write_count,
            initial_weight_writes,
        )
        self.assertEqual(controller.selected_p_nodes, ("P0",))
        self.assertEqual(len(controller.demands["SSD0"]), 1)
        self.assertEqual(len(controller.demands["SSD1"]), 1)

    def test_real_qos_gate_promotes_waiter_and_completes_all_requests(self):
        """真实QoS中被门控Queue会在前一GPU排空后同时刻解锁。"""
        qos_config = load_yaml(SIMULATION_CONFIG_FILE)["simulation"]["qos"]
        qos = build_qos_simulator(qos_config=qos_config, start_time_us=0)
        queue_ids = qos.token_stage.queue_order
        binding = build_queue_binding_strategy(
            "balanced_exclusive",
            ["P0", "P1"],
            {"SSD0": queue_ids},
        )
        controller = CoflowPriorityController(
            {"SSD0": SSD_CAPACITY_BYTES_PER_SECOND},
            ordering="fifo",
            selection_width=1,
        )
        gateway = DPURequestGateway(
            {"SSD0": queue_ids},
            binding,
            qos.input,
            {"SSD0": qos},
            controller,
        )
        gateway.submit_batch([
            make_request(
                "selected",
                "P0",
                "SSD0",
                147_456,
                "p0_layer_1",
                1_000_000_000,
            ),
            make_request(
                "waiter",
                "P1",
                "SSD0",
                147_456,
                "p1_layer_1",
                1_000_000_000,
            ),
        ], arrival_time_us=0)

        result = qos.end()
        self.assertEqual(
            [
                request["request_id"]
                for request in result["dispatched_requests"]
            ],
            ["selected", "waiter"],
        )
        self.assertEqual(controller.statistics()["active_demand_count"], 0)
        self.assertEqual(
            controller.statistics()[
                "completed_demand_count_by_storage_target"
            ],
            {"SSD0": 2},
        )
        self.assertEqual(gateway.group_weight_write_count, 0)
        self.assertGreaterEqual(gateway.queue_weight_write_count, 2)

    def test_real_qos_paced_queues_use_only_cir_and_finish_without_residue(self):
        """有限PIR使选中Queue按80 us节拍走CIR，等待Queue轮转前不下发。"""
        qos_config = load_yaml(SIMULATION_CONFIG_FILE)["simulation"]["qos"]
        qos = build_qos_simulator(qos_config=qos_config, start_time_us=0)
        queue_ids = qos.token_stage.queue_order
        binding = build_queue_binding_strategy(
            "balanced_exclusive",
            ["P0", "P1"],
            {"SSD0": queue_ids},
        )
        controller = CoflowPriorityController(
            {"SSD0": SSD_CAPACITY_BYTES_PER_SECOND},
            ordering="fifo",
            selection_width=1,
            finite_selected_pir=True,
        )
        gateway = DPURequestGateway(
            {"SSD0": queue_ids},
            binding,
            qos.input,
            {"SSD0": qos},
            controller,
        )
        submitted = gateway.submit_batch([
            make_request(
                "selected",
                "P0",
                "SSD0",
                147_456,
                "p0_group_0",
                1_000_000_000,
            ),
            make_request(
                "waiter",
                "P1",
                "SSD0",
                147_456,
                "p1_group_0",
                1_000_000_000,
            ),
        ], arrival_time_us=0)
        selected_queue = submitted[0]["queue_id"]
        waiter_queue = submitted[1]["queue_id"]

        self.assertEqual(
            controller.queue_pirs["SSD0"][selected_queue],
            SSD_CAPACITY_BYTES_PER_SECOND,
        )
        self.assertEqual(controller.queue_pirs["SSD0"][waiter_queue], 0)
        qos.process_at(0)
        self.assertEqual(len(qos.dispatched_requests), 0)

        qos.process_at(80)
        self.assertEqual(
            [request["request_id"] for request in qos.dispatched_requests],
            ["selected"],
        )
        self.assertEqual(qos.queue_io_counts()[waiter_queue], 1)

        result = qos.end()
        self.assertEqual(
            [request["request_id"] for request in result["dispatched_requests"]],
            ["selected", "waiter"],
        )
        self.assertEqual(
            [request["dispatch_time_us"] for request in result["dispatched_requests"]],
            [80, 160],
        )
        self.assertEqual(result["cir_dispatched_request_count"], 2)
        self.assertEqual(result["excess_dispatched_request_count"], 0)
        self.assertTrue(result["completed"])
        self.assertTrue(
            all(count == 0 for count in qos.queue_io_counts().values())
        )
        self.assertEqual(controller.statistics()["active_demand_count"], 0)
        self.assertEqual(gateway.group_weight_write_count, 0)


if __name__ == "__main__":
    unittest.main()
