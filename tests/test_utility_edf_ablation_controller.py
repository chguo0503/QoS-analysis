"""三因子Utility/EDF消融控制器的组合与等价性测试。"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from DPU import (
    DPURequestGateway,
    UtilityEDFAblationController,
    UtilityEDFController,
)
from qos_ssd_simulator import (
    JointSimulation,
    install_summary_only_logs,
    summarize_run,
)
from simulation_common.config_utils import load_yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SIMULATION_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
CAPACITY_BYTES_PER_SECOND = 1_000_000


def register_path(
    controller,
    target,
    queue_id,
    p_node_id,
    group_id,
    byte_count,
    *,
    service_window_us=10,
    arrival_time_us=0,
    deadline_us=10_000,
    is_initial=True,
    request_count=1,
):
    """以1 MB/s容量登记路径，使byte数与服务微秒数相同。"""
    block_size_bytes = (
        byte_count // request_count
        if byte_count % request_count == 0
        else None
    )
    controller.register_demand(
        storage_target_id=target,
        queue_id=queue_id,
        requested_cir_bytes_per_second=CAPACITY_BYTES_PER_SECOND,
        arrival_time_us=arrival_time_us,
        p_node_id=p_node_id,
        demand_group_id=group_id,
        batch_total_bytes=byte_count,
        path_bytes=byte_count,
        path_request_count=request_count,
        block_size_bytes=block_size_bytes,
        service_window_us=service_window_us,
        deadline_us=deadline_us,
        compute_layer_index=None if is_initial else 0,
        prefetch_layer_index=0 if is_initial else 1,
        inference_arrival_time_us=0,
    )


def build_ablation(targets, c, u, e, layer_count=4):
    """创建指定因子的消融控制器。"""
    return UtilityEDFAblationController(
        {
            target: CAPACITY_BYTES_PER_SECOND
            for target in targets
        },
        coordination_enabled=c,
        utility_enabled=u,
        edf_enabled=e,
        compute_layer_count=layer_count,
    )


class AblationFactorTests(unittest.TestCase):
    """验证u/e排序与c协调的正交语义。"""

    def test_u_factor_switches_stage0_between_utility_and_fcfs(self):
        selected = {}
        for utility_enabled in (False, True):
            controller = build_ablation(
                ("SSD0",),
                c=True,
                u=utility_enabled,
                e=False,
            )
            # FCFS选先登记的P0；integer utility的参考值选P1。
            register_path(
                controller,
                "SSD0",
                "q0",
                "P0",
                "P0_initial",
                1,
                service_window_us=1,
            )
            register_path(
                controller,
                "SSD0",
                "q1",
                "P1",
                "P1_initial",
                1,
                service_window_us=2,
            )
            controller.recalculate(
                "SSD0",
                event_time_us=0,
                queue_depths={"q0": 1, "q1": 1},
            )
            selected[utility_enabled] = controller.selected_p_node_id

            selected_queue = "q1" if utility_enabled else "q0"
            waiting_queue = "q0" if utility_enabled else "q1"
            self.assertIsNone(
                controller.queue_pirs["SSD0"][selected_queue]
            )
            self.assertEqual(
                controller.queue_pirs["SSD0"][waiting_queue],
                0,
            )

        self.assertEqual(selected[False], "P0")
        self.assertEqual(selected[True], "P1")

    def test_e_factor_switches_prefix_protection_and_stage0_priority(self):
        selected = {}
        for edf_enabled in (False, True):
            controller = build_ablation(
                ("SSD0",),
                c=True,
                u=False,
                e=edf_enabled,
            )
            register_path(
                controller,
                "SSD0",
                "qi",
                "Pinitial",
                "initial",
                800,
            )
            register_path(
                controller,
                "SSD0",
                "qp",
                "Pprefetch",
                "prefetch",
                1,
                deadline_us=0,
                is_initial=False,
            )
            controller.recalculate(
                "SSD0",
                event_time_us=0,
                queue_depths={"qi": 1, "qp": 1},
            )
            selected[edf_enabled] = controller.selected_p_node_id

        self.assertEqual(selected[False], "Pinitial")
        self.assertEqual(selected[True], "Pprefetch")

    def test_e0_prefetch_order_is_fcfs_when_no_stage0_exists(self):
        controller = build_ablation(
            ("SSD0",),
            c=True,
            u=False,
            e=False,
        )
        register_path(
            controller,
            "SSD0",
            "qlate",
            "Plate",
            "late",
            1,
            arrival_time_us=20,
            deadline_us=0,
            is_initial=False,
        )
        register_path(
            controller,
            "SSD0",
            "qearly",
            "Pearly",
            "early",
            1,
            arrival_time_us=10,
            deadline_us=100,
            is_initial=False,
        )
        controller.recalculate(
            "SSD0",
            event_time_us=20,
            queue_depths={"qlate": 1, "qearly": 1},
        )
        self.assertEqual(controller.selected_p_node_id, "Pearly")

    def test_c0_can_choose_different_local_owners(self):
        controllers = {
            coordination_enabled: build_ablation(
                ("SSD0", "SSD1"),
                c=coordination_enabled,
                u=True,
                e=False,
            )
            for coordination_enabled in (False, True)
        }
        for controller in controllers.values():
            # P0/P1的全局最慢路径都是100 us，c1稳定选P0；
            # 每块SSD本地的短路径则分别属于P0和P1。
            register_path(
                controller, "SSD0", "q0", "P0", "P0_g", 1
            )
            register_path(
                controller, "SSD0", "q1", "P1", "P1_g", 100
            )
            register_path(
                controller, "SSD1", "q0", "P0", "P0_g", 100
            )
            register_path(
                controller, "SSD1", "q1", "P1", "P1_g", 1
            )
            controller.recalculate(
                "SSD0", 0, {"q0": 1, "q1": 1}
            )
            controller.recalculate(
                "SSD1", 0, {"q0": 1, "q1": 1}
            )

        coupled = controllers[True]
        self.assertEqual(coupled.selected_p_node_id, "P0")
        self.assertEqual(coupled.queue_rates["SSD1"]["q0"], 1_000_000)
        self.assertEqual(coupled.queue_rates["SSD1"]["q1"], 0)

        independent = controllers[False]
        self.assertEqual(
            independent.selected_p_node_id_by_storage_target,
            {"SSD0": "P0", "SSD1": "P1"},
        )
        self.assertEqual(independent.queue_rates["SSD1"]["q0"], 0)
        self.assertEqual(
            independent.queue_rates["SSD1"]["q1"],
            1_000_000,
        )

    def test_c0_started_lock_is_local_to_one_ssd(self):
        controller = build_ablation(
            ("SSD0", "SSD1"),
            c=False,
            u=True,
            e=False,
        )
        for target in ("SSD0", "SSD1"):
            register_path(
                controller,
                target,
                "q0",
                "P0",
                "P0_g",
                100,
                request_count=10,
            )
            controller.recalculate(target, 0, {"q0": 10})
        for target in ("SSD0", "SSD1"):
            register_path(
                controller,
                target,
                "q1",
                "P1",
                "P1_g",
                1,
                service_window_us=100,
            )

        controller.recalculate(
            "SSD0", 1, {"q0": 5, "q1": 1}
        )
        controller.recalculate(
            "SSD1", 1, {"q0": 10, "q1": 1}
        )
        self.assertEqual(
            controller.selected_p_node_id_by_storage_target,
            {"SSD0": "P0", "SSD1": "P1"},
        )
        self.assertEqual(
            controller.owner_locked_by_storage_target,
            {"SSD0": True, "SSD1": False},
        )

    def test_c0_empty_callback_does_not_unlock_another_ssd(self):
        controller = build_ablation(
            ("SSD0", "SSD1"),
            c=False,
            u=True,
            e=True,
            layer_count=1,
        )
        for target in ("SSD0", "SSD1"):
            register_path(
                controller,
                target,
                "q0",
                "P0",
                "P0_g",
                100,
                request_count=10,
            )
            controller.recalculate(target, 0, {"q0": 10})
            controller.recalculate(target, 1, {"q0": 5})
        self.assertEqual(
            controller.owner_locked_by_storage_target,
            {"SSD0": True, "SSD1": True},
        )

        # SSD0的本地路径和唯一读组结束，只解锁/恢复SSD0。
        controller.release_empty_demands(
            "SSD0",
            {"q0": 0},
            event_time_us=2,
        )
        self.assertEqual(
            controller.selected_p_node_id_by_storage_target,
            {"SSD0": None, "SSD1": "P0"},
        )
        self.assertEqual(
            controller.owner_locked_by_storage_target,
            {"SSD0": False, "SSD1": True},
        )
        self.assertNotIn(
            "q0",
            controller._programmed_queue_states["SSD0"],
        )
        self.assertEqual(
            controller._programmed_queue_states["SSD1"]["q0"],
            (CAPACITY_BYTES_PER_SECOND, None, 1),
        )


class AblationEquivalenceTests(unittest.TestCase):
    """验证完整组合与原控制器的强等价契约。"""

    @staticmethod
    def _register_one_ssd_mix(controller):
        register_path(
            controller,
            "SSD0",
            "q0",
            "P0",
            "P0_initial",
            800,
            service_window_us=1,
        )
        register_path(
            controller,
            "SSD0",
            "q1",
            "P1",
            "P1_initial",
            1,
            service_window_us=100,
            arrival_time_us=1,
        )
        register_path(
            controller,
            "SSD0",
            "q2",
            "P2",
            "P2_prefetch",
            1,
            deadline_us=0,
            is_initial=False,
            arrival_time_us=2,
        )

    def test_c0_and_c1_are_decision_equivalent_with_one_ssd(self):
        for utility_enabled in (False, True):
            for edf_enabled in (False, True):
                with self.subTest(u=utility_enabled, e=edf_enabled):
                    local = build_ablation(
                        ("SSD0",),
                        c=False,
                        u=utility_enabled,
                        e=edf_enabled,
                    )
                    coupled = build_ablation(
                        ("SSD0",),
                        c=True,
                        u=utility_enabled,
                        e=edf_enabled,
                    )
                    for controller in (local, coupled):
                        self._register_one_ssd_mix(controller)

                    depths = {"q0": 1, "q1": 1, "q2": 1}
                    local_updates = local.recalculate("SSD0", 2, depths)
                    coupled_updates = coupled.recalculate("SSD0", 2, depths)
                    self.assertEqual(local_updates, coupled_updates)
                    self.assertEqual(
                        local.selected_p_node_id,
                        coupled.selected_p_node_id,
                    )
                    self.assertEqual(
                        local.decision_history,
                        coupled.decision_history,
                    )

                    selected_queue = {
                        "P0": "q0",
                        "P1": "q1",
                        "P2": "q2",
                    }[local.selected_p_node_id]
                    empty_depths = dict(depths)
                    empty_depths[selected_queue] = 0
                    local_updates = local.release_empty_demands(
                        "SSD0", empty_depths, event_time_us=3
                    )
                    coupled_updates = coupled.release_empty_demands(
                        "SSD0", empty_depths, event_time_us=3
                    )
                    self.assertEqual(local_updates, coupled_updates)
                    self.assertEqual(
                        local.selected_p_node_id,
                        coupled.selected_p_node_id,
                    )
                    self.assertEqual(
                        local.decision_history,
                        coupled.decision_history,
                    )
                    self.assertEqual(
                        local._programmed_queue_states,
                        coupled._programmed_queue_states,
                    )

    def test_c1u1e1_matches_existing_controller_decision_by_decision(self):
        reference = UtilityEDFController(
            {
                "SSD0": CAPACITY_BYTES_PER_SECOND,
                "SSD1": CAPACITY_BYTES_PER_SECOND,
            },
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=4,
            finite_selected_pir=False,
        )
        ablation = build_ablation(
            ("SSD0", "SSD1"),
            c=True,
            u=True,
            e=True,
        )
        for controller in (reference, ablation):
            for target, slow_bytes, fast_bytes in (
                ("SSD0", 20, 2),
                ("SSD1", 30, 2),
            ):
                register_path(
                    controller,
                    target,
                    "q0",
                    "Pslow",
                    "slow_initial",
                    slow_bytes,
                    service_window_us=1,
                    request_count=2,
                )
                register_path(
                    controller,
                    target,
                    "q1",
                    "Pfast",
                    "fast_initial",
                    fast_bytes,
                    service_window_us=100,
                    request_count=2,
                )
                register_path(
                    controller,
                    target,
                    "q2",
                    "Pprefetch",
                    "prefetch",
                    2,
                    deadline_us=1_000,
                    is_initial=False,
                    request_count=2,
                )

        events = [
            ("recalculate", "SSD0", 0, {"q0": 2, "q1": 2, "q2": 2}),
            ("recalculate", "SSD1", 0, {"q0": 2, "q1": 2, "q2": 2}),
            ("recalculate", "SSD0", 1, {"q0": 2, "q1": 1, "q2": 2}),
            ("release", "SSD0", 2, {"q0": 2, "q1": 0, "q2": 2}),
            ("release", "SSD1", 3, {"q0": 2, "q1": 0, "q2": 2}),
        ]
        for operation, target, event_time_us, depths in events:
            with self.subTest(operation=operation, target=target):
                if operation == "recalculate":
                    reference_updates = reference.recalculate(
                        target, event_time_us, depths
                    )
                    ablation_updates = ablation.recalculate(
                        target, event_time_us, depths
                    )
                else:
                    reference_updates = reference.release_empty_demands(
                        target, depths, event_time_us=event_time_us
                    )
                    ablation_updates = ablation.release_empty_demands(
                        target, depths, event_time_us=event_time_us
                    )
                self.assertEqual(ablation_updates, reference_updates)
                self.assertEqual(
                    ablation.selected_p_node_id,
                    reference.selected_p_node_id,
                )
                self.assertEqual(
                    ablation.owner_locked,
                    reference.owner_locked,
                )
                self.assertEqual(
                    ablation.decision_history,
                    reference.decision_history,
                )
                self.assertEqual(
                    ablation._programmed_queue_states,
                    reference._programmed_queue_states,
                )

    def test_c1u1e1_small_simulation_matches_existing_strategy(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 2
        config["workload_generation"].update({
            "inference_count_per_gpu": 1,
            "input_tokens_range": [128, 128],
            "prefill_layer_hit_ratio_range": [0.5, 0.5],
            "unique_across_gpus": False,
        })
        config["workload"]["first_layer_index"] = 0
        config["workload"]["last_layer_index"] = 1

        reference = JointSimulation(
            config=config,
            rate_control_strategy_name="utility_edf_integer_l750",
        ).run()
        ablation = JointSimulation(
            config=config,
            rate_control_strategy_name="ablation_c1_u1_e1",
        ).run()

        self.assertEqual(ablation["gpus"], reference["gpus"])
        self.assertEqual(
            ablation["storage_paths"],
            reference["storage_paths"],
        )
        self.assertEqual(
            ablation["request_conservation"],
            reference["request_conservation"],
        )
        self.assertEqual(ablation["event_loop"], reference["event_loop"])

        reference_rate_control = deepcopy(
            reference["dpu"]["rate_control"]
        )
        ablation_rate_control = deepcopy(
            ablation["dpu"]["rate_control"]
        )
        reference_rate_control.pop("strategy")
        ablation_rate_control.pop("strategy")
        ablation_rate_control.pop("coordination_enabled")
        ablation_rate_control.pop("utility_enabled")
        ablation_rate_control.pop("edf_enabled")
        self.assertEqual(ablation_rate_control, reference_rate_control)

    def test_c0_and_c1_one_ssd_small_simulations_match_all_metrics(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        config["workload_generation"].update({
            "inference_count_per_gpu": 1,
            "input_tokens_range": [128, 128],
            "prefill_layer_hit_ratio_range": [0.5, 0.5],
            "unique_across_gpus": False,
        })
        config["workload"]["first_layer_index"] = 0
        config["workload"]["last_layer_index"] = 1

        for utility_enabled in (False, True):
            for edf_enabled in (False, True):
                with self.subTest(u=utility_enabled, e=edf_enabled):
                    results = {}
                    for coordination_enabled in (False, True):
                        strategy_name = (
                            f"ablation_c{int(coordination_enabled)}"
                            f"_u{int(utility_enabled)}"
                            f"_e{int(edf_enabled)}"
                        )
                        results[coordination_enabled] = JointSimulation(
                            config=config,
                            rate_control_strategy_name=strategy_name,
                        ).run()

                    local = results[False]
                    coupled = results[True]
                    for result_key in (
                        "gpus",
                        "storage_paths",
                        "request_conservation",
                        "event_loop",
                    ):
                        self.assertEqual(
                            local[result_key],
                            coupled[result_key],
                        )

                    local_dpu = deepcopy(local["dpu"])
                    coupled_dpu = deepcopy(coupled["dpu"])
                    local_rate_control = local_dpu["rate_control"]
                    coupled_rate_control = coupled_dpu["rate_control"]
                    local_history = local_rate_control["decision_history"]
                    coupled_history = coupled_rate_control[
                        "decision_history"
                    ]
                    # c1会在无IO的末层batch边界做全局idle重算；
                    # c0没有affected target时不做这些no-op。所有真正
                    # Demand/Queue决策的内容和时钟必须完全相同。
                    self.assertEqual(
                        local_history,
                        coupled_history[:len(local_history)],
                    )
                    for idle_decision in coupled_history[
                        len(local_history):
                    ]:
                        self.assertIsNone(
                            idle_decision["selected_p_node_id"]
                        )
                        self.assertEqual(idle_decision["reason"], "idle")
                    for rate_control in (
                        local_rate_control,
                        coupled_rate_control,
                    ):
                        rate_control.pop("strategy")
                        rate_control.pop("coordination_enabled")
                        rate_control.pop("utility_enabled")
                        rate_control.pop("edf_enabled")
                        rate_control.pop("decision_count")
                        rate_control.pop("decision_history")
                    for local_only_key in (
                        "selected_p_node_id_by_storage_target",
                        "owner_locked_by_storage_target",
                        "completed_path_group_count_by_storage_target",
                    ):
                        local_rate_control.pop(local_only_key)
                    self.assertEqual(local_dpu, coupled_dpu)


class _ReleaseClockQoS:
    """仅实现Gateway Queue状态回调需要的QoS接口。"""

    start_time_us = 0

    def __init__(self):
        self.token_stage = SimpleNamespace(update_period_us=80)

    def set_queue_state_observer(self, observer):
        self.observer = observer

    def queue_io_counts(self):
        return {"q0": 0}


class _ReleaseClockBinding:
    """不参与数据面的最小Queue binding。"""

    strategy_name = "release_clock_test"
    bindings = {}


class _ReleaseClockController:
    """记录release时钟并拒绝任何跨SSD recalculate。"""

    coordinates_storage_targets = False
    uses_event_time_for_release = True

    def __init__(self):
        self.capacity = {"SSD0": 1, "SSD1": 1}
        self.release_calls = []
        self.recalculate_calls = []

    def release_empty_demands(
        self,
        storage_target_id,
        queue_depths,
        event_time_us=None,
    ):
        self.release_calls.append((
            storage_target_id,
            dict(queue_depths),
            event_time_us,
        ))
        return {
            "queue_rates": {},
            "queue_pirs": {},
            "queue_weights": {},
            "group_weights": None,
            # 即使返回True，coordinates=False也绝不得跨盘重算。
            "coordinates_changed": True,
        }

    def recalculate(self, *args, **kwargs):
        self.recalculate_calls.append((args, kwargs))
        raise AssertionError("local release must not recalculate another SSD")


class ReleaseEventCapabilityTests(unittest.TestCase):
    """验证release时钟传递与跨SSD协调是两个独立capability。"""

    def test_local_controller_gets_real_clock_without_cross_ssd_recalculate(self):
        controller = _ReleaseClockController()
        qos_by_target = {
            target: _ReleaseClockQoS()
            for target in ("SSD0", "SSD1")
        }
        gateway = DPURequestGateway(
            {
                "SSD0": ["q0"],
                "SSD1": ["q0"],
            },
            _ReleaseClockBinding(),
            lambda request: request,
            qos_by_target,
            controller,
        )

        gateway.on_qos_queue_state_change("SSD0", event_time_us=137)

        self.assertEqual(
            controller.release_calls,
            [("SSD0", {"q0": 0}, 137)],
        )
        self.assertEqual(controller.recalculate_calls, [])


class AblationParserTests(unittest.TestCase):
    """验证八个策略名与固定硬件契约。"""

    def test_parser_accepts_exactly_all_eight_factor_combinations(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 1
        config["topology"]["storage_path_count"] = 1
        for coordination_enabled in (False, True):
            for utility_enabled in (False, True):
                for edf_enabled in (False, True):
                    strategy_name = (
                        f"ablation_c{int(coordination_enabled)}"
                        f"_u{int(utility_enabled)}"
                        f"_e{int(edf_enabled)}"
                    )
                    with self.subTest(strategy_name=strategy_name):
                        simulation = JointSimulation(
                            config=config,
                            rate_control_strategy_name=strategy_name,
                        )
                        controller = simulation.dpu.rate_controller
                        self.assertIsInstance(
                            controller,
                            UtilityEDFAblationController,
                        )
                        self.assertEqual(
                            controller.coordination_enabled,
                            coordination_enabled,
                        )
                        self.assertEqual(
                            controller.utility_enabled,
                            utility_enabled,
                        )
                        self.assertEqual(
                            controller.edf_enabled,
                            edf_enabled,
                        )
                        self.assertEqual(
                            controller.coordinates_storage_targets,
                            coordination_enabled,
                        )
                        self.assertEqual(
                            controller.deadline_allowance_us,
                            750,
                        )
                        self.assertEqual(controller.score_mode, "integer")
                        self.assertFalse(controller.finite_selected_pir)
                        self.assertTrue(
                            controller.requires_tick_aligned_control
                        )
                        self.assertTrue(
                            controller.strict_control_update_grid
                        )

    def test_parser_rejects_non_binary_or_noncanonical_names(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 1
        config["topology"]["storage_path_count"] = 1
        for invalid_name in (
            "ablation",
            "ablation_c2_u1_e1",
            "ablation_c1_u1_e1_extra",
            "ablation_c1u1e1",
        ):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(ValueError):
                    JointSimulation(
                        config=config,
                        rate_control_strategy_name=invalid_name,
                    )

    def test_compact_summary_keeps_factors_without_local_runtime_state(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 1
        config["topology"]["storage_path_count"] = 1
        config["workload_generation"].update({
            "inference_count_per_gpu": 1,
            "input_tokens_range": [128, 128],
            "prefill_layer_hit_ratio_range": [0.5, 0.5],
            "unique_across_gpus": False,
        })
        config["workload"]["first_layer_index"] = 0
        config["workload"]["last_layer_index"] = 0
        simulation = JointSimulation(
            config=config,
            rate_control_strategy_name="ablation_c0_u1_e0",
        )
        dispatch_logs = install_summary_only_logs(simulation)
        simulation.event_loop.run_until(simulation._all_gpus_complete)
        summary = summarize_run(
            simulation,
            dispatch_logs,
            wall_time_seconds=0,
        )

        rate_control = summary["rate_control"]
        self.assertEqual(
            {
                key: rate_control[key]
                for key in (
                    "coordination_enabled",
                    "utility_enabled",
                    "edf_enabled",
                )
            },
            {
                "coordination_enabled": False,
                "utility_enabled": True,
                "edf_enabled": False,
            },
        )
        self.assertNotIn(
            "selected_p_node_id_by_storage_target",
            rate_control,
        )
        self.assertNotIn("owner_locked_by_storage_target", rate_control)


if __name__ == "__main__":
    unittest.main()
