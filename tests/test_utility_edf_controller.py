"""Utility+EDF DPU控制器的整数参考与接线测试。"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from DPU import (
    DPURequestGateway,
    UtilityEDFController,
)
from qos_ssd_simulator import JointSimulation
from simulation_common.config_utils import load_yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SIMULATION_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
CAPACITY_BYTES_PER_SECOND = 1_000_000


def register_demand(
    controller,
    queue_id,
    p_node_id,
    service_us,
    byte_count,
    *,
    deadline_us=None,
    compute_layer_index=None,
    arrival_time_us=0,
    request_count=1,
    block_size_bytes=None,
    inference_arrival_time_us=0,
    demand_group_id=None,
):
    """以1 MB/s容量登记，使Byte数数值等于服务微秒。"""
    if block_size_bytes is None and request_count == 1:
        block_size_bytes = byte_count
    controller.register_demand(
        storage_target_id="SSD0",
        queue_id=queue_id,
        requested_cir_bytes_per_second=CAPACITY_BYTES_PER_SECOND,
        arrival_time_us=arrival_time_us,
        p_node_id=p_node_id,
        demand_group_id=(
            f"{p_node_id}_group"
            if demand_group_id is None
            else demand_group_id
        ),
        batch_total_bytes=byte_count,
        path_bytes=byte_count,
        path_request_count=request_count,
        block_size_bytes=block_size_bytes,
        service_window_us=service_us,
        deadline_us=(
            arrival_time_us + service_us
            if deadline_us is None
            else deadline_us
        ),
        compute_layer_index=compute_layer_index,
        prefetch_layer_index=(
            0 if compute_layer_index is None else compute_layer_index + 1
        ),
        inference_arrival_time_us=inference_arrival_time_us,
    )


class UtilityEDFReferenceTests(unittest.TestCase):
    """验证价值密度精确比较和EDF插入边界。"""

    def build_controller(self, mode="integer", allowance_us=2):
        return UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode=mode,
            deadline_allowance_us=allowance_us,
            compute_layer_count=4,
        )

    def test_integer_and_power_exact_comparisons_choose_different_reference(self):
        """(c,b)=(1,1)/(2,1)使两种规定公式产生相反顺序。"""
        selected_by_mode = {}
        for mode in ("integer", "power"):
            controller = self.build_controller(mode=mode)
            register_demand(controller, "q0", "P0", 1, 1)
            register_demand(controller, "q1", "P1", 2, 1)
            controller.recalculate(
                "SSD0",
                event_time_us=0,
                queue_depths={"q0": 1, "q1": 1},
            )
            selected_by_mode[mode] = controller.selected_p_node_id

        # integer: 1/(5*1^2) < 2/(9*1^2)，选P1。
        self.assertEqual(selected_by_mode["integer"], "P1")
        # power用规定的10/13/12次整数交叉乘法，选P0。
        self.assertEqual(selected_by_mode["power"], "P0")

    def test_edf_conflict_and_exact_allowance_boundary(self):
        """累计服务大于deadline+L才冲突，等于时可插stage0。"""
        conflict = self.build_controller(allowance_us=2)
        register_demand(conflict, "qi", "PI", 10, 5)
        register_demand(
            conflict,
            "qp",
            "PP",
            10,
            4,
            deadline_us=6,
            compute_layer_index=0,
        )
        updates = conflict.recalculate(
            "SSD0",
            event_time_us=0,
            queue_depths={"qi": 1, "qp": 1},
        )
        self.assertEqual(conflict.selected_p_node_id, "PP")
        self.assertEqual(updates["queue_rates"]["qp"], CAPACITY_BYTES_PER_SECOND)
        self.assertEqual(updates["queue_pirs"], {"qi": 0})
        self.assertEqual(conflict.feasibility_conflict_count, 1)

        boundary = self.build_controller(allowance_us=2)
        register_demand(boundary, "qi", "PI", 10, 5)
        register_demand(
            boundary,
            "qp",
            "PP",
            10,
            4,
            deadline_us=7,
            compute_layer_index=0,
        )
        boundary.recalculate(
            "SSD0",
            event_time_us=0,
            queue_depths={"qi": 1, "qp": 1},
        )
        self.assertEqual(boundary.selected_p_node_id, "PI")
        self.assertEqual(boundary.feasibility_conflict_count, 0)

    def test_depth_scales_remaining_bytes_and_incomplete_path_is_parked(self):
        """新arrival用完整depth重算；未完成四读组的路径保持park。"""
        controller = self.build_controller()
        register_demand(
            controller,
            "q0",
            "P0",
            100,
            100,
            request_count=10,
            block_size_bytes=10,
        )
        controller.recalculate(
            "SSD0",
            event_time_us=20,
            queue_depths={"q0": 4},
        )
        self.assertEqual(
            controller.decision_history[-1]["remaining_service_us"],
            40,
        )

        updates = controller.release_empty_demands(
            "SSD0",
            {"q0": 0},
            event_time_us=60,
        )
        self.assertEqual(updates["queue_rates"], {"q0": 0})
        self.assertEqual(controller.selected_p_node_id, None)
        self.assertEqual(
            controller._programmed_queue_states["SSD0"],
            {"q0": (0, 0, 0)},
        )
        self.assertEqual(controller.statistics()["active_demand_count"], 0)
        self.assertEqual(
            controller.statistics()["p_node_statistics"]["P0"][
                "completed_coflow_count"
            ],
            1,
        )

    def test_optional_finite_selected_pir_is_supported(self):
        """默认uncapped避免交接冷启，显式模式可限制PIR。"""
        uncapped = self.build_controller()
        register_demand(uncapped, "q0", "P0", 10, 1)
        uncapped.recalculate(
            "SSD0", event_time_us=0, queue_depths={"q0": 1}
        )
        self.assertIsNone(uncapped.queue_pirs["SSD0"]["q0"])

        finite = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            deadline_allowance_us=2,
            finite_selected_pir=True,
        )
        register_demand(finite, "q0", "P0", 10, 1)
        finite.recalculate(
            "SSD0", event_time_us=0, queue_depths={"q0": 1}
        )
        self.assertEqual(
            finite.queue_pirs["SSD0"]["q0"],
            CAPACITY_BYTES_PER_SECOND,
        )

    def test_same_p_node_can_complete_two_consecutive_inferences(self):
        """新Layer 0重新park固定路径，每次4层后恢复。"""
        controller = self.build_controller()
        controller.prepark_all_queues(
            {"SSD0": ["q0"]},
            {"SSD0": {"q0": "P0"}},
        )

        for inference_index, inference_arrival_us in enumerate((10, 1_000)):
            for layer_index in range(4):
                arrival_time_us = inference_arrival_us + layer_index * 10
                register_demand(
                    controller,
                    "q0",
                    "P0",
                    10,
                    10,
                    arrival_time_us=arrival_time_us,
                    compute_layer_index=(
                        None if layer_index == 0 else layer_index - 1
                    ),
                    inference_arrival_time_us=inference_arrival_us,
                    demand_group_id=(
                        f"inference_{inference_index}_layer_{layer_index}"
                    ),
                )
                if layer_index == 0:
                    self.assertNotIn(
                        ("SSD0", "q0"),
                        controller.restored_queue_paths,
                    )
                    self.assertEqual(
                        controller.current_inference_completed_layer_count_by_p_node[
                            "P0"
                        ],
                        0,
                    )
                    self.assertEqual(
                        controller._desired_states("SSD0", None)["q0"],
                        controller._PARKED_QUEUE_STATE,
                    )

                controller.recalculate(
                    "SSD0",
                    event_time_us=arrival_time_us,
                    queue_depths={"q0": 1},
                )
                controller.release_empty_demands(
                    "SSD0",
                    {"q0": 0},
                    event_time_us=arrival_time_us + 1,
                )

            self.assertIn(("SSD0", "q0"), controller.restored_queue_paths)
            self.assertEqual(
                controller.current_inference_completed_layer_count_by_p_node[
                    "P0"
                ],
                4,
            )

        self.assertEqual(controller.completed_layer_count_by_p_node["P0"], 8)
        self.assertEqual(
            controller.completed_coflow_count_by_p_node["P0"],
            8,
        )
        statistics = controller.statistics()
        self.assertEqual(statistics["completed_layer_count"], 8)
        self.assertEqual(
            statistics["current_inference_arrival_time_us_by_p_node"],
            {"P0": 1_000},
        )

    def test_started_owner_is_not_preempted_by_new_arrival(self):
        """未开始时可换更优候选；depth减小后锁定到Queue-empty。"""
        controller = self.build_controller()
        register_demand(
            controller,
            "q0",
            "Pslow",
            10,
            100,
            request_count=10,
            block_size_bytes=10,
        )
        controller.recalculate(
            "SSD0",
            event_time_us=0,
            queue_depths={"q0": 10},
        )
        self.assertEqual(controller.selected_p_node_id, "Pslow")

        # 尚未下发，同时后登记的高价值密度候选可替换。
        register_demand(controller, "q1", "Pfast", 100, 1)
        controller.recalculate(
            "SSD0",
            event_time_us=0,
            queue_depths={"q0": 10, "q1": 1},
        )
        self.assertEqual(controller.selected_p_node_id, "Pfast")
        self.assertFalse(controller.owner_locked)

        # Pfast已消耗部分depth后，新来的更优Pbest也不抢占。
        register_demand(controller, "q2", "Pbest", 1_000, 1)
        controller.recalculate(
            "SSD0",
            event_time_us=1,
            queue_depths={"q0": 10, "q1": 0, "q2": 1},
        )
        # q1一个IO已全部离开Queue，应由empty回调轮换。
        self.assertEqual(controller.selected_p_node_id, "Pbest")

        locked = self.build_controller()
        register_demand(
            locked,
            "q0",
            "Powner",
            10,
            100,
            request_count=10,
            block_size_bytes=10,
        )
        locked.recalculate(
            "SSD0", event_time_us=0, queue_depths={"q0": 10}
        )
        register_demand(locked, "q1", "Pbetter", 1_000, 1)
        locked.recalculate(
            "SSD0",
            event_time_us=1,
            queue_depths={"q0": 5, "q1": 1},
        )
        self.assertEqual(locked.selected_p_node_id, "Powner")
        self.assertTrue(locked.owner_locked)
        self.assertEqual(
            locked.decision_history[-1]["reason"],
            "owner_locked",
        )

        locked.release_empty_demands(
            "SSD0",
            {"q0": 0, "q1": 1},
            event_time_us=2,
        )
        self.assertEqual(locked.selected_p_node_id, "Pbetter")
        self.assertFalse(locked.owner_locked)


class _FixedBinding:
    strategy_name = "fixed"

    def select_queue(self, request, queue_ids):
        return queue_ids[0]


class _FakeQoS:
    def __init__(self):
        self.token_stage = SimpleNamespace(update_period_us=80)
        self.depths = {"q0": 0}
        self.rate_updates = []
        self.weight_updates = []

    def set_queue_state_observer(self, observer):
        self.observer = observer

    def input(self, request):
        self.depths[request["queue_id"]] += 1

    def queue_io_counts(self):
        return dict(self.depths)

    def schedule_queue_rate_update(self, queue_id, cir, pir, effective):
        self.rate_updates.append((queue_id, cir, pir, effective))

    def schedule_queue_weight_update(self, weights, effective):
        self.weight_updates.append((dict(weights), effective))

    def schedule_group_weight_update(self, weights, effective):
        raise AssertionError("UtilityEDF must not write Group WRR")


class UtilityEDFIntegrationTests(unittest.TestCase):
    """验证parser与DPU完整元数据接线。"""

    def test_dispatcher_passes_layer_and_remaining_depth_metadata(self):
        qos = _FakeQoS()
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            deadline_allowance_us=1_000,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q0"]},
            _FixedBinding(),
            qos.input,
            {"SSD0": qos},
            controller,
        )
        requests = []
        for index in range(3):
            requests.append({
                "basic": {
                    "request_id": f"r{index}",
                    "p_node_id": "P0",
                    "storage_target_id": "SSD0",
                    "size_bytes": 10,
                },
                "demand_bw": {
                    "demand_group_id": "P0_layer0",
                    "compute_layer_index": None,
                    "prefetch_layer_index": 0,
                    "inference_arrival_time_us": 7,
                    "service_window_us": 100,
                    "aggregate_bytes_on_storage_target": 30,
                    "aggregate_required_bytes_per_second": 300_000,
                },
            })

        gateway.submit_batch(requests, arrival_time_us=7)
        demand = controller.demands["SSD0"]["q0"]
        self.assertEqual(demand["path_request_count"], 3)
        self.assertEqual(demand["block_size_bytes"], 10)
        self.assertIsNone(demand["compute_layer_index"])
        self.assertEqual(demand["prefetch_layer_index"], 0)
        self.assertEqual(demand["inference_arrival_time_us"], 7)
        self.assertEqual(controller.queue_depths["SSD0"]["q0"], 3)
        self.assertEqual(gateway.group_weight_write_count, 0)

    def test_parser_supports_both_modes_and_strict_positive_allowance(self):
        config = deepcopy(load_yaml(SIMULATION_CONFIG_FILE)["simulation"])
        config["topology"]["gpu_count"] = 2
        config["topology"]["storage_path_count"] = 1
        for strategy_name, mode, allowance in (
            ("utility_edf_integer_l750", "integer", 750),
            ("utility_edf_power_l2000", "power", 2_000),
        ):
            with self.subTest(strategy_name=strategy_name):
                simulation = JointSimulation(
                    config=config,
                    rate_control_strategy_name=strategy_name,
                )
                controller = simulation.dpu.rate_controller
                self.assertEqual(controller.score_mode, mode)
                self.assertEqual(
                    controller.deadline_allowance_us,
                    allowance,
                )
                self.assertEqual(controller.compute_layer_count, 4)
                self.assertFalse(controller.finite_selected_pir)

        for invalid_name in (
            "utility_edf_float_l1000",
            "utility_edf_integer_l0",
            "utility_edf_power_l01000",
        ):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(ValueError):
                    JointSimulation(
                        config=config,
                        rate_control_strategy_name=invalid_name,
                    )


if __name__ == "__main__":
    unittest.main()
