"""需求感知Queue速率、Group WRR权重和多GPU/SSD联合仿真测试。"""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from DPU import (
    DPURequestGateway,
    DemandAwareRateController,
    build_queue_binding_strategy,
)
from qos import build_qos_simulator
from qos.schedulers import SmoothWeightedRoundRobinScheduler
import qos_ssd_simulator


PROJECT_DIR = Path(__file__).resolve().parents[1]
QOS_CONFIG_DIR = PROJECT_DIR / "qos" / "config"


def build_qos():
    """功能：使用项目YAML创建一套纯QoS。

    目的：在不启动SSD流水线的情况下验证DPU状态传递和设置。
    输入：无。输出：未连接SSD的QoS仿真器。
    """
    return build_qos_simulator(
        QOS_CONFIG_DIR / "queue_layout_config.yaml",
        QOS_CONFIG_DIR / "token_bucket_config.yaml",
        QOS_CONFIG_DIR / "wrr_config.yaml",
        QOS_CONFIG_DIR / "qos_runtime_config.yaml",
        0,
    )


def flat_request(request_id, node, group, queue, rate):
    """功能：构造需求控制器使用的展平请求。

    目的：用最小字段集独立测试速率和权重计算。
    输入：请求ID、P节点、需求组、Queue和整数Byte/s。
    输出：DPU展平请求字典。
    """
    return {
        "request_id": request_id,
        "p_node_id": node,
        "storage_target_id": "SSD0",
        "size_bytes": 147_456,
        "demand_group_id": group,
        "aggregate_required_bytes_per_second": rate,
        "queue_id": queue,
        "arrival_time_us": 0,
        "qos_admitted": False,
    }


def dpu_request(request_id, node, group, rate):
    """功能：构造DPU入口使用的KV请求。

    目的：验证互斥绑定、Queue登记和控制事件的完整链路。
    输入：请求ID、P节点、需求组和整数Byte/s。
    输出：带 ``basic`` 和 ``demand_bw`` 的DPU请求。
    """
    return {
        "basic": {
            "request_id": request_id,
            "p_node_id": node,
            "storage_target_id": "SSD0",
            "size_bytes": 147_456,
        },
        "demand_bw": {
            "demand_group_id": group,
            "aggregate_required_bytes_per_second": rate,
        },
    }


class QueueBindingTests(unittest.TestCase):
    """验证随机互斥绑定在实验前完成且可复现。"""

    def test_unique_sticky_binding_has_no_collisions(self):
        """功能：为10个P节点在两块SSD上预绑定Queue。

        目的：断言每SSD的10个Queue互不重复，且重建策略结果相同。
        输入：无。输出：通过unittest断言报告结果。
        """
        p_nodes = [f"P{index}" for index in range(10)]
        queues = {
            target: [f"q{index:03d}" for index in range(256)]
            for target in ("SSD0", "SSD1")
        }
        first = build_queue_binding_strategy(
            "random_unique_sticky",
            5102,
            p_nodes,
            queues,
        )
        second = build_queue_binding_strategy(
            "random_unique_sticky",
            5102,
            p_nodes,
            queues,
        )
        self.assertEqual(first.bindings, second.bindings)
        for target in queues:
            selected = [first.bindings[(node, target)] for node in p_nodes]
            self.assertEqual(len(selected), len(set(selected)))


class DemandAwareControllerTests(unittest.TestCase):
    """验证Queue速率、Group权重和过载等待规则。"""

    def build_controller(self, capacity=40):
        """功能：创建只含三个Queue的需求感知控制器。

        目的：用q000所在g0和q032/q033所在g1验证组级聚合。
        输入：SSD容量。输出：空的DemandAwareRateController。
        """
        return DemandAwareRateController(
            {"SSD0": capacity},
            {"SSD0": {"q000": "g0", "q032": "g1", "q033": "g1"}},
        )

    def test_queue_rates_and_group_weights_have_separate_roles(self):
        """功能：登记三份4 Byte/s需求并计算控制量。

        目的：断言每Queue速率为4，g0/g1调度权重分别为4/8。
        输入：无。输出：通过unittest断言报告结果。
        """
        controller = self.build_controller()
        requests = [
            flat_request("r0", "P0", "d0", "q000", 4),
            flat_request("r1", "P1", "d1", "q032", 4),
            flat_request("r2", "P2", "d2", "q033", 4),
        ]
        for request in requests:
            controller.register(request)
        updates = controller.update("SSD0")
        self.assertEqual(
            updates["queue_rates"],
            {"q000": 4, "q032": 4, "q033": 4},
        )
        self.assertEqual(updates["group_weights"], {"g0": 4, "g1": 8})

    def test_later_demand_waits_without_scaling_active_rates(self):
        """功能：用30+20需求测试40容量的整项准入。

        目的：断言先到需求保持30，后到需求等待释放后获得20。
        输入：无。输出：通过unittest断言报告结果。
        """
        controller = self.build_controller()
        first = flat_request("r0", "P0", "d0", "q000", 30)
        second = flat_request("r1", "P1", "d1", "q032", 20)
        controller.register(first)
        controller.register(second)

        updates = controller.update("SSD0")
        self.assertEqual(updates["queue_rates"], {"q000": 30})
        self.assertTrue(first["qos_admitted"])
        self.assertFalse(second["qos_admitted"])

        updates = controller.dispatched([first])
        self.assertEqual(updates["queue_rates"], {"q000": 0, "q032": 20})
        self.assertTrue(second["qos_admitted"])
        self.assertEqual(controller.reserved["SSD0"], 20)

    def test_waiting_io_stays_in_real_qos_queue(self):
        """功能：经过真实QoS运行30+20 GB/s的两个IO。

        目的：断言第二个IO在第一个离队后才获得速率并下发。
        输入：无。输出：通过unittest断言报告结果。
        """
        qos = build_qos()
        queues_by_target = {"SSD0": qos.token_stage.queue_order}
        controller = DemandAwareRateController(
            {"SSD0": 40_000_000_000},
            {"SSD0": qos.queue_layout.queue_to_group},
        )
        binding = build_queue_binding_strategy(
            "random_unique_sticky",
            7,
            ["P0", "P1"],
            queues_by_target,
        )
        dpu = DPURequestGateway(
            queues_by_target,
            binding,
            qos.input,
            {"SSD0": qos},
            controller,
        )
        submitted = dpu.submit_batch([
            dpu_request("r0", "P0", "d0", 30_000_000_000),
            dpu_request("r1", "P1", "d1", 20_000_000_000),
        ], 0)
        self.assertTrue(submitted[0]["qos_admitted"])
        self.assertFalse(submitted[1]["qos_admitted"])

        result = qos.end()
        dispatch_times = {
            request["request_id"]: request["dispatch_time_us"]
            for request in result["dispatched_requests"]
        }
        self.assertLess(dispatch_times["r0"], dispatch_times["r1"])
        self.assertFalse(any(result["queue_io_counts"].values()))
        self.assertEqual(result["group_weight_bitmap"], [0] * 8)


class WeightedRoundRobinTests(unittest.TestCase):
    """验证动态整数平滑WRR的比例和更新能力。"""

    def test_dynamic_three_to_one_ratio(self):
        """功能：用3:1权重执行40次仲裁。

        目的：断言调度器不展开权重仍精确给出30:10次机会。
        输入：无。输出：通过unittest断言报告结果。
        """
        scheduler = SmoothWeightedRoundRobinScheduler(["g0", "g1"], [1, 1])
        scheduler.set_weights({"g0": 3, "g1": 1})
        selected = [scheduler.select_next(lambda _: True) for _ in range(40)]
        self.assertEqual(selected.count("g0"), 30)
        self.assertEqual(selected.count("g1"), 10)


class DemandSatisfactionMetricTests(unittest.TestCase):
    """验证需求满足率使用SSD最后完成时刻和GPU计算截止时刻。"""

    def test_only_nonempty_layers_completed_by_deadline_are_satisfied(self):
        """功能：构造一份按时、一份超时和一份空层结果。

        目的：断言分子为1、分母为2，且空层不是SSD读取demand。
        输入：无。输出：通过unittest断言报告结果。
        """
        inference_results = [{
            "gpu_id": "GPU0",
            "inference_index": 0,
            "layers": [
                {
                    "layer_request_id": "d0",
                    "block_count": 1,
                    "compute_done_time_us": 100,
                    "io_completion_time_us": 100,
                },
                {
                    "layer_request_id": "d1",
                    "block_count": 1,
                    "compute_done_time_us": 200,
                    "io_completion_time_us": 201,
                },
                {
                    "layer_request_id": "d2",
                    "block_count": 0,
                    "compute_done_time_us": 300,
                    "io_completion_time_us": 0,
                },
            ],
        }]
        result = (
            qos_ssd_simulator.JointSimulation
            ._demand_satisfaction_statistics(inference_results)
        )
        self.assertEqual(result["satisfied_demand_count"], 1)
        self.assertEqual(result["total_demand_count"], 2)
        self.assertEqual(result["satisfaction_ratio"], 0.5)


class TopologyTests(unittest.TestCase):
    """使用缩短负载验证不同GPU/SSD数量和需求满足率输出。"""

    def run_topology(self, gpu_count, ssd_count):
        """功能：在内存中覆盖拓扑并运行一层仿真。

        目的：快速验证任意多GPU/多SSD组件装配与请求守恒。
        输入：GPU和SSD数量。输出：联合仿真结果。
        """
        original_load = qos_ssd_simulator.load_yaml
        simulation_file = (
            PROJECT_DIR / "config" / "simulation_config.yaml"
        ).resolve()

        def test_load(path):
            """功能：为本次测试缩短顶层工作负载。

            目的：不修改项目YAML即可完成多种拓扑的快速回归。
            输入：YAML路径。输出：可能包含内存覆盖的配置。
            """
            data = deepcopy(original_load(path))
            resolved = Path(path).resolve()
            if resolved == simulation_file:
                simulation = data["simulation"]
                simulation["topology"]["gpu_count"] = gpu_count
                simulation["topology"]["storage_path_count"] = ssd_count
                generation = simulation["workload_generation"]
                generation["inference_count_per_gpu"] = 1
                generation["input_tokens_range"] = [512, 768]
                generation["prefill_layer_hit_ratio_range"] = [0.5, 0.8]
            elif resolved == qos_ssd_simulator.MULTI_GPU_CONFIG_FILE.resolve():
                data["multi_gpu"]["defaults"].update({
                    "first_layer_index": 0,
                    "last_layer_index": 0,
                })
            return data

        with patch.object(qos_ssd_simulator, "load_yaml", side_effect=test_load):
            return qos_ssd_simulator.run_joint_simulation()

    def test_multiple_topologies_complete(self):
        """功能：运行1x1、3x2和10x5拓扑。

        目的：断言请求守恒、控制状态归零且需求满足率字段完整。
        输入：无。输出：通过unittest断言报告结果。
        """
        for gpu_count, ssd_count in ((1, 1), (3, 2), (10, 5)):
            with self.subTest(gpu_count=gpu_count, ssd_count=ssd_count):
                result = self.run_topology(gpu_count, ssd_count)
                self.assertEqual(result["gpu_count"], gpu_count)
                self.assertEqual(result["ssd_count"], ssd_count)
                self.assertEqual(
                    result["dpu"]["rate_control"]["active_demand_count"],
                    0,
                )
                self.assertEqual(
                    result["dpu"]["rate_control"]["waiting_demand_count"],
                    0,
                )
                self.assertEqual(
                    result["demand_satisfaction"]["total_demand_count"],
                    gpu_count,
                )
                self.assertTrue(all(
                    path["qos"]["completed"]
                    for path in result["storage_paths"].values()
                ))


if __name__ == "__main__":
    unittest.main()
