"""DPU FCFS-CIR控制面与QoS两轮调度的回归测试。"""

from pathlib import Path
import unittest

from DPU import (
    DPURequestGateway,
    DemandAwareFCFSCIRController,
    build_queue_binding_strategy,
)
from qos import build_qos_simulator
from simulation_common.config_utils import load_yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SIMULATION_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
SSD_CAPACITY_BYTES_PER_SECOND = 40_000_000_000


def build_uniform_qos():
    """功能：创建CIR=0、PIR=uncapped、全部WRR=1的纯QoS实例。

    目的：隔离SSD后端，直接验证DPU控制写入、Queue状态回传
    以及CIR-first/EXCESS调度语义。

    输入：无。

    输出：未连接SSD的QoS离散事件仿真器。
    """
    qos_config = load_yaml(SIMULATION_CONFIG_FILE)["simulation"]["qos"]
    return build_qos_simulator(qos_config=qos_config, start_time_us=0)


def make_dpu_request(request_id, gpu_index, requested_cir):
    """功能：构造一个最小KV Placement输出请求。

    目的：测试DPU只在内部保存聚合带宽诉求，向QoS只传递
    普通IO数据面字段和最终CIR/PIR控制。

    输入：请求ID、GPU索引和KV Placement生成的整数Byte/s诉求。

    输出：包含 ``basic`` 和 ``demand_bw`` 的DPU入口字典。
    """
    return {
        "basic": {
            "request_id": request_id,
            "p_node_id": f"P{gpu_index}",
            "storage_target_id": "SSD0",
            "size_bytes": 147_456,
        },
        "demand_bw": {
            "demand_group_id": f"demand_{gpu_index}",
            "aggregate_required_bytes_per_second": requested_cir,
        },
    }


def build_gateway(gpu_count, controller=None):
    """功能：装配一套单SSD的DPU↔QoS测试链路。

    目的：复用真实Queue布局和互斥绑定，避免单元测试自行
    伪造调度器行为。

    输入：GPU数量和可选FCFS-CIR控制器。

    输出： ``(qos, dpu)`` 二元组。
    """
    qos = build_uniform_qos()
    queues_by_target = {"SSD0": qos.token_stage.queue_order}
    p_node_ids = [f"P{index}" for index in range(gpu_count)]
    binding = build_queue_binding_strategy(
        "balanced_exclusive",
        p_node_ids,
        queues_by_target,
    )
    dpu = DPURequestGateway(
        queues_by_target,
        binding,
        qos.input,
        {"SSD0": qos},
        controller,
    )
    return qos, dpu


class BalancedExclusiveBindingTests(unittest.TestCase):
    """验证64 GPU在每块SSD上的固定互斥Queue绑定。"""

    def test_exact_formula_and_group_distribution(self):
        """功能：在两个SSD namespace中预绑定64张GPU。

        目的：验证每盘64条Queue互不冲突、8个Group各8条，
        且Queue下标严格符合实验公式。

        输入：无。

        输出：通过unittest断言报告绑定结果。
        """
        p_node_ids = [f"P{index}" for index in range(64)]
        queues = [f"q{index:03d}" for index in range(256)]
        strategy = build_queue_binding_strategy(
            "balanced_exclusive",
            p_node_ids,
            {"SSD0": queues, "SSD1": queues},
        )

        for storage_target_id in ("SSD0", "SSD1"):
            selected = [
                strategy.bindings[(p_node_id, storage_target_id)]
                for p_node_id in p_node_ids
            ]
            self.assertEqual(len(set(selected)), 64)
            group_counts = {group_index: 0 for group_index in range(8)}
            for gpu_index, queue_id in enumerate(selected):
                expected_index = 32 * (gpu_index % 8) + gpu_index // 8
                self.assertEqual(queue_id, f"q{expected_index:03d}")
                group_counts[expected_index // 32] += 1
            self.assertEqual(set(group_counts.values()), {8})


class OneGroupPerGpuBindingTests(unittest.TestCase):
    """验证每张GPU固定使用自己连续Group的首Queue。"""

    def test_each_gpu_uses_the_first_queue_of_its_group(self):
        p_node_ids = [f"P{index}" for index in range(8)]
        queues = [f"q{index:03d}" for index in range(256)]
        strategy = build_queue_binding_strategy(
            "one_group_per_gpu",
            p_node_ids,
            {"SSD0": queues, "SSD1": queues},
        )

        expected = [f"q{index:03d}" for index in range(0, 256, 32)]
        for storage_target_id in ("SSD0", "SSD1"):
            self.assertEqual(
                [
                    strategy.bindings[(p_node_id, storage_target_id)]
                    for p_node_id in p_node_ids
                ],
                expected,
            )

    def test_gpu_count_must_divide_256(self):
        queues = [f"q{index:03d}" for index in range(256)]
        with self.assertRaisesRegex(ValueError, "divide 256"):
            build_queue_binding_strategy(
                "one_group_per_gpu",
                [f"P{index}" for index in range(10)],
                {"SSD0": queues},
            )


class FCFSCIRControllerTests(unittest.TestCase):
    """验证仅使用整数运算的先到先服务CIR分配。"""

    def build_controller(self):
        """功能：创建一块40 GB/s SSD的空FCFS-CIR控制器。

        目的：让各个分配规则测试共用相同物理容量。

        输入：无。

        输出：新的 ``DemandAwareFCFSCIRController``。
        """
        return DemandAwareFCFSCIRController({
            "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
        })

    def register(self, controller, queue_id, requested_gb_s, arrival_order):
        """功能：按测试顺序登记一个Queue Demand。

        目的：将易读的GB/s用例转换成控制器真实整数Byte/s输入。

        输入：控制器、Queue ID、GB/s诉求和到达序号。

        输出：无；原地登记Demand。
        """
        controller.register_demand(
            "SSD0",
            queue_id,
            requested_gb_s * 1_000_000_000,
            arrival_order,
        )

    def test_fcfs_partial_assignment_for_30_plus_20(self):
        """功能：将先到30 GB/s和后到20 GB/s放入40 GB/s SSD。

        目的：断言FCFS部分分配是30+10，而不是比例缩放或整项拒绝。

        输入：无。

        输出：通过unittest断言报告分配结果。
        """
        controller = self.build_controller()
        self.register(controller, "q000", 30, 0)
        self.register(controller, "q032", 20, 1)
        updates = controller.recalculate("SSD0")

        self.assertEqual(
            updates["queue_rates"],
            {"q000": 30_000_000_000, "q032": 10_000_000_000},
        )
        self.assertIsNone(updates["group_weights"])
        self.assertEqual(
            sum(controller.queue_rates["SSD0"].values()),
            SSD_CAPACITY_BYTES_PER_SECOND,
        )

    def test_zero_assignment_does_not_remove_demand(self):
        """功能：登记一个用完容量的Demand和一个后到Demand。

        目的：验证assigned CIR=0只是没有保障，Demand仍保留为活跃状态。

        输入：无。

        输出：通过unittest断言报告Demand状态。
        """
        controller = self.build_controller()
        self.register(controller, "q000", 40, 0)
        self.register(controller, "q032", 20, 1)
        controller.recalculate("SSD0")

        self.assertEqual(controller.demands["SSD0"]["q032"]["assigned_cir"], 0)
        self.assertIn("q032", controller.demands["SSD0"])

    def test_releasing_first_queue_restores_second_cir(self):
        """功能：模拟30+20分配后先到Queue变空。

        目的：断言DPU只依据Queue depth释放A，并将B从10恢复到20 GB/s。

        输入：无。

        输出：通过unittest断言报告重分配结果。
        """
        controller = self.build_controller()
        self.register(controller, "q000", 30, 0)
        self.register(controller, "q032", 20, 1)
        controller.recalculate("SSD0")
        updates = controller.release_empty_demands(
            "SSD0",
            {"q000": 0, "q032": 1},
        )

        self.assertNotIn("q000", controller.demands["SSD0"])
        self.assertEqual(controller.demands["SSD0"]["q032"]["assigned_cir"], 20_000_000_000)
        self.assertEqual(
            updates["queue_rates"],
            {"q000": 0, "q032": 20_000_000_000},
        )

    def test_total_assignment_never_exceeds_capacity(self):
        """功能：按到达顺序登记多个过载Demand。

        目的：对每次重算断言assigned CIR总和不超过40 GB/s。

        输入：无。

        输出：通过unittest断言报告容量上限。
        """
        controller = self.build_controller()
        for index, requested_gb_s in enumerate((17, 19, 23, 29)):
            self.register(
                controller,
                f"q{index * 32:03d}",
                requested_gb_s,
                index,
            )
            controller.recalculate("SSD0")
            self.assertLessEqual(
                sum(controller.queue_rates["SSD0"].values()),
                SSD_CAPACITY_BYTES_PER_SECOND,
            )


class DPUQoSInterfaceTests(unittest.TestCase):
    """验证DPU↔QoS接口限制和两轮调度语义。"""

    def test_baseline_dispatches_only_through_excess(self):
        """功能：在Baseline配置下下发两条IO。

        目的：断言CIR=0、PIR=uncapped时没有CIR dispatch，
        非空Queue仍能全部通过EXCESS完成。

        输入：无。

        输出：通过unittest断言报告两轮计数。
        """
        qos, dpu = build_gateway(gpu_count=2)
        dpu.submit_batch([
            make_dpu_request("r0", 0, 10_000_000_000),
            make_dpu_request("r1", 1, 10_000_000_000),
        ], 0)
        result = qos.end()

        self.assertEqual(result["cir_dispatched_request_count"], 0)
        self.assertEqual(result["excess_dispatched_request_count"], 2)
        self.assertEqual(len(result["dispatched_requests"]), 2)

    def test_zero_assigned_queue_still_dispatches_through_excess(self):
        """功能：使先到Demand占用40 GB/s，后到Demand获得0 CIR。

        目的：断言两条Queue都能下发，assigned CIR=0不构成Admission Gate。

        输入：无。

        输出：通过unittest断言报告请求完成情况。
        """
        controller = DemandAwareFCFSCIRController({
            "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
        })
        qos, dpu = build_gateway(gpu_count=2, controller=controller)
        submitted = dpu.submit_batch([
            make_dpu_request("r0", 0, 40_000_000_000),
            make_dpu_request("r1", 1, 20_000_000_000),
        ], 0)
        second_queue = submitted[1]["queue_id"]
        self.assertEqual(
            controller.demands["SSD0"][second_queue]["assigned_cir"],
            0,
        )

        result = qos.end()
        self.assertEqual(
            {request["request_id"] for request in result["dispatched_requests"]},
            {"r0", "r1"},
        )
        self.assertGreaterEqual(result["excess_dispatched_request_count"], 1)
        self.assertEqual(controller.statistics()["active_demand_count"], 0)

    def test_qos_request_contains_only_data_plane_fields(self):
        """功能：检查DPU完成Queue绑定后的QoS请求格式。

        目的：断言KV Placement的Demand ID和requested/assigned CIR
        只留在DPU内部，不通过每IO接口写入QoS。

        输入：无。

        输出：通过unittest断言报告字段集合。
        """
        qos, dpu = build_gateway(gpu_count=1)
        submitted = dpu.submit_batch([
            make_dpu_request("r0", 0, 4_000_000_000),
        ], 0)

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
        qos.end()

    def test_queue_depth_observer_releases_demand(self):
        """功能：运行一条由DPU登记的Queue Demand至Queue排空。

        目的：断言QoS只用无请求载荷的状态唤醒通知DPU，
        DPU主动读depth=0后释放Demand和CIR。

        输入：无。

        输出：通过unittest断言报告控制器终态。
        """
        controller = DemandAwareFCFSCIRController({
            "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
        })
        qos, dpu = build_gateway(gpu_count=1, controller=controller)
        dpu.submit_batch([
            make_dpu_request("r0", 0, 4_000_000_000),
        ], 0)
        self.assertEqual(controller.statistics()["active_demand_count"], 1)

        qos.end()
        self.assertEqual(controller.statistics()["active_demand_count"], 0)
        self.assertEqual(
            controller.statistics()["completed_demand_count_by_storage_target"],
            {"SSD0": 1},
        )

    def test_repeated_blocks_register_one_aggregate_demand(self):
        """功能：将同一GPU/层/SSD的两个Block批量交给DPU。

        目的：断言每个Block重复携带的KV Placement聚合带宽只登记
        一次，不会被DPU按IO数量再次累加。

        输入：无。

        输出：通过unittest断言报告DPU内部Demand数量和requested CIR。
        """
        controller = DemandAwareFCFSCIRController({
            "SSD0": SSD_CAPACITY_BYTES_PER_SECOND,
        })
        qos, dpu = build_gateway(gpu_count=1, controller=controller)
        requested_cir = 4_000_000_000
        dpu.submit_batch([
            make_dpu_request("r0", 0, requested_cir),
            make_dpu_request("r1", 0, requested_cir),
        ], 0)

        self.assertEqual(len(controller.demands["SSD0"]), 1)
        demand = next(iter(controller.demands["SSD0"].values()))
        self.assertEqual(demand["requested_cir"], requested_cir)
        self.assertEqual(sum(qos.queue_io_counts().values()), 2)
        qos.end()

    def test_all_pir_is_uncapped_and_weights_stay_one(self):
        """功能：检查实验QoS的256条Queue和两级WRR初值。

        目的：断言两策略的PIR全部uncapped，Group和Queue权重全为1。

        输入：无。

        输出：通过unittest断言报告静态QoS配置。
        """
        qos = build_uniform_qos()
        controllers = qos.token_stage.controllers
        self.assertTrue(all(
            controller.pir_bucket is None
            for controller in controllers.values()
        ))
        self.assertEqual(
            set(qos.scheduler.group_scheduler.weights.values()),
            {1},
        )
        for queue_scheduler in qos.scheduler.queue_schedulers.values():
            slots = queue_scheduler.rr_scheduler.item_order
            self.assertEqual(len(slots), 32)
            self.assertEqual(len(set(slots)), 32)

    def test_dynamic_group_weight_interface_is_preserved(self):
        """功能：向QoS独立提交一次动态Group WRR更新。

        目的：确认未来策略仍可使用该硬件接口，而本次
        Baseline和FCFS-CIR只是默认不调用它。

        输入：无。

        输出：通过unittest断言报告更新后的权重。
        """
        qos = build_uniform_qos()
        weights = {
            group_id: index + 1
            for index, group_id in enumerate(qos.queue_layout.group_order)
        }
        qos.schedule_group_weight_update(weights, 0)
        qos.process_at(0)
        self.assertEqual(qos.scheduler.group_scheduler.weights, weights)


if __name__ == "__main__":
    unittest.main()
