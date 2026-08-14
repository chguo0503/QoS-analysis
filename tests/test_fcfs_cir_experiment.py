"""64 GPU、2 SSD、3层FCFS-CIR对照实验的配置与端到端测试。"""

from copy import deepcopy
from pathlib import Path
import subprocess
import unittest

from experiments.compare_fcfs_cir import (
    TOKEN_CONFIG_FILE,
    WRR_CONFIG_FILE,
    build_simulation_override,
    execution_identity,
    gpu_layer_samples,
    load_experiment_config,
    summarize_pair,
    summarize_policy,
)
from llm_workload.inference_workload_sampler import (
    UniformRandomInferenceSampler,
)
from llm_workload.kv_placement_manager import KVPlacementManager
from llm_workload.layer_request import DEFAULT_WORKLOAD, LLMWorkload
from qos_ssd_simulator import run_joint_simulation


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_FILE = (
    PROJECT_DIR / "experiments" / "config" / "fcfs_cir_comparison.yaml"
)
SSD_CAPACITY_BYTES_PER_SECOND = 40_000_000_000


def build_base_workloads(gpu_count, workload_override):
    """功能：为指定数量GPU生成实验工作负载模板。

    目的：在不运行SSD后端时复现64 GPU唯一随机序列，
    便于独立验证输入、Placement和requested CIR。

    输入：GPU数量和实验的LLM默认字段覆盖。

    输出： ``gpu_id -> 完整workload模板`` 映射。
    """
    workloads = {}
    for gpu_index in range(gpu_count):
        workload = deepcopy(DEFAULT_WORKLOAD)
        workload.update(deepcopy(workload_override))
        workload["workload_id"] = f"experiment_GPU{gpu_index}"
        workload["p_node_id"] = f"P{gpu_index}"
        workloads[f"GPU{gpu_index}"] = workload
    return workloads


def build_sampled_workloads(config, load_name):
    """功能：根据固定种子生成一组64 GPU工作负载。

    目的：让配置和容量测试与真实实验入口共用同一
    ``seed=6103``、Token区间和命中率区间。

    输入：完整实验配置和 ``normal_load`` 或 ``overload_load``。

    输出： ``gpu_id -> 第一次推理workload`` 映射。
    """
    simulation = build_simulation_override(
        config,
        config["loads"][load_name],
    )
    gpu_count = simulation["topology"]["gpu_count"]
    base_workloads = build_base_workloads(gpu_count, config["workload"])
    sampler = UniformRandomInferenceSampler(
        simulation["workload_generation"]
    )
    sequences = sampler.build_sequences(base_workloads)
    return {
        gpu_id: workloads[0]
        for gpu_id, workloads in sequences.items()
    }


def requested_cir_totals(config, load_name):
    """功能：汇总一层在每块SSD上的requested CIR。

    目的：使用 ``KVPlacementManager`` 作为唯一带宽诉求计算源，
    同一GPU/层/SSD只取一次聚合值，不按Block重复累加。

    输入：完整实验配置和负载名称。

    输出： ``SSD ID -> 该层总requested Byte/s`` 映射。
    """
    totals = {"SSD0": 0, "SSD1": 0}
    for workload in build_sampled_workloads(config, load_name).values():
        layer_plan = LLMWorkload(workload=workload).start_next_layer()
        placement = KVPlacementManager(
            ["SSD0", "SSD1"],
            workload["placement"],
        )
        requests = placement.build_requests(layer_plan)
        one_rate_per_storage_target = {}
        for request in requests:
            storage_target_id = request["basic"]["storage_target_id"]
            one_rate_per_storage_target[storage_target_id] = request[
                "demand_bw"
            ]["aggregate_required_bytes_per_second"]
        for storage_target_id, requested_cir in one_rate_per_storage_target.items():
            totals[storage_target_id] += requested_cir
    return totals


def run_small_policy(policy):
    """功能：运行4 GPU、2 SSD、1层的快速端到端仿真。

    目的：以真实DPU、QoS和ASU SSD低成本检查策略输入一致性、
    权重写入次数、Queue数量和全链路请求/字节守恒。

    输入： ``baseline`` 或 ``demand_aware_fcfs_cir``。

    输出：一次完整联合仿真结果。
    """
    return run_joint_simulation(
        binding_strategy="balanced_exclusive",
        rate_control_strategy=policy,
        simulation_config_override={
            "start_time_us": 0,
            "topology": {
                "gpu_count": 4,
                "storage_path_count": 2,
            },
            "workload_generation": {
                "inference_count_per_gpu": 1,
                "random_seed": 6103,
                "input_tokens_range": [1_000, 2_000],
                "prefill_layer_hit_ratio_range": [0.90, 0.99],
                "unique_across_gpus": True,
                "inter_inference_gap_us": 0,
            },
        },
        workload_defaults_override={
            "first_layer_index": 0,
            "last_layer_index": 0,
            "arrival_time_us": 0,
            "batch_size": 1,
            "placement": {
                "strategy": "balanced_round_robin",
                "allowed_storage_targets": "all",
                "random_seed": 6103,
            },
        },
        token_config_file=TOKEN_CONFIG_FILE,
        scheduler_config_file=WRR_CONFIG_FILE,
    )


class WorkloadAndPlacementTests(unittest.TestCase):
    """验证实验随机负载和确定性平衡Block放置。"""

    @classmethod
    def setUpClass(cls):
        """功能：读取一次实验YAML供本类所有测试共用。

        目的：避免各测试重复解析配置，同时保证测试读取的
        就是命令行实验使用的YAML。

        输入：unittest类初始化调用。

        输出：无；设置 ``cls.config``。
        """
        cls.config, _ = load_experiment_config(EXPERIMENT_CONFIG_FILE)

    def test_seeded_values_are_unique_and_reproducible(self):
        """功能：两次生成正常负载的64 GPU样本。

        目的：断言input_tokens互不相同、hit ratio互不相同，
        并且固定种子6103能完全复现。

        输入：无。

        输出：通过unittest断言报告随机序列。
        """
        first = build_sampled_workloads(self.config, "normal_load")
        second = build_sampled_workloads(self.config, "normal_load")
        first_identity = [
            (workload["input_tokens"], workload["prefill_layer_hit_ratio"])
            for workload in first.values()
        ]
        second_identity = [
            (workload["input_tokens"], workload["prefill_layer_hit_ratio"])
            for workload in second.values()
        ]

        self.assertEqual(first_identity, second_identity)
        self.assertEqual(len({value[0] for value in first_identity}), 64)
        self.assertEqual(len({value[1] for value in first_identity}), 64)

    def test_balanced_mapping_is_stable_and_uses_both_ssds(self):
        """功能：用正常负载逐GPU放置Layer 0的全部Block。

        目的：断言每层同时使用两盘、Block数差不超过1，
        且逆序遍历不改变 ``block_index -> SSD`` 映射。

        输入：无。

        输出：通过unittest断言报告Placement结果。
        """
        workloads = build_sampled_workloads(self.config, "normal_load")
        for workload in workloads.values():
            layer_plan = LLMWorkload(workload=workload).start_next_layer()
            manager = KVPlacementManager(
                ["SSD0", "SSD1"],
                workload["placement"],
            )
            forward_requests = manager.build_requests(layer_plan)
            counts = {"SSD0": 0, "SSD1": 0}
            forward_mapping = {}
            for block, request in zip(layer_plan["blocks"], forward_requests):
                target = request["basic"]["storage_target_id"]
                counts[target] += 1
                forward_mapping[block["block_index"]] = target

            reverse_plan = deepcopy(layer_plan)
            reverse_plan["blocks"].reverse()
            reverse_manager = KVPlacementManager(
                ["SSD0", "SSD1"],
                workload["placement"],
            )
            reverse_requests = reverse_manager.build_requests(reverse_plan)
            reverse_mapping = {
                block["block_index"]: request["basic"]["storage_target_id"]
                for block, request in zip(
                    reverse_plan["blocks"],
                    reverse_requests,
                )
            }

            self.assertGreater(counts["SSD0"], 0)
            self.assertGreater(counts["SSD1"], 0)
            self.assertLessEqual(abs(counts["SSD0"] - counts["SSD1"]), 1)
            self.assertEqual(forward_mapping, reverse_mapping)

    def test_normal_and_overload_capacity_preconditions(self):
        """功能：用KV Placement计算两组负载的每盘总诉求。

        目的：断言正常负载每盘小于40 GB/s，过载负载每盘
        大于40 GB/s，确保两组实验真正覆盖不同容量区间。

        输入：无。

        输出：通过unittest断言报告容量前置条件。
        """
        normal_totals = requested_cir_totals(self.config, "normal_load")
        overload_totals = requested_cir_totals(self.config, "overload_load")
        self.assertTrue(all(
            total < SSD_CAPACITY_BYTES_PER_SECOND
            for total in normal_totals.values()
        ))
        self.assertTrue(all(
            total > SSD_CAPACITY_BYTES_PER_SECOND
            for total in overload_totals.values()
        ))


class EndToEndPolicyTests(unittest.TestCase):
    """使用小型真实后端验证两策略的公平对比和守恒。"""

    @classmethod
    def setUpClass(cls):
        """功能：各运行一次小型Baseline和FCFS-CIR仿真。

        目的：让多个端到端断言共用结果，缩短全量测试时间。

        输入：unittest类初始化调用。

        输出：无；设置两份策略结果。
        """
        cls.baseline = run_small_policy("baseline")
        cls.demand_aware = run_small_policy("demand_aware_fcfs_cir")

    def test_policies_use_identical_workload_placement_and_binding(self):
        """功能：对比两份结果的非时序执行身份。

        目的：断言工作负载、Block→SSD和GPU→Queue完全一致，
        策略差异只来自Queue CIR。

        输入：无。

        输出：通过unittest断言报告身份比较。
        """
        self.assertEqual(
            execution_identity(self.baseline),
            execution_identity(self.demand_aware),
        )

    def test_group_weights_are_not_written_by_either_policy(self):
        """功能：检查两种策略的Group写入计数和终态权重。

        目的：断言动态Group接口虽保留，本次策略写入次数为0，
        两块SSD的Group WRR始终全为1。

        输入：无。

        输出：通过unittest断言报告Group控制状态。
        """
        for result in (self.baseline, self.demand_aware):
            self.assertEqual(result["dpu"]["group_weight_write_count"], 0)
            for path in result["storage_paths"].values():
                self.assertEqual(path["qos"]["group_weight_bitmap"], [1] * 8)

    def test_each_ssd_uses_one_queue_per_gpu(self):
        """功能：统计小型实验每块SSD的非空Queue。

        目的：断言4张GPU在每盘正好使用4条互不相同的独占Queue。

        输入：无。

        输出：通过unittest断言报告活跃Queue数。
        """
        for result in (self.baseline, self.demand_aware):
            for path in result["storage_paths"].values():
                active_queues = [
                    queue_id
                    for queue_id, statistics in path["qos"][
                        "queue_statistics"
                    ].items()
                    if statistics["dispatched_requests"] > 0
                ]
                self.assertEqual(len(active_queues), 4)

    def test_request_block_and_byte_conservation(self):
        """功能：比较GPU、QoS和SSD三层计数与字节数。

        目的：断言每个Block只生成一条IO，无丢失、重复或字节数改变。

        输入：无。

        输出：通过unittest断言报告全链路守恒。
        """
        for result in (self.baseline, self.demand_aware):
            conservation = result["request_conservation"]
            request_counts = {
                conservation["gpu_requests"],
                conservation["gpu_completed_requests"],
                conservation["qos_input_requests"],
                conservation["qos_dispatched_requests"],
                conservation["ssd_completed_requests"],
            }
            byte_counts = {
                conservation["gpu_bytes"],
                conservation["qos_dispatched_bytes"],
                conservation["ssd_completed_bytes"],
            }
            self.assertEqual(len(request_counts), 1)
            self.assertEqual(len(byte_counts), 1)

    def test_baseline_has_no_cir_dispatch_and_fcfs_releases_all_demands(self):
        """功能：检查Baseline的速率类别和FCFS-CIR的Demand终态。

        目的：断言Baseline全部走EXCESS，且FCFS根据Queue空状态
        最终释放所有Demand，不依赖SSD完成回调。

        输入：无。

        输出：通过unittest断言报告调度和Demand终态。
        """
        self.assertTrue(all(
            path["qos"]["cir_dispatched_request_count"] == 0
            for path in self.baseline["storage_paths"].values()
        ))
        rate_statistics = self.demand_aware["dpu"]["rate_control"]
        self.assertEqual(rate_statistics["active_demand_count"], 0)
        self.assertTrue(all(
            peak <= SSD_CAPACITY_BYTES_PER_SECOND
            for peak in rate_statistics[
                "peak_assigned_cir_bytes_per_second"
            ].values()
        ))


class SummaryMetricTests(unittest.TestCase):
    """验证跨SSD读取时间、P95和配对GPU计数定义。"""

    def synthetic_result(self, ssd0_completion, ssd1_completion):
        """功能：构造一张GPU的三层极小结果。

        目的：用可手算的时刻验证summary函数，不引入QoS/SSD时序噪声。

        输入：SSD0和SSD1的每层完成时刻。

        输出：符合summary函数所需最小字段的联合结果。
        """
        layers = []
        for layer_index in range(3):
            layer_start = layer_index * 100
            layers.append({
                "layer_start_time_us": layer_start,
                "compute_done_time_us": layer_start + 50,
                "ssd_completion_times_us": {
                    "SSD0": layer_start + ssd0_completion,
                    "SSD1": layer_start + ssd1_completion,
                },
            })
        return {
            "gpus": {
                "GPU0": {"inferences": [{"layers": layers}]},
            },
            "storage_paths": {
                "SSD0": {"ssd": {"last_completion_time_us": ssd0_completion + 200}},
                "SSD1": {"ssd": {"last_completion_time_us": ssd1_completion + 200}},
            },
        }

    def test_actual_read_uses_maximum_ssd_completion(self):
        """功能：使SSD0用30 us、SSD1用70 us完成同一层。

        目的：断言actual read为70 us，signed delta为20 us，
        不是两盘时间的求和或平均。

        输入：无。

        输出：通过unittest断言报告逐层样本。
        """
        result = self.synthetic_result(30, 70)
        samples = gpu_layer_samples(result, ["SSD0", "SSD1"])
        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["actual_read_us"] for sample in samples}, {70})
        self.assertEqual({sample["signed_delta_us"] for sample in samples}, {20})

    def test_summary_has_only_requested_metrics(self):
        """功能：对一份合成结果生成策略和配对摘要。

        目的：断言每策略只有7个最终指标，配对结果只有3个GPU计数。

        输入：无。

        输出：通过unittest断言报告summary schema。
        """
        baseline = self.synthetic_result(30, 70)
        demand_aware = self.synthetic_result(20, 60)
        baseline_summary, baseline_samples = summarize_policy(
            baseline,
            ["SSD0", "SSD1"],
        )
        demand_summary, demand_samples = summarize_policy(
            demand_aware,
            ["SSD0", "SSD1"],
        )
        paired = summarize_pair(baseline_samples, demand_samples)

        self.assertEqual(len(baseline_summary), 7)
        self.assertEqual(len(demand_summary), 7)
        self.assertEqual(
            set(paired),
            {
                "improved_gpu_count",
                "worsened_gpu_count",
                "unchanged_gpu_count",
            },
        )
        self.assertEqual(paired["improved_gpu_count"], 1)


class RepositoryCleanupTests(unittest.TestCase):
    """验证旧的每IO准入字段已从代码库清除。"""

    def test_removed_admission_field_has_no_repository_match(self):
        """功能：用git grep搜索旧的QoS每IO准入字段。

        目的：防止代码、测试或文档重新引入目标硬件不存在的
        Admission Gate控制变量。

        输入：当前Git工作树。

        输出：通过unittest断言报告全仓库搜索结果。
        """
        forbidden_field = "qos" + "_admitted"
        completed = subprocess.run(
            ["git", "grep", "-n", forbidden_field],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout)


if __name__ == "__main__":
    unittest.main()
