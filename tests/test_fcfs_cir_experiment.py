"""统一配置、Baseline和Demand-aware FCFS CIR的端到端回归测试。"""

from copy import deepcopy
from pathlib import Path
import subprocess
import unittest

from qos_ssd_simulator import (
    build_simulation,
    load_simulation_config,
    run_joint_simulation,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SSD_CAPACITY_BYTES_PER_SECOND = 40_000_000_000


def build_small_config():
    """功能：构造快速端到端测试使用的统一配置副本。

    目的：沿用正式QoS、DPU和SSD配置，只缩小GPU、Token和层数以控制测试时间。

    输入：无。

    输出：
        dict: 4 GPU、2 SSD、1层的独立simulation配置。
    """
    config = load_simulation_config()
    config["topology"]["gpu_count"] = 4
    config["topology"]["ssd_counts"] = [2]
    config["workload_generation"].update({
        "inference_count_per_gpu": 1,
        "input_tokens_range": [1_000, 2_000],
        "prefill_layer_hit_ratio_range": [0.90, 0.99],
    })
    config["workload"].update({
        "first_layer_index": 0,
        "last_layer_index": 0,
        "batch_size": 1,
    })
    return config


def execution_identity(result):
    """功能：提取不包含策略时序的工作负载、Placement和Queue绑定身份。

    目的：确认两种策略比较只改变Queue CIR，不改变输入、SSD目标或固定Queue。

    输入：
        result: 一次完整联合仿真结果。

    输出：
        tuple: 推理随机值与每个请求的数据路径身份。
    """
    workloads = tuple(sorted(
        (
            inference["gpu_id"],
            inference["input_tokens"],
            inference["prefill_layer_hit_ratio"],
        )
        for inference in result["llms"]
    ))
    paths = tuple(sorted(
        (
            request["request_id"],
            storage_target_id,
            request["p_node_id"],
            request["queue_id"],
            request["size_bytes"],
        )
        for storage_target_id, path in result["storage_paths"].items()
        for request in path["qos"]["dispatched_requests"]
    ))
    return workloads, paths


class UnifiedConfigurationTests(unittest.TestCase):
    """验证项目只保留一个YAML和一个可执行仿真入口。"""

    def test_default_experiment_parameters(self):
        """功能：读取项目统一YAML的正式实验参数。

        目的：固定当前128 GPU、Batch 1、4层、1～10 SSD和两种策略入口。

        输入：无。

        输出：通过unittest断言报告配置内容。
        """
        config = load_simulation_config()
        self.assertEqual(config["topology"]["gpu_count"], 128)
        self.assertEqual(
            config["topology"]["ssd_counts"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(
            config["dpu"]["rate_control"]["strategies"],
            ["baseline", "utility_edf_integer_l750"],
        )
        self.assertEqual(config["workload"]["batch_size"], 1)
        self.assertEqual(config["workload"]["first_layer_index"], 0)
        self.assertEqual(config["workload"]["last_layer_index"], 3)
        self.assertEqual(config["gpu"]["effective_compute_tflops"], 512.0)

    def test_only_one_yaml_and_one_production_main(self):
        """功能：枚举仓库中的YAML和生产代码main守卫。

        目的：防止后续重新引入相互覆盖的组件YAML或第二个仿真脚本入口。

        输入：当前项目工作树。

        输出：通过unittest断言报告唯一配置与入口路径。
        """
        yaml_files = sorted(
            path.relative_to(PROJECT_DIR).as_posix()
            for path in PROJECT_DIR.rglob("*")
            if path.suffix in {".yaml", ".yml"}
        )
        self.assertEqual(yaml_files, ["config/simulation_config.yaml"])

        production_mains = []
        for path in PROJECT_DIR.rglob("*.py"):
            if "tests" in path.parts:
                continue
            if 'if __name__ == "__main__":' in path.read_text(
                encoding="utf-8"
            ):
                production_mains.append(
                    path.relative_to(PROJECT_DIR).as_posix()
                )
        self.assertEqual(production_mains, ["qos_ssd_simulator.py"])

    def test_seeded_gpu_workloads_are_unique_and_reproducible(self):
        """功能：从相同统一配置构造两次64 GPU实验状态。

        目的：断言固定种子下128张GPU的Token和命中率各自互不相同且可复现。

        输入：默认simulation配置。

        输出：通过unittest断言报告随机序列身份。
        """
        config = load_simulation_config()
        first = build_simulation(config, 2, "baseline")
        second = build_simulation(config, 2, "baseline")
        first_values = [
            (
                sequence[0]["input_tokens"],
                sequence[0]["prefill_layer_hit_ratio"],
            )
            for sequence in first.gpu_workload_sequences.values()
        ]
        second_values = [
            (
                sequence[0]["input_tokens"],
                sequence[0]["prefill_layer_hit_ratio"],
            )
            for sequence in second.gpu_workload_sequences.values()
        ]
        self.assertEqual(first_values, second_values)
        self.assertEqual(len({value[0] for value in first_values}), 128)
        self.assertEqual(len({value[1] for value in first_values}), 128)


class EndToEndPolicyTests(unittest.TestCase):
    """使用真实QoS和ASU SSD验证两种策略的公平对比与守恒。"""

    @classmethod
    def setUpClass(cls):
        """功能：各运行一次小型Baseline和FCFS-CIR仿真。

        目的：让端到端断言复用同一对结果，缩短全套测试时间。

        输入：unittest类初始化调用。

        输出：None；保存两份完整策略结果。
        """
        config = build_small_config()
        cls.baseline = run_joint_simulation(
            config=deepcopy(config),
            rate_control_strategy="baseline",
        )
        cls.demand_aware = run_joint_simulation(
            config=deepcopy(config),
            rate_control_strategy="demand_aware_fcfs_cir",
        )

    def test_workload_placement_and_binding_are_identical(self):
        """功能：比较两种策略的非时序执行身份。

        目的：断言Block随机放置和GPU固定Queue绑定在策略之间完全一致。

        输入：两份已完成仿真结果。

        输出：通过unittest等值断言报告执行身份。
        """
        self.assertEqual(
            execution_identity(self.baseline),
            execution_identity(self.demand_aware),
        )

    def test_group_weights_stay_one_without_dynamic_writes(self):
        """功能：检查两种策略的Group写入计数和终态权重。

        目的：保留动态接口，但确认当前策略只写Queue CIR且WRR始终为1。

        输入：两份已完成仿真结果。

        输出：通过unittest断言报告Group控制状态。
        """
        for result in (self.baseline, self.demand_aware):
            self.assertEqual(result["dpu"]["group_weight_write_count"], 0)
            for path in result["storage_paths"].values():
                self.assertEqual(path["qos"]["group_weight_bitmap"], [1] * 8)

    def test_request_and_byte_conservation(self):
        """功能：比较GPU、QoS和SSD的请求数与字节数。

        目的：断言随机多盘放置和两种速率策略都不丢失、重复或改变Block IO。

        输入：两份已完成仿真结果。

        输出：通过unittest守恒断言报告数据通路正确性。
        """
        for result in (self.baseline, self.demand_aware):
            conservation = result["request_conservation"]
            self.assertEqual(len({
                conservation["gpu_requests"],
                conservation["gpu_completed_requests"],
                conservation["qos_input_requests"],
                conservation["qos_dispatched_requests"],
                conservation["ssd_completed_requests"],
            }), 1)
            self.assertEqual(len({
                conservation["gpu_bytes"],
                conservation["qos_dispatched_bytes"],
                conservation["ssd_completed_bytes"],
            }), 1)

    def test_baseline_excess_and_fcfs_demand_release(self):
        """功能：检查Baseline下发类别和FCFS-CIR终态。

        目的：Baseline必须全部走EXCESS；Demand-aware必须依靠Queue空状态
        释放全部Demand，且峰值assigned CIR不超过40 GB/s。

        输入：两份已完成仿真结果。

        输出：通过unittest断言报告速率控制语义。
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


class RepositoryCleanupTests(unittest.TestCase):
    """验证旧Admission Gate字段已从仓库清除。"""

    def test_removed_admission_field_has_no_repository_match(self):
        """功能：用git grep搜索旧的每IO准入字段。

        目的：防止重新引入目标硬件不存在的Admission Gate控制变量。

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
