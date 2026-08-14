#!/usr/bin/env python3
"""比较Baseline与Demand-aware FCFS CIR的64 GPU、2 SSD、3层读取时间。"""

import argparse
import json
import math
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from qos_ssd_simulator import JointSimulation  # noqa: E402
from simulation_common.config_utils import load_yaml  # noqa: E402


DEFAULT_CONFIG_FILE = (
    PROJECT_DIR / "experiments" / "config" / "fcfs_cir_comparison.yaml"
)
TOKEN_CONFIG_FILE = (
    PROJECT_DIR
    / "experiments"
    / "config"
    / "uniform_baseline_token_bucket.yaml"
)
WRR_CONFIG_FILE = (
    PROJECT_DIR / "experiments" / "config" / "uniform_wrr.yaml"
)


class SummaryOnlyNANDServiceLog:
    """在不改变NAND事件执行的前提下丢弃逐4 KiB绘图明细。"""

    def append(self, event):
        """功能：接收但不保存一条NAND服务明细。

        目的：64 GPU三层会执行数百万个4 KiB命令；最终实验
        只需要完成时刻，因此避免为绘图明细占用大量内存。

        输入：后端已经执行的NAND服务事件字典。

        输出：无；不改变任何调度、反压或计数状态。
        """

    def __iter__(self):
        """功能：为SSD结果整理提供空迭代器。

        目的：保持后端`list(nand_service_events)`接口兼容，
        同时使summary-only实验不返回逐4 KiB明细。

        输入：无。

        输出：空迭代器。
        """
        return iter(())


def parse_arguments():
    """功能：读取FCFS CIR对照实验的命令行参数。

    目的：默认一条命令运行固定正常/过载负载，同时允许
    测试或临时实验替换YAML和唯一summary输出位置。

    输入：进程命令行。

    输出：
        argparse.Namespace: 配置文件和可选输出文件。
    """
    parser = argparse.ArgumentParser(
        description="Compare baseline and demand-aware FCFS CIR."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="experiment YAML path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override summary.json path",
    )
    return parser.parse_args()


def load_experiment_config(config_file):
    """功能：读取实验YAML并解析唯一输出文件路径。

    目的：使64 GPU、2 SSD、层范围、随机种子和两组命中率
    参数全部由一份独立配置控制，不改写项目默认YAML。

    输入：
        config_file: `fcfs_cir_comparison` YAML文件路径。

    输出：
        tuple: 实验配置字典和相对YAML目录解析后的summary路径。
    """
    config_file = Path(config_file).resolve()
    config = load_yaml(config_file)["fcfs_cir_comparison"]
    output_file = (config_file.parent / config["output_file"]).resolve()
    return config, output_file


def build_simulation_override(config, load_config):
    """功能：为一组正常或过载负载生成项目级仿真覆盖。

    目的：复用共同的64 GPU/2 SSD/种子6103配置，只替换
    当前负载的hit-ratio区间。两种QoS策略会收到完全相同
    的覆盖字典。

    输入：
        config: 完整实验配置。
        load_config: 当前负载的hit-ratio区间字典。

    输出：
        dict: 可传给 ``run_joint_simulation`` 的simulation递归覆盖。
    """
    simulation_override = json.loads(json.dumps(config["simulation"]))
    simulation_override["workload_generation"][
        "prefill_layer_hit_ratio_range"
    ] = list(load_config["prefill_layer_hit_ratio_range"])
    return simulation_override


def execution_identity(result):
    """功能：构造不包含时序结果的工作负载、Placement和Binding身份。

    目的：在写summary前确认Baseline与Demand-aware的GPU参数、
    Block→SSD映射和 ``(GPU, SSD)→Queue`` 映射完全相同。
    函数只在内存中构造比较键，不生成manifest或明细文件。

    输入：
        result: 一种策略的完整联合仿真结果。

    输出：
        tuple: GPU工作负载元组和按request_id排序的SSD/Queue映射。
    """
    workloads = tuple(
        (
            gpu_id,
            gpu_result["inferences"][0]["input_tokens"],
            gpu_result["inferences"][0]["prefill_layer_hit_ratio"],
        )
        for gpu_id, gpu_result in result["gpus"].items()
    )
    request_paths = []
    for storage_target_id, path_result in result["storage_paths"].items():
        for request in path_result["qos"]["dispatched_requests"]:
            request_paths.append((
                request["request_id"],
                storage_target_id,
                request["queue_id"],
            ))
    return workloads, tuple(sorted(request_paths))


def gpu_layer_samples(result, storage_target_ids):
    """功能：计算每个GPU、每一层的目标窗口、实际读取和有符号差值。

    目的：严格使用两块SSD对当前GPU层的最晚完成时刻之
    最大值。跨SSD时间不求和、不求平均。样本只用于内存统计，
    不写入summary。

    输入：
        result: 一种策略的完整联合仿真结果。
        storage_target_ids: 需要同时等待的SSD ID列表。

    输出：
        list[dict]: 192个GPU×Layer样本，包含actual和signed delta。
    """
    samples = []
    for gpu_id, gpu_result in result["gpus"].items():
        inference = gpu_result["inferences"][0]
        for layer in inference["layers"]:
            completion_times = layer["ssd_completion_times_us"]
            latest_storage_completion_us = max(
                completion_times[storage_target_id]
                for storage_target_id in storage_target_ids
            )
            layer_start_time_us = layer["layer_start_time_us"]
            target_window_us = (
                layer["compute_done_time_us"] - layer_start_time_us
            )
            actual_read_us = (
                latest_storage_completion_us - layer_start_time_us
            )
            samples.append({
                "gpu_id": gpu_id,
                "actual_read_us": actual_read_us,
                "signed_delta_us": actual_read_us - target_window_us,
            })
    return samples


def nearest_rank_p95(values):
    """功能：计算一组数值的nearest-rank P95。

    目的：对192个有符号delta使用固定、无插值的
    ``sorted[ceil(N*0.95)-1]`` 定义，避免不同统计库默认
    插值方式导致结果不一致。

    输入：
        values: 非空数值序列。

    输出：
        number: nearest-rank定义的P95值。
    """
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def summarize_policy(result, storage_target_ids):
    """功能：将一种策略压缩为用户指定的7个最终指标。

    目的：不输出逐GPU、逐层、逐SSD明细，也不泄露DPU内部
    requested/assigned CIR。最终summary只保留迟到数、delta分布、
    平均读取时间和两块SSD尾部完成时刻。

    输入：
        result: 一种策略的联合仿真结果。
        storage_target_ids: 按SSD0、SSD1排列的两块SSD ID。

    输出：
        tuple: 7项策略摘要和供配对统计使用的内存样本。
    """
    samples = gpu_layer_samples(result, storage_target_ids)
    deltas = [sample["signed_delta_us"] for sample in samples]
    actual_reads = [sample["actual_read_us"] for sample in samples]
    ssd0, ssd1 = storage_target_ids
    ssd0_last = result["storage_paths"][ssd0]["ssd"][
        "last_completion_time_us"
    ]
    ssd1_last = result["storage_paths"][ssd1]["ssd"][
        "last_completion_time_us"
    ]
    summary = {
        "late_gpu_layer_count": sum(delta > 0 for delta in deltas),
        "p95_delta_us": nearest_rank_p95(deltas),
        "worst_delta_us": max(deltas),
        "mean_actual_read_us": sum(actual_reads) / len(actual_reads),
        "SSD0_last_completion_us": ssd0_last,
        "SSD1_last_completion_us": ssd1_last,
        "overall_last_completion_us": max(ssd0_last, ssd1_last),
    }
    return summary, samples


def summarize_pair(baseline_samples, demand_aware_samples):
    """功能：按GPU比较两种策略的3层平均实际读取时间。

    目的：将每个GPU的3个 ``actual_read_us`` 取算术平均，
    Demand-aware更小记为improved，更大记为worsened，精确相等
    记为unchanged。

    输入：
        baseline_samples: Baseline的192个内存样本。
        demand_aware_samples: Demand-aware的192个内存样本。

    输出：
        dict: improved/worsened/unchanged GPU数量。
    """
    means_by_policy = []
    for samples in (baseline_samples, demand_aware_samples):
        reads_by_gpu = {}
        for sample in samples:
            reads_by_gpu.setdefault(sample["gpu_id"], []).append(
                sample["actual_read_us"]
            )
        means_by_policy.append({
            gpu_id: sum(reads) / len(reads)
            for gpu_id, reads in reads_by_gpu.items()
        })

    baseline_means, demand_aware_means = means_by_policy
    return {
        "improved_gpu_count": sum(
            demand_aware_means[gpu_id] < baseline_mean
            for gpu_id, baseline_mean in baseline_means.items()
        ),
        "worsened_gpu_count": sum(
            demand_aware_means[gpu_id] > baseline_mean
            for gpu_id, baseline_mean in baseline_means.items()
        ),
        "unchanged_gpu_count": sum(
            demand_aware_means[gpu_id] == baseline_mean
            for gpu_id, baseline_mean in baseline_means.items()
        ),
    }


def run_policy(policy, simulation_override, workload_override):
    """功能：为一种策略创建全新联合仿真并运行。

    目的：保留完整ASU阶段事件、反压、计数和完成时刻，
    但不保存最终summary完全不需要的逐4 KiB NAND绘图记录。

    输入：策略名、项目级simulation覆盖和LLM workload覆盖。

    输出：一次完整联合仿真的内存结果。
    """
    simulation = JointSimulation(
        binding_strategy_name="balanced_exclusive",
        rate_control_strategy_name=policy,
        simulation_config_override=simulation_override,
        workload_defaults_override=workload_override,
        token_config_file=TOKEN_CONFIG_FILE,
        scheduler_config_file=WRR_CONFIG_FILE,
    )
    for storage_path in simulation.storage_paths.values():
        # 只替换绘图明细容器；NAND命令仍在同一时刻启动、
        # 完成并参与全部上下游反压。
        storage_path.ssd.backend.nand_service_events = (
            SummaryOnlyNANDServiceLog()
        )
    return simulation.run()


def run_load(load_name, config):
    """功能：用相同负载连续运行Baseline和Demand-aware。

    目的：每种策略都重建GPU、QoS、SSD和事件日历，然后用
    ``execution_identity`` 确认工作负载、Block Placement和Queue
    Binding不变，最后只返回精简summary。

    输入：
        load_name: ``normal_load`` 或 ``overload_load``。
        config: 完整实验配置。

    输出：
        dict: 两种策略各7项指标和3项GPU配对计数。
    """
    simulation_override = build_simulation_override(
        config,
        config["loads"][load_name],
    )
    storage_target_ids = ["SSD0", "SSD1"]
    identities = {}
    summaries = {}
    samples_by_policy = {}
    for policy in config["policies"]:
        result = run_policy(
            policy,
            simulation_override,
            config["workload"],
        )
        identities[policy] = execution_identity(result)
        summaries[policy], samples_by_policy[policy] = summarize_policy(
            result,
            storage_target_ids,
        )
        # 本策略的完成时间和执行身份已提取；立即释放
        # 逐IO完成记录，避免两份完整结果同时常驻内存。
        del result

    if (
        identities["baseline"]
        != identities["demand_aware_fcfs_cir"]
    ):
        raise RuntimeError(
            "policies did not receive identical workload, placement and binding"
        )

    return {
        "baseline": summaries["baseline"],
        "demand_aware_fcfs_cir": summaries[
            "demand_aware_fcfs_cir"
        ],
        "paired": summarize_pair(
            samples_by_policy["baseline"],
            samples_by_policy["demand_aware_fcfs_cir"],
        ),
    }


def run_experiment(config):
    """功能：运行正常和过载两组完整对照实验。

    目的：以固定顺序生成最终summary字典，不保存任何逐GPU、
    逐层或逐SSD的中间文件。

    输入：
        config: 完整实验配置。

    输出：
        dict: ``normal_load`` 和 ``overload_load`` 两个摘要。
    """
    return {
        load_name: run_load(load_name, config)
        for load_name in ("normal_load", "overload_load")
    }


def write_and_print_summary(summary, output_file):
    """功能：将同一份最终JSON写入磁盘并原样打印到终端。

    目的：整个实验只创建 ``summary.json``，终端输出与文件
    内容使用同一次序列化，不生成CSV、manifest或明细报告。

    输入：
        summary: ``run_experiment`` 生成的最终字典。
        output_file: 唯一summary.json路径。

    输出：
        None: 写文件并向标准输出打印相同JSON。
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2, ensure_ascii=False)
    output_file.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def main():
    """功能：执行FCFS CIR对照实验的完整命令行流程。

    目的：串联配置读取、四次联合仿真、配对统计和唯一
    summary输出，作为用户可直接重现的独立实验入口。

    输入：进程命令行。
    输出：无。
    """
    arguments = parse_arguments()
    config, configured_output = load_experiment_config(arguments.config)
    output_file = (
        configured_output if arguments.output is None else arguments.output
    )
    write_and_print_summary(run_experiment(config), output_file)


if __name__ == "__main__":
    main()
