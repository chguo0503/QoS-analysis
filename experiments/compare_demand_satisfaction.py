#!/usr/bin/env python3
"""对比Baseline与需求感知策略的SSD读取需求满足率。"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from DPU import (  # noqa: E402
    DPURequestGateway,
    DemandAwareRateController,
    build_queue_binding_strategy,
)
from backends.asu_ssd import SSDSimulator, load_ssd_config  # noqa: E402
from discrete_simulation import EventLoop  # noqa: E402
from qos import build_qos_simulator, load_queue_layout  # noqa: E402
from simulation_common.storage_path import StoragePath  # noqa: E402


QOS_CONFIG_DIR = PROJECT_DIR / "qos" / "config"
BASELINE_TOKEN_CONFIG = (
    PROJECT_DIR / "experiments" / "config" / "uniform_baseline_token_bucket.yaml"
)
UNIFORM_WRR_CONFIG = (
    PROJECT_DIR / "experiments" / "config" / "uniform_wrr.yaml"
)
BACKEND_CONFIG = (
    PROJECT_DIR / "backends" / "asu_ssd" / "config" / "asu_backend_config.yaml"
)
POLICIES = ("baseline", "demand_aware")


def build_dpu_request(request_id, p_node_id, demand_id, size_bytes, rate):
    """功能：构造一个固定诉求的合成DPU读请求。

    目的：排除LLM随机长度和KV Placement的影响，只比较QoS对
    10份相同4 GB/s需求的速率与调度机会分配。

    输入：请求ID、P节点、需求ID、IO Byte数和整数Byte/s诉求。
    输出：带 ``basic`` 和 ``demand_bw`` 的DPU请求字典。
    """
    return {
        "basic": {
            "request_id": request_id,
            "p_node_id": p_node_id,
            "storage_target_id": "SSD0",
            "size_bytes": size_bytes,
        },
        "demand_bw": {
            "demand_group_id": demand_id,
            "aggregate_required_bytes_per_second": rate,
        },
    }


def run_policy(policy, arguments):
    """功能：运行一次单SSD合成需求满足率实验。

    目的：Baseline使用CIR=0、PIR=uncapped和全1 WRR；需求感知
    策略在相同互斥Queue上设置每路径CIR=PIR，并动态设置Group WRR。

    输入：策略名和命令行实验参数。
    输出：需求满足数/比例、每需求完成时刻和绑定分布。
    """
    event_loop = EventLoop()
    queue_layout = load_queue_layout(
        QOS_CONFIG_DIR / "queue_layout_config.yaml"
    )
    qos = build_qos_simulator(
        layout_config_file=QOS_CONFIG_DIR / "queue_layout_config.yaml",
        token_config_file=BASELINE_TOKEN_CONFIG,
        scheduler_config_file=UNIFORM_WRR_CONFIG,
        qos_runtime_config_file=QOS_CONFIG_DIR / "qos_runtime_config.yaml",
        start_time_us=0,
        queue_layout=queue_layout,
    )

    request_to_demand = {}
    completion_times_by_demand = {}
    completed_io_count = 0

    def on_storage_complete(completion):
        """功能：将SSD完成的IO归并到所属合成需求。

        目的：每份需求只用最后一个IO的SSD完成时刻与GPU计算
        窗口比较，QoS dispatch_time不参与需求满足判定。

        输入：SSD发布的完成记录。
        输出：无；更新需求最晚完成时刻和全局IO完成数。
        """
        nonlocal completed_io_count
        demand_id = request_to_demand[completion["request_id"]]
        completion_times_by_demand[demand_id] = max(
            completion_times_by_demand.get(demand_id, 0),
            completion["completion_time_us"],
        )
        completed_io_count += 1

    backend_config = load_ssd_config(BACKEND_CONFIG)
    ssd = SSDSimulator(
        backend_config=deepcopy(backend_config),
        completion_sink=on_storage_complete,
        storage_target_id="SSD0",
    )
    storage_path = StoragePath("SSD0", qos, ssd, event_loop)
    p_node_ids = [f"P{index}" for index in range(arguments.demand_count)]
    queues_by_target = {"SSD0": queue_layout.queue_order}
    binding = build_queue_binding_strategy(
        "random_unique_sticky",
        arguments.binding_seed,
        p_node_ids,
        queues_by_target,
    )
    controller = None
    if policy == "demand_aware":
        controller = DemandAwareRateController(
            {"SSD0": backend_config["nand"]["read_bandwidth_bytes_per_second"]},
            {"SSD0": queue_layout.queue_to_group},
        )
    dpu = DPURequestGateway(
        queue_ids_by_storage_target=queues_by_target,
        queue_binding_strategy=binding,
        request_sink=storage_path.input,
        qos_interfaces_by_storage_target={"SSD0": qos},
        rate_controller=controller,
    )

    for demand_index, p_node_id in enumerate(p_node_ids):
        demand_id = f"demand_{demand_index:02d}"
        requests = []
        for io_index in range(arguments.io_count_per_demand):
            request_id = f"{demand_id}_io_{io_index:04d}"
            request_to_demand[request_id] = demand_id
            requests.append(build_dpu_request(
                request_id=request_id,
                p_node_id=p_node_id,
                demand_id=demand_id,
                size_bytes=arguments.io_size_bytes,
                rate=arguments.rate_bytes_per_second,
            ))
        dpu.submit_batch(requests, arrival_time_us=0)

    total_io_count = arguments.demand_count * arguments.io_count_per_demand
    started = time.perf_counter()
    event_loop.run_until(lambda: completed_io_count == total_io_count)
    wall_time_s = time.perf_counter() - started
    path_result = storage_path.end()

    transfer_window_us = (
        arguments.io_count_per_demand
        * arguments.io_size_bytes
        * 1_000_000
        / arguments.rate_bytes_per_second
    )
    compute_done_time_us = transfer_window_us + arguments.deadline_slack_us
    demand_results = []
    for demand_index in range(arguments.demand_count):
        demand_id = f"demand_{demand_index:02d}"
        completion_time_us = completion_times_by_demand[demand_id]
        demand_results.append({
            "demand_id": demand_id,
            "queue_id": binding.bindings[(f"P{demand_index}", "SSD0")],
            "compute_done_time_us": compute_done_time_us,
            "io_completion_time_us": completion_time_us,
            "satisfied": completion_time_us <= compute_done_time_us,
        })

    satisfied_count = sum(item["satisfied"] for item in demand_results)
    group_distribution = Counter(
        queue_layout.queue_to_group[item["queue_id"]]
        for item in demand_results
    )
    dispatched_requests = path_result["qos"]["dispatched_requests"]
    dpu_statistics = dpu.statistics()
    return {
        "policy": policy,
        "satisfied_demand_count": satisfied_count,
        "total_demand_count": len(demand_results),
        "satisfaction_ratio": satisfied_count / len(demand_results),
        "configured_rate_gb_s_per_demand": (
            arguments.rate_bytes_per_second / 1_000_000_000
        ),
        "aggregate_configured_rate_gb_s": (
            arguments.demand_count
            * arguments.rate_bytes_per_second
            / 1_000_000_000
        ),
        "nominal_transfer_window_us": transfer_window_us,
        "deadline_slack_us": arguments.deadline_slack_us,
        "compute_done_time_us": compute_done_time_us,
        "group_distribution": dict(sorted(group_distribution.items())),
        "dpu": {
            "binding_strategy": dpu_statistics["strategy"],
            "rate_control_strategy": (
                None
                if dpu_statistics["rate_control"] is None
                else dpu_statistics["rate_control"]["strategy"]
            ),
            "rate_control_write_count": dpu_statistics[
                "rate_control_write_count"
            ],
            "group_weight_write_count": dpu_statistics[
                "group_weight_write_count"
            ],
        },
        "qos": {
            "cir_dispatched_request_count": path_result["qos"][
                "cir_dispatched_request_count"
            ],
            "excess_dispatched_request_count": path_result["qos"][
                "excess_dispatched_request_count"
            ],
            "first_dispatch_time_us": min(
                request["dispatch_time_us"] for request in dispatched_requests
            ),
            "last_dispatch_time_us": max(
                request["dispatch_time_us"] for request in dispatched_requests
            ),
        },
        "ssd": {
            "first_submit_time_us": path_result["ssd"]["first_submit_time_us"],
            "last_completion_time_us": path_result["ssd"][
                "last_completion_time_us"
            ],
        },
        "demand_results": demand_results,
        "wall_time_s": wall_time_s,
    }


def write_result(results, arguments):
    """功能：将两种策略的需求满足率和明细写入JSON。

    目的：保留可复现参数和每份需求完成时刻，不生成TTFT或
    带宽图，使本实验只聚焦用户定义的需求满足率。

    输入：Baseline/需求感知结果和命令行参数。
    输出：无；创建输出目录并写入 ``results.json``。
    """
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": {
            "demand_count": arguments.demand_count,
            "rate_bytes_per_second_per_demand": (
                arguments.rate_bytes_per_second
            ),
            "io_count_per_demand": arguments.io_count_per_demand,
            "io_size_bytes": arguments.io_size_bytes,
            "deadline_slack_us": arguments.deadline_slack_us,
            "binding_seed": arguments.binding_seed,
        },
        "results": results,
    }
    (arguments.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def print_result(results):
    """功能：仅打印两种策略的需求满足率对比。

    目的：让命令行输出直接回答“在GPU计算窗口内完成了多少
    份SSD读取需求”，详细完成时刻保留在JSON中。

    输入：两种策略的结果列表。
    输出：无；向标准输出写入简表。
    """
    first = results[0]
    print(
        f"{first['total_demand_count']} demands x "
        f"{first['configured_rate_gb_s_per_demand']:.3f} GB/s -> 1 SSD"
    )
    print(
        f"nominal GPU window={first['nominal_transfer_window_us']:.3f} us, "
        f"deadline slack={first['deadline_slack_us']:.3f} us"
    )
    print(f"{'policy':<18}{'satisfied':>14}{'ratio':>14}")
    for result in results:
        print(
            f"{result['policy']:<18}"
            f"{result['satisfied_demand_count']:>5}/"
            f"{result['total_demand_count']:<8}"
            f"{result['satisfaction_ratio']:>13.2%}"
        )


def parse_arguments():
    """功能：读取合成需求满足率实验参数。

    目的：默认构造10份4 GB/s需求；125个144 KiB IO恰好对应
    4608 us的名义传输窗口，可选slack用于单独观察固定流水线延迟。

    输入：进程命令行。
    输出：argparse Namespace。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-count", type=int, default=10)
    parser.add_argument(
        "--rate-bytes-per-second",
        type=int,
        default=4_000_000_000,
    )
    parser.add_argument("--io-count-per-demand", type=int, default=125)
    parser.add_argument("--io-size-bytes", type=int, default=147_456)
    parser.add_argument("--deadline-slack-us", type=float, default=0)
    parser.add_argument("--binding-seed", type=int, default=5102)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "experiments"
            / "results"
            / "demand_satisfaction_10x4"
        ),
    )
    return parser.parse_args()


def main():
    """功能：顺序运行Baseline和需求感知对照实验。

    目的：两次仿真使用相同随机互斥Queue绑定、请求和SSD配置，
    使需求满足率差异只来自Queue速率和Group权重控制。

    输入：无；读取命令行。
    输出：无；打印简表并写入JSON结果。
    """
    arguments = parse_arguments()
    results = [run_policy(policy, arguments) for policy in POLICIES]
    write_result(results, arguments)
    print_result(results)


if __name__ == "__main__":
    main()
