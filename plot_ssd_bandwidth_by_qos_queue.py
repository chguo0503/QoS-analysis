#!/usr/bin/env python3
"""按SSD分别画出真实NAND带宽，并反向归属到原始QoS Queue。

统计对象是后端NAND阶段真正启动的4 KiB内部命令，不使用QoS dispatch时间。
每块SSD使用一个独立子图；子图中的每条曲线表示一个实际服务过IO的QoS
Queue。所有SSD共用同一个时间原点和窗口边界，因此不同设备可以直接对齐观察。

带宽换算公式：
    GB/s = 窗口内NAND启动字节数 / (窗口长度us * 1000)
"""

import argparse
from pathlib import Path

import matplotlib

# 使用无图形界面的渲染后端，服务器环境执行脚本也可以直接生成PNG。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from qos_ssd_simulator import run_joint_simulation


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "ssd_bandwidth_by_qos_queue.png"
DEFAULT_WINDOW_US = 1000.0


def parse_arguments():
    """功能：读取后端SSD带宽图的命令行参数。

    目的：允许用户选择时间窗口、DPU绑定策略、SSD子集和输出文件，
    而不需要修改任何仿真原始代码。

    输入：
        无；由argparse读取当前进程命令行。

    输出：
        argparse.Namespace: 包含窗口、策略、SSD筛选、输出路径和DPI。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot each SSD's physical NAND bandwidth attributed to QoS queues."
        )
    )
    parser.add_argument(
        "--window-us",
        type=float,
        default=DEFAULT_WINDOW_US,
        help="fixed aggregation window in microseconds (default: 1000)",
    )
    parser.add_argument(
        "--binding-strategy",
        choices=("balanced_exclusive",),
        default=None,
        help="override the DPU queue binding strategy from YAML",
    )
    parser.add_argument(
        "--storage-target",
        action="append",
        default=None,
        help=(
            "plot only this SSD ID; repeat the option to select multiple SSDs "
            "(default: all configured SSDs)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="output PNG path",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="output image resolution (default: 160)",
    )
    return parser.parse_args()


def _select_ssd_results(simulation_result, requested_storage_targets=None):
    """功能：从联合仿真结果中取出需要绘图的SSD结果。

    目的：统一使用 ``storage_paths`` 多SSD命名空间，并在绘图前就报告
    用户输入的不存在SSD ID。

    输入：
        simulation_result: ``run_joint_simulation`` 返回的联合结果。
        requested_storage_targets: 可选的SSD ID列表；None表示全部SSD。

    输出：
        dict: 按拓扑顺序保存的 ``storage_target_id -> ssd_result`` 映射。
    """
    storage_paths = simulation_result["storage_paths"]
    if requested_storage_targets is None:
        selected_ids = list(storage_paths)
    else:
        # 用dict去重同时保留用户在命令行中给出的先后顺序。
        selected_ids = list(dict.fromkeys(requested_storage_targets))

    unknown_ids = [
        storage_target_id
        for storage_target_id in selected_ids
        if storage_target_id not in storage_paths
    ]
    if unknown_ids:
        available = ", ".join(storage_paths)
        raise ValueError(
            "unknown storage target(s): "
            f"{', '.join(unknown_ids)}; available targets: {available}"
        )
    if not selected_ids:
        raise ValueError("at least one storage target must be selected")
    return {
        storage_target_id: storage_paths[storage_target_id]["ssd"]
        for storage_target_id in selected_ids
    }


def _global_nand_time_range(ssd_results):
    """功能：计算参考SSD集合共用的NAND服务时间范围。

    目的：让每块SSD的固定窗口拥有相同时间原点和边界，防止各自从
    首个IO开始而导致子图看似对齐、实际错位。

    输入：
        ssd_results: 按SSD ID保存的后端结果映射。

    输出：
        tuple[float, float]: 全局首个和最后一个NAND启动时刻，单位us。
    """
    event_times = [
        event["start_time_us"]
        for ssd_result in ssd_results.values()
        for event in ssd_result["nand_service_events"]
    ]
    if not event_times:
        raise RuntimeError("selected SSDs produced no NAND service events")
    return min(event_times), max(event_times)


def aggregate_queue_bandwidth(
    ssd_result,
    window_us,
    timeline_start_us=None,
    timeline_end_us=None,
):
    """功能：按固定仿真时间窗口聚合一块SSD的分Queue NAND带宽。

    目的：根据真实NAND命令启动时刻反向归属物理服务字节，并补齐
    没有NAND服务的0值窗口。

    输入：
        ssd_result: 某个StoragePath的SSD最终结果。
        window_us: 固定聚合窗口长度，单位us。
        timeline_start_us: 可选共用时间轴开始时刻。
        timeline_end_us: 可选共用时间轴结束时刻。

    输出：
        dict: 时间轴、每Queue窗口字节和对应GB/s序列。
    """
    if window_us <= 0:
        raise ValueError("--window-us must be greater than 0")

    nand_events = ssd_result["nand_service_events"]
    if timeline_start_us is None or timeline_end_us is None:
        if not nand_events:
            raise RuntimeError("SSD produced no NAND service events")
        timeline_start_us = nand_events[0]["start_time_us"]
        timeline_end_us = nand_events[-1]["start_time_us"]
    if timeline_end_us < timeline_start_us:
        raise ValueError("timeline_end_us cannot precede timeline_start_us")

    # 最后一个启动事件所在的窗口也必须被创建，因此末尾加1。
    window_count = int(
        (timeline_end_us - timeline_start_us) // window_us
    ) + 1
    queue_ids = sorted({event["queue_id"] for event in nand_events})
    bytes_by_queue = {
        queue_id: [0] * window_count
        for queue_id in queue_ids
    }

    for event in nand_events:
        relative_time_us = event["start_time_us"] - timeline_start_us
        window_index = int(relative_time_us // window_us)
        if not 0 <= window_index < window_count:
            raise RuntimeError(
                "NAND event fell outside the shared plotting timeline"
            )
        bytes_by_queue[event["queue_id"]][window_index] += event["size_bytes"]

    bandwidth_by_queue = {
        queue_id: [
            # 1 GB/s等于1000 Byte/us，因此窗口Byte数除以us*1000。
            byte_count / (window_us * 1_000)
            for byte_count in byte_series
        ]
        for queue_id, byte_series in bytes_by_queue.items()
    }
    window_center_ms = [
        (window_index + 0.5) * window_us / 1_000
        for window_index in range(window_count)
    ]
    return {
        "storage_target_id": ssd_result["storage_target_id"],
        "timeline_start_us": timeline_start_us,
        "timeline_end_us": timeline_end_us,
        "window_us": window_us,
        "window_center_ms": window_center_ms,
        "bytes_by_queue": bytes_by_queue,
        "bandwidth_by_queue": bandwidth_by_queue,
    }


def validate_aggregation(ssd_result, series):
    """功能：校验单块SSD的NAND埋点、Queue归属和带宽上限。

    目的：在绘图前发现事件遗漏、字节重复归属或窗口换算突破物理
    NAND配置等统计错误。

    输入：
        ssd_result: 当前SSD的原始最终结果。
        series: ``aggregate_queue_bandwidth`` 返回的聚合序列。

    输出：
        dict: 事件数、物理字节、NAND容量和窗口总带宽峰值。
    """
    nand_events = ssd_result["nand_service_events"]
    nand_started_count = ssd_result["stage_statistics"]["NAND"]["started"]
    if len(nand_events) != nand_started_count:
        raise RuntimeError(
            f"NAND event count mismatch: events={len(nand_events)}, "
            f"started={nand_started_count}"
        )

    event_byte_count = sum(event["size_bytes"] for event in nand_events)
    grouped_byte_count = sum(
        sum(byte_series)
        for byte_series in series["bytes_by_queue"].values()
    )
    if grouped_byte_count != event_byte_count:
        raise RuntimeError(
            f"NAND byte attribution mismatch: grouped={grouped_byte_count}, "
            f"events={event_byte_count}"
        )

    window_count = len(series["window_center_ms"])
    bandwidth_by_queue = series["bandwidth_by_queue"]
    total_bandwidth = [
        sum(
            queue_series[window_index]
            for queue_series in bandwidth_by_queue.values()
        )
        for window_index in range(window_count)
    ]
    capacity_gb_s = (
        ssd_result["nand_read_bandwidth_bytes_per_second"] / 1_000_000_000
    )
    chunk_quantization_gb_s = (
        ssd_result["backend_chunk_size_bytes"]
        / (series["window_us"] * 1_000)
    )
    peak_total_bandwidth_gb_s = max(total_bandwidth, default=0.0)
    if (
        peak_total_bandwidth_gb_s
        > capacity_gb_s + chunk_quantization_gb_s + 1e-12
    ):
        raise RuntimeError(
            "NAND bandwidth exceeds configured capacity: "
            f"peak={peak_total_bandwidth_gb_s:.6f} GB/s, "
            f"capacity={capacity_gb_s:.6f} GB/s"
        )
    return {
        "event_count": len(nand_events),
        "physical_byte_count": event_byte_count,
        "capacity_gb_s": capacity_gb_s,
        "peak_total_bandwidth_gb_s": peak_total_bandwidth_gb_s,
    }


def build_storage_bandwidth_series(
    simulation_result,
    window_us,
    requested_storage_targets=None,
):
    """功能：为已选SSD构造时间对齐的分Queue带宽数据。

    目的：集中完成SSD筛选、全局时间轴建立和逐SSD一致性校验，使绘图
    函数只处理已验证的数据。

    输入：
        simulation_result: 多GPU/多SSD联合仿真结果。
        window_us: 固定聚合窗口长度，单位us。
        requested_storage_targets: 可选的SSD ID列表。

    输出：
        tuple[dict, dict]: 按SSD ID保存的带宽序列和校验结果。
    """
    selected_ssd_results = _select_ssd_results(
        simulation_result,
        requested_storage_targets,
    )
    # 时间轴始终由本次仿真的全部SSD建立。这样即使用户单独
    # 选中一块没有IO的SSD，也能看到它在整段仿真内为空闲，而不是报错。
    all_ssd_results = _select_ssd_results(simulation_result)
    timeline_start_us, timeline_end_us = _global_nand_time_range(
        all_ssd_results
    )

    series_by_storage_target = {}
    validation_by_storage_target = {}
    for storage_target_id, ssd_result in selected_ssd_results.items():
        series = aggregate_queue_bandwidth(
            ssd_result=ssd_result,
            window_us=window_us,
            timeline_start_us=timeline_start_us,
            timeline_end_us=timeline_end_us,
        )
        series_by_storage_target[storage_target_id] = series
        validation_by_storage_target[storage_target_id] = validate_aggregation(
            ssd_result,
            series,
        )
    return series_by_storage_target, validation_by_storage_target


def _bandwidth_to_utilization_functions(capacity_gb_s):
    """功能：为Matplotlib右侧使用率坐标轴创建双向换算函数。

    目的：让每块SSD子图同时显示GB/s和相对该盘NAND配置的百分比，
    右轴不会引入额外数据曲线。

    输入：
        capacity_gb_s: 当前SSD的NAND读带宽上限。

    输出：
        tuple[callable, callable]: GB/s到百分比及其反向换算函数。
    """
    def bandwidth_to_utilization(bandwidth_gb_s):
        """功能：把GB/s换算为NAND带宽使用率。

        目的：为右侧百分比坐标轴提供正向变换。

        输入：
            bandwidth_gb_s: 左轴上的带宽值。

        输出：
            float | array: 对应的使用率百分比。
        """
        return bandwidth_gb_s / capacity_gb_s * 100

    def utilization_to_bandwidth(utilization_percent):
        """功能：把NAND带宽使用率换算为GB/s。

        目的：为右侧百分比坐标轴提供反向变换。

        输入：
            utilization_percent: 右轴上的使用率百分比。

        输出：
            float | array: 对应的左轴GB/s值。
        """
        return utilization_percent / 100 * capacity_gb_s

    return bandwidth_to_utilization, utilization_to_bandwidth


def plot_storage_bandwidth(
    series_by_storage_target,
    validation_by_storage_target,
    output_file,
    dpi,
):
    """功能：为每块SSD创建一个子图并按QoS Queue绘制NAND带宽。

    目的：保持每块SSD独立的物理带宽上限，避免将不同SSD中同名Queue
    合并，同时让多个子图共享一条对齐时间轴。

    输入：
        series_by_storage_target: 每块SSD的分Queue带宽序列。
        validation_by_storage_target: 每块SSD已验证的容量与峰值。
        output_file: 目标PNG路径。
        dpi: PNG输出分辨率。

    输出：
        Path: 已生成PNG的绝对路径。
    """
    storage_count = len(series_by_storage_target)
    figure_height = max(5.2, 3.8 * storage_count)
    figure, axes = plt.subplots(
        storage_count,
        1,
        figsize=(12.5, figure_height),
        sharex=True,
        squeeze=False,
    )

    for row_index, (storage_target_id, series) in enumerate(
        series_by_storage_target.items()
    ):
        axis = axes[row_index][0]
        validation = validation_by_storage_target[storage_target_id]
        capacity_gb_s = validation["capacity_gb_s"]
        bandwidth_by_queue = series["bandwidth_by_queue"]

        # 每个实际进入该SSD之NAND阶段的QoS Queue只绘制一条曲线。
        for queue_id, bandwidth in bandwidth_by_queue.items():
            axis.plot(
                series["window_center_ms"],
                bandwidth,
                linewidth=1.5,
                label=f"QoS queue {queue_id}",
            )

        if not bandwidth_by_queue:
            # 保留未被Placement选中的SSD子图，明确表示本次实验无IO，
            # 不将“没有请求”误解为设备内部卡住。
            axis.text(
                0.5,
                0.5,
                "No NAND service requests",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )

        window_label = f"{series['window_us'] / 1_000:g} ms"
        axis.set_title(
            f"{storage_target_id}: NAND bandwidth by QoS queue "
            f"({window_label} windows)"
        )
        axis.set_ylabel("Bandwidth (GB/s)")
        axis.set_ylim(0, capacity_gb_s * 1.05)
        axis.grid(True, linewidth=0.6, alpha=0.3)

        # Queue数较少时显示完整图例；GPU较多时互斥绑定
        # 也可能激活数百个Queue，此时改为显示活跃Queue数。
        if 0 < len(bandwidth_by_queue) <= 12:
            axis.legend(loc="upper right")
        elif len(bandwidth_by_queue) > 12:
            axis.text(
                0.99,
                0.96,
                f"{len(bandwidth_by_queue)} active queues",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )

        conversion_functions = _bandwidth_to_utilization_functions(
            capacity_gb_s
        )
        utilization_axis = axis.secondary_yaxis(
            "right",
            functions=conversion_functions,
        )
        utilization_axis.set_ylabel("Utilization (%)")

    axes[-1][0].set_xlabel("Time since first selected NAND service (ms)")
    figure.tight_layout()

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=dpi)
    plt.close(figure)
    return output_file


def print_summary(
    series_by_storage_target,
    validation_by_storage_target,
    output_file,
):
    """功能：打印图片位置以及每块SSD和Queue的NAND带宽摘要。

    目的：为图形提供可人工复核的事件数、物理容量、总峰值和分Queue
    平均/峰值文本统计。

    输入：
        series_by_storage_target: 每块SSD的分Queue带宽序列。
        validation_by_storage_target: 每块SSD的校验摘要。
        output_file: 已生成图片的绝对路径。

    输出：
        None: 将人类可读摘要写入标准输出。
    """
    print(f"Output image: {output_file}")
    for storage_target_id, series in series_by_storage_target.items():
        validation = validation_by_storage_target[storage_target_id]
        print(
            f"{storage_target_id}: window={series['window_us'] / 1_000:g} ms, "
            f"capacity={validation['capacity_gb_s']:.3f} GB/s, "
            f"NAND events={validation['event_count']}, "
            "peak total="
            f"{validation['peak_total_bandwidth_gb_s']:.3f} GB/s"
        )
        for queue_id, bandwidth in series["bandwidth_by_queue"].items():
            average_gb_s = sum(bandwidth) / len(bandwidth)
            peak_gb_s = max(bandwidth)
            print(
                f"  Queue {queue_id:<8}: average={average_gb_s:.3f} GB/s, "
                f"peak={peak_gb_s:.3f} GB/s "
                f"({peak_gb_s / validation['capacity_gb_s'] * 100:.2f}%)"
            )


def main():
    """功能：运行联合仿真、聚合分Queue后端带宽并生成PNG。

    目的：提供一个不修改仿真模块的独立绘图入口，直接支持YAML中的
    任意GPU/SSD数量和两种DPU Queue绑定策略。

    输入：
        无；使用 ``parse_arguments`` 解析的命令行参数。

    输出：
        None: 生成PNG并打印统计摘要。
    """
    arguments = parse_arguments()
    simulation_result = run_joint_simulation(
        binding_strategy=arguments.binding_strategy,
    )
    series_by_storage_target, validation_by_storage_target = (
        build_storage_bandwidth_series(
            simulation_result=simulation_result,
            window_us=arguments.window_us,
            requested_storage_targets=arguments.storage_target,
        )
    )
    output_file = plot_storage_bandwidth(
        series_by_storage_target=series_by_storage_target,
        validation_by_storage_target=validation_by_storage_target,
        output_file=arguments.output,
        dpi=arguments.dpi,
    )
    print_summary(
        series_by_storage_target,
        validation_by_storage_target,
        output_file,
    )


if __name__ == "__main__":
    main()
