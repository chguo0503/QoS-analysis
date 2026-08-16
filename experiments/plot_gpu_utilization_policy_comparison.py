"""Plot the exact 1–10 SSD utilization comparison for three DPU policies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
UNIFIED_RESULT = RESULTS_ROOT / "unified_topology_scan" / "summary.json"
STRICT_ONE_SSD_RESULT = (
    RESULTS_ROOT
    / "utility_edf_strict_80us_validated"
    / "summary.json"
)
STRICT_CHUNK_RESULTS = (
    RESULTS_ROOT
    / "utility_edf_strict_80us_topology_chunks"
    / "ssd_2_4.json",
    RESULTS_ROOT
    / "utility_edf_strict_80us_topology_chunks"
    / "ssd_5_7.json",
    RESULTS_ROOT
    / "utility_edf_strict_80us_topology_chunks"
    / "ssd_8_10.json",
)
OUTPUT_DIR = RESULTS_ROOT / "utility_edf_policy_comparison"
OUTPUT_STEM = "gpu_utilization_vs_ssd_count_policy_comparison"

BASELINE_POLICY = "baseline"
LEGACY_POLICY = "demand_aware_fcfs_cir"
NEW_POLICY = "utility_edf_integer_l750"

EXPECTED_EXPERIMENT = {
    "gpu_count": 128,
    "inference_count_per_gpu": 1,
    "batch_size": 1,
    "effective_compute_tflops": 512.0,
    "first_layer_index": 0,
    "last_layer_index": 3,
    "seed": 6103,
    "input_tokens_range": [100_000, 200_000],
    "hit_ratio_range": [0.5, 0.99],
    "placement_strategy": "random",
    "queue_binding_strategy": "balanced_exclusive",
    "backend_execution_mode": "batched_exact",
    "backend_batch_commands": 32,
}
EXPECTED_REQUEST_COUNT = 456_116
EXPECTED_BYTE_COUNT = 67_257_040_896


def load_result(path: Path) -> dict:
    """Load one summary and reject a mismatched experimental identity."""
    document = json.loads(path.read_text(encoding="utf-8"))
    experiment = document["experiment"]
    for field, expected in EXPECTED_EXPERIMENT.items():
        actual = experiment.get(field)
        if actual != expected:
            raise RuntimeError(
                f"{path}: experiment.{field}={actual!r}, expected {expected!r}"
            )
    return document


def validate_common_summary(
    summary: dict,
    source: Path,
    topology: str,
    *,
    require_completion_audit: bool = False,
) -> None:
    """Check workload conservation and all completion fields that exist.

    The unified baseline/legacy artifact predates the explicit completion and
    starvation fields.  Its request/byte conservation fields are still
    mandatory; strict Utility+EDF artifacts must additionally contain and pass
    the newer completion audit.
    """
    expected = {
        "request_count": EXPECTED_REQUEST_COUNT,
        "byte_count": EXPECTED_BYTE_COUNT,
    }
    completion_audit = {
        "completed_inference_count": 128,
        "starvation_free": True,
    }
    for field, expected_value in expected.items():
        actual = summary.get(field)
        if actual != expected_value:
            raise RuntimeError(
                f"{source}:{topology}: {field}={actual!r}, "
                f"expected {expected_value!r}"
            )
    for field, expected_value in completion_audit.items():
        if field not in summary and not require_completion_audit:
            continue
        actual = summary.get(field)
        if actual != expected_value:
            raise RuntimeError(
                f"{source}:{topology}: {field}={actual!r}, "
                f"expected {expected_value!r}"
            )


def validate_strict_utility(
    summary: dict,
    source: Path,
    topology: str,
    ssd_count: int,
) -> None:
    """Check the strict 80-us and no-starvation invariants for Utility+EDF."""
    validate_common_summary(
        summary,
        source,
        topology,
        require_completion_audit=True,
    )
    rate_control = summary["rate_control"]
    expected_periods = {f"SSD{index}": 80 for index in range(ssd_count)}
    checks = {
        "group_weight_write_count": 0,
        "control_update_non_tick_write_count": 0,
        "control_update_period_us_by_storage_target": expected_periods,
    }
    for field, expected in checks.items():
        if summary.get(field) != expected:
            raise RuntimeError(
                f"{source}:{topology}: {field}={summary.get(field)!r}, "
                f"expected {expected!r}"
            )
    if summary["control_update_tick_aligned_write_count"] != (
        summary["rate_control_write_count"]
        + summary["queue_weight_write_count"]
        + summary["group_weight_write_count"]
    ):
        raise RuntimeError(f"{source}:{topology}: control write audit failed")
    if rate_control.get("active_demand_count") != 0:
        raise RuntimeError(f"{source}:{topology}: active demand remains")
    if rate_control.get("completed_coflow_count") != 512:
        raise RuntimeError(f"{source}:{topology}: coflow completion mismatch")
    if rate_control.get("starved_p_node_count") != 0:
        raise RuntimeError(f"{source}:{topology}: starved p_node detected")


def collect_curves() -> tuple[list[int], list[float], list[float], list[float]]:
    """Collect the three exact curves from their machine-readable summaries."""
    unified = load_result(UNIFIED_RESULT)
    strict_one = load_result(STRICT_ONE_SSD_RESULT)
    strict_documents = [strict_one]
    strict_documents.extend(load_result(path) for path in STRICT_CHUNK_RESULTS)

    utility_points: dict[int, tuple[dict, Path, str]] = {}
    strict_sources = (STRICT_ONE_SSD_RESULT,) + STRICT_CHUNK_RESULTS
    for document, source in zip(strict_documents, strict_sources):
        for topology, point in document["topologies"].items():
            ssd_count = int(topology.split("_", 1)[0])
            if ssd_count in utility_points:
                raise RuntimeError(f"duplicate Utility+EDF point for {ssd_count} SSD")
            utility_points[ssd_count] = (point[NEW_POLICY], source, topology)

    ssd_counts = list(range(1, 11))
    if sorted(utility_points) != ssd_counts:
        raise RuntimeError(
            f"Utility+EDF topology coverage is {sorted(utility_points)}, "
            f"expected {ssd_counts}"
        )

    baseline_values = []
    legacy_values = []
    utility_values = []
    for ssd_count in ssd_counts:
        topology = f"{ssd_count}_ssd"
        unified_point = unified["topologies"][topology]
        baseline = unified_point[BASELINE_POLICY]
        legacy = unified_point[LEGACY_POLICY]
        utility, source, source_topology = utility_points[ssd_count]
        validate_common_summary(baseline, UNIFIED_RESULT, topology)
        validate_common_summary(legacy, UNIFIED_RESULT, topology)
        validate_strict_utility(
            utility,
            source,
            source_topology,
            ssd_count,
        )
        baseline_values.append(baseline["mean_gpu_utilization_percent"])
        legacy_values.append(legacy["mean_gpu_utilization_percent"])
        utility_values.append(utility["mean_gpu_utilization_percent"])

    strict_baseline = strict_one["topologies"]["1_ssd"][BASELINE_POLICY]
    if strict_baseline["mean_gpu_utilization_percent"] != baseline_values[0]:
        raise RuntimeError("the strict and unified 1-SSD baselines differ")
    return ssd_counts, baseline_values, legacy_values, utility_values


def write_csv(
    output_path: Path,
    ssd_counts: list[int],
    baseline_values: list[float],
    legacy_values: list[float],
    utility_values: list[float],
) -> None:
    """Write exact plotted values in a compact wide table."""
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "ssd_count",
            "baseline_mean_gpu_utilization_percent",
            "legacy_fcfs_cir_mean_gpu_utilization_percent",
            "utility_edf_integer_l750_mean_gpu_utilization_percent",
        ))
        writer.writerows(zip(
            ssd_counts,
            baseline_values,
            legacy_values,
            utility_values,
        ))


def draw_plot(
    png_path: Path,
    svg_path: Path,
    ssd_counts: list[int],
    baseline_values: list[float],
    legacy_values: list[float],
    utility_values: list[float],
) -> None:
    """Render a reference-style comparison plot as PNG and vector SVG."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    })
    figure, axis = plt.subplots(figsize=(14, 8.5), dpi=160)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    axis.plot(
        ssd_counts,
        baseline_values,
        color="#6B7280",
        marker="o",
        linewidth=3.2,
        markersize=9,
        label="Baseline",
    )
    axis.plot(
        ssd_counts,
        legacy_values,
        color="#2563EB",
        marker="s",
        linewidth=3.2,
        markersize=9,
        label="Legacy FCFS-CIR",
    )
    axis.plot(
        ssd_counts,
        utility_values,
        color="#DC2626",
        marker="^",
        linewidth=3.2,
        markersize=10,
        label="Utility+EDF (New)",
    )

    axis.set_title(
        "GPU Utilization vs. SSD Count: 128 GPUs, Batch 1, Layers 0–3",
        fontsize=20,
        pad=16,
    )
    axis.set_xlabel("SSD count (40 GB/s each)", fontsize=15)
    axis.set_ylabel("Mean GPU utilization (%)", fontsize=15)
    axis.set_xlim(0.55, 10.45)
    axis.set_ylim(0, 100)
    axis.set_xticks(ssd_counts)
    axis.set_yticks(range(0, 101, 20))
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(True, color="#D1D5DB", alpha=0.62, linewidth=1.0)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=14)

    figure.text(
        0.5,
        0.012,
        "Full batched-exact simulation · 512 TFLOPS/GPU · "
        "seed 6103 · random KV placement · strict 80 µs Utility control",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.995, 0.985))
    figure.savefig(png_path, dpi=160, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Validate sources, export the value table, and draw both formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ssd_counts, baseline, legacy, utility = collect_curves()
    csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    write_csv(csv_path, ssd_counts, baseline, legacy, utility)
    draw_plot(png_path, svg_path, ssd_counts, baseline, legacy, utility)
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    for values in zip(ssd_counts, baseline, legacy, utility):
        print(
            f"SSD={values[0]:2d} baseline={values[1]:.6f}% "
            f"legacy_fcfs={values[2]:.6f}% utility_edf={values[3]:.6f}%"
        )
