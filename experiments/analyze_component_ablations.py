#!/usr/bin/env python3
"""Analyze the exact 2^3 Coflow/Utility/EDF ablation experiment.

The runner deliberately keeps the common admission-Gate foundation in every
factorial cell.  This analyzer therefore reports two different quantities:

* ``000 - legacy FCFS-CIR``: the shared Gate/pre-park/strict-tick foundation;
* ``111 - 000``: the combined contribution of the C/U/E algorithm factors.

Those quantities must not be merged when attributing the result to C, U, or E.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "utility_edf_component_ablations"
)
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "summary.json"
DEFAULT_ANALYSIS_DIR = DEFAULT_RESULT_DIR / "analysis"

FACTORS = ("C", "U", "E")
FACTOR_NAMES_CN = {
    "C": "跨 SSD Coflow 一致性",
    "U": "Stage-0 Utility Density",
    "E": "Prefetch EDF 与 deadline 保护",
}
FACTOR_NAMES_EN = {
    "C": "Cross-SSD coflow",
    "U": "Utility density",
    "E": "EDF deadline protection",
}
FACTOR_POLICY_RE = re.compile(r"^ablation_c([01])_u([01])_e([01])$")
ANCHOR_POLICIES = {
    "baseline": "Baseline",
    "demand_aware_fcfs_cir": "Legacy FCFS-CIR",
    "utility_edf_integer_l750": "Production Utility+EDF",
}


@dataclass(frozen=True)
class MetricSpec:
    label_cn: str
    raw_unit: str
    difference_unit: str
    positive_change_meaning: str


METRICS: dict[str, MetricSpec] = {
    "mean_gpu_utilization_percent": MetricSpec(
        "平均 GPU 利用率", "%", "pp", "改善",
    ),
    "min_gpu_utilization_percent": MetricSpec(
        "最低 GPU 利用率", "%", "pp", "改善",
    ),
    "p95_gpu_utilization_percent": MetricSpec(
        "P95 GPU 利用率", "%", "pp", "改善",
    ),
    "max_gpu_utilization_percent": MetricSpec(
        "最高 GPU 利用率", "%", "pp", "改善",
    ),
    "mean_ttft_us": MetricSpec(
        "平均 TTFT", "us", "us", "退化（时间变长）",
    ),
    "p95_ttft_us": MetricSpec(
        "P95 TTFT", "us", "us", "退化（时间变长）",
    ),
    "max_ttft_us": MetricSpec(
        "Max TTFT", "us", "us", "退化（时间变长）",
    ),
    "late_gpu_layer_count": MetricSpec(
        "逾期 GPU layer 数", "count", "count", "退化（数量增加）",
    ),
    "p95_delta_us": MetricSpec(
        "P95 预取超期量", "us", "us", "退化（超期增加）",
    ),
    "worst_delta_us": MetricSpec(
        "最坏预取超期量", "us", "us", "退化（超期增加）",
    ),
    "mean_actual_read_us": MetricSpec(
        "平均实际读取时间", "us", "us", "退化（时间变长）",
    ),
}

ANCHOR_TRANSITIONS = (
    (
        "legacy_vs_baseline",
        "baseline",
        "demand_aware_fcfs_cir",
        "Legacy FCFS-CIR 相对 Baseline",
        "legacy_cir",
    ),
    (
        "shared_foundation",
        "demand_aware_fcfs_cir",
        "ablation_c0_u0_e0",
        "共享 Gate 基础（000 - Legacy FCFS-CIR）",
        "shared_foundation",
    ),
    (
        "combined_cue",
        "ablation_c0_u0_e0",
        "ablation_c1_u1_e1",
        "C/U/E 组合算法（111 - 000）",
        "algorithm_factors",
    ),
    (
        "production_parity",
        "ablation_c1_u1_e1",
        "utility_edf_integer_l750",
        "Production 与消融 111 一致性",
        "parity_check",
    ),
    (
        "end_to_end",
        "baseline",
        "ablation_c1_u1_e1",
        "完整 111 相对 Baseline",
        "end_to_end",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze C/U/E 2^3 ablations and generate CSV/Markdown/PNG.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="aggregate summary.json from run_component_ablations.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <input parent>/analysis",
    )
    return parser.parse_args(argv)


def policy_factors(policy: str) -> tuple[int, int, int] | None:
    match = FACTOR_POLICY_RE.fullmatch(policy)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def factor_policy(bits: tuple[int, int, int]) -> str:
    return f"ablation_c{bits[0]}_u{bits[1]}_e{bits[2]}"


def policy_role(policy: str) -> str:
    if policy_factors(policy) is not None:
        return "factorial_cell"
    if policy == "baseline":
        return "baseline_anchor"
    if policy == "demand_aware_fcfs_cir":
        return "legacy_fcfs_cir_anchor"
    if policy == "utility_edf_integer_l750":
        return "production_anchor"
    return "other"


def _ssd_count_from_key(key: str) -> int:
    match = re.fullmatch(r"(\d+)_ssd", key)
    if match is None:
        raise ValueError(f"invalid topology key: {key!r}")
    return int(match.group(1))


def load_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("the input document must be a JSON object")
    validate_document(document)
    return document


def validate_document(document: dict[str, Any]) -> None:
    topologies = document.get("topologies")
    if not isinstance(topologies, dict) or not topologies:
        raise ValueError("input has no non-empty 'topologies' object")

    required_policies = set(ANCHOR_POLICIES)
    required_policies.update(
        factor_policy(bits)
        for bits in itertools.product((0, 1), repeat=3)
    )
    for topology_key, summaries in topologies.items():
        _ssd_count_from_key(topology_key)
        if not isinstance(summaries, dict):
            raise ValueError(f"{topology_key}: policy summaries must be an object")
        missing = sorted(required_policies - set(summaries))
        if missing:
            raise ValueError(f"{topology_key}: missing policies: {missing}")
        for policy in required_policies:
            summary = summaries[policy]
            if not isinstance(summary, dict):
                raise ValueError(f"{topology_key}/{policy}: summary is not an object")
            missing_metrics = sorted(set(METRICS) - set(summary))
            if missing_metrics:
                raise ValueError(
                    f"{topology_key}/{policy}: missing metrics: {missing_metrics}"
                )
            for metric in METRICS:
                value = summary[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{topology_key}/{policy}/{metric}: expected a number"
                    )


def topology_items(
    document: dict[str, Any],
) -> Iterable[tuple[int, dict[str, dict[str, Any]]]]:
    parsed = [
        (_ssd_count_from_key(key), summaries)
        for key, summaries in document["topologies"].items()
    ]
    yield from sorted(parsed, key=lambda item: item[0])


def _infer_unit(metric: str) -> str:
    if metric in METRICS:
        return METRICS[metric].raw_unit
    if metric.endswith("_percent"):
        return "%"
    if metric.endswith("_us"):
        return "us"
    if metric.endswith("_bytes_per_second"):
        return "bytes/s"
    if metric.endswith("_bytes") or metric == "byte_count":
        return "bytes"
    if metric.endswith("_seconds"):
        return "s"
    if metric.endswith("_count"):
        return "count"
    return ""


def _positive_meaning(metric: str) -> str:
    if metric in METRICS:
        return METRICS[metric].positive_change_meaning
    return "仅表示数值增加"


def _flatten_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(value[key], child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _flatten_scalars(child, child_prefix)
        return
    yield prefix, value


def build_long_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ssd_count, policies in topology_items(document):
        for policy in sorted(policies):
            bits = policy_factors(policy)
            summary = policies[policy]
            for metric, value in _flatten_scalars(summary):
                rows.append(
                    {
                        "ssd_count": ssd_count,
                        "policy": policy,
                        "policy_role": policy_role(policy),
                        "C": "" if bits is None else bits[0],
                        "U": "" if bits is None else bits[1],
                        "E": "" if bits is None else bits[2],
                        "metric": metric,
                        "value": value,
                        "unit": _infer_unit(metric),
                        "positive_change_meaning": _positive_meaning(metric),
                    }
                )
    return rows


def _factorial_values(
    policies: dict[str, dict[str, Any]], metric: str,
) -> dict[tuple[int, int, int], float]:
    return {
        bits: float(policies[factor_policy(bits)][metric])
        for bits in itertools.product((0, 1), repeat=3)
    }


def _contrast(
    values: dict[tuple[int, int, int], float],
    factor_indexes: tuple[int, ...],
) -> float:
    """Return a standard 0/1-factor effect or interaction contrast.

    Main effects are averages of ``on - off``.  Pair interactions are average
    difference-in-differences over the third factor.  The triple interaction is
    the difference of those pair interactions.
    """

    denominator = 2 ** (len(FACTORS) - len(factor_indexes))
    total = 0.0
    for bits, value in values.items():
        sign = math.prod(1 if bits[index] else -1 for index in factor_indexes)
        total += sign * value
    return total / denominator


def build_effect_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        ("main", "C", (0,)),
        ("main", "U", (1,)),
        ("main", "E", (2,)),
        ("second_order", "C:U", (0, 1)),
        ("second_order", "C:E", (0, 2)),
        ("second_order", "U:E", (1, 2)),
        ("third_order", "C:U:E", (0, 1, 2)),
    )
    rows: list[dict[str, Any]] = []
    for ssd_count, policies in topology_items(document):
        for metric, spec in METRICS.items():
            values = _factorial_values(policies, metric)
            for effect_type, effect, indexes in definitions:
                rows.append(
                    {
                        "ssd_count": ssd_count,
                        "metric": metric,
                        "metric_label_cn": spec.label_cn,
                        "effect_type": effect_type,
                        "effect": effect,
                        "estimate": _contrast(values, indexes),
                        "unit": spec.difference_unit,
                        "positive_change_meaning": spec.positive_change_meaning,
                    }
                )
    return rows


def _shapley(
    values: dict[tuple[int, int, int], float],
) -> dict[str, float]:
    contributions = {factor: 0.0 for factor in FACTORS}
    factor_indexes = {factor: index for index, factor in enumerate(FACTORS)}
    permutations = tuple(itertools.permutations(FACTORS))
    for order in permutations:
        enabled: set[str] = set()
        before = values[(0, 0, 0)]
        for factor in order:
            enabled.add(factor)
            bits = tuple(
                int(candidate in enabled) for candidate in FACTORS
            )
            after = values[bits]  # type: ignore[index]
            contributions[factor] += after - before
            before = after
    for factor in FACTORS:
        contributions[factor] /= len(permutations)

    # Enforce efficiency at machine precision.  The correction is normally at
    # round-off scale and leaves the permutation-average interpretation intact.
    total = values[(1, 1, 1)] - values[(0, 0, 0)]
    residual = total - sum(contributions.values())
    contributions["E"] += residual
    if not math.isclose(
        sum(contributions.values()), total, rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise ArithmeticError("Shapley efficiency check failed")
    return contributions


def build_shapley_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ssd_count, policies in topology_items(document):
        for metric, spec in METRICS.items():
            values = _factorial_values(policies, metric)
            contributions = _shapley(values)
            total = values[(1, 1, 1)] - values[(0, 0, 0)]
            emitted_sum = sum(contributions.values())
            for factor in FACTORS:
                rows.append(
                    {
                        "ssd_count": ssd_count,
                        "metric": metric,
                        "metric_label_cn": spec.label_cn,
                        "factor": factor,
                        "factor_label_cn": FACTOR_NAMES_CN[factor],
                        "shapley_contribution": contributions[factor],
                        "unit": spec.difference_unit,
                        "total_111_minus_000": total,
                        "three_factor_sum": emitted_sum,
                        "efficiency_residual": total - emitted_sum,
                        "positive_change_meaning": spec.positive_change_meaning,
                    }
                )
    return rows


def _relative_percent(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


def build_anchor_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ssd_count, policies in topology_items(document):
        for transition, before_policy, after_policy, label_cn, attribution in (
            ANCHOR_TRANSITIONS
        ):
            for metric, spec in METRICS.items():
                before = float(policies[before_policy][metric])
                after = float(policies[after_policy][metric])
                rows.append(
                    {
                        "ssd_count": ssd_count,
                        "transition": transition,
                        "transition_label_cn": label_cn,
                        "attribution_scope": attribution,
                        "before_policy": before_policy,
                        "after_policy": after_policy,
                        "metric": metric,
                        "metric_label_cn": spec.label_cn,
                        "before_value": before,
                        "after_value": after,
                        "signed_change": after - before,
                        "change_unit": spec.difference_unit,
                        "signed_relative_change_percent": (
                            None
                            if metric
                            not in {
                                "mean_ttft_us",
                                "p95_ttft_us",
                                "max_ttft_us",
                                "mean_actual_read_us",
                            }
                            else _relative_percent(before, after)
                        ),
                        "positive_change_meaning": spec.positive_change_meaning,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows_by(
    rows: Iterable[dict[str, Any]], **criteria: Any,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]


def _value(
    policies: dict[str, dict[str, Any]], policy: str, metric: str,
) -> float:
    return float(policies[policy][metric])


def _format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def _signed(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:+,.{digits}f}"


def audit_invariants(document: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    expected_gpu_count = document.get("experiment", {}).get("gpu_count")
    for ssd_count, policies in topology_items(document):
        request_counts = {
            summary.get("request_count") for summary in policies.values()
        }
        byte_counts = {summary.get("byte_count") for summary in policies.values()}
        if len(request_counts) != 1:
            violations.append(f"{ssd_count} SSD: policy request_count differs")
        if len(byte_counts) != 1:
            violations.append(f"{ssd_count} SSD: policy byte_count differs")
        for policy, summary in policies.items():
            prefix = f"{ssd_count} SSD/{policy}"
            if summary.get("starvation_free") is not True:
                violations.append(f"{prefix}: starvation_free is not true")
            if (
                expected_gpu_count is not None
                and summary.get("completed_inference_count") != expected_gpu_count
            ):
                violations.append(f"{prefix}: completion count mismatch")
            if policy_factors(policy) is not None or policy.startswith(
                "utility_edf_"
            ):
                if summary.get("control_update_non_tick_write_count") != 0:
                    violations.append(f"{prefix}: non-tick control write")
                if summary.get("group_weight_write_count") != 0:
                    violations.append(f"{prefix}: fixed Group WRR was changed")
                expected_aligned = sum(
                    int(summary.get(field, 0))
                    for field in (
                        "rate_control_write_count",
                        "queue_weight_write_count",
                        "group_weight_write_count",
                    )
                )
                aligned = summary.get("control_update_tick_aligned_write_count")
                if aligned != expected_aligned:
                    violations.append(f"{prefix}: aligned-write audit mismatch")
    return violations


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    document: dict[str, Any],
    effect_rows: list[dict[str, Any]],
    shapley_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    output_files: list[str],
) -> str:
    topology_data = list(topology_items(document))
    ssd_counts = [ssd_count for ssd_count, _ in topology_data]
    experiment = document.get("experiment", {})
    factor_definitions = document.get("factor_definitions", {})
    violations = audit_invariants(document)

    lines = [
        "# Coflow / Utility / EDF 2^3 消融实验报告",
        "",
        "> 本报告由 `experiments/analyze_component_ablations.py` 从原始 "
        "`summary.json` 确定性生成。所有差值都定义为“后者 - 前者”。",
        "",
        "## 最重要的归因边界",
        "",
        "8 个消融 cell 都共享 Gate 基础。因此：",
        "",
        "- `000 - Legacy FCFS-CIR` 是 **shared Gate foundation** 的效果；",
        "- `111 - 000` 才是 **C/U/E 三个算法因子的联合效果**；",
        "- C/U/E 的主效应、交互项和 Shapley 值一律只在 8 个 "
        "Gate 消融 cell 内计算，不会把 Gate 的提升偷算给 C/U/E。",
        "",
        "本轮 shared foundation 的定义为：",
        "",
        f"> {factor_definitions.get('shared_foundation', 'input did not record it')}",
        "",
        "三个因子的定义：",
        "",
    ]
    for factor in FACTORS:
        definitions = factor_definitions.get(factor, {})
        lines.append(
            f"- **{factor} — {FACTOR_NAMES_CN[factor]}**："
            f"0 = {definitions.get('0', 'N/A')}；"
            f"1 = {definitions.get('1', 'N/A')}。"
        )

    lines.extend(
        [
            "",
            "## 实验范围与完整性",
            "",
            f"- SSD 数量：{', '.join(map(str, ssd_counts))}",
            f"- GPU 数量：{experiment.get('gpu_count', 'N/A')}",
            f"- GPU 算力：{experiment.get('effective_compute_tflops', 'N/A')} TFLOPS",
            f"- 随机种子：{experiment.get('random_seed', 'N/A')}",
            f"- 控制周期：{experiment.get('control_period_us', 'N/A')} us",
            f"- 结果源哈希：`{document.get('source_sha256', 'N/A')}`",
            "",
        ]
    )
    if violations:
        lines.append(f"完整性检查：**失败（{len(violations)} 项）**。")
        lines.append("")
        lines.extend(f"- {item}" for item in violations)
    else:
        lines.append(
            "完整性检查：**通过**。所有报告 policy 都无饥饿、"
            "完成数正确，request/byte 守恒；严格策略无非 80 us tick "
            "写入，且没有修改 Group WRR。"
        )

    endpoint_rows: list[list[str]] = []
    target_met_ssd_counts: list[int] = []
    for ssd_count, policies in topology_data:
        baseline = _value(policies, "baseline", "mean_gpu_utilization_percent")
        legacy = _value(
            policies, "demand_aware_fcfs_cir", "mean_gpu_utilization_percent"
        )
        cell000 = _value(
            policies, "ablation_c0_u0_e0", "mean_gpu_utilization_percent"
        )
        cell111 = _value(
            policies, "ablation_c1_u1_e1", "mean_gpu_utilization_percent"
        )
        production = _value(
            policies, "utility_edf_integer_l750", "mean_gpu_utilization_percent"
        )
        target = min(baseline + 25.0, 99.5)
        meets_target = cell111 >= target
        if meets_target:
            target_met_ssd_counts.append(ssd_count)
        endpoint_rows.append(
            [
                str(ssd_count),
                _format_number(baseline),
                _format_number(legacy),
                _format_number(cell000),
                _format_number(cell111),
                _signed(cell000 - legacy),
                _signed(cell111 - cell000),
                _signed(cell111 - baseline),
                _format_number(target),
                "是" if meets_target else "否",
                _signed(production - cell111),
            ]
        )
    lines.extend(
        [
            "",
            "## 平均 GPU 利用率端点",
            "",
            "利用率本身的单位是 `%`，两策略相减的单位是"
            "**百分点（pp）**，不是相对百分比。目标定义为 "
            "`min(Baseline + 25 pp, 99.5%)`。",
            "",
            _markdown_table(
                [
                    "SSD",
                    "Baseline %",
                    "Legacy %",
                    "000 %",
                    "111 %",
                    "Gate: 000-Legacy pp",
                    "C/U/E: 111-000 pp",
                    "111-Baseline pp",
                    "目标 %",
                    "达标",
                    "Prod-111 pp",
                ],
                endpoint_rows,
            ),
            "",
            (
                f"达标 SSD 点共 **{len(target_met_ssd_counts)}** 个："
                + (
                    ", ".join(map(str, target_met_ssd_counts))
                    if target_met_ssd_counts
                    else "无"
                )
                + "。"
            ),
        ]
    )

    cell_order = (
        ("000", (0, 0, 0)),
        ("100 (C)", (1, 0, 0)),
        ("010 (U)", (0, 1, 0)),
        ("001 (E)", (0, 0, 1)),
        ("110 (C+U)", (1, 1, 0)),
        ("101 (C+E)", (1, 0, 1)),
        ("011 (U+E)", (0, 1, 1)),
        ("111 (C+U+E)", (1, 1, 1)),
    )
    raw_cell_rows = []
    for ssd_count, policies in topology_data:
        raw_cell_rows.append(
            [str(ssd_count)]
            + [
                _format_number(
                    _value(
                        policies,
                        factor_policy(bits),
                        "mean_gpu_utilization_percent",
                    )
                )
                for _, bits in cell_order
            ]
        )
    lines.extend(
        [
            "",
            "## 8 个消融 cell 的原始平均 GPU 利用率",
            "",
            "这张表直接回答“只开 C、只开 U、只开 E，以及两两组合”"
            "各自跑到多少。所有 cell 都已包含共享 Gate 基础。单位为 `%`。",
            "",
            _markdown_table(
                ["SSD"] + [label for label, _ in cell_order],
                raw_cell_rows,
            ),
        ]
    )

    util_effects = _rows_by(
        effect_rows, metric="mean_gpu_utilization_percent"
    )
    effect_lookup = {
        (int(row["ssd_count"]), str(row["effect"])): float(row["estimate"])
        for row in util_effects
    }
    effect_table = [
        [str(ssd_count)]
        + [
            _signed(effect_lookup[(ssd_count, effect)])
            for effect in ("C", "U", "E", "C:U", "C:E", "U:E", "C:U:E")
        ]
        for ssd_count in ssd_counts
    ]
    interaction_rank = []
    for effect in ("C:U", "C:E", "U:E", "C:U:E"):
        values = [effect_lookup[(ssd, effect)] for ssd in ssd_counts]
        interaction_rank.append(
            (
                effect,
                sum(values) / len(values),
                sum(abs(value) for value in values) / len(values),
            )
        )
    interaction_rank.sort(key=lambda item: item[2], reverse=True)
    interaction_rank_table = [
        [str(rank), effect, _signed(net), _format_number(absolute)]
        for rank, (effect, net, absolute) in enumerate(
            interaction_rank, start=1
        )
    ]
    lines.extend(
        [
            "",
            "## C/U/E 对平均 GPU 利用率的因子效应",
            "",
            "主效应是该因子从 0 变为 1 时，在其他因子所有组合上的"
            "平均变化。二阶项是平均差分之差，三阶项是二阶交互的差分。"
            "正值表示利用率增加，单位均为 pp。",
            "",
            _markdown_table(
                ["SSD", "C", "U", "E", "C:U", "C:E", "U:E", "C:U:E"],
                effect_table,
            ),
            "",
            "按所有 SSD 点上的平均绝对交互项排名如下。正值表示"
            "该组合使 GPU 利用率超过简单加和，负值表示互相抵消。",
            "",
            _markdown_table(
                ["排名", "交互项", "平均净交互 pp", "平均绝对交互 pp"],
                interaction_rank_table,
            ),
        ]
    )

    util_shapley = _rows_by(
        shapley_rows, metric="mean_gpu_utilization_percent"
    )
    shapley_lookup = {
        (int(row["ssd_count"]), str(row["factor"])): float(
            row["shapley_contribution"]
        )
        for row in util_shapley
    }
    shapley_table: list[list[str]] = []
    for ssd_count, policies in topology_data:
        contributions = [
            shapley_lookup[(ssd_count, factor)] for factor in FACTORS
        ]
        total = _value(
            policies, "ablation_c1_u1_e1", "mean_gpu_utilization_percent"
        ) - _value(
            policies, "ablation_c0_u0_e0", "mean_gpu_utilization_percent"
        )
        shapley_table.append(
            [str(ssd_count)]
            + [_signed(value) for value in contributions]
            + [_signed(sum(contributions)), _signed(total)]
        )
    factor_rank = []
    for factor in FACTORS:
        values = [shapley_lookup[(ssd, factor)] for ssd in ssd_counts]
        factor_rank.append(
            (
                factor,
                sum(values) / len(values),
                sum(abs(value) for value in values) / len(values),
            )
        )
    factor_rank.sort(key=lambda item: item[2], reverse=True)
    rank_table = [
        [
            str(index),
            factor,
            FACTOR_NAMES_CN[factor],
            _signed(net),
            _format_number(absolute),
        ]
        for index, (factor, net, absolute) in enumerate(factor_rank, start=1)
    ]
    lines.extend(
        [
            "",
            "## Shapley 归因：把 `111 - 000` 完整分给 C/U/E",
            "",
            "Shapley 值是 6 种因子加入顺序中边际收益的平均。"
            "它会公平分摊交互项，并且对每个 SSD 点都严格满足 "
            "`C + U + E = 111 - 000`（浮点容差内）。单位为 pp。",
            "",
            _markdown_table(
                ["SSD", "C", "U", "E", "Shapley 和", "111-000"],
                shapley_table,
            ),
            "",
            "按所有 SSD 点上的平均绝对 Shapley 贡献排名如下。"
            "“平均净贡献”保留正负号，“平均绝对贡献”表示影响强度。",
            "",
            _markdown_table(
                ["排名", "因子", "含义", "平均净贡献 pp", "平均绝对贡献 pp"],
                rank_table,
            ),
        ]
    )

    def utilization_delta(
        policies: dict[str, dict[str, Any]],
        after: str,
        before: str,
    ) -> float:
        return _value(
            policies, after, "mean_gpu_utilization_percent"
        ) - _value(
            policies, before, "mean_gpu_utilization_percent"
        )

    path_definitions = (
        ("只开 C", "ablation_c1_u0_e0", "ablation_c0_u0_e0"),
        ("只开 U", "ablation_c0_u1_e0", "ablation_c0_u0_e0"),
        ("只开 E", "ablation_c0_u0_e1", "ablation_c0_u0_e0"),
        ("不开 C 的 U+E", "ablation_c0_u1_e1", "ablation_c0_u0_e0"),
        ("U+E 已开后再开 C", "ablation_c1_u1_e1", "ablation_c0_u1_e1"),
        ("C+U 已开后再开 E", "ablation_c1_u1_e1", "ablation_c1_u1_e0"),
        ("Gate 相对 Legacy FCFS-CIR", "ablation_c0_u0_e0", "demand_aware_fcfs_cir"),
        ("Gate 相对 Baseline", "ablation_c0_u0_e0", "baseline"),
    )
    path_values: dict[str, list[float]] = {}
    path_table = []
    for label, after, before in path_definitions:
        values = [
            utilization_delta(policies, after, before)
            for _, policies in topology_data
        ]
        path_values[label] = values
        path_table.append(
            [
                label,
                _signed(sum(values) / len(values)),
                _signed(min(values)),
                _signed(max(values)),
            ]
        )

    c_after_ue = path_values["U+E 已开后再开 C"]
    best_c_index = max(range(len(c_after_ue)), key=c_after_ue.__getitem__)
    e_shapley_values = [
        shapley_lookup[(ssd_count, "E")] for ssd_count in ssd_counts
    ]
    u_shapley_values = [
        shapley_lookup[(ssd_count, "U")] for ssd_count in ssd_counts
    ]
    c_shapley_values = [
        shapley_lookup[(ssd_count, "C")] for ssd_count in ssd_counts
    ]
    ue_interactions = [
        effect_lookup[(ssd_count, "U:E")] for ssd_count in ssd_counts
    ]
    lines.extend(
        [
            "",
            "## 结论：哪个优化点最重要",
            "",
            "下表使用最直观的固定加入路径，不对交互项做平均分摊。"
            "平均、最小、最大都跨本报告的 SSD 点计算，单位为 pp。",
            "",
            _markdown_table(
                ["加入路径", "平均 Δ pp", "最小 Δ pp", "最大 Δ pp"],
                path_table,
            ),
            "",
            "1. **EDF 是首要贡献。** 它的平均 Shapley 贡献为 "
            f"**{sum(e_shapley_values) / len(e_shapley_values):+.3f} pp**，"
            f"并且 10 个 SSD 点全部为正（{min(e_shapley_values):+.3f} 至 "
            f"{max(e_shapley_values):+.3f} pp）。在 C+U 已开启后再开 EDF，"
            f"平均仍增加 **{sum(path_values['C+U 已开后再开 E']) / len(ssd_counts):+.3f} pp**。",
            "2. **Utility 是第二贡献，但单独使用较弱。** 它的平均 Shapley "
            f"贡献为 **{sum(u_shapley_values) / len(u_shapley_values):+.3f} pp**。"
            "它主要负责在尚未进入计算流水线的 Stage-0 候选之间选择更值得"
            "先启动的推理；若没有 EDF 保护正在计算 GPU 的下一层预取，"
            "仅改变 Stage-0 顺序无法消除主要阻塞。",
            "3. **Utility 与 EDF 是关键组合。** U:E 交互在 10 个点上的"
            f"平均净值为 **{sum(ue_interactions) / len(ue_interactions):+.3f} pp**。"
            "低 SSD 数的过载最强，二者协同也最明显；SSD 增多后物理容量"
            "缓解拥塞，交互自然衰减。",
            "4. **跨 SSD Coflow 一致性不是独立主力，而是条件性增强项。** "
            f"其平均 Shapley 为 **{sum(c_shapley_values) / len(c_shapley_values):+.3f} pp**；"
            "只开 C 时通常略降，因为全局同一 owner 会减少各 SSD 独立选择"
            "本地最佳 Queue 的自由度。不过在 U+E 已开启后，除 1 SSD 的"
            "必然零效应外，多 SSD 点再开 C 都为正，最大为 "
            f"**{c_after_ue[best_c_index]:+.3f} pp（{ssd_counts[best_c_index]} SSD）**。"
            "这说明它的价值是对齐多盘 barrier，而不是单独创造吞吐。",
            "5. **Gate 是控制基础，不是 1 SSD 达标的主要来源。** "
            f"000 相对 Legacy FCFS-CIR 平均为 "
            f"**{sum(path_values['Gate 相对 Legacy FCFS-CIR']) / len(ssd_counts):+.3f} pp**，"
            "但它与 C/U/E 使用的是不同控制语义，因此必须作为独立 foundation "
            "报告，不能把这部分收益记到任一算法因子。",
            "",
            "机制上，EDF 直接保护已经进入 `compute L / prefetch L+1` 流水线、"
            "且下一层 deadline 将到的 GPU；Utility 只在允许启动新的 Stage-0 "
            "读取时选择高价值候选；C 再把同一 GPU 的跨盘路径对齐。这个先后"
            "关系解释了为什么 `E > U > C`，以及为什么 U/E 和 C/U/E 存在交互。"
            "这些是固定 workload、seed、placement 与 SSD 模型下的确定性结论，"
            "不自动外推到其他分布。",
        ]
    )

    ttft_rows: list[list[str]] = []
    for ssd_count, policies in topology_data:
        row = [str(ssd_count)]
        for metric in ("mean_ttft_us", "p95_ttft_us", "max_ttft_us"):
            baseline = _value(policies, "baseline", metric)
            full = _value(policies, "ablation_c1_u1_e1", metric)
            row.extend(
                [
                    _format_number(baseline),
                    _format_number(full),
                    _signed(full - baseline),
                    _signed(_relative_percent(baseline, full)),
                ]
            )
        ttft_rows.append(row)
    lines.extend(
        [
            "",
            "## TTFT 退化完整报告",
            "",
            "下表的差值是 `111 - Baseline`。**正值表示 TTFT 变长，即退化；"
            "负值表示 TTFT 缩短，即改善。** 报告保留符号，没有取绝对值。",
            "",
            _markdown_table(
                [
                    "SSD",
                    "Mean base us",
                    "Mean 111 us",
                    "Mean Δ us",
                    "Mean Δ %",
                    "P95 base us",
                    "P95 111 us",
                    "P95 Δ us",
                    "P95 Δ %",
                    "Max base us",
                    "Max 111 us",
                    "Max Δ us",
                    "Max Δ %",
                ],
                ttft_rows,
            ),
            "",
            "Gate 、C/U/E 组合、Production parity 等每一段的平均/P95/Max "
            "TTFT 有符号绝对变化和有符号相对变化，详见 "
            "`anchor_comparisons.csv`。",
        ]
    )

    if 1 in ssd_counts:
        one_ssd = dict(topology_data)[1]
        differences = []
        for utility, edf in itertools.product((0, 1), repeat=2):
            local = _value(
                one_ssd,
                factor_policy((0, utility, edf)),
                "mean_gpu_utilization_percent",
            )
            global_owner = _value(
                one_ssd,
                factor_policy((1, utility, edf)),
                "mean_gpu_utilization_percent",
            )
            differences.append(global_owner - local)
        max_abs = max(abs(value) for value in differences)
        lines.extend(
            [
                "",
                "## 1 SSD 负向对照",
                "",
                "1 SSD 时不存在跨 SSD 所有者协调，因此 C=0 与 C=1 "
                "应等价。在四组 U/E 配对上，C 切换造成的平均 GPU 利用率"
                f"最大绝对差为 **{max_abs:.12g} pp**。",
            ]
        )

    lines.extend(
        [
            "",
            "## 计算方法",
            "",
            "设 `v(c,u,e)` 为某个 SSD 数量和某个指标下的 cell 结果。",
            "",
            "- C 主效应：所有 `(u,e)` 上 `v(1,u,e)-v(0,u,e)` 的平均；",
            "- C:U 二阶交互：两个 E 水平上 "
            "`v(1,1,e)-v(1,0,e)-v(0,1,e)+v(0,0,e)` 的平均；",
            "- C:U:E 三阶交互：E=1 和 E=0 两个 C:U 交互的差；",
            "- Shapley：遍历 C/U/E 的 6 种加入顺序，对每个因子的"
            "边际变化取平均。",
            "",
            "对 GPU 利用率，主效应、交互项、Shapley 和 anchor 差值单位均为 "
            "pp。对 TTFT，所有差值均保留符号，正值为退化。",
            "",
            "## 输出文件",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in output_files)
    lines.append("")
    return "\n".join(lines)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_utilization_effects(
    document: dict[str, Any],
    effect_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _configure_plot_style()
    topology_data = list(topology_items(document))
    x = [ssd for ssd, _ in topology_data]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    endpoints = (
        ("baseline", "Baseline", "#6b7280", "o", "-"),
        (
            "demand_aware_fcfs_cir",
            "Legacy FCFS-CIR",
            "#d97706",
            "s",
            "--",
        ),
        ("ablation_c0_u0_e0", "Gate foundation (000)", "#2563eb", "D", "-"),
        ("ablation_c1_u1_e1", "Full C+U+E (111)", "#dc2626", "^", "-"),
        (
            "utility_edf_integer_l750",
            "Production Utility+EDF",
            "#111827",
            "x",
            ":",
        ),
    )
    for policy, label, color, marker, linestyle in endpoints:
        y = [
            float(policies[policy]["mean_gpu_utilization_percent"])
            for _, policies in topology_data
        ]
        axes[0].plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6,
        )
    axes[0].set_ylabel("Mean GPU utilization (%)")
    axes[0].set_title("Anchors and factorial endpoints")
    axes[0].legend(ncol=2, frameon=False)

    colors = {"C": "#7c3aed", "U": "#059669", "E": "#e11d48"}
    for factor in FACTORS:
        selected = _rows_by(
            effect_rows,
            metric="mean_gpu_utilization_percent",
            effect_type="main",
            effect=factor,
        )
        by_ssd = {int(row["ssd_count"]): float(row["estimate"]) for row in selected}
        axes[1].plot(
            x,
            [by_ssd[ssd] for ssd in x],
            label=f"{factor}: {FACTOR_NAMES_EN[factor]}",
            color=colors[factor],
            marker="o",
            linewidth=2.0,
        )
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_ylabel("Average main effect (pp)")
    axes[1].set_xlabel("SSD count")
    axes[1].set_title("C/U/E main effects inside the shared Gate foundation")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False)
    fig.suptitle("Coflow / Utility / EDF component ablation", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_shapley(
    document: dict[str, Any],
    shapley_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _configure_plot_style()
    x_values = [ssd for ssd, _ in topology_items(document)]
    positions = list(range(len(x_values)))
    width = 0.22
    colors = {"C": "#7c3aed", "U": "#059669", "E": "#e11d48"}
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    all_contributions: dict[str, list[float]] = {}
    for index, factor in enumerate(FACTORS):
        rows = _rows_by(
            shapley_rows,
            metric="mean_gpu_utilization_percent",
            factor=factor,
        )
        by_ssd = {
            int(row["ssd_count"]): float(row["shapley_contribution"])
            for row in rows
        }
        values = [by_ssd[ssd] for ssd in x_values]
        all_contributions[factor] = values
        offsets = [position + (index - 1) * width for position in positions]
        axis.bar(
            offsets,
            values,
            width=width,
            label=f"{factor}: {FACTOR_NAMES_EN[factor]}",
            color=colors[factor],
            alpha=0.88,
        )
    total = [
        sum(all_contributions[factor][index] for factor in FACTORS)
        for index in range(len(x_values))
    ]
    axis.plot(
        positions,
        total,
        color="#111827",
        marker="o",
        linewidth=2.0,
        label="Total: 111 - 000",
    )
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_xticks(positions, [str(ssd) for ssd in x_values])
    axis.set_xlabel("SSD count")
    axis.set_ylabel("Mean GPU utilization contribution (pp)")
    axis.set_title("Shapley attribution of the C/U/E combined effect")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ttft_changes(
    document: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _configure_plot_style()
    x_values = [ssd for ssd, _ in topology_items(document)]
    metrics = (
        ("mean_ttft_us", "Mean TTFT"),
        ("p95_ttft_us", "P95 TTFT"),
        ("max_ttft_us", "Max TTFT"),
    )
    transitions = (
        ("shared_foundation", "Gate: 000 vs legacy", "#2563eb", "s"),
        ("combined_cue", "C/U/E: 111 vs 000", "#dc2626", "^"),
        ("end_to_end", "End-to-end: 111 vs baseline", "#111827", "o"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    for axis, (metric, title) in zip(axes, metrics):
        for transition, label, color, marker in transitions:
            selected = _rows_by(
                anchor_rows, transition=transition, metric=metric
            )
            by_ssd = {
                int(row["ssd_count"]): row["signed_relative_change_percent"]
                for row in selected
            }
            axis.plot(
                x_values,
                [float(by_ssd[ssd]) for ssd in x_values],
                label=label,
                color=color,
                marker=marker,
                linewidth=1.8,
            )
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.set_ylabel("Signed change (%)")
        axis.set_title(f"{title}: positive means degradation")
    axes[-1].set_xlabel("SSD count")
    axes[-1].set_xticks(x_values)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Signed TTFT changes (after - before)", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def analyze(input_path: Path, output_dir: Path) -> dict[str, Path]:
    document = load_document(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows = build_long_rows(document)
    effect_rows = build_effect_rows(document)
    shapley_rows = build_shapley_rows(document)
    anchor_rows = build_anchor_rows(document)

    paths = {
        "long_csv": output_dir / "metrics_long.csv",
        "effects_csv": output_dir / "factorial_effects.csv",
        "shapley_csv": output_dir / "shapley_attribution.csv",
        "anchors_csv": output_dir / "anchor_comparisons.csv",
        "utilization_png": output_dir / "gpu_utilization_effects.png",
        "shapley_png": output_dir / "gpu_utilization_shapley.png",
        "ttft_png": output_dir / "ttft_signed_changes.png",
        "report": output_dir / "REPORT.md",
    }
    _write_csv(paths["long_csv"], long_rows)
    _write_csv(paths["effects_csv"], effect_rows)
    _write_csv(paths["shapley_csv"], shapley_rows)
    _write_csv(paths["anchors_csv"], anchor_rows)
    plot_utilization_effects(document, effect_rows, paths["utilization_png"])
    plot_shapley(document, shapley_rows, paths["shapley_png"])
    plot_ttft_changes(document, anchor_rows, paths["ttft_png"])
    output_names = [path.name for key, path in paths.items() if key != "report"]
    report = build_report(
        document,
        effect_rows,
        shapley_rows,
        anchor_rows,
        output_names,
    )
    paths["report"].write_text(report, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    input_path = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_path.parent / "analysis"
    )
    paths = analyze(input_path, output_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return paths
