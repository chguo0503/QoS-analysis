import csv
import json
from pathlib import Path

import pytest

from experiments.analyze_component_ablations import (
    METRICS,
    analyze,
    build_anchor_rows,
    build_effect_rows,
    build_shapley_rows,
    factor_policy,
)


def _polynomial(c: int, u: int, e: int) -> float:
    return (
        40.0
        + 2.0 * c
        + 4.0 * u
        + 6.0 * e
        + 3.0 * c * u
        - 2.0 * c * e
        + 1.0 * u * e
        + 5.0 * c * u * e
    )


def _summary(utilization: float, ttft: float) -> dict:
    summary = {
        "mean_gpu_utilization_percent": utilization,
        "min_gpu_utilization_percent": utilization - 5.0,
        "p95_gpu_utilization_percent": utilization + 2.0,
        "max_gpu_utilization_percent": utilization + 3.0,
        "mean_ttft_us": ttft,
        "p95_ttft_us": ttft * 1.2,
        "max_ttft_us": ttft * 1.5,
        "late_gpu_layer_count": utilization,
        "p95_delta_us": ttft - 1_000.0,
        "worst_delta_us": ttft - 900.0,
        "mean_actual_read_us": ttft / 2.0,
        "completed_inference_count": 128,
        "starvation_free": True,
        "request_count": 456_116,
        "byte_count": 67_257_040_896,
        "rate_control_write_count": 10,
        "queue_weight_write_count": 20,
        "group_weight_write_count": 0,
        "control_update_tick_aligned_write_count": 30,
        "control_update_non_tick_write_count": 0,
        "wall_time_seconds": 1.25,
        "rate_control": {"decision_count": 9},
    }
    assert set(METRICS).issubset(summary)
    return summary


def _fixture_document() -> dict:
    policies = {
        "baseline": _summary(35.0, 900.0),
        "demand_aware_fcfs_cir": _summary(38.0, 950.0),
        "utility_edf_integer_l750": _summary(59.0, 1_190.0),
    }
    for c in (0, 1):
        for u in (0, 1):
            for e in (0, 1):
                value = _polynomial(c, u, e)
                policies[factor_policy((c, u, e))] = _summary(
                    value,
                    1_000.0 + 10.0 * (value - 40.0),
                )
    return {
        "experiment": {
            "gpu_count": 128,
            "effective_compute_tflops": 512.0,
            "random_seed": 6103,
            "control_period_us": 80,
        },
        "source_sha256": "synthetic-source-hash",
        "factor_definitions": {
            "C": {"0": "local", "1": "global"},
            "U": {"0": "FCFS", "1": "Utility"},
            "E": {"0": "FCFS", "1": "EDF"},
            "shared_foundation": "synthetic Gate foundation",
        },
        "topologies": {"2_ssd": policies},
    }


def _lookup(rows: list[dict], **criteria) -> dict:
    matches = [
        row
        for row in rows
        if all(row[key] == value for key, value in criteria.items())
    ]
    assert len(matches) == 1
    return matches[0]


def test_factorial_effects_and_shapley_efficiency() -> None:
    document = _fixture_document()
    effects = build_effect_rows(document)
    expected = {
        "C": 3.75,
        "U": 7.25,
        "E": 6.75,
        "C:U": 5.5,
        "C:E": 0.5,
        "U:E": 3.5,
        "C:U:E": 5.0,
    }
    for effect, value in expected.items():
        row = _lookup(
            effects,
            ssd_count=2,
            metric="mean_gpu_utilization_percent",
            effect=effect,
        )
        assert row["estimate"] == pytest.approx(value)
        assert row["unit"] == "pp"

    shapley = build_shapley_rows(document)
    util_rows = [
        row
        for row in shapley
        if row["metric"] == "mean_gpu_utilization_percent"
    ]
    contributions = {
        row["factor"]: row["shapley_contribution"] for row in util_rows
    }
    assert contributions == pytest.approx(
        {
            "C": 2.0 + 3.0 / 2.0 - 2.0 / 2.0 + 5.0 / 3.0,
            "U": 4.0 + 3.0 / 2.0 + 1.0 / 2.0 + 5.0 / 3.0,
            "E": 6.0 - 2.0 / 2.0 + 1.0 / 2.0 + 5.0 / 3.0,
        }
    )
    assert sum(contributions.values()) == pytest.approx(19.0)
    assert all(row["efficiency_residual"] == pytest.approx(0.0) for row in util_rows)


def test_anchor_differences_keep_util_pp_and_signed_ttft() -> None:
    anchors = build_anchor_rows(_fixture_document())
    foundation = _lookup(
        anchors,
        ssd_count=2,
        transition="shared_foundation",
        metric="mean_gpu_utilization_percent",
    )
    assert foundation["signed_change"] == pytest.approx(2.0)
    assert foundation["change_unit"] == "pp"
    assert foundation["signed_relative_change_percent"] is None

    ttft = _lookup(
        anchors,
        ssd_count=2,
        transition="end_to_end",
        metric="mean_ttft_us",
    )
    assert ttft["signed_change"] == pytest.approx(290.0)
    assert ttft["signed_relative_change_percent"] == pytest.approx(
        290.0 / 900.0 * 100.0
    )
    assert ttft["positive_change_meaning"].startswith("退化")


def test_analyze_writes_csv_report_and_png_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(_fixture_document(), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    paths = analyze(input_path, output_dir)

    assert set(paths) == {
        "long_csv",
        "effects_csv",
        "shapley_csv",
        "anchors_csv",
        "utilization_png",
        "shapley_png",
        "ttft_png",
        "report",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    for key in ("utilization_png", "shapley_png", "ttft_png"):
        assert paths[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    report = paths["report"].read_text(encoding="utf-8")
    assert "shared Gate foundation" in report
    assert "`111 - 000`" in report
    assert "正值表示 TTFT 变长" in report

    with paths["effects_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    util_c = next(
        row
        for row in rows
        if row["metric"] == "mean_gpu_utilization_percent"
        and row["effect"] == "C"
    )
    assert float(util_c["estimate"]) == pytest.approx(3.75)
