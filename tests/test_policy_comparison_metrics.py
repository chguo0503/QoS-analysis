"""Tests for generic policy-versus-Baseline experiment metrics."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from qos_ssd_simulator import (
    load_simulation_config,
    run_configured_experiment,
    summarize_policy_comparison,
)


def make_run_summary(mean_utilization, min_utilization, ttft_values):
    """Build the metric subset consumed by policy comparison code."""
    return {
        "mean_gpu_utilization_percent": mean_utilization,
        "min_gpu_utilization_percent": min_utilization,
        "mean_ttft_us": ttft_values[0],
        "p95_ttft_us": ttft_values[1],
        "max_ttft_us": ttft_values[2],
    }


class PolicyComparisonMetricTests(unittest.TestCase):
    """Verify target, utilization, and signed TTFT comparison semantics."""

    def test_reports_target_and_all_required_metric_changes(self):
        """A policy at Baseline+25 points meets an uncapped target exactly."""
        baseline = make_run_summary(45.0, 20.0, (1_000, 2_000, 4_000))
        policy = make_run_summary(70.0, 32.5, (800, 2_200, 6_000))

        self.assertEqual(
            summarize_policy_comparison(baseline, policy),
            {
                "mean_gpu_utilization_gain_percentage_points": 25.0,
                "target_mean_gpu_utilization_percent": 70.0,
                "meets_target": True,
                "min_gpu_utilization_change_percentage_points": 12.5,
                "mean_ttft_change_us": -200,
                "mean_ttft_change_percent": -20.0,
                "p95_ttft_change_us": 200,
                "p95_ttft_change_percent": 10.0,
                "max_ttft_change_us": 2_000,
                "max_ttft_change_percent": 50.0,
            },
        )

    def test_caps_target_at_99_point_5_percent(self):
        """A high-utilization Baseline only requires reaching the target cap."""
        baseline = make_run_summary(90.0, 80.0, (100, 110, 120))
        below_cap = make_run_summary(99.4, 79.0, (101, 111, 121))
        at_cap = make_run_summary(99.5, 79.0, (101, 111, 121))

        below_comparison = summarize_policy_comparison(baseline, below_cap)
        at_comparison = summarize_policy_comparison(baseline, at_cap)
        self.assertEqual(
            below_comparison["target_mean_gpu_utilization_percent"],
            99.5,
        )
        self.assertFalse(below_comparison["meets_target"])
        self.assertTrue(at_comparison["meets_target"])

    def test_zero_baseline_ttft_has_json_safe_relative_change(self):
        """A synthetic zero denominator produces None instead of Infinity."""
        baseline = make_run_summary(0.0, 0.0, (0, 0, 0))
        policy = make_run_summary(1.0, 1.0, (1, 2, 3))

        comparison = summarize_policy_comparison(baseline, policy)
        self.assertIsNone(comparison["mean_ttft_change_percent"])
        self.assertIsNone(comparison["p95_ttft_change_percent"])
        self.assertIsNone(comparison["max_ttft_change_percent"])


class ConfiguredExperimentComparisonTests(unittest.TestCase):
    """Verify every non-Baseline policy receives a comparison mapping."""

    def test_outputs_generic_comparisons_and_legacy_paired_result(self):
        """The generic mapping covers two policies while paired stays legacy."""
        config = deepcopy(load_simulation_config())
        config["topology"]["ssd_counts"] = [3]
        config["dpu"]["rate_control"]["strategies"] = [
            "baseline",
            "demand_aware_fcfs_cir",
            "queue_priority",
        ]
        run_summaries = {
            "baseline": make_run_summary(
                50.0, 20.0, (1_000, 2_000, 3_000)
            ),
            "demand_aware_fcfs_cir": make_run_summary(
                60.0, 18.0, (900, 2_100, 3_300)
            ),
            "queue_priority": make_run_summary(
                75.0, 25.0, (750, 2_400, 4_500)
            ),
        }

        with (
            patch(
                "qos_ssd_simulator.run_one",
                side_effect=lambda _config, _count, policy: deepcopy(
                    run_summaries[policy]
                ),
            ),
            patch("qos_ssd_simulator.write_summary") as write_summary,
        ):
            summary = run_configured_experiment(config)

        topology = summary["topologies"]["3_ssd"]
        self.assertEqual(
            set(topology["comparisons"]),
            {"demand_aware_fcfs_cir", "queue_priority"},
        )
        self.assertFalse(
            topology["comparisons"]["demand_aware_fcfs_cir"][
                "meets_target"
            ]
        )
        self.assertTrue(
            topology["comparisons"]["queue_priority"]["meets_target"]
        )
        self.assertEqual(topology["paired"], {
            "mean_gpu_utilization_gain_percentage_points": 10.0,
        })
        self.assertGreaterEqual(write_summary.call_count, 1)


if __name__ == "__main__":
    unittest.main()
