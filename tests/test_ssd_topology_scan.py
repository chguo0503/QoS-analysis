"""Test GPU-utilization metrics and stable-random topology-scan placement."""

from argparse import Namespace
import unittest

from experiments.scan_ssd_topologies import (
    build_simulation,
    gpu_utilization_percent,
    summarize_pair,
)


def build_test_arguments():
    """Function: create a compact argument set for scan-construction tests.

    Purpose: exercise the production configuration builder without running a
    large SSD simulation or duplicating its command-line parser.

    Input: none.

    Output: namespace containing all fields consumed by ``build_simulation``.
    """
    return Namespace(
        gpu_count=4,
        batch_size=1,
        layer_count=4,
        input_token_min=1_000,
        input_token_max=2_000,
        hit_ratio_min=0.50,
        hit_ratio_max=0.99,
        seed=6103,
    )


class GPUUtilizationTests(unittest.TestCase):
    """Verify the whole-inference modeled GPU utilization definition."""

    def test_no_ssd_stall_is_full_utilization(self):
        """Function: evaluate an inference whose TTFT is all computation.

        Purpose: establish that an IO-hidden inference reports exactly 100%.

        Input: synthetic 400-us compute-only and actual TTFT values.

        Output: unittest exact-equality assertion.
        """
        inference = {
            "compute_only_ttft_us": 400,
            "ttft_us": 400,
        }
        self.assertEqual(gpu_utilization_percent(inference), 100)

    def test_ssd_stall_reduces_utilization(self):
        """Function: evaluate computation followed by modeled SSD waiting.

        Purpose: verify that 100 us busy within 125 us elapsed equals 80%.

        Input: synthetic compute-only and actual TTFT values.

        Output: unittest exact-equality assertion.
        """
        inference = {
            "compute_only_ttft_us": 100,
            "ttft_us": 125,
        }
        self.assertEqual(gpu_utilization_percent(inference), 80)

    def test_pair_reports_percentage_point_gain(self):
        """Function: compare two already averaged utilization summaries.

        Purpose: ensure paired output reports percentage points, not a relative
        percent and not the old improved/worsened GPU count.

        Input: synthetic 97.5% Baseline and 100% Demand-aware summaries.

        Output: exact paired dictionary with a +2.5-point gain.
        """
        paired = summarize_pair(
            {"mean_gpu_utilization_percent": 97.5},
            {"mean_gpu_utilization_percent": 100.0},
        )
        self.assertEqual(paired, {
            "mean_gpu_utilization_gain_percentage_points": 2.5,
        })


class RandomPlacementConfigurationTests(unittest.TestCase):
    """Verify both policies receive the same stable-random Block placement."""

    def test_scan_selects_random_placement_for_both_policies(self):
        """Function: construct Baseline and Demand-aware scan simulations.

        Purpose: confirm the scan no longer overrides Placement with balanced
        round robin, while workloads and exclusive Queue bindings remain paired.

        Input: four GPUs, three SSDs and fixed seed 6103.

        Output: assertions on strategy, workload identity and Queue bindings.
        """
        arguments = build_test_arguments()
        baseline = build_simulation(arguments, 3, "baseline")
        demand_aware = build_simulation(
            arguments,
            3,
            "demand_aware_fcfs_cir",
        )
        self.assertTrue(all(
            manager.strategy.strategy_name == "random"
            for manager in baseline.placement_managers.values()
        ))
        self.assertEqual(
            baseline.gpu_workload_sequences,
            demand_aware.gpu_workload_sequences,
        )
        self.assertEqual(
            baseline.dpu.binding.bindings,
            demand_aware.dpu.binding.bindings,
        )


if __name__ == "__main__":
    unittest.main()
