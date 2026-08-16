# GPU utilization policy comparison

This directory contains the 1–10 SSD comparison requested for:

- Baseline
- Legacy `demand_aware_fcfs_cir`
- New strict-80-us `utility_edf_integer_l750`

The plot uses full `batched_exact` simulation results, not fluid
interpolation.  The common experiment fingerprint is 128 GPUs, 512
TFLOPS/GPU, batch size 1, layers 0–3, seed 6103, random KV placement, and
40 GB/s per SSD.

## Files

- `gpu_utilization_vs_ssd_count_policy_comparison.png`: raster plot
- `gpu_utilization_vs_ssd_count_policy_comparison.svg`: vector plot
- `gpu_utilization_vs_ssd_count_policy_comparison.csv`: exact plotted values

## Raw sources

- Baseline and legacy FCFS-CIR, SSD 1–10:
  `../unified_topology_scan/summary.json`
- New Utility+EDF, SSD 1:
  `../utility_edf_strict_80us_validated/summary.json`
- New Utility+EDF, SSD 2–10:
  `../utility_edf_strict_80us_topology_chunks/ssd_2_4.json`,
  `ssd_5_7.json`, and `ssd_8_10.json`

Before plotting, the script validates the experiment fingerprint, request and
byte conservation, all 128 inference completions for the new policy, zero
starved p-nodes, fixed Group WRR, and that every new-policy control write is
aligned to the 80 us control grid.

Regenerate with:

```bash
python experiments/plot_gpu_utilization_policy_comparison.py
```
