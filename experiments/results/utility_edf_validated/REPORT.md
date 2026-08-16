# 旧版即时控制结果（仅供历史对照）

本目录的 `summary.json` 来自较早实现：Token Bucket refill 周期为 80 μs，
但 DPU 控制命令可以在任意 Queue arrival/empty 事件时刻立即生效。因此它不能
作为“控制写只能落在 80 μs 网格”的正式证据。

当前严格 80 μs 的自包含 Baseline+策略结果位于：

- [`../utility_edf_strict_80us_validated/REPORT.md`](../utility_edf_strict_80us_validated/REPORT.md)
- [`../utility_edf_strict_80us_validated/summary.json`](../utility_edf_strict_80us_validated/summary.json)
- [`../../../docs/UTILITY_EDF_DESIGN.md`](../../../docs/UTILITY_EDF_DESIGN.md)

旧版整数策略的性能数字恰好与严格版一致，但这不能替代时序契约验证；严格版
额外证明 2,188 个按 Queue/Group 计的动态控制写项全部落在 80 μs 边界，
非边界写入为 0。
