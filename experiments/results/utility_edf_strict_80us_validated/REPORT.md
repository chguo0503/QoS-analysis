# 128 GPU Utility+EDF 严格 80 μs 正式验证报告

这份目录是当前正式证据。机器可读原始结果见
[`summary.json`](summary.json)，策略原理、逐文件改动和复现教程见
[`docs/UTILITY_EDF_DESIGN.md`](../../../docs/UTILITY_EDF_DESIGN.md)。

## 实验口径

- 128 张 GPU，每张定义为 512 TFLOPS。
- 四个计算层：先读取 Layer 0；计算 Layer L 时预取 Layer L+1；最后一层只计算。
- 每张 GPU 一次推理，seed 6103，输入 100K～200K Token，命中率 0.50～0.99。
- KV Placement、负载、Queue 绑定和 SSD 物理参数在 Baseline 与策略之间完全相同。
- 1 块 SSD，标称读带宽 40,000,000,000 Byte/s。
- Group WRR 固定；仅动态控制 Queue CIR、PIR 和 Queue WRR。
- 所有 Utility+EDF 控制写项只在 80 μs 网格生效；DPU 不读取 SSD completion、inflight 或未来状态。
- 目标是 `min(Baseline mean utilization + 25pp, 99.5%)`。

这里的 GPU utilization 是每次推理的
`四层纯计算时间 / 实际 TTFT × 100%`，再对 128 次推理取算术平均；
它不是整段墙钟时间上的集群硬件 busy ratio。

## 正式结果

| 指标 | Baseline | `utility_edf_integer_l750` | 变化 |
|---|---:|---:|---:|
| Mean GPU utilization | 21.242814% | 46.523392% | **+25.280578pp** |
| 目标利用率 | — | 46.242814% | **超过 0.280578pp** |
| Mean TTFT | 1491.870 ms | 921.478 ms | -38.233% |
| P95 TTFT | 1719.905 ms | 1595.285 ms | -7.246% |
| Max TTFT | 1741.571 ms | 1685.365 ms | -3.227% |
| Min GPU utilization | 0.939300% | 0.912643% | -0.026657pp |
| P95 GPU utilization | 45.005833% | 96.868239% | +51.862407pp |

因此，严格 80 μs 控制版本仍满足约定的 25 个百分点目标。

## 严格 80 μs 证据

新策略运行共写入：

```text
Queue CIR/PIR joint commands = 1,158
Queue WRR per-Queue writes   = 1,030
Group WRR writes             = 0
tick-aligned write entries   = 2,188
non-tick write entries       = 0
control period               = 80 us
```

启动时只把 128 条实际绑定 Queue 预先 park 为 `(CIR=0, PIR=0,
weight=0)`。GPU arrival 若已位于 tick，允许在该 tick 的
`rate_update` 阶段写入；非边界 arrival 向上取整到下一 tick。
Queue-empty 是在本 tick 的调度阶段之后观察到的，因此即使恰好发生在 tick，
换手也必须等严格的下一 tick。同一 tick 的重复字段写入由最后值覆盖。

## 完整性、无饥饿和物理工作量

- 128/128 次推理完成。
- 512/512 个 KV 读组完成。
- GPU、QoS、SSD 三侧均完成 456,116 个请求和 67,257,040,896 Byte。
- `starved_p_node_count=0`；这是当前有限闭合负载的实测结果，不是无限在线到达的形式证明。
- Baseline 与新策略的 SSD 最后完成时刻均为 1,681,488.92 μs。
- 新策略 439,215 个 IO 走 CIR，16,901 个 IO 走 selected Queue 的 uncapped EXCESS；等待 Queue 始终被 PIR=0 和 weight=0 双重门控。

新策略没有增加带宽或减少数据量。提升来自重新安排完成顺序，使更多 GPU
较早进入计算并让下一层读取与当前层计算重叠。

## 尾部影响

TTFT 的 mean、P95 和 max 都改善，但 read-window lateness 被集中到更少的
GPU/层窗口，必须同时披露：

| Read-window 指标 | Baseline | 新策略 | 变化 |
|---|---:|---:|---:|
| late window count（共 512 个） | 510 | 207 | -303 / -59.412% |
| Mean actual read | 353.944 ms | 208.932 ms | -40.970% |
| P95 lateness delta | 552.923 ms | 1221.268 ms | **+120.875%** |
| Worst lateness delta | 706.326 ms | 1652.204 ms | **+133.915%** |

也就是说，大多数窗口明显变好，少数被延后的窗口更差。用户允许这种尾部退化，
前提是完整报告且没有永久饥饿；本表就是对应的代价说明。

## 策略摘要

尚未启动 Layer-0 读取的 GPU 使用整数价值密度：

```text
F = elapsed + remaining_service + 4 * compute_window
U = compute_window / (F * remaining_service^2)
```

DPU 用交叉整数乘法比较 U，不执行浮点除法。已经进入计算流水线的下一层
预取按 EDF 排序。若在最高 U 的初始读取前插入所有 ready prefetch 后，
任一 prefetch 预计晚于 `deadline + 750 μs`，就先服务 EDF；否则启动新的 GPU。

当前 owner 一旦有 IO 离开 QoS 就锁到该读组所有 SSD 路径 Queue-empty。
选中 Queue 为 `(40 GB/s, uncapped, 1)`，等待 Queue 为 `(0, 0, 0)`。
前三个读组结束后 Queue 保持 park，但不跨层保留 owner 槽；第四个读组结束后
才恢复 Baseline 默认 `(0, uncapped, 1)`。

## 复现

在项目根目录运行：

```bash
python - <<'PY'
from qos_ssd_simulator import load_simulation_config, run_configured_experiment

config = load_simulation_config("config/simulation_config.yaml")
config["topology"]["ssd_counts"] = [1]
config["dpu"]["rate_control"]["strategies"] = [
    "baseline",
    "utility_edf_integer_l750",
]
config["experiment"]["output_file"] = (
    "experiments/results/utility_edf_strict_80us_reproduced/summary.json"
)
run_configured_experiment(config)
PY
```

验证命令：

```bash
python -m unittest discover -s tests -v
python -m py_compile \
  DPU/dispatcher.py DPU/rate_controller.py qos_ssd_simulator.py \
  discrete_simulation/simulator.py simulation_common/storage_path.py \
  llm_workload/layer_request.py llm_workload/kv_placement_manager.py
git diff --check
```

当前验收为 81/81 测试通过，`py_compile` 和 `git diff --check` 通过。
运行环境为 Python 3.10.10、PyYAML 6.0。关键源码与配置的 SHA-256
见 [`MANIFEST.md`](MANIFEST.md)。

## 证据边界

- 本目录的 raw JSON 只正式证明 `1 SSD × integer L750`。
- Power score 和 2/3 SSD 数字属于搜索记录，当前没有与本目录同等级的 raw 配对文件。
- owner lock 只锁到 QoS Queue drain，不锁到 SSD completion；DPU 看不到后端 inflight。
- DPU 排序本身的指令执行延迟尚未单独加入仿真；整数版用于降低真实实现成本。
- 当前无饥饿结论只适用于所有 arrival 最终停止的有限工作负载。
