# DPU Queue 绑定与 Utility+EDF 控制

DPURequestGateway 连接 KV Placement、每块 SSD 的独立 QoS，以及可选的
速率/准入控制器。正式拓扑为 128 GPU；每个 GPU 到每块 SSD 的路径固定使用
一条独占 Queue。

Utility+EDF 的推导、伪代码、实验口径与复现方法见
[完整设计文档](../docs/UTILITY_EDF_DESIGN.md)。本文件只说明 DPU 组件实际
承担的接口和控制语义。

## 硬件接口边界

DPU 向 QoS 提供：

- 普通 IO：request_id、p_node_id、storage_target_id、size_bytes、
  queue_id 和 arrival_time_us。
- 每 Queue 的 CIR/PIR。
- 每 Queue 的动态 WRR 权重。

动态 Group WRR 接口仍然保留，但 Utility+EDF 不调用它；正式配置中八个
Group 的权重始终固定为 1。

QoS 向 DPU 只提供：

- Queue depth 快照；
- Queue 状态发生变化时的无负载唤醒。

唤醒不携带 request ID、Demand ID、逐 IO dispatch、SSD completion 或
inflight 信息。DPU 也不读取 NAND、FCP 或其他 SSD 内部状态。

KV Placement 还会在 DPU 内部使用的 demand_bw 中提供：

~~~python
{
    "demand_group_id": "...",
    "compute_layer_index": 0,
    "prefetch_layer_index": 1,
    "inference_arrival_time_us": 0,
    "service_window_us": 10_000,
    "aggregate_bytes_on_storage_target": 147_456,
    "aggregate_required_bytes_per_second": 14_745_600,
}
~~~

同一读组的全部 Block 先进入逻辑 Queue occupancy，DPU 再按
(SSD, queue_id) 登记一次路径 Demand，避免批量尚未完整入队时误判完成。
这些聚合元数据不会逐 IO 转发给 QoS。

## 128 GPU 独占 Queue 绑定

balanced_exclusive 在仿真开始前为每个 (GPU, SSD) 固定 Queue：

~~~text
group_id    = gpu_index % 8
queue_index = 32 × (gpu_index % 8) + floor(gpu_index / 8)
~~~

对 gpu_index=0..127：

- 每块 SSD 使用 128 条互不冲突的 Queue；
- 8 个 Group 各有 16 条绑定 Queue；
- 每个 Group 的另外 16 条 Queue 未被本拓扑使用；
- 不同 SSD 拥有独立 Queue namespace，可以复用相同 queue_id。

同一 GPU 后续四个 KV 读组始终复用它在该 SSD 上的固定 Queue。

## 三种控制策略

### Baseline

Baseline 不创建 Demand 控制器，也不动态写控制状态：

~~~text
Queue CIR = 0
Queue PIR = uncapped
Queue WRR = 1
Group WRR = 1
~~~

非空 Queue 可以通过 EXCESS 轮下发。

### Demand-aware FCFS CIR

demand_aware_fcfs_cir 按到达顺序分配每块 SSD 的 CIR。它只创建带宽保证，
仍保持 PIR=uncapped 和 Queue WRR=1，所以获得 0 CIR 的 Queue 仍可通过
EXCESS 下发；该策略不是 Admission Gate。

### Utility+EDF

Utility+EDF 全局一次只准入一个 p_node。它使用请求元数据、当前时间和
Queue depth：

1. Stage-0 候选按整数价值密度排序：

   ~~~text
   F = elapsed + remaining_service + 4 × compute_window
   U = compute_window / (F × remaining_service²)
   ~~~

   实现用大整数交叉相乘比较，不执行除法。

2. 已进入计算—预取流水线的读组按绝对 deadline 执行 EDF。
3. 尝试在 EDF 队列前插入最高 Utility 的 Stage-0；若任一 EDF 前缀预计超过
   deadline + allowance，先服务 EDF，否则启动该 Stage-0。
4. 多 SSD 读组按 (p_node_id, demand_group_id) 聚合；路径并行，因此估计
   服务时间取各 SSD 路径中的最大值。

完整公式、平局规则和数值例子见
[Utility+EDF 设计文档](../docs/UTILITY_EDF_DESIGN.md)。

## Queue Gate 状态

Utility+EDF 使用以下三种状态：

| 状态 | CIR | PIR | Queue WRR |
|---|---:|---:|---:|
| 当前选中的 p_node | 当前 SSD 容量 | uncapped | 1 |
| 等待或 parked | 0 | 0 | 0 |
| 整次四读组完成后的默认态 | 0 | uncapped | 1 |

CIR=0 本身不能阻止 EXCESS，所以等待 Queue 同时使用 PIR=0 和
Queue WRR=0。选中 Queue 保持 PIR uncapped，使 owner 切换后即使 CIR
token 从 0 开始，也可以立即通过 EXCESS 使用 SSD；物理上限仍由 SSD 后端
保证。

Utility+EDF 只返回 Queue 更新，group_weights 始终为 None。

## 严格 80 us 控制边界

正式语义不是“每 80 us 轮询一次 Queue”，而是：

- arrival 和 Queue-empty 仍会立即触发 DPU 计算；
- 计算所得 CIR/PIR/Queue WRR 命令只能在 80 us 边界生效；
- GPU arrival 发生在 QoS 的 rate_update 阶段之前：若 t 已是边界可用当前
  tick，否则使用 ceil(t / 80) × 80 us；
- Queue-empty 从 scheduler_dispatch 阶段回调，此时当前 tick 的 rate_update
  已结束，所以即使 t 恰好是边界也必须使用严格的下一 tick；
- 同一 Queue、同一 tick 的多次写入以最后状态为准；
- 到下一边界前继续使用上一边界已经生效的 Gate。

Utility 管理的独占 Queue 在首次非边界 arrival 前必须已经处于 parked，
防止默认 PIR=uncapped 在首个开门边界前产生 EXCESS 泄漏。相同原则也适用
于非边界 Queue-empty 后的 owner 切换：等待者只有到下一个控制边界才打开。

80 us 同时也是 Token Bucket refill 周期。DPU 将 Byte/s 向上换算为每 tick
整数 Byte：

~~~text
fill_bytes = ceil(rate_bytes_per_second × 80 / 1_000_000)
~~~

## Owner lock 与 Queue park

这两个状态不能混淆。

### Owner lock

选中候选尚未下发时，新的更优候选可以替换它。任一路径出现
depth < submitted_count 后，当前 owner 被锁定，直到当前读组在所有 SSD
上的 QoS Queue 路径全部排空。某块 SSD 路径先排空时仍保留 sticky lock，
防止另一块 SSD 的 depth 快照暂时陈旧导致错误抢占。

### Queue park

owner lock 不跨层保留。当前读组排空后，DPU 可以立刻选择其他活跃
p_node；但原 Queue 在所属 p_node 尚未完成四个读组时保持：

~~~text
parked = (CIR=0, PIR=0, Queue WRR=0)
~~~

下一层到达并登记新 Demand 时，parked Queue 不会自动恢复 uncapped。它必须
重新参加 Utility+EDF，只有再次被选中后才在下一个 80 us 边界打开。该
p_node 的第四个读组完成后，Queue 才恢复默认 (0, uncapped, 1)。

## Queue-empty 不等于 SSD completion

Queue depth 变成 0 只表示该 Demand 的 IO 已全部离开 QoS、进入 SSD。此时
这些 IO 仍可能位于 SSD 的 FCP、BCP、NFI、NAND 或其他流水线阶段。

DPU 在 Queue-empty 时释放 Demand、轮换 owner 是允许的，因为它严格看不到
SSD completion/inflight。真正的层完成和下一层启动仍由 LLM 收到全部 SSD
completion 后决定。

## 有限负载下的无饥饿含义

正式实验是有限闭合负载：128 个 GPU 各一次推理，每次固定四个读组。选中
Queue 的 PIR=uncapped、weight=1，SSD 可接收时会持续前进；每个有限读组
排空后重新选择，最终可以验证所有推理和读组完成。

这里的“无饥饿”仅指该有限工作负载完整结束，不是对无限在线到达流证明了
最大等待时间。Utility score 本身没有面向无限流的严格 aging 上界。

## 可审计统计

DPURequestGateway.statistics() 报告：

- Queue CIR/PIR 写入次数；
- Queue WRR 写入次数；
- Group WRR 写入次数；
- Queue assignment 和终态 depth；
- Utility+EDF 的决策、冲突、owner 切换和已完成读组计数。

Utility+EDF 正式运行中应满足：

~~~text
group_weight_write_count == 0
active_demand_count == 0
每个 p_node 完成 4 个读组
~~~
