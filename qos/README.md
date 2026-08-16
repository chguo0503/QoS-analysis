# 256 Queue 分层 QoS

每块 SSD 独立实例化一个 QoS。一个 StoragePath 由一个 QoS 和一个 ASU SSD
后端组成，不同 SSD 的 Queue、Token Bucket、WRR 游标和反压状态彼此独立。

Utility+EDF 的整体设计和实验说明见
[完整设计文档](../docs/UTILITY_EDF_DESIGN.md)。

## Queue 布局

每块 SSD 固定创建 q000～q255 共 256 条 Queue，平均分成 8 个 Group：

~~~text
g0: q000–q031    g4: q128–q159
g1: q032–q063    g5: q160–q191
g2: q064–q095    g6: q192–q223
g3: q096–q127    g7: q224–q255
~~~

正式 128 GPU 拓扑使用 balanced_exclusive 绑定。每块 SSD 使用 128 条
独占 Queue，每个 Group 绑定 16 条；同一 GPU 的四个读组复用同一条
(GPU, SSD) Queue。

Group 只是两级 WRR 的第一层调度单元，不创建 Group CIR/PIR Token Bucket。
每条 Queue 独立保存 FIFO、CIR Token Bucket 和可选 PIR Token Bucket。
单 SSD 的物理带宽和入口上限由 SSD 后端及其反压建模，QoS 不重复创建
Root Token Bucket。

## CIR-first 与 EXCESS

每次仲裁只选择一个完整 IO，随后重新从 CIR 轮开始：

~~~text
CIR轮：
  Queue非空
  && Queue CIR token足够
  && Queue PIR允许
  && Queue WRR权重>0

EXCESS轮：
  Queue非空
  && Queue PIR允许
  && Queue WRR权重>0
~~~

QoS 先在所有 CIR-eligible Queue 中执行分层 WRR。全局没有 CIR 候选时才
进入 EXCESS 轮。

PIR=uncapped 不创建 PIR Bucket，因此 CIR=0 的 Queue 仍可通过 EXCESS。
真正的 Utility Admission Gate 使用 PIR=0 和 Queue WRR=0 双重关闭，而不是
只依靠 CIR。

QoS 不解析 Demand ID、requested rate、deadline、层号或 Utility score；
它只执行 DPU 最终写入的 Queue 状态。

## 两级 WRR 与动态 Queue 权重

~~~text
固定Group平滑WRR选择group_id
                  ↓
获胜Group内的动态Queue WRR选择queue_id
                  ↓
从该Queue FIFO取出一个IO
~~~

正式配置中八个 Group 权重始终为 1。Utility+EDF 只动态写 Queue 权重：

- weight=1：Queue 正常参加组内仲裁；
- weight=0：Queue 不参加仲裁；
- 部分权重更新只改变命中的 Queue，其他 Queue 保持原状态；
- 一个 Group 内全部 Queue 为 0 时，该 Group 不消耗 Group WRR 机会；
- 全部候选为 0 时安全返回“无 Queue 可发”，不会死循环。

schedule_group_weight_update() 接口仍保留给其他策略，但 Utility+EDF 不调用
它；其 group_weight_write_count 应为 0。

## Utility Queue 状态

| 状态 | CIR | PIR | Queue WRR |
|---|---:|---:|---:|
| selected | SSD capacity | uncapped | 1 |
| waiting / parked | 0 | 0 | 0 |
| 四读组完成后的默认态 | 0 | uncapped | 1 |

等待和 parked 在 QoS 寄存器中的状态相同，区别只存在于 DPU 生命周期：

- waiting 有当前活跃 Demand，但尚未被选中；
- parked 的当前读组已经排空，不过其 p_node 尚未完成四个读组；
- 下一读组到达不会自动解除 park；
- 第四个读组完成后才恢复默认状态。

这不是跨层保留 owner。读组排空后可以服务其他 p_node，原 Queue 只是保持
关闭，直到后续 Demand 再次被 Utility+EDF 选中。

## 严格 80 us 控制边界

Token refill 和 Utility 控制生效都使用 80 us 边界。GPU arrival 在
rate_update 之前发生，使用 inclusive ceil：

~~~text
effective_time = ceil(decision_time / 80) × 80 us
~~~

Queue-empty 则由 scheduler_dispatch 之后回调，已经错过当前边界的
rate_update；它使用严格的下一 tick，即 t 恰好为 80 的整数倍时也再加
80 us。

非边界 arrival 或 Queue-empty 可以立即触发 DPU 决策，但新 CIR/PIR/Queue
WRR 只能在下一边界应用。边界前保持上一控制状态，因此：

- Utility 管理的 Queue 在首次非边界 arrival 前需要预先 parked；
- 非边界 Queue-empty 后，等待 Queue 不能在下一边界前泄漏；
- 同一 Queue、同一 tick 的多次 rate 写入只保留最后一次；
- 同一 tick 的 Queue 权重部分更新按 Queue 合并，重复 Queue 取最后值。

同一时间戳的 QoS 阶段顺序固定为：

~~~text
token_refill
rate_update（CIR/PIR、Queue WRR；Group接口也在此阶段）
io_arrival
scheduler_dispatch
~~~

控制更新会保留 Queue FIFO，但 Queue 速率租约改变时清空旧 Token。选中 Queue
使用 PIR=uncapped，所以即使新 CIR Token 为 0，也能立即通过 EXCESS；
parked Queue 的 PIR=0、weight=0 则不能泄漏。

DPU 的 Byte/s 按当前 QoS 周期向上换算：

~~~text
fill_bytes_per_tick =
    ceil(rate_bytes_per_second × 80 / 1_000_000)
~~~

## Queue depth 接口

queue_io_counts() 返回：

~~~text
queue_id -> 已登记但尚未成功下发到SSD的完整IO数量
~~~

逻辑 occupancy 同时覆盖已经进入 FIFO 的请求和已登记、等待当前 arrival
阶段处理的请求。因此同一层批量提交期间，DPU 能看到完整读组，而不会把中间
空状态误认为 Demand 完成。

QoS 只在 Queue 状态变化后用 event_time_us 唤醒 DPU；DPU 必须主动读取
depth 快照。通知不包含 request、Demand 或 SSD 内部信息。

## Queue-empty 与 SSD 的关系

QoS 只在 SSD 入口可接收时非阻塞下发。SSD 可以把一个完整 Block 拆成多个
4 KiB 命令进入内部流水线，但这不改变 DPU/QoS 的完整 IO 边界。

~~~text
Queue-empty:
  最后一个IO已经离开QoS并进入SSD

SSD completion:
  SSD内部所有相关阶段真正完成该IO
~~~

二者不是同一个事件。DPU 只能根据 Queue-empty 释放 Demand 和改变 Gate；
LLM 层屏障仍等待真实 SSD completion。

多 SSD owner 切换时，一块盘上的 Queue-empty 可能导致同一控制边界更新其他
StoragePath。控制事件观察者负责把更早的跨盘控制时刻加入全局事件日历，并对
同一时刻的多次唤醒去重。

## 有限负载完成语义

正式实验为有限闭合负载。只要 SSD 后端持续具备进展能力，selected Queue 的
PIR=uncapped、weight=1 会持续下发；每个有限读组 Queue 排空后重新选择，
最终验证全部推理和读组完成。

结果中的 starvation_free 表示这次有限负载没有未完成的推理或读组，不表示
对无限在线到达流证明了等待时间上界。

## 公共入口

~~~python
from qos import build_qos_simulator, build_queue_layout
from simulation_common.config_utils import load_yaml

config = load_yaml("config/simulation_config.yaml")["simulation"]
layout = build_queue_layout(config["qos"]["queue_layout"])
qos = build_qos_simulator(
    qos_config=config["qos"],
    start_time_us=config["start_time_us"],
    queue_layout=layout,
)
qos.input(request)
result = qos.end()
~~~

联合 SSD 仿真由全局事件日历交错推进 GPU、DPU、各 QoS 和各 SSD；不要为每个
QoS 单独运行到结束，否则会破坏跨设备事件顺序。
