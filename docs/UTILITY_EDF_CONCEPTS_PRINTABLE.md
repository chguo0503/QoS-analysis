# Utility+EDF 九问：一张可独立打印的概念说明

> 适用实现：`UtilityEDFController` / `utility_edf_integer_l750`
> 阅读门槛：不要求了解 DPU、QoS、LLM 或离散事件仿真
> 口径：本文的数字是便于手算的教学例子；公式、字段含义和状态转换与当前实现一致，
> 不是从某次实验结果中抽出的一条真实 trace。

## 摘要

可以把系统想成“两座 SSD 仓库给 GPU 厨房送 KV 食材”：

- 一个 KV Block 是一箱货；`block_size_bytes` 是每箱多大。
- 一条 SSD path 是一座仓库到一个 GPU 的装货通道；`path_request_count` 是原有箱数，
  `path_bytes` 是这条通道原有货物总量。
- Queue depth 是还没有离开 QoS 装货台的完整箱数。箱子离开装货台不等于已经送到厨房，
  所以 **Queue-empty 不等于 SSD completion**。
- Stage 0 是 GPU 尚未开始计算时先读 L0。它按 Utility 排序；
  已经进入“计算当前层、预取下一层”流水线的读组按 EDF 排序。
- 控制器只看请求元数据、当前 DPU 时钟和完整 Queue depth，不看 SSD 内部的
  inflight、NAND 状态或 completion。
- 当前 owner 一旦有 IO 离开 Queue，就 sticky 到这个读组的所有 SSD path 都
  Queue-empty；owner 不跨层继承。
- 所有受管 Queue 在 `t=0` 先 pre-park。控制计算可以在任意数据事件时发生，
  但 Gate 只能在 `0, 80, 160, ... us` 的控制边界生效。

本文用同一条时间线回答九个问题：

1. `inference_arrival_time_us` 是谁的“出生时间”？
2. `arrival_time_us` 又是谁到达？
3. current clock 是哪个时间？
4. `service_window_us` 是时长还是时刻？
5. `deadline_us` 怎样产生、怎样使用？
6. `path_bytes` 为什么不是整个读组的字节数？
7. `path_request_count` 为什么不是当前 Queue depth？
8. `block_size_bytes` 与前两者是什么关系？
9. Queue depth 怎样变成 remaining service，Queue-empty 又意味着什么？

---

## 统一数字例子

### 固定参数

例子沿用正式配置中的关键硬件口径：

| 参数 | 符号 | 数值 |
|---|---:|---:|
| 计算层数 | `G` | 4 |
| SSD 数量 | — | 2 |
| 每块 SSD 标称容量 | `C_s` | 40,000,000,000 Byte/s = 40,000 Byte/us |
| KV Block 大小 | `S` | 147,456 Byte |
| EDF allowance | `L` | 750 us |
| QoS 控制周期 | `T` | 80 us |

例子中有两个 GPU/P 节点：

- `P_A` 已经计算 L0，并在 `r_A=9,000 us` 提交 L1 prefetch；它的计算窗口为
  `c_A=1,000 us`，所以绝对 deadline 是 `d_A=10,000 us`。
- `P_B` 的整次推理在 `a_B=9,953 us` 到达；它同时提交 Stage-0/L0 初始读，
  所以该读组的到达时刻也是 `r_B0=9,953 us`。它的单层计算窗口是
  `c_B=500 us`。

两者的原始 path 如下。所有除法都向上取整到整数微秒：

| 候选 | 类型 | SSD | `N=path_request_count` | `B_s=path_bytes=N*S` | 满 Queue 服务时间 `b_s=ceil(B_s/C_s)` |
|---|---|---:|---:|---:|---:|
| `P_A:L1` | Prefetch | SSD0 | 109 | 16,072,704 B | 402 us |
| `P_A:L1` | Prefetch | SSD1 | 80 | 11,796,480 B | 295 us |
| `P_B:L0` | Stage 0 | SSD0 | 163 | 24,035,328 B | 601 us |
| `P_B:L0` | Stage 0 | SSD1 | 120 | 17,694,720 B | 443 us |

同一读组在两块 SSD 上并行，因此候选服务时间取最慢 path，而不是相加：

```text
b_A = max(402, 295) = 402 us
b_B = max(601, 443) = 601 us
```

### 符号表

九个核心概念用粗体编号。其余符号是推导量或控制参数。

| 编号 | 字段或符号 | 一句话定义 | 在例子中的值 |
|---:|---|---|---:|
| **1** | `a` / `inference_arrival_time_us` | 整次推理第一次进入系统的绝对时刻 | `a_B=9,953 us` |
| **2** | `r` / `arrival_time_us` | 当前这个层读组进入 DPU/QoS 的绝对时刻 | L0 为 `9,953`，L1 为 `11,105 us` |
| **3** | `t` / current clock | 控制器本次做决定时看到的仿真时刻 | 如 `9,953`、`10,402 us` |
| **4** | `c` / `service_window_us` | 当前 GPU 一层计算可用于隐藏预取的持续时间 | `c_B=500 us` |
| **5** | `d` / `deadline_us` | Prefetch 应赶上的绝对时刻 | `P_A` 为 `10,000 us` |
| **6** | `B_s` / `path_bytes` | 一个读组在某一块 SSD path 上最初提交的总字节 | `P_B/SSD0` 为 `24,035,328 B` |
| **7** | `N_s` / `path_request_count` | 该 path 最初提交的完整 IO 数量 | `P_B/SSD0` 为 `163` |
| **8** | `S` / `block_size_bytes` | 一个等长 IO/KV Block 的大小 | `147,456 B` |
| **9** | `q_s` / Queue depth | 当前仍未离开 QoS 的完整 IO 数量 | 如 `120` |
| — | `R_s` / remaining bytes | 由 `q_s` 推出的当前剩余字节 | `min(B_s, q_s*S)` |
| — | `b_s` / remaining service | 一条 path 的标称剩余服务时间 | `ceil(R_s/C_s)` |
| — | `b` | 多 SSD 读组的候选剩余服务时间 | `max_s(b_s)` |
| — | `e` / effective time | 本次控制写真正生效的 80-us 边界 | 如 `10,480 us` |
| — | `L` / allowance | 做插入可行性检查时允许越过 deadline 的裕量 | `750 us` |

注意：`a`、`r`、`t`、`d`、`e` 都是绝对时刻；`c`、`b`、`L` 是时长。
这是全文最重要的一条单位检查。

### 统一时间线

```text
9,000       P_A 开始计算 L0，同时提交 L1 prefetch
            r_A=9,000，c_A=1,000，所以 d_A=10,000

9,953       P_B 整次推理到达，并提交 Stage-0/L0 初始读
            a_B=r_B0=t=9,953
            假设此刻需要重新选择 owner；控制器在 P_A 和 P_B 中决策

10,000      选择 P_A 的 Gate 在下一个 80-us 边界真正打开

10,160      教学快照：P_A depth 已下降，SSD0 为 66，SSD1 为 37
            若此刻有新 arrival 触发 DPU 读取，它会确认 owner 已开始并 sticky

10,295      P_A/SSD1 Queue-empty；SSD0 尚未空，sticky 继续

10,402      P_A/SSD0 Queue-empty；整个 P_A:L1 读组才从 DPU 视角结束
            当前控制器时钟 t=10,402，重新选择 P_B

10,430      假设 P_A 的最后一个 SSD completion 此时才回来
            它晚于 Queue-empty；DPU 不读取这个 completion

10,480      P_B 的 Gate 在严格下一控制边界打开

10,640      教学快照：P_B depth 已下降，SSD0 为 120，SSD1 为 77

10,923      P_B/SSD1 Queue-empty；SSD0 尚未空

11,081      P_B/SSD0 Queue-empty；P_B:L0 的 owner lock 释放

11,105      假设 P_B 的 L0 最后 SSD completion 返回
            LLM 才能开始计算 L0，并提交 L1 prefetch
            r_B1=11,105，c_B=500，所以 d_B1=11,605

11,120      L1 arrival 对应的控制状态最早在这个 80-us 边界生效
            P_B:L1 必须重新参加 EDF；它不会自动继承 L0 owner
```

`10,430 us` 和 `11,105 us` 是为了展示“Queue-empty 早于 completion”而设的
教学时刻。实际 completion 由 SSD 后端流水线计算，DPU 既不能预知，也不能读取。

---

## 1. `inference_arrival_time_us` 是谁的“出生时间”？

它是**整次推理**第一次进入系统的绝对时刻，不是某一层读请求的到达时刻。

对 `P_B` 而言：

```text
a_B = inference_arrival_time_us = 9,953 us
```

这个值随后随 L0、L1、L2、L3 四个读组一起传递，不会在换层时重置。比如
`P_B:L1` 到 `11,105 us` 才提交，它携带的 `inference_arrival_time_us` 仍是
`9,953 us`。

Stage-0 Utility 用它计算整次推理已经等待多久：

```text
elapsed = max(0, t - a)
```

在 `t=9,953 us`，`P_B` 的 `elapsed=0`。如果它一直等到
`t=10,402 us` 才重新参加决策，则：

```text
elapsed = 10,402 - 9,953 = 449 us
```

当前实现的要点：

- LLM 明确把 workload 的原始到达时刻放入每个层计划。
- 若调用者没有提供该字段，DPU 才回退到当前读组的 `arrival_time_us`。
- Utility 中使用 `max(0, t-a)`，防止异常的未来时间产生负等待。
- Prefetch 的 EDF 主排序不使用这个字段；它主要服务于 Stage-0 Utility。

一句话记忆：**`a` 是顾客第一次进店的时间，后面每次催菜都不能把它改成“刚进店”。**

---

## 2. `arrival_time_us` 又是谁到达？

它是**当前这个 demand group/path 批次**进入 DPU/QoS 的绝对时刻。一个推理会有
多个读组，所以会有多个 `r`：

```text
P_B:L0  Stage 0 arrival: r_B0 =  9,953 us
P_B:L1  Prefetch arrival: r_B1 = 11,105 us
```

Stage 0 通常在推理一开始提交，所以当前模型中常见 `r_B0=a_B`。从 L1 开始，
读组要等上一个层屏障越过后才能提交，因此通常 `r_B1>a_B`。

`arrival_time_us` 的当前用途包括：

- 没有显式 deadline 时，生成默认 `deadline_us=r+c`；
- Utility 平局时，较早的读组 arrival 优先；
- 多 SSD path 聚成一个候选时，候选取各 path 中最早的 arrival；
- 形成稳定的 arrival sequence，供 EDF 平局时使用。

它不是 inference arrival，也不是控制写生效时刻。`P_B:L1` 在
`11,105 us` 真实到达 Queue，但它的 Gate 最早到 `11,120 us` 才能生效；
IO arrival 没有被量化到 80-us 网格，被量化的是控制写。

一句话记忆：**`a` 是整桌客人进店，`r` 是这一道菜的单子送进后厨。**

---

## 3. current clock 是哪个时间？

current clock 是控制器**本次做决策时**的仿真时刻，也就是传入
`recalculate(..., event_time_us=t)` 的事件时间。控制器内部的
`current_time_us` 只会单调前进：

```text
current_time_us = max(current_time_us, event_time_us)
```

在时间线中，`P_A` 最后一条 path 于 `10,402 us` Queue-empty，控制器立即重算：

```text
t = current clock = 10,402 us
```

但新 Gate 不能在 `10,402 us` 立即生效。Queue-empty 回调发生在本时刻的
QoS `rate_update` 阶段之后，必须使用严格下一边界：

```text
e = floor(10,402 / 80) * 80 + 80 = 10,480 us
```

所以一定要区分：

```text
t = 算优先级时“现在几点”
e = 这次 CIR/PIR/WRR 写入何时真正生效
```

严格 80-us 规则有两种：

```text
新 arrival:    e = inclusive_ceil_to_80(t)
Queue-empty:   e = strictly_next_80_boundary(t)
```

因此，arrival 恰好在 `10,000 us` 时可以使用 `10,000 us` 这个 tick；
Queue-empty 即使恰好在 `10,000 us`，也只能使用 `10,080 us`。

一句话记忆：**`t` 是调度员看表做决定，`e` 是仓库下一次允许扳动闸门的钟点。**

---

## 4. `service_window_us` 是时长还是时刻？

`service_window_us=c` 是一个**持续时间**。在当前 LLM 流水线里，它等于一层的
GPU 计算时间：计算当前层 L 时，SSD 可以在这段时间内预取 L+1。

对 `P_B`：

```text
c_B = 500 us
```

当 L0 真正在 `11,105 us` 开始计算时，L1 的理想隐藏窗口是：

```text
[11,105, 11,105+500] = [11,105, 11,605] us
```

它不等于下面任何一个量：

- 不是绝对 deadline；`500 us` 不是钟表上的时刻。
- 不是 SSD 剩余服务时间 `b`；例子中 Stage-0 的 `b_B=601 us`。
- 不是“已经等待多久”；那是 `t-a`。
- 不是 `deadline-now`；窗口是在读组创建时随计算计划给出的，不随时钟减少。

Stage 0 也携带同一个 `c`，但此时 GPU 还没有开始任何层计算。因此 Stage 0 的
`c` 只是 Utility 对“启动这个 GPU 后可得到多少计算收益”的输入，**不是一个正在
倒计时的真实 prefetch window**。

Stage-0 的整数版价值密度定义为：

```text
F = max(0, t-a) + b + G*c
U = c / (F*b^2)
```

在 `t=a_B=9,953 us` 时：

```text
F_B = 0 + 601 + 4*500 = 2,601 us
U_B = 500 / (2,601 * 601^2)
    = 500 / 939,483,801
```

实现不会真的做浮点除法；比较两个 Stage-0 候选时用大整数交叉相乘。`b` 被平方
惩罚，所以占 SSD 很久的候选会明显降权。

一句话记忆：**`c` 是厨师接下来能连续做菜多久，不是食材几点必须送到。**

---

## 5. `deadline_us` 怎样产生、怎样使用？

`deadline_us=d` 是一个**绝对时刻**。当前请求没有显式 deadline 时，DPU gateway
和 controller 都采用同一默认规则：

```text
d = r + c
```

### Prefetch 的 deadline 是真的

`P_A` 在 `9,000 us` 开始计算 L0，同时预取 L1：

```text
r_A = 9,000 us
c_A = 1,000 us
d_A = 10,000 us
```

这表示如果 L1 的 IO 在 `10,000 us` 之后才真正完成，L0 计算结束后可能要等
SSD，产生 layer stall。EDF 以 deadline 升序排列 ready prefetch；deadline
相同时再比较 remaining service、已完成 coflow 数、arrival sequence 和
`p_node_id`。

### Stage 0 的 deadline 只是占位值

`P_B:L0` 也会因默认规则存下：

```text
d_B0 = 9,953 + 500 = 10,453 us
```

但 Stage 0 的 `compute_layer_index is None`，此时根本没有正在进行的 GPU 计算，
所以这个 `10,453 us` **不能拿来做 EDF deadline**。控制器按 stage type 分流：

```text
compute_layer_index is None      -> Stage 0 -> Utility
compute_layer_index is not None  -> Prefetch -> EDF
```

### Allowance 不会改写 deadline

`L=750 us` 只在“能否把一个新 Stage 0 插到 EDF 队列前面”的可行性检查中使用。
在 `t=9,953 us`，若先做 `P_B` 再做 `P_A`：

```text
预计 P_A 完成 = t + b_B + b_A
              = 9,953 + 601 + 402
              = 10,956 us

允许最晚       = d_A + L
              = 10,000 + 750
              = 10,750 us
```

因为 `10,956 > 10,750`，插入会冲突，控制器先选 EDF 的 `P_A`。实现使用严格
`>`；若两边恰好相等，仍视为可行。多个 prefetch 存在时会按 EDF 顺序逐个累加，
检查每一个前缀，而不是只看队首。

一句话记忆：**`d` 是墙上写着的交付钟点；`L` 是决策时容许的缓冲，不会把墙上
的钟点擦掉重写。**

---

## 6. `path_bytes` 为什么不是整个读组的字节数？

`path_bytes=B_s` 是一个读组在**某一块 SSD path** 上最初提交的总字节。不同 SSD
上分别保存自己的值。

对 `P_B:L0`：

```text
SSD0 path_bytes = 24,035,328 B
SSD1 path_bytes = 17,694,720 B
```

整个多 SSD 读组的字节和是：

```text
24,035,328 + 17,694,720 = 41,730,048 B
```

但当前 Utility+EDF 不把 `41,730,048 B` 当成一条串行 path。两块 SSD 可以并行，
所以先分别算 `b_s`，再取最慢 path：

```text
b_B = max(601, 443) = 601 us
```

若错误地把两个 path 时间相加，会得到 `1,044 us`，这与当前并行 coflow 语义不符。

`path_bytes` 是注册 Demand 时的原始总量，之后不会随着 Queue depth 下降而改写。
当前剩余字节另算为 `R_s`。保留原值有两个作用：

- 可以把 live depth 转成 remaining bytes；
- 用 `min(path_bytes, q*S)` 防止异常 depth 把剩余量算得比原始提交量还大。

一句话记忆：**`path_bytes` 是“这座仓库原来一共要发多少货”，不是两座仓库的
合计，也不是现在还剩多少。**

---

## 7. `path_request_count` 为什么不是当前 Queue depth？

`path_request_count=N_s` 是该 SSD path 在 Demand 注册时**最初提交的完整 IO 数量**。
它是原始元数据，不随下发过程递减。

对 `P_B:L0`：

```text
SSD0 original path_request_count = 163
SSD1 original path_request_count = 120
```

它和 live Queue depth 的关系是：

```text
刚完整入队时：q 通常等于 N
开始下发后：  0 <= q < N
全部离开 QoS：q = 0，但 SSD 内仍可能有 inflight IO
```

当前实现用 `q<N` 判断 owner 是否已经真正开始下发。一旦 DPU 在重算时观察到任一
path 出现这个条件，owner 就 sticky，后来的更优 Utility 或更早 deadline 都不能
抢占当前读组。若中间没有事件唤醒 DPU，当前选择本来就不会改变；第一条 path
Queue-empty 时，释放逻辑也会在删除它之前持久化 sticky lock。

在例子中，假设 `P_A` 到 `10,160 us` 时因另一个 arrival 触发一次完整 depth
读取，则会看到：

```text
SSD0: q=66 < N=109
SSD1: q=37 < N=80
```

所以 `P_A` 已开始，必须保持 owner，直到其所有 SSD path 都 Queue-empty。

DPU gateway 会先把一个层读组的所有普通 IO 完整提交进逻辑 Queue，再按
`(SSD, queue_id)` 注册一次聚合 Demand。这样 `N` 表示完整批次，而不是“登记到
一半时碰巧看见的数量”。

一句话记忆：**`N` 是发货单原来有几箱，`q` 是此刻装货台上还压着几箱。**

---

## 8. `block_size_bytes` 与前两者是什么关系？

`block_size_bytes=S` 是一个完整 IO/KV Block 的大小。正式工作负载中 Block 等长：

```text
S = 147,456 Byte
```

这个数也与正式配置的 `queue_max_io_size_bytes` 一致。按模型参数可以手算为：

```text
128 tokens/block * (512 KV-LoRA rank + 64 RoPE dim) * 2 Byte
= 147,456 Byte/block
```

在等长例子中：

```text
path_bytes = path_request_count * block_size_bytes
B_s        = N_s * S
```

例如 `P_B/SSD0`：

```text
163 * 147,456 = 24,035,328 B
```

三个字段单位不同，不能互换：

| 字段 | 单位 | 是否随运行变化 |
|---|---|---|
| `path_bytes` | Byte/path | 否 |
| `path_request_count` | IO/path | 否 |
| `block_size_bytes` | Byte/IO | 否 |
| Queue depth | IO | 是 |

如果一个 path 内的普通 IO 大小不一致，gateway 会把
`uniform_block_size_bytes` 设为 `None`。此时 controller 不使用 `q*S`，而按
原始字节与原始请求数做比例回退：

```text
remaining_bytes = ceil(path_bytes * q / path_request_count)
```

这只是无法知道每个剩余 IO 大小时的近似；正式 KV 工作负载使用等长 Block，走的
是精确的 `q*S` 分支。

一句话记忆：**一箱多重是 `S`，原来几箱是 `N`，原来总重是 `B=N*S`。**

---

## 9. Queue depth 怎样变成 remaining service，Queue-empty 又意味着什么？

Queue depth `q_s` 是某条 path 上**尚未离开 QoS 的完整 IO 数量**。等长 Block 下，
当前实现计算：

```text
R_s = min(path_bytes, q_s * block_size_bytes)
b_s = ceil(R_s * 1,000,000 / SSD_capacity_bytes_per_second)
b   = max_s(b_s)     # 同一读组的多 SSD path 并行
```

### 用 `P_B` 在 `10,640 us` 的教学快照手算

假设此时读取一次 depth（这一步用于手算；QoS 不会为每个 IO 出队都回调 DPU）：

```text
SSD0 q_0 = 120
SSD1 q_1 = 77
```

则：

```text
SSD0:
R_0 = 120 * 147,456 = 17,694,720 B
b_0 = ceil(17,694,720 / 40,000) = 443 us

SSD1:
R_1 = 77 * 147,456 = 11,354,112 B
b_1 = ceil(11,354,112 / 40,000) = 284 us

P_B group remaining service:
b_B = max(443, 284) = 443 us
```

这里的 `b` 是按标称整盘带宽估出的、仍由 DPU Gate 控制的 Queue 内容需要多久，
不是 SSD 后端的精确完成时间。已经离开 QoS、进入 FCP/BCP/NFI/NAND/BDP/DAS
流水线的 IO 不再计入 `q`，但可能仍未 completion。

因此：

```text
Queue-empty:   q=0，表示装货台清空，可以释放 Demand/轮换 owner
SSD completion: IO 真正走完 SSD 后端，LLM 才能越过层屏障
```

时间线中 `P_B:L0` 的两条 path 分别在 `10,923` 和 `11,081 us` Queue-empty，
controller 在第二条 path 也空后释放 owner；LLM 却要等假设的最后 completion
`11,105 us` 才开始 L0 计算并提交 L1。

一句话记忆：**装货台没有箱子，只说明货都上路了，不说明货已经到厨房。**

---

## 完整状态机

下面的状态机把九个量放回实际控制流程。`PIR=None` 在代码中表示 uncapped。

```text
┌──────────────────────────────────────────────────────────────┐
│ t=0: PRE-PARK                                                │
│ balanced_exclusive 实际绑定的受管 Queue -> (CIR=0,PIR=0,W=0) │
└──────────────────────────┬───────────────────────────────────┘
                           │ 新读组完整入队并注册 Demand
                           v
┌──────────────────────────────────────────────────────────────┐
│ CLASSIFY                                                     │
│ compute_layer_index is None     -> Stage 0 / Utility         │
│ compute_layer_index is not None -> Prefetch / EDF            │
└──────────────────────────┬───────────────────────────────────┘
                           │ 数据事件触发重算，使用 current clock t
                           v
┌──────────────────────────────────────────────────────────────┐
│ CHOOSE                                                       │
│ 1. 若当前 owner 已开始：保持 sticky                         │
│ 2. 否则找最高 Utility 的 Stage 0                            │
│ 3. Prefetch 按 EDF 排序                                      │
│ 4. 检查插入 Stage 0 是否使任一 EDF 前缀超过 d+allowance     │
│    冲突 -> EDF 队首；不冲突 -> 最佳 Stage 0                  │
└──────────────────────────┬───────────────────────────────────┘
                           │ 控制写排到合法 80-us effective time e
                           v
┌──────────────────────────────────────────────────────────────┐
│ GATE                                                         │
│ selected p_node paths -> (CIR=SSD capacity,PIR=uncapped,W=1) │
│ all other managed paths -> (0,0,0)                           │
└──────────────────────────┬───────────────────────────────────┘
                           │ 任一路径 q < original N
                           v
┌──────────────────────────────────────────────────────────────┐
│ STICKY OWNER                                                 │
│ 当前读组不可抢占；一块 SSD 先 Queue-empty 也不换 owner       │
└──────────────────────────┬───────────────────────────────────┘
                           │ 当前 demand group 的所有 SSD path q=0
                           v
┌──────────────────────────────────────────────────────────────┐
│ RELEASE                                                      │
│ 删除空 Demand、读组完成计数 +1、解除 owner、立即重新选择     │
│ Queue-empty 引起的控制变化只能在严格下一 80-us tick 生效      │
└──────────────────────────┬───────────────────────────────────┘
                           │ SSD 后端稍后给 LLM completion
                           v
┌──────────────────────────────────────────────────────────────┐
│ NEXT LAYER                                                   │
│ LLM 越过计算/IO 屏障，提交下一读组；它必须重新参加 Utility/EDF│
│ owner 不跨层继承，固定 Queue 仍 pre-park，防止层间 EXCESS 偷跑│
└──────────────────────────┬───────────────────────────────────┘
                           │ 同一 p_node 的第 4 个读组全部 path 排空
                           v
┌──────────────────────────────────────────────────────────────┐
│ RESTORE DEFAULT                                              │
│ 该 p_node 的固定 Queue -> (CIR=0,PIR=uncapped,W=1)           │
└──────────────────────────────────────────────────────────────┘
```

### Sticky owner 的准确边界

Sticky 不是“选中就永远不能换”：

- 刚选中、所有 path 仍满足 `q=N` 时，说明还没有 IO 真正离开 Queue；后注册的更优
  候选仍可替换它，避免 Python 提交顺序决定结果。
- 任一路径出现 `q<N` 后，owner 才被视为 started，并保持到当前 demand group 的
  所有 SSD path 都 Queue-empty。
- 某一块 SSD path 先空时，controller 在删除该 path 之前先持久化 lock，避免另一块
  SSD 的 depth 快照暂时陈旧而错误抢占。
- 最后一条 path 消失后 lock 自动释放；同一 GPU 的下一层不会继承这个 lock。

### Pre-park 的准确范围

正式 `balanced_exclusive` 拓扑每块 SSD 有 256 条 Queue，但 128 个 GPU 只实际绑定
其中 128 条。初始化时优先只 pre-park 这些受管 Queue；未使用 Queue 保持默认值。
若 binding 没有提供预绑定映射，gateway 才保守回退为 pre-park 全部 Queue。

Pre-park 状态为：

```text
CIR=0, PIR=0, Queue WRR weight=0
```

仅把 CIR 设为 0 不足以阻断 Queue，因为它仍可能走 EXCESS。`PIR=0` 和
`weight=0` 才让等待 Queue 真正关门。

### 同一控制 tick 的覆盖

同一 Queue 可能因 Queue-empty、跨 SSD 协同重算和新层 arrival，在同一个 80-us
tick 收到多次写入。当前事件引擎对相同 `(effective_time, queue_id)` 保留最后状态。
因此时间线里 `11,081 us` 的 Stage-0 释放和 `11,105 us` 的 L1 arrival 都可能指向
`11,120 us`；最终以该 tick 的最后一次全局决策为准。

---

## 一页纠错清单

| 容易说错的话 | 当前实现的准确说法 |
|---|---|
| “service window 是 deadline。” | `c` 是时长；`d` 是绝对时刻。默认 `d=r+c`。 |
| “inference arrival 就是每一层 arrival。” | `a` 一次推理固定不变；每个读组有自己的 `r`。 |
| “current clock 就是控制生效时刻。” | `t` 用于算优先级；写入只能在稍后的合法 tick `e` 生效。 |
| “Stage 0 也有 `r+c`，所以进 EDF。” | Stage 0 的存储 deadline 是占位值；`compute_layer_index=None`，按 Utility。 |
| “allowance 把 deadline 延长了 750 us。” | deadline 本身不变；只在插入可行性检查中比较 `finish>d+750`。 |
| “path bytes 是这一层所有 SSD 的总字节。” | 它只属于当前 SSD path；多 path 分别计算。 |
| “request count 是当前还剩的请求数。” | `N` 是原始提交数；live 剩余数是 `q`。 |
| “block size、count、bytes 都是在说数据量。” | 单位分别是 Byte/IO、IO/path、Byte/path，等长时 `B=N*S`。 |
| “两块 SSD 的服务时间应该相加。” | path 并行，当前候选 `b=max_s(b_s)`。 |
| “Queue-empty 就说明这层读完了。” | 只说明 IO 离开 QoS；LLM 仍等 SSD completion。 |
| “owner 一选中就不能换，且一直跟着这个 GPU。” | 首个 IO 下发后才 sticky；只持续到当前读组所有 path Queue-empty。 |
| “换层后 Queue 自动恢复默认 EXCESS。” | 未完成四个读组前仍保持 pre-park，下一层必须重新获选。 |
| “Utility 使用 Placement 算出的 requested CIR。” | Utility 用元数据排序；选中后 Gate 写整盘 capacity CIR。 |

---

## 关键限制

1. **DPU 看不到 SSD 内部状态。** 它只读 Queue depth，不读 completion、inflight、
   FCP、NAND 等状态，因此 `b` 不是后端精确完成时间。
2. **服务时间是标称容量估计。** `ceil(remaining_bytes/capacity)` 没有把后端固定
   延迟、命令槽竞争和已经 inflight 的 IO 算进去。
3. **Utility 是启发式代理，不是精确 TTFT。** `F=(t-a)+b+G*c` 只是在线完成时间
   代理；实现不声称它给出全局最优调度。
4. **Stage-0 Utility 没有无限流 starvation 上界。** 等待变长会增大 `F`，不构成
   严格 aging 保证。当前“最终都完成”依赖正式实验是有限闭合负载。
5. **一次全局只准入一个 p_node。** 这便于形成完整 burst 和跨 SSD owner lock，
   但在 path 极不均衡时可能留下某些 SSD 空闲。
6. **EDF 保护的是估计可行性。** `deadline+750 us` 是调度裕量，不能保证真实 SSD
   completion 一定在这个时刻之前。
7. **等长 Block 分支最准确。** 非等长 IO 时 controller 只能按 `B*q/N` 比例估计
   remaining bytes。
8. **时间元数据会整数化。** Controller 对非负时间使用 `round` 转成整数微秒，
   因此亚微秒差异不会保留。
9. **控制量化会产生额外等待。** 数据按真实时刻到达 Queue，但 Gate 只能在 80-us
   网格改变；Queue-empty 即使发生在边界上，也必须等严格下一 tick。
10. **DPU 不会为每次 depth 递减都被唤醒。** 当前 QoS 只在至少一条 Queue 从
    非空变为空后发状态回调；新 arrival 也会触发重算。每次重算都主动读取完整
    depth 快照，但不存在逐 IO dispatch 通知。
11. **`requested_cir_bytes_per_second` 不是 Utility 分数。** Placement/gateway 仍可
    生成并传入该值，但 `UtilityEDFController` 的候选元数据不保存它；正式默认下，
    被选 path 使用整块 SSD capacity 作为 CIR，PIR 保持 uncapped。

---

## 源码索引

下表行号对应本文编写时的当前工作区版本。链接指向实现文件；行号用于快速定位。

| 主题 | 源码位置 | 关键内容 |
|---|---|---|
| 正式硬件与工作负载参数 | [`config/simulation_config.yaml`](../config/simulation_config.yaml)，10–68、70–104、128–155 行 | 128 GPU、4 层、Block、80 us、40 GB/s、策略入口 |
| Stage 0 / compute-prefetch 语义 | [`llm_workload/layer_request.py`](../llm_workload/layer_request.py)，261–355 行 | 初始读 `compute_layer_index=None`；计算 L 时预取 L+1；推理 arrival 与 service window 透传 |
| Placement 按 SSD 聚合 path bytes | [`llm_workload/kv_placement_manager.py`](../llm_workload/kv_placement_manager.py)，76–138 行 | 每块 SSD 字节聚合、requested rate、Demand 元数据 |
| Gateway 完整批次与 path 元数据 | [`DPU/dispatcher.py`](../DPU/dispatcher.py)，365–581 行 | 先完整入队；推导 path bytes/count/block size；默认 `deadline=r+c` |
| Demand 字段登记 | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1224–1326 行 | 时间整数化、deadline fallback、原始 path 元数据、初始 depth |
| Remaining bytes/service | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1328–1419 行 | `q*S`、非等长回退、容量向上取整、多 SSD 取最大值 |
| Stage-0 Utility | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1421–1479 行 | `F`、整数交叉乘法、平局规则 |
| Prefetch EDF 与插入检查 | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1482–1554 行 | EDF key、started 判断、sticky、`deadline+allowance` |
| Queue Gate 状态 | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1597–1628 行 | selected capacity/uncapped/1，其他 0/0/0 |
| current clock 与 Queue-empty 释放 | [`DPU/rate_controller.py`](../DPU/rate_controller.py)，1679–1793 行 | 单调时钟、depth 更新、多 path 删除、第四组恢复默认 |
| 80-us effective time | [`DPU/dispatcher.py`](../DPU/dispatcher.py)，218–257 行 | arrival inclusive ceil、Queue-empty strictly next |
| Queue 状态回调与跨 SSD 重算 | [`DPU/dispatcher.py`](../DPU/dispatcher.py)，617–682 行 | 只读完整 depth、empty 后控制阶段、协同写入 |
| 接口边界与状态摘要 | [`DPU/README.md`](../DPU/README.md)，11–47、88–183 行 | DPU 可见信息、Gate、sticky、park、Queue-empty 与 completion |
| 严格 80-us 回归测试 | [`tests/test_utility_edf_strict_80us_contract.py`](../tests/test_utility_edf_strict_80us_contract.py) | pre-park、边界对齐、同 tick 覆盖、跨层 park |
| 多 SSD sticky 回归测试 | [`tests/test_utility_edf_multissd_owner_lock.py`](../tests/test_utility_edf_multissd_owner_lock.py) | 一条 path 先空时仍保持 owner |

配套的 1–10 SSD、C/U/E `2^3` 正交消融结果与完整 TTFT 报告见
[`experiments/results/utility_edf_component_ablations/analysis/REPORT.md`](../experiments/results/utility_edf_component_ablations/analysis/REPORT.md)。

## 最后用九句话复述

1. inference arrival 是整次推理的出生时间，四个读组一直携带它。
2. path arrival 是当前读组到 DPU 的时间，每层都可能不同。
3. current clock 是这次决策的“现在”，不是 80-us Gate 的生效时刻。
4. service window 是 GPU 计算持续时间，Stage 0 只把它用于 Utility。
5. deadline 是 Prefetch 的绝对截止时刻；Stage-0 的默认值不参与 EDF。
6. path bytes 是一个读组落在一块 SSD 上的原始总字节。
7. path request count 是该 path 的原始 IO 数，不是 live 剩余数。
8. block size 是每个 IO 的字节数；等长时 `path_bytes=count*block_size`。
9. Queue depth 是 QoS 尚未下发的 IO 数；它能估 remaining service，但不能说明
   SSD 已 completion。
