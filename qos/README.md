# 256队列分层QoS仿真

## 公共入口

外部代码只从 `qos` 包导入装配函数，不直接导入 `entry.py`、令牌桶或具体调度器：

```python
from qos import build_qos_simulator, load_queue_layout

queue_layout = load_queue_layout(layout_config_file)
qos = build_qos_simulator(
    layout_config_file=layout_config_file,
    token_config_file=token_config_file,
    scheduler_config_file=scheduler_config_file,
    simulation_config_file=simulation_config_file,
    queue_layout=queue_layout,
)

qos.input(request)
qos.run()
result = qos.end()
```

`qos/__init__.py` 是唯一公共入口，`qos/entry.py` 负责读取配置并组装实例。
`entry.py` 会创建令牌桶和调度器，再把它们注入离散事件引擎。
QoS实例对普通请求只提供 `input()`、`run()` 和 `end()`；`set_backend()` 只由
LLM、DPU、QoS和SSD的联合装配代码使用。

## 目录结构

```text
qos/
├── __init__.py                         包级公共入口
├── entry.py                            创建各组件并注入事件引擎
├── queue_layout.py                     生成队列和分组映射
├── token_bucket.py                     单个基础令牌桶
├── token_bucket_stage.py               Group/Queue两级令牌路径和每队列FIFO
├── schedulers/
│   ├── __init__.py                     调度器子目录入口
│   ├── round_robin.py                  RR指针
│   ├── weighted_round_robin.py         基于整数权重的WRR
│   └── hierarchical.py                 组间WRR和组内WRR
└── config/                              队列、速率、权重和仿真配置
```

## 固定队列布局

QoS仿真器启动时固定创建 `q000`～`q255` 共256个输入队列，并平均分成8组：

```text
g0: q000～q031    g4: q128～q159
g1: q032～q063    g5: q160～q191
g2: q064～q095    g6: q192～q223
g3: q096～q127    g7: q224～q255
```

布局来自 `config/queue_layout_config.yaml`。

## 调度代码层次

调度逻辑按可复用关系拆成三个文件，离散事件引擎仍只调用一个
`select_next_queue(is_eligible)`：

```text
schedulers/round_robin.py
  RoundRobinScheduler              # 只维护循环指针
          ↑ 复用
schedulers/weighted_round_robin.py
  WeightedRoundRobinScheduler      # 展开整数权重位图，再交给RR
          ↑ 组合
schedulers/hierarchical.py
  HierarchicalScheduler            # 先组间WRR，再获胜组内WRR
```

`entry.py` 统一读取布局、令牌桶、权重和事件配置，联合SSD入口通过包级入口调用它。

## Group/Queue CIR、PIR和物理Root

QoS数据面只维护Group和Queue两级路径，不创建QoS Root令牌桶。单SSD联合
仿真的唯一共享物理出口就是SSD NAND：它以聚合40 GB/s的命令启动节奏工作，
有限流水线槽位和入口反压会把QoS时钟推进到真正可接收的时刻。在QoS中再创建
40 GB/s Root PIR会重复整形同一资源，并额外引入80 μs令牌量化，所以本实现
明确没有Root CIR、PIR、CBS或PBS。

`config/token_bucket_config.yaml` 使用十进制GB/s。CIR（Committed Information Rate）
是全部队列同时繁忙时的最低保障份额，不是峰值上限。PIR（Peak Information
Rate）是可选的本节点长期上限：

- `pir_gb_s: uncapped` 表示本节点不创建PIR桶，检查路径时跳过本级PIR。
- Group和Queue默认都是 `uncapped`，因此活跃队列可以借用同组、其他组和
  整个SSD的空闲能力。例如g7的0.384 GB/s Queue CIR只是保障值，不会把该
  队列的实际下发速度限在0.384 GB/s。
- Group或Queue也可以配置数值PIR。请求在CIR和EXCESS两轮中都必须通过
  路径上的每个有限PIR，下发后扣除这些PIR令牌。`PIR = CIR` 表示该
  节点没有长期可借用空间。

默认8个Group的CIR合计为32 GB/s：

| Group | Queue范围 | Group CIR (GB/s) | Group PIR |
|---|---|---:|---|
| g0 | q000–q031 | 1.28 | `uncapped` |
| g1 | q032–q063 | 1.92 | `uncapped` |
| g2 | q064–q095 | 2.56 | `uncapped` |
| g3 | q096–q127 | 3.20 | `uncapped` |
| g4 | q128–q159 | 3.84 | `uncapped` |
| g5 | q160–q191 | 5.12 | `uncapped` |
| g6 | q192–q223 | 6.40 | `uncapped` |
| g7 | q224–q255 | 7.68 | `uncapped` |

每组32个Queue使用交错权重 `[4, 3, 2, 1] × 8`，Queue CIR按组内总权重80
归一化：

```text
Queue CIR = Group CIR × Queue权重 / 80
```

因此g0的四档每Queue CIR为0.064/0.048/0.032/0.016 GB/s，g7为
0.384/0.288/0.192/0.096 GB/s，每个数值在组内出现8次。每组Queue CIR
之和精确等于该Group CIR，256个Queue合计仍为32 GB/s。Queue CBS为
589,824 B（4个144 KiB Block）；Group CBS为该组80 μs CIR补充量加一个
147,456 B Block，使上个周期不足一个Block的余量不会在refill时被截断。

## 两轮CIR/EXCESS调度

每次仲裁只使用一个 `HierarchicalScheduler`、同一套Group/Queue WRR权重和
同一套RR游标。事件引擎先扫描CIR；全系统没有CIR-eligible Queue时才扫描
EXCESS。一次成功只下发一个完整请求，随后立即回到CIR轮，而不会一次性
排空所有EXCESS请求。

```text
CIR轮：
  FIFO非空
  + Queue CIR令牌足够
  + Group CIR令牌足够
  + Group/Queue路径上所有有限PIR令牌足够
  -> 扣Queue CIR、Group CIR和所有有限PIR
  -> qos_rate_class = CIR

EXCESS轮：
  FIFO非空
  + Group/Queue路径上所有有限PIR令牌足够
  -> 不检查、不扣除任何CIR
  -> 只扣所有有限PIR
  -> qos_rate_class = EXCESS
```

这个顺序使最低保障始终优先，同时让 `uncapped` 队列在保障份额暂时不可用时
继续借用SSD空闲容量。因为不存在QoS Root，最终40 GB/s总带宽上限由SSD的
真实接收时刻和反压实现。

## 分层WRR和最终RR

一次调度按下面顺序完成：

```text
组间WRR选择group_id
        ↓
获胜组内WRR选择queue_id
        ↓
从该队列唯一FIFO取出一个IO
```

两个WRR都由“整数权重位图 + 最终RR指针”实现：权重N会展开成N个连续槽位，RR指针从上次
位置继续扫描，并跳过当前没有合格IO的槽位。

组间权重位图是：

```text
group:  g0 g1 g2 g3 g4 g5 g6 g7
weight:  2  3  4  5  6  8 10 12
```

8张组内权重位图均为32项，完整值保存在 `config/wrr_config.yaml`。
所有Group使用同一个重复模式：

```text
g0–g7: [4,3,2,1]×8
```

这里的“最终RR”是扫描加权槽位的仲裁动作。将来如果增加多个SSD或多个输出端口，端口之间的
RR属于后端路由层，应在顶层单独建模。

## 联合仿真的请求边界

默认联合场景由 `llm_workload/config/layer_request.yaml` 描述。GLM-5.1共有78层；输入
10,000 tokens、KV Cache命中率99%、Batch Size 1时，每层命中9,900 tokens，未命中
100 tokens。命中部分从SSD读取，未命中部分在GPU上重算。

LLM先把命中的KV Cache按128 tokens对齐成完整KV Block。当前每Token、每层的KV数据是
1,152 B，因此一个Block固定为147,456 B，也就是144 KiB；9,900个命中Token向上取整后，
每层生成78个Block。LLM只生成中立的层读取计划，其中包含P节点、服务窗口和Block列表，
不包含DPU、QoS或SSD的请求格式。

```text
有效KV字节数 = 命中Token数 × (kv_lora_rank + qk_rope_head_dim) × bits_per_element / 8
Block字节数 = tokens_per_block × 每Token、每层KV字节数
Block数 = ceil(命中Token数 / tokens_per_block)
实际传输量 = Block数 × Block字节数
```

默认单层有效数据是11,404,800 B，对齐后的实际传输量是11,501,568 B。GPU重算窗口约
2,051.282 us。`KVPlacementManager` 用固定随机种子把每个Block稳定映射到SSD，并按
“层 + SSD”汇总字节数和需求带宽。默认只有 `SSD0`，所以整层聚合需求约为
5.6070144 GB/s；同一组内的Block会重复携带这一组的聚合值，不能再按Block相加。

Placement Manager随后为每个Block封装DPU请求。`basic` 中包含 `request_id`、
`p_node_id`、`storage_target_id` 和 `size_bytes`；`demand_bw` 中包含
`demand_group_id` 和 `aggregate_required_gb_s`。当前DPU仍只按P节点绑定QoS队列，尚未使用
存储目标和聚合需求做调度。`aggregate_required_gb_s` 即使未来用于需求感知选队，
也只是队列选择依据，不会直接增加、扣除或改写任何QoS令牌。QoS真正的放行节奏
由请求大小、Group/Queue CIR、路径上的有限PIR和分层WRR共同决定。

## 从LLM到完成回调

```text
qos_ssd_simulator.py装配并运行完整链路
        ↓
LLM生成中立的层读取计划
        ↓
KVPlacementManager稳定映射Block到SSD、按SSD聚合需求并封装请求
        ↓ 逐个调用DPU.submit(request)
DPU首次看到P节点时随机选择队列并缓存绑定，暂不解释聚合需求
        ↓ 展平字段，一对一提交，不拆IO
QoS的Group/Queue两轮令牌准入 + 组间WRR + 组内WRR
        ↓ 逐个调用SSD.input(request)
SSD内部按硬件粒度处理
        ↓ 每个完整144 KiB请求只完成一次
completion callback直接通知LLM
```

DPU绑定是静态的：同一P节点的全部层和全部Block始终使用首次随机选中的QoS队列。当前DPU
不读取QoS或SSD状态，也不根据负载调整Group/Queue CIR、有限PIR或WRR权重。

默认联合入口只创建一个SSD后端，因此所有请求最终进入 `SSD0`。Placement Manager的独立
演示可以展示1、5、10个SSD的稳定分布和按SSD聚合结果，但联合仿真尚未连接多个SSD后端，
也没有按 `storage_target_id` 路由。

SSD可以在后端内部把144 KiB请求展开成4 KiB命令，但这只是SSD流水线实现，不会改变
LLM、DPU和QoS之间的一个Block请求边界。SSD完成通知只包含 `request_id` 和
`completion_time_us`，直接交给LLM匹配原请求。
QoS调用SSD时也会构造最小描述符，只传 `request_id`、`queue_id`、`size_bytes`
和 `dispatch_time_us`；P节点、存储目标和聚合需求不会进入当前SSD内部流水线。

## 层屏障与TTFT

LLM为当前层同时登记GPU计算结束时刻和78个未完成Block。只有收到本层全部78个SSD完成
通知后，才通过这一层的屏障：

```text
本层结束时刻 = max(GPU计算结束时刻, 本层最后一个Block的SSD完成时刻)
下一层开始时刻 = 本层结束时刻
TTFT = 最后一层结束时刻 - 工作负载到达时刻
```

因此SSD在GPU窗口内完成时不会增加该层时延；超过GPU窗口的部分才计入SSD等待。默认78层
的纯GPU计算TTFT是160 ms，最终TTFT等于160 ms再加上各层屏障产生的SSD额外等待。

## 结束接口

QoS只接收普通非零字节请求，不使用0 Byte结束符。联合仿真每层调用
`qos.run()` 下发该层请求，再调用SSD的 `run_until_idle()` 获得完成回调；
全部层结束后，`qos.end()` 整理QoS统计，SSD的 `end()` 由顶层单独调用。
