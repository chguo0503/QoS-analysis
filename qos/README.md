# 256 Queue分层QoS

QoS为每块SSD独立实例化。一个StoragePath由一个QoS和一个ASU SSD后端组成，
不同SSD的Queue、令牌、调度器和反压状态彼此独立。

## Queue布局

QoS固定创建`q000`～`q255`共256条Queue，平均分成8个Group：

```text
g0: q000–q031    g4: q128–q159
g1: q032–q063    g5: q160–q191
g2: q064–q095    g6: q192–q223
g3: q096–q127    g7: q224–q255
```

Group只是分层WRR的调度单元，不创建Group CIR/PIR令牌桶。每条Queue独立
保存FIFO、CIR令牌桶和可选PIR令牌桶。单块SSD的40 GB/s物理总上限由
SSD后端及其反压建模，QoS不重复创建Root令牌桶。

## CIR-first与EXCESS两轮调度

每次成功只选择一个完整IO，然后重新从CIR轮开始：

```text
CIR轮：
  Queue非空
  && Queue CIR token足够
  && Queue PIR允许

EXCESS轮：
  Queue非空
  && Queue PIR允许
```

QoS先在所有CIR-eligible Queue中执行分层WRR。只有当全局没有CIR-eligible
Queue时，才进入EXCESS轮。`PIR=uncapped`时不创建PIR桶，因此即使
`CIR=0`，非空Queue仍可以通过EXCESS下发。

不存在每IO Admission Gate，CIR与EXCESS轮也不读取Demand ID、requested CIR
或assigned CIR。QoS只能看到DPU最终写入的Queue CIR/PIR。

## WRR

两轮调度共用同一套分层调度器：

```text
Group平滑加权RR选择group_id
              ↓
获胜Group内的Queue加权RR选择queue_id
              ↓
从该Queue的FIFO取出一个IO
```

`experiments/config/uniform_wrr.yaml`将全部Group和Queue权重设为1。
Baseline与`demand_aware_fcfs_cir`全程使用这个固定权重，不写动态
Group权重。

QoS仍保留`schedule_group_weight_update()`和`set_group_weights()`，以后的策略可以
通过它调整Group间的调度机会；该接口只修改WRR，不创建Group令牌桶。

## 动态Queue CIR/PIR

DPU使用`schedule_queue_rate_update()`指定Queue、CIR每周期补充字节、可选PIR
每周期补充字节和生效时刻。速率更新保留Queue FIFO，但会用新参数重建
该Queue的令牌状态。

实验使用80 μs refill周期和`initial_tokens: empty`。因此：

- t=0动态设置CIR时，新CIR桶的token为0。
- 第一次80 μs refill前，Demand-aware Queue可能先通过EXCESS下发。
- CIR-first带来的差异从后续token refill开始出现。

`experiments/config/uniform_baseline_token_bucket.yaml`中所有Queue的静态CIR为0、
PIR为uncapped。Demand-aware策略在整层IO入队后只覆盖活跃Queue的CIR，
PIR仍为uncapped。

## Queue状态接口

`queue_io_counts()`返回`queue_id -> 尚未下发IO数`快照。统计包括已登记但
尚未在当前事件阶段移入FIFO的同时刻请求，避免DPU把批量入队中间状态
误认为空Queue。

Queue成功出队后，QoS可以用仅携带`event_time_us`的事件唤醒DPU。DPU再主动
读取depth快照。该通道不向DPU暴露具体请求或Demand信息。

## 与SSD后端的关系

QoS在后端能够接收时非阻塞下发IO。SSD可以将一个144 KiB Block拆成
4 KiB命令进入硬件流水线，但这不改变DPU和QoS之间的完整IO边界。
Queue变空只表示IO已离开QoS，不代表SSD已完成。层读取时间仍由
LLM收到的完整SSD completion计算。

## 公共入口

```python
from qos import build_qos_simulator, load_queue_layout

qos = build_qos_simulator(
    layout_config_file,
    token_config_file,
    scheduler_config_file,
    qos_runtime_config_file,
    start_time_us=0,
)
qos.input(request)
result = qos.end()
```

联合SSD仿真不单独调用`qos.run()`，而是由顶层全局事件日历交错推进
GPU、DPU、QoS和各个SSD。
