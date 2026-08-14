# DPU Queue绑定与CIR控制

`DPURequestGateway` 连接KV Placement和每块SSD的独立QoS。DPU不解析
LLM模型、不拆分IO，也不读取SSD内部流水线状态。

## 硬件接口边界

DPU→QoS只有两类操作：

- 提交普通IO：`request_id`、`p_node_id`、`storage_target_id`、
  `size_bytes`、`queue_id`和`arrival_time_us`。
- 设置指定Queue的CIR/PIR。CIR/PIR写入使用整数Byte/s；
  `PIR=None`表示uncapped。

QoS→DPU只提供Queue空/非空或depth快照。Queue状态变化事件只用来
唤醒DPU，不携带request ID、Demand ID或逐IO dispatch信息。

KV Placement输入DPU的每个Block还会携带：

```python
{
    "demand_bw": {
        "demand_group_id": "...",
        "aggregate_required_bytes_per_second": 4_000_000_000,
    }
}
```

这些Demand字段只在DPU内部使用，不转发给QoS。
`aggregate_required_bytes_per_second`由`KVPlacementManager`按下式唯一计算：

```text
requested CIR = ceil(bytes_on_ssd × 1,000,000 / service_window_us)
```

同一`(GPU, layer, SSD)`的多个Block重复携带同一聚合值。DPU在整层IO
全部提交给QoS后，每个`(SSD, queue_id)`只登记一次Demand，不按Block
重复累加。

## 独占Queue绑定

`balanced_exclusive`在实验开始前将每条`(GPU, SSD)`路径绑定到独占
Queue。64 GPU的固定公式是：

```text
group_id   = gpu_index % 8
queue_index = 32 × (gpu_index % 8) + floor(gpu_index / 8)
```

因此每块SSD正好使用64条不同Queue，8个Group各有8条。SSD0和SSD1
拥有独立Queue namespace，所以可以复用相同`queue_id`。

## Baseline

Baseline不创建Demand控制器，也不动态写速率：

```text
Queue CIR = 0
Queue PIR = uncapped
Group WRR = 1
Queue WRR = 1
```

所有非空Queue通过QoS的EXCESS轮参与调度。

## Demand-aware FCFS CIR

`demand_aware_fcfs_cir`为每块SSD维护独立的活跃Demand到达顺序，
并只使用整数比较、减法和`min`分配CIR：

```text
remaining_capacity = 40,000,000,000 Byte/s

for demand in active_demands_by_arrival_order:
    assigned_cir = min(requested_cir, remaining_capacity)
    remaining_capacity -= assigned_cir
```

例如A先到并请求30 GB/s，B后到并请求20 GB/s，则A获得30、B获得10。
A结束后，B按原到达顺序重算并恢复到20 GB/s。

assigned CIR=0只表示没有带宽保障，不禁止IO下发。所有Queue的PIR
始终uncapped，所以仍可通过EXCESS借用空闲带宽。该策略不修改
Group或Queue WRR，两者均保持为1。

## Demand结束

DPU按`(SSD, queue_id)`保存当前Demand。整层的全部IO已批量入队后才
登记Demand，因此不会把批量入队前的初始空Queue误判为完成。

Queue depth变为0表示该Demand的IO已全部离开QoS；它不表示SSD已完成这些IO。
此时DPU可以释放QoS CIR，删除Demand，并按原到达顺序重算剩余Demand。

## 动态Group WRR接口

`set_group_weights()`和QoS的动态Group权重事件仍保留，供未来DPU策略使用。
本次Baseline和`demand_aware_fcfs_cir`都不调用它，因此实验的
`group_weight_write_count`必须为0，Group WRR权重始终为1。
