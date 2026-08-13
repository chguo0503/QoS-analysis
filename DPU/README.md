# DPU请求网关

`DPURequestGateway` 是LLM与QoS之间唯一的请求数据边界。DPU不导入
`llm_workload`，不生成工作负载，也不拆分IO。

## 请求格式

LLM通过 `submit(request)` 逐个提交完整KV Block。输入只包含：

```python
{
    "basic": {
        "request_id": "glm51_P0_all_layers_layer_00_block_00000",
        "p_node_id": "P0",
        "storage_target_id": "SSD0",
        "size_bytes": 147_456,
    },
    "demand_bw": {
        "demand_group_id": "glm51_P0_all_layers_layer_00_SSD0",
        "aggregate_required_gb_s": 5.6070144,
    },
}
```

`size_bytes=147_456` 就是144 KiB。`storage_target_id` 标识该Block所在的
存储目标；当前只有 `SSD0`，DPU只传递这个字段，不用它选择队列或
切换后端。

`demand_group_id` 标识一份聚合需求属于哪个需求组。默认场景中，
同一P节点、同一模型层、同一存储目标使用同一个组编号。
`aggregate_required_gb_s` 是该组在这个存储目标上的聚合带宽需求。
例如，默认每层78个144 KiB Block全部位于 `SSD0`时，聚合需求为
5.6070144 GB/s。同一组的每个Block可以携带同样的聚合值，这些值
表示同一份组状态，不能再按Block相加。

DPU只把 `basic` 和 `demand_bw` 展平，不解析模型、层或Token语义，也不在
`submit()` 中累加需求。当前随机绑定策略仍然只使用 `p_node_id`，
尚不使用 `storage_target_id` 或 `aggregate_required_gb_s` 做选队决策。

`aggregate_required_gb_s` 始终是DPU选队所需的需求描述，不是令牌补充指令。
即使后续实现需求感知策略，DPU也只能用它比较候选Queue和所在Group的
可借用容量；不能用它直接增加、扣除或改写Group/Queue CIR、有限PIR、
CBS或PBS。QoS令牌只能由QoS配置、80 μs补充周期和实际下发字节数驱动。

当前QoS中的Queue CIR是全256个队列同时繁忙时的最低保障，不是队列的
峰值能力。Group和Queue默认PIR都是 `uncapped`，请求可以在CIR轮之后通过
EXCESS借用同组、其他Group和SSD的空闲能力。因此，例如5.6070144 GB/s的
P节点不应因为g7某个Queue的CIR只有0.384 GB/s就把它判定为无法承载；
需求感知DPU应考察候选路径的当前积压、保障份额和可借用容量。最终40 GB/s
聚合上限由单SSD物理出口和反压实现，DPU不会构造QoS Root。

## 展平后的QoS请求

`submit()` 不拆分、合并或丢弃请求。一个输入Block对应一个输出字典：

```python
{
    "request_id": "glm51_P0_all_layers_layer_00_block_00000",
    "p_node_id": "P0",
    "storage_target_id": "SSD0",
    "size_bytes": 147_456,
    "demand_group_id": "glm51_P0_all_layers_layer_00_SSD0",
    "aggregate_required_gb_s": 5.6070144,
    "queue_id": "q046",
    "arrival_time_us": 0.0,
}
```

## P节点绑定

P节点首次调用 `submit()` 时，网关使用YAML中的 `random_seed` 从可用
QoS队列中选择一个，并在 `p_node_to_queue` 中缓存绑定。同一P节点
之后的所有请求始终进入该队列，即使请求的存储目标不同也不会重新绑定。
多SSD环境下的按存储目标绑定属于后续策略，本类目前不实现。

`submit()` 为请求添加 `queue_id` 和由 `clock()` 生成的 `arrival_time_us`，
调用注入的 `request_sink` 后返回同一份扁平QoS字典。返回字典而不是
`request_sink` 的返回值，可以让调用方直接核对实际入队的八个字段。

```python
gateway = DPURequestGateway(
    queue_ids=["q000", "q001"],
    random_seed=5102,
    request_sink=qos.input,
    clock=lambda: qos.current_time_us,
)
qos_request = gateway.submit(request)
```

SSD内部仍可把144 KiB请求分成4 KiB命令进入硬件流水线；这是SSD
后端行为，不是DPU拆分IO。

## 完成通知与TTFT

DPU只负责请求的前向提交，不接收或汇总SSD完成。SSD在一个完整144 KiB请求完成后，
通过completion callback把 `request_id` 和 `completion_time_us` 直接通知LLM。

LLM为每层维护78个Block的完成屏障，同时记录该层GPU计算结束时刻。收到本层全部
78个完成通知后，本层结束时刻取“GPU计算结束”和“最后一个SSD完成”中的较晚者，
下一层才会开始。最终TTFT由78层GPU计算时间和各层屏障产生的SSD额外等待共同决定。

QoS和DPU之间没有0 Byte结束请求。所有Block完成后，顶层分别调用QoS和SSD的结束
接口整理统计；结束动作不会伪造一个业务IO。
