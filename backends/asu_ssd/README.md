# ASU SSD离散事件后端

本目录只实现SSD内部流水线，整体装配位于项目根目录。

## 公开接口

`simulator.py` 提供 `SSDSimulator`：

```python
ssd = SSDSimulator(ssd_config, completion_sink=on_complete)
ssd.input(request)
ssd.run_until_idle()
result = ssd.end()
```

- `input()` 接收普通IO，返回真正进入FCP的 `accepted_time_us`。
- 输入字典只需 `request_id`、`queue_id`、`size_bytes` 和 `dispatch_time_us`。
- FCP暂时不能接收时，SSD内部事件会推进到可接收时刻，不丢请求。
- `run_until_idle()` 排空当前已进入SSD的请求，但不结束SSD；下一层仍可继续输入。
- 完整请求的最后一个DAS原子完成时，`completion_sink` 只收到一次
  `request_id` 和 `completion_time_us`。
- `end()` 排空全部已接收IO并返回统计；重复调用返回同一结果。
- QoS不再发送0 Byte结束符。顶层在LLM算出TTFT后直接调用 `end()`。

## 内部数据流

```text
上层可变大小SSD请求
    |
    v
FCP(8 KiB原子) -> BCP(4 KiB) -> NFI(4 KiB) -> NAND(4 KiB)
    -> BDP(4 KiB) -> DAS(8 KiB原子) -> 请求完成
```

- 请求大小来自每个描述符的 `size_bytes`，后端再动态拆成多个4 KiB命令。
- 上层KV block是UCM layerwise模式中某一层128 tokens的KV Cache，当前为144 KiB。
- 后端4 KiB命令是SSD流水线内部粒度，与上层KV block不等价。
- FCP和DAS把相邻两个4 KiB命令组成8 KiB原子。
- 所有阶段均使用事件启动、完成和下游传递，不按CPU时钟逐周期扫描。
- 已完成但下游暂时不能接收的数据继续占用本阶段槽位，从而逐级形成反压。
- NAND当前是“聚合带宽 + 固定读取延迟 + 命令槽位”的抽象模型，不包含真实Channel/Die/Plane、FTL和GC。

详细参数见 [ASU后端仿真参数.txt](./ASU后端仿真参数.txt)，可运行配置见
`config/asu_backend_config.yaml`。

## 单SSD联合仿真的物理Root

本后端是当前联合仿真中唯一的共享物理出口。NAND通过4 KiB命令的
启动间隔建模聚合40 GB/s上限，各流水线阶段的有限槽位保留在途命令。
当FCP入口暂时无法接收新请求时，`input()` 在SSD内部推进事件，并向上游
返回真正进入FCP的 `accepted_time_us`。QoS根据这个时刻推进自己的时钟，
因此SSD的物理容量和反压会直接约束QoS下发时间线。

QoS不再创建一个同样40 GB/s的Root令牌桶。QoS的Group/Queue CIR用于确定
最低保障的优先顺序，`uncapped` 和EXCESS让队列借用空闲容量；本SSD则负责
统一实现最终总带宽、入口反压和完成时刻。这样只有一个组件拥有物理容量，
不会在QoS中用80 μs令牌周期重复整形SSD已经建模的同一出口。

## 扩展新的SSD

新增SSD时，在 `backends/` 下建立新的独立目录，并实现相同的
`input()`、`run_until_idle()` 和 `end()` 接口。当前只有一个SSD，等多SSD场景确定后再在
顶层增加路由，QoS模块无需修改。

## 运行

从项目根目录执行：

```bash
python qos_ssd_simulator.py
```

联合模式的实际完成带宽取决于
工作负载、QoS两轮仲裁和流水线状态；40 GB/s是本SSD的聚合物理上限，
不代表每次联合实验都必然达到满载。默认Group和Queue PIR均为
`uncapped`，持续积压可以通过EXCESS把下发速度推到SSD的可接收上限；
只有路径上显式配置数值PIR时，QoS才可能在请求到达SSD之前先形成长期限速。
