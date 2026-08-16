# UCM Prefix KV SQE 到 QoS/SSD 的 Layerwise TTFT 闭环仿真设计

> 文档状态：实现稿
>
> 适用代码：`ucm-sqe-simulator` 与 `QoS分析` 当前实现
>
> 实验主题：GLM-5.1 Prefix Prefill / TTFT，不包含 Decode
>
> 最后核对：2026-08-17

## 1. 一句话目标

先由独立的 `ucm-sqe-simulator` 在普通 CPU 电脑上生成 UCM 协议合法的
KV SQE trace，再由 `QoS分析` 将每个 Retrieve Entry 作为一条 SSD IO，
通过 DPU/QoS/SSD 仿真完成逐层读取，并用 SSD completion 重算后续层的
真实下发时间和每张 GPU 的 TTFT。

这套方案回答三个核心问题：

1. 给定 Prefix 工作负载后，UCM 把每个 KV Block 放到哪个 ASU。
2. 多张 GPU 同时读取时，DPU/QoS 策略如何编排每块 SSD 的流量。
3. 路由和读取策略确定后，每层实际何时发出哪些 KV SQE，
   何时读完，最终 TTFT 是多少。

## 2. 当前实验边界

本设计的默认边界已经固定。

| 项目 | 当前取值 | 含义 |
|---|---:|---|
| 模型 | GLM-5.1 | 只使用与 KV 尺寸和 Prefill 计算相关的几何参数 |
| NPU/GPU 数 | 128 | 仿真中统一用 `gpu_id` 表示计算 worker |
| ASU 数 | 10 | 一个 ASU 就是一块 SSD |
| Transformer 层数 | 78 | `layer_id` 为 0～77 |
| Tensor Parallel | TP=1 | 每张 GPU/NPU 都拥有完整模型参数 |
| rank | `rank_id=0` | TP=1 worker 内唯一 rank，不是全局 GPU 编号 |
| 有效算力 | 512 TFLOPS/GPU | 用于估算每张 GPU 的单层 Prefill 计算时间 |
| 输入长度 | [100K, 200K] token | 每张 GPU 独立做闭区间整数均匀采样 |
| 缓存 Prefix 比例 | [0.60, 0.99] | 每张 GPU 独立采样 |
| 共享热点 GPU 比例 | 0.60 | 128 张中四舍五入得到 77 张 |
| 热点长度 | 50,000 token | 参考系是完整输入的前 50K token，可配置 |
| UCM Block | 128 token | 50K 中只使用完整 block |
| 实际共享热点 | 390 blocks | 390 × 128 = 49,920 token |
| 请求到达 | Uniform[0, 0.1s] | 每张 GPU 独立采样，单位 ns |
| 执行模式 | Layerwise | 读取与逐层计算形成闭环 |
| 操作 | read-only | 只回放 Retrieve，不生成 Store/写回 |
| Exist | 忽略 | 可解析和计数，不进入 QoS/SSD |
| Decode | 不模拟 | TTFT 在最后一层 Prefill 计算完成时结束 |
| ASU 后链路 | 0 延迟 | SSD 读完后视为瞬间到达客户端内存/HBM |
| UB | 不模拟 | 本实验不关心 UB 协议和硬件链路 |

GPU 和 NPU 在本文的语义相同：都表示一个独立的推理 worker。
代码为了兼容 QoS 项目的现有命名，使用 `gpu_id` 和 `P{gpu_id}`。

上表保留完整 GLM-5.1 workload 的模型边界。当前对比实验的
trace bundle 只生成并回放前 4 层：单次扫描使用 128 GPU 和
1～10 SSD，稳态扫描使用 64 GPU 和 1～5 SSD。

## 3. 明确不做的事

下列内容不在当前仿真边界内：

- 不启动 vLLM，不加载 GLM-5.1 权重。
- 不依赖真实 NPU、ASU、CANN 或 UB 设备。
- 不模拟 vLLM continuous batching、抢占、scheduler 抖动或 Decode。
- 不模拟 GPU HBM 内部的 KV 替换。
- 不把 `hbm_hit_ratio`、`ucm_hit_ratio`、`miss_ratio` 拆成三个独立参数。
- 不在 QoS trace 模式里再做一次 KV Placement。
- 不把 SQE 里的合成 DMA 地址真正提交给硬件。
- 不将 generator 的简化 FIFO TTFT 当成最终 QoS 结果。

当前参数已经足够生成对应的 UCM SQE：输入长度决定候选
Block 数，缓存 Prefix 比例决定 Retrieve 数量，热点成员和 50K 热点决定
跨 GPU 重复访问，UCM Ring Hash 决定 ASU，512 TFLOPS 决定计算窗口。

## 4. 两个工程的责任分工

当前方案是两阶段，而不是把 UCM 代码直接塞进 QoS 项目。

```text
/home/chguo/PycharmProjects/ucm-sqe-simulator
    生成 Prefix workload
    调用 UCM native helper
    生成原始 SQE
    记录 UCM 已决定的 ASU 路由
    写出 trace bundle
                   |
                   | 只通过文件边界连接
                   v
/home/chguo/PycharmProjects/QoS分析
    DPU/ucm_trace.py 解析 bundle
    ucm_trace_qos_simulator.py 管理 Layerwise 闭环
    DPU 做聚合带宽需求和策略控制
    QoS 做 Queue/CIR/PIR/WRR 调度
    SSD backend 产生 completion
    输出 effective manifest 和 TTFT summary
```

### 4.1 `ucm-sqe-simulator` 负责什么

它负责在 QoS 仿真开始前固化以下事实：

- 每张 GPU 的输入 token 数。
- 每张 GPU 的缓存 Prefix 比例和完整 Block 数。
- 哪 77 张 GPU 是热点成员。
- 热点成员共享的前 390 个 BlockId。
- 每个 BlockId 通过 UCM `RING_HASH` 落到哪个 ASU。
- 每层的 offset、Entry length、key 和 SQE 切批顺序。
- 每张 GPU 的请求到达时间。
- 每张 GPU 按 512 TFLOPS 估算的单层计算时间。

它复用 UCM 的 key 压缩、Ring Hash 路由和 SQE packer，因此输出的
`raw_sqe.bin` 不是自定义的仿协议 JSON，而是 UCM packer 生成的原始字节。

### 4.2 `QoS分析` 负责什么

它不改变 UCM 已经决定的 key 和 ASU，只负责：

- 从 raw SQE 中解析 Retrieve Entry。
- 把每个 Entry 转换成一条 DPU/QoS/SSD IO。
- 把同一源请求、同一层的所有 ASU/SQE 分片合并为一个逻辑批次。
- 计算每个 ASU 路径的聚合字节数和 CIR 需求。
- 运行 baseline、CIR-only（PIR uncapped）和 Utility+EDF。
- 用 SSD completion 推进每张 GPU 的下一层。
- 计算 TTFT、SSD 压力、GPU 利用率和 storage stall。

## 5. 为什么不需要 vLLM

本实验关心的不是 vLLM scheduler 如何做 online batching，而是给定
Prefix 长度、命中比例、共享关系和 GPU 算力后，UCM 和 DPU/QoS/SSD
的结果。

`ucm-sqe-simulator` 使用 PyTorch CPU 生成可复现 token，但不加载模型权重。
一个请求需要的信息已经全部在配置里：

```text
input_tokens
+ cached_prefix_ratio
+ same-prefix membership
+ fixed hot-prefix tokens
+ arrival time
+ effective compute TFLOPS
        -> 可生成 BlockId、ASU 路由、SQE 和计算窗口
```

未来如果要研究 continuous batching 或真实 scheduler 抖动，可以替换最上层
workload/timeline，但现在不需要为生成 SQE 而引入 vLLM。

## 6. Trace bundle 是唯一工程边界

QoS trace 模式依赖同一目录中的四个必需文件：

```text
<trace_bundle_dir>/
    raw_sqe.bin
    sqe_manifest.jsonl
    metadata.json
    workload_summary.json
```

generator 还会生成 placement、pressure、hotspot、图片等分析文件，但对
QoS replay 来说，上面四个文件才是必需 bundle。

这四个文件不能只复制其中一两个。

### 6.1 `raw_sqe.bin`

该文件是 UCM `PackRequest`/packer 返回的 SQE 字节直接拼接。

它的重要特征是：

- header、Entry 布局和 pack 字节由 UCM 原生 packer 生成。
- DMA address、MR key 和 CID 是协议合法的确定性合成值，
  不对应真实硬件连接或可 DMA 的内存。
- 没有自定义帧头。
- 没有每条记录的长度前缀。
- 没有 `gpu_id`。
- 没有 `source_request_id`。
- 没有 `layer_id`。
- 没有 `timestamp_ns`。
- 没有 `target_asu_id`。
- 没有 SSD completion 时间。

特别需要注意：UCM raw SQE 本身不编码目标 ASU。真实系统是通过
“将 SQE 提交到哪个 ASU connection”表示目标。离线 trace 必须在 manifest
里保存这个带外信息。

### 6.2 `sqe_manifest.jsonl`

manifest 每行描述一条 SQE，提供 raw 文件中没有的边界和上下文。

关键字段包括：

| 字段 | 用途 |
|---|---|
| `record_index` | 原 trace 中的稳定顺序 |
| `sqe_uid` | 一条 SQE 的唯一标识 |
| `opcode` | `Exist` 或 `BatchRetrieve` |
| `phase` | Prefix Query 或 Layer Retrieve |
| `raw_offset` | 这条 SQE 在 `raw_sqe.bin` 中的起点 |
| `raw_length` | 这条 SQE 的字节长度 |
| `gpu_id` | 源 GPU/NPU worker |
| `source_request_id` | 源 Prefix 请求 |
| `layer_id` | Retrieve 对应的层 |
| `target_asu_id` | UCM 已选定的目标 ASU |
| `timestamp_ns` | generator 时间线中的原始参考时间 |
| `batch_number` | BatchRetrieve 中的 Entry 数 |
| `payload_bytes` | 该 SQE 内所有 Entry 的数据字节总和 |

读取第 N 条 SQE 时，必须使用：

```text
raw_sqe = raw_sqe.bin[raw_offset : raw_offset + raw_length]
```

### 6.3 `metadata.json`

metadata 固化这份 trace 是如何生成的，包括：

- GLM-5.1 模型几何。
- 78 层、128 token/block、TP=1。
- UCM helper 的 commit、ABI 和字节序。
- workload、arrival、router 和 pressure 配置。
- 每 token/每 layer KV 字节数。
- 单 Block/单 layer 的 KV 字节数。
- Query/Retrieve SQE 和 Entry 的计数。
- 是否完成生成，以及输出契约。

QoS parser 使用它确认小端 SQE，实验对齐时使用它确认 UCM
commit 和 KV 几何没有变化。

### 6.4 `workload_summary.json`

该文件保存每张 GPU 的上层请求语义。

对每个 `source_request_id`，QoS 闭环使用：

- `gpu_id`
- `arrival_time_ns`
- `input_tokens`
- `sampled_cached_prefix_ratio`
- `cached_block_count`
- `cached_token_count`
- `recompute_token_count`
- `single_layer_compute_ns`

其中 `arrival_time_ns` 是 Layer 0 真实可以开始提交的时间，
`single_layer_compute_ns` 是闭环递推的计算窗口。

### 6.5 为什么四个文件缺一不可

| 缺少的文件 | 丢失的能力 |
|---|---|
| `raw_sqe.bin` | 无法使用 UCM 真实字段和 Entry 顺序 |
| `sqe_manifest.jsonl` | 无法切分 raw，也不知道 GPU、层、ASU 和原时间 |
| `metadata.json` | 无法核对模型几何、字节序、UCM commit 和完整性 |
| `workload_summary.json` | 无法得到 GPU arrival 和单层计算时间，因而无法闭环 TTFT |

所以，仅有 `raw_sqe.bin` 和 `sqe_manifest.jsonl` 也不足以完成 TTFT 闭环。

## 7. SQE 和 IO 的转换语义

### 7.1 Exist

generator 会把 Prefix Query 的 Exist SQE 写入 raw 和 manifest。

QoS trace replay 里的处理是：

```text
opcode == Exist
    -> 跳过
    -> 不调用 DPU
    -> 不进入 QoS Queue
    -> 不产生 SSD IO
    -> 不计入 TTFT 存储等待
```

这是当前实验假设，不是说真实 Exist 永远没有开销。

### 7.2 BatchRetrieve

`DPU/ucm_trace.py` 按 UCM 小端格式解析 BatchRetrieve：

- 校验 opcode `0x46`。
- 解析 64-byte SQE header。
- 校验 `batch_number` 为 1～110。
- 按 36 byte 解析每个 Entry。
- 保留 offset、ASU key、buffer address、MR key 和 length。
- 校验 manifest 与 raw 的 Entry 数、descriptor 长度和 payload 字节数一致。

对当前 GLM-5.1/TP=1/BF16/128-token block：

```text
KV bytes per token per layer
    = (kv_lora_rank + qk_rope_head_dim) * 2
    = (512 + 64) * 2
    = 1,152 bytes

one Retrieve Entry
    = 128 * 1,152
    = 147,456 bytes
    = one QoS/SSD IO
```

适配器不把一条 BatchRetrieve SQE 当成一条大 IO。
它把 SQE 内的每个 Entry 展开成一条 IO。

因此：

```text
1 BatchRetrieve SQE with N Entries
    -> N DPU basic requests
    -> N QoS requests
    -> N SSD completions
```

实现不硬编码 IO 长度，而是读 Entry 里的 `length_bytes`。
在当前配置下，该值应为 147,456 byte。

### 7.3 多 SQE 分片不改变逻辑层边界

一张 GPU 的一层可以访问多个 ASU。
同一个 ASU 上的 Entry 数也可以超过 110，因此 UCM 会切成多条 SQE。

这些 SQE 分片仍然属于同一个逻辑层：

```text
logical layer key = (source_request_id, layer_id)
```

trace adapter 必须先收集该 key 下所有 ASU 和所有 SQE 分片，
再且只能调用一次：

```python
dpu.submit_batch(requests=whole_layer_entries, arrival_time_us=issue_time_us)
```

以下做法是错误的：

- 每个 SQE 调用一次 `submit_batch()`。
- 每个 ASU 调用一次 `submit_batch()`。
- 超过 110 Entry 切批后将一层变成多个 Demand。

这些错误会改变 DPU 的聚合带宽需求，也会改变 CIR-only 和
Utility+EDF 的策略结果。

## 8. `DPU/ucm_trace.py` 的责任

该模块是四文件 bundle 到现有 DPU 输入格式的边界。

它完成三层工作：

1. `parse_batch_retrieve(raw_sqe)` 解析一条 UCM raw SQE。
2. `UcmTraceBundle` 加载四个必需文件，建立 workload 查询。
3. `iter_layer_submissions()` 按完整逻辑层输出 `UcmLayerSubmission`。

`UcmLayerSubmission` 中有：

- 源请求、GPU、层和原始时间。
- 单层计算窗口。
- 整层总字节数。
- 每个 `SSD{asu_id}` 的聚合字节数。
- 已展开为 Entry 粒度的 DPU requests。

每个 Entry 的稳定 ID 为：

```text
{sqe_uid}:entry:{entry_index:03d}
```

这个 ID 使 SSD completion 能够回到正确的源请求、逻辑层和 SQE。

## 9. DPU 输入格式

每个 Retrieve Entry 会生成一个 request：

```python
{
    "basic": {
        "request_id": "<sqe_uid>:entry:000",
        "p_node_id": "P<gpu_id>",
        "storage_target_id": "SSD<target_asu_id>",
        "size_bytes": 147456,
    },
    "demand_bw": {
        "demand_group_id": "<source_request_id>:layer:<layer_id>",
        "compute_layer_index": None,   # Layer 0；后续层为 layer_id - 1
        "prefetch_layer_index": 0,
        "inference_arrival_time_us": 0.0,
        "service_window_us": 0.0,
        "deadline_us": 0.0,
        "batch_total_bytes": 0,
        "aggregate_bytes_on_storage_target": 0,
        "aggregate_required_bytes_per_second": 0,
    },
}
```

`basic` 是真正传给 QoS 的普通 IO 字段。
`demand_bw` 是 DPU 为一个逻辑层做聚合控制所用的带宽元数据。

同一层所有 Entry 都重复携带相同的 demand group 信息，但
`DPURequestGateway` 在每个 storage path 上只登记一次聚合 Demand。

## 10. 每个 ASU 的聚合带宽

对一个逻辑层，定义：

```text
path_bytes[a]
    = 该层中 target ASU = a 的所有 Entry.length_bytes 之和

batch_total_bytes
    = sum(path_bytes[a] for every ASU a)

service_window_ns
    = 该 GPU 的 single_layer_compute_ns

requested_CIR[a]
    = ceil(path_bytes[a] * 1,000,000,000 / service_window_ns)
```

例如，某 GPU 的某层在 SSD0 上有 100 个 Entry，每个 147,456 byte，
该 GPU 单层计算窗口为 1 ms，则：

```text
path_bytes[SSD0] = 100 * 147,456 = 14,745,600 bytes
requested_CIR[SSD0] = 14,745,600,000 bytes/s
```

此聚合不改变 IO 粒度。SSD 仍然收到 100 条 147,456-byte IO，
DPU 只是用 14,745,600 byte 和计算窗口表达这个路径的带宽需求。

这个聚合逻辑以前与 `KVPlacementManager` 绑在一起。
trace 模式不需要它的放置决策，因此聚合被移到 trace adapter。

## 11. Trace 模式绕过哪些旧模块

QoS 项目原来的 synthetic 路径大致是：

```text
llm_workload
    -> 生成 synthetic layer request
    -> KVPlacementManager 选 SSD
    -> DPU
    -> QoS
    -> SSD
```

UCM trace 路径是：

```text
four-file UCM trace bundle
    -> DPU/ucm_trace.py
    -> UcmTraceQosSimulation closed-loop coordinator
    -> DPU
    -> QoS
    -> SSD
```

在 trace 模式中，下列部分完全绕过：

- synthetic `llm_workload` 的请求生成。
- synthetic 命中率采样。
- `KVPlacementManager` 的 storage target 选择。
- 旧的随机 Block 放置。
- 只运行 0～3 层的 synthetic workload 配置。

这些模块的代码仍然保留，以免破坏原有 synthetic 实验，但它们不参与
`ucm_trace_qos_simulator.py` 的数据路径。

## 12. 为什么仍然需要 Layerwise coordinator

绕过 `llm_workload` 不等于可以丢掉 GPU 逐层状态。

静态 raw trace 只能告诉我们“某个逻辑层有哪些 SQE”，不能预知
该层在某个 QoS 策略下何时读完。

而下一层何时发出，又依赖：

- 当前层的所有 SSD IO 是否完成。
- 上一层 GPU 计算是否结束。

因此 `ucm_trace_qos_simulator.py` 中保留了一个独立的闭环 coordinator。
它不是 LLM 模型，也不是 vLLM；它只保存每张 GPU 的最小状态：

- 请求到达时间。
- 单层计算时间。
- 当前正在等待的层。
- 该层还有多少 Entry 未完成。
- 上一层计算完成时间。
- 最后一层计算完成时间。

## 13. Layerwise TTFT 闭环递推

对 GPU `g` 和层 `L`，定义：

```text
A[g]       请求到达时间
I[g,L]     该层 Retrieve 实际发出时间
D[g,L]     该层最后一个 Entry 的 SSD completion 时间
S[g,L]     该层 GPU 计算开始时间
F[g,L]     该层 GPU 计算完成时间
C[g]       该 GPU 的 single_layer_compute_ns
```

Layer 0 在 GPU 请求到达时发出：

```text
I[g,0] = A[g]
```

Layer 0 的所有 Entry 完成后，才能开始计算：

```text
S[g,0] = D[g,0]
F[g,0] = S[g,0] + C[g]
```

开始计算 Layer 0 的同一时刻，预取 Layer 1：

```text
I[g,1] = S[g,0]
```

对 `L >= 1`：

```text
S[g,L] = max(F[g,L-1], D[g,L])
F[g,L] = S[g,L] + C[g]
I[g,L+1] = S[g,L]       # 如果 L+1 存在
```

最终：

```text
TTFT[g] = F[g,77] - A[g]
```

这个递推同时表达了两种重叠：

- Layer L 计算可以与 Layer L+1 读取重叠。
- 如果 Layer L+1 在计算结束前已读完，GPU 无需等待存储。
- 如果 Layer L+1 读取较慢，超出的部分成为 storage stall。

## 14. 原 manifest 时间戳的真正用途

generator 在 `sqe_manifest.jsonl` 里写了 `timestamp_ns`。
这个时间戳对 generator 自己的时间线和 offered-load 图有用。

但对 QoS 闭环，后续层的该时间只是 reference，不是实际下发时间。

原因是：

```text
不同 QoS 策略
    -> 不同 Queue 等待
    -> 不同 SSD completion
    -> 不同的下一层 issue time
    -> 不同 TTFT
```

所以 QoS replay 使用如下时间规则：

- Layer 0 使用 `workload_summary.requests[].arrival_time_ns`。
- Layer 1～77 不按原 manifest `timestamp_ns` 直接排队。
- 后续层的 issue time 由上一节的 completion 闭环递推得到。
- 原 `timestamp_ns` 保留在结果中，只用于对照和排障。
- EDF deadline 也按当次实际 issue time 重算，不使用已过期的原时间。

## 15. Effective manifest

每个策略会单独输出：

```text
<output_dir>/<policy>/effective_sqe_manifest.jsonl
```

该文件每行对应一条 Retrieve SQE，字段包括：

| 字段 | 含义 |
|---|---|
| `effective_record_index` | 按层完成后写入文件的输出顺序 |
| `effective_issue_sequence` | 本策略下 SQE 实际发出的全局稳定顺序 |
| `inference_index` | 稳态回放中该 GPU 的第几次推理，从 0 开始 |
| `original_record_index` | 原 manifest 记录序号 |
| `sqe_uid` | 本次回放的运行时 SQE 标识 |
| `template_sqe_uid` | raw bundle 中的原 SQE 标识 |
| `source_request_id` | 本次回放的运行时源请求 |
| `template_source_request_id` | bundle 中的原源请求 |
| `gpu_id` | 源 GPU |
| `layer_id` | 逻辑层 |
| `target_asu_id` | UCM 原定的目标 ASU |
| `batch_number` | 该 SQE 内 Entry 数 |
| `payload_bytes` | 该 SQE 总 payload |
| `raw_offset` / `raw_length` | 还能定位到原 raw SQE |
| `original_timestamp_ns` | generator 给出的参考时间 |
| `effective_issue_time_ns` | QoS 闭环重算的实际下发时间 |
| `effective_completion_time_ns` | 该 SQE 最后一个 Entry 的 SSD completion 时间 |

同一条 SQE 内不同 Entry 可能在不同时刻完成。
effective manifest 中的 completion 是该 SQE 内最后一个 Entry 的完成时间。
因为文件在层完成后才写入，文件行顺序不一定就是发出顺序。
需要比较同时刻多张 GPU 的确定性全序时，使用
`effective_issue_sequence`。

对某一层而言，层完成时间还要再取该层所有 SQE/ASU 的最大值。

QoS 闭环不修改 raw SQE 里由 generator 按 UCM stream 规则分配的
合成 CID和 ASU 局部顺序。单次模式保留原 `sqe_uid`；稳态模式
复用同一份 raw 模板，但为每次推理生成唯一的运行时
source/request/demand/SQE ID，并用 `template_sqe_uid` 回指原 raw 记录。
`effective_issue_sequence` 只表示当前 QoS 策略改变时序后的实际回放顺序。

## 16. 三种对比策略

单次 128 GPU 扫描运行 `baseline` 和 `cir_only`；稳态 64 GPU
扫描再加入 `utility_edf_integer_l750`。

### 16.1 `baseline`

- 不创建 DPU rate controller。
- Queue CIR 默认为 0。
- Queue PIR 为 `None`/`uncapped`。
- IO 可以通过 EXCESS 轮下发。
- 用于观察不做 Demand-aware 准入时的基准。

### 16.2 `cir_only`

- 实验名 `cir_only` 映射到 `DemandAwareFCFSCIRController`。
- 按 Demand 到达顺序分配每块 SSD 的 CIR。
- 先到 Demand 先获得带宽保证。
- DPU 只动态分配 CIR，不给 Queue 设置有限 PIR。
- Queue PIR 始终为 `None`/`uncapped`。
- 没有获得 CIR 的 Queue 仍可能走 EXCESS。
- 因此这里的“CIR-only”只表示控制器动态分配 CIR；Queue
  PIR 保持 uncapped，EXCESS 允许下发，它不构成硬准入门控。

### 16.3 `utility_edf_integer_l750`

- 使用整数 Utility score 和 EDF。
- `l750` 表示 deadline allowance 为 750 us。
- 对已进入计算—预取流水线的 Demand 考虑 deadline。
- 对未开始 GPU 考虑 Utility。
- 多 ASU 逻辑层仍以同一 `demand_group_id` 做 coflow 聚合。

同一拓扑的对比策略必须使用同一份 trace bundle。
每种策略创建一套全新 EventLoop、DPU、QoS、SSD 和 GPU 状态，
不能在策略之间复用运行中的 Queue/token 状态。

## 17. ASU、SSD 和 Queue 映射

当前拓扑定义：

```text
target_asu_id = a
    -> storage_target_id = SSDa

gpu_id = g
    -> p_node_id = Pg
```

每个 ASU 创建独立的：

- QoS scheduler。
- Queue CIR token bucket；PIR uncapped 时 Queue 不创建 PIR bucket。
- SSD backend。
- completion 路径。

Group CIR 只是给组内 Queue 分配 CIR 的配置预算。Group 没有运行时
PIR bucket，也不用 Group token bucket 重复做 IO 准入；物理出口上限
继续由 SSD backend 保证。

单次 128 GPU 扫描使用 `balanced_exclusive` 绑定。每个
`(GPU, SSD)` 在实验开始前固定到一条专用 Queue。每块 SSD
仍为 256 条 Queue、8 个 Group、每组 32 条 Queue。

### 17.1 64 GPU 稳态拓扑

稳态实验使用 `one_group_per_gpu` 固定绑定。每块 SSD 上的
256 条 Queue 会在该仿真实例的深拷贝配置中动态重排为：

```text
64 Group × 4 Queue/Group = 256 Queue
GPU i -> P{i} -> q{i*4} -> g{i}, 0 <= i < 64
```

`group_rates` 展开为 `g0`～`g63`，Queue CIR 权重为 4 项，Group WRR
权重为 64 项，每个 Group 的 Queue WRR 权重也是 4 项。这样每张
GPU 在所有 SSD 上都固定使用自己 Group 的首 Queue，不会在不同
推理次数间重新绑定。

## 18. SSD completion 是什么层级的信号

DPU 看到 Queue-empty 不等于数据已经读完。

```text
Queue-empty
    = 该 Demand 的 IO 已经离开 QoS Queue
    != IO 已经离开 SSD 流水线
```

Layerwise coordinator 等待的是每个 Entry 的真正 SSD completion，
不是 DPU Demand release，也不是 Queue depth 变成 0。

当一层的最后一个 Entry completion 到达时，coordinator 才设置 `D[g,L]`。

当前假设 SSD completion 之后的客户端内存/HBM 传输时间为 0，
因此不再增加网络或 DMA completion 事件。

## 19. 内存和流式边界

默认完整 trace 包含数百 MB raw/manifest 和数百万 Retrieve Entry。
代码不能把所有 Entry 一次性转成 Python 对象。

当前实现的内存边界是：

1. 启动时只扫描 manifest，记录每个 `(source_request_id, layer_id)`
   的 JSONL 字节起止位置。
2. 只保存每个源请求的层 ID 列表和 ASU 集合。
3. 当某张 GPU 真正需要发出某层时，才 seek manifest 和 raw。
4. 只解析这一个逻辑层的 SQE 和 Entry。
5. 这一层的 Entry completion 全部到达后，立即释放 request ownership。
6. QoS 和 SSD 使用计数/字节聚合日志，不常驻保存数百万条逐 IO 日志。
7. effective manifest 边完成边顺序写盘。

因此常驻解析对象的规模与“当前正在读的层”相关，
而不是与整个 trace 中的总 Entry 数相关。

### 19.1 对 manifest 顺序的要求

同一 `(source_request_id, layer_id)` 的所有 Retrieve SQE 必须在 manifest 中连续。
该层结束后不能在后面再次出现。

这是当前 generator 已满足的输出契约，也是简单流式索引的基础。

## 20. QoS 项目不复制大 trace

`raw_sqe.bin` 和 `sqe_manifest.jsonl` 可以非常大，不应该复制到
`QoS分析` Git 工作树，也不应该提交到 QoS Git 仓库。

QoS 通过项目唯一配置文件 `config/simulation_config.yaml` 中的
bundle root 和目录 pattern 直接指向 generator 输出：

```yaml
simulation:
  ucm_trace:
    trace_bundle_root: ../ucm-sqe-simulator/outputs/glm51_128npu_4layer_asu_1_10
    trace_bundle_pattern: gpu_128_asu_{ssd_count}
    output_dir: experiments/results/ucm_trace_4layer_ssd_sweep
    policies:
      - baseline
      - cir_only

  ucm_trace_steady:
    trace_bundle_root: ../ucm-sqe-simulator/outputs/glm51_64npu_4layer_asu_1_5
    trace_bundle_pattern: gpu_64_asu_{ssd_count}
    output_dir: experiments/results/ucm_trace_steady_4layer_ssd_sweep
    ssd_counts: [1, 2, 3, 4, 5]
    inference_count_per_gpu: 5
    warmup_inference_count: 0
    stop_mode: first_gpu_reaches_limit
    parallel_workers: 15
    queue_binding_strategy: one_group_per_gpu
    policies:
      - baseline
      - cir_only
      - utility_edf_integer_l750
```

每个 pattern 展开后的子目录就是一个 trace-dir 边界。
路径相对 QoS 项目根目录解析，也可以写绝对路径。
trace 模式与 synthetic 模式共用这一份 YAML 中的 DPU/QoS/SSD 硬件参数，
不再维护第二份 trace YAML。

## 21. 生成 trace bundle

### 21.1 安装和构建 UCM helper

```bash
cd /home/chguo/PycharmProjects/ucm-sqe-simulator
python -m pip install -e '.[test]'
ucm-sqe-sim build-helper --ucm-source /home/chguo/work/UCM
```

helper 会检查 UCM checkout 是否等于配置中锁定的 commit：

```text
e55ddc0ab30770e757fd15c4335dd296db72d11b
```

先做配置校验：

```bash
ucm-sqe-sim run \
  -c configs/glm51_prefix.yaml \
  --validate-only
```

### 21.2 生成当前 128 NPU / 10 ASU trace

```bash
cd /home/chguo/PycharmProjects/ucm-sqe-simulator

ucm-sqe-sim run \
  -c configs/glm51_prefix.yaml \
  -o outputs/glm51_128npu_10asu_100k_200k_hit60_99_hot50k_arrival100ms \
  --no-plots
```

需要 generator 的 offered-load 图时，去掉 `--no-plots`，或在生成后单独执行：

```bash
ucm-sqe-sim plot \
  -i outputs/glm51_128npu_10asu_100k_200k_hit60_99_hot50k_arrival100ms
```

### 21.3 先做小规模 smoke trace

```bash
ucm-sqe-sim run \
  -c configs/glm51_prefix.yaml \
  -o outputs/smoke_2npu_2asu \
  --npu-count 2 \
  --asu-count 2 \
  --no-plots
```

这个命令只覆盖 NPU/ASU 数，其他 Prefix 参数继续使用 YAML。

## 22. 运行 QoS 闭环 replay

确认 `config/simulation_config.yaml` 的 bundle root 和 pattern 指向目标
bundle，然后选择单次或稳态模式：

```bash
cd /home/chguo/PycharmProjects/QoS分析

python qos_ssd_simulator.py \
  --mode ucm-trace \
  --config config/simulation_config.yaml

python qos_ssd_simulator.py \
  --mode ucm-trace-steady \
  --config config/simulation_config.yaml
```

`qos_ssd_simulator.py` 是项目唯一 production main；`--mode synthetic` 运行原有
synthetic workload，`--mode ucm-trace` 运行单次 UCM trace 扫描，
`--mode ucm-trace-steady` 运行稳态扫描。
`ucm_trace_qos_simulator.py` 只是被主入口延迟导入的实现模块，不是第二个 CLI。
trace 目录通过唯一 YAML 设置，不需要复制文件。

程序依次打印：

```text
START UCM trace policy=<policy>
DONE UCM trace policy=<policy> mean_ttft_us=<value>
```

单次模式串行回放各个 point。稳态模式有 5 个 SSD 数量和
3 种策略，共 15 个 `(ssd_count, policy)` point。父进程使用
15 个 worker 进程并行运行这些 point；每个 worker 内部的 EventLoop
仍为单线程，并且只写自己的 topology/policy 目录。父进程独占写
总 `summary.json`、CSV 和图片，所以跨 point 并行不改变单点仿真语义。

### 22.1 稳态回放和统计口径

每块 SSD 数量使用对应的 64 GPU、4 层 bundle 作为模板。每张 GPU
最多连续回放 5 次：

1. 第 0 次推理的 Layer 0 使用 bundle 中该 GPU 的原 arrival。
2. 一次推理的最终层计算完成时，同一 GPU 立即发出下一次
   推理的 Layer 0。
3. 同一 GPU 的 Queue 和 Group 绑定在全部推理中保持不变。
4. 没有 warmup，停止前已完整完成的每次推理都进入指标。

`stop_mode: first_gpu_reaches_limit` 表示任意第一张 GPU 完成第 5 次
推理的 completion 事件后，EventLoop 立即停止。它不继续排空其他
GPU、QoS Queue 或 SSD 流水线；因此其他 GPU 可能只完成 0～4 次，
也可能保留正在读取或计算的推理。

若多张 GPU 的第 5 次 completion 具有相同时间戳，EventLoop 仍按
`(time, priority, stable sequence)` 逐个处理事件，并在每个事件后检查
停止条件。因此稳定顺序中第一个被处理的 GPU 是 winner，同时戳的
后续 completion 仍留在未处理事件中，不计为已完成。

稳态 Utility+EDF 不在一次推理的最终层后立即恢复 uncapped 路径。
该 GPU 的路径保持 parked，新 Layer 0 登记后由下一个 80 us 控制周期
重新选择，避免跨推理边界偷跑。

主曲线使用全 GPU observation-window 利用率。对 GPU `g`，观察窗口从
该 GPU 首次 arrival `a_g` 开始，到全局早停时刻 `T_stop` 结束。
分子包含该 GPU 已完成推理的所有层计算区间，以及当前未完成
推理中已经开始的计算部分；每个区间都裁剪到 `[a_g, T_stop]`：

```text
observation_busy_g = sum(clipped compute intervals of GPU g)
observation_window_g = T_stop - a_g

observation_utilization_g = observation_busy_g / observation_window_g * 100

mean_observation_window_gpu_utilization_percent
    = sum(observation_utilization_g for all observed GPUs)
      / observation_window_gpu_count
```

当 winner 的第 5 次推理已在停止事件中完成时，其当前 layer list
不会再重复计入分子。正式扫描的 64 张 GPU 都在早停前到达，
因此每个 point 的 `observation_window_gpu_count` 都是 64。

completed-only 利用率仍保留为诊断指标。它对每张至少完整完成
一次推理的 GPU，只聚合停止前的完整推理：

```text
completed_g = 停止前 GPU g 已完整完成的推理集合

U_g_percent = sum(compute_time[g, i] for i in completed_g)
              / sum(completion_time[g, i] - arrival_time[g, i]
                    for i in completed_g)
              * 100

measured_gpus = {g | len(completed_g) > 0}
mean_gpu_utilization_percent = (
    sum(U_g_percent for g in measured_gpus) / len(measured_gpus)
)
```

停止时未完成推理的部分 IO 和计算不进入 TTFT 或 completed-only
GPU 利用率。
没有完成任何推理的 GPU 将 `gpu_utilization_percent` 记为 `null`，
不进入 completed-only 平均值或最小值。因此该诊断指标的样本集
会随策略改变，有选择偏差，不用作正式曲线和主结论。

这一停止方式输出 partial snapshot，不要求满足全量 trace 守恒，也不要求
rate controller 的 active Demand 为 0。定义：

```text
S = 已提交到 DPU 的 Entry/字节
Q = 已由 QoS 接收并下发到 SSD 的 Entry/字节
C = 已收到 SSD completion 的 Entry/字节

0 <= C <= Q <= S
before_qos = S - Q
in_ssd = Q - C
total_outstanding = S - C
```

`len(request_owner)` 和所有 active layer 的 pending Entry 总数都必须等于
`S-C`。`submitted_sqe_count` 记录已发出 SQE；
`completed_layer_sqe_count` 只记录所在逻辑层已整层读完并写入 effective
manifest 的 SQE。每个 SSD 使用全局 stop time 作为吞吐区间终点，
分别输出 QoS-dispatched 和 SSD-completed 吞吐；若还没有 completion，
completed 吞吐为 0。

### 22.2 正式 5 次早停结果

下表是 64 GPU、4 层模板、每 GPU 最多 5 次推理、无 warmup，
任意首张 GPU 完成第 5 次后立即全局停止的正式结果。数值为
`mean_observation_window_gpu_utilization_percent`，单位是 `%`：

| SSD 数 | Baseline | FCFS CIR-only (PIR uncapped) | Utility+EDF (`L=750`) |
|---:|---:|---:|---:|
| 1 | 31.1006283 | 26.1915348 | 39.4487903 |
| 2 | 55.6788030 | 44.6272111 | 61.1034279 |
| 3 | 64.5829063 | 51.9562727 | 70.0875163 |
| 4 | 80.7867787 | 62.4122058 | 86.9093850 |
| 5 | 85.7772670 | 68.2591662 | 91.6829370 |

主指标在每个 point 都统计全部 64 张 GPU。原 completed-only 指标
另存为 CSV，由于不同策略在早停时拥有完整推理样本的 GPU 数不同，
它存在选择偏差，不作为主结论。

本次 Utility+EDF 在 1～5 SSD 的每个 point 都是三种策略中最好的。
FCFS CIR-only 在每个 point 都低于 baseline；这只说明当前 FCFS
CIR-only 策略在该 workload 和早停口径下效果较差，不等于
所有 demand-aware 策略都会变差。

正式产物：

- [主指标 CSV](../experiments/results/ucm_trace_steady_4layer_ssd_sweep/steady_gpu_utilization_vs_ssd_count.csv)
- [主曲线 PNG](../experiments/results/ucm_trace_steady_4layer_ssd_sweep/steady_gpu_utilization_vs_ssd_count.png)
- [主曲线 SVG](../experiments/results/ucm_trace_steady_4layer_ssd_sweep/steady_gpu_utilization_vs_ssd_count.svg)
- [总 summary](../experiments/results/ucm_trace_steady_4layer_ssd_sweep/summary.json)
- [completed-only 诊断 CSV](../experiments/results/ucm_trace_steady_4layer_ssd_sweep/steady_completed_only_gpu_utilization_vs_ssd_count.csv)

## 23. QoS replay 输出

默认输出目录：

```text
experiments/results/ucm_trace_4layer_ssd_sweep/
    summary.json
    gpu_utilization_vs_ssd_count.{csv,png,svg}
    <1..10>_ssd/
        baseline/{summary.json,effective_sqe_manifest.jsonl}
        cir_only/{summary.json,effective_sqe_manifest.jsonl}

experiments/results/ucm_trace_steady_4layer_ssd_sweep/
    summary.json
    steady_gpu_utilization_vs_ssd_count.{csv,png,svg}
    steady_completed_only_gpu_utilization_vs_ssd_count.csv
    <1..5>_ssd/
        baseline/{summary.json,effective_sqe_manifest.jsonl}
        cir_only/{summary.json,effective_sqe_manifest.jsonl}
        utility_edf_integer_l750/{summary.json,effective_sqe_manifest.jsonl}
```

每种策略的 `summary.json` 包含：

- `mean_ttft_us`、`p95_ttft_us`、`max_ttft_us`。
- 平均/最低 GPU 利用率。
- 每张 GPU 的 arrival、first-token time、TTFT、compute-only TTFT。
- 每张 GPU 的 storage stall。
- 每张 GPU 每层的 issue、load completion、compute start 和 compute done。
- 每个 SSD 的 IO 数、字节数、CIR/EXCESS dispatch 数。
- 每个 SSD 的首次提交、最后 completion、活跃时间和平均带宽。
- DPU rate-control 决策和完成 Demand 统计。
- trace→QoS→SSD 的全量守恒或 partial snapshot 检查。
- EventLoop 完成时间、处理事件数和实际运行时间。

稳态策略的 `summary.json` 还保存每张 GPU 在 stop time 前已完成的
0～5 条逐次推理记录、未完成推理快照、winner、待处理事件数、
completed-only GPU 利用率，以及 submitted/QoS-dispatched/SSD-completed/
outstanding 计数。这些字段是停止时快照，不暗示所有 GPU 或 IO 已排空。

generator 的 `asu_offered_bandwidth_*` 图表示原始 issue time 上的 offered load，
不是 QoS 策略下的实际 SSD 完成带宽。两者不能混为同一个指标。

## 24. 当前实现文件对照

### 24.1 `ucm-sqe-simulator`

| 文件 | 责任 |
|---|---|
| `configs/glm51_prefix.yaml` | 128/10、50K 热点、arrival、512 TFLOPS 等默认实验参数 |
| `ucm_sqe_sim/workload.py` | 生成每 GPU 输入、Prefix 命中、热点成员和 arrival |
| `ucm_sqe_sim/model.py` | GLM-5.1 KV 字节几何和单层计算时间 |
| `ucm_sqe_sim/keygen.py` | UCM/vLLM connector 同形 BlockId |
| `ucm_sqe_sim/planner.py` | 放置、Retrieve 任务和 SQE 切批 |
| `ucm_sqe_sim/ucm_core/` | 调用 native UCM helper |
| `ucm_sqe_sim/trace.py` | 流式写 raw SQE 和 manifest |
| `ucm_sqe_sim/simulator.py` | 组合 workload、UCM、trace 和 generator 侧分析 |

### 24.2 `QoS分析`

| 文件 | 责任 |
|---|---|
| `DPU/ucm_trace.py` | 解析 raw+manifest，跳过 Exist，建立整层 DPU submission |
| `qos_ssd_simulator.py` | 项目唯一 production main，通过 `--mode ucm-trace` 或 `ucm-trace-steady` 进入 trace 回放 |
| `ucm_trace_qos_simulator.py` | 无独立 main 的实现模块：流式层索引、Layerwise/稳态闭环、跨 point 并行、策略运行和 effective manifest |
| `simulation_common/aggregate_logs.py` | 只聚合 IO 计数和字节，避免常驻保存数百万条日志 |
| `DPU/dispatcher.py` | 整层 `submit_batch()`、Queue 绑定和 Demand 登记 |
| `DPU/rate_controller.py` | FCFS CIR（Queue PIR uncapped）和 Utility+EDF |
| `qos/` | Queue、token bucket、CIR/PIR/WRR 和调度 |
| `backends/asu_ssd/` | 每个 ASU 对应的 SSD 流水线和 completion |
| `config/simulation_config.yaml` | 项目唯一 YAML；同时保存 trace 路径/策略与 DPU/QoS/SSD 硬件参数 |

## 25. 测试与验收命令

### 25.1 generator 单元测试

```bash
cd /home/chguo/PycharmProjects/ucm-sqe-simulator
python -m pytest -q
```

重点覆盖：

- 128 × 0.60 得到 77 个热点成员。
- 50,000 token 取完整 block 后得到 390 blocks/49,920 token。
- 每 GPU 输入长度在 [100K, 200K] 内，并且固定 seed 可复现。
- Prefix 比例在 [0.60, 0.99] 内。
- arrival 在 [0, 100,000,000] ns 内，并且与其他随机流独立。
- raw SQE 和 manifest 边界一致。
- Retrieve Entry 数等于 workload 估算。

### 25.2 QoS trace adapter 测试

```bash
cd /home/chguo/PycharmProjects/QoS分析
python -m pytest -q \
  tests/test_ucm_trace_adapter.py \
  tests/test_ucm_trace_qos_simulator.py
```

重点覆盖：

- 合法 BatchRetrieve 的 header/Entry 解析。
- Exist 被忽略。
- 同一 ASU 上的多个 SQE 分片能正确聚合。
- 多 ASU 仍只形成一个逻辑层 submission。
- CIR 按每 ASU 聚合字节数和计算窗口正确计算。
- manifest/raw payload 不一致会被发现。

### 25.3 DPU 策略测试

```bash
python -m pytest -q \
  tests/test_dpu_qos_rate_control.py \
  tests/test_fcfs_cir_experiment.py \
  tests/test_coflow_priority_controller.py \
  tests/test_utility_edf_controller.py
```

重点覆盖：

- 原 FCFS CIR 行为不变。
- `cir_only` 只分配 CIR，Queue PIR 保持 `None`/`uncapped`。
- 0-CIR Queue 仍可通过 EXCESS 前进。
- Group 没有运行时 PIR bucket。
- 旧 Demand 完成后释放容量，后续 Demand 能够前进。
- Utility+EDF 的 coflow、deadline 和重复推理语义不被 trace 接入破坏。
- 64 GPU 时 `q{i*4}` 属于 `g{i}`，且绑定在全部推理中不变。
- 2 GPU × 2 次推理 × 2 层的小型闭环能守恒 Entry/SQE/层数。
- 异步 2 GPU 测试在首 GPU 完成第 5 次时立即停止，另一张
  GPU 少于 5 次且可保留 inflight 推理。
- SSD 已接收 IO 但还没有 completion 时，partial throughput 仍可输出。

### 25.4 全量回归

```bash
python -m pytest -q
git diff --check
```

trace 模式是新入口，原 `qos_ssd_simulator.py` 的 synthetic 实验应继续通过。

## 26. 必须满足的验收不变量

### 26.1 Bundle 完整性

- 四个必需文件全部存在。
- `metadata.status == "completed"`。
- UCM helper commit 与锁定 commit 一致。
- `raw_offset` 和 `raw_length` 能覆盖每条 SQE，不越界。
- 每条 BatchRetrieve 的 raw/manifest `batch_number` 一致。
- 每条 BatchRetrieve 的 raw/manifest `payload_bytes` 一致。
- generator 的 Retrieve Entry 总数等于 workload 估算数。

### 26.2 工作负载不变量

- GLM-5.1 完整模型参数仍记录 78 层，当前 bundle 的
  `trace_layer_count` 为 4。
- 单次扫描为 128 GPU，ASU/SSD 数与当前 bundle 一致，范围为 1～10。
- 稳态扫描为 64 GPU，ASU/SSD 数与当前 bundle 一致，范围为 1～5。
- 稳态每张 GPU 最多回放 5 次，warmup 为 0。
- 任意第一张 GPU 完成第 5 次时全局立即停止。
- 128 GPU workload 的热点 GPU 数为 77。
- 热点完整 Block 数为 390。
- 实际热点 token 数为 49,920。
- 每 GPU 输入长度在 100K～200K 之间。
- 每 GPU 的缓存 Prefix 比例在 0.60～0.99 之间。
- 每 GPU arrival 在 0～0.1s 之间。
- 每个完整 Block/单层 Retrieve Entry 为 147,456 byte。

### 26.3 层边界不变量

- Exist 不进入 QoS/SSD。
- 一个 Retrieve Entry 恰好生成一条 QoS/SSD IO。
- 一层可以含多个 ASU 和多条 SQE。
- 每个 `(source_request_id, layer_id)` 恰好调用一次 `submit_batch()`。
- 层 completion 是该层所有 Entry completion 的最大值。
- 下一层只在闭环递推规定的时刻发出。
- 最后一层 GPU 计算完成后才产生 first token。

### 26.4 全量守恒与部分快照

`all_gpus_complete` 模式结束时必须满足：

```text
submitted Retrieve Entry count
    == metadata/workload expected Retrieve Entry count
    == QoS dispatched request count
    == SSD completed request count

submitted Retrieve Entry bytes
    == QoS dispatched bytes
    == SSD completed bytes
```

同时：

- completion ownership map 为空。
- 实际层数等于 GPU 数乘 `trace_layer_count` 再乘配置推理次数。
- effective Retrieve SQE 数等于 metadata Retrieve SQE 数再乘配置推理次数。
- 当前 bundle 中的所有 GPU 都完成。
- 没有 GPU 保留 active layer。
- rate controller 的 active Demand 数最终为 0。

`first_gpu_reaches_limit` 模式结束时不使用上述全量相等式。它必须满足：

- winner 完成 5 次，其他 GPU 完成数小于 5。
- Entry 和字节都满足 `0 <= SSD-completed <= QoS-dispatched <= submitted`。
- `total_outstanding == submitted - SSD-completed`。
- completion ownership 数和 active layer pending Entry 总数都等于
  outstanding Entry 数。
- 允许 GPU active layer、pending EventLoop 事件和 rate-controller active Demand/coflow。
- Utility+EDF 快照满足 `completed_layer_count + active_coflow_count == submitted_layer_count`。

### 26.5 Effective time 不变量

对每条 Retrieve SQE：

```text
effective_completion_time_ns >= effective_issue_time_ns
```

对每张 GPU：

```text
Layer 0 effective issue == request arrival
Layer L compute start >= Layer L load completion
Layer L compute start >= Layer L-1 compute done, when L >= 1
Layer L+1 effective issue == Layer L compute start
TTFT == 最后一个回放层的 compute done - request arrival
```

`all_gpus_complete` 模式的 effective manifest Retrieve SQE 数应等于
全量期望数。`first_gpu_reaches_limit` 模式只要求该数等于
`completed_layer_sqe_count`，它可以小于 `submitted_sqe_count`。
`template_sqe_uid`、`raw_offset`、`raw_length`、`target_asu_id` 必须与原 trace
一致；稳态运行时 `sqe_uid` 必须在已提交的最多 5 次推理中唯一。

### 26.6 策略不变量

- baseline 的 `rate_control` 为 `null`。
- `cir_only` 只动态分配 CIR，Queue PIR 为 `None`/`uncapped`。
- `cir_only` 允许 EXCESS dispatch，未获得 CIR 的 Queue 不因 PIR 被强制阻塞。
- Group 没有运行时 PIR bucket。
- CIR-only 中已分配的 Queue CIR 不超过 SSD 容量总和。
- Utility+EDF 使用当前闭环 issue time 重算 deadline。
- 同一拓扑下各策略复用同一份原始 trace 模板和单次推理输入定义。
- 由于各策略的全局早停时刻不同，停止快照中实际 submitted Entry
  和字节数允许不同。
