#!/usr/bin/env python3
"""生成GLM-5.1的层读取计划，并计算SSD完成对TTFT的影响。

本文件只保存LLM能够理解的信息：模型层、KV Block、GPU计算窗口和逐层完成状态。
它不再生成DPU请求，也不知道Block位于哪个SSD；这些工作交给KV Placement Manager。

模型、GPU、KV Cache和默认工作负载分别放在
``config/layer_request.yaml`` 的四个分区中。
"""

from copy import deepcopy
import math
from pathlib import Path

from simulation_common.config_utils import load_yaml


# ------------------------------ YAML配置加载 ------------------------------

CONFIG_DIR = Path(__file__).resolve().parent / "config"
LAYER_REQUEST_CONFIG_FILE = CONFIG_DIR / "layer_request.yaml"


def load_layer_request_config(config_file=LAYER_REQUEST_CONFIG_FILE):
    """功能：读取LLM层请求YAML配置。

    目的：把模型、GPU算力、KV Block布局和默认工作负载与Python逻辑分离。

    输入：
        config_file: ``layer_request.yaml`` 文件路径。

    输出：
        dict: YAML中 ``layer_request`` 节点对应的完整配置。
    """
    return load_yaml(config_file)["layer_request"]


# 模块导入时加载一次，让现有函数默认参数和公开常量保持稳定。
_CONFIG = load_layer_request_config()
_MODEL_CONFIG = _CONFIG["model"]
_GPU_CONFIG = _CONFIG["gpu"]
_KV_CACHE_CONFIG = _CONFIG["kv_cache"]
_WORKLOAD_CONFIG = _CONFIG["workload"]

MODEL_NAME = _MODEL_CONFIG["name"]
MODEL_PROFILE = deepcopy(_MODEL_CONFIG["profile"])
MODEL_PROFILE["tokens_per_kv_block"] = _KV_CACHE_CONFIG["tokens_per_block"]
GPU_ESTIMATE = deepcopy(_GPU_CONFIG)
GPU_ESTIMATE["gpu_count"] = GPU_ESTIMATE.pop("count")
KV_CACHE_BITS_PER_ELEMENT = _KV_CACHE_CONFIG["bits_per_element"]
DEFAULT_WORKLOAD = deepcopy(_WORKLOAD_CONFIG)


def build_scenario():
    """功能：构造模型、KV粒度和GPU算力的独立场景字典。

    目的：返回值会被LLM层计划和联合仿真使用。函数每次都创建深拷贝，调用方可以为
    临时实验修改字段，而不会污染已加载的默认配置。

    输入：
        无。

    输出：
        dict: 包含 ``model_profiles``、``gpu_estimate`` 和KV精度。
    """
    return {
        "kv_cache_bits_per_element": KV_CACHE_BITS_PER_ELEMENT,
        "model_profiles": {MODEL_NAME: deepcopy(MODEL_PROFILE)},
        "gpu_estimate": deepcopy(GPU_ESTIMATE),
    }


def _kv_bytes_per_token_per_layer(scenario, model_name):
    """功能：计算一个命中Token在一个模型层中的逻辑KV字节数。

    目的：GLM-5.1当前按 ``(kv_lora_rank + qk_rope_head_dim) ×
    元素字节数`` 估算，为后续Block对齐前提供有效逻辑大小。

    输入：
        scenario: ``build_scenario`` 生成的模型和KV精度配置。
        model_name: 需要查询的模型名称。

    输出：
        float: 单Token、单层的逻辑KV字节数，未做Block对齐。
    """
    profile = scenario["model_profiles"][model_name]
    bytes_per_element = scenario["kv_cache_bits_per_element"] / 8
    return (
        profile["kv_lora_rank"] + profile["qk_rope_head_dim"]
    ) * bytes_per_element


def build_layer_plan(
    input_tokens=DEFAULT_WORKLOAD["input_tokens"],
    hit_ratio=DEFAULT_WORKLOAD["prefill_layer_hit_ratio"],
    batch_size=DEFAULT_WORKLOAD["batch_size"],
    scenario=None,
    model_name=DEFAULT_WORKLOAD["model"],
):
    """功能：计算一个模型层的KV Block读取规模和GPU重算窗口。

    目的：返回值只描述LLM内部的模型和读取规模。Block到SSD的放置、每SSD聚合
    需求以及DPU请求格式均由KV Placement Manager负责。

    输入：
        input_tokens: 单个Batch样本的输入Token数。
        hit_ratio: 每层需要从SSD读取的KV命中比例。
        batch_size: 当前推理Batch Size。
        scenario: 可选的模型与GPU场景；None时使用YAML默认值。
        model_name: 当前工作负载使用的模型名称。

    输出：
        dict: Token划分、Block大小/数量、传输字节和GPU计算时间。
    """
    scenario = build_scenario() if scenario is None else scenario
    profile = scenario["model_profiles"][model_name]
    total_input_tokens = input_tokens * batch_size
    cached_tokens = int(round(total_input_tokens * hit_ratio))
    recompute_tokens = total_input_tokens - cached_tokens
    kv_bytes_per_token = _kv_bytes_per_token_per_layer(scenario, model_name)
    useful_read_bytes = math.ceil(cached_tokens * kv_bytes_per_token)

    active_parameters_per_layer = (
        profile["active_parameters"] / profile["hidden_layers"]
    )
    layer_operations = (
        active_parameters_per_layer
        * recompute_tokens
        * scenario["gpu_estimate"]["operation_factor_per_parameter_token"]
    )
    effective_flops = (
        scenario["gpu_estimate"]["effective_compute_tflops"]
        * scenario["gpu_estimate"]["gpu_count"]
        * 1_000_000_000_000
    )
    compute_time_us = layer_operations / effective_flops * 1_000_000

    tokens_per_block = profile["tokens_per_kv_block"]
    block_size_bytes = math.ceil(tokens_per_block * kv_bytes_per_token)
    block_count = math.ceil(cached_tokens / tokens_per_block)
    transport_bytes = block_count * block_size_bytes
    return {
        "input_tokens": input_tokens,
        "batch_size": batch_size,
        "total_input_tokens": total_input_tokens,
        "prefill_layer_hit_ratio": hit_ratio,
        "cached_tokens": cached_tokens,
        "recompute_tokens": recompute_tokens,
        "kv_cache_bytes_per_token_per_layer": kv_bytes_per_token,
        "tokens_per_kv_block": tokens_per_block,
        "block_size_bytes": block_size_bytes,
        "block_count": block_count,
        "useful_read_bytes": useful_read_bytes,
        "transport_bytes": transport_bytes,
        "compute_time_us": compute_time_us,
    }


class LLMWorkload:
    """按层生成KV Block，并根据SSD完成时间计算当前P节点的TTFT。"""

    def __init__(self, workload=None, scenario=None):
        """功能：创建一张GPU/P节点的独立逐层LLM工作负载。

        目的：为多GPU仿真分别保存层进度、待完成Block、GPU计算窗口和TTFT，
        避免不同GPU共享可变状态。

        输入：
            workload: 可选的完整GPU工作负载；None时使用YAML默认值。
            scenario: 可选的模型/GPU场景；None时使用默认场景。

        输出：
            None: 初始化尚未启动任何层的GPU状态。
        """
        self.workload = deepcopy(DEFAULT_WORKLOAD if workload is None else workload)
        self.scenario = build_scenario() if scenario is None else deepcopy(scenario)
        self.layer_plan = build_layer_plan(
            input_tokens=self.workload["input_tokens"],
            hit_ratio=self.workload["prefill_layer_hit_ratio"],
            batch_size=self.workload["batch_size"],
            scenario=self.scenario,
            model_name=self.workload["model"],
        )
        self.layer_indexes = list(range(
            self.workload["first_layer_index"],
            self.workload["last_layer_index"] + 1,
        ))
        self.next_layer_position = 0
        self.next_layer_start_time_us = self.workload["arrival_time_us"]
        self.current_layer = None
        self.request_to_layer = {}
        self.layer_results = []
        self.request_count = 0
        self.completed_request_count = 0
        self.first_token_time_us = None
        self.ttft_us = None

    def next_layer_start_time(self):
        """功能：查询下一层可启动的仿真时刻。

        目的：让全局事件引擎为每张GPU独立安排层就绪事件。

        输入：
            无。

        输出：
            float: 下一层计算与IO可开始的微秒时刻。
        """
        return self.next_layer_start_time_us

    def can_start_next_layer(self):
        """功能：判断这张GPU当前是否可以独立启动下一层。

        目的：多GPU顶层事件引擎使用这个公开状态，避免访问
        ``current_layer`` 等内部成员，也避免一个GPU等待其他GPU。

        输入：
            无。

        输出：
            bool: 当前无未完成层、仍有后续层且未产生首Token时返回True。
        """
        has_more_layers = self.next_layer_position < len(self.layer_indexes)
        return (
            self.current_layer is None
            and has_more_layers
            and not self.is_complete()
        )

    def start_next_layer(self):
        """功能：生成下一层的中立Block读取计划并登记完成关系。

        目的：在请求分散到多块SSD前建立 ``request_id -> layer`` 屏障，
        使任意SSD返回顺序都能正确判断本层最后一个Block。

        输入：
            无；使用当前GPU的下一层位置和通用层计划。

        输出：
            dict: demand group、P节点、服务窗口和全部中立Block。
        """
        layer_index = self.layer_indexes[self.next_layer_position]
        layer_id = f"{self.workload['workload_id']}_layer_{layer_index:02d}"
        layer_start_time_us = self.next_layer_start_time_us
        layer_state = {
            "layer_request_id": layer_id,
            "layer_index": layer_index,
            "layer_start_time_us": layer_start_time_us,
            "compute_done_time_us": (
                layer_start_time_us + self.layer_plan["compute_time_us"]
            ),
            "pending_request_ids": set(),
            "io_completion_time_us": None,
            # 每块SSD分别保存当前GPU层的最晚Block完成时刻；
            # 层级读取完成时间取这些值的最大值，不求和或平均。
            "ssd_completion_times_us": {},
        }

        blocks = []
        for block_index in range(self.layer_plan["block_count"]):
            request_id = f"{layer_id}_block_{block_index:05d}"
            layer_state["pending_request_ids"].add(request_id)
            self.request_to_layer[request_id] = layer_state
            blocks.append({
                "request_id": request_id,
                # 显式层内下标供确定性平衡Placement使用；
                # 即使调用方改变Block遍历顺序，映射也不变。
                "block_index": block_index,
                "size_bytes": self.layer_plan["block_size_bytes"],
            })

        self.current_layer = layer_state
        self.next_layer_position += 1
        self.request_count += len(blocks)
        if not blocks:
            self._finish_layer(layer_state)
        return {
            "demand_group_id": layer_id,
            "p_node_id": self.workload["p_node_id"],
            "service_window_us": self.layer_plan["compute_time_us"],
            "blocks": blocks,
        }

    def _finish_layer(self, layer_state):
        """功能：固化一层的GPU计算和SSD IO完成结果。

        目的：将层结束定义为计算与IO两者的较晚时刻，计算SSD stall，
        并使下一层只从该屏障时刻开始。

        输入：
            layer_state: 当前层的启动时间、计算截止和IO完成状态。

        输出：
            None: 追加层结果、释放当前层，并可能设置首Token时刻。
        """
        io_completion_time_us = layer_state["io_completion_time_us"]
        if io_completion_time_us is None:
            io_completion_time_us = layer_state["layer_start_time_us"]
        layer_end_time_us = max(
            layer_state["compute_done_time_us"],
            io_completion_time_us,
        )
        ssd_stall_us = max(
            0,
            io_completion_time_us - layer_state["compute_done_time_us"],
        )
        self.layer_results.append({
            "layer_request_id": layer_state["layer_request_id"],
            "layer_index": layer_state["layer_index"],
            "layer_start_time_us": layer_state["layer_start_time_us"],
            "compute_done_time_us": layer_state["compute_done_time_us"],
            "io_completion_time_us": io_completion_time_us,
            "ssd_completion_times_us": dict(
                layer_state["ssd_completion_times_us"]
            ),
            "layer_end_time_us": layer_end_time_us,
            "ssd_stall_us": ssd_stall_us,
            "block_count": self.layer_plan["block_count"],
        })
        self.next_layer_start_time_us = layer_end_time_us
        self.current_layer = None

        if len(self.layer_results) == len(self.layer_indexes):
            self.first_token_time_us = layer_end_time_us
            self.ttft_us = (
                self.first_token_time_us - self.workload["arrival_time_us"]
            )

    def on_storage_complete(self, completion):
        """功能：接收任意SSD返回的一个完整Block完成通知。

        目的：将跨SSD、乱序完成的Block汇聚回所属层，只在最后一个Block
        完成后解除层屏障。

        输入：
            completion: 至少包含 ``request_id`` 和 ``completion_time_us`` 的完成字典。

        输出：
            None: 更新待完成集合，并在集合清空时结束当前层。
        """
        request_id = completion["request_id"]
        layer_state = self.request_to_layer.pop(request_id)
        layer_state["pending_request_ids"].remove(request_id)
        self.completed_request_count += 1
        completion_time_us = completion["completion_time_us"]
        storage_target_id = completion["storage_target_id"]
        previous_storage_time = layer_state["ssd_completion_times_us"].get(
            storage_target_id
        )
        layer_state["ssd_completion_times_us"][storage_target_id] = (
            completion_time_us
            if previous_storage_time is None
            else max(previous_storage_time, completion_time_us)
        )
        previous_time = layer_state["io_completion_time_us"]
        layer_state["io_completion_time_us"] = (
            completion_time_us
            if previous_time is None
            else max(previous_time, completion_time_us)
        )

        if layer_state["pending_request_ids"]:
            return
        self._finish_layer(layer_state)

    def is_complete(self):
        """功能：查询这张GPU是否已计算出最后一层首Token时刻。

        目的：让顶层判断LLM状态；全局时钟是否真正到达该时刻，仍由
        ``JointSimulation`` 的显式GPU完成事件保证。

        输入：
            无。

        输出：
            bool: ``ttft_us`` 已经确定时返回True。
        """
        return self.ttft_us is not None

    def result(self):
        """功能：整理一张GPU/P节点的工作负载与逐层最终结果。

        目的：为多GPU顶层统计、DPU策略比较和请求守恒校验提供稳定字典。

        输入：
            无。

        输出：
            dict: 工作负载元数据、请求计数、计算/SSD时间、TTFT和逐层记录。
        """
        return {
            "workload_id": self.workload["workload_id"],
            "p_node_id": self.workload["p_node_id"],
            "model": self.workload["model"],
            "input_tokens": self.workload["input_tokens"],
            "batch_size": self.workload["batch_size"],
            "prefill_layer_hit_ratio": self.workload["prefill_layer_hit_ratio"],
            "layer_count": len(self.layer_indexes),
            "block_count_per_layer": self.layer_plan["block_count"],
            "block_size_bytes": self.layer_plan["block_size_bytes"],
            "request_count": self.request_count,
            "completed_request_count": self.completed_request_count,
            "compute_time_us_per_layer": self.layer_plan["compute_time_us"],
            "compute_only_ttft_us": (
                len(self.layer_indexes) * self.layer_plan["compute_time_us"]
            ),
            "ssd_stall_us": sum(
                layer["ssd_stall_us"] for layer in self.layer_results
            ),
            "first_token_time_us": self.first_token_time_us,
            "ttft_us": self.ttft_us,
            "layers": list(self.layer_results),
        }
