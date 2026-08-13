#!/usr/bin/env python3
"""统一运行单/多GPU、单/多SSD的QoS与SSD离散事件仿真。"""

import argparse
from copy import deepcopy
from functools import partial
import math
from pathlib import Path

from DPU import (
    DPURequestGateway,
    DemandAwareRateController,
    build_queue_binding_strategy,
)
from backends.asu_ssd import SSDSimulator, load_ssd_config
from backends.asu_ssd.time_utils import time_to_us, us_to_time
from discrete_simulation import EventLoop
from llm_workload.inference_workload_sampler import (
    UniformRandomInferenceSampler,
)
from llm_workload.kv_placement_manager import KVPlacementManager
from llm_workload.layer_request import DEFAULT_WORKLOAD, LLMWorkload
from qos import build_qos_simulator, load_queue_layout
from simulation_common.config_utils import load_yaml
from simulation_common.storage_path import StoragePath


PROJECT_DIR = Path(__file__).resolve().parent
QOS_CONFIG_DIR = PROJECT_DIR / "qos" / "config"
INTEGRATION_CONFIG_FILE = QOS_CONFIG_DIR / "qos_ssd_config.yaml"
MULTI_GPU_CONFIG_FILE = (
    PROJECT_DIR / "llm_workload" / "config" / "multi_gpu_workloads.yaml"
)
GPU_LAYER_READY_PRIORITY = 10
# 同时刻先固化已完成推理，再让所有GPU的下一次推理一起进入层就绪阶段。
GPU_COMPLETION_PRIORITY = 5


def _resolve_integration_path(path_text):
    """功能：解析联合配置中相对于qos/config的文件路径。

    目的：保证从任意工作目录启动脚本时都读取同一组项目配置。

    输入：
        path_text: ``qos_ssd_config.yaml`` 中记录的相对路径文本。

    输出：
        Path: 规范化后的绝对文件路径。
    """
    return (QOS_CONFIG_DIR / path_text).resolve()


def _deep_merge(base, override):
    """功能：递归合并默认字典和局部覆盖字典。

    目的：允许某张GPU只覆盖 ``placement.allowed_storage_targets`` 等嵌套字段，
    而不会意外删除同级的策略名称或随机种子。

    输入：
        base: 提供完整默认值的字典。
        override: 只包含需要替换字段的字典。

    输出：
        dict: 不修改输入对象的新合并字典。
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _build_topology(simulation_config):
    """功能：展开GPU和StoragePath拓扑配置。

    目的：动态生成稳定设备ID；每个StoragePath表示独立QoS+SSD。

    输入：
        simulation_config: 项目级 ``config/simulation_config.yaml`` 的simulation字典。

    输出：
        dict: 包含GPU ID、P节点ID和Storage Target ID列表的规范化拓扑。
    """
    topology_config = simulation_config["topology"]
    gpu_count = topology_config["gpu_count"]
    storage_path_count = topology_config["storage_path_count"]
    gpu_ids = [
        f"{topology_config['gpu_id_prefix']}{index}"
        for index in range(gpu_count)
    ]
    p_node_ids = [
        f"{topology_config['p_node_id_prefix']}{index}"
        for index in range(gpu_count)
    ]
    storage_target_ids = [
        f"{topology_config['storage_target_id_prefix']}{index}"
        for index in range(storage_path_count)
    ]
    return {
        "gpu_ids": gpu_ids,
        "p_node_ids": p_node_ids,
        "storage_target_ids": storage_target_ids,
    }


def _build_gpu_workloads(
    topology,
    multi_gpu_config,
    single_workload=None,
):
    """功能：为拓扑中的每张独立GPU构造工作负载模板。

    目的：GPU数量只由项目级simulation YAML决定；multi_gpu YAML仅提供默认值和按ID
    覆盖。模板中的输入长度和命中率会在后续每次推理采样时替换。

    输入：
        topology: ``_build_topology`` 生成的设备ID列表。
        multi_gpu_config: multi_gpu_workloads YAML中的默认值和覆盖项。
        single_workload: 可选旧接口单GPU完整工作负载；提供时只创建GPU0。

    输出：
        dict: ``gpu_id -> 完整workload模板`` 映射。
    """
    if single_workload is not None:
        workload = _deep_merge(DEFAULT_WORKLOAD, single_workload)
        workload.setdefault("p_node_id", topology["p_node_ids"][0])
        return {topology["gpu_ids"][0]: workload}

    defaults = multi_gpu_config.get("defaults", {})
    overrides = multi_gpu_config.get("gpu_overrides", {})

    workloads = {}
    for index, gpu_id in enumerate(topology["gpu_ids"]):
        workload = _deep_merge(DEFAULT_WORKLOAD, defaults)
        gpu_override = overrides.get(gpu_id, {})
        workload = _deep_merge(workload, gpu_override)

        # 默认YAML只描述一张GPU，因此拓扑展开时必须重新生成全局唯一身份。
        # 用户若在GPU覆盖项中显式提供这两个字段，则尊重实验配置。
        if "workload_id" not in gpu_override:
            workload["workload_id"] = (
                f"{DEFAULT_WORKLOAD['workload_id']}_{gpu_id}"
            )
        if "p_node_id" not in gpu_override:
            workload["p_node_id"] = topology["p_node_ids"][index]
        workloads[gpu_id] = workload
    return workloads


class JointSimulation:
    """装配并运行多GPU通过DPU访问多条独立QoS+SSD路径的仿真。"""

    def __init__(self, binding_strategy_name=None, workload=None):
        """功能：根据全部YAML创建一次全新的联合仿真状态。

        目的：每次策略比较都重新创建GPU推理序列、QoS、SSD和事件日历，
        确保前一次实验的随机状态、令牌、队列和流水线不会污染下一次结果。

        输入：
            binding_strategy_name: 可选DPU Queue绑定策略覆盖名称。
            workload: 可选旧接口单GPU工作负载；提供时强制使用1个GPU。

        输出：
            None: 完成全部组件装配、随机序列生成和GPU首次推理安排。
        """
        integration = load_yaml(INTEGRATION_CONFIG_FILE)["integration"]
        global_simulation_config_file = _resolve_integration_path(
            integration["global_simulation_config"]
        )
        self.global_simulation_config = load_yaml(
            global_simulation_config_file
        )["simulation"]
        if workload is not None:
            self.global_simulation_config = deepcopy(
                self.global_simulation_config
            )
            self.global_simulation_config["topology"]["gpu_count"] = 1

        self.topology = _build_topology(self.global_simulation_config)
        self.start_time_us = self.global_simulation_config["start_time_us"]
        self.event_loop = EventLoop(
            start_time=us_to_time(self.start_time_us)
        )
        self.request_owner = {}
        self.scheduled_gpu_events = set()
        self.scheduled_gpu_completion_events = set()
        self.completed_gpu_ids = set()

        layout_file = _resolve_integration_path(
            integration["queue_layout_config"]
        )
        qos_runtime_config_file = _resolve_integration_path(
            integration["qos_runtime_config"]
        )
        queue_layout = load_queue_layout(layout_file)
        backend_config = load_ssd_config(
            _resolve_integration_path(integration["backend_config"])
        )

        self.storage_paths = {}
        for storage_target_id in self.topology["storage_target_ids"]:
            qos = build_qos_simulator(
                layout_config_file=layout_file,
                token_config_file=_resolve_integration_path(
                    integration["token_bucket_config"]
                ),
                scheduler_config_file=_resolve_integration_path(
                    integration["wrr_config"]
                ),
                qos_runtime_config_file=qos_runtime_config_file,
                start_time_us=self.start_time_us,
                queue_layout=queue_layout,
            )
            ssd = SSDSimulator(
                backend_config=deepcopy(backend_config),
                completion_sink=self._on_storage_complete,
                storage_target_id=storage_target_id,
            )
            self.storage_paths[storage_target_id] = StoragePath(
                storage_target_id=storage_target_id,
                qos=qos,
                ssd=ssd,
                event_loop=self.event_loop,
            )

        multi_gpu_config = load_yaml(MULTI_GPU_CONFIG_FILE)["multi_gpu"]
        gpu_workload_templates = _build_gpu_workloads(
            topology=self.topology,
            multi_gpu_config=multi_gpu_config,
            single_workload=workload,
        )

        # 旧的workload参数是单GPU快速实验接口，保持只运行一次；
        # 正常YAML入口则在事件循环前一次性生成全部随机序列。
        if workload is None:
            generation_config = self.global_simulation_config[
                "workload_generation"
            ]
            sampler = UniformRandomInferenceSampler(generation_config)
            self.inter_inference_gap_us = sampler.inter_inference_gap_us
            self.gpu_workload_sequences = {
                gpu_id: sampler.build_sequence(template, gpu_id)
                for gpu_id, template in gpu_workload_templates.items()
            }
            self.workload_generation_config = deepcopy(generation_config)
        else:
            self.inter_inference_gap_us = 0
            self.gpu_workload_sequences = {
                gpu_id: [template]
                for gpu_id, template in gpu_workload_templates.items()
            }
            self.workload_generation_config = {
                "mode": "single_workload_override",
                "inference_count_per_gpu": 1,
            }

        # llms只保存每张GPU正在运行的一次推理。完成后结果
        # 转移到completed_inference_results，再用下一个新实例替换它。
        self.llms = {}
        self.active_inference_indexes = {}
        self.next_inference_indexes = {
            gpu_id: 0 for gpu_id in self.gpu_workload_sequences
        }
        self.completed_inference_results = {
            gpu_id: [] for gpu_id in self.gpu_workload_sequences
        }
        self.placement_managers = {
            gpu_id: KVPlacementManager(
                storage_target_ids=self.topology["storage_target_ids"],
                placement_config=template["placement"],
            )
            for gpu_id, template in gpu_workload_templates.items()
        }

        dpu_config = load_yaml(
            _resolve_integration_path(integration["dpu_config"])
        )["dpu"]
        binding_config = dpu_config["queue_binding"]
        selected_strategy = (
            binding_config["strategy"]
            if binding_strategy_name is None
            else binding_strategy_name
        )
        queue_ids_by_storage_target = {
            storage_target_id: queue_layout.queue_order
            for storage_target_id in self.topology["storage_target_ids"]
        }
        queue_binding_strategy = build_queue_binding_strategy(
            strategy_name=selected_strategy,
            random_seed=binding_config["random_seed"],
            p_node_ids=self.topology["p_node_ids"],
            queue_ids_by_storage_target=queue_ids_by_storage_target,
        )
        # 每个storage_target_id绑定到其独立QoS实例。顶层只在初始化时完成
        # DPU与QoS的状态/设置接口接线；运行期间DPU直接读取Queue数量并写CIR/PIR，
        # 不经StoragePath转发，也不读取SSD完成或NAND内部状态。
        qos_interfaces_by_storage_target = {
            storage_target_id: storage_path.qos
            for storage_target_id, storage_path in self.storage_paths.items()
        }
        rate_controller = None
        if dpu_config["rate_control"]["enabled"]:
            # DPU控制面直接使用SSD整数Byte/s容量，不转成浮点GB/s。
            rate_controller = DemandAwareRateController(
                capacity_bytes_per_second_by_storage_target={
                    storage_target_id: backend_config["nand"][
                        "read_bandwidth_bytes_per_second"
                    ]
                    for storage_target_id in self.topology[
                        "storage_target_ids"
                    ]
                },
                # 每块SSD有独立QoS实例，因此也传入独立的
                # Queue->Group固定连线，用于生成该路径的WRR权重。
                queue_to_group_by_storage_target={
                    storage_target_id: queue_layout.queue_to_group
                    for storage_target_id in self.topology[
                        "storage_target_ids"
                    ]
                },
            )
        self.dpu = DPURequestGateway(
            queue_ids_by_storage_target=queue_ids_by_storage_target,
            queue_binding_strategy=queue_binding_strategy,
            request_sink=self._route_qos_request,
            qos_interfaces_by_storage_target=(
                qos_interfaces_by_storage_target
            ),
            rate_controller=rate_controller,
        )

        for gpu_id, template in gpu_workload_templates.items():
            self._start_next_inference(
                gpu_id=gpu_id,
                start_time_us=template["arrival_time_us"],
            )

    def _start_next_inference(self, gpu_id, start_time_us):
        """功能：在指定时刻启动某张GPU的下一次推理。

        目的：为每次推理创建全新 ``LLMWorkload`` 状态，使同一GPU
        内的层进度、TTFT和待完成Block不会跨推理污染。

        输入：
            gpu_id: 要启动下一次推理的GPU编号。
            start_time_us: 新推理的全局到达时刻，单位微秒。

        输出：
            bool: 成功启动返回True；该GPU已无剩余推理时返回False。
        """
        inference_index = self.next_inference_indexes[gpu_id]
        sequence = self.gpu_workload_sequences[gpu_id]
        if inference_index >= len(sequence):
            return False

        workload = deepcopy(sequence[inference_index])
        workload["arrival_time_us"] = start_time_us
        self.llms[gpu_id] = LLMWorkload(workload=workload)
        self.active_inference_indexes[gpu_id] = inference_index
        self.next_inference_indexes[gpu_id] += 1
        self._schedule_gpu_layer(gpu_id)
        return True

    def _route_qos_request(self, request):
        """功能：根据DPU请求中的storage_target_id路由到对应StoragePath。

        目的：让KV Placement选择成为唯一SSD路由依据，DPU和QoS不再重新选择
        或忽略目标设备。

        输入：
            request: DPU已经展平并添加Queue及到达时刻的QoS请求。

        输出：
            dict: 目标StoragePath登记并返回的原请求。
        """
        storage_target_id = request["storage_target_id"]
        return self.storage_paths[storage_target_id].input(request)

    def _schedule_gpu_layer(self, gpu_id):
        """功能：安排某张GPU下一层就绪事件。

        目的：GPU计算窗口和SSD完成屏障都通过未来事件表达，不在完成回调中
        直接跨越时间生成请求。

        输入：
            gpu_id: 需要检查和安排下一层的GPU编号。

        输出：
            None: GPU不可继续时不安排；可继续时登记一次去重事件。
        """
        llm = self.llms[gpu_id]
        if not llm.can_start_next_layer():
            return
        inference_index = self.active_inference_indexes[gpu_id]
        event_time = us_to_time(llm.next_layer_start_time())
        event_time = max(event_time, self.event_loop.current_time)
        event_key = (gpu_id, inference_index, event_time)
        if event_key in self.scheduled_gpu_events:
            return
        self.scheduled_gpu_events.add(event_key)
        self.event_loop.schedule_at(
            event_time=event_time,
            priority=GPU_LAYER_READY_PRIORITY,
            event_name=f"gpu-layer-ready:{gpu_id}:{inference_index}",
            callback=partial(
                self._process_gpu_layer_ready,
                gpu_id,
                inference_index,
            ),
        )

    def _process_gpu_layer_ready(self, gpu_id, inference_index, event_time):
        """功能：在全局时刻生成某张GPU当前可开始层的全部Block请求。

        目的：让多张GPU同一时间戳的层事件先全部执行，再由较低优先级QoS
        统一接收到达；零Block层则继续安排下一计算完成时刻。

        输入：
            gpu_id: 当前事件所属GPU编号。
            inference_index: 当前GPU上的推理序号。
            event_time: 全局事件日历传入的内部整数时刻。

        输出：
            None: 请求经Placement和DPU登记到各StoragePath。
        """
        self.scheduled_gpu_events.discard(
            (gpu_id, inference_index, event_time)
        )
        llm = self.llms[gpu_id]
        placement_manager = self.placement_managers[gpu_id]

        while (
            llm.can_start_next_layer()
            and us_to_time(llm.next_layer_start_time()) <= event_time
        ):
            layer_read_plan = llm.start_next_layer()
            layer_requests = placement_manager.build_requests(layer_read_plan)
            arrival_time_us = time_to_us(event_time)

            for request in layer_requests:
                request_id = request["basic"]["request_id"]
                self.request_owner[request_id] = (gpu_id, inference_index)

            # 同一层的全部Block在同一个硬件时刻到达。DPU先完成所有Queue绑定
            # 和状态登记，再按受影响SSD各计算一次CIR/PIR，避免把同一组聚合需求
            # 按Block重复相加，也避免为每个IO生成一轮冗余控制设置。
            self.dpu.submit_batch(
                requests=layer_requests,
                arrival_time_us=arrival_time_us,
            )

            # 非空层必须等待分散在所有SSD上的Block全部完成；零Block层已经由
            # LLMWorkload立即结束，可以在while下一轮检查其未来计算完成时刻。
            if layer_requests:
                return

        self._schedule_gpu_layer(gpu_id)
        self._schedule_gpu_completion(gpu_id)

    def _schedule_gpu_completion(self, gpu_id):
        """功能：为已经算出首Token时刻的当前推理安排完成事件。

        目的：最后一层IO可能早于GPU计算窗口完成；全局事件时钟必须
        真正推进到 ``first_token_time_us``，不能在SSD回调时刻提前停止。

        输入：
            gpu_id: 需要检查当前推理首Token完成时刻的GPU编号。

        输出：
            None: 当前推理尚未完成最后一层时不安排，否则登记去重事件。
        """
        llm = self.llms[gpu_id]
        if not llm.is_complete() or gpu_id in self.completed_gpu_ids:
            return

        inference_index = self.active_inference_indexes[gpu_id]
        completion_time = us_to_time(llm.first_token_time_us)
        # IO晚于计算完成时，首Token时刻就是当前SSD完成时刻；
        # max同时防止浮点到整数时间换算产生极小的时钟回退。
        completion_time = max(completion_time, self.event_loop.current_time)
        event_key = (gpu_id, inference_index, completion_time)
        if event_key in self.scheduled_gpu_completion_events:
            return
        self.scheduled_gpu_completion_events.add(event_key)
        self.event_loop.schedule_at(
            event_time=completion_time,
            priority=GPU_COMPLETION_PRIORITY,
            event_name=f"gpu-complete:{gpu_id}:{inference_index}",
            callback=partial(
                self._process_gpu_completion,
                gpu_id,
                inference_index,
            ),
        )

    def _process_gpu_completion(self, gpu_id, inference_index, event_time):
        """功能：固化一次推理结果并按需启动下一次推理。

        目的：将LLM计算出的未来首Token时刻转成真实事件；
        同一GPU的推理严格串行，不同GPU仍由全局日历并行推进。

        输入：
            gpu_id: 当前完成事件所属GPU编号。
            inference_index: 当前GPU上已完成的推理序号。
            event_time: 全局事件日历传入的内部整数时刻。

        输出：
            None: 保存本次结果，启动后续推理或标记整张GPU完成。
        """
        self.scheduled_gpu_completion_events.discard(
            (gpu_id, inference_index, event_time)
        )
        llm = self.llms[gpu_id]

        inference_result = llm.result()
        inference_result["gpu_id"] = gpu_id
        inference_result["inference_index"] = inference_index
        self.completed_inference_results[gpu_id].append(inference_result)

        # 下一次推理的到达时间必须基于本次真实完成事件，
        # 而不是预估计算时间，否则SSD stall会造成同一GPU上的推理重叠。
        next_start_time_us = (
            time_to_us(event_time) + self.inter_inference_gap_us
        )
        if self._start_next_inference(gpu_id, next_start_time_us):
            return
        self.completed_gpu_ids.add(gpu_id)

    def _on_storage_complete(self, completion):
        """功能：把任意SSD的完整请求完成通知路由回所属GPU。

        目的：一层Block可以分布在多块SSD上，但只有最后一个Block完成后，
        LLMWorkload才解除本层屏障并安排下一层。

        输入：
            completion: 包含request_id、storage_target_id和完成时刻的字典。

        输出：
            None: 更新所属GPU状态并按需安排下一层事件。
        """
        request_id = completion["request_id"]
        gpu_id, _ = self.request_owner.pop(request_id)
        llm = self.llms[gpu_id]
        llm.on_storage_complete(completion)
        self._schedule_gpu_layer(gpu_id)
        self._schedule_gpu_completion(gpu_id)

    def _all_gpus_complete(self):
        """功能：检查全部GPU是否已完成配置的所有推理。

        目的：作为全局事件循环唯一正常停止条件，避免提前按某个QoS或SSD空闲
        状态结束仍在计算或等待其他设备的GPU。

        输入：
            无。

        输出：
            bool: 每张GPU的最后一次推理完成事件均已处理时返回True。
        """
        return len(self.completed_gpu_ids) == len(self.gpu_workload_sequences)

    def _build_gpu_result(self, gpu_id):
        """功能：汇总一张GPU的全部随机推理结果。

        目的：在保留每次输入长度、命中率和TTFT明细的同时，
        提供每GPU的请求守恒、平均/P95/最大TTFT和累计SSD等待统计。

        输入：
            gpu_id: 需要汇总的GPU编号。

        输出：
            dict: GPU级聚合指标和按序保存的 ``inferences`` 明细。
        """
        inference_results = self.completed_inference_results[gpu_id]
        expected_count = len(self.gpu_workload_sequences[gpu_id])
        ttft_values = [result["ttft_us"] for result in inference_results]
        ordered_ttft = sorted(ttft_values)
        p95_index = math.ceil(0.95 * len(ordered_ttft)) - 1
        generated_bytes = sum(
            result["request_count"] * result["block_size_bytes"]
            for result in inference_results
        )
        return {
            "gpu_id": gpu_id,
            "p_node_id": inference_results[0]["p_node_id"],
            "inference_count": expected_count,
            "completed_inference_count": len(inference_results),
            "request_count": sum(
                result["request_count"] for result in inference_results
            ),
            "completed_request_count": sum(
                result["completed_request_count"]
                for result in inference_results
            ),
            "generated_bytes": generated_bytes,
            "ssd_stall_us": sum(
                result["ssd_stall_us"] for result in inference_results
            ),
            "mean_ttft_us": sum(ttft_values) / len(ttft_values),
            "p95_ttft_us": ordered_ttft[p95_index],
            "max_ttft_us": ordered_ttft[-1],
            "inferences": list(inference_results),
        }

    def _conservation_statistics(self, gpu_results, path_results):
        """功能：汇总GPU、QoS和SSD的请求/字节计数。

        目的：让测试或分析程序可以直接比较守恒关系，不在主流程重复报错。

        输入：
            gpu_results: 按GPU保存的多次推理聚合结果。
            path_results: 按SSD保存的QoS和SSD最终结果。

        输出：
            dict: 全局与每块SSD请求/字节统计。
        """
        gpu_request_count = sum(
            result["request_count"] for result in gpu_results.values()
        )
        gpu_completed_count = sum(
            result["completed_request_count"]
            for result in gpu_results.values()
        )
        qos_input_count = sum(
            result["qos"]["input_request_count"]
            for result in path_results.values()
        )
        qos_dispatched_count = sum(
            result["qos"]["dispatched_request_count"]
            for result in path_results.values()
        )
        ssd_completed_count = sum(
            len(result["ssd"]["completed_requests"])
            for result in path_results.values()
        )
        gpu_generated_bytes = sum(
            result["generated_bytes"] for result in gpu_results.values()
        )
        qos_dispatched_bytes = sum(
            result["qos"]["dispatched_bytes"]
            for result in path_results.values()
        )
        ssd_completed_bytes = sum(
            result["ssd"]["completed_bytes"]
            for result in path_results.values()
        )
        dpu_statistics = self.dpu.statistics()
        per_storage_target = {}
        for storage_target_id, path_result in path_results.items():
            target_assignment_counts = dpu_statistics[
                "assignment_counts"
            ].get(storage_target_id, {})
            dpu_assigned_count = sum(
                request_count
                for queue_counts in target_assignment_counts.values()
                for request_count in queue_counts.values()
            )
            qos_result = path_result["qos"]
            ssd_result = path_result["ssd"]
            per_storage_target[storage_target_id] = {
                "dpu_assigned_requests": dpu_assigned_count,
                "qos_input_requests": qos_result["input_request_count"],
                "qos_dispatched_requests": qos_result[
                    "dispatched_request_count"
                ],
                "ssd_completed_requests": len(
                    ssd_result["completed_requests"]
                ),
                "qos_dispatched_bytes": qos_result["dispatched_bytes"],
                "ssd_completed_bytes": ssd_result["completed_bytes"],
            }

        return {
            "gpu_requests": gpu_request_count,
            "gpu_completed_requests": gpu_completed_count,
            "qos_input_requests": qos_input_count,
            "qos_dispatched_requests": qos_dispatched_count,
            "ssd_completed_requests": ssd_completed_count,
            "gpu_bytes": gpu_generated_bytes,
            "qos_dispatched_bytes": qos_dispatched_bytes,
            "ssd_completed_bytes": ssd_completed_bytes,
            "per_storage_target": per_storage_target,
        }

    @staticmethod
    def _demand_satisfaction_statistics(inference_results):
        """功能：统计SSD读取需求在GPU计算窗口内的完成比例。

        目的：以一张GPU的一层为一个demand；该层分散在所有SSD
        上的Block全部完成，且最晚完成时刻不超过GPU计算窗口，
        才记为满足。因此指标使用真实SSD完成时刻，不使用QoS下发时刻。

        输入：全部GPU已完成的单次推理结果列表。
        输出：满足数、总需求数、满足率和每份需求明细。
        """
        demands = []
        for inference in inference_results:
            for layer in inference["layers"]:
                if layer["block_count"] == 0:
                    continue
                demands.append({
                    "gpu_id": inference["gpu_id"],
                    "inference_index": inference["inference_index"],
                    "demand_group_id": layer["layer_request_id"],
                    "compute_done_time_us": layer["compute_done_time_us"],
                    "io_completion_time_us": layer["io_completion_time_us"],
                    "satisfied": (
                        layer["io_completion_time_us"]
                        <= layer["compute_done_time_us"]
                    ),
                })
        satisfied_count = sum(demand["satisfied"] for demand in demands)
        return {
            "satisfied_demand_count": satisfied_count,
            "total_demand_count": len(demands),
            "satisfaction_ratio": (
                satisfied_count / len(demands) if demands else 1.0
            ),
            "demands": demands,
        }

    def run(self):
        """功能：运行全局事件日历直到全部GPU完成所有推理。

        目的：在一条单调时间线上交错推进任意数量的GPU、QoS和SSD，并以设备ID
        命名空间保存结果，供策略比较和后端带宽绘图使用。

        输入：
            无。

        输出：
            dict: 每GPU聚合/逐推理结果、每StoragePath结果和事件统计。
        """
        self.event_loop.run_until(self._all_gpus_complete)

        gpu_results = {
            gpu_id: self._build_gpu_result(gpu_id)
            for gpu_id in self.gpu_workload_sequences
        }
        inference_results = [
            inference_result
            for gpu_result in gpu_results.values()
            for inference_result in gpu_result["inferences"]
        ]
        path_results = {
            storage_target_id: storage_path.end()
            for storage_target_id, storage_path in self.storage_paths.items()
        }
        conservation = self._conservation_statistics(gpu_results, path_results)
        demand_satisfaction = self._demand_satisfaction_statistics(
            inference_results
        )

        dpu_statistics = self.dpu.statistics()
        result = {
            "gpu_count": len(gpu_results),
            "storage_path_count": len(path_results),
            # 保留ssd_count作为旧绘图/分析调用方的兼容字段。
            "ssd_count": len(path_results),
            "inference_count": len(inference_results),
            "queue_binding_strategy": dpu_statistics["strategy"],
            "rate_control_strategy": (
                None
                if dpu_statistics["rate_control"] is None
                else dpu_statistics["rate_control"]["strategy"]
            ),
            "gpus": gpu_results,
            # llms保留为展平后的单次推理结果列表，方便逐请求分析。
            "llms": inference_results,
            "workload_generation": deepcopy(
                self.workload_generation_config
            ),
            "storage_paths": path_results,
            "dpu": dpu_statistics,
            "request_conservation": conservation,
            "demand_satisfaction": demand_satisfaction,
            "event_loop": {
                "completion_time_us": time_to_us(self.event_loop.current_time),
                "processed_event_count": self.event_loop.processed_event_count,
            },
        }

        # 全局只有一次推理时保留旧llm键，其含义仍是单次推理。
        if len(inference_results) == 1:
            result["llm"] = inference_results[0]
        if len(path_results) == 1:
            only_path = next(iter(path_results.values()))
            result["qos"] = only_path["qos"]
            result["ssd"] = only_path["ssd"]
        return result


def run_joint_simulation(workload=None, binding_strategy=None):
    """功能：创建并运行一次全新的统一联合仿真。

    目的：提供脚本、测试和绘图共同使用的稳定入口，并支持临时覆盖DPU策略。

    输入：
        workload: 可选旧接口单GPU完整工作负载。
        binding_strategy: 可选互斥固定、普通固定或逐IO随机绑定。

    输出：
        dict: ``JointSimulation.run`` 返回的完整结果。
    """
    return JointSimulation(
        binding_strategy_name=binding_strategy,
        workload=workload,
    ).run()


def compare_queue_binding_strategies(strategy_names=None, workload=None):
    """功能：使用完全相同配置分别运行多种DPU Queue绑定策略。

    目的：建立项目最终策略评测入口，确保每种策略都从空令牌、空SSD和相同
    KV Placement映射开始，结果之间可以直接比较。

    输入：
        strategy_names: 可选策略名称列表；默认读取DPU YAML比较列表。
        workload: 可选用于快速单GPU实验的完整工作负载。

    输出：
        dict: ``strategy_name -> 联合仿真结果`` 映射。
    """
    if strategy_names is None:
        integration = load_yaml(INTEGRATION_CONFIG_FILE)["integration"]
        dpu_config = load_yaml(
            _resolve_integration_path(integration["dpu_config"])
        )["dpu"]["queue_binding"]
        strategy_names = dpu_config["comparison_strategies"]
    return {
        strategy_name: run_joint_simulation(
            workload=workload,
            binding_strategy=strategy_name,
        )
        for strategy_name in strategy_names
    }


def _effective_bandwidth_gb_s(ssd_result):
    """功能：计算一块SSD从首次接收到最后完成之间的平均有效带宽。

    目的：使用真实后端服务区间比较不同绑定策略下每块SSD的利用情况。

    输入：
        ssd_result: 单个StoragePath中的SSD最终结果。

    输出：
        float: 十进制GB/s；空SSD或零时长结果返回0。
    """
    first_time_us = ssd_result["first_submit_time_us"]
    last_time_us = ssd_result["last_completion_time_us"]
    if first_time_us is None or last_time_us is None:
        return 0.0
    elapsed_us = last_time_us - first_time_us
    if elapsed_us <= 0:
        return 0.0
    return ssd_result["completed_bytes"] / (elapsed_us * 1_000)


def _active_queue_count(dpu_result):
    """功能：统计一次仿真实际使用的全局Queue数量。

    目的：用 ``(SSD, Queue)`` 命名空间量化固定绑定与逐请求随机的分散程度。

    输入：
        dpu_result: DPU ``statistics`` 返回的绑定计数。

    输出：
        int: 至少接收过一个请求的不同 ``(storage_target_id, queue_id)`` 数量。
    """
    active_queues = set()
    for storage_target_id, p_node_counts in dpu_result[
        "assignment_counts"
    ].items():
        for queue_counts in p_node_counts.values():
            for queue_id, request_count in queue_counts.items():
                if request_count > 0:
                    active_queues.add((storage_target_id, queue_id))
    return len(active_queues)


def print_result(result):
    """功能：打印一次多GPU、多StoragePath联合仿真的核心结果。

    目的：同时展示每GPU的多次推理TTFT、每条QoS+SSD路径吞吐和DPU分散度，
    便于快速验证项目级拓扑配置。

    输入：
        result: ``run_joint_simulation`` 返回的完整结果。

    输出：
        None: 将人类可读摘要写入标准输出。
    """
    print("=" * 82)
    print(
        "多GPU -> DPU -> 多StoragePath联合仿真 | "
        f"GPU={result['gpu_count']} "
        f"StoragePath={result['storage_path_count']} "
        f"推理={result['inference_count']} "
        f"绑定策略={result['queue_binding_strategy']} "
        f"速率策略={result['rate_control_strategy']}"
    )
    print("=" * 82)

    for gpu_id, gpu_result in result["gpus"].items():
        print(
            f"{gpu_id}/{gpu_result['p_node_id']}: "
            f"推理={gpu_result['completed_inference_count']}/"
            f"{gpu_result['inference_count']}, "
            f"IO={gpu_result['completed_request_count']}/"
            f"{gpu_result['request_count']}, "
            f"平均TTFT={gpu_result['mean_ttft_us'] / 1000:.3f} ms, "
            f"P95={gpu_result['p95_ttft_us'] / 1000:.3f} ms, "
            f"最大={gpu_result['max_ttft_us'] / 1000:.3f} ms"
        )
        for inference in gpu_result["inferences"]:
            print(
                f"  #{inference['inference_index']:02d}: "
                f"输入={inference['input_tokens']:,} Token, "
                f"命中率={inference['prefill_layer_hit_ratio']:.2%}, "
                f"IO={inference['request_count']:,}, "
                f"TTFT={inference['ttft_us'] / 1000:.3f} ms"
            )

    for storage_target_id, path_result in result["storage_paths"].items():
        qos_result = path_result["qos"]
        ssd_result = path_result["ssd"]
        rate_statistics = result["dpu"]["rate_control"]
        rate_text = ""
        if rate_statistics is not None:
            peak_reserved = rate_statistics[
                "peak_reserved_bytes_per_second"
            ].get(storage_target_id, 0)
            peak_waiting = rate_statistics[
                "peak_waiting_demand_count"
            ].get(storage_target_id, 0)
            rate_text = (
                f", DPU峰值预留={peak_reserved / 1_000_000_000:.3f} GB/s, "
                f"峰值等待需求={peak_waiting}"
            )
        print(
            f"{storage_target_id}: QoS下发="
            f"{qos_result['dispatched_request_count']}/"
            f"{qos_result['input_request_count']}, "
            f"SSD完成字节={ssd_result['completed_bytes']:,}, "
            f"有效带宽={_effective_bandwidth_gb_s(ssd_result):.3f} GB/s"
            f"{rate_text}"
        )
    print(f"实际使用全局Queue数={_active_queue_count(result['dpu'])}")
    demand_satisfaction = result["demand_satisfaction"]
    print(
        "需求满足率="
        f"{demand_satisfaction['satisfied_demand_count']}/"
        f"{demand_satisfaction['total_demand_count']} "
        f"({demand_satisfaction['satisfaction_ratio']:.2%})"
    )
    print(
        "DPU CIR/PIR设置="
        f"{result['dpu']['rate_control_write_count']}, "
        "Group WRR权重设置="
        f"{result['dpu']['group_weight_write_count']}"
    )


def print_comparison(comparison_results):
    """功能：并排打印多种DPU Queue绑定策略的核心指标。

    目的：以相同随机工作负载快速比较平均/P95/最大TTFT、
    Queue分散度和总SSD字节。

    输入：
        comparison_results: ``compare_queue_binding_strategies`` 返回的结果映射。

    输出：
        None: 将策略对比表写入标准输出。
    """
    print("=" * 112)
    print("DPU Queue绑定策略比较")
    print("=" * 112)
    print(
        f"{'策略':<24}{'平均TTFT(ms)':>16}{'P95 TTFT(ms)':>16}"
        f"{'最大TTFT(ms)':>16}"
        f"{'使用Queue数':>14}{'SSD完成字节':>20}"
    )
    for strategy_name, result in comparison_results.items():
        ttft_values = sorted(
            inference_result["ttft_us"] / 1000
            for gpu_result in result["gpus"].values()
            for inference_result in gpu_result["inferences"]
        )
        p95_index = math.ceil(0.95 * len(ttft_values)) - 1
        total_completed_bytes = sum(
            path_result["ssd"]["completed_bytes"]
            for path_result in result["storage_paths"].values()
        )
        print(
            f"{strategy_name:<24}"
            f"{sum(ttft_values) / len(ttft_values):>16.3f}"
            f"{ttft_values[p95_index]:>16.3f}"
            f"{ttft_values[-1]:>16.3f}"
            f"{_active_queue_count(result['dpu']):>14d}"
            f"{total_completed_bytes:>20,d}"
        )


def parse_arguments():
    """功能：读取统一仿真入口的命令行参数。

    目的：允许用户运行YAML默认策略、临时选择单个策略或直接比较两种策略。

    输入：
        无；由argparse读取进程命令行。

    输出：
        argparse.Namespace: 解析后的策略覆盖和比较开关。
    """
    parser = argparse.ArgumentParser(
        description="Run multi-GPU, multi-SSD QoS simulation."
    )
    parser.add_argument(
        "--binding-strategy",
        choices=(
            "random_unique_sticky",
            "random_sticky",
            "random_per_request",
        ),
        default=None,
        help="override DPU queue binding strategy from YAML",
    )
    parser.add_argument(
        "--compare-binding-strategies",
        action="store_true",
        help="run and compare all strategies listed in DPU YAML",
    )
    return parser.parse_args()


def main():
    """功能：执行命令行选择的联合仿真或策略比较。

    目的：提供项目唯一可直接运行入口，替代已经删除的多GPU专用脚本。

    输入：
        无；使用 ``parse_arguments`` 的命令行结果。

    输出：
        None: 打印单次结果或策略对比表。
    """
    arguments = parse_arguments()
    if arguments.compare_binding_strategies:
        print_comparison(compare_queue_binding_strategies())
        return
    print_result(
        run_joint_simulation(binding_strategy=arguments.binding_strategy)
    )


if __name__ == "__main__":
    main()
