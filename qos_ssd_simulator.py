#!/usr/bin/env python3
"""统一运行单/多GPU、单/多SSD的QoS与SSD离散事件仿真。"""

import argparse
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import re
from time import perf_counter

from DPU import (
    CoflowPriorityController,
    DPURequestGateway,
    DemandAwareFCFSCIRController,
    UtilityEDFAblationController,
    UtilityEDFController,
    build_queue_binding_strategy,
)
from backends.asu_ssd import SSDSimulator
from backends.asu_ssd.time_utils import time_to_us, us_to_time
from discrete_simulation import EventLoop
from llm_workload.inference_workload_sampler import (
    UniformRandomInferenceSampler,
)
from llm_workload.kv_placement_manager import KVPlacementManager
from llm_workload.layer_request import (
    DEFAULT_WORKLOAD,
    LLMWorkload,
    build_scenario,
)
from qos import build_qos_simulator, build_queue_layout
from simulation_common.config_utils import load_yaml
from simulation_common.storage_path import StoragePath


PROJECT_DIR = Path(__file__).resolve().parent
SIMULATION_CONFIG_FILE = PROJECT_DIR / "config" / "simulation_config.yaml"
GPU_LAYER_READY_PRIORITY = 10
# 同时刻先固化已完成推理，再让所有GPU的下一次推理一起进入层就绪阶段。
GPU_COMPLETION_PRIORITY = 5


def load_simulation_config(config_file=SIMULATION_CONFIG_FILE):
    """功能：读取项目唯一仿真YAML中的完整simulation配置。

    目的：让统一入口、测试和组件默认值引用同一份GPU、SSD、DPU、QoS
    与工作负载配置，不再维护模块级YAML之间的路径关系。

    输入：
        config_file: 项目统一YAML路径。

    输出：
        dict: ``simulation`` 节点的完整深拷贝。
    """
    return deepcopy(load_yaml(config_file)["simulation"])


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
    # 正式扫描从topology.ssd_counts逐项覆盖storage_path_count；直接创建
    # JointSimulation时则默认使用列表中的第一项，避免在YAML重复写SSD数量。
    storage_path_count = topology_config.get(
        "storage_path_count",
        topology_config["ssd_counts"][0],
    )
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
    workload_config,
    single_workload=None,
):
    """功能：为拓扑中的每张独立GPU构造工作负载模板。

    目的：GPU数量、默认工作负载和按GPU覆盖都来自项目唯一YAML。模板中的
    输入长度和命中率会在后续每次推理采样时替换。

    输入：
        topology: ``_build_topology`` 生成的设备ID列表。
        workload_config: ``simulation.workload`` 默认值及 ``gpu_overrides``。
        single_workload: 可选旧接口单GPU完整工作负载；提供时只创建GPU0。

    输出：
        dict: ``gpu_id -> 完整workload模板`` 映射。
    """
    if single_workload is not None:
        workload = _deep_merge(DEFAULT_WORKLOAD, single_workload)
        workload.setdefault("p_node_id", topology["p_node_ids"][0])
        return {topology["gpu_ids"][0]: workload}

    defaults = deepcopy(workload_config)
    overrides = defaults.pop("gpu_overrides", {})

    workloads = {}
    for index, gpu_id in enumerate(topology["gpu_ids"]):
        workload = _deep_merge(DEFAULT_WORKLOAD, defaults)
        gpu_override = overrides.get(gpu_id, {})
        workload = _deep_merge(workload, gpu_override)

        # 默认YAML只描述一张GPU，因此拓扑展开时必须重新生成全局唯一身份。
        # 用户若在GPU覆盖项中显式提供这两个字段，则尊重实验配置。
        if "workload_id" not in gpu_override:
            workload["workload_id"] = f"{defaults['workload_id']}_{gpu_id}"
        if "p_node_id" not in gpu_override:
            workload["p_node_id"] = topology["p_node_ids"][index]
        workloads[gpu_id] = workload
    return workloads


class JointSimulation:
    """装配并运行多GPU通过DPU访问多条独立QoS+SSD路径的仿真。"""

    def __init__(
        self,
        binding_strategy_name=None,
        workload=None,
        rate_control_strategy_name=None,
        config=None,
        config_file=SIMULATION_CONFIG_FILE,
        simulation_config_override=None,
        workload_defaults_override=None,
        backend_config_override=None,
    ):
        """功能：根据项目唯一YAML创建一次全新的联合仿真状态。

        目的：每次策略比较都重新创建GPU推理序列、QoS、SSD和事件日历，
        确保前一次实验的随机状态、令牌、队列和流水线不会污染下一次结果。

        输入：
            binding_strategy_name: 可选DPU Queue绑定策略覆盖名称。
            workload: 可选旧接口单GPU工作负载；提供时强制使用1个GPU。
            rate_control_strategy_name: ``baseline`` 或DPU速率策略名称。
            config: 可选的完整 ``simulation`` 配置字典；测试可直接传入副本。
            config_file: config为None时读取的项目统一YAML路径。
            simulation_config_override: 项目级simulation字典的递归覆盖。
            workload_defaults_override: 全部GPU共用的LLM工作负载覆盖。
            backend_config_override: SSD后端配置的递归覆盖；用于
                在同一工作负载下对照detailed与batched_exact。
        输出：
            None: 完成全部组件装配、随机序列生成和GPU首次推理安排。
        """
        self.global_simulation_config = (
            load_simulation_config(config_file)
            if config is None
            else deepcopy(config)
        )
        if simulation_config_override is not None:
            # 实验覆盖只作用于本次新建实例，不改写项目YAML。
            self.global_simulation_config = _deep_merge(
                self.global_simulation_config,
                simulation_config_override,
            )
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

        qos_config = self.global_simulation_config["qos"]
        queue_layout = build_queue_layout(qos_config["queue_layout"])
        backend_config = deepcopy(
            self.global_simulation_config["ssd"]["backend"]
        )
        if backend_config_override is not None:
            # 后端模式覆盖只作用于本次仿真，不改写公共YAML；
            # 这使严格等价测试能从完全相同的其余状态启动。
            backend_config = _deep_merge(
                backend_config,
                backend_config_override,
            )

        self.storage_paths = {}
        for storage_target_id in self.topology["storage_target_ids"]:
            qos = build_qos_simulator(
                qos_config=qos_config,
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

        workload_config = deepcopy(self.global_simulation_config["workload"])
        if workload_defaults_override is not None:
            workload_config = _deep_merge(
                workload_config,
                workload_defaults_override,
            )
        gpu_workload_templates = _build_gpu_workloads(
            topology=self.topology,
            workload_config=workload_config,
            single_workload=workload,
        )
        self.llm_scenario = build_scenario(self.global_simulation_config)

        # 旧的workload参数是单GPU快速实验接口，保持只运行一次；
        # 正常YAML入口则在事件循环前一次性生成全部随机序列。
        if workload is None:
            generation_config = self.global_simulation_config[
                "workload_generation"
            ]
            sampler = UniformRandomInferenceSampler(generation_config)
            self.inter_inference_gap_us = sampler.inter_inference_gap_us
            self.gpu_workload_sequences = sampler.build_sequences(
                gpu_workload_templates
            )
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

        dpu_config = self.global_simulation_config["dpu"]
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
        configured_rate_control = dpu_config["rate_control"]
        if rate_control_strategy_name is None:
            selected_rate_control_strategy = configured_rate_control[
                "strategies"
            ][0]
        else:
            selected_rate_control_strategy = rate_control_strategy_name
        self.rate_control_strategy_name = selected_rate_control_strategy

        rate_controller = None
        capacity_by_storage_target = {
            storage_target_id: backend_config["nand"][
                "read_bandwidth_bytes_per_second"
            ]
            for storage_target_id in self.topology["storage_target_ids"]
        }
        if selected_rate_control_strategy == "demand_aware_fcfs_cir":
            # DPU控制面直接使用SSD整数Byte/s容量，不转成浮点GB/s。
            rate_controller = DemandAwareFCFSCIRController(
                capacity_bytes_per_second_by_storage_target=(
                    capacity_by_storage_target
                ),
            )
        elif selected_rate_control_strategy.startswith("utility_edf_"):
            strategy_match = re.fullmatch(
                r"utility_edf_(integer|power)_l([1-9][0-9]*)",
                selected_rate_control_strategy,
            )
            if strategy_match is None:
                raise ValueError(
                    "utility EDF strategy must match "
                    "utility_edf_<integer|power>_l<positive integer>"
                )
            compute_layer_count = (
                workload_config["last_layer_index"]
                - workload_config["first_layer_index"]
                + 1
            )
            rate_controller = UtilityEDFController(
                capacity_bytes_per_second_by_storage_target=(
                    capacity_by_storage_target
                ),
                score_mode=strategy_match.group(1),
                deadline_allowance_us=int(strategy_match.group(2)),
                compute_layer_count=compute_layer_count,
            )
        elif selected_rate_control_strategy.startswith("ablation"):
            strategy_match = re.fullmatch(
                r"ablation_c([01])_u([01])_e([01])",
                selected_rate_control_strategy,
            )
            if strategy_match is None:
                raise ValueError(
                    "ablation strategy must match "
                    "ablation_c<0|1>_u<0|1>_e<0|1>"
                )
            compute_layer_count = (
                workload_config["last_layer_index"]
                - workload_config["first_layer_index"]
                + 1
            )
            rate_controller = UtilityEDFAblationController(
                capacity_bytes_per_second_by_storage_target=(
                    capacity_by_storage_target
                ),
                coordination_enabled=(strategy_match.group(1) == "1"),
                utility_enabled=(strategy_match.group(2) == "1"),
                edf_enabled=(strategy_match.group(3) == "1"),
                compute_layer_count=compute_layer_count,
            )
        elif selected_rate_control_strategy.startswith((
            "coflow_",
            "cohort_",
            "paced_",
        )):
            # 策略名格式：
            #   coflow_<ordering>_k<并发GPU数>
            #   cohort_<ordering>_k<并发GPU数>
            #   paced_<ordering>_k<并发GPU数>
            # 前者只在有活跃Queue时占槽；后者在一次推理的
            # 多个KV读组之间保留owner，避免compute空档让全部
            # 初始层请求提前灌入SSD。
            persistent_cohort = selected_rate_control_strategy.startswith(
                "cohort_"
            )
            finite_selected_pir = selected_rate_control_strategy.startswith(
                "paced_"
            )
            strategy_prefix = (
                "cohort_"
                if persistent_cohort
                else "paced_" if finite_selected_pir else "coflow_"
            )
            strategy_suffix = selected_rate_control_strategy[
                len(strategy_prefix):
            ]
            ordering, width_text = strategy_suffix.rsplit("_k", 1)
            # 普通coflow与persistent cohort都获得每次推理的
            # KV读组数；只有cohort_前缀会启用跨层占槽。
            expected_coflow_count = (
                workload_config["last_layer_index"]
                - workload_config["first_layer_index"]
                + 1
            )
            rate_controller = CoflowPriorityController(
                capacity_bytes_per_second_by_storage_target=(
                    capacity_by_storage_target
                ),
                ordering=ordering,
                selection_width=int(width_text),
                persistent_cohort=persistent_cohort,
                expected_coflow_count=expected_coflow_count,
                finite_selected_pir=finite_selected_pir,
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
        self.llms[gpu_id] = LLMWorkload(
            workload=workload,
            scenario=self.llm_scenario,
        )
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
        dpu_statistics = self.dpu.statistics()
        result = {
            "gpu_count": len(gpu_results),
            "storage_path_count": len(path_results),
            # 保留ssd_count作为旧绘图/分析调用方的兼容字段。
            "ssd_count": len(path_results),
            "inference_count": len(inference_results),
            "queue_binding_strategy": dpu_statistics["strategy"],
            "rate_control_strategy": self.rate_control_strategy_name,
            "gpus": gpu_results,
            # llms保留为展平后的单次推理结果列表，方便逐请求分析。
            "llms": inference_results,
            "workload_generation": deepcopy(
                self.workload_generation_config
            ),
            "storage_paths": path_results,
            "dpu": dpu_statistics,
            "request_conservation": conservation,
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


def run_joint_simulation(
    workload=None,
    binding_strategy=None,
    rate_control_strategy=None,
    config=None,
    config_file=SIMULATION_CONFIG_FILE,
    simulation_config_override=None,
    workload_defaults_override=None,
    backend_config_override=None,
):
    """功能：创建并运行一次全新的统一联合仿真。

    目的：提供脚本、测试和绘图共同使用的稳定入口，并支持临时覆盖DPU策略。

    输入：
        workload: 可选旧接口单GPU完整工作负载。
        binding_strategy: 可选的DPU Queue绑定策略名称。
        rate_control_strategy: ``baseline`` 或DPU速率策略名称。
        config: 可选完整 ``simulation`` 配置字典。
        config_file: config为None时读取的统一YAML路径。
        simulation_config_override: 可选项目级仿真参数覆盖。
        workload_defaults_override: 可选全GPU LLM工作负载覆盖。
        backend_config_override: 可选SSD后端配置递归覆盖。

    输出：
        dict: ``JointSimulation.run`` 返回的完整结果。
    """
    return JointSimulation(
        binding_strategy_name=binding_strategy,
        workload=workload,
        rate_control_strategy_name=rate_control_strategy,
        config=config,
        config_file=config_file,
        simulation_config_override=simulation_config_override,
        workload_defaults_override=workload_defaults_override,
        backend_config_override=backend_config_override,
    ).run()


class CountOnlyAppendLog:
    """只统计追加次数，不保留逐请求诊断字典。"""

    def __init__(self):
        """功能：创建只计数的记录容器。

        目的：摘要实验不保留数百万条SSD完成或NAND事件字典，降低内存与运行时间。

        输入：无。

        输出：None；把累计记录数初始化为0。
        """
        self.count = 0

    def append(self, record):
        """功能：消费一条诊断记录并累计数量。

        目的：保持后端原有append调用位置和时序，仅省略与最终摘要无关的字典保存。

        输入：
            record: 一条SSD完成或NAND服务记录；内容不会被保留。

        输出：None；累计数量加1。
        """
        self.count += 1

    def __len__(self):
        """功能：返回已消费记录数。

        目的：提供与普通list相同的长度查询接口，供请求守恒统计使用。

        输入：无。

        输出：
            int: 已消费的记录数量。
        """
        return self.count


class DispatchAggregateLog:
    """聚合QoS下发数量、字节和速率类别，不保存逐IO字典。"""

    def __init__(self):
        """功能：创建空的QoS下发聚合器。

        目的：摘要实验保留守恒与CIR/EXCESS统计，同时避免长期保存每个请求对象。

        输入：无。

        输出：None；初始化全部累计字段。
        """
        self.count = 0
        self.byte_count = 0
        self.cir_count = 0
        self.excess_count = 0

    def append(self, request):
        """功能：聚合一条已成功提交SSD的QoS请求。

        目的：不改变请求经过Queue、WRR、令牌和SSD反压的路径，只替换最终日志保存。

        输入：
            request: 已完成QoS下发的普通请求字典。

        输出：None；更新请求数、字节数和速率类别计数。
        """
        self.count += 1
        self.byte_count += request["size_bytes"]
        if request["qos_rate_class"] == "CIR":
            self.cir_count += 1
        else:
            self.excess_count += 1

    def __len__(self):
        """功能：返回成功下发请求数。

        目的：保持QoS内部通过len计算dispatch_index的语义不变。

        输入：无。

        输出：
            int: 成功下发请求数量。
        """
        return self.count


def nearest_rank_p95(values):
    """功能：按nearest-rank定义计算P95。

    目的：避免不同统计库的插值规则让多次实验摘要不可直接比较。

    输入：
        values: 非空数值序列。

    输出：
        number: 排序后ceil(0.95*N)-1位置的数值。
    """
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def gpu_utilization_percent(inference):
    """功能：计算一次推理窗口内的GPU模型利用率。

    目的：量化TTFT中GPU计算所占比例；SSD等待越长，该比例越低。

    输入：
        inference: 包含compute_only_ttft_us和ttft_us的推理结果。

    输出：
        float: GPU计算时间占实际TTFT的百分比。
    """
    return (
        inference["compute_only_ttft_us"]
        / inference["ttft_us"]
        * 100
    )


def build_simulation(config, ssd_count, policy):
    """功能：根据统一配置创建一个拓扑与策略实验实例。

    目的：每个实验点都从空Queue、空令牌和空SSD启动，并仅覆盖当前SSD数量与策略。

    输入：
        config: 完整simulation配置字典。
        ssd_count: 本实验点要实例化的独立QoS+SSD数量。
        policy: baseline或demand_aware_fcfs_cir。

    输出：
        JointSimulation: 尚未运行的联合仿真实例。
    """
    run_config = deepcopy(config)
    run_config["topology"]["storage_path_count"] = ssd_count
    return JointSimulation(
        config=run_config,
        binding_strategy_name=run_config["dpu"]["queue_binding"]["strategy"],
        rate_control_strategy_name=policy,
    )


def install_summary_only_logs(simulation):
    """功能：把高频明细日志替换为等价聚合容器。

    目的：完整执行数据通路和SSD时序，只关闭最终摘要不需要的逐请求记录保留。

    输入：
        simulation: 尚未开始运行的JointSimulation实例。

    输出：
        dict: storage_target_id到QoS下发聚合器的映射。
    """
    dispatch_logs = {}
    for storage_target_id, storage_path in simulation.storage_paths.items():
        dispatch_log = DispatchAggregateLog()
        storage_path.qos.dispatched_requests = dispatch_log
        storage_path.ssd.backend.completed_requests = CountOnlyAppendLog()
        storage_path.ssd.backend.nand_service_events = CountOnlyAppendLog()
        dispatch_logs[storage_target_id] = dispatch_log
    return dispatch_logs


def collect_inference_results(simulation):
    """功能：按GPU拓扑顺序展开全部已完成推理。

    目的：统一支持每GPU一次或多次推理，并为层时序和GPU利用率摘要提供输入。

    输入：
        simulation: 所有GPU均已完成的JointSimulation实例。

    输出：
        list: 按GPU、推理序号排列的完整推理结果。
    """
    return [
        inference
        for gpu_id in simulation.gpu_workload_sequences
        for inference in simulation.completed_inference_results[gpu_id]
    ]


def summarize_run(simulation, dispatch_logs, wall_time_seconds):
    """功能：生成一个策略实验点的紧凑性能与守恒摘要。

    目的：报告层读取延迟、TTFT、GPU利用率、SSD尾时刻和CIR/EXCESS数量，
    同时确认GPU、QoS和SSD的请求数与字节数严格守恒。

    输入：
        simulation: 已运行至全部GPU完成的联合仿真实例。
        dispatch_logs: 每块SSD的QoS下发聚合器。
        wall_time_seconds: 本实验点的真实运行耗时。

    输出：
        dict: 当前SSD数量与DPU策略的紧凑摘要。
    """
    inferences = collect_inference_results(simulation)
    signed_deltas_us = []
    actual_reads_us = []
    for inference in inferences:
        initial_read = inference["initial_layer_read"]
        initial_read_us = initial_read["read_time_us"]
        actual_reads_us.append(initial_read_us)
        # 首层没有可重叠的前一层计算窗口，全部读取时间都属于启动缺口。
        signed_deltas_us.append(initial_read_us)
        for layer in inference["layers"]:
            # 末层不预取测量窗口外数据，不能把它的0us读取误当成
            # 一个提前完成的负delta样本。
            if layer["prefetch_layer_index"] is None:
                continue
            layer_start_us = layer["layer_start_time_us"]
            actual_read_us = layer["io_completion_time_us"] - layer_start_us
            target_window_us = layer["compute_done_time_us"] - layer_start_us
            actual_reads_us.append(actual_read_us)
            signed_deltas_us.append(actual_read_us - target_window_us)

    expected_request_count = sum(
        inference["request_count"] for inference in inferences
    )
    expected_completed_count = sum(
        inference["completed_request_count"] for inference in inferences
    )
    qos_request_count = sum(log.count for log in dispatch_logs.values())
    qos_byte_count = sum(log.byte_count for log in dispatch_logs.values())
    ssd_request_count = sum(
        len(path.ssd.backend.completed_requests)
        for path in simulation.storage_paths.values()
    )
    ssd_byte_count = sum(
        path.ssd.backend.completed_bytes()
        for path in simulation.storage_paths.values()
    )
    expected_byte_count = sum(
        inference["request_count"] * inference["block_size_bytes"]
        for inference in inferences
    )
    if len({
        expected_request_count,
        expected_completed_count,
        qos_request_count,
        ssd_request_count,
    }) != 1:
        raise RuntimeError("request conservation failed")
    if len({expected_byte_count, qos_byte_count, ssd_byte_count}) != 1:
        raise RuntimeError("byte conservation failed")

    last_completion_by_ssd_us = {}
    for storage_target_id, path in simulation.storage_paths.items():
        completion_time = path.ssd.backend.last_completion_time
        last_completion_by_ssd_us[storage_target_id] = (
            None if completion_time is None else time_to_us(completion_time)
        )
    completed_ssd_times = [
        value for value in last_completion_by_ssd_us.values()
        if value is not None
    ]

    ttft_values_us = [inference["ttft_us"] for inference in inferences]
    gpu_utilizations_percent = [
        gpu_utilization_percent(inference) for inference in inferences
    ]
    dpu_statistics = simulation.dpu.statistics()
    rate_control = dpu_statistics["rate_control"]
    if rate_control is not None and rate_control["active_demand_count"] != 0:
        raise RuntimeError("demand-aware run ended with active Queue demands")

    expected_inference_count = sum(
        len(sequence)
        for sequence in simulation.gpu_workload_sequences.values()
    )
    if len(inferences) != expected_inference_count:
        raise RuntimeError("inference completion/starvation check failed")

    compact_rate_control = None
    if rate_control is not None:
        compact_rate_control = {
            key: rate_control[key]
            for key in (
                "strategy",
                "ordering",
                "selection_width",
                "active_demand_count",
                "active_p_node_count",
                "completed_coflow_count",
                "selection_change_count",
                "selected_queue_count",
                "max_queue_wait_us",
                "peak_assigned_cir_bytes_per_second",
                "persistent_cohort",
                "finite_selected_pir",
                "min_window_threshold_us",
                "score_mode",
                "deadline_allowance_us",
                "compute_layer_count",
                "coordination_enabled",
                "utility_enabled",
                "edf_enabled",
                "decision_count",
                "initial_decision_count",
                "prefetch_decision_count",
                "feasibility_conflict_count",
            )
            if key in rate_control
        }
        p_node_statistics = rate_control.get("p_node_statistics", {})
        if p_node_statistics:
            expected_coflow_count = len(
                simulation.global_simulation_config["workload"].get(
                    "layer_indexes",
                    range(
                        simulation.global_simulation_config["workload"][
                            "first_layer_index"
                        ],
                        simulation.global_simulation_config["workload"][
                            "last_layer_index"
                        ] + 1,
                    ),
                )
            )
            compact_rate_control["p_node_count"] = len(p_node_statistics)
            compact_rate_control["starved_p_node_count"] = sum(
                profile["completed_coflow_count"] < expected_coflow_count
                for profile in p_node_statistics.values()
            )

    return {
        "late_gpu_layer_count": sum(
            delta_us > 0 for delta_us in signed_deltas_us
        ),
        "p95_delta_us": nearest_rank_p95(signed_deltas_us),
        "worst_delta_us": max(signed_deltas_us),
        "mean_actual_read_us": sum(actual_reads_us) / len(actual_reads_us),
        "mean_ttft_us": sum(ttft_values_us) / len(ttft_values_us),
        "p95_ttft_us": nearest_rank_p95(ttft_values_us),
        "max_ttft_us": max(ttft_values_us),
        "mean_gpu_utilization_percent": (
            sum(gpu_utilizations_percent) / len(gpu_utilizations_percent)
        ),
        "min_gpu_utilization_percent": min(gpu_utilizations_percent),
        "p95_gpu_utilization_percent": nearest_rank_p95(
            gpu_utilizations_percent
        ),
        "max_gpu_utilization_percent": max(gpu_utilizations_percent),
        "completed_inference_count": len(inferences),
        "starvation_free": True,
        "last_completion_by_ssd_us": last_completion_by_ssd_us,
        "overall_last_completion_us": (
            max(completed_ssd_times)
            if completed_ssd_times
            else simulation.start_time_us
        ),
        "request_count": expected_request_count,
        "byte_count": expected_byte_count,
        "cir_dispatch_count": sum(
            log.cir_count for log in dispatch_logs.values()
        ),
        "excess_dispatch_count": sum(
            log.excess_count for log in dispatch_logs.values()
        ),
        "processed_event_count": simulation.event_loop.processed_event_count,
        "rate_control_write_count": dpu_statistics[
            "rate_control_write_count"
        ],
        "queue_weight_write_count": dpu_statistics[
            "queue_weight_write_count"
        ],
        "group_weight_write_count": dpu_statistics[
            "group_weight_write_count"
        ],
        "control_update_tick_aligned_write_count": dpu_statistics[
            "control_update_tick_aligned_write_count"
        ],
        "control_update_non_tick_write_count": dpu_statistics[
            "control_update_non_tick_write_count"
        ],
        "control_update_period_us_by_storage_target": dpu_statistics[
            "control_update_period_us_by_storage_target"
        ],
        "rate_control": compact_rate_control,
        "wall_time_seconds": wall_time_seconds,
    }


def summarize_pair(baseline_summary, demand_aware_summary):
    """功能：比较两种策略的平均GPU利用率。

    目的：用百分点表达Demand-aware相对Baseline的整体GPU利用率变化。

    输入：
        baseline_summary: Baseline实验点摘要。
        demand_aware_summary: Demand-aware FCFS CIR实验点摘要。

    输出：
        dict: Demand-aware减Baseline的平均利用率百分点。
    """
    return {
        "mean_gpu_utilization_gain_percentage_points": (
            demand_aware_summary["mean_gpu_utilization_percent"]
            - baseline_summary["mean_gpu_utilization_percent"]
        ),
    }


def _relative_change_percent(baseline_value, policy_value):
    """功能：计算策略值相对Baseline的有符号百分比变化。

    目的：为Mean、P95和Max TTFT提供方向一致的相对变化；正数
    表示TTFT退化，负数表示TTFT改善。

    输入：
        baseline_value: Baseline指标值。
        policy_value: 待比较策略的指标值。

    输出：
        float | None: ``(policy-baseline)/baseline*100``；Baseline为0时
        返回None，避免在summary.json中写入非标准Infinity。
    """
    if baseline_value == 0:
        return None
    return (policy_value - baseline_value) / baseline_value * 100


def summarize_policy_comparison(baseline_summary, policy_summary):
    """功能：通用比较一个非Baseline策略与Baseline的关键指标。

    目的：直接报告平均GPU利用率是否达到约定目标，并完整保留
    平均、P95、Max TTFT以及最低GPU利用率的变化。

    输入：
        baseline_summary: Baseline实验点的summarize_run输出。
        policy_summary: 待比较策略的summarize_run输出。

    输出：
        dict: 利用率百分点变化、目标与达标状态，以及有符号
        TTFT绝对/百分比变化。TTFT正数变化表示退化。
    """
    baseline_mean_utilization = baseline_summary[
        "mean_gpu_utilization_percent"
    ]
    policy_mean_utilization = policy_summary[
        "mean_gpu_utilization_percent"
    ]
    target_mean_utilization = min(
        baseline_mean_utilization + 25.0,
        99.5,
    )
    comparison = {
        "mean_gpu_utilization_gain_percentage_points": (
            policy_mean_utilization - baseline_mean_utilization
        ),
        "target_mean_gpu_utilization_percent": target_mean_utilization,
        "meets_target": policy_mean_utilization >= target_mean_utilization,
        "min_gpu_utilization_change_percentage_points": (
            policy_summary["min_gpu_utilization_percent"]
            - baseline_summary["min_gpu_utilization_percent"]
        ),
    }
    for statistic in ("mean", "p95", "max"):
        metric_name = f"{statistic}_ttft_us"
        baseline_value = baseline_summary[metric_name]
        policy_value = policy_summary[metric_name]
        comparison[f"{statistic}_ttft_change_us"] = (
            policy_value - baseline_value
        )
        comparison[f"{statistic}_ttft_change_percent"] = (
            _relative_change_percent(baseline_value, policy_value)
        )
    return comparison


def write_summary(summary, output_file):
    """功能：原子写入唯一summary.json检查点。

    目的：长时间扫描中每完成一个策略就保存进度，且不生成CSV或manifest。

    输入：
        summary: 当前已完成实验点的摘要字典。
        output_file: summary.json目标路径。

    输出：None；通过同目录临时文件替换目标JSON。
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_file.replace(output_file)


def run_one(config, ssd_count, policy):
    """功能：运行并汇总一个SSD数量与DPU策略组合。

    目的：各实验点串行执行以限制峰值内存，并打印可观察的长任务进度。

    输入：
        config: 完整simulation配置字典。
        ssd_count: 当前独立QoS+SSD数量。
        policy: 当前DPU速率策略名称。

    输出：
        dict: 当前实验点的紧凑摘要。
    """
    print(f"START ssd_count={ssd_count} policy={policy}", flush=True)
    simulation = build_simulation(config, ssd_count, policy)
    dispatch_logs = install_summary_only_logs(simulation)
    started_at = perf_counter()
    simulation.event_loop.run_until(simulation._all_gpus_complete)
    wall_time_seconds = perf_counter() - started_at
    summary = summarize_run(simulation, dispatch_logs, wall_time_seconds)
    print(
        f"DONE ssd_count={ssd_count} policy={policy} "
        f"wall={wall_time_seconds:.3f}s "
        f"mean_ttft={summary['mean_ttft_us']:.3f}us",
        flush=True,
    )
    return summary


def resolve_output_file(config):
    """功能：解析统一配置中的summary.json目标路径。

    目的：相对路径始终以项目根目录为基准，使入口可从任意工作目录启动。

    输入：
        config: 包含experiment.output_file的完整simulation配置。

    输出：
        Path: 绝对输出路径。
    """
    output_file = Path(config["experiment"]["output_file"])
    if output_file.is_absolute():
        return output_file
    return PROJECT_DIR / output_file


def run_configured_experiment(config):
    """功能：运行统一YAML声明的全部SSD数量和策略。

    目的：用一个入口替代experiments目录中的多个脚本，并在每个实验点后更新
    同一个summary.json。

    输入：
        config: 完整simulation配置字典。

    输出：
        dict: 实验元数据、各拓扑策略摘要、通用comparisons映射
        及兼容旧Demand-aware输出的paired结果。
    """
    experiment = config["experiment"]
    workload = config["workload"]
    generation = config["workload_generation"]
    backend = config["ssd"]["backend"]
    output_file = resolve_output_file(config)
    policies = list(config["dpu"]["rate_control"]["strategies"])
    ssd_counts = list(config["topology"]["ssd_counts"])

    summary = {
        "experiment": {
            "gpu_count": config["topology"]["gpu_count"],
            "inference_count_per_gpu": generation[
                "inference_count_per_gpu"
            ],
            "batch_size": workload["batch_size"],
            "effective_compute_tflops": config["gpu"][
                "effective_compute_tflops"
            ],
            "first_layer_index": workload["first_layer_index"],
            "last_layer_index": workload["last_layer_index"],
            "ssd_counts": ssd_counts,
            "policies": policies,
            "seed": generation["random_seed"],
            "input_tokens_range": generation["input_tokens_range"],
            "hit_ratio_range": generation[
                "prefill_layer_hit_ratio_range"
            ],
            "placement_strategy": workload["placement"]["strategy"],
            "queue_binding_strategy": config["dpu"]["queue_binding"][
                "strategy"
            ],
            "backend_execution_mode": backend["execution_mode"],
            "backend_batch_commands": backend["exact_batch_max_commands"],
        },
        "topologies": {},
    }

    for ssd_count in ssd_counts:
        topology_summary = {}
        summary["topologies"][f"{ssd_count}_ssd"] = topology_summary
        for policy in policies:
            topology_summary[policy] = run_one(
                config,
                ssd_count,
                policy,
            )
            write_summary(summary, output_file)
        if "baseline" in topology_summary:
            topology_summary["comparisons"] = {
                policy: summarize_policy_comparison(
                    topology_summary["baseline"],
                    topology_summary[policy],
                )
                for policy in policies
                if policy != "baseline" and policy in topology_summary
            }
            if "demand_aware_fcfs_cir" in topology_summary:
                topology_summary["paired"] = summarize_pair(
                    topology_summary["baseline"],
                    topology_summary["demand_aware_fcfs_cir"],
                )
            write_summary(summary, output_file)
    return summary


def parse_arguments():
    """功能：解析统一仿真入口的唯一命令行参数。

    目的：所有实验参数保存在一个YAML；命令行只负责选择这份配置文件。

    输入：无；由argparse读取进程命令行。

    输出：
        argparse.Namespace: 包含统一YAML路径。
    """
    parser = argparse.ArgumentParser(
        description="Run the unified multi-GPU, multi-SSD QoS simulation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SIMULATION_CONFIG_FILE,
        help="project-wide simulation YAML",
    )
    return parser.parse_args()


def main():
    """功能：运行统一YAML实验并打印最终summary.json内容。

    目的：作为项目唯一可执行仿真入口，依次完成全部SSD数量和DPU策略组合。

    输入：无；读取parse_arguments返回的统一YAML路径。

    输出：None；写入一个summary.json并在终端打印相同JSON。
    """
    arguments = parse_arguments()
    config = load_simulation_config(arguments.config)
    summary = run_configured_experiment(config)
    write_summary(summary, resolve_output_file(config))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
