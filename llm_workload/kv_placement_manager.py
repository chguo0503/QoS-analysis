#!/usr/bin/env python3
"""把LLM层读取计划转换成带SSD目标和聚合需求的DPU请求。"""

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import sys


# 直接运行本文件时加入项目根目录，使包导入和脚本运行使用同一套路径。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_workload.placement_strategies import build_placement_strategy  # noqa: E402
from simulation_common.config_utils import load_yaml  # noqa: E402


CONFIG_DIR = Path(__file__).resolve().parent / "config"
KV_PLACEMENT_CONFIG_FILE = CONFIG_DIR / "kv_placement.yaml"


def load_kv_placement_config(config_file=KV_PLACEMENT_CONFIG_FILE):
    """功能：读取KV Placement独立演示配置。

    目的：让示例所需的SSD数量和随机种子保存在YAML中；联合仿真的真实SSD
    列表仍由全局simulation配置动态生成。

    输入：
        config_file: KV Placement YAML文件路径。

    输出：
        dict: YAML中 ``kv_placement`` 节点对应的配置字典。
    """
    return load_yaml(config_file)["kv_placement"]


_CONFIG = load_kv_placement_config()
DEMO_RANDOM_SEED = _CONFIG["demo_random_seed"]
DEMO_SSD_COUNTS = tuple(_CONFIG["demo_ssd_counts"])


def _resolve_allowed_targets(storage_target_ids, allowed_storage_targets):
    """功能：解析某张GPU实际允许访问的SSD集合。

    目的：支持YAML中的 ``all`` 简写和显式SSD子集。

    输入：
        storage_target_ids: 全局拓扑创建的全部SSD ID。
        allowed_storage_targets: 字符串 ``all`` 或显式SSD ID列表。

    输出：
        list[str]: 保持全局拓扑顺序的合法、非空SSD ID列表。
    """
    all_targets = list(storage_target_ids)
    if allowed_storage_targets == "all":
        return all_targets

    # 按全局拓扑顺序返回，而不是使用集合顺序，确保重复仿真完全可复现。
    requested_set = set(allowed_storage_targets)
    return [target_id for target_id in all_targets if target_id in requested_set]


class KVPlacementManager:
    """通过可替换策略放置KV Block，并按SSD计算层级聚合需求。"""

    def __init__(self, storage_target_ids, placement_config):
        """功能：创建一张GPU独立使用的KV Placement Manager。

        目的：为该GPU固定可访问SSD集合和放置策略，避免不同GPU共享有状态
        随机源，同时让未来策略能够独立保存自己的运行状态。

        输入：
            storage_target_ids: 全局仿真中实际创建的全部SSD ID。
            placement_config: LLM workload中的策略、允许目标和随机种子配置。

        输出：
            None: 初始化目标列表、放置策略和Block映射记录。
        """
        self.storage_target_ids = _resolve_allowed_targets(
            storage_target_ids,
            placement_config.get("allowed_storage_targets", "all"),
        )
        self.strategy = build_placement_strategy(
            strategy_name=placement_config["strategy"],
            random_seed=placement_config["random_seed"],
        )
        self.block_to_storage_target = {}

    def _storage_target_for(self, block):
        """功能：返回Block已经确定或首次选择的SSD目标。

        目的：保证即使未来同一个逻辑Block被多次请求，也不会在重试或重读时
        改变物理位置；首次选择的具体算法由放置策略负责。

        输入：
            block: 至少包含唯一 ``request_id`` 的LLM Block字典。

        输出：
            str: 当前Block稳定使用的 ``storage_target_id``。
        """
        request_id = block["request_id"]
        if request_id not in self.block_to_storage_target:
            self.block_to_storage_target[request_id] = self.strategy.select_target(
                block=block,
                storage_target_ids=self.storage_target_ids,
            )
        return self.block_to_storage_target[request_id]

    def build_requests(self, layer_read_plan):
        """功能：放置一层全部Block并生成DPU请求。

        目的：先按SSD聚合同一GPU、同一层的总读取字节，再把对应聚合带宽诉求
        写入每个Block请求；DPU只需消费结果，不需要理解层级聚合过程。

        输入：
            layer_read_plan: LLMWorkload生成的单层Block计划和GPU服务窗口。

        输出：
            list[dict]: 与输入Block一一对应的 ``basic``/``demand_bw`` 请求列表。
        """
        placed_blocks = []
        bytes_by_target = Counter()

        # 第一遍只完成放置和字节累计。必须先知道某个SSD在这一层的全部Block，
        # 才能为每个请求写入完整而不是逐步增长的聚合Byte/s诉求。
        for block in layer_read_plan["blocks"]:
            storage_target_id = self._storage_target_for(block)
            placed_blocks.append((block, storage_target_id))
            bytes_by_target[storage_target_id] += block["size_bytes"]

        service_window_us = layer_read_plan["service_window_us"]
        # 请求格式向DPU传递整数Byte/s。向上取整避免一份
        # 非零需求在换算时被截断为0，DPU内部不再需要浮点运算。
        aggregate_demand_by_target = {
            storage_target_id: math.ceil(
                byte_count * 1_000_000 / service_window_us
            )
            for storage_target_id, byte_count in bytes_by_target.items()
        }

        requests = []
        for block, storage_target_id in placed_blocks:
            requests.append({
                "basic": {
                    "request_id": block["request_id"],
                    "p_node_id": layer_read_plan["p_node_id"],
                    "storage_target_id": storage_target_id,
                    "size_bytes": block["size_bytes"],
                },
                "demand_bw": {
                    "demand_group_id": layer_read_plan["demand_group_id"],
                    # 同一需求组的Block重复携带同一整数聚合值；
                    # DPU按demand_group_id只预留一次，不逐IO相加。
                    "aggregate_required_bytes_per_second": (
                        aggregate_demand_by_target[storage_target_id]
                    ),
                },
            })
        return requests


def _build_demo_layer_read_plan():
    """功能：生成KV Placement独立演示使用的一层读取计划。

    目的：通过真实LLMWorkload验证100K输入、99%命中率和Batch=1时的放置结果。

    输入：
        无。

    输出：
        dict: 只包含第一层的中立LLM读取计划。
    """
    from llm_workload.layer_request import DEFAULT_WORKLOAD, LLMWorkload

    workload = deepcopy(DEFAULT_WORKLOAD)
    workload["input_tokens"] = 100_000
    workload["batch_size"] = 1
    workload["prefill_layer_hit_ratio"] = 0.99
    workload["last_layer_index"] = workload["first_layer_index"]
    return LLMWorkload(workload=workload).start_next_layer()


def _summarize_requests(requests, storage_target_ids):
    """功能：按SSD整理演示请求数和不重复的聚合带宽诉求。

    目的：证明同一SSD上的多个Block已经在Placement阶段聚合，而不是让打印
    代码错误地把每个Block重复携带的聚合值再次相加。

    输入：
        requests: KVPlacementManager生成的DPU请求列表。
        storage_target_ids: 演示中创建的全部SSD ID。

    输出：
        dict: 每个SSD的请求数和一次性聚合带宽诉求。
    """
    request_counts = Counter(
        request["basic"]["storage_target_id"] for request in requests
    )
    demand_by_target = {}
    for request in requests:
        storage_target_id = request["basic"]["storage_target_id"]
        demand_by_target[storage_target_id] = request["demand_bw"][
            "aggregate_required_bytes_per_second"
        ]
    return {
        storage_target_id: {
            "request_count": request_counts[storage_target_id],
            "aggregate_required_bytes_per_second": demand_by_target.get(
                storage_target_id,
                0,
            ),
        }
        for storage_target_id in storage_target_ids
    }


def main():
    """功能：运行1、5、10个SSD的稳定随机放置演示。

    目的：在不启动DPU、QoS和SSD流水线的情况下快速检查Block分布与需求聚合。

    输入：
        无；使用本模块YAML中的演示参数。

    输出：
        None: 将每个SSD的Block数量和聚合需求打印到标准输出。
    """
    layer_read_plan = _build_demo_layer_read_plan()
    print("100K输入、99%命中率、Batch=1的KV Block放置示例")
    print(
        f"上层总请求={len(layer_read_plan['blocks'])}，"
        f"共同GPU窗口={layer_read_plan['service_window_us'] / 1000:.3f} ms"
    )

    for ssd_count in DEMO_SSD_COUNTS:
        storage_target_ids = [f"SSD{index}" for index in range(ssd_count)]
        manager = KVPlacementManager(
            storage_target_ids=storage_target_ids,
            placement_config={
                "strategy": "random",
                "allowed_storage_targets": "all",
                "random_seed": DEMO_RANDOM_SEED,
            },
        )
        requests = manager.build_requests(deepcopy(layer_read_plan))
        summary = _summarize_requests(requests, storage_target_ids)
        total_demand = sum(
            item["aggregate_required_bytes_per_second"]
            for item in summary.values()
        )

        print(f"\n下层SSD数量={ssd_count}，输出Block请求={len(requests)}")
        for storage_target_id, item in summary.items():
            print(
                f"  {storage_target_id}: Block={item['request_count']:>3d}，"
                "聚合需求="
                f"{item['aggregate_required_bytes_per_second']:,} Byte/s"
            )
        print(f"  全部SSD聚合需求之和={total_demand:,} Byte/s")


if __name__ == "__main__":
    main()
