"""为每张GPU生成可复现的随机推理工作负载序列。"""

from copy import deepcopy
import hashlib
import random


class UniformRandomInferenceSampler:
    """在配置区间内均匀采样输入Token数和KV Cache命中率。"""

    def __init__(self, config):
        """功能：保存随机推理生成配置。

        目的：将工作负载随机性与KV Placement和DPU Queue随机性隔离。

        输入：
            config: 项目级YAML中 ``workload_generation`` 节点的字典。

        输出：
            None: 初始化采样范围、次数、间隔和随机种子。
        """
        self.inference_count_per_gpu = config["inference_count_per_gpu"]
        self.random_seed = config["random_seed"]
        self.input_tokens_range = tuple(config["input_tokens_range"])
        self.hit_ratio_range = tuple(
            config["prefill_layer_hit_ratio_range"]
        )
        self.inter_inference_gap_us = config.get("inter_inference_gap_us", 0)
        self.unique_across_gpus = config.get("unique_across_gpus", False)

    def sample(self, base_workload, gpu_id, inference_index):
        """功能：为一张GPU的一次推理生成完整工作负载。

        目的：使用 ``(全局种子, GPU ID, 推理序号)`` 派生独立
        随机源，保证事件完成顺序或DPU策略变化时仍使用相同负载。

        输入：
            base_workload: 已合并默认值和GPU局部覆盖的工作负载模板。
            gpu_id: 当前GPU的唯一编号。
            inference_index: 这张GPU上从0开始的推理序号。

        输出：
            dict: 已写入唯一ID、随机Token数和随机命中率的完整工作负载。
        """
        identity = f"{self.random_seed}\x1f{gpu_id}\x1f{inference_index}"
        digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, byteorder="big"))

        workload = deepcopy(base_workload)
        workload["workload_id"] = (
            f"{base_workload['workload_id']}_inference_{inference_index:05d}"
        )
        workload["input_tokens"] = rng.randint(*self.input_tokens_range)
        workload["prefill_layer_hit_ratio"] = rng.uniform(
            *self.hit_ratio_range
        )
        return workload

    def build_sequence(self, base_workload, gpu_id):
        """功能：生成一张GPU的全部随机推理配置。

        目的：在事件循环开始前固定完整序列，避免运行时调度顺序
        影响后续随机样本，使策略对比保持公平。

        输入：
            base_workload: 当前GPU的完整工作负载模板。
            gpu_id: 当前GPU的唯一编号。

        输出：
            list[dict]: 按推理序号排列的可复现工作负载列表。
        """
        return [
            self.sample(base_workload, gpu_id, inference_index)
            for inference_index in range(self.inference_count_per_gpu)
        ]

    def build_sequences(self, base_workloads_by_gpu):
        """功能：为全部GPU一次性生成可复现的推理序列。

        目的：普通模式保留每GPU独立采样；实验模式开启
        ``unique_across_gpus`` 后，在每个推理序号上为GPU不放回
        采样input_tokens，并保证hit ratio不重复。完整序列在
        事件循环启动前固定，因此QoS策略不会改变工作负载。

        输入：
            base_workloads_by_gpu: ``gpu_id -> 完整工作负载模板`` 映射。

        输出：
            dict: ``gpu_id -> 按推理序号排列的工作负载列表``。
        """
        if not self.unique_across_gpus:
            return {
                gpu_id: self.build_sequence(base_workload, gpu_id)
                for gpu_id, base_workload in base_workloads_by_gpu.items()
            }

        gpu_ids = list(base_workloads_by_gpu)
        sequences = {gpu_id: [] for gpu_id in gpu_ids}
        token_low, token_high = self.input_tokens_range
        hit_low, hit_high = self.hit_ratio_range

        for inference_index in range(self.inference_count_per_gpu):
            # 唯一采样使用独立的全局随机源，不依赖GPU完成顺序。
            identity = (
                f"{self.random_seed}\x1funique-across-gpus\x1f"
                f"{inference_index}"
            )
            digest = hashlib.blake2b(
                identity.encode("utf-8"),
                digest_size=8,
            ).digest()
            rng = random.Random(int.from_bytes(digest, byteorder="big"))
            input_tokens = rng.sample(
                range(token_low, token_high + 1),
                len(gpu_ids),
            )

            hit_ratios = []
            used_hit_ratios = set()
            while len(hit_ratios) < len(gpu_ids):
                hit_ratio = rng.uniform(hit_low, hit_high)
                if hit_ratio not in used_hit_ratios:
                    used_hit_ratios.add(hit_ratio)
                    hit_ratios.append(hit_ratio)

            for gpu_id, token_count, hit_ratio in zip(
                gpu_ids,
                input_tokens,
                hit_ratios,
            ):
                workload = deepcopy(base_workloads_by_gpu[gpu_id])
                workload["workload_id"] = (
                    f"{workload['workload_id']}_inference_"
                    f"{inference_index:05d}"
                )
                workload["input_tokens"] = token_count
                workload["prefill_layer_hit_ratio"] = hit_ratio
                sequences[gpu_id].append(workload)

        return sequences
