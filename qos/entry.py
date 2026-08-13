"""QoS的公共装配入口。

项目外部只需从 ``qos`` 导入 ``build_qos_simulator`` 和
``load_queue_layout``。令牌桶、调度器和事件引擎的连接都在这里完成。
"""

from discrete_simulation.simulator import DiscreteEventSimulator
from .queue_layout import QueueLayout
from .schedulers import HierarchicalScheduler
from .token_bucket_stage import PerQueueTokenBucketStage
from simulation_common.config_utils import load_yaml


def load_queue_layout(layout_config_file):
    """功能：读取Queue布局YAML并创建固定队列/分组对象。

    目的：让DPU绑定策略和每个QoS实例共用相同的Queue ID与Group映射。

    输入：
        layout_config_file: 包含 ``queue_layout`` 节点的YAML文件路径。

    输出：
        QueueLayout: 已校验和展开的Queue布局对象。
    """
    layout_config = load_yaml(layout_config_file)["queue_layout"]
    return QueueLayout(layout_config)


def build_qos_simulator(
    layout_config_file,
    token_config_file,
    scheduler_config_file,
    qos_runtime_config_file,
    start_time_us,
    queue_layout=None,
):
    """功能：装配一套独立QoS的Queue、令牌桶、调度器和事件组件。

    目的：使QoS只读取自身运行参数，由顶层显式传入全局起始时刻，
    从而不依赖GPU/StoragePath拓扑配置的文件位置。

    输入：
        layout_config_file: Queue数量和Group布局YAML路径。
        token_config_file: 每Queue CIR/PIR令牌参数YAML路径。
        scheduler_config_file: 分层WRR调度参数YAML路径。
        qos_runtime_config_file: 单个QoS实例的同时间戳事件顺序YAML路径。
        start_time_us: 顶层全局仿真传入的起始微秒时刻。
        queue_layout: 可选的已展开QueueLayout，用于多个StoragePath复用。

    输出：
        DiscreteEventSimulator: 状态独立、尚未连接SSD的QoS仿真器。
    """
    if queue_layout is None:
        queue_layout = load_queue_layout(layout_config_file)
    # 联合仿真可以传入同一个layout对象，保证DPU和QoS使用完全相同的队列映射。

    # 布局已在上方处理；这里再分别读令牌桶、调度和QoS运行文档。
    token_config = load_yaml(token_config_file)["token_bucket"]
    scheduler_config = load_yaml(scheduler_config_file)["scheduler"]
    qos_runtime_config = load_yaml(qos_runtime_config_file)["qos_runtime"]

    # DiscreteEventSimulator仍使用一个局部字典，但它由顶层时钟与
    # QoS自身事件顺序组合而成，不再携带任何GPU/SSD拓扑字段。
    simulation_config = {
        "start_time_us": start_time_us,
        "same_timestamp_event_order": list(
            qos_runtime_config["same_timestamp_event_order"]
        ),
    }

    # 先组装会影响请求下发的数据通路。
    token_stage = PerQueueTokenBucketStage(queue_layout, token_config)
    scheduler = HierarchicalScheduler(queue_layout, scheduler_config)
    simulator = DiscreteEventSimulator(
        token_stage=token_stage,
        scheduler=scheduler,
        simulation_config=simulation_config,
    )
    # 把已展开的队列布局一并交给调用方，便于查看queue_id与group_id的关系。
    simulator.queue_layout = queue_layout
    return simulator
