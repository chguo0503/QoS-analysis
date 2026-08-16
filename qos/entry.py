"""QoS的公共装配入口。

项目外部只需从 ``qos`` 导入 ``build_qos_simulator`` 和
``build_queue_layout``。令牌桶、调度器和事件引擎的连接都在这里完成。
"""

from discrete_simulation.simulator import DiscreteEventSimulator
from .queue_layout import QueueLayout
from .schedulers import HierarchicalScheduler
from .token_bucket_stage import PerQueueTokenBucketStage


def build_queue_layout(layout_config):
    """功能：根据统一配置创建固定Queue与Group布局。

    目的：让DPU绑定策略和每个QoS实例共用相同的Queue ID与Group映射。

    输入：
        layout_config: ``simulation.qos.queue_layout`` 配置字典。

    输出：
        QueueLayout: 已校验和展开的Queue布局对象。
    """
    return QueueLayout(layout_config)


def build_qos_simulator(
    qos_config,
    start_time_us,
    queue_layout=None,
):
    """功能：装配一套独立QoS的Queue、令牌桶、调度器和事件组件。

    目的：由顶层把统一YAML中的QoS分区直接传入组件，避免QoS再次读取
    Queue、令牌、WRR和运行时等分散配置文件。

    输入：
        qos_config: ``simulation.qos`` 完整配置字典。
        start_time_us: 顶层全局仿真传入的起始微秒时刻。
        queue_layout: 可选的已展开QueueLayout，用于多个StoragePath复用。

    输出：
        DiscreteEventSimulator: 状态独立、尚未连接SSD的QoS仿真器。
    """
    if queue_layout is None:
        queue_layout = build_queue_layout(qos_config["queue_layout"])
    # 多块SSD复用不可变布局，但每块SSD仍创建独立的令牌、Queue和调度器状态。
    token_config = qos_config["token_bucket"]
    scheduler_config = qos_config["scheduler"]
    qos_runtime_config = qos_config["runtime"]

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
