"""QoS公共入口；外部代码只需从这里导入装配函数。"""

from .entry import build_qos_simulator, build_queue_layout


__all__ = ["build_qos_simulator", "build_queue_layout"]
