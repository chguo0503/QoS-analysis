"""跨StoragePath控制写入的全局事件唤醒回归测试。"""

import unittest

from backends.asu_ssd.time_utils import time_to_us
from discrete_simulation import EventLoop
from qos import build_qos_simulator
from simulation_common.storage_path import StoragePath


def build_minimal_qos():
    """创建两条Queue、80 us周期且初始PIR不封顶的真实QoS。"""
    return build_qos_simulator(
        qos_config={
            "queue_layout": {
                "queue_count": 2,
                "queue_id_prefix": "q",
                "queue_id_width": 3,
                "group_count": 1,
                "queues_per_group": 2,
                "group_id_prefix": "g",
            },
            "token_bucket": {
                "update_period_us": 80,
                "initial_state": "empty",
                "queue_max_io_size_bytes": 4,
                "queue_cbs_bytes": 4,
                "queue_pbs_bytes": 4,
                "group_rates": [{"group_id": "g0", "cir_gb_s": 0}],
                "queue_cir_weight_bitmap": [1, 1],
                "queue_default_pir_gb_s": "uncapped",
                "queue_overrides": {},
            },
            "scheduler": {
                "group_weight_bitmap": [1],
                "queue_weight_bitmaps": {"g0": [1, 1]},
            },
            "runtime": {
                "same_timestamp_event_order": [
                    "token_refill",
                    "rate_update",
                    "io_arrival",
                    "scheduler_dispatch",
                ],
            },
        },
        start_time_us=0,
    )


class RecordingSSD:
    """始终可接收且没有内部事件的最小非阻塞SSD。"""

    def __init__(self):
        self.accepted_requests = []

    def can_accept_at_us(self, _event_time_us):
        return True

    def try_input_at_us(self, request, requested_time_us):
        self.accepted_requests.append((dict(request), requested_time_us))
        return {
            "accepted": True,
            "accepted_time_us": requested_time_us,
        }

    def next_event_time(self):
        return None


def make_request(request_id, target, queue_id, arrival_time_us):
    """构造StoragePath可直接登记的最小完整QoS请求。"""
    return {
        "request_id": request_id,
        "p_node_id": request_id,
        "storage_target_id": target,
        "queue_id": queue_id,
        "size_bytes": 4,
        "arrival_time_us": arrival_time_us,
    }


class ControlObserverTests(unittest.TestCase):
    """验证三类控制heap只在最早时刻提前时通知。"""

    def test_rate_group_and_queue_weight_share_one_earliest_notification(self):
        qos = build_minimal_qos()
        observed_times = []
        qos.set_control_event_observer(
            lambda event_time_us: observed_times.append(event_time_us)
        )

        qos.schedule_queue_rate_update("q000", 0, None, 100)
        qos.schedule_group_weight_update({"g0": 1}, 90)
        qos.schedule_queue_weight_update({"q000": 0}, 80)
        # 同一最早时刻的覆盖与更晚事件都不能产生重复通知。
        qos.schedule_queue_rate_update("q001", 0, 0, 80)
        qos.schedule_queue_weight_update({"q001": 0}, 120)

        self.assertEqual(observed_times, [100, 90, 80])


class CrossStorageControlWakeupTests(unittest.TestCase):
    """验证一条StoragePath回调能同时刻唤醒另一条路径。"""

    def test_cross_path_unlock_preempts_existing_80us_token_event(self):
        event_loop = EventLoop()
        qos0 = build_minimal_qos()
        qos1 = build_minimal_qos()
        ssd0 = RecordingSSD()
        ssd1 = RecordingSSD()
        path0 = StoragePath("SSD0", qos0, ssd0, event_loop)
        path1 = StoragePath("SSD1", qos1, ssd1, event_loop)

        # SSD1的请求在0 us进入FIFO，但PIR=0且weight=0，第一次QoS回调
        # 后只能留下一个自然的80 us token事件。
        qos1.schedule_queue_rate_update("q001", 0, 0, 0)
        qos1.schedule_queue_weight_update({"q001": 0}, 0)
        path1.input(make_request("waiter", "SSD1", "q001", 0))

        # SSD0在10 us排空时模拟DPU跨盘晋升waiter。两个控制写入发生在
        # 同一个QoS回调内，目标StoragePath只能补排一个10 us事件。
        def unlock_other_path(event_time_us):
            qos1.schedule_queue_rate_update(
                "q001",
                0,
                None,
                event_time_us,
            )
            qos1.schedule_queue_weight_update(
                {"q001": 1},
                event_time_us,
            )

        qos0.set_queue_state_observer(unlock_other_path)
        path0.input(make_request("trigger", "SSD0", "q000", 10))

        processed_events = []
        while event_loop.events:
            processed_events.append(event_loop.run_next_event())

        self.assertEqual(
            processed_events,
            ["qos:SSD1", "qos:SSD0", "qos:SSD1", "qos:SSD1"],
        )
        self.assertEqual(
            [request[0]["request_id"] for request in ssd1.accepted_requests],
            ["waiter"],
        )
        self.assertEqual(ssd1.accepted_requests[0][1], 10)
        self.assertEqual(qos1.current_time_us, 80)
        self.assertEqual(qos1.queue_io_counts()["q001"], 0)
        self.assertEqual(time_to_us(event_loop.current_time), 80)


if __name__ == "__main__":
    unittest.main()
