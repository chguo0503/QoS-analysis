"""Utility+EDF严格80 us控制边界验收测试。

所有Utility控制命令只在80 us边界生效；等待下一层的Queue保持parked；
同一tick以最后写入为准；只有第四个读组完成后才恢复baseline默认状态。
"""

from types import SimpleNamespace
import unittest

from DPU import DPURequestGateway, UtilityEDFController
from qos import build_qos_simulator


CONTROL_PERIOD_US = 80
CAPACITY_BYTES_PER_SECOND = 1_000_000


def build_minimal_qos():
    """创建2条Queue、1个Group和80 us周期的真实QoS。"""
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
                "update_period_us": CONTROL_PERIOD_US,
                "initial_state": "empty",
                "queue_max_io_size_bytes": 100,
                "queue_cbs_bytes": 100,
                "queue_pbs_bytes": 100,
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


class _FixedBinding:
    """以(p_node, SSD)固定绑定互不冲突的Queue。"""

    strategy_name = "fixed"

    def __init__(self, bindings):
        self.bindings = dict(bindings)

    def select_queue(self, request, queue_ids):
        basic = request["basic"]
        queue_id = self.bindings[(
            basic["p_node_id"],
            basic["storage_target_id"],
        )]
        if queue_id not in queue_ids:
            raise KeyError(queue_id)
        return queue_id


class _RecordingQoS:
    """记录控制命令和逻辑depth的最小DPU硬件边界。"""

    def __init__(self, queue_ids):
        self.token_stage = SimpleNamespace(
            update_period_us=CONTROL_PERIOD_US,
        )
        self.depths = {queue_id: 0 for queue_id in queue_ids}
        self.rate_updates = []
        self.queue_weight_updates = []
        self.group_weight_updates = []
        self.observer = None

    def set_queue_state_observer(self, observer):
        self.observer = observer

    def queue_io_counts(self):
        return dict(self.depths)

    def input(self, request):
        self.depths[request["queue_id"]] += 1
        return request

    def schedule_queue_rate_update(
        self,
        queue_id,
        cir_fill,
        pir_fill,
        effective_time_us,
    ):
        self.rate_updates.append((
            queue_id,
            cir_fill,
            pir_fill,
            effective_time_us,
        ))

    def schedule_queue_weight_update(self, weights, effective_time_us):
        self.queue_weight_updates.append((
            dict(weights),
            effective_time_us,
        ))

    def schedule_group_weight_update(self, weights, effective_time_us):
        self.group_weight_updates.append((
            dict(weights),
            effective_time_us,
        ))


class _BudgetBackend:
    """按时刻给定接收名额，用于制造非边界Queue-empty。"""

    def __init__(self):
        self.budget_by_time = {}
        self.accepted = []

    def set_budget(self, event_time_us, count):
        self.budget_by_time[event_time_us] = count

    def can_accept_at_us(self, event_time_us):
        return self.budget_by_time.get(event_time_us, 0) > 0

    def try_input_at_us(self, request, requested_time_us):
        if not self.can_accept_at_us(requested_time_us):
            return {"accepted": False}
        self.budget_by_time[requested_time_us] -= 1
        self.accepted.append((dict(request), requested_time_us))
        return {
            "accepted": True,
            "accepted_time_us": requested_time_us,
        }


def make_request(
    request_id,
    p_node_id,
    demand_group_id,
    size_bytes,
    *,
    arrival_time_us,
    compute_layer_index=None,
    prefetch_layer_index=0,
    service_window_us=100,
):
    """构造带Utility+EDF必需元数据的单Block请求。"""
    return {
        "basic": {
            "request_id": request_id,
            "p_node_id": p_node_id,
            "storage_target_id": "SSD0",
            "size_bytes": size_bytes,
        },
        "demand_bw": {
            "demand_group_id": demand_group_id,
            "compute_layer_index": compute_layer_index,
            "prefetch_layer_index": prefetch_layer_index,
            "inference_arrival_time_us": 7,
            "service_window_us": service_window_us,
            "deadline_us": arrival_time_us + service_window_us,
            "aggregate_bytes_on_storage_target": size_bytes,
            "aggregate_required_bytes_per_second": (
                size_bytes * 1_000_000 // service_window_us
            ),
        },
    }


def assert_all_control_times_are_tick_aligned(test_case, qos):
    """断言三类控制记录中没有非80 us生效时刻。"""
    effective_times = [update[3] for update in qos.rate_updates]
    effective_times.extend(
        event_time_us
        for _, event_time_us in qos.queue_weight_updates
    )
    effective_times.extend(
        event_time_us
        for _, event_time_us in qos.group_weight_updates
    )
    test_case.assertTrue(effective_times)
    test_case.assertTrue(all(
        event_time_us % CONTROL_PERIOD_US == 0
        for event_time_us in effective_times
    ), effective_times)


def last_rate_state(qos, queue_id):
    """返回Queue最后一次(CIR fill, PIR fill, time)写入。"""
    return next(
        (cir, pir, event_time_us)
        for current_queue_id, cir, pir, event_time_us
        in reversed(qos.rate_updates)
        if current_queue_id == queue_id
    )


def last_weight_state(qos, queue_id):
    """返回Queue最后一次(weight, time)写入。"""
    return next(
        (weights[queue_id], event_time_us)
        for weights, event_time_us in reversed(qos.queue_weight_updates)
        if queue_id in weights
    )


class StrictTickEventEngineTests(unittest.TestCase):
    """验证QoS在同tick覆盖与边界前不泄漏的基础语义。"""

    def test_same_tick_last_state_wins_and_park_blocks_dispatch(self):
        qos = build_minimal_qos()

        # t=0预先park，使7 us到达的IO不能经由默认PIR=uncapped
        # 在首个控制边界前泄漏。
        qos.schedule_queue_rate_update("q000", 0, 0, 0)
        qos.schedule_queue_weight_update({"q000": 0}, 0)
        qos.process_at(0)
        qos.input({
            "request_id": "parked",
            "p_node_id": "P0",
            "storage_target_id": "SSD0",
            "queue_id": "q000",
            "size_bytes": 1,
            "arrival_time_us": 7,
        })

        # 同一tick先开后关；实际应用的必须是最后的park。
        qos.schedule_queue_rate_update("q000", 80, None, 80)
        qos.schedule_queue_weight_update({"q000": 1}, 80)
        qos.schedule_queue_rate_update("q000", 0, 0, 80)
        qos.schedule_queue_weight_update({"q000": 0}, 80)

        qos.process_at(79)
        self.assertEqual(qos.dispatched_requests, [])
        qos.process_at(80)
        self.assertEqual(qos.dispatched_requests, [])
        queue_controller = qos.token_stage.controllers["q000"]
        self.assertEqual(queue_controller.cir_bucket.fill_per_tick, 0)
        self.assertIsNotNone(queue_controller.pir_bucket)
        self.assertEqual(queue_controller.pir_bucket.fill_per_tick, 0)
        self.assertEqual(
            qos.scheduler.queue_schedulers["g0"].weights["q000"],
            0,
        )

        qos.schedule_queue_rate_update("q000", 80, None, 160)
        qos.schedule_queue_weight_update({"q000": 1}, 160)
        qos.process_at(160)
        self.assertEqual(
            [request["request_id"] for request in qos.dispatched_requests],
            ["parked"],
        )


class StrictUtilityIntegrationTests(unittest.TestCase):
    """验证Utility端到端只在80 us边界改变Gate。"""

    def test_prepark_uses_bound_queues_and_leaves_unused_queues_default(self):
        """有预绑定时只park DPU实际管理的Queue。"""
        qos = _RecordingQoS(["q000", "q001", "q002", "q003"])
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            deadline_allowance_us=750,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000", "q001", "q002", "q003"]},
            _FixedBinding({
                ("P0", "SSD0"): "q000",
                ("P1", "SSD0"): "q001",
            }),
            qos.input,
            {"SSD0": qos},
            controller,
        )

        self.assertEqual(
            controller.managed_queue_ids["SSD0"],
            {"q000", "q001"},
        )
        self.assertEqual(
            set(controller._programmed_queue_states["SSD0"]),
            {"q000", "q001"},
        )
        self.assertEqual(
            set(queue_id for _, queue_id in gateway.queue_rate_settings),
            {"q000", "q001"},
        )
        self.assertNotIn(("SSD0", "q002"), gateway.queue_weight_settings)
        self.assertNotIn(("SSD0", "q003"), gateway.queue_weight_settings)

    def test_arrival_tick_is_inclusive_but_empty_tick_is_strict_next(self):
        """GPU在rate_update前可用当前tick；dispatch回调必须等下一tick。"""
        qos = _RecordingQoS(["q000"])
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            deadline_allowance_us=750,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000"]},
            _FixedBinding({("P0", "SSD0"): "q000"}),
            qos.input,
            {"SSD0": qos},
            controller,
        )

        before_arrival = len(qos.rate_updates)
        gateway.submit_batch([
            make_request(
                "p0_group_0",
                "P0",
                "p0_group_0",
                1,
                arrival_time_us=80,
            ),
        ], arrival_time_us=80)
        self.assertEqual(
            {update[3] for update in qos.rate_updates[before_arrival:]},
            {80},
        )

        before_empty = len(qos.rate_updates)
        qos.depths["q000"] = 0
        gateway.on_qos_queue_state_change("SSD0", event_time_us=80)
        self.assertEqual(
            {update[3] for update in qos.rate_updates[before_empty:]},
            {160},
        )
        self.assertEqual(gateway.group_weight_write_count, 0)
        self.assertEqual(gateway.control_update_non_tick_write_count, 0)
        self.assertGreater(
            gateway.control_update_tick_aligned_write_count,
            0,
        )

    def test_all_utility_control_effective_times_are_tick_aligned(self):
        qos = _RecordingQoS(["q000", "q001"])
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=4,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000", "q001"]},
            _FixedBinding({
                ("P0", "SSD0"): "q000",
                ("P1", "SSD0"): "q001",
            }),
            qos.input,
            {"SSD0": qos},
            controller,
        )
        initial_rate_count = len(qos.rate_updates)
        initial_weight_count = len(qos.queue_weight_updates)
        gateway.submit_batch([
            make_request(
                "p0",
                "P0",
                "p0_group_0",
                1,
                arrival_time_us=7,
            ),
            make_request(
                "p1",
                "P1",
                "p1_group_0",
                100,
                arrival_time_us=7,
            ),
        ], arrival_time_us=7)
        arrival_rate_updates = qos.rate_updates[initial_rate_count:]
        arrival_weight_updates = qos.queue_weight_updates[
            initial_weight_count:
        ]
        self.assertTrue(arrival_rate_updates)
        self.assertTrue(arrival_weight_updates)
        self.assertEqual(
            {update[3] for update in arrival_rate_updates},
            {80},
        )
        self.assertEqual(
            {event_time_us for _, event_time_us in arrival_weight_updates},
            {80},
        )

        # 在非边界排空当前owner，迫使等待Queue产生第二轮Gate写入。
        before_empty_rate_count = len(qos.rate_updates)
        before_empty_weight_count = len(qos.queue_weight_updates)
        qos.depths["q000"] = 0
        gateway.on_qos_queue_state_change(
            "SSD0",
            event_time_us=91,
        )
        empty_rate_updates = qos.rate_updates[before_empty_rate_count:]
        empty_weight_updates = qos.queue_weight_updates[
            before_empty_weight_count:
        ]
        self.assertTrue(empty_rate_updates)
        self.assertTrue(empty_weight_updates)
        self.assertEqual(
            {update[3] for update in empty_rate_updates},
            {160},
        )
        self.assertEqual(
            {event_time_us for _, event_time_us in empty_weight_updates},
            {160},
        )

        assert_all_control_times_are_tick_aligned(self, qos)
        self.assertEqual(gateway.group_weight_write_count, 0)
        self.assertEqual(qos.group_weight_updates, [])

    def test_non_boundary_arrival_and_empty_do_not_leak_dispatch(self):
        qos = build_minimal_qos()
        backend = _BudgetBackend()
        qos.set_backend(backend)
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=4,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000", "q001"]},
            _FixedBinding({
                ("P0", "SSD0"): "q000",
                ("P1", "SSD0"): "q001",
            }),
            qos.input,
            {"SSD0": qos},
            controller,
        )

        # P0的b=2 us，P1的b=100 us，价值密度保证P0先选中。
        gateway.submit_batch([
            make_request(
                "p0_0",
                "P0",
                "p0_group_0",
                1,
                arrival_time_us=7,
            ),
            make_request(
                "p0_1",
                "P0",
                "p0_group_0",
                1,
                arrival_time_us=7,
            ),
            make_request(
                "p1_0",
                "P1",
                "p1_group_0",
                100,
                arrival_time_us=7,
            ),
        ], arrival_time_us=7)

        backend.set_budget(7, 10)
        qos.process_at(7)
        self.assertEqual(
            backend.accepted,
            [],
            "non-boundary Stage-0 arrival leaked before the 80 us gate",
        )

        # 80 us只允许P0下发一个Block，保留一个Block以在91 us
        # 制造真实的非边界Queue-empty。
        backend.set_budget(80, 1)
        qos.process_at(80)
        self.assertEqual(
            [request[0]["request_id"] for request in backend.accepted],
            ["p0_0"],
        )

        backend.set_budget(91, 10)
        qos.process_at(91)
        self.assertEqual(
            [request[0]["request_id"] for request in backend.accepted],
            ["p0_0", "p0_1"],
        )

        # Queue-empty回调后再处理一次同时间控制。若DPU把P1
        # 立即写成91 us，它会在这次第二轮中泄漏。
        qos.process_at(91)
        self.assertEqual(
            [request[0]["request_id"] for request in backend.accepted],
            ["p0_0", "p0_1"],
            "waiter leaked after a non-boundary Queue-empty callback",
        )

        backend.set_budget(160, 10)
        qos.process_at(160)
        self.assertEqual(
            [request[0]["request_id"] for request in backend.accepted],
            ["p0_0", "p0_1", "p1_0"],
        )
        self.assertEqual(gateway.group_weight_write_count, 0)

    def test_next_layer_stays_parked_until_p_node_is_selected_again(self):
        qos = _RecordingQoS(["q000", "q001"])
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=4,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000", "q001"]},
            _FixedBinding({
                ("P0", "SSD0"): "q000",
                ("P1", "SSD0"): "q001",
            }),
            qos.input,
            {"SSD0": qos},
            controller,
        )

        # P0第一读组完成，但整次四组推理尚未完成，q000必须park。
        gateway.submit_batch([
            make_request(
                "p0_group_0",
                "P0",
                "p0_group_0",
                1,
                arrival_time_us=7,
            ),
        ], arrival_time_us=7)
        qos.depths["q000"] = 0
        gateway.on_qos_queue_state_change("SSD0", event_time_us=91)
        self.assertEqual(last_rate_state(qos, "q000")[:2], (0, 0))
        self.assertEqual(last_weight_state(qos, "q000")[0], 0)

        # P1随后成为owner，并以下降但非空的depth证明已经开始下发。
        gateway.submit_batch([
            make_request(
                "p1_group_0_a",
                "P1",
                "p1_group_0",
                100,
                arrival_time_us=167,
            ),
            make_request(
                "p1_group_0_b",
                "P1",
                "p1_group_0",
                100,
                arrival_time_us=167,
            ),
        ], arrival_time_us=167)
        qos.depths["q001"] = 1
        gateway.on_qos_queue_state_change("SSD0", event_time_us=201)
        self.assertEqual(controller.selected_p_node_id, "P1")
        self.assertTrue(controller.owner_locked)

        # P0下一层在P1锁定期间到达。注册Demand不能隐式把park恢复成
        # uncapped；只有P1排空、Utility重新选中P0后才允许开门。
        gateway.submit_batch([
            make_request(
                "p0_group_1",
                "P0",
                "p0_group_1",
                1,
                arrival_time_us=207,
                compute_layer_index=0,
                prefetch_layer_index=1,
            ),
        ], arrival_time_us=207)
        self.assertEqual(controller.selected_p_node_id, "P1")
        self.assertEqual(last_rate_state(qos, "q000")[:2], (0, 0))
        self.assertEqual(last_weight_state(qos, "q000")[0], 0)

        qos.depths["q001"] = 0
        gateway.on_qos_queue_state_change("SSD0", event_time_us=251)
        self.assertEqual(controller.selected_p_node_id, "P0")
        self.assertEqual(
            last_rate_state(qos, "q000")[:2],
            (CONTROL_PERIOD_US, None),
        )
        self.assertEqual(last_weight_state(qos, "q000")[0], 1)
        self.assertEqual(gateway.group_weight_write_count, 0)
        assert_all_control_times_are_tick_aligned(self, qos)

    def test_incomplete_p_node_parks_queue_and_fourth_group_restores(self):
        qos = _RecordingQoS(["q000"])
        controller = UtilityEDFController(
            {"SSD0": CAPACITY_BYTES_PER_SECOND},
            score_mode="integer",
            deadline_allowance_us=750,
            compute_layer_count=4,
        )
        gateway = DPURequestGateway(
            {"SSD0": ["q000"]},
            _FixedBinding({("P0", "SSD0"): "q000"}),
            qos.input,
            {"SSD0": qos},
            controller,
        )

        for group_index in range(4):
            arrival_time_us = 7 + group_index * 160
            empty_time_us = 91 + group_index * 160
            gateway.submit_batch([
                make_request(
                    f"p0_group_{group_index}",
                    "P0",
                    f"p0_group_{group_index}",
                    1,
                    arrival_time_us=arrival_time_us,
                    compute_layer_index=(
                        None if group_index == 0 else group_index - 1
                    ),
                    prefetch_layer_index=group_index,
                ),
            ], arrival_time_us=arrival_time_us)
            qos.depths["q000"] = 0
            gateway.on_qos_queue_state_change(
                "SSD0",
                event_time_us=empty_time_us,
            )

            self.assertEqual(
                controller.completed_coflow_count_by_p_node["P0"],
                group_index + 1,
            )
            if group_index < 3:
                # 前三个读组结束只park，不恢复baseline默认值。
                self.assertEqual(
                    last_rate_state(qos, "q000")[:2],
                    (0, 0),
                )
                self.assertEqual(
                    last_weight_state(qos, "q000")[0],
                    0,
                )

        # 第四组后才恢复(CIR=0, PIR=uncapped, weight=1)。
        self.assertEqual(
            last_rate_state(qos, "q000")[:2],
            (0, None),
        )
        self.assertEqual(last_weight_state(qos, "q000")[0], 1)
        self.assertEqual(gateway.group_weight_write_count, 0)
        self.assertEqual(qos.group_weight_updates, [])
        self.assertEqual(gateway.control_update_non_tick_write_count, 0)
        assert_all_control_times_are_tick_aligned(self, qos)


if __name__ == "__main__":
    unittest.main()
