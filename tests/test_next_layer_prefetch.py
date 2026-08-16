"""验证GPU计算当前层、SSD预取下一层的流水语义。"""

from copy import deepcopy
import unittest

from llm_workload.layer_request import DEFAULT_WORKLOAD, LLMWorkload


class NextLayerPrefetchTests(unittest.TestCase):
    """覆盖四层窗口的首层、层间屏障和末层边界。"""

    def setUp(self):
        workload = deepcopy(DEFAULT_WORKLOAD)
        workload.update({
            "workload_id": "prefetch_test",
            "p_node_id": "P0",
            "first_layer_index": 0,
            "last_layer_index": 3,
            "arrival_time_us": 0,
            "input_tokens": 1_000,
            "batch_size": 1,
            "prefill_layer_hit_ratio": 0.5,
        })
        self.llm = LLMWorkload(workload=workload)

    def complete_prefetch(self, plan, completion_time_us):
        for block in plan["blocks"]:
            self.llm.on_storage_complete({
                "request_id": block["request_id"],
                "storage_target_id": "SSD0",
                "completion_time_us": completion_time_us,
            })

    def test_initial_read_then_four_compute_layers_prefetch_ahead(self):
        compute_us = self.llm.layer_plan["compute_time_us"]
        block_count = self.llm.layer_plan["block_count"]

        initial = self.llm.start_next_layer()
        self.assertIsNone(initial["compute_layer_index"])
        self.assertEqual(initial["prefetch_layer_index"], 0)
        self.assertTrue(all("_layer_00_block_" in block["request_id"]
                            for block in initial["blocks"]))
        self.complete_prefetch(initial, 10)

        first = self.llm.start_next_layer()
        self.assertEqual(first["compute_layer_index"], 0)
        self.assertEqual(first["prefetch_layer_index"], 1)
        self.assertTrue(all("_layer_01_block_" in block["request_id"]
                            for block in first["blocks"]))
        # 第1层预取比第0层计算晚10us，下一层从这个较晚屏障开始。
        self.complete_prefetch(first, compute_us + 20)

        second = self.llm.start_next_layer()
        self.assertEqual(second["compute_layer_index"], 1)
        self.assertEqual(second["prefetch_layer_index"], 2)
        second_start = self.llm.current_layer["layer_start_time_us"]
        self.complete_prefetch(second, second_start + 1)

        third = self.llm.start_next_layer()
        self.assertEqual(third["compute_layer_index"], 2)
        self.assertEqual(third["prefetch_layer_index"], 3)
        third_start = self.llm.current_layer["layer_start_time_us"]
        self.complete_prefetch(third, third_start + 1)

        last = self.llm.start_next_layer()
        self.assertEqual(last["compute_layer_index"], 3)
        self.assertIsNone(last["prefetch_layer_index"])
        self.assertEqual(last["blocks"], [])

        result = self.llm.result()
        self.assertEqual(result["layer_count"], 4)
        self.assertEqual(result["request_count"], 4 * block_count)
        self.assertEqual(result["completed_request_count"], 4 * block_count)
        self.assertEqual(result["initial_layer_read"]["layer_index"], 0)
        self.assertAlmostEqual(result["initial_layer_read"]["read_time_us"], 10)
        self.assertEqual(
            [layer["prefetch_layer_index"] for layer in result["layers"]],
            [1, 2, 3, None],
        )
        self.assertAlmostEqual(result["ssd_stall_us"], 20)
        self.assertAlmostEqual(result["ttft_us"], 4 * compute_us + 20)


if __name__ == "__main__":
    unittest.main()
