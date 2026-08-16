"""UCM raw-SQE 到 DPU 请求适配器的小型 golden 测试。"""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from DPU.ucm_trace import UcmTraceBundle, parse_batch_retrieve


def _batch_retrieve(entries, cid=1, kv_ns_id=100):
    """只构造解析器需要的最小合法 BatchRetrieve 二进制数据。"""

    raw = bytearray(64 + 36 * len(entries))
    header = [0] * 16
    header[0] = (cid << 16) | (3 << 14) | 0x46
    header[1] = kv_ns_id
    header[8] = 36 * len(entries)
    header[9] = 1 << 24
    header[10] = len(entries)
    struct.pack_into("<16I", raw, 0, *header)
    for index, (offset, key, length) in enumerate(entries):
        base = 64 + index * 36
        words = [0] * 9
        words[0] = offset
        words[5] = 0x1000 + index * 0x1000
        words[7] = (0x12 << 24) | length
        words[8] = (0x40 << 24) | 0x3456
        struct.pack_into("<9I", raw, base, *words)
        raw[base + 4:base + 12] = key
    return bytes(raw)


class UcmTraceBundleTests(unittest.TestCase):
    def _write_bundle(self, directory, *, corrupt_payload=False):
        directory = Path(directory)
        metadata = {
            "schema_version": "ucm-sqe-simulation/v1",
            "ucm_helper": {"endianness": "little"},
        }
        workload = {
            "schema_version": "ucm-sqe-simulation/v1",
            "requests": [{
                "gpu_id": 7,
                "source_request_id": "gpu-0007-prefix-0000",
                "arrival_time_ns": 50,
                "single_layer_compute_ns": 2_000,
            }],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (directory / "workload_summary.json").write_text(
            json.dumps(workload), encoding="utf-8"
        )

        raw_parts = [bytes(64)]
        records = [{
            "sqe_uid": "query",
            "phase": "prefix_query",
            "opcode": "Exist",
            "timestamp_ns": 100,
            "gpu_id": 7,
            "source_request_id": "gpu-0007-prefix-0000",
            "layer_id": None,
            "target_asu_id": 0,
            "batch_number": 1,
            "descriptor_bytes": 64,
            "payload_bytes": 0,
            "record_index": 0,
            "raw_offset": 0,
            "raw_length": 64,
        }]

        specifications = [
            # ASU0 上的两个 SQE 分片必须合并成一条 3,072-byte 路径。
            ("a0", 0, 0, 100, [(0, b"key00001", 1_024)]),
            ("a1", 0, 0, 100, [(512, b"key00002", 2_048)]),
            ("b0", 1, 0, 100, [(0, b"key00003", 4_096)]),
            # 第二个逻辑层用来验证层边界。
            ("c0", 0, 1, 2_100, [(1_024, b"key00004", 512)]),
        ]
        raw_offset = 64
        for record_index, (uid, asu_id, layer_id, timestamp, entries) in enumerate(
            specifications, 1
        ):
            raw = _batch_retrieve(entries, cid=record_index + 1)
            payload = sum(entry[2] for entry in entries)
            records.append({
                "sqe_uid": uid,
                "phase": "layer_retrieve",
                "opcode": "BatchRetrieve",
                "timestamp_ns": timestamp,
                "gpu_id": 7,
                "source_request_id": "gpu-0007-prefix-0000",
                "layer_id": layer_id,
                "target_asu_id": asu_id,
                "batch_number": len(entries),
                "descriptor_bytes": len(raw),
                "payload_bytes": (
                    payload + 1 if corrupt_payload and uid == "a0" else payload
                ),
                "record_index": record_index,
                "raw_offset": raw_offset,
                "raw_length": len(raw),
            })
            raw_parts.append(raw)
            raw_offset += len(raw)

        (directory / "raw_sqe.bin").write_bytes(b"".join(raw_parts))
        (directory / "sqe_manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_parse_batch_retrieve_golden(self):
        raw = _batch_retrieve([
            (512, b"12345678", 1_024),
            (1_024, b"abcdefgh", 2_048),
        ], cid=9, kv_ns_id=17)
        parsed = parse_batch_retrieve(raw)

        self.assertEqual(parsed["cid"], 9)
        self.assertEqual(parsed["kv_ns_id"], 17)
        self.assertEqual(parsed["batch_number"], 2)
        self.assertEqual(parsed["payload_bytes"], 3_072)
        self.assertEqual(parsed["entries"][0]["asu_key_hex"], "3132333435363738")
        self.assertEqual(parsed["entries"][1]["length_bytes"], 2_048)

    def test_bundle_skips_exist_and_aggregates_complete_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_bundle(directory)
            submissions = list(UcmTraceBundle(directory).iter_layer_submissions())

        self.assertEqual(len(submissions), 2)
        first = submissions[0]
        self.assertEqual(first.source_request_id, "gpu-0007-prefix-0000")
        self.assertEqual(first.layer_id, 0)
        self.assertEqual(first.timestamp_ns, 100)
        self.assertEqual(first.arrival_time_us, 0.1)
        self.assertEqual(first.inference_arrival_time_us, 0.05)
        self.assertEqual(first.batch_total_bytes, 7_168)
        self.assertEqual(
            first.path_bytes_by_storage_target,
            {"SSD0": 3_072, "SSD1": 4_096},
        )
        self.assertEqual(len(first.requests), 3)
        self.assertEqual(len(set(first.request_ids)), 3)

        ssd0 = [
            request for request in first.requests
            if request["basic"]["storage_target_id"] == "SSD0"
        ]
        self.assertEqual({request["basic"]["p_node_id"] for request in ssd0}, {"P7"})
        self.assertEqual(
            {request["basic"]["size_bytes"] for request in ssd0},
            {1_024, 2_048},
        )
        demand = ssd0[0]["demand_bw"]
        self.assertEqual(demand["aggregate_bytes_on_storage_target"], 3_072)
        self.assertEqual(demand["batch_total_bytes"], 7_168)
        self.assertEqual(
            demand["aggregate_required_bytes_per_second"],
            1_536_000_000,
        )
        self.assertIsNone(demand["compute_layer_index"])
        self.assertEqual(demand["prefetch_layer_index"], 0)
        self.assertEqual(demand["service_window_us"], 2.0)
        self.assertEqual(demand["inference_arrival_time_us"], 0.05)

        second = submissions[1]
        self.assertEqual(second.layer_id, 1)
        self.assertEqual(second.requests[0]["demand_bw"]["compute_layer_index"], 0)

    def test_manifest_payload_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_bundle(directory, corrupt_payload=True)
            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                list(UcmTraceBundle(directory).iter_layer_submissions())


if __name__ == "__main__":
    unittest.main()
