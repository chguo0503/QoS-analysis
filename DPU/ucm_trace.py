"""读取 UCM SQE trace bundle，将 Retrieve Entry 转成 DPU 请求。

raw 文件本身没有记录边界；每条 SQE 的位置、源 GPU、目标 ASU
和发出时间都以 ``sqe_manifest.jsonl`` 为准。模块输出到现有
``DPURequestGateway`` 的边界为止：调用方对每个返回的完整层调用一次
``submit_batch()`` 即可。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct


_SQE_HEADER_BYTES = 64
_BATCH_ENTRY_BYTES = 36
_BATCH_RETRIEVE_OPCODE = 0x46
_BATCH_DPTR_TYPE = 0x01
_STANDARD_DPTR_TYPE = 0x40


def _ceil_div(numerator, denominator):
    """用整数计算向上取整，避免大数带宽转浮点数丢失精度。"""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def parse_batch_retrieve(raw_sqe):
    """解析一条小端 UCM BatchRetrieve SQE。

    只返回重放和排障会用到的字段。DMA 地址和 MR key 保留供排障，
    QoS/SSD 数据面实际只消费每个 Entry 的长度。
    """

    if len(raw_sqe) < _SQE_HEADER_BYTES:
        raise ValueError("raw SQE is shorter than the 64-byte header")

    header = struct.unpack_from("<16I", raw_sqe, 0)
    opcode = header[0] & 0xFF
    if opcode != _BATCH_RETRIEVE_OPCODE:
        raise ValueError(
            f"expected BatchRetrieve opcode 0x46, got 0x{opcode:02x}"
        )

    batch_number = header[10] & 0xFFFF
    if not 1 <= batch_number <= 110:
        raise ValueError("BatchRetrieve batch_number must be in [1, 110]")
    expected_length = _SQE_HEADER_BYTES + batch_number * _BATCH_ENTRY_BYTES
    if len(raw_sqe) != expected_length:
        raise ValueError(
            f"raw SQE length {len(raw_sqe)} does not match "
            f"batch_number={batch_number} ({expected_length} bytes)"
        )
    if header[8] != batch_number * _BATCH_ENTRY_BYTES:
        raise ValueError("BatchRetrieve header descriptor length is inconsistent")
    if header[9] >> 24 != _BATCH_DPTR_TYPE:
        raise ValueError("BatchRetrieve header does not use the batch DPTR type")

    entries = []
    for entry_index in range(batch_number):
        base = _SQE_HEADER_BYTES + entry_index * _BATCH_ENTRY_BYTES
        words = struct.unpack_from("<9I", raw_sqe, base)
        length_bytes = words[7] & 0xFFFFFF
        if length_bytes <= 0:
            raise ValueError(f"BatchRetrieve entry {entry_index} has zero length")
        if words[8] >> 24 != _STANDARD_DPTR_TYPE:
            raise ValueError(
                f"BatchRetrieve entry {entry_index} does not use standard DPTR"
            )
        entries.append({
            "entry_index": entry_index,
            "offset_bytes": words[0],
            "asu_key_hex": raw_sqe[base + 4:base + 12].hex(),
            "buffer_addr": words[5] | (words[6] << 32),
            "mr_key": (words[7] >> 24) | ((words[8] & 0xFFFFFF) << 8),
            "length_bytes": length_bytes,
        })

    return {
        "cid": header[0] >> 16,
        "kv_ns_id": header[1],
        "batch_number": batch_number,
        "entries": entries,
        "payload_bytes": sum(entry["length_bytes"] for entry in entries),
    }


@dataclass(frozen=True)
class UcmLayerSubmission:
    """一个完整的 ``(源请求, 层)`` DPU 提交单元。"""

    source_request_id: str
    gpu_id: int
    layer_id: int
    timestamp_ns: int
    arrival_time_us: float
    inference_arrival_time_us: float
    service_window_us: float
    batch_total_bytes: int
    path_bytes_by_storage_target: dict
    requests: tuple

    @property
    def request_ids(self):
        """返回 Entry 请求 ID，便于调用方登记 SSD completion 归属。"""

        return tuple(request["basic"]["request_id"] for request in self.requests)


class UcmTraceBundle:
    """将四文件 UCM trace bundle 暴露为完整层提交流。"""

    def __init__(self, bundle_dir):
        self.bundle_dir = Path(bundle_dir)
        self.raw_path = self.bundle_dir / "raw_sqe.bin"
        self.manifest_path = self.bundle_dir / "sqe_manifest.jsonl"
        self.metadata_path = self.bundle_dir / "metadata.json"
        self.workload_summary_path = self.bundle_dir / "workload_summary.json"
        for path in (
            self.raw_path,
            self.manifest_path,
            self.metadata_path,
            self.workload_summary_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing UCM trace bundle file: {path}")

        self.metadata = self._read_json(self.metadata_path)
        self.workload_summary = self._read_json(self.workload_summary_path)
        helper = self.metadata.get("ucm_helper", {})
        if helper.get("endianness", "little") != "little":
            raise ValueError("only little-endian UCM SQE traces are supported")

        request_rows = self.workload_summary.get("requests")
        if not isinstance(request_rows, list) or not request_rows:
            raise ValueError("workload_summary.requests must be a non-empty list")
        self.workload_by_source_request = {}
        for row in request_rows:
            source_request_id = row.get("source_request_id")
            if not source_request_id:
                raise ValueError("workload request is missing source_request_id")
            if source_request_id in self.workload_by_source_request:
                raise ValueError(f"duplicate workload request {source_request_id!r}")
            compute_window_ns = int(row.get("single_layer_compute_ns", 0))
            if compute_window_ns <= 0:
                raise ValueError(
                    f"{source_request_id!r} has a non-positive compute window"
                )
            self.workload_by_source_request[source_request_id] = row

    @staticmethod
    def _read_json(path):
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        return value

    @staticmethod
    def _read_raw_record(raw_stream, record):
        raw_offset = int(record["raw_offset"])
        raw_length = int(record["raw_length"])
        if raw_offset < 0 or raw_length <= 0:
            raise ValueError("manifest raw_offset/raw_length is invalid")
        raw_stream.seek(raw_offset)
        raw_sqe = raw_stream.read(raw_length)
        if len(raw_sqe) != raw_length:
            raise ValueError("raw_sqe.bin is shorter than a manifest record")
        return raw_sqe

    def _iter_retrieve_records(self, raw_stream):
        """逐条返回校验后的 Retrieve；Exist 不进入 QoS。"""

        last_record_index = -1
        last_retrieve_timestamp_ns = -1
        seen_sqe_uids = set()
        with self.manifest_path.open("r", encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_index = int(record["record_index"])
                if record_index <= last_record_index:
                    raise ValueError("manifest record_index must be strictly increasing")
                last_record_index = record_index

                opcode = record.get("opcode")
                if opcode == "Exist" or record.get("phase") == "prefix_query":
                    continue
                if opcode != "BatchRetrieve":
                    raise ValueError(
                        f"unsupported manifest opcode {opcode!r} on line {line_number}"
                    )

                timestamp_ns = int(record["timestamp_ns"])
                if timestamp_ns < last_retrieve_timestamp_ns:
                    raise ValueError("Retrieve manifest timestamps must be non-decreasing")
                last_retrieve_timestamp_ns = timestamp_ns
                sqe_uid = record.get("sqe_uid")
                if not sqe_uid or sqe_uid in seen_sqe_uids:
                    raise ValueError(f"invalid or duplicate sqe_uid {sqe_uid!r}")
                seen_sqe_uids.add(sqe_uid)

                raw_sqe = self._read_raw_record(raw_stream, record)
                parsed = parse_batch_retrieve(raw_sqe)
                if parsed["batch_number"] != int(record["batch_number"]):
                    raise ValueError(f"{sqe_uid}: manifest/raw batch_number mismatch")
                if len(raw_sqe) != int(record.get("descriptor_bytes", len(raw_sqe))):
                    raise ValueError(f"{sqe_uid}: manifest/raw descriptor size mismatch")
                if parsed["payload_bytes"] != int(record["payload_bytes"]):
                    raise ValueError(f"{sqe_uid}: manifest/raw payload size mismatch")
                yield record, parsed

    def iter_layer_submissions(self):
        """流式返回可直接交给 ``submit_batch()`` 的完整层批次。

        现有 UCM trace 会把同一逻辑层的所有 ASU/SQE 分片连续写入。
        若已结束的层再次出现，则直接报错，防止损坏的 trace 悄悄为
        同一层创建两个 DPU Demand。
        """

        finished_layer_keys = set()
        current_key = None
        current_records = []
        with self.raw_path.open("rb") as raw_stream:
            for record, parsed in self._iter_retrieve_records(raw_stream):
                layer_key = (
                    record["source_request_id"],
                    int(record["layer_id"]),
                )
                if current_key is None:
                    current_key = layer_key
                elif layer_key != current_key:
                    finished_layer_keys.add(current_key)
                    yield self._build_layer_submission(current_records)
                    if layer_key in finished_layer_keys:
                        raise ValueError(
                            f"layer {layer_key!r} is not contiguous in the manifest"
                        )
                    current_key = layer_key
                    current_records = []
                current_records.append((record, parsed))

            if current_records:
                yield self._build_layer_submission(current_records)

    def _build_layer_submission(self, records):
        first = records[0][0]
        source_request_id = first["source_request_id"]
        layer_id = int(first["layer_id"])
        gpu_id = int(first["gpu_id"])
        timestamp_ns = int(first["timestamp_ns"])
        for record, _ in records[1:]:
            if (
                record["source_request_id"] != source_request_id
                or int(record["layer_id"]) != layer_id
                or int(record["gpu_id"]) != gpu_id
                or int(record["timestamp_ns"]) != timestamp_ns
            ):
                raise ValueError("one logical layer has inconsistent manifest metadata")

        workload = self.workload_by_source_request.get(source_request_id)
        if workload is None:
            raise ValueError(f"manifest request {source_request_id!r} is not in workload")
        if int(workload["gpu_id"]) != gpu_id:
            raise ValueError(f"{source_request_id!r} has inconsistent gpu_id")
        inference_arrival_ns = int(workload["arrival_time_ns"])
        compute_window_ns = int(workload["single_layer_compute_ns"])

        path_bytes = {}
        parsed_rows = []
        for record, parsed in records:
            storage_target_id = f"SSD{int(record['target_asu_id'])}"
            path_bytes[storage_target_id] = (
                path_bytes.get(storage_target_id, 0) + parsed["payload_bytes"]
            )
            parsed_rows.append((record, parsed, storage_target_id))
        batch_total_bytes = sum(path_bytes.values())

        demand_group_id = f"{source_request_id}:layer:{layer_id}"
        requested_cir = {
            storage_target_id: _ceil_div(
                byte_count * 1_000_000_000,
                compute_window_ns,
            )
            for storage_target_id, byte_count in path_bytes.items()
        }
        requests = []
        for record, parsed, storage_target_id in parsed_rows:
            for entry in parsed["entries"]:
                request_id = (
                    f"{record['sqe_uid']}:entry:{entry['entry_index']:03d}"
                )
                requests.append({
                    "basic": {
                        "request_id": request_id,
                        "p_node_id": f"P{gpu_id}",
                        "storage_target_id": storage_target_id,
                        "size_bytes": entry["length_bytes"],
                    },
                    "demand_bw": {
                        "demand_group_id": demand_group_id,
                        "compute_layer_index": (
                            None if layer_id == 0 else layer_id - 1
                        ),
                        "prefetch_layer_index": layer_id,
                        "inference_arrival_time_us": (
                            inference_arrival_ns / 1_000
                        ),
                        "service_window_us": compute_window_ns / 1_000,
                        "deadline_us": (
                            timestamp_ns + compute_window_ns
                        ) / 1_000,
                        "batch_total_bytes": batch_total_bytes,
                        "aggregate_bytes_on_storage_target": path_bytes[
                            storage_target_id
                        ],
                        "aggregate_required_bytes_per_second": requested_cir[
                            storage_target_id
                        ],
                    },
                })

        return UcmLayerSubmission(
            source_request_id=source_request_id,
            gpu_id=gpu_id,
            layer_id=layer_id,
            timestamp_ns=timestamp_ns,
            arrival_time_us=timestamp_ns / 1_000,
            inference_arrival_time_us=inference_arrival_ns / 1_000,
            service_window_us=compute_window_ns / 1_000,
            batch_total_bytes=batch_total_bytes,
            path_bytes_by_storage_target=dict(sorted(path_bytes.items())),
            requests=tuple(requests),
        )
