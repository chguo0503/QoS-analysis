#!/usr/bin/env python3
"""Run resumable 2^3 Coflow/Utility/EDF component ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qos_ssd_simulator import load_simulation_config, run_one  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "utility_edf_component_ablations"
)
SCHEMA_VERSION = 1
SOURCE_FILES = (
    "config/simulation_config.yaml",
    "DPU/rate_controller.py",
    "DPU/dispatcher.py",
    "qos_ssd_simulator.py",
    "llm_workload/layer_request.py",
    "llm_workload/kv_placement_manager.py",
    "discrete_simulation/simulator.py",
    "qos/token_bucket.py",
    "qos/token_bucket_stage.py",
    "qos/schedulers/hierarchical.py",
    "qos/schedulers/weighted_round_robin.py",
)
FACTOR_DEFINITIONS = {
    "C": {
        "0": "per-SSD local owner with local path lock",
        "1": "global cross-SSD p-node owner with sticky coflow lock",
    },
    "U": {
        "0": "stable FCFS Stage-0 ordering",
        "1": "integer Utility Density Stage-0 ordering",
    },
    "E": {
        "0": "stable FCFS prefetch ordering without deadline protection",
        "1": "EDF prefetch ordering with deadline+750us prefix protection",
    },
    "shared_foundation": (
        "single-owner admission Gate per SSD; waiting Queue=(0,0,0); "
        "selected Queue=(SSD capacity,uncapped,1); t=0 pre-park; "
        "strict 80us control grid; fixed Group WRR"
    ),
}

ANCHOR_POLICIES = (
    "baseline",
    "demand_aware_fcfs_cir",
    "utility_edf_integer_l750",
)
FACTOR_POLICIES = tuple(
    f"ablation_c{coflow}_u{utility}_e{edf}"
    for coflow in (0, 1)
    for utility in (0, 1)
    for edf in (0, 1)
)
DEFAULT_POLICIES = ANCHOR_POLICIES + FACTOR_POLICIES

EXPECTED_EXPERIMENT = {
    "gpu_count": 128,
    "inference_count_per_gpu": 1,
    "batch_size": 1,
    "effective_compute_tflops": 512.0,
    "first_layer_index": 0,
    "last_layer_index": 3,
    "random_seed": 6103,
    "input_tokens_range": [100_000, 200_000],
    "prefill_layer_hit_ratio_range": [0.50, 0.99],
    "placement_strategy": "random",
    "queue_binding_strategy": "balanced_exclusive",
    "backend_execution_mode": "batched_exact",
    "backend_batch_commands": 32,
    "control_period_us": 80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact, resumable Coflow/Utility/EDF component ablations."
        ),
    )
    parser.add_argument(
        "--ssd-counts",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun valid existing point checkpoints",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help=(
            "write only independent point checkpoints; use this for "
            "parallel workers, then run once without this flag to merge"
        ),
    )
    return parser.parse_args()


def experiment_identity(config: dict) -> dict:
    workload = config["workload"]
    generation = config["workload_generation"]
    backend = config["ssd"]["backend"]
    return {
        "gpu_count": config["topology"]["gpu_count"],
        "inference_count_per_gpu": generation[
            "inference_count_per_gpu"
        ],
        "batch_size": workload["batch_size"],
        "effective_compute_tflops": config["gpu"][
            "effective_compute_tflops"
        ],
        "first_layer_index": workload["first_layer_index"],
        "last_layer_index": workload["last_layer_index"],
        "random_seed": generation["random_seed"],
        "input_tokens_range": generation["input_tokens_range"],
        "prefill_layer_hit_ratio_range": generation[
            "prefill_layer_hit_ratio_range"
        ],
        "placement_strategy": workload["placement"]["strategy"],
        "queue_binding_strategy": config["dpu"]["queue_binding"][
            "strategy"
        ],
        "backend_execution_mode": backend["execution_mode"],
        "backend_batch_commands": backend["exact_batch_max_commands"],
        "control_period_us": config["qos"]["token_bucket"][
            "update_period_us"
        ],
    }


def validate_experiment_identity(config: dict) -> dict:
    identity = experiment_identity(config)
    if identity != EXPECTED_EXPERIMENT:
        raise RuntimeError(
            "ablation experiment identity differs from the validated scope:\n"
            f"actual={json.dumps(identity, ensure_ascii=False, sort_keys=True)}\n"
            f"expected={json.dumps(EXPECTED_EXPERIMENT, ensure_ascii=False, sort_keys=True)}"
        )
    return identity


def identity_hash(identity: dict) -> str:
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def point_path(output_dir: Path, ssd_count: int, policy: str) -> Path:
    return output_dir / "raw" / f"ssd_{ssd_count:02d}" / f"{policy}.json"


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_summary(summary: dict, policy: str, ssd_count: int) -> None:
    required = {
        "completed_inference_count": EXPECTED_EXPERIMENT["gpu_count"],
        "starvation_free": True,
        "request_count": 456_116,
        "byte_count": 67_257_040_896,
    }
    for field, expected in required.items():
        actual = summary.get(field)
        if actual != expected:
            raise RuntimeError(
                f"{policy}/{ssd_count}SSD: {field}={actual!r}, "
                f"expected {expected!r}"
            )
    periods = summary.get("control_update_period_us_by_storage_target")
    expected_periods = {
        f"SSD{index}": EXPECTED_EXPERIMENT["control_period_us"]
        for index in range(ssd_count)
    }
    if periods != expected_periods:
        raise RuntimeError(
            f"{policy}/{ssd_count}SSD: control periods {periods!r}, "
            f"expected {expected_periods!r}"
        )
    if policy.startswith("ablation_") or policy.startswith("utility_edf_"):
        if summary.get("control_update_non_tick_write_count") != 0:
            raise RuntimeError(
                f"{policy}/{ssd_count}SSD emitted non-tick control writes"
            )
        if summary.get("group_weight_write_count") != 0:
            raise RuntimeError(
                f"{policy}/{ssd_count}SSD changed fixed Group WRR"
            )
        aligned = summary.get("control_update_tick_aligned_write_count")
        written = sum(
            summary.get(field, 0)
            for field in (
                "rate_control_write_count",
                "queue_weight_write_count",
                "group_weight_write_count",
            )
        )
        if aligned != written:
            raise RuntimeError(
                f"{policy}/{ssd_count}SSD control-write audit mismatch: "
                f"aligned={aligned!r}, written={written!r}"
            )


def load_valid_checkpoint(
    path: Path,
    *,
    identity: dict,
    ssd_count: int,
    policy: str,
    current_source_hash: str,
) -> dict | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("experiment") != identity:
        return None
    if document.get("experiment_sha256") != identity_hash(identity):
        return None
    if document.get("source_sha256") != current_source_hash:
        return None
    if document.get("ssd_count") != ssd_count:
        return None
    if document.get("policy") != policy:
        return None
    summary = document.get("summary")
    if not isinstance(summary, dict):
        return None
    validate_summary(summary, policy, ssd_count)
    return document


def run_point(
    config: dict,
    output_dir: Path,
    identity: dict,
    current_source_hash: str,
    ssd_count: int,
    policy: str,
    *,
    force: bool,
) -> dict:
    path = point_path(output_dir, ssd_count, policy)
    if not force:
        existing = load_valid_checkpoint(
            path,
            identity=identity,
            ssd_count=ssd_count,
            policy=policy,
            current_source_hash=current_source_hash,
        )
        if existing is not None:
            print(
                f"SKIP ssd_count={ssd_count} policy={policy} "
                f"path={path}",
                flush=True,
            )
            return existing

    summary = run_one(deepcopy(config), ssd_count, policy)
    validate_summary(summary, policy, ssd_count)
    document = {
        "experiment": identity,
        "experiment_sha256": identity_hash(identity),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": current_source_hash,
        "factor_definitions": FACTOR_DEFINITIONS,
        "ssd_count": ssd_count,
        "policy": policy,
        "summary": summary,
    }
    atomic_write_json(path, document)
    return document


def aggregate_checkpoints(
    output_dir: Path,
    identity: dict,
    current_source_hash: str,
    ssd_counts: list[int],
    policies: list[str],
) -> dict:
    aggregate = {
        "experiment": identity,
        "experiment_sha256": identity_hash(identity),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": current_source_hash,
        "factor_definitions": FACTOR_DEFINITIONS,
        "ssd_counts": list(ssd_counts),
        "policies": list(policies),
        "topologies": {},
    }
    for ssd_count in ssd_counts:
        topology = aggregate["topologies"].setdefault(
            f"{ssd_count}_ssd",
            {},
        )
        for policy in policies:
            path = point_path(output_dir, ssd_count, policy)
            document = load_valid_checkpoint(
                path,
                identity=identity,
                ssd_count=ssd_count,
                policy=policy,
                current_source_hash=current_source_hash,
            )
            if document is None:
                raise RuntimeError(f"missing valid checkpoint: {path}")
            topology[policy] = document["summary"]
    atomic_write_json(output_dir / "summary.json", aggregate)
    return aggregate


def main() -> None:
    args = parse_args()
    if not args.ssd_counts or any(
        count < 1 or count > 10 for count in args.ssd_counts
    ):
        raise ValueError("--ssd-counts must contain integers from 1 through 10")
    unknown = sorted(set(args.policies) - set(DEFAULT_POLICIES))
    if unknown:
        raise ValueError(f"unknown ablation policies: {unknown}")
    ssd_counts = sorted(set(args.ssd_counts))
    policies = list(dict.fromkeys(args.policies))
    output_dir = args.output_dir.resolve()
    config = load_simulation_config()
    identity = validate_experiment_identity(config)
    current_source_hash = source_hash()
    print(
        f"EXPERIMENT sha256={identity_hash(identity)} "
        f"source_sha256={current_source_hash} "
        f"points={len(ssd_counts) * len(policies)} "
        f"output={output_dir}",
        flush=True,
    )
    for ssd_count in ssd_counts:
        for policy in policies:
            run_point(
                config,
                output_dir,
                identity,
                current_source_hash,
                ssd_count,
                policy,
                force=args.force,
            )
    if not args.no_aggregate:
        aggregate_checkpoints(
            output_dir,
            identity,
            current_source_hash,
            ssd_counts,
            policies,
        )
