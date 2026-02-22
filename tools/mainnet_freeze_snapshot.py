from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from powx.chain import Chain
from powx.config import CONFIG


CONSENSUS_CRITICAL_KEYS = [
    "symbol",
    "chain_id",
    "consensus_lock_enabled",
    "protocol_version",
    "protocol_upgrade_v2_height",
    "target_schedule",
    "target_block_time",
    "difficulty_window",
    "asert_half_life",
    "mtp_window",
    "halving_interval",
    "initial_block_reward",
    "max_total_supply",
    "min_tx_fee",
    "max_target",
    "initial_target",
    "max_adjust_factor_up",
    "max_adjust_factor_down",
    "max_transactions_per_block",
    "max_mempool_transactions",
    "max_mempool_virtual_bytes",
    "min_mempool_fee_rate",
    "mempool_ancestor_limit",
    "mempool_descendant_limit",
    "mempool_rbf_enabled",
    "mempool_cpfp_enabled",
    "max_rbf_replacements",
    "min_rbf_fee_delta",
    "min_rbf_feerate_delta",
    "max_tx_inputs",
    "max_tx_outputs",
    "max_future_block_seconds",
    "max_block_timestamp_step_seconds",
    "max_mempool_tx_age_seconds",
    "max_future_tx_seconds",
    "max_reorg_depth",
    "pow_algorithm",
    "fixed_genesis_hash",
    "fixed_genesis_txid",
    "fixed_genesis_timestamp",
    "fixed_genesis_address",
    "fixed_genesis_supply",
    "fixed_genesis_tx_nonce",
    "fixed_genesis_block_nonce",
]


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _config_snapshot() -> dict[str, Any]:
    config_dict = asdict(CONFIG)
    return {key: config_dict[key] for key in CONSENSUS_CRITICAL_KEYS}


def _chain_snapshot(data_dir: Path) -> dict[str, Any]:
    chain = Chain(data_dir)
    if not chain.exists():
        return {"chain_state_present": False}

    chain.load()
    tip = chain.tip
    genesis_hash = chain.chain[0].block_hash if chain.chain else ""
    return {
        "chain_state_present": True,
        "height": chain.height,
        "tip_hash": tip.block_hash if tip else "",
        "chain_work": int(tip.chain_work) if tip else 0,
        "genesis_hash": genesis_hash,
        "issued_supply": chain.issued_supply(),
    }


def build_snapshot(data_dir: Path) -> dict[str, Any]:
    config_snapshot = _config_snapshot()
    chain_snapshot = _chain_snapshot(data_dir)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "created_utc": now,
        "consensus_config": config_snapshot,
        "chain_snapshot": chain_snapshot,
    }
    payload["consensus_config_hash"] = hashlib.sha256(_canonical_bytes(config_snapshot)).hexdigest()
    payload["freeze_snapshot_hash"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mainnet freeze snapshot for KK91.")
    parser.add_argument("--data-dir", default="./data", help="Chain data directory")
    parser.add_argument(
        "--out",
        default="./docs/mainnet_freeze_snapshot.json",
        help="Output file for freeze snapshot JSON",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(data_dir)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)

    print("Freeze snapshot written")
    print(json.dumps({"out": str(output), "hash": snapshot["freeze_snapshot_hash"]}, indent=2))


if __name__ == "__main__":
    main()
