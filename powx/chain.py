from __future__ import annotations

import copy
import json
import math
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

from .config import CONFIG, ChainConfig
from .crypto import (
    address_from_public_key,
    private_key_to_public_key,
    sign_digest,
    verify_signature,
)
from .gpu_miner import OpenCLMiner
from .models import Block, Transaction, TxInput, TxOutput, block_work, merkle_root
from .pow_hash import hash_from_seed_words, seed_words_from_block, target_to_words


class ValidationError(Exception):
    pass


class MiningInterruptedError(Exception):
    pass


class Chain:
    MAINNET_LOCKED_TARGET_SCHEDULE = "asert-v3"

    def __init__(self, data_dir: str | Path, config: ChainConfig = CONFIG):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "chain_state.json"

        self.chain: list[Block] = []
        self.utxos: dict[str, dict[str, Any]] = {}
        self.mempool: list[Transaction] = []
        self.nft_registry: dict[str, dict[str, Any]] = {}
        self.nft_listings: dict[str, dict[str, Any]] = {}
        self.contracts: dict[str, dict[str, Any]] = {}
        self._legacy_target_schedule = False
        self._asert_target_schedule = False
        self._consensus_chain_id = self.config.chain_id
        self._consensus_protocol_version = self.config.protocol_version
        self._consensus_protocol_upgrade_v2_height = self.config.protocol_upgrade_v2_height
        self._consensus_target_block_time = self.config.target_block_time
        self._consensus_max_target = self.config.max_target
        self._consensus_max_adjust_factor_up = self.config.max_adjust_factor_up
        self._consensus_max_adjust_factor_down = self.config.max_adjust_factor_down
        self._consensus_asert_half_life = self.config.asert_half_life
        self._consensus_mtp_window = self.config.mtp_window
        self._consensus_max_block_timestamp_step_seconds = self.config.max_block_timestamp_step_seconds
        self._gpu_miner: OpenCLMiner | None = None
        self._gpu_probe_attempted = False
        # Fresh chains must default to runtime consensus schedule, not legacy/window fallback.
        configured_schedule = self._normalize_target_schedule_name(self.config.target_schedule)
        if self._consensus_lock_enabled():
            configured_schedule = self._locked_target_schedule_name()
        self._set_target_schedule_name(configured_schedule or self.MAINNET_LOCKED_TARGET_SCHEDULE)

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def _consensus_lock_enabled(self) -> bool:
        return bool(getattr(self.config, "consensus_lock_enabled", False))

    def _locked_target_schedule_name(self) -> str:
        # Mainnet lock explicitly requires ASERT.
        return self.MAINNET_LOCKED_TARGET_SCHEDULE

    @staticmethod
    def _normalize_target_schedule_name(raw: Any) -> str:
        schedule = str(raw).strip().lower()
        if schedule in {"legacy", "legacy-v1", "v1"}:
            return "legacy-v1"
        if schedule in {"window", "window-v2", "v2"}:
            return "window-v2"
        if schedule in {"asert", "asert-v3", "v3"}:
            return "asert-v3"
        return ""

    @classmethod
    def _detect_target_schedule(cls, raw: Any) -> str:
        if not isinstance(raw, dict):
            return "window-v2"

        normalized = cls._normalize_target_schedule_name(raw.get("target_schedule", ""))
        if normalized:
            return normalized

        if "difficulty_window" not in raw:
            return "legacy-v1"
        try:
            window = int(raw.get("difficulty_window"))
        except (TypeError, ValueError):
            return "window-v2"
        return "legacy-v1" if window <= 2 else "window-v2"

    @staticmethod
    def _is_legacy_schedule_name(name: str) -> bool:
        return name == "legacy-v1"

    @staticmethod
    def _is_asert_schedule_name(name: str) -> bool:
        return name == "asert-v3"

    def _set_target_schedule_name(self, name: str) -> None:
        normalized = self._normalize_target_schedule_name(name)
        if not normalized:
            normalized = self._locked_target_schedule_name() if self._consensus_lock_enabled() else "window-v2"
        self._legacy_target_schedule = self._is_legacy_schedule_name(normalized)
        self._asert_target_schedule = self._is_asert_schedule_name(normalized)

    def _target_schedule_name(self) -> str:
        if self._legacy_target_schedule:
            return "legacy-v1"
        if self._asert_target_schedule:
            return "asert-v3"
        return "window-v2"

    def _resolve_target_schedule_name(self, name: str | None = None) -> str:
        if name is None:
            return self._target_schedule_name()
        normalized = self._normalize_target_schedule_name(name)
        if normalized:
            return normalized
        return self._target_schedule_name()

    def _mtp_window_for_schedule(self, target_schedule: str | None = None) -> int:
        schedule_name = self._resolve_target_schedule_name(target_schedule)
        if self._is_asert_schedule_name(schedule_name):
            return max(11, int(self._consensus_mtp_window))
        return 11

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> None:
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}")

        with self.state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self._reset_consensus_overrides()
        raw_config = data.get("config")
        loaded_chain = [Block.from_dict(item) for item in data.get("chain", [])]
        detected_schedule = self._detect_target_schedule(raw_config)
        state_has_explicit_schedule = isinstance(raw_config, dict) and "target_schedule" in raw_config
        schedule_migrated = False

        if self._consensus_lock_enabled():
            target_schedule = self._locked_target_schedule_name()
            schedule_candidates: list[str] = [target_schedule]
            if not state_has_explicit_schedule:
                schedule_migrated = True
        else:
            target_schedule = detected_schedule
            schedule_candidates = [target_schedule]

            # Auto-heal early bugged local states: empty/genesis-only files could be persisted
            # as window-v2 even when runtime consensus is asert-v3.
            runtime_schedule = self._target_schedule_name()
            if (
                runtime_schedule == "asert-v3"
                and target_schedule == "window-v2"
                and len(loaded_chain) <= 1
            ):
                target_schedule = runtime_schedule
                schedule_migrated = True
                schedule_candidates[0] = target_schedule

            if not state_has_explicit_schedule:
                for candidate_schedule in (runtime_schedule, "asert-v3", "window-v2", "legacy-v1"):
                    if candidate_schedule not in schedule_candidates:
                        schedule_candidates.append(candidate_schedule)

        self._validate_state_config(raw_config, target_schedule=target_schedule)
        self._apply_state_consensus_overrides(raw_config, target_schedule=target_schedule)
        base_max_adjust_down = self._consensus_max_adjust_factor_down

        if loaded_chain:
            selected_schedule = target_schedule
            last_error: ValidationError | None = None
            for candidate_schedule in schedule_candidates:
                self._consensus_max_adjust_factor_down = base_max_adjust_down
                try:
                    self.chain, self.utxos, self.nft_registry, self.nft_listings, self.contracts = self.validate_chain_blocks(
                        loaded_chain,
                        target_schedule=candidate_schedule,
                    )
                    selected_schedule = candidate_schedule
                    last_error = None
                    break
                except ValidationError as exc:
                    if (
                        self._is_legacy_schedule_name(candidate_schedule)
                        and self._legacy_missing_down_adjust(raw_config)
                        and "target mismatch" in str(exc).lower()
                        and base_max_adjust_down != 0.5
                    ):
                        # Some early local chains used 0.5 as down-adjust clamp.
                        self._consensus_max_adjust_factor_down = 0.5
                        try:
                            self.chain, self.utxos, self.nft_registry, self.nft_listings, self.contracts = self.validate_chain_blocks(
                                loaded_chain,
                                target_schedule=candidate_schedule,
                            )
                            selected_schedule = candidate_schedule
                            last_error = None
                            break
                        except ValidationError as inner_exc:
                            last_error = inner_exc
                            continue
                    last_error = exc

            if last_error is not None:
                raise last_error

            target_schedule = selected_schedule
            if target_schedule != detected_schedule:
                schedule_migrated = True
            self._set_target_schedule_name(target_schedule)
        else:
            self.chain = []
            self.utxos = {}
            self.nft_registry = {}
            self.nft_listings = {}
            self.contracts = {}
            if target_schedule != detected_schedule:
                schedule_migrated = True
            self._set_target_schedule_name(target_schedule)

        self.mempool = [Transaction.from_dict(item) for item in data.get("mempool", [])]
        self.mempool = self._sanitize_mempool(self.mempool)
        if not state_has_explicit_schedule:
            schedule_migrated = True
        if schedule_migrated:
            self.save()

    def save(self) -> None:
        target_schedule = self._target_schedule_name()
        difficulty_window = 2 if self._is_legacy_schedule_name(target_schedule) else self.config.difficulty_window
        initial_target = self.chain[0].target if self.chain else self.config.initial_target
        data = {
            "config": {
                "symbol": self.config.symbol,
                "chain_id": self._consensus_chain_id,
                "consensus_lock_enabled": self._consensus_lock_enabled(),
                "protocol_version": self._consensus_protocol_version,
                "protocol_upgrade_v2_height": self._consensus_protocol_upgrade_v2_height,
                "target_schedule": target_schedule,
                "target_block_time": self._consensus_target_block_time,
                "difficulty_window": difficulty_window,
                "asert_half_life": self._consensus_asert_half_life,
                "mtp_window": self._consensus_mtp_window,
                "pow_algorithm": self.config.pow_algorithm,
                "halving_interval": self.config.halving_interval,
                "initial_block_reward": self.config.initial_block_reward,
                "max_total_supply": self.config.max_total_supply,
                "coinbase_maturity": self.config.coinbase_maturity,
                "min_tx_fee": self.config.min_tx_fee,
                "max_target": self._consensus_max_target,
                "initial_target": initial_target,
                "max_adjust_factor_up": self._consensus_max_adjust_factor_up,
                "max_adjust_factor_down": self._consensus_max_adjust_factor_down,
                "max_transactions_per_block": self.config.max_transactions_per_block,
                "max_mempool_transactions": self.config.max_mempool_transactions,
                "max_mempool_virtual_bytes": self.config.max_mempool_virtual_bytes,
                "min_mempool_fee_rate": self.config.min_mempool_fee_rate,
                "min_standard_tx_fee_rate": self.config.min_standard_tx_fee_rate,
                "min_dust_output": self.config.min_dust_output,
                "max_standard_tx_virtual_bytes": self.config.max_standard_tx_virtual_bytes,
                "smart_contracts_enabled": self.config.smart_contracts_enabled,
                "nft_market_enabled": self.config.nft_market_enabled,
                "max_contract_payload_bytes": self.config.max_contract_payload_bytes,
                "max_contract_kv_entries": self.config.max_contract_kv_entries,
                "max_contract_key_bytes": self.config.max_contract_key_bytes,
                "max_contract_value_bytes": self.config.max_contract_value_bytes,
                "mempool_ancestor_limit": self.config.mempool_ancestor_limit,
                "mempool_descendant_limit": self.config.mempool_descendant_limit,
                "mempool_rbf_enabled": self.config.mempool_rbf_enabled,
                "mempool_cpfp_enabled": self.config.mempool_cpfp_enabled,
                "max_rbf_replacements": self.config.max_rbf_replacements,
                "min_rbf_fee_delta": self.config.min_rbf_fee_delta,
                "min_rbf_feerate_delta": self.config.min_rbf_feerate_delta,
                "max_tx_inputs": self.config.max_tx_inputs,
                "max_tx_outputs": self.config.max_tx_outputs,
                "max_future_block_seconds": self.config.max_future_block_seconds,
                "max_block_timestamp_step_seconds": self._consensus_max_block_timestamp_step_seconds,
                "max_mempool_tx_age_seconds": self.config.max_mempool_tx_age_seconds,
                "max_future_tx_seconds": self.config.max_future_tx_seconds,
                "max_reorg_depth": self.config.max_reorg_depth,
                "fixed_genesis_hash": self.config.fixed_genesis_hash,
                "fixed_genesis_txid": self.config.fixed_genesis_txid,
                "fixed_genesis_timestamp": self.config.fixed_genesis_timestamp,
                "fixed_genesis_address": self.config.fixed_genesis_address,
                "fixed_genesis_supply": self.config.fixed_genesis_supply,
                "fixed_genesis_tx_nonce": self.config.fixed_genesis_tx_nonce,
                "fixed_genesis_block_nonce": self.config.fixed_genesis_block_nonce,
            },
            "height": self.height,
            "chain": [block.to_dict() for block in self.chain],
            "utxos": self.utxos,
            "nft_registry": self.nft_registry,
            "nft_listings": self.nft_listings,
            "contracts": self.contracts,
            "mempool": [tx.to_dict() for tx in self.mempool],
        }
        temp_path = self.state_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.state_path)

    @staticmethod
    def _detect_legacy_target_schedule(raw: Any) -> bool:
        return Chain._is_legacy_schedule_name(Chain._detect_target_schedule(raw))

    @staticmethod
    def _legacy_missing_down_adjust(raw: Any) -> bool:
        if not isinstance(raw, dict):
            return True
        return "max_adjust_factor_down" not in raw

    def _allowed_chain_ids(self) -> set[str]:
        return {self.config.chain_id}

    def _reset_consensus_overrides(self) -> None:
        if self._consensus_lock_enabled():
            configured_schedule = self._locked_target_schedule_name()
        else:
            configured_schedule = self._normalize_target_schedule_name(self.config.target_schedule)
        self._set_target_schedule_name(configured_schedule or self.MAINNET_LOCKED_TARGET_SCHEDULE)
        self._consensus_chain_id = str(self.config.chain_id)
        self._consensus_protocol_version = int(self.config.protocol_version)
        self._consensus_protocol_upgrade_v2_height = int(self.config.protocol_upgrade_v2_height)
        self._consensus_target_block_time = int(self.config.target_block_time)
        self._consensus_max_target = int(self.config.max_target)
        self._consensus_max_adjust_factor_up = float(self.config.max_adjust_factor_up)
        self._consensus_max_adjust_factor_down = float(self.config.max_adjust_factor_down)
        self._consensus_asert_half_life = max(60, int(self.config.asert_half_life))
        self._consensus_mtp_window = max(11, int(self.config.mtp_window))
        self._consensus_max_block_timestamp_step_seconds = max(
            self._consensus_target_block_time * 2,
            int(self.config.max_block_timestamp_step_seconds),
        )

    def _apply_state_consensus_overrides(self, raw: Any, target_schedule: str) -> None:
        if not isinstance(raw, dict):
            return

        self._set_target_schedule_name(target_schedule)

        if "chain_id" in raw:
            chain_id = str(raw.get("chain_id", "")).strip()
            if chain_id:
                self._consensus_chain_id = chain_id

        if "protocol_version" in raw:
            try:
                parsed = int(raw.get("protocol_version"))
                if parsed < 1:
                    raise ValueError
                self._consensus_protocol_version = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'protocol_version' in state file") from exc

        if "protocol_upgrade_v2_height" in raw:
            try:
                parsed = int(raw.get("protocol_upgrade_v2_height"))
                if parsed < 0:
                    raise ValueError
                self._consensus_protocol_upgrade_v2_height = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'protocol_upgrade_v2_height' in state file") from exc

        if "target_block_time" in raw:
            try:
                parsed = int(raw.get("target_block_time"))
                if parsed <= 0:
                    raise ValueError
                self._consensus_target_block_time = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'target_block_time' in state file") from exc

        if "max_target" in raw:
            try:
                parsed = int(raw.get("max_target"))
                if parsed <= 0:
                    raise ValueError
                self._consensus_max_target = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'max_target' in state file") from exc

        if "max_adjust_factor_up" in raw:
            try:
                parsed = float(raw.get("max_adjust_factor_up"))
                if parsed <= 0:
                    raise ValueError
                self._consensus_max_adjust_factor_up = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'max_adjust_factor_up' in state file") from exc

        if "max_adjust_factor_down" in raw:
            try:
                parsed = float(raw.get("max_adjust_factor_down"))
                if parsed <= 0:
                    raise ValueError
                self._consensus_max_adjust_factor_down = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'max_adjust_factor_down' in state file") from exc

        if "asert_half_life" in raw:
            try:
                parsed = int(raw.get("asert_half_life"))
                if parsed < 60:
                    raise ValueError
                self._consensus_asert_half_life = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'asert_half_life' in state file") from exc

        if "mtp_window" in raw:
            try:
                parsed = int(raw.get("mtp_window"))
                if parsed < 11:
                    raise ValueError
                self._consensus_mtp_window = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'mtp_window' in state file") from exc

        if "max_block_timestamp_step_seconds" in raw:
            try:
                parsed = int(raw.get("max_block_timestamp_step_seconds"))
                if parsed < max(2, self._consensus_target_block_time):
                    raise ValueError
                self._consensus_max_block_timestamp_step_seconds = parsed
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'max_block_timestamp_step_seconds' in state file") from exc

    def _validate_state_config(self, raw: Any, target_schedule: str) -> None:
        if not isinstance(raw, dict):
            return

        critical: dict[str, Any] = {
            "symbol": self.config.symbol,
            "pow_algorithm": self.config.pow_algorithm,
            "protocol_version": self.config.protocol_version,
            "coinbase_maturity": self.config.coinbase_maturity,
            "smart_contracts_enabled": self.config.smart_contracts_enabled,
            "nft_market_enabled": self.config.nft_market_enabled,
            "max_contract_payload_bytes": self.config.max_contract_payload_bytes,
            "max_contract_kv_entries": self.config.max_contract_kv_entries,
            "max_contract_key_bytes": self.config.max_contract_key_bytes,
            "max_contract_value_bytes": self.config.max_contract_value_bytes,
        }

        for key, expected in critical.items():
            if key not in raw:
                continue
            value = raw.get(key)
            if isinstance(expected, float):
                try:
                    value_float = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(f"Invalid config value for '{key}' in state file") from exc
                if abs(value_float - expected) > 1e-12:
                    raise ValidationError(
                        f"State config mismatch for '{key}': file={value_float} runtime={expected}"
                    )
                continue

            if value != expected:
                raise ValidationError(f"State config mismatch for '{key}': file={value} runtime={expected}")

        if "chain_id" in raw:
            chain_id = str(raw.get("chain_id", "")).strip()
            if chain_id not in self._allowed_chain_ids():
                raise ValidationError(
                    f"State config mismatch for 'chain_id': file={chain_id} runtime={self.config.chain_id}"
                )

        if "target_schedule" in raw:
            normalized = self._normalize_target_schedule_name(raw.get("target_schedule", ""))
            if not normalized:
                raise ValidationError("Invalid config value for 'target_schedule' in state file")
            if self._consensus_lock_enabled() and normalized != self._locked_target_schedule_name():
                raise ValidationError(
                    f"State config mismatch for 'target_schedule': file={normalized} runtime={self._locked_target_schedule_name()}"
                )

        resolved_schedule = self._resolve_target_schedule_name(target_schedule)
        if self._consensus_lock_enabled() and resolved_schedule != self._locked_target_schedule_name():
            raise ValidationError(
                f"Mainnet consensus lock requires target_schedule={self._locked_target_schedule_name()}"
            )

        if self._consensus_lock_enabled() and "protocol_upgrade_v2_height" in raw:
            try:
                file_upgrade = int(raw.get("protocol_upgrade_v2_height"))
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'protocol_upgrade_v2_height' in state file") from exc
            if file_upgrade != int(self.config.protocol_upgrade_v2_height):
                raise ValidationError(
                    "State config mismatch for 'protocol_upgrade_v2_height': "
                    f"file={file_upgrade} runtime={self.config.protocol_upgrade_v2_height}"
                )

        if resolved_schedule == "window-v2" and "difficulty_window" in raw:
            try:
                window = int(raw.get("difficulty_window"))
            except (TypeError, ValueError) as exc:
                raise ValidationError("Invalid config value for 'difficulty_window' in state file") from exc
            if window != int(self.config.difficulty_window):
                raise ValidationError(
                    f"State config mismatch for 'difficulty_window': file={window} runtime={self.config.difficulty_window}"
                )

    def snapshot(self) -> dict[str, Any]:
        tip = self.tip
        return {
            "height": self.height,
            "tip_hash": tip.block_hash if tip else None,
            "chain_work": int(tip.chain_work) if tip else 0,
            "chain": [block.to_dict() for block in self.chain],
            "mempool": [tx.to_dict() for tx in self.mempool],
            "nft_registry": self.nft_registry,
            "nft_listings": self.nft_listings,
            "contracts": self.contracts,
        }

    @property
    def height(self) -> int:
        return len(self.chain) - 1

    @property
    def tip(self) -> Block | None:
        return self.chain[-1] if self.chain else None

    def initialize(self, genesis_address: str, genesis_supply: int = 0) -> Block:
        if self.chain:
            raise ValidationError("Chain is already initialized")
        self._reset_consensus_overrides()

        if self._consensus_lock_enabled():
            genesis_address = str(self.config.fixed_genesis_address).strip()
            genesis_supply = int(self.config.fixed_genesis_supply)
            timestamp = int(self.config.fixed_genesis_timestamp)
            coinbase_nonce = int(self.config.fixed_genesis_tx_nonce)
            genesis_block_nonce = int(self.config.fixed_genesis_block_nonce)
        else:
            timestamp = self._now()
            coinbase_nonce = secrets.randbits(64)
            genesis_block_nonce = 0

        if genesis_supply < 0:
            raise ValidationError("Genesis supply must be >= 0")
        if genesis_supply > self.config.max_total_supply:
            raise ValidationError(
                f"Genesis supply exceeds max_total_supply ({self.config.max_total_supply})"
            )
        self._validate_address_format(genesis_address)

        coinbase = Transaction(
            version=self.tx_version_for_height(0),
            timestamp=timestamp,
            nonce=coinbase_nonce,
            inputs=[],
            outputs=[TxOutput(amount=genesis_supply, address=genesis_address)],
        )
        coinbase.txid = coinbase.compute_txid()

        block = Block(
            index=0,
            prev_hash="0" * 64,
            timestamp=timestamp,
            target=self.config.initial_target,
            nonce=genesis_block_nonce,
            merkle_root=merkle_root([coinbase.txid]),
            miner=genesis_address,
            transactions=[coinbase],
            chain_work=block_work(self.config.initial_target),
        )
        if self._consensus_lock_enabled():
            digest = self._pow_digest(block)
            if int.from_bytes(digest, "big") >= block.target:
                raise ValidationError("Configured fixed genesis nonce does not satisfy target")
            block.block_hash = digest.hex()

            expected_hash = str(self.config.fixed_genesis_hash).strip().lower()
            if expected_hash and block.block_hash != expected_hash:
                raise ValidationError("Configured fixed genesis hash does not match runtime genesis template")
            expected_txid = str(self.config.fixed_genesis_txid).strip().lower()
            if expected_txid and coinbase.txid.lower() != expected_txid:
                raise ValidationError("Configured fixed genesis txid does not match runtime genesis template")
        else:
            block.block_hash = self._mine_block_hash(block)

        self.chain = [block]
        self.utxos = {}
        # Genesis allocation is treated as pre-mine allocation, not a mineable subsidy output.
        self._add_outputs(coinbase, self.utxos, block_height=0, coinbase=False)
        self.nft_registry = {}
        self.nft_listings = {}
        self.contracts = {}
        self.mempool = []
        self.save()
        return block

    def _pow_digest(
        self,
        block: Block,
        seed_words: tuple[int, int, int, int, int, int, int, int] | None = None,
    ) -> bytes:
        if seed_words is None:
            seed_words = seed_words_from_block(block, self._consensus_chain_id)
        return hash_from_seed_words(seed_words, block.nonce)

    def _get_gpu_miner(self) -> OpenCLMiner | None:
        if self._gpu_miner is not None:
            return self._gpu_miner
        if self._gpu_probe_attempted:
            return None

        self._gpu_probe_attempted = True
        try:
            self._gpu_miner = OpenCLMiner()
        except Exception:
            self._gpu_miner = None
        return self._gpu_miner

    def _mine_block_hash(
        self,
        block: Block,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        progress_interval: int = 2500,
        mining_backend: str = "auto",
    ) -> str:
        backend = mining_backend.lower().strip()
        if backend not in {"auto", "gpu", "cpu"}:
            raise ValidationError("Mining backend must be one of: auto, gpu, cpu")

        seed_words = seed_words_from_block(block, self._consensus_chain_id)
        started = time.perf_counter()
        attempts = 0

        if backend in {"auto", "gpu"}:
            gpu_miner = self._get_gpu_miner()
            if gpu_miner is None and backend == "gpu":
                raise ValidationError(
                    "GPU mining requested but no OpenCL GPU backend is available. Install pyopencl and GPU drivers."
                )

            if gpu_miner is not None:
                try:
                    found_nonce, gpu_attempts = gpu_miner.search_nonce(
                        seed_words=seed_words,
                        target_words=target_to_words(block.target),
                        start_nonce=block.nonce,
                        batch_size=self.config.gpu_batch_size,
                        stop_requested=stop_requested,
                        progress_callback=progress_callback,
                    )
                    attempts += gpu_attempts
                    if found_nonce is not None:
                        block.nonce = found_nonce
                        digest = self._pow_digest(block, seed_words=seed_words)
                        if progress_callback:
                            elapsed = max(0.0001, time.perf_counter() - started)
                            progress_callback(
                                {
                                    "backend": "gpu",
                                    "nonce": block.nonce,
                                    "attempts": attempts,
                                    "elapsed": elapsed,
                                    "hash_rate": attempts / elapsed,
                                    "hash_preview": digest.hex()[:16],
                                    "solved": True,
                                }
                            )
                        return digest.hex()

                    if stop_requested and stop_requested():
                        raise MiningInterruptedError("Mining interrupted by user")
                except MiningInterruptedError:
                    raise
                except Exception as exc:
                    if backend == "gpu":
                        raise ValidationError(f"GPU mining failed: {exc}") from exc

                    if progress_callback:
                        progress_callback(
                            {
                                "backend": "cpu",
                                "nonce": block.nonce,
                                "attempts": attempts,
                                "elapsed": max(0.0001, time.perf_counter() - started),
                                "hash_rate": 0.0,
                                "hash_preview": "gpu-fallback",
                            }
                        )

        while True:
            if stop_requested and stop_requested():
                raise MiningInterruptedError("Mining interrupted by user")

            digest = self._pow_digest(block, seed_words=seed_words)
            attempts += 1

            if progress_callback and attempts % progress_interval == 0:
                elapsed = max(0.0001, time.perf_counter() - started)
                progress_callback(
                    {
                        "backend": "cpu",
                        "nonce": block.nonce,
                        "attempts": attempts,
                        "elapsed": elapsed,
                        "hash_rate": attempts / elapsed,
                        "hash_preview": digest.hex()[:16],
                    }
                )

            if int.from_bytes(digest, "big") < block.target:
                if progress_callback:
                    elapsed = max(0.0001, time.perf_counter() - started)
                    progress_callback(
                        {
                            "backend": "cpu",
                            "nonce": block.nonce,
                            "attempts": attempts,
                            "elapsed": elapsed,
                            "hash_rate": attempts / elapsed,
                            "hash_preview": digest.hex()[:16],
                            "solved": True,
                        }
                    )
                return digest.hex()
            block.nonce += 1
            if block.nonce >= 1 << 64:
                block.nonce = 0

    def _next_target_from_history(
        self,
        history: list[Block],
        timestamp: int,
        target_schedule: str | None = None,
    ) -> int:
        if not history:
            return self.config.initial_target

        prev = history[-1]
        schedule_name = self._resolve_target_schedule_name(target_schedule)

        if self._is_asert_schedule_name(schedule_name):
            max_step = max(
                self._consensus_target_block_time * 2,
                int(self._consensus_max_block_timestamp_step_seconds),
            )
            delta = int(timestamp) - int(prev.timestamp)
            bounded_delta = max(-max_step, min(max_step, delta))
            exponent = (bounded_delta - self._consensus_target_block_time) / float(
                max(60, self._consensus_asert_half_life)
            )
            ratio = math.exp2(exponent)
        else:
            if self._is_legacy_schedule_name(schedule_name):
                lookback = 1
            else:
                window = max(2, int(self.config.difficulty_window))
                lookback = min(len(history), window - 1)
            anchor = history[-lookback]

            expected_span = max(1, self._consensus_target_block_time * lookback)
            actual_span = max(1, timestamp - anchor.timestamp)
            ratio = actual_span / expected_span

        ratio = min(self._consensus_max_adjust_factor_up, ratio)
        ratio = max(self._consensus_max_adjust_factor_down, ratio)
        target = int(prev.target * ratio)
        target = max(1, target)
        target = min(self._consensus_max_target, target)
        return target

    def _validate_block_timestamp(
        self,
        timestamp: int,
        median_past: int,
        context: str,
        prev_timestamp: int | None = None,
    ) -> None:
        now_ts = self._now()

        if timestamp <= median_past:
            raise ValidationError(f"{context} timestamp is too old")

        if prev_timestamp is not None:
            if timestamp < prev_timestamp:
                raise ValidationError(f"{context} timestamp regresses")
            max_step = max(
                self._consensus_target_block_time * 2,
                int(self._consensus_max_block_timestamp_step_seconds),
            )
            step = timestamp - prev_timestamp
            # Enforce per-block jump bounds only while tip-time is still near local wall-clock.
            if step > max_step and (now_ts - prev_timestamp) <= max_step:
                raise ValidationError(f"{context} timestamp jump is too large")

        max_allowed = now_ts + self.config.max_future_block_seconds
        if timestamp > max_allowed:
            raise ValidationError(f"{context} timestamp is too far in the future")

    def next_target(self, timestamp: int) -> int:
        return self._next_target_from_history(
            self.chain,
            timestamp,
            target_schedule=self._target_schedule_name(),
        )

    def block_reward(self, height: int) -> int:
        halvings = height // self.config.halving_interval
        if halvings >= 64:
            return 0
        return self.config.initial_block_reward >> halvings

    def protocol_version_for_height(self, height: int) -> int:
        base = max(1, int(self._consensus_protocol_version))
        upgrade_height = int(self._consensus_protocol_upgrade_v2_height)
        if height >= upgrade_height:
            return max(base, 2)
        return base

    def tx_version_for_height(self, height: int) -> int:
        protocol = self.protocol_version_for_height(height)
        if protocol >= 2:
            return 2
        return 1

    def active_protocol_version(self) -> int:
        if self.height < 0:
            return self.protocol_version_for_height(0)
        return self.protocol_version_for_height(self.height)

    def next_protocol_version(self) -> int:
        return self.protocol_version_for_height(max(0, self.height + 1))

    @staticmethod
    def _coinbase_amount_from_block(block: Block) -> int:
        if not block.transactions:
            return 0
        coinbase = block.transactions[0]
        if not coinbase.is_coinbase():
            return 0
        return sum(output.amount for output in coinbase.outputs)

    def issued_supply(self, chain_view: list[Block] | None = None) -> int:
        blocks = self.chain if chain_view is None else chain_view
        issued = 0
        for block in blocks:
            issued += self._coinbase_amount_from_block(block)
        return issued

    def median_time_past(self, count: int = 11) -> int:
        if not self.chain:
            return 0
        times = [block.timestamp for block in self.chain[-count:]]
        times.sort()
        return times[len(times) // 2]

    @staticmethod
    def _is_hex_string(value: str, expected_len: int | None = None) -> bool:
        text = value.strip().lower()
        if expected_len is not None and len(text) != expected_len:
            return False
        if not text:
            return False
        try:
            int(text, 16)
        except ValueError:
            return False
        return True

    def _effective_min_relay_fee_rate(self) -> float:
        mempool_floor = max(0.0, float(self.config.min_mempool_fee_rate))
        standard_floor = max(0.0, float(getattr(self.config, "min_standard_tx_fee_rate", 0.0)))
        return max(mempool_floor, standard_floor)

    def _min_dust_output(self) -> int:
        return max(1, int(getattr(self.config, "min_dust_output", 1)))

    def _validate_tx_outputs(
        self,
        tx: Transaction,
        allow_zero_amount: bool = False,
        enforce_standard_policy: bool = False,
    ) -> int:
        if len(tx.outputs) == 0:
            raise ValidationError("Transaction has no outputs")
        if len(tx.outputs) > self.config.max_tx_outputs:
            raise ValidationError(f"Transaction has too many outputs (max {self.config.max_tx_outputs})")

        total_out = 0
        dust_floor = self._min_dust_output()
        for output in tx.outputs:
            if output.amount < 0:
                raise ValidationError("Output amount must not be negative")
            if output.amount == 0 and not allow_zero_amount:
                raise ValidationError("Output amount must be positive")
            if enforce_standard_policy and output.amount > 0 and output.amount < dust_floor:
                raise ValidationError(f"Dust output below policy minimum ({dust_floor})")
            self._validate_address_format(output.address)
            total_out += output.amount
        return total_out

    def _validate_address_format(self, address: str) -> None:
        prefix = self.config.symbol
        if not address.startswith(prefix):
            raise ValidationError("Address has wrong prefix")
        body = address[len(prefix) :]
        if len(body) != 40:
            raise ValidationError("Address must contain 40 hex chars after prefix")
        try:
            int(body, 16)
        except ValueError as exc:
            raise ValidationError("Address contains non-hex characters") from exc

    def _contracts_enabled(self) -> bool:
        return bool(getattr(self.config, "smart_contracts_enabled", True))

    def _nft_market_enabled(self) -> bool:
        return bool(getattr(self.config, "nft_market_enabled", True))

    @staticmethod
    def _identifier_valid(text: str, max_len: int = 96) -> bool:
        if not text or len(text) > max_len:
            return False
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/")
        return all(char in allowed for char in text)

    def _validate_identifier(self, value: Any, field_name: str, max_len: int = 96) -> str:
        text = str(value).strip()
        if not self._identifier_valid(text, max_len=max_len):
            raise ValidationError(f"Invalid {field_name}")
        return text

    def _normalize_contract_payload(self, tx: Transaction) -> dict[str, Any] | None:
        raw = tx.contract
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValidationError("Contract payload must be a JSON object")
        try:
            normalized = json.loads(json.dumps(raw, sort_keys=True))
        except Exception as exc:
            raise ValidationError("Contract payload cannot be canonicalized") from exc
        if not isinstance(normalized, dict):
            raise ValidationError("Contract payload must be a JSON object")
        payload_size = len(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        max_payload_bytes = max(128, int(getattr(self.config, "max_contract_payload_bytes", 4_096)))
        if payload_size > max_payload_bytes:
            raise ValidationError(f"Contract payload exceeds size limit ({max_payload_bytes} bytes)")
        return normalized

    def _normalize_contract_state_map(self, raw_state: Any) -> dict[str, str]:
        if raw_state is None:
            return {}
        if not isinstance(raw_state, dict):
            raise ValidationError("Contract state must be an object")
        normalized: dict[str, str] = {}
        max_entries = max(1, int(getattr(self.config, "max_contract_kv_entries", 1_024)))
        max_key_len = max(8, int(getattr(self.config, "max_contract_key_bytes", 64)))
        max_value_len = max(16, int(getattr(self.config, "max_contract_value_bytes", 1_024)))
        for key, value in raw_state.items():
            key_text = self._validate_identifier(key, "contract key", max_len=max_key_len)
            value_text = str(value)
            if len(value_text.encode("utf-8")) > max_value_len:
                raise ValidationError("Contract value exceeds size limit")
            normalized[key_text] = value_text
            if len(normalized) > max_entries:
                raise ValidationError("Contract state exceeds max entries")
        return normalized

    def _apply_nft_contract_action(
        self,
        payload: dict[str, Any],
        tx: Transaction,
        sender_address: str,
        block_height: int,
        nft_view: dict[str, dict[str, Any]],
        nft_listing_view: dict[str, dict[str, Any]],
    ) -> None:
        if not self._nft_market_enabled():
            raise ValidationError("NFT market is disabled by config")

        action = str(payload.get("action", "")).strip().lower()
        token_id = self._validate_identifier(payload.get("token_id", ""), "token_id", max_len=128)

        if action == "mint":
            if token_id in nft_view:
                raise ValidationError("NFT token already exists")
            metadata_uri = str(payload.get("metadata_uri", "")).strip()
            if not metadata_uri:
                raise ValidationError("NFT mint requires metadata_uri")
            if len(metadata_uri.encode("utf-8")) > 512:
                raise ValidationError("NFT metadata_uri exceeds size limit")
            owner_address = str(payload.get("owner", sender_address)).strip()
            self._validate_address_format(owner_address)
            nft_view[token_id] = {
                "token_id": token_id,
                "creator": sender_address,
                "owner": owner_address,
                "metadata_uri": metadata_uri,
                "mint_height": int(block_height),
                "mint_txid": tx.txid,
                "updated_height": int(block_height),
                "updated_txid": tx.txid,
            }
            nft_listing_view.pop(token_id, None)
            return

        existing = nft_view.get(token_id)
        if existing is None:
            raise ValidationError("NFT token not found")
        owner = str(existing.get("owner", "")).strip()

        if action == "list":
            if owner != sender_address:
                raise ValidationError("Only NFT owner can list token")
            if token_id in nft_listing_view:
                raise ValidationError("NFT token is already listed")
            try:
                price = int(payload.get("price"))
            except Exception as exc:
                raise ValidationError("NFT listing requires integer price") from exc
            if price <= 0:
                raise ValidationError("NFT listing price must be positive")
            nft_listing_view[token_id] = {
                "token_id": token_id,
                "seller": sender_address,
                "price": price,
                "listed_height": int(block_height),
                "listed_txid": tx.txid,
            }
            return

        if action == "cancel":
            listing = nft_listing_view.get(token_id)
            if listing is None:
                raise ValidationError("NFT token is not listed")
            seller = str(listing.get("seller", "")).strip()
            if seller != sender_address:
                raise ValidationError("Only listing seller can cancel")
            del nft_listing_view[token_id]
            return

        if action == "buy":
            listing = nft_listing_view.get(token_id)
            if listing is None:
                raise ValidationError("NFT token is not listed")
            seller = str(listing.get("seller", "")).strip()
            if seller == sender_address:
                raise ValidationError("Listing seller cannot buy own NFT")
            if owner != seller:
                raise ValidationError("NFT listing seller is no longer owner")
            price = int(listing.get("price", 0))
            paid_to_seller = sum(int(output.amount) for output in tx.outputs if output.address == seller)
            if paid_to_seller < price:
                raise ValidationError("NFT buy transaction does not pay seller listing price")
            updated = dict(existing)
            updated["owner"] = sender_address
            updated["updated_height"] = int(block_height)
            updated["updated_txid"] = tx.txid
            nft_view[token_id] = updated
            del nft_listing_view[token_id]
            return

        raise ValidationError("Unsupported NFT contract action")

    def _apply_smart_contract_action(
        self,
        payload: dict[str, Any],
        tx: Transaction,
        sender_address: str,
        block_height: int,
        contract_view: dict[str, dict[str, Any]],
    ) -> None:
        if not self._contracts_enabled():
            raise ValidationError("Smart contracts are disabled by config")

        action = str(payload.get("action", "")).strip().lower()
        if action == "deploy":
            template = str(payload.get("template", "")).strip().lower()
            if template != "kv_v1":
                raise ValidationError("Unsupported smart contract template")
            contract_id = str(payload.get("contract_id", "")).strip() or tx.txid
            contract_id = self._validate_identifier(contract_id, "contract_id", max_len=128)
            if contract_id in contract_view:
                raise ValidationError("Smart contract already exists")
            init_state = self._normalize_contract_state_map(payload.get("init", {}))
            contract_view[contract_id] = {
                "contract_id": contract_id,
                "template": template,
                "owner": sender_address,
                "state": init_state,
                "created_height": int(block_height),
                "created_txid": tx.txid,
                "updated_height": int(block_height),
                "updated_txid": tx.txid,
            }
            return

        if action == "call":
            contract_id = self._validate_identifier(payload.get("contract_id", ""), "contract_id", max_len=128)
            existing = contract_view.get(contract_id)
            if existing is None:
                raise ValidationError("Smart contract not found")
            template = str(existing.get("template", "")).strip().lower()
            if template != "kv_v1":
                raise ValidationError("Unsupported deployed contract template")
            owner = str(existing.get("owner", "")).strip()
            if owner != sender_address:
                raise ValidationError("Only contract owner can call this method")

            method = str(payload.get("method", "")).strip().lower()
            args = payload.get("args", {})
            if not isinstance(args, dict):
                raise ValidationError("Contract call args must be an object")
            key = self._validate_identifier(args.get("key", ""), "contract key", max_len=max(8, int(getattr(self.config, "max_contract_key_bytes", 64))))
            state_map = self._normalize_contract_state_map(existing.get("state", {}))
            if method == "set":
                value_text = str(args.get("value", ""))
                if len(value_text.encode("utf-8")) > max(16, int(getattr(self.config, "max_contract_value_bytes", 1_024))):
                    raise ValidationError("Contract value exceeds size limit")
                state_map[key] = value_text
            elif method in {"delete", "del"}:
                state_map.pop(key, None)
            else:
                raise ValidationError("Unsupported contract call method")

            max_entries = max(1, int(getattr(self.config, "max_contract_kv_entries", 1_024)))
            if len(state_map) > max_entries:
                raise ValidationError("Contract state exceeds max entries")

            updated_contract = dict(existing)
            updated_contract["state"] = state_map
            updated_contract["updated_height"] = int(block_height)
            updated_contract["updated_txid"] = tx.txid
            contract_view[contract_id] = updated_contract
            return

        raise ValidationError("Unsupported smart contract action")

    def _apply_contract_action(
        self,
        payload: dict[str, Any],
        tx: Transaction,
        sender_address: str,
        block_height: int,
        nft_view: dict[str, dict[str, Any]],
        nft_listing_view: dict[str, dict[str, Any]],
        contract_view: dict[str, dict[str, Any]],
    ) -> None:
        kind = str(payload.get("kind", "")).strip().lower()
        if not kind:
            raise ValidationError("Contract payload missing kind")
        if kind == "nft":
            self._apply_nft_contract_action(payload, tx, sender_address, block_height, nft_view, nft_listing_view)
            return
        if kind in {"sc", "contract"}:
            self._apply_smart_contract_action(payload, tx, sender_address, block_height, contract_view)
            return
        raise ValidationError("Unsupported contract kind")

    def validate_transaction(
        self,
        tx: Transaction,
        utxo_view: dict[str, dict[str, Any]],
        allow_coinbase: bool = False,
        for_mempool: bool = False,
        block_height: int | None = None,
        nft_view: dict[str, dict[str, Any]] | None = None,
        nft_listing_view: dict[str, dict[str, Any]] | None = None,
        contract_view: dict[str, dict[str, Any]] | None = None,
    ) -> int:
        if block_height is None:
            block_height = max(0, self.height + 1)
        if nft_view is None:
            nft_view = self.nft_registry
        if nft_listing_view is None:
            nft_listing_view = self.nft_listings
        if contract_view is None:
            contract_view = self.contracts

        expected_tx_version = self.tx_version_for_height(block_height)
        if tx.version != expected_tx_version:
            raise ValidationError(
                f"Unsupported transaction version {tx.version} at height {block_height}; expected {expected_tx_version}"
            )
        if tx.timestamp <= 0:
            raise ValidationError("Transaction timestamp is invalid")
        if for_mempool:
            now = self._now()
            if tx.timestamp > now + self.config.max_future_tx_seconds:
                raise ValidationError("Transaction timestamp is too far in the future")
            if tx.timestamp < now - self.config.max_mempool_tx_age_seconds:
                raise ValidationError("Transaction is too old for mempool")
            max_standard_vbytes = max(80, int(getattr(self.config, "max_standard_tx_virtual_bytes", 100_000)))
            tx_vbytes = self._tx_virtual_size(tx)
            if tx_vbytes > max_standard_vbytes:
                raise ValidationError(f"Transaction exceeds standard size limit ({max_standard_vbytes} vbytes)")

        contract_payload = self._normalize_contract_payload(tx)

        computed = tx.compute_txid()
        if tx.txid != computed:
            raise ValidationError("Transaction ID mismatch")

        allow_zero_outputs = tx.is_coinbase() and allow_coinbase
        total_out = self._validate_tx_outputs(
            tx,
            allow_zero_amount=allow_zero_outputs,
            enforce_standard_policy=for_mempool and not tx.is_coinbase(),
        )

        if tx.is_coinbase():
            if not allow_coinbase:
                raise ValidationError("Coinbase transaction is not allowed in this context")
            if contract_payload is not None:
                raise ValidationError("Coinbase transaction must not contain contract payload")
            return 0

        if len(tx.inputs) == 0:
            raise ValidationError("Regular transaction requires inputs")
        if len(tx.inputs) > self.config.max_tx_inputs:
            raise ValidationError(f"Transaction has too many inputs (max {self.config.max_tx_inputs})")

        seen_inputs: set[str] = set()
        total_in = 0
        sighash = tx.signing_hash()
        input_owners: set[str] = set()

        for tx_input in tx.inputs:
            if tx_input.index < 0:
                raise ValidationError("Input index must be >= 0")
            if not self._is_hex_string(tx_input.txid, expected_len=64):
                raise ValidationError("Input references malformed txid")

            key = f"{tx_input.txid}:{tx_input.index}"
            if key in seen_inputs:
                raise ValidationError("Duplicate input in transaction")
            seen_inputs.add(key)

            prev_out = utxo_view.get(key)
            if prev_out is None:
                raise ValidationError(f"Missing UTXO: {key}")
            if not tx_input.signature or not tx_input.pubkey:
                raise ValidationError("Missing signature or pubkey")

            pubkey_text = str(tx_input.pubkey).strip().lower()
            signature_text = str(tx_input.signature).strip().lower()
            if for_mempool:
                if not self._is_hex_string(signature_text, expected_len=128):
                    raise ValidationError("Non-standard signature encoding")
                if not self._is_hex_string(pubkey_text, expected_len=66) or pubkey_text[:2] not in {"02", "03"}:
                    raise ValidationError("Non-standard pubkey encoding")

            if bool(prev_out.get("coinbase", False)):
                maturity = max(0, int(getattr(self.config, "coinbase_maturity", 0)))
                if maturity > 0:
                    origin_height_raw = prev_out.get("height")
                    try:
                        origin_height = int(origin_height_raw)
                    except (TypeError, ValueError) as exc:
                        raise ValidationError("Coinbase UTXO metadata missing height") from exc
                    confirmations = int(block_height) - origin_height
                    if confirmations < maturity:
                        raise ValidationError(
                            f"Coinbase UTXO is immature ({confirmations}/{maturity} confirmations)"
                        )

            try:
                expected_addr = address_from_public_key(pubkey_text, self.config.symbol)
            except Exception as exc:
                raise ValidationError("Invalid public key encoding") from exc
            if expected_addr != prev_out["address"]:
                raise ValidationError("Input pubkey does not match UTXO owner")
            input_owners.add(expected_addr)

            if not verify_signature(pubkey_text, sighash, signature_text):
                raise ValidationError("Invalid input signature")

            total_in += int(prev_out["amount"])

        if total_in < total_out:
            raise ValidationError("Inputs do not cover outputs")

        fee = total_in - total_out
        if fee < self.config.min_tx_fee:
            raise ValidationError("Transaction fee is below minimum")
        if for_mempool:
            fee_rate = fee / max(1, self._tx_virtual_size(tx))
            if fee_rate + 1e-12 < self._effective_min_relay_fee_rate():
                raise ValidationError("Transaction feerate is below mempool minimum")

        if contract_payload is not None:
            if len(input_owners) != 1:
                raise ValidationError("Contract transaction must spend UTXOs from exactly one owner address")
            sender_address = next(iter(input_owners))
            self._apply_contract_action(
                contract_payload,
                tx,
                sender_address,
                int(block_height),
                nft_view,
                nft_listing_view,
                contract_view,
            )

        return fee

    @staticmethod
    def _consume_inputs(tx: Transaction, utxo_view: dict[str, dict[str, Any]]) -> None:
        for tx_input in tx.inputs:
            key = f"{tx_input.txid}:{tx_input.index}"
            if key in utxo_view:
                del utxo_view[key]

    @staticmethod
    def _add_outputs(
        tx: Transaction,
        utxo_view: dict[str, dict[str, Any]],
        *,
        block_height: int | None,
        coinbase: bool = False,
    ) -> None:
        for index, output in enumerate(tx.outputs):
            key = f"{tx.txid}:{index}"
            utxo_view[key] = {
                "amount": int(output.amount),
                "address": output.address,
                "coinbase": bool(coinbase),
                "height": None if block_height is None else int(block_height),
            }

    def _apply_block_transactions(
        self,
        block: Block,
        utxo_start: dict[str, dict[str, Any]],
        nft_start: dict[str, dict[str, Any]],
        nft_listing_start: dict[str, dict[str, Any]],
        contract_start: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        if not block.transactions:
            raise ValidationError("Block must contain at least one transaction")

        coinbase = block.transactions[0]
        if not coinbase.is_coinbase():
            raise ValidationError("First transaction must be coinbase")
        self.validate_transaction(
            coinbase,
            utxo_start,
            allow_coinbase=True,
            block_height=block.index,
            nft_view=nft_start,
            nft_listing_view=nft_listing_start,
            contract_view=contract_start,
        )

        for tx in block.transactions[1:]:
            if tx.is_coinbase():
                raise ValidationError("Only the first transaction can be coinbase")

        utxo_view = copy.deepcopy(utxo_start)
        nft_view = copy.deepcopy(nft_start)
        nft_listing_view = copy.deepcopy(nft_listing_start)
        contract_view = copy.deepcopy(contract_start)
        total_fees = 0

        for tx in block.transactions[1:]:
            fee = self.validate_transaction(
                tx,
                utxo_view,
                block_height=block.index,
                nft_view=nft_view,
                nft_listing_view=nft_listing_view,
                contract_view=contract_view,
            )
            total_fees += fee
            self._consume_inputs(tx, utxo_view)
            self._add_outputs(tx, utxo_view, block_height=block.index, coinbase=False)

        coinbase_amount = sum(output.amount for output in coinbase.outputs)
        allowed_reward = self.block_reward(block.index) + total_fees
        if coinbase_amount > allowed_reward:
            raise ValidationError("Coinbase exceeds allowed reward")

        self._add_outputs(coinbase, utxo_view, block_height=block.index, coinbase=True)
        nft_start.clear()
        nft_start.update(nft_view)
        nft_listing_start.clear()
        nft_listing_start.update(nft_listing_view)
        contract_start.clear()
        contract_start.update(contract_view)
        return utxo_view, total_fees, coinbase_amount

    @staticmethod
    def _median_time_for_chain(chain: list[Block], count: int = 11) -> int:
        if not chain:
            return 0
        times = [block.timestamp for block in chain[-count:]]
        times.sort()
        return times[len(times) // 2]

    def _next_target_for_chain(
        self,
        chain: list[Block],
        timestamp: int,
        target_schedule: str | None = None,
    ) -> int:
        return self._next_target_from_history(
            chain,
            timestamp,
            target_schedule=target_schedule,
        )

    def _validate_genesis_block(self, genesis: Block) -> dict[str, dict[str, Any]]:
        if genesis.index != 0:
            raise ValidationError("Genesis block index must be 0")
        if genesis.prev_hash != "0" * 64:
            raise ValidationError("Genesis previous hash is invalid")
        if genesis.target <= 0:
            raise ValidationError("Genesis target must be positive")
        if genesis.target > self._consensus_max_target:
            raise ValidationError("Genesis target exceeds configured maximum")
        if genesis.chain_work != block_work(genesis.target):
            raise ValidationError("Genesis chain work is invalid")
        if genesis.timestamp <= 0:
            raise ValidationError("Genesis timestamp is invalid")
        if genesis.timestamp > self._now() + self.config.max_future_block_seconds:
            raise ValidationError("Genesis timestamp is too far in the future")

        if not genesis.transactions:
            raise ValidationError("Genesis block must include coinbase transaction")
        if len(genesis.transactions) != 1:
            raise ValidationError("Genesis block must only contain the coinbase transaction")

        coinbase = genesis.transactions[0]
        if not coinbase.is_coinbase():
            raise ValidationError("Genesis transaction must be coinbase")
        self.validate_transaction(coinbase, {}, allow_coinbase=True, block_height=0)

        txids = [tx.txid for tx in genesis.transactions]
        if genesis.merkle_root != merkle_root(txids):
            raise ValidationError("Genesis merkle root mismatch")

        digest = self._pow_digest(genesis)
        if int.from_bytes(digest, "big") >= genesis.target:
            raise ValidationError("Genesis proof of work does not satisfy target")
        if digest.hex() != genesis.block_hash:
            raise ValidationError("Genesis block hash mismatch")

        if self._consensus_lock_enabled():
            expected_hash = str(self.config.fixed_genesis_hash).strip().lower()
            expected_txid = str(self.config.fixed_genesis_txid).strip().lower()
            expected_timestamp = int(self.config.fixed_genesis_timestamp)
            expected_address = str(self.config.fixed_genesis_address).strip()
            expected_supply = int(self.config.fixed_genesis_supply)
            expected_tx_nonce = int(self.config.fixed_genesis_tx_nonce)
            expected_block_nonce = int(self.config.fixed_genesis_block_nonce)

            if expected_hash and genesis.block_hash.lower() != expected_hash:
                raise ValidationError("Genesis hash does not match locked mainnet genesis")
            if expected_txid and coinbase.txid.lower() != expected_txid:
                raise ValidationError("Genesis txid does not match locked mainnet genesis")
            if genesis.timestamp != expected_timestamp:
                raise ValidationError("Genesis timestamp does not match locked mainnet genesis")
            if genesis.nonce != expected_block_nonce:
                raise ValidationError("Genesis nonce does not match locked mainnet genesis")
            if coinbase.nonce != expected_tx_nonce:
                raise ValidationError("Genesis coinbase nonce does not match locked mainnet genesis")
            if len(coinbase.outputs) != 1:
                raise ValidationError("Locked mainnet genesis must have exactly one coinbase output")
            output = coinbase.outputs[0]
            if output.address != expected_address:
                raise ValidationError("Genesis payout address does not match locked mainnet genesis")
            if int(output.amount) != expected_supply:
                raise ValidationError("Genesis supply does not match locked mainnet genesis")

        utxos: dict[str, dict[str, Any]] = {}
        # Genesis allocation is treated as pre-mine allocation, not a mineable subsidy output.
        self._add_outputs(coinbase, utxos, block_height=0, coinbase=False)
        return utxos

    def validate_chain_blocks(
        self,
        blocks: list[Block],
        target_schedule: str | None = None,
    ) -> tuple[
        list[Block],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        if not blocks:
            raise ValidationError("Candidate chain is empty")

        schedule_name = self._resolve_target_schedule_name(target_schedule)
        if self._consensus_lock_enabled() and schedule_name != self._locked_target_schedule_name():
            raise ValidationError(
                f"Mainnet consensus lock requires target_schedule={self._locked_target_schedule_name()}"
            )
        mtp_window = self._mtp_window_for_schedule(schedule_name)
        normalized = [Block.from_dict(block.to_dict()) for block in blocks]

        chain_view: list[Block] = [normalized[0]]
        utxo_view = self._validate_genesis_block(normalized[0])
        nft_view: dict[str, dict[str, Any]] = {}
        nft_listing_view: dict[str, dict[str, Any]] = {}
        contract_view: dict[str, dict[str, Any]] = {}
        issued_supply = self._coinbase_amount_from_block(normalized[0])
        if issued_supply > self.config.max_total_supply:
            raise ValidationError("Total supply exceeds max_total_supply at genesis")

        for block in normalized[1:]:
            prev = chain_view[-1]

            if block.index != prev.index + 1:
                raise ValidationError("Wrong block index in candidate chain")
            if block.prev_hash != prev.block_hash:
                raise ValidationError("Wrong previous block hash in candidate chain")
            self._validate_block_timestamp(
                block.timestamp,
                self._median_time_for_chain(chain_view, count=mtp_window),
                context="Candidate block",
                prev_timestamp=prev.timestamp,
            )

            expected_target = self._next_target_for_chain(
                chain_view,
                block.timestamp,
                target_schedule=schedule_name,
            )
            if block.target != expected_target:
                raise ValidationError("Candidate block target mismatch")

            txids = [tx.txid for tx in block.transactions]
            if block.merkle_root != merkle_root(txids):
                raise ValidationError("Candidate block merkle root mismatch")

            digest = self._pow_digest(block)
            if int.from_bytes(digest, "big") >= block.target:
                raise ValidationError("Candidate block PoW does not satisfy target")
            if digest.hex() != block.block_hash:
                raise ValidationError("Candidate block hash mismatch")

            expected_work = prev.chain_work + block_work(block.target)
            if block.chain_work != expected_work:
                raise ValidationError("Candidate block chain work mismatch")

            utxo_view, total_fees, coinbase_amount = self._apply_block_transactions(
                block,
                utxo_view,
                nft_view,
                nft_listing_view,
                contract_view,
            )

            remaining = max(0, self.config.max_total_supply - issued_supply)
            max_subsidy = min(self.block_reward(block.index), remaining)
            allowed_coinbase = total_fees + max_subsidy
            if coinbase_amount > allowed_coinbase:
                raise ValidationError("Coinbase exceeds remaining supply cap")

            minted = max(0, coinbase_amount - total_fees)
            issued_supply += minted
            if issued_supply > self.config.max_total_supply:
                raise ValidationError("Total supply exceeds max_total_supply")
            chain_view.append(block)

        return chain_view, utxo_view, nft_view, nft_listing_view, contract_view

    @staticmethod
    def _tx_input_outpoint(tx_input: TxInput) -> str:
        return f"{tx_input.txid}:{tx_input.index}"

    def _tx_virtual_size(self, tx: Transaction) -> int:
        # Lightweight vsize approximation for fee-rate policy.
        size = 12
        for tx_input in tx.inputs:
            sig_bytes = max(0, len(tx_input.signature) // 2) if tx_input.signature else 0
            pub_bytes = max(0, len(tx_input.pubkey) // 2) if tx_input.pubkey else 0
            size += 41 + sig_bytes + pub_bytes
        for output in tx.outputs:
            size += 9 + len(output.address.encode("utf-8"))
        return max(80, size)

    def _collect_ancestors(self, txid: str, parents: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        stack = list(parents.get(txid, set()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(parents.get(current, set()))
        return visited

    def _collect_descendants(self, txid: str, children: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        stack = list(children.get(txid, set()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(children.get(current, set()))
        return visited

    def _topological_order_subset(
        self,
        txids: set[str],
        parents: dict[str, set[str]],
    ) -> list[str]:
        ordered: list[str] = []
        marked: set[str] = set()

        def visit(current: str) -> None:
            if current in marked or current not in txids:
                return
            marked.add(current)
            for parent in sorted(parents.get(current, set())):
                if parent in txids:
                    visit(parent)
            ordered.append(current)

        for txid in sorted(txids):
            visit(txid)
        return ordered

    def _build_mempool_index(self, txs: list[Transaction]) -> dict[str, Any]:
        by_txid: dict[str, Transaction] = {tx.txid: tx for tx in txs}
        parents: dict[str, set[str]] = {txid: set() for txid in by_txid}
        children: dict[str, set[str]] = {txid: set() for txid in by_txid}
        spending_index: dict[str, str] = {}
        fees: dict[str, int] = {}
        vbytes: dict[str, int] = {}
        fee_rates: dict[str, float] = {}

        for tx in txs:
            txid = tx.txid
            vbytes[txid] = self._tx_virtual_size(tx)
            total_in = 0
            valid_inputs = True

            for tx_input in tx.inputs:
                outpoint = self._tx_input_outpoint(tx_input)
                existing_spender = spending_index.get(outpoint)
                if existing_spender and existing_spender != txid:
                    valid_inputs = False
                spending_index[outpoint] = txid

                parent = by_txid.get(tx_input.txid)
                if parent is not None:
                    if tx_input.index < 0 or tx_input.index >= len(parent.outputs):
                        valid_inputs = False
                        continue
                    parents[txid].add(parent.txid)
                    total_in += int(parent.outputs[tx_input.index].amount)
                    continue

                prev = self.utxos.get(outpoint)
                if prev is None:
                    valid_inputs = False
                    continue
                total_in += int(prev["amount"])

            for parent_txid in parents[txid]:
                children[parent_txid].add(txid)

            if tx.is_coinbase() or not valid_inputs:
                fee = 0
            else:
                fee = max(0, total_in - sum(out.amount for out in tx.outputs))
            fees[txid] = fee
            fee_rates[txid] = fee / max(1, vbytes[txid])

        return {
            "by_txid": by_txid,
            "parents": parents,
            "children": children,
            "spending_index": spending_index,
            "fees": fees,
            "vbytes": vbytes,
            "fee_rates": fee_rates,
        }

    def _enforce_chain_limits_for_tx(
        self,
        txid: str,
        parents: dict[str, set[str]],
        children: dict[str, set[str]],
    ) -> None:
        ancestor_limit = max(1, int(self.config.mempool_ancestor_limit))
        descendant_limit = max(1, int(self.config.mempool_descendant_limit))

        ancestors = self._collect_ancestors(txid, parents)
        if len(ancestors) > ancestor_limit:
            raise ValidationError(
                f"Transaction has too many unconfirmed ancestors (max {ancestor_limit})"
            )

        to_check = set(ancestors)
        to_check.add(txid)
        for item in to_check:
            descendants = self._collect_descendants(item, children)
            if len(descendants) > descendant_limit:
                raise ValidationError(
                    f"Transaction chain exceeds descendant limit (max {descendant_limit})"
                )

    def _evict_pool_to_limits(
        self,
        txs: list[Transaction],
        protected_txids: set[str] | None = None,
    ) -> tuple[list[Transaction], set[str]]:
        protected = set(protected_txids or set())
        pool = [Transaction.from_dict(tx.to_dict()) for tx in txs]
        evicted: set[str] = set()

        while True:
            index = self._build_mempool_index(pool)
            total_vbytes = sum(index["vbytes"].values())
            over_count = len(pool) - int(self.config.max_mempool_transactions)
            over_vbytes = total_vbytes - int(self.config.max_mempool_virtual_bytes)
            if over_count <= 0 and over_vbytes <= 0:
                break

            victim_package: set[str] | None = None
            victim_rate = 0.0
            victim_oldest = 0

            for txid, tx in index["by_txid"].items():
                if txid in protected:
                    continue
                descendants = self._collect_descendants(txid, index["children"])
                package = {txid, *descendants}
                if package & protected:
                    continue
                package_fee = sum(int(index["fees"].get(item, 0)) for item in package)
                package_vbytes = sum(int(index["vbytes"].get(item, 0)) for item in package)
                package_rate = package_fee / max(1, package_vbytes)
                package_oldest = min(index["by_txid"][item].timestamp for item in package)

                if victim_package is None:
                    victim_package = package
                    victim_rate = package_rate
                    victim_oldest = package_oldest
                    continue

                if package_rate < victim_rate or (
                    abs(package_rate - victim_rate) <= 1e-12 and package_oldest < victim_oldest
                ):
                    victim_package = package
                    victim_rate = package_rate
                    victim_oldest = package_oldest

            if not victim_package:
                break

            evicted.update(victim_package)
            pool = [tx for tx in pool if tx.txid not in victim_package]

        return pool, evicted

    def _sanitize_mempool(self, txs: list[Transaction]) -> list[Transaction]:
        accepted: list[Transaction] = []
        seen: set[str] = set()
        utxo_view = copy.deepcopy(self.utxos)
        nft_view = copy.deepcopy(self.nft_registry)
        nft_listing_view = copy.deepcopy(self.nft_listings)
        contract_view = copy.deepcopy(self.contracts)
        next_block_height = max(0, self.height + 1)

        for tx in txs:
            if tx.txid in seen:
                continue
            seen.add(tx.txid)
            try:
                _fee = self.validate_transaction(
                    tx,
                    utxo_view,
                    for_mempool=True,
                    block_height=next_block_height,
                    nft_view=nft_view,
                    nft_listing_view=nft_listing_view,
                    contract_view=contract_view,
                )
                self._consume_inputs(tx, utxo_view)
                self._add_outputs(tx, utxo_view, block_height=None, coinbase=False)
                accepted.append(tx)
            except ValidationError:
                continue

        # Drop low fee-rate entries and txs violating ancestor/descendant policy.
        while True:
            changed = False
            index = self._build_mempool_index(accepted)
            if not index["by_txid"]:
                break

            min_fee_rate = self._effective_min_relay_fee_rate()
            low_fee = [
                txid
                for txid, rate in index["fee_rates"].items()
                if rate + 1e-12 < min_fee_rate
            ]
            if low_fee:
                for txid in low_fee:
                    package = {txid, *self._collect_descendants(txid, index["children"])}
                    accepted = [tx for tx in accepted if tx.txid not in package]
                changed = True
                if changed:
                    continue

            ancestor_limit = max(1, int(self.config.mempool_ancestor_limit))
            descendant_limit = max(1, int(self.config.mempool_descendant_limit))
            violating: set[str] = set()
            for txid in index["by_txid"]:
                if len(self._collect_ancestors(txid, index["parents"])) > ancestor_limit:
                    violating.add(txid)
                if len(self._collect_descendants(txid, index["children"])) > descendant_limit:
                    violating.add(txid)

            if violating:
                victim = min(
                    violating,
                    key=lambda item: (
                        float(index["fee_rates"].get(item, 0.0)),
                        int(index["by_txid"][item].timestamp),
                    ),
                )
                package = {victim, *self._collect_descendants(victim, index["children"])}
                accepted = [tx for tx in accepted if tx.txid not in package]
                changed = True
                if changed:
                    continue

            break

        accepted, _evicted = self._evict_pool_to_limits(accepted, protected_txids=set())
        return accepted

    @staticmethod
    def _reorg_depth(current: list[Block], candidate: list[Block]) -> int:
        if not current:
            return 0

        common = 0
        for left, right in zip(current, candidate):
            if left.block_hash != right.block_hash:
                break
            common += 1

        return len(current) - common

    def prune_mempool(self, save: bool = True) -> int:
        before = len(self.mempool)
        self.mempool = self._sanitize_mempool(self.mempool)
        removed = before - len(self.mempool)
        if save and removed > 0:
            self.save()
        return removed

    def replace_chain(
        self,
        blocks: list[Block],
        incoming_mempool: list[Transaction] | None = None,
        require_better: bool = True,
    ) -> bool:
        candidate_chain, candidate_utxos, candidate_nfts, candidate_nft_listings, candidate_contracts = self.validate_chain_blocks(
            blocks
        )
        candidate_work = candidate_chain[-1].chain_work if candidate_chain else 0
        current_work = self.tip.chain_work if self.tip else 0

        if self.chain and candidate_chain and self.chain[0].block_hash != candidate_chain[0].block_hash:
            raise ValidationError("Genesis block mismatch")

        if self.chain and candidate_chain:
            rollback_depth = self._reorg_depth(self.chain, candidate_chain)
            if rollback_depth > self.config.max_reorg_depth:
                raise ValidationError(
                    f"Reorg depth {rollback_depth} exceeds max_reorg_depth {self.config.max_reorg_depth}"
                )

        if require_better and self.chain and candidate_work <= current_work:
            return False

        existing_mempool = [Transaction.from_dict(tx.to_dict()) for tx in self.mempool]
        self.chain = candidate_chain
        self.utxos = candidate_utxos
        self.nft_registry = candidate_nfts
        self.nft_listings = candidate_nft_listings
        self.contracts = candidate_contracts

        combined_pool: list[Transaction] = []
        if incoming_mempool is not None:
            combined_pool.extend(Transaction.from_dict(tx.to_dict()) for tx in incoming_mempool)
        combined_pool.extend(existing_mempool)

        self.mempool = self._sanitize_mempool(combined_pool)
        self.save()
        return True

    def validate_block(
        self,
        block: Block,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        if not self.chain:
            raise ValidationError("Chain not initialized")

        prev = self.chain[-1]
        if block.index != prev.index + 1:
            raise ValidationError("Wrong block index")
        if block.prev_hash != prev.block_hash:
            raise ValidationError("Wrong previous block hash")
        mtp_window = self._mtp_window_for_schedule()
        self._validate_block_timestamp(
            block.timestamp,
            self.median_time_past(count=mtp_window),
            context="Block",
            prev_timestamp=prev.timestamp,
        )

        expected_target = self.next_target(block.timestamp)
        if block.target != expected_target:
            raise ValidationError("Block target does not match expected difficulty")

        txids = [tx.txid for tx in block.transactions]
        if block.merkle_root != merkle_root(txids):
            raise ValidationError("Merkle root mismatch")

        digest = self._pow_digest(block)
        pow_value = int.from_bytes(digest, "big")
        if pow_value >= block.target:
            raise ValidationError("Proof of work does not satisfy target")

        computed_hash = digest.hex()
        if computed_hash != block.block_hash:
            raise ValidationError("Block hash mismatch")

        expected_work = prev.chain_work + block_work(block.target)
        if block.chain_work != expected_work:
            raise ValidationError("Invalid chain work")

        nft_view = copy.deepcopy(self.nft_registry)
        nft_listing_view = copy.deepcopy(self.nft_listings)
        contract_view = copy.deepcopy(self.contracts)
        utxo_view, total_fees, coinbase_amount = self._apply_block_transactions(
            block,
            self.utxos,
            nft_view,
            nft_listing_view,
            contract_view,
        )
        issued_before = self.issued_supply()
        remaining = max(0, self.config.max_total_supply - issued_before)
        max_subsidy = min(self.block_reward(block.index), remaining)
        if coinbase_amount > total_fees + max_subsidy:
            raise ValidationError("Coinbase exceeds remaining supply cap")

        minted = max(0, coinbase_amount - total_fees)
        if issued_before + minted > self.config.max_total_supply:
            raise ValidationError("Total supply exceeds max_total_supply")
        return utxo_view, nft_view, nft_listing_view, contract_view

    def add_block(self, block: Block) -> None:
        utxo_view, nft_view, nft_listing_view, contract_view = self.validate_block(block)

        self.chain.append(block)
        self.utxos = utxo_view
        self.nft_registry = nft_view
        self.nft_listings = nft_listing_view
        self.contracts = contract_view

        block_txids = {tx.txid for tx in block.transactions[1:]}
        self.mempool = [tx for tx in self.mempool if tx.txid not in block_txids]
        self.mempool = self._sanitize_mempool(self.mempool)
        self.save()

    def _mempool_utxo_view(self, txs: list[Transaction]) -> dict[str, dict[str, Any]]:
        view = copy.deepcopy(self.utxos)
        index = self._build_mempool_index(txs)
        order = self._topological_order_subset(set(index["by_txid"].keys()), index["parents"])
        for txid in order:
            tx = index["by_txid"].get(txid)
            if tx is None:
                continue
            self._consume_inputs(tx, view)
            self._add_outputs(tx, view, block_height=None, coinbase=False)
        return view

    def _mempool_contract_views(
        self,
        txs: list[Transaction],
        utxo_view: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        applied_utxo = copy.deepcopy(self.utxos) if utxo_view is None else copy.deepcopy(utxo_view)
        nft_view = copy.deepcopy(self.nft_registry)
        nft_listing_view = copy.deepcopy(self.nft_listings)
        contract_view = copy.deepcopy(self.contracts)
        next_block_height = max(0, self.height + 1)

        index = self._build_mempool_index(txs)
        order = self._topological_order_subset(set(index["by_txid"].keys()), index["parents"])
        for txid in order:
            tx = index["by_txid"].get(txid)
            if tx is None:
                continue
            try:
                self.validate_transaction(
                    tx,
                    applied_utxo,
                    for_mempool=True,
                    block_height=next_block_height,
                    nft_view=nft_view,
                    nft_listing_view=nft_listing_view,
                    contract_view=contract_view,
                )
            except ValidationError:
                continue
            self._consume_inputs(tx, applied_utxo)
            self._add_outputs(tx, applied_utxo, block_height=None, coinbase=False)
        return nft_view, nft_listing_view, contract_view

    def _tx_fee_rate_for_pool(self, txid: str, index: dict[str, Any]) -> float:
        fee = int(index["fees"].get(txid, 0))
        vbytes = int(index["vbytes"].get(txid, 0))
        return fee / max(1, vbytes)

    def _package_fee_rate(self, txids: set[str], index: dict[str, Any]) -> float:
        fee = sum(int(index["fees"].get(txid, 0)) for txid in txids)
        vbytes = sum(int(index["vbytes"].get(txid, 0)) for txid in txids)
        return fee / max(1, vbytes)

    def add_transaction(self, tx: Transaction) -> None:
        if not self.chain:
            raise ValidationError("Initialize the chain first")

        self.prune_mempool(save=False)

        if any(existing.txid == tx.txid for existing in self.mempool):
            raise ValidationError("Transaction already exists in mempool")

        current_pool = [Transaction.from_dict(item.to_dict()) for item in self.mempool]
        current_index = self._build_mempool_index(current_pool)

        candidate_outpoints = {self._tx_input_outpoint(tx_input) for tx_input in tx.inputs}
        conflicting: set[str] = {
            spender
            for outpoint, spender in current_index["spending_index"].items()
            if outpoint in candidate_outpoints
        }

        replacement_set: set[str] = set()
        if conflicting:
            if not bool(self.config.mempool_rbf_enabled):
                raise ValidationError("Conflicting mempool transaction (RBF disabled)")
            for txid in conflicting:
                replacement_set.add(txid)
                replacement_set.update(self._collect_descendants(txid, current_index["children"]))
            if len(replacement_set) > int(self.config.max_rbf_replacements):
                raise ValidationError(
                    f"RBF replacement set too large (max {self.config.max_rbf_replacements})"
                )

        base_pool = [item for item in current_pool if item.txid not in replacement_set]
        base_index = self._build_mempool_index(base_pool)
        utxo_view = self._mempool_utxo_view(base_pool)
        nft_view, nft_listing_view, contract_view = self._mempool_contract_views(base_pool)
        next_height = max(0, self.height + 1)

        fee = self.validate_transaction(
            tx,
            utxo_view,
            for_mempool=True,
            block_height=next_height,
            nft_view=nft_view,
            nft_listing_view=nft_listing_view,
            contract_view=contract_view,
        )
        fee_rate = fee / max(1, self._tx_virtual_size(tx))
        if fee_rate + 1e-12 < self._effective_min_relay_fee_rate():
            raise ValidationError("Transaction feerate is below mempool minimum")

        if replacement_set:
            old_fee = sum(int(current_index["fees"].get(txid, 0)) for txid in replacement_set)
            old_vbytes = sum(int(current_index["vbytes"].get(txid, 0)) for txid in replacement_set)
            old_rate = old_fee / max(1, old_vbytes)
            if fee < old_fee + int(self.config.min_rbf_fee_delta):
                raise ValidationError("RBF replacement fee delta too small")
            if fee_rate <= old_rate + float(self.config.min_rbf_feerate_delta):
                raise ValidationError("RBF replacement feerate too low")

        candidate_pool = base_pool + [tx]
        candidate_index = self._build_mempool_index(candidate_pool)

        self._enforce_chain_limits_for_tx(
            tx.txid,
            candidate_index["parents"],
            candidate_index["children"],
        )

        protected = self._collect_ancestors(tx.txid, candidate_index["parents"])
        protected.add(tx.txid)
        final_pool, _evicted = self._evict_pool_to_limits(candidate_pool, protected_txids=protected)
        final_ids = {item.txid for item in final_pool}
        if tx.txid not in final_ids:
            raise ValidationError("Mempool full and candidate feerate too low")

        sanitized = self._sanitize_mempool(final_pool)
        if tx.txid not in {item.txid for item in sanitized}:
            raise ValidationError("Transaction rejected by mempool policy")
        self.mempool = sanitized
        self.save()

    def _estimate_base_fee(self, tx: Transaction) -> int:
        index = self._build_mempool_index([tx])
        return int(index["fees"].get(tx.txid, 0))

    def _select_spendable_utxos(
        self,
        sender_address: str,
        needed: int,
    ) -> tuple[list[tuple[str, dict[str, Any]]], int]:
        reserved_utxos = {
            f"{tx_input.txid}:{tx_input.index}"
            for mem_tx in self.mempool
            for tx_input in mem_tx.inputs
        }

        candidates: list[tuple[str, dict[str, Any]]] = []
        spend_height = max(0, self.height + 1)
        maturity = max(0, int(getattr(self.config, "coinbase_maturity", 0)))
        for key, value in self.utxos.items():
            if key in reserved_utxos:
                continue
            if value["address"] == sender_address:
                if bool(value.get("coinbase", False)) and maturity > 0:
                    try:
                        origin_height = int(value.get("height"))
                    except (TypeError, ValueError):
                        # Defensive: malformed metadata should not be picked by wallet selection.
                        continue
                    if spend_height - origin_height < maturity:
                        continue
                candidates.append((key, value))

        selected: list[tuple[str, dict[str, Any]]] = []
        total_in = 0
        for key, value in candidates:
            selected.append((key, value))
            total_in += int(value["amount"])
            if total_in >= needed:
                break

        if total_in < needed:
            raise ValidationError("Insufficient balance")
        return selected, total_in

    def create_unsigned_transaction(
        self,
        sender_pubkey: str,
        to_address: str,
        amount: int,
        fee: int | None = None,
    ) -> Transaction:
        if amount <= 0:
            raise ValidationError("Amount must be positive")
        if amount < self._min_dust_output():
            raise ValidationError(f"Amount is below dust policy minimum ({self._min_dust_output()})")
        self._validate_address_format(to_address)

        tx_fee = self.config.min_tx_fee if fee is None else fee
        if tx_fee < self.config.min_tx_fee:
            raise ValidationError("Fee is below minimum")

        sender_pubkey_text = str(sender_pubkey).strip().lower()
        if not sender_pubkey_text:
            raise ValidationError("Sender pubkey is required")
        try:
            sender_address = address_from_public_key(sender_pubkey_text, self.config.symbol)
        except Exception as exc:
            raise ValidationError("Invalid sender pubkey") from exc

        needed = amount + tx_fee
        selected, total_in = self._select_spendable_utxos(sender_address=sender_address, needed=needed)

        inputs: list[TxInput] = []
        for key, _ in selected:
            txid, index_text = key.split(":", 1)
            inputs.append(TxInput(txid=txid, index=int(index_text), pubkey=sender_pubkey_text))

        outputs = [TxOutput(amount=amount, address=to_address)]
        change = total_in - needed
        if change >= self._min_dust_output():
            outputs.append(TxOutput(amount=change, address=sender_address))
        elif change > 0:
            # Avoid creating dust change; treat it as extra fee.
            change = 0

        return Transaction(
            version=self.tx_version_for_height(max(0, self.height + 1)),
            timestamp=self._now(),
            nonce=secrets.randbits(64),
            inputs=inputs,
            outputs=outputs,
        )

    def sign_transaction(self, tx: Transaction, private_key_hex: str) -> Transaction:
        if tx.is_coinbase():
            raise ValidationError("Coinbase transaction cannot be signed this way")
        if len(tx.inputs) == 0:
            raise ValidationError("Transaction requires at least one input")

        signed_tx = Transaction.from_dict(tx.to_dict())
        try:
            signer_pubkey = private_key_to_public_key(private_key_hex).hex()
        except Exception as exc:
            raise ValidationError("Invalid private key") from exc

        for tx_input in signed_tx.inputs:
            existing_pubkey = tx_input.pubkey.strip().lower()
            if existing_pubkey and existing_pubkey != signer_pubkey:
                raise ValidationError("Private key does not match transaction input pubkey")
            tx_input.pubkey = signer_pubkey

        sighash = signed_tx.signing_hash()
        for tx_input in signed_tx.inputs:
            tx_input.signature = sign_digest(private_key_hex, sighash)
        signed_tx.txid = signed_tx.compute_txid()
        return signed_tx

    def create_transaction(self, private_key_hex: str, to_address: str, amount: int, fee: int | None = None) -> Transaction:
        try:
            sender_pubkey = private_key_to_public_key(private_key_hex).hex()
        except Exception as exc:
            raise ValidationError("Invalid private key") from exc

        unsigned_tx = self.create_unsigned_transaction(
            sender_pubkey=sender_pubkey,
            to_address=to_address,
            amount=amount,
            fee=fee,
        )
        return self.sign_transaction(unsigned_tx, private_key_hex=private_key_hex)

    def create_unsigned_contract_transaction(
        self,
        sender_pubkey: str,
        contract_payload: dict[str, Any],
        fee: int | None = None,
        to_address: str | None = None,
        amount: int | None = None,
    ) -> Transaction:
        sender_pubkey_text = str(sender_pubkey).strip().lower()
        if not sender_pubkey_text:
            raise ValidationError("Sender pubkey is required")
        try:
            sender_address = address_from_public_key(sender_pubkey_text, self.config.symbol)
        except Exception as exc:
            raise ValidationError("Invalid sender pubkey") from exc

        if to_address is None:
            destination = sender_address
        else:
            destination = str(to_address).strip()
            self._validate_address_format(destination)

        transfer_amount = self._min_dust_output() if amount is None else int(amount)
        if transfer_amount < self._min_dust_output():
            raise ValidationError(f"Contract transaction amount is below dust policy minimum ({self._min_dust_output()})")

        if not isinstance(contract_payload, dict):
            raise ValidationError("Contract payload must be a JSON object")
        normalized_payload = json.loads(json.dumps(contract_payload, sort_keys=True))
        if not isinstance(normalized_payload, dict):
            raise ValidationError("Contract payload must be a JSON object")

        tx = self.create_unsigned_transaction(
            sender_pubkey=sender_pubkey_text,
            to_address=destination,
            amount=transfer_amount,
            fee=fee,
        )
        tx.contract = normalized_payload
        tx.contract = self._normalize_contract_payload(tx)
        tx.txid = ""
        return tx

    def create_contract_transaction(
        self,
        private_key_hex: str,
        contract_payload: dict[str, Any],
        fee: int | None = None,
        to_address: str | None = None,
        amount: int | None = None,
    ) -> Transaction:
        try:
            sender_pubkey = private_key_to_public_key(private_key_hex).hex()
        except Exception as exc:
            raise ValidationError("Invalid private key") from exc
        unsigned_tx = self.create_unsigned_contract_transaction(
            sender_pubkey=sender_pubkey,
            contract_payload=contract_payload,
            fee=fee,
            to_address=to_address,
            amount=amount,
        )
        return self.sign_transaction(unsigned_tx, private_key_hex=private_key_hex)

    def mine_block(
        self,
        miner_address: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        mining_backend: str = "auto",
    ) -> Block:
        if not self.chain:
            raise ValidationError("Initialize the chain first")
        self._validate_address_format(miner_address)

        prev = self.chain[-1]
        mtp_window = self._mtp_window_for_schedule()
        timestamp = max(self._now(), self.median_time_past(count=mtp_window) + 1, prev.timestamp)
        target = self.next_target(timestamp)

        self.prune_mempool(save=False)

        chosen: list[Transaction] = []
        total_fees = 0
        utxo_view = copy.deepcopy(self.utxos)
        nft_view = copy.deepcopy(self.nft_registry)
        nft_listing_view = copy.deepcopy(self.nft_listings)
        contract_view = copy.deepcopy(self.contracts)
        max_mempool_tx_in_block = max(0, self.config.max_transactions_per_block - 1)
        pool_index = self._build_mempool_index(self.mempool)
        remaining: dict[str, Transaction] = dict(pool_index["by_txid"])
        parents = pool_index["parents"]
        children = pool_index["children"]

        while remaining and len(chosen) < max_mempool_tx_in_block:
            remaining_ids = set(remaining.keys())
            best_package: set[str] = set()
            best_score = -1.0
            best_fee = -1
            best_oldest_ts = 0

            for txid, tx in remaining.items():
                unresolved_ancestors = self._collect_ancestors(txid, parents) & remaining_ids
                package = set(unresolved_ancestors)
                package.add(txid)

                if bool(self.config.mempool_cpfp_enabled):
                    score = self._package_fee_rate(package, pool_index)
                    package_fee = sum(int(pool_index["fees"].get(item, 0)) for item in package)
                else:
                    score = self._tx_fee_rate_for_pool(txid, pool_index)
                    package = {txid}
                    package_fee = int(pool_index["fees"].get(txid, 0))

                oldest_ts = min(remaining[item].timestamp for item in package if item in remaining)
                if (
                    score > best_score
                    or (
                        abs(score - best_score) <= 1e-12
                        and (package_fee > best_fee or (package_fee == best_fee and oldest_ts < best_oldest_ts))
                    )
                ):
                    best_package = package
                    best_score = score
                    best_fee = package_fee
                    best_oldest_ts = oldest_ts

            if not best_package:
                break

            ordered = self._topological_order_subset(best_package, parents)
            admitted_any = False
            for txid in ordered:
                if txid not in remaining:
                    continue
                if len(chosen) >= max_mempool_tx_in_block:
                    break

                candidate = remaining[txid]
                try:
                    fee = self.validate_transaction(
                        candidate,
                        utxo_view,
                        for_mempool=True,
                        block_height=prev.index + 1,
                        nft_view=nft_view,
                        nft_listing_view=nft_listing_view,
                        contract_view=contract_view,
                    )
                except ValidationError:
                    reject_set = {txid, *self._collect_descendants(txid, children)}
                    for rejected in reject_set:
                        remaining.pop(rejected, None)
                    continue

                self._consume_inputs(candidate, utxo_view)
                self._add_outputs(candidate, utxo_view, block_height=None, coinbase=False)
                total_fees += fee
                chosen.append(candidate)
                remaining.pop(txid, None)
                admitted_any = True

            if not admitted_any:
                # Defensive progress guard.
                for txid in ordered:
                    remaining.pop(txid, None)

        issued_before = self.issued_supply()
        remaining = max(0, self.config.max_total_supply - issued_before)
        subsidy = min(self.block_reward(prev.index + 1), remaining)
        reward = subsidy + total_fees
        coinbase = Transaction(
            version=self.tx_version_for_height(prev.index + 1),
            timestamp=timestamp,
            nonce=secrets.randbits(64),
            inputs=[],
            outputs=[TxOutput(amount=reward, address=miner_address)],
        )
        coinbase.txid = coinbase.compute_txid()

        transactions = [coinbase] + chosen
        txids = [tx.txid for tx in transactions]

        block = Block(
            index=prev.index + 1,
            prev_hash=prev.block_hash,
            timestamp=timestamp,
            target=target,
            nonce=0,
            merkle_root=merkle_root(txids),
            miner=miner_address,
            transactions=transactions,
            chain_work=prev.chain_work + block_work(target),
        )

        block.block_hash = self._mine_block_hash(
            block,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
            mining_backend=mining_backend,
        )
        self.add_block(block)
        return block

    def balance_of(self, address: str) -> int:
        total = 0
        for value in self.utxos.values():
            if value["address"] == address:
                total += int(value["amount"])
        return total

    def nft_state(self, token_id: str | None = None) -> dict[str, Any]:
        if token_id is None:
            return {
                "tokens": {key: dict(value) for key, value in sorted(self.nft_registry.items())},
                "count": len(self.nft_registry),
            }
        token_key = str(token_id).strip()
        entry = self.nft_registry.get(token_key)
        return {"token_id": token_key, "exists": entry is not None, "token": dict(entry) if entry is not None else None}

    def nft_listings_state(self) -> dict[str, Any]:
        rows = [dict(value) for _key, value in sorted(self.nft_listings.items())]
        return {"listings": rows, "count": len(rows)}

    def smart_contract_state(self, contract_id: str | None = None) -> dict[str, Any]:
        if contract_id is None:
            return {
                "contracts": {key: dict(value) for key, value in sorted(self.contracts.items())},
                "count": len(self.contracts),
            }
        contract_key = str(contract_id).strip()
        entry = self.contracts.get(contract_key)
        return {
            "contract_id": contract_key,
            "exists": entry is not None,
            "contract": dict(entry) if entry is not None else None,
        }

    def status(self) -> dict[str, Any]:
        tip = self.tip
        issued = self.issued_supply()
        next_height = max(0, self.height + 1)
        active_protocol = self.active_protocol_version()
        next_protocol = self.next_protocol_version()
        upgrade_height = int(self._consensus_protocol_upgrade_v2_height)
        if next_height >= upgrade_height:
            blocks_until_upgrade = 0
        else:
            blocks_until_upgrade = upgrade_height - next_height
        return {
            "chain_id": self._consensus_chain_id,
            "symbol": self.config.symbol,
            "consensus_lock_enabled": self._consensus_lock_enabled(),
            "protocol_version": active_protocol,
            "next_protocol_version": next_protocol,
            "protocol_upgrade_v2_height": upgrade_height,
            "blocks_until_protocol_v2": blocks_until_upgrade,
            "target_schedule": self._target_schedule_name(),
            "fixed_genesis_hash": str(self.config.fixed_genesis_hash).strip().lower(),
            "asert_half_life": self._consensus_asert_half_life,
            "mtp_window": self._mtp_window_for_schedule(),
            "max_block_timestamp_step_seconds": self._consensus_max_block_timestamp_step_seconds,
            "height": self.height,
            "tip_hash": tip.block_hash if tip else None,
            "chain_work": int(tip.chain_work) if tip else 0,
            "issued_supply": issued,
            "max_total_supply": self.config.max_total_supply,
            "remaining_supply": max(0, self.config.max_total_supply - issued),
            "mempool_size": len(self.mempool),
            "tx_policy": {
                "coinbase_maturity": int(getattr(self.config, "coinbase_maturity", 0)),
                "min_dust_output": self._min_dust_output(),
                "max_standard_tx_virtual_bytes": int(getattr(self.config, "max_standard_tx_virtual_bytes", 100_000)),
                "smart_contracts_enabled": self._contracts_enabled(),
                "nft_market_enabled": self._nft_market_enabled(),
                "max_contract_payload_bytes": int(getattr(self.config, "max_contract_payload_bytes", 4_096)),
            },
            "mempool_policy": {
                "max_transactions": self.config.max_mempool_transactions,
                "max_virtual_bytes": self.config.max_mempool_virtual_bytes,
                "min_fee_rate": self.config.min_mempool_fee_rate,
                "min_standard_fee_rate": float(getattr(self.config, "min_standard_tx_fee_rate", 0.0)),
                "effective_min_fee_rate": self._effective_min_relay_fee_rate(),
                "ancestor_limit": self.config.mempool_ancestor_limit,
                "descendant_limit": self.config.mempool_descendant_limit,
                "rbf_enabled": self.config.mempool_rbf_enabled,
                "cpfp_enabled": self.config.mempool_cpfp_enabled,
                "max_rbf_replacements": self.config.max_rbf_replacements,
                "min_rbf_fee_delta": self.config.min_rbf_fee_delta,
                "min_rbf_feerate_delta": self.config.min_rbf_feerate_delta,
            },
            "nft_count": len(self.nft_registry),
            "nft_listings_count": len(self.nft_listings),
            "smart_contract_count": len(self.contracts),
            "utxo_count": len(self.utxos),
            "target": tip.target if tip else None,
        }
