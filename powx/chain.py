from __future__ import annotations

import copy
import json
import os
import secrets
import time
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
    def __init__(self, data_dir: str | Path, config: ChainConfig = CONFIG):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "chain_state.json"

        self.chain: list[Block] = []
        self.utxos: dict[str, dict[str, Any]] = {}
        self.mempool: list[Transaction] = []
        self._legacy_target_schedule = False
        self._consensus_chain_id = self.config.chain_id
        self._consensus_protocol_version = self.config.protocol_version
        self._consensus_protocol_upgrade_v2_height = self.config.protocol_upgrade_v2_height
        self._consensus_target_block_time = self.config.target_block_time
        self._consensus_max_target = self.config.max_target
        self._consensus_max_adjust_factor_up = self.config.max_adjust_factor_up
        self._consensus_max_adjust_factor_down = self.config.max_adjust_factor_down
        self._gpu_miner: OpenCLMiner | None = None
        self._gpu_probe_attempted = False

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> None:
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}")

        with self.state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self._reset_consensus_overrides()
        raw_config = data.get("config")
        legacy_target_schedule = self._detect_legacy_target_schedule(raw_config)
        self._validate_state_config(raw_config, legacy_target_schedule=legacy_target_schedule)
        self._apply_state_consensus_overrides(raw_config, legacy_target_schedule=legacy_target_schedule)

        loaded_chain = [Block.from_dict(item) for item in data.get("chain", [])]
        if loaded_chain:
            try:
                self.chain, self.utxos = self.validate_chain_blocks(
                    loaded_chain,
                    legacy_target_schedule=legacy_target_schedule,
                )
            except ValidationError as exc:
                if (
                    legacy_target_schedule
                    and self._legacy_missing_down_adjust(raw_config)
                    and "target mismatch" in str(exc).lower()
                    and self._consensus_max_adjust_factor_down != 0.5
                ):
                    # Some early local chains used 0.5 as down-adjust clamp.
                    self._consensus_max_adjust_factor_down = 0.5
                    self.chain, self.utxos = self.validate_chain_blocks(
                        loaded_chain,
                        legacy_target_schedule=legacy_target_schedule,
                    )
                else:
                    raise
            self._legacy_target_schedule = legacy_target_schedule
        else:
            self.chain = []
            self.utxos = {}
            self._legacy_target_schedule = legacy_target_schedule

        self.mempool = [Transaction.from_dict(item) for item in data.get("mempool", [])]
        self.mempool = self._sanitize_mempool(self.mempool)

    def save(self) -> None:
        target_schedule = "legacy-v1" if self._legacy_target_schedule else "window-v2"
        difficulty_window = 2 if self._legacy_target_schedule else self.config.difficulty_window
        initial_target = self.chain[0].target if self.chain else self.config.initial_target
        data = {
            "config": {
                "symbol": self.config.symbol,
                "chain_id": self._consensus_chain_id,
                "protocol_version": self._consensus_protocol_version,
                "protocol_upgrade_v2_height": self._consensus_protocol_upgrade_v2_height,
                "target_schedule": target_schedule,
                "target_block_time": self._consensus_target_block_time,
                "difficulty_window": difficulty_window,
                "pow_algorithm": self.config.pow_algorithm,
                "halving_interval": self.config.halving_interval,
                "initial_block_reward": self.config.initial_block_reward,
                "max_total_supply": self.config.max_total_supply,
                "min_tx_fee": self.config.min_tx_fee,
                "max_target": self._consensus_max_target,
                "initial_target": initial_target,
                "max_adjust_factor_up": self._consensus_max_adjust_factor_up,
                "max_adjust_factor_down": self._consensus_max_adjust_factor_down,
                "max_transactions_per_block": self.config.max_transactions_per_block,
                "max_mempool_transactions": self.config.max_mempool_transactions,
                "max_tx_inputs": self.config.max_tx_inputs,
                "max_tx_outputs": self.config.max_tx_outputs,
                "max_future_block_seconds": self.config.max_future_block_seconds,
                "max_mempool_tx_age_seconds": self.config.max_mempool_tx_age_seconds,
                "max_future_tx_seconds": self.config.max_future_tx_seconds,
                "max_reorg_depth": self.config.max_reorg_depth,
            },
            "height": self.height,
            "chain": [block.to_dict() for block in self.chain],
            "utxos": self.utxos,
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
        if not isinstance(raw, dict):
            return False

        schedule = str(raw.get("target_schedule", "")).strip().lower()
        if schedule in {"legacy", "legacy-v1", "v1"}:
            return True
        if schedule in {"window", "window-v2", "v2"}:
            return False

        if "difficulty_window" not in raw:
            return True
        try:
            window = int(raw.get("difficulty_window"))
        except (TypeError, ValueError):
            return False
        return window <= 2

    @staticmethod
    def _legacy_missing_down_adjust(raw: Any) -> bool:
        if not isinstance(raw, dict):
            return True
        return "max_adjust_factor_down" not in raw

    def _allowed_chain_ids(self) -> set[str]:
        allowed = {self.config.chain_id}
        if self.config.chain_id == "kk91-gpu-main":
            allowed.add("kk91-main")
        return allowed

    def _reset_consensus_overrides(self) -> None:
        self._consensus_chain_id = str(self.config.chain_id)
        self._consensus_protocol_version = int(self.config.protocol_version)
        self._consensus_protocol_upgrade_v2_height = int(self.config.protocol_upgrade_v2_height)
        self._consensus_target_block_time = int(self.config.target_block_time)
        self._consensus_max_target = int(self.config.max_target)
        self._consensus_max_adjust_factor_up = float(self.config.max_adjust_factor_up)
        self._consensus_max_adjust_factor_down = float(self.config.max_adjust_factor_down)

    def _apply_state_consensus_overrides(self, raw: Any, legacy_target_schedule: bool) -> None:
        if not isinstance(raw, dict):
            return

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

    def _validate_state_config(self, raw: Any, legacy_target_schedule: bool) -> None:
        if not isinstance(raw, dict):
            return

        critical: dict[str, Any] = {
            "symbol": self.config.symbol,
            "pow_algorithm": self.config.pow_algorithm,
            "protocol_version": self.config.protocol_version,
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

        if not legacy_target_schedule and "difficulty_window" in raw:
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
        if genesis_supply < 0:
            raise ValidationError("Genesis supply must be >= 0")
        if genesis_supply > self.config.max_total_supply:
            raise ValidationError(
                f"Genesis supply exceeds max_total_supply ({self.config.max_total_supply})"
            )
        self._validate_address_format(genesis_address)
        self._legacy_target_schedule = False
        self._reset_consensus_overrides()

        timestamp = self._now()
        coinbase = Transaction(
            version=self.tx_version_for_height(0),
            timestamp=timestamp,
            nonce=secrets.randbits(64),
            inputs=[],
            outputs=[TxOutput(amount=genesis_supply, address=genesis_address)],
        )
        coinbase.txid = coinbase.compute_txid()

        block = Block(
            index=0,
            prev_hash="0" * 64,
            timestamp=timestamp,
            target=self.config.initial_target,
            nonce=0,
            merkle_root=merkle_root([coinbase.txid]),
            miner=genesis_address,
            transactions=[coinbase],
            chain_work=block_work(self.config.initial_target),
        )

        block.block_hash = self._mine_block_hash(block)

        self.chain = [block]
        self.utxos = {f"{coinbase.txid}:0": {"amount": genesis_supply, "address": genesis_address}}
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
        legacy_target_schedule: bool | None = None,
    ) -> int:
        if not history:
            return self.config.initial_target

        prev = history[-1]
        use_legacy = self._legacy_target_schedule if legacy_target_schedule is None else bool(legacy_target_schedule)
        if use_legacy:
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

    def _validate_block_timestamp(self, timestamp: int, median_past: int, context: str) -> None:
        if timestamp <= median_past:
            raise ValidationError(f"{context} timestamp is too old")

        max_allowed = self._now() + self.config.max_future_block_seconds
        if timestamp > max_allowed:
            raise ValidationError(f"{context} timestamp is too far in the future")

    def next_target(self, timestamp: int) -> int:
        return self._next_target_from_history(
            self.chain,
            timestamp,
            legacy_target_schedule=self._legacy_target_schedule,
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

    def _validate_tx_outputs(self, tx: Transaction, allow_zero_amount: bool = False) -> int:
        if len(tx.outputs) == 0:
            raise ValidationError("Transaction has no outputs")
        if len(tx.outputs) > self.config.max_tx_outputs:
            raise ValidationError(f"Transaction has too many outputs (max {self.config.max_tx_outputs})")

        total_out = 0
        for output in tx.outputs:
            if output.amount < 0:
                raise ValidationError("Output amount must not be negative")
            if output.amount == 0 and not allow_zero_amount:
                raise ValidationError("Output amount must be positive")
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

    def validate_transaction(
        self,
        tx: Transaction,
        utxo_view: dict[str, dict[str, Any]],
        allow_coinbase: bool = False,
        for_mempool: bool = False,
        block_height: int | None = None,
    ) -> int:
        if block_height is None:
            block_height = max(0, self.height + 1)

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

        computed = tx.compute_txid()
        if tx.txid != computed:
            raise ValidationError("Transaction ID mismatch")

        allow_zero_outputs = tx.is_coinbase() and allow_coinbase
        total_out = self._validate_tx_outputs(tx, allow_zero_amount=allow_zero_outputs)

        if tx.is_coinbase():
            if not allow_coinbase:
                raise ValidationError("Coinbase transaction is not allowed in this context")
            return 0

        if len(tx.inputs) == 0:
            raise ValidationError("Regular transaction requires inputs")
        if len(tx.inputs) > self.config.max_tx_inputs:
            raise ValidationError(f"Transaction has too many inputs (max {self.config.max_tx_inputs})")

        seen_inputs: set[str] = set()
        total_in = 0
        sighash = tx.signing_hash()

        for tx_input in tx.inputs:
            key = f"{tx_input.txid}:{tx_input.index}"
            if key in seen_inputs:
                raise ValidationError("Duplicate input in transaction")
            seen_inputs.add(key)

            prev_out = utxo_view.get(key)
            if prev_out is None:
                raise ValidationError(f"Missing UTXO: {key}")
            if not tx_input.signature or not tx_input.pubkey:
                raise ValidationError("Missing signature or pubkey")

            try:
                expected_addr = address_from_public_key(tx_input.pubkey, self.config.symbol)
            except Exception as exc:
                raise ValidationError("Invalid public key encoding") from exc
            if expected_addr != prev_out["address"]:
                raise ValidationError("Input pubkey does not match UTXO owner")

            if not verify_signature(tx_input.pubkey, sighash, tx_input.signature):
                raise ValidationError("Invalid input signature")

            total_in += int(prev_out["amount"])

        if total_in < total_out:
            raise ValidationError("Inputs do not cover outputs")

        fee = total_in - total_out
        if fee < self.config.min_tx_fee:
            raise ValidationError("Transaction fee is below minimum")

        return fee

    @staticmethod
    def _consume_inputs(tx: Transaction, utxo_view: dict[str, dict[str, Any]]) -> None:
        for tx_input in tx.inputs:
            key = f"{tx_input.txid}:{tx_input.index}"
            if key in utxo_view:
                del utxo_view[key]

    @staticmethod
    def _add_outputs(tx: Transaction, utxo_view: dict[str, dict[str, Any]]) -> None:
        for index, output in enumerate(tx.outputs):
            key = f"{tx.txid}:{index}"
            utxo_view[key] = {"amount": output.amount, "address": output.address}

    def _apply_block_transactions(
        self,
        block: Block,
        utxo_start: dict[str, dict[str, Any]],
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
        )

        for tx in block.transactions[1:]:
            if tx.is_coinbase():
                raise ValidationError("Only the first transaction can be coinbase")

        utxo_view = copy.deepcopy(utxo_start)
        total_fees = 0

        for tx in block.transactions[1:]:
            fee = self.validate_transaction(tx, utxo_view, block_height=block.index)
            total_fees += fee
            self._consume_inputs(tx, utxo_view)
            self._add_outputs(tx, utxo_view)

        coinbase_amount = sum(output.amount for output in coinbase.outputs)
        allowed_reward = self.block_reward(block.index) + total_fees
        if coinbase_amount > allowed_reward:
            raise ValidationError("Coinbase exceeds allowed reward")

        self._add_outputs(coinbase, utxo_view)
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
        legacy_target_schedule: bool | None = None,
    ) -> int:
        return self._next_target_from_history(
            chain,
            timestamp,
            legacy_target_schedule=legacy_target_schedule,
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

        utxos: dict[str, dict[str, Any]] = {}
        self._add_outputs(coinbase, utxos)
        return utxos

    def validate_chain_blocks(
        self,
        blocks: list[Block],
        legacy_target_schedule: bool | None = None,
    ) -> tuple[list[Block], dict[str, dict[str, Any]]]:
        if not blocks:
            raise ValidationError("Candidate chain is empty")

        use_legacy = self._legacy_target_schedule if legacy_target_schedule is None else bool(legacy_target_schedule)
        normalized = [Block.from_dict(block.to_dict()) for block in blocks]

        chain_view: list[Block] = [normalized[0]]
        utxo_view = self._validate_genesis_block(normalized[0])
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
                self._median_time_for_chain(chain_view),
                context="Candidate block",
            )

            expected_target = self._next_target_for_chain(
                chain_view,
                block.timestamp,
                legacy_target_schedule=use_legacy,
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

            utxo_view, total_fees, coinbase_amount = self._apply_block_transactions(block, utxo_view)

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

        return chain_view, utxo_view

    def _sanitize_mempool(self, txs: list[Transaction]) -> list[Transaction]:
        clean: list[Transaction] = []
        seen: set[str] = set()
        utxo_view = copy.deepcopy(self.utxos)
        next_block_height = max(0, self.height + 1)

        for tx in txs:
            if len(clean) >= self.config.max_mempool_transactions:
                break
            if tx.txid in seen:
                continue
            seen.add(tx.txid)
            try:
                _fee = self.validate_transaction(
                    tx,
                    utxo_view,
                    for_mempool=True,
                    block_height=next_block_height,
                )
                self._consume_inputs(tx, utxo_view)
                self._add_outputs(tx, utxo_view)
                clean.append(tx)
            except ValidationError:
                continue

        return clean

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
        candidate_chain, candidate_utxos = self.validate_chain_blocks(blocks)
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

        combined_pool: list[Transaction] = []
        if incoming_mempool is not None:
            combined_pool.extend(Transaction.from_dict(tx.to_dict()) for tx in incoming_mempool)
        combined_pool.extend(existing_mempool)

        self.mempool = self._sanitize_mempool(combined_pool)
        self.save()
        return True

    def validate_block(self, block: Block) -> dict[str, dict[str, Any]]:
        if not self.chain:
            raise ValidationError("Chain not initialized")

        prev = self.chain[-1]
        if block.index != prev.index + 1:
            raise ValidationError("Wrong block index")
        if block.prev_hash != prev.block_hash:
            raise ValidationError("Wrong previous block hash")
        self._validate_block_timestamp(block.timestamp, self.median_time_past(), context="Block")

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

        utxo_view, total_fees, coinbase_amount = self._apply_block_transactions(block, self.utxos)
        issued_before = self.issued_supply()
        remaining = max(0, self.config.max_total_supply - issued_before)
        max_subsidy = min(self.block_reward(block.index), remaining)
        if coinbase_amount > total_fees + max_subsidy:
            raise ValidationError("Coinbase exceeds remaining supply cap")

        minted = max(0, coinbase_amount - total_fees)
        if issued_before + minted > self.config.max_total_supply:
            raise ValidationError("Total supply exceeds max_total_supply")
        return utxo_view

    def add_block(self, block: Block) -> None:
        utxo_view = self.validate_block(block)

        self.chain.append(block)
        self.utxos = utxo_view

        block_txids = {tx.txid for tx in block.transactions[1:]}
        self.mempool = [tx for tx in self.mempool if tx.txid not in block_txids]
        self.mempool = self._sanitize_mempool(self.mempool)
        self.save()

    def add_transaction(self, tx: Transaction) -> None:
        if not self.chain:
            raise ValidationError("Initialize the chain first")

        self.prune_mempool(save=False)

        if any(existing.txid == tx.txid for existing in self.mempool):
            raise ValidationError("Transaction already exists in mempool")

        if len(self.mempool) >= self.config.max_mempool_transactions:
            raise ValidationError(f"Mempool is full (max {self.config.max_mempool_transactions} transactions)")

        # Build a temporary view that already includes mempool spends to prevent double-spends.
        utxo_view = copy.deepcopy(self.utxos)
        for mem_tx in self.mempool:
            self._consume_inputs(mem_tx, utxo_view)
            self._add_outputs(mem_tx, utxo_view)

        _fee = self.validate_transaction(
            tx,
            utxo_view,
            for_mempool=True,
            block_height=max(0, self.height + 1),
        )
        self._consume_inputs(tx, utxo_view)
        self._add_outputs(tx, utxo_view)

        self.mempool.append(tx)
        self.save()

    def _estimate_base_fee(self, tx: Transaction) -> int:
        # Fee estimate against current UTXO set (without mempool dependencies) for mempool ranking.
        if tx.is_coinbase():
            return 0
        total_in = 0
        for tx_input in tx.inputs:
            key = f"{tx_input.txid}:{tx_input.index}"
            prev = self.utxos.get(key)
            if prev is None:
                return 0
            total_in += int(prev["amount"])
        total_out = sum(out.amount for out in tx.outputs)
        return max(0, total_in - total_out)

    def create_transaction(self, private_key_hex: str, to_address: str, amount: int, fee: int | None = None) -> Transaction:
        if amount <= 0:
            raise ValidationError("Amount must be positive")
        self._validate_address_format(to_address)

        tx_fee = self.config.min_tx_fee if fee is None else fee
        if tx_fee < self.config.min_tx_fee:
            raise ValidationError("Fee is below minimum")

        sender_pubkey = private_key_to_public_key(private_key_hex).hex()
        sender_address = address_from_public_key(sender_pubkey, self.config.symbol)

        reserved_utxos = {
            f"{tx_input.txid}:{tx_input.index}"
            for mem_tx in self.mempool
            for tx_input in mem_tx.inputs
        }

        candidates: list[tuple[str, dict[str, Any]]] = []
        for key, value in self.utxos.items():
            if key in reserved_utxos:
                continue
            if value["address"] == sender_address:
                candidates.append((key, value))

        selected: list[tuple[str, dict[str, Any]]] = []
        total_in = 0
        needed = amount + tx_fee

        for key, value in candidates:
            selected.append((key, value))
            total_in += int(value["amount"])
            if total_in >= needed:
                break

        if total_in < needed:
            raise ValidationError("Insufficient balance")

        inputs: list[TxInput] = []
        for key, _ in selected:
            txid, index_text = key.split(":", 1)
            inputs.append(TxInput(txid=txid, index=int(index_text), pubkey=sender_pubkey))

        outputs = [TxOutput(amount=amount, address=to_address)]
        change = total_in - needed
        if change > 0:
            outputs.append(TxOutput(amount=change, address=sender_address))

        tx = Transaction(
            version=self.tx_version_for_height(max(0, self.height + 1)),
            timestamp=self._now(),
            nonce=secrets.randbits(64),
            inputs=inputs,
            outputs=outputs,
        )

        sighash = tx.signing_hash()
        for tx_input in tx.inputs:
            tx_input.signature = sign_digest(private_key_hex, sighash)

        tx.txid = tx.compute_txid()
        return tx

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

        timestamp = max(self._now(), self.median_time_past() + 1)
        target = self.next_target(timestamp)
        prev = self.chain[-1]

        self.prune_mempool(save=False)

        chosen: list[Transaction] = []
        total_fees = 0
        utxo_view = copy.deepcopy(self.utxos)
        ranked = sorted(
            self.mempool,
            key=lambda tx: (self._estimate_base_fee(tx), -tx.timestamp),
            reverse=True,
        )

        for tx in ranked:
            if len(chosen) >= self.config.max_transactions_per_block - 1:
                break
            try:
                fee = self.validate_transaction(tx, utxo_view, for_mempool=True)
                self._consume_inputs(tx, utxo_view)
                self._add_outputs(tx, utxo_view)
                total_fees += fee
                chosen.append(tx)
            except ValidationError:
                continue

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
            "protocol_version": active_protocol,
            "next_protocol_version": next_protocol,
            "protocol_upgrade_v2_height": upgrade_height,
            "blocks_until_protocol_v2": blocks_until_upgrade,
            "target_schedule": "legacy-v1" if self._legacy_target_schedule else "window-v2",
            "height": self.height,
            "tip_hash": tip.block_hash if tip else None,
            "chain_work": int(tip.chain_work) if tip else 0,
            "issued_supply": issued,
            "max_total_supply": self.config.max_total_supply,
            "remaining_supply": max(0, self.config.max_total_supply - issued),
            "mempool_size": len(self.mempool),
            "utxo_count": len(self.utxos),
            "target": tip.target if tip else None,
        }
