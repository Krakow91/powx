from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key
from powx.models import Transaction


PRIV_A = "1" * 64
ADDR_A = address_from_public_key(private_key_to_public_key(PRIV_A).hex(), "KK91")
PRIV_B = "2" * 64
ADDR_B = address_from_public_key(private_key_to_public_key(PRIV_B).hex(), "KK91")

FAST_UPGRADE_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-upgrade-test",
    coinbase_maturity=0,
    initial_target=2**255,
    max_target=2**255,
    max_adjust_factor_up=1.0,
    max_adjust_factor_down=1.0,
    target_block_time=1,
    protocol_version=1,
    protocol_upgrade_v2_height=2,
)


class ProtocolUpgradeCompatTest(unittest.TestCase):
    def test_protocol_upgrade_transition_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_UPGRADE_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=0)
            for _ in range(3):
                chain.mine_block(ADDR_A, mining_backend="cpu")

            versions_by_height = [block.transactions[0].version for block in chain.chain]
            self.assertEqual(versions_by_height, [1, 1, 2, 2])

            status = chain.status()
            self.assertEqual(status["height"], 3)
            self.assertEqual(status["protocol_version"], 2)
            self.assertEqual(status["next_protocol_version"], 2)
            self.assertEqual(status["protocol_upgrade_v2_height"], 2)
            self.assertEqual(status["blocks_until_protocol_v2"], 0)

    def test_transaction_version_gate_after_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_UPGRADE_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=0)
            chain.mine_block(ADDR_A, mining_backend="cpu")

            tx_v2 = chain.create_transaction(PRIV_A, ADDR_B, amount=2, fee=1)
            self.assertEqual(tx_v2.version, 2)

            fee = chain.validate_transaction(
                tx_v2,
                utxo_view=copy.deepcopy(chain.utxos),
                block_height=2,
            )
            self.assertGreaterEqual(fee, 1)

            tx_legacy = Transaction.from_dict(tx_v2.to_dict())
            tx_legacy.version = 1
            with self.assertRaisesRegex(ValidationError, "Unsupported transaction version"):
                chain.validate_transaction(
                    tx_legacy,
                    utxo_view=copy.deepcopy(chain.utxos),
                    block_height=2,
                )

    def test_state_compat_missing_protocol_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_UPGRADE_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=0)
            chain.mine_block(ADDR_A, mining_backend="cpu")

            state_path = Path(td) / "chain_state.json"
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            if isinstance(cfg, dict):
                cfg.pop("protocol_version", None)
                cfg.pop("protocol_upgrade_v2_height", None)
            state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

            reloaded = Chain(td, config=FAST_UPGRADE_CONFIG)
            reloaded.load()
            status = reloaded.status()
            self.assertEqual(status["protocol_version"], 1)
            self.assertEqual(status["next_protocol_version"], 2)
            self.assertEqual(status["protocol_upgrade_v2_height"], 2)

    def test_state_override_upgrade_height_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_UPGRADE_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=0)
            chain.mine_block(ADDR_A, mining_backend="cpu")
            chain.mine_block(ADDR_A, mining_backend="cpu")

            runtime_config = replace(
                CONFIG,
                consensus_lock_enabled=False,
                chain_id="kk91-upgrade-test",
                coinbase_maturity=0,
                initial_target=2**255,
                max_target=2**255,
                max_adjust_factor_up=1.0,
                max_adjust_factor_down=1.0,
                target_block_time=1,
                protocol_version=1,
                protocol_upgrade_v2_height=2_147_483_647,
            )
            reloaded = Chain(td, config=runtime_config)
            reloaded.load()
            status = reloaded.status()
            self.assertEqual(status["protocol_upgrade_v2_height"], 2)
            self.assertEqual(status["protocol_version"], 2)

    def test_invalid_upgrade_height_in_state_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_UPGRADE_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=0)

            state_path = Path(td) / "chain_state.json"
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            if isinstance(cfg, dict):
                cfg["protocol_upgrade_v2_height"] = -1
            state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

            broken = Chain(td, config=FAST_UPGRADE_CONFIG)
            with self.assertRaisesRegex(ValidationError, "Invalid config value for 'protocol_upgrade_v2_height'"):
                broken.load()


if __name__ == "__main__":
    unittest.main()
