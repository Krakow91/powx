from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key


PRIV_B = "2" * 64
ADDR_B = address_from_public_key(private_key_to_public_key(PRIV_B).hex(), "KK91")


class MainnetConsensusLockTest(unittest.TestCase):
    def test_initialize_uses_fixed_genesis_template_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            with mock.patch.object(chain, "_now", return_value=1_900_000_000):
                genesis = chain.initialize(ADDR_B, genesis_supply=123456)

            self.assertTrue(chain.status()["consensus_lock_enabled"])
            self.assertEqual(genesis.block_hash, CONFIG.fixed_genesis_hash)
            self.assertEqual(genesis.timestamp, int(CONFIG.fixed_genesis_timestamp))
            self.assertEqual(genesis.nonce, int(CONFIG.fixed_genesis_block_nonce))
            self.assertEqual(genesis.transactions[0].nonce, int(CONFIG.fixed_genesis_tx_nonce))
            self.assertEqual(genesis.transactions[0].txid, CONFIG.fixed_genesis_txid)
            self.assertEqual(genesis.transactions[0].outputs[0].address, CONFIG.fixed_genesis_address)
            self.assertEqual(int(genesis.transactions[0].outputs[0].amount), int(CONFIG.fixed_genesis_supply))

    def test_state_rejects_non_asert_schedule_when_lock_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            chain.initialize(ADDR_B, genesis_supply=0)

            state_path = Path(td) / "chain_state.json"
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            if isinstance(cfg, dict):
                cfg["target_schedule"] = "window-v2"
            state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

            broken = Chain(td)
            with self.assertRaisesRegex(ValidationError, "target_schedule"):
                broken.load()

    def test_state_rejects_upgrade_height_mismatch_when_lock_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            chain.initialize(ADDR_B, genesis_supply=0)

            state_path = Path(td) / "chain_state.json"
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            if isinstance(cfg, dict):
                cfg["protocol_upgrade_v2_height"] = int(CONFIG.protocol_upgrade_v2_height) + 1
            state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

            broken = Chain(td)
            with self.assertRaisesRegex(ValidationError, "protocol_upgrade_v2_height"):
                broken.load()

    def test_state_rejects_chain_id_alias_when_lock_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            chain.initialize(ADDR_B, genesis_supply=0)

            state_path = Path(td) / "chain_state.json"
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", {})
            if isinstance(cfg, dict):
                cfg["chain_id"] = "kk91-main"
            state_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

            broken = Chain(td)
            with self.assertRaisesRegex(ValidationError, "chain_id"):
                broken.load()


if __name__ == "__main__":
    unittest.main()

