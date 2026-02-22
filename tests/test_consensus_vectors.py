from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key, sign_digest
from powx.models import Block, Transaction, TxInput, TxOutput


PRIV_A = "1" * 64
PUB_A = private_key_to_public_key(PRIV_A).hex()
ADDR_A = address_from_public_key(PUB_A, "KK91")

PRIV_B = "2" * 64
PUB_B = private_key_to_public_key(PRIV_B).hex()
ADDR_B = address_from_public_key(PUB_B, "KK91")

VECTOR_GENESIS_HASH = "00000330c8fe2065d8f358842cee842734c98e6ce46d302440db095297252160"
VECTOR_GENESIS_TXID = "90ed3f83e3e35b517af2ec9116bba8296d51be31b8a57462cf92e23de2a8c008"
VECTOR_B1_HASH = "000062242a22cb4bf73e2e98c335039d7247888cd3d0c3e017ccf50eb2f1420f"
VECTOR_B1_TXID = "4cd9f90c162e3ba5b0b063340e6d6685dca3fca0aa0a49d575b4ac4c3035e9f0"
VECTOR_B1_TARGET = 1766847064778384329583297500742918515827483896875618958121606201292619776
VECTOR_B1_CHAIN_WORK = 131070

VECTOR_TX_SIG = "cb2105bd2d4184b8999dde26600fdae3fbf609aabd3c1f6cbd6a4d81c298420f44d26c3e4a83e678485d1dfe010e5a82639728a071334b97ead27f39bd752679"
VECTOR_TX_TXID = "a0762d0a439734af6b89b36b0b033168100862ed546f2665f7066b6e7b21e99c"
VECTOR_TX_SIGHASH = "2670f838a3601a10000d54777c7e22530723d3ec25364c250654a08bfea11a13"

FAST_POLICY_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-vectors-fast",
    initial_target=2**255,
    max_target=2**255,
    max_adjust_factor_up=1.0,
    max_adjust_factor_down=1.0,
    target_block_time=1,
)


class ConsensusVectorsTest(unittest.TestCase):
    def test_addresses_vector(self) -> None:
        self.assertEqual(ADDR_A, "KK915b6b92b37b765963ab61d52a3171a54da33778c1")
        self.assertEqual(ADDR_B, "KK914c98afd617dc61c12995e7cd96067f2c4c29fd97")

    def test_genesis_and_first_block_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            with mock.patch("powx.chain.secrets.randbits", side_effect=[1111111111111111, 2222222222222222]):
                with mock.patch.object(chain, "_now", return_value=1700000000):
                    genesis = chain.initialize(ADDR_A, genesis_supply=0)
                with mock.patch.object(chain, "_now", return_value=1700000030):
                    b1 = chain.mine_block(ADDR_A, mining_backend="cpu")

            self.assertEqual(genesis.block_hash, VECTOR_GENESIS_HASH)
            self.assertEqual(genesis.transactions[0].txid, VECTOR_GENESIS_TXID)
            self.assertEqual(b1.block_hash, VECTOR_B1_HASH)
            self.assertEqual(b1.transactions[0].txid, VECTOR_B1_TXID)
            self.assertEqual(b1.target, VECTOR_B1_TARGET)
            self.assertEqual(b1.chain_work, VECTOR_B1_CHAIN_WORK)

            # Reload and re-validate from disk.
            chain_reloaded = Chain(td)
            chain_reloaded.load()
            self.assertEqual(chain_reloaded.height, 1)
            self.assertEqual(chain_reloaded.tip.block_hash if chain_reloaded.tip else "", VECTOR_B1_HASH)

    def test_transaction_signature_and_txid_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            tx = Transaction(
                version=1,
                timestamp=1700000100,
                nonce=4444444444444444,
                inputs=[TxInput(txid="aa" * 32, index=0, pubkey=PUB_A)],
                outputs=[
                    TxOutput(amount=900, address=ADDR_B),
                    TxOutput(amount=99, address=ADDR_A),
                ],
            )
            sig = sign_digest(PRIV_A, tx.signing_hash())
            tx.inputs[0].signature = sig
            tx.txid = tx.compute_txid()

            self.assertEqual(tx.signing_hash().hex(), VECTOR_TX_SIGHASH)
            self.assertEqual(sig, VECTOR_TX_SIG)
            self.assertEqual(tx.txid, VECTOR_TX_TXID)

            utxo_view = {
                f"{'aa' * 32}:0": {
                    "amount": 1000,
                    "address": ADDR_A,
                }
            }
            fee = chain.validate_transaction(tx, utxo_view, block_height=1)
            self.assertEqual(fee, 1)

    def test_candidate_target_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            with mock.patch("powx.chain.secrets.randbits", side_effect=[1111111111111111, 2222222222222222]):
                with mock.patch.object(chain, "_now", return_value=1700000000):
                    genesis = chain.initialize(ADDR_A, genesis_supply=0)
                with mock.patch.object(chain, "_now", return_value=1700000030):
                    b1 = chain.mine_block(ADDR_A, mining_backend="cpu")

            tampered = b1.to_dict()
            tampered["target"] = str(int(tampered["target"]) + 1)

            with self.assertRaises(ValidationError):
                chain.validate_chain_blocks([genesis, Block.from_dict(tampered)])

    def test_halving_schedule_vector(self) -> None:
        cfg = replace(FAST_POLICY_CONFIG, halving_interval=2, initial_block_reward=50)
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            expected = {
                0: 50,
                1: 50,
                2: 25,
                3: 25,
                4: 12,
                5: 12,
                6: 6,
                7: 6,
                128: 0,
            }
            for height, reward in expected.items():
                self.assertEqual(chain.block_reward(height), reward, msg=f"height={height}")

    def test_supply_cap_vector(self) -> None:
        cfg = replace(
            FAST_POLICY_CONFIG,
            halving_interval=2,
            initial_block_reward=50,
            max_total_supply=120,
        )
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)

            rewards: list[int] = []
            for _ in range(6):
                block = chain.mine_block(ADDR_A, mining_backend="cpu")
                rewards.append(sum(out.amount for out in block.transactions[0].outputs))

            self.assertEqual(rewards, [50, 25, 25, 12, 8, 0])
            self.assertEqual(chain.issued_supply(), 120)
            self.assertEqual(chain.status()["remaining_supply"], 0)

    def test_default_tokenomics_calibration_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td)
            halving_interval = int(chain.config.halving_interval)
            theoretical_uncapped = halving_interval * sum(
                chain.config.initial_block_reward >> era for era in range(64)
            )

            self.assertEqual(chain.config.max_total_supply, 911_000_000)
            self.assertEqual(chain.config.initial_block_reward, 2_173)
            self.assertEqual(theoretical_uncapped, 911_190_000)
            self.assertEqual(theoretical_uncapped - chain.config.max_total_supply, 190_000)

    def test_reorg_depth_limit_vector(self) -> None:
        cfg = replace(FAST_POLICY_CONFIG, max_reorg_depth=1)

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            chain_main = Chain(d1, config=cfg)
            chain_alt = Chain(d2, config=cfg)

            # Force identical genesis on both chains.
            with mock.patch("powx.chain.secrets.randbits", return_value=777):
                with mock.patch.object(chain_main, "_now", return_value=1700000000):
                    chain_main.initialize(ADDR_A, genesis_supply=0)
                with mock.patch.object(chain_alt, "_now", return_value=1700000000):
                    chain_alt.initialize(ADDR_A, genesis_supply=0)

            for _ in range(3):
                chain_main.mine_block(ADDR_A, mining_backend="cpu")
            for _ in range(4):
                chain_alt.mine_block(ADDR_A, mining_backend="cpu")

            with self.assertRaisesRegex(ValidationError, "Reorg depth 3 exceeds max_reorg_depth 1"):
                chain_main.replace_chain(chain_alt.chain, require_better=True)


if __name__ == "__main__":
    unittest.main()
