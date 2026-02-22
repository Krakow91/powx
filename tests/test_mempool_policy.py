from __future__ import annotations

import copy
import secrets
import tempfile
import unittest
from dataclasses import replace

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key, sign_digest
from powx.models import Transaction, TxInput, TxOutput


PRIV_A = "1" * 64
PUB_A = private_key_to_public_key(PRIV_A).hex()
ADDR_A = address_from_public_key(PUB_A, "KK91")

PRIV_B = "2" * 64
PUB_B = private_key_to_public_key(PRIV_B).hex()
ADDR_B = address_from_public_key(PUB_B, "KK91")


FAST_MEMPOOL_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-mempool-test",
    coinbase_maturity=0,
    initial_target=2**255,
    max_target=2**255,
    max_adjust_factor_up=1.0,
    max_adjust_factor_down=1.0,
    target_block_time=1,
    asert_half_life=60,
    max_block_timestamp_step_seconds=3600,
)


def _build_spend_tx(
    chain: Chain,
    prev_txid: str,
    prev_index: int,
    prev_amount: int,
    fee: int,
    to_address: str = ADDR_A,
) -> Transaction:
    amount = prev_amount - fee
    if amount <= 0:
        raise ValueError("Invalid spend amount")
    tx = Transaction(
        version=chain.tx_version_for_height(max(0, chain.height + 1)),
        timestamp=chain._now(),
        nonce=secrets.randbits(64),
        inputs=[TxInput(txid=prev_txid, index=prev_index, pubkey=PUB_A)],
        outputs=[TxOutput(amount=amount, address=to_address)],
    )
    sighash = tx.signing_hash()
    tx.inputs[0].signature = sign_digest(PRIV_A, sighash)
    tx.txid = tx.compute_txid()
    return tx


class MempoolPolicyTest(unittest.TestCase):
    def _mine_blocks(self, chain: Chain, count: int) -> None:
        for _ in range(count):
            chain.mine_block(ADDR_A, mining_backend="cpu")

    def test_fee_rate_eviction_keeps_higher_fee_txs(self) -> None:
        cfg = replace(
            FAST_MEMPOOL_CONFIG,
            max_mempool_transactions=2,
            max_mempool_virtual_bytes=500_000,
        )
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)
            self._mine_blocks(chain, 4)

            low = chain.create_transaction(PRIV_A, ADDR_B, amount=49, fee=1)
            chain.add_transaction(low)
            mid = chain.create_transaction(PRIV_A, ADDR_B, amount=46, fee=4)
            chain.add_transaction(mid)
            high = chain.create_transaction(PRIV_A, ADDR_B, amount=42, fee=8)
            chain.add_transaction(high)

            txids = {tx.txid for tx in chain.mempool}
            self.assertEqual(len(txids), 2)
            self.assertIn(high.txid, txids)
            self.assertIn(mid.txid, txids)
            self.assertNotIn(low.txid, txids)

    def test_ancestor_limit_blocks_too_deep_chain(self) -> None:
        cfg = replace(
            FAST_MEMPOOL_CONFIG,
            mempool_ancestor_limit=1,
            mempool_descendant_limit=10,
            max_mempool_transactions=50,
        )
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)
            self._mine_blocks(chain, 2)

            parent = chain.create_transaction(PRIV_A, ADDR_B, amount=2, fee=1)
            chain.add_transaction(parent)

            parent_change = int(parent.outputs[1].amount)
            child = _build_spend_tx(
                chain,
                prev_txid=parent.txid,
                prev_index=1,
                prev_amount=parent_change,
                fee=2,
                to_address=ADDR_A,
            )
            chain.add_transaction(child)

            child_amount = int(child.outputs[0].amount)
            grandchild = _build_spend_tx(
                chain,
                prev_txid=child.txid,
                prev_index=0,
                prev_amount=child_amount,
                fee=2,
                to_address=ADDR_A,
            )
            with self.assertRaisesRegex(ValidationError, "too many unconfirmed ancestors"):
                chain.add_transaction(grandchild)

    def test_rbf_replaces_conflicting_tx_with_higher_fee(self) -> None:
        cfg = replace(
            FAST_MEMPOOL_CONFIG,
            mempool_rbf_enabled=True,
            min_rbf_fee_delta=1,
            min_rbf_feerate_delta=0.0,
        )
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)
            self._mine_blocks(chain, 1)

            utxos_before = dict(chain.utxos)
            original = chain.create_transaction(PRIV_A, ADDR_B, amount=49, fee=1)
            chain.add_transaction(original)

            total_in = 0
            for tx_input in original.inputs:
                key = f"{tx_input.txid}:{tx_input.index}"
                total_in += int(utxos_before[key]["amount"])

            bad_replacement = Transaction(
                version=original.version,
                timestamp=chain._now(),
                nonce=secrets.randbits(64),
                inputs=[TxInput(txid=item.txid, index=item.index, pubkey=item.pubkey) for item in original.inputs],
                outputs=[TxOutput(amount=total_in - 1, address=ADDR_B)],
            )
            bad_sighash = bad_replacement.signing_hash()
            for tx_input in bad_replacement.inputs:
                tx_input.signature = sign_digest(PRIV_A, bad_sighash)
            bad_replacement.txid = bad_replacement.compute_txid()
            with self.assertRaisesRegex(ValidationError, "RBF replacement fee delta too small"):
                chain.add_transaction(bad_replacement)

            replacement = Transaction(
                version=original.version,
                timestamp=chain._now(),
                nonce=secrets.randbits(64),
                inputs=[TxInput(txid=item.txid, index=item.index, pubkey=item.pubkey) for item in original.inputs],
                outputs=[TxOutput(amount=total_in - 3, address=ADDR_B)],
            )
            sighash = replacement.signing_hash()
            for tx_input in replacement.inputs:
                tx_input.signature = sign_digest(PRIV_A, sighash)
            replacement.txid = replacement.compute_txid()
            chain.add_transaction(replacement)

            txids = {tx.txid for tx in chain.mempool}
            self.assertIn(replacement.txid, txids)
            self.assertNotIn(original.txid, txids)

    def test_cpfp_package_can_beat_single_tx_in_mining(self) -> None:
        cfg = replace(
            FAST_MEMPOOL_CONFIG,
            max_transactions_per_block=3,
            mempool_cpfp_enabled=True,
            max_mempool_transactions=200,
        )
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)
            self._mine_blocks(chain, 4)

            parent = chain.create_transaction(PRIV_A, ADDR_B, amount=2, fee=1)
            chain.add_transaction(parent)
            child = _build_spend_tx(
                chain,
                prev_txid=parent.txid,
                prev_index=1,
                prev_amount=int(parent.outputs[1].amount),
                fee=40,
                to_address=ADDR_A,
            )
            chain.add_transaction(child)

            single = chain.create_transaction(PRIV_A, ADDR_B, amount=39, fee=11)
            chain.add_transaction(single)

            mined = chain.mine_block(ADDR_A, mining_backend="cpu")
            mined_txids = {tx.txid for tx in mined.transactions[1:]}
            self.assertEqual(len(mined_txids), 2)
            self.assertIn(parent.txid, mined_txids)
            self.assertIn(child.txid, mined_txids)
            self.assertNotIn(single.txid, mined_txids)

    def test_coinbase_maturity_rejects_immature_spend(self) -> None:
        cfg = replace(FAST_MEMPOOL_CONFIG, coinbase_maturity=5)
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=0)
            self._mine_blocks(chain, 1)

            coinbase = chain.chain[-1].transactions[0]
            amount = int(coinbase.outputs[0].amount)
            spend = _build_spend_tx(
                chain,
                prev_txid=coinbase.txid,
                prev_index=0,
                prev_amount=amount,
                fee=1,
                to_address=ADDR_B,
            )
            with self.assertRaisesRegex(ValidationError, "immature"):
                chain.validate_transaction(
                    spend,
                    copy.deepcopy(chain.utxos),
                    for_mempool=True,
                    block_height=max(0, chain.height + 1),
                )

    def test_dust_output_rejected_by_standard_policy(self) -> None:
        cfg = replace(FAST_MEMPOOL_CONFIG, coinbase_maturity=0, min_dust_output=5)
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=1_000)
            genesis = chain.chain[0].transactions[0]
            dust_tx = _build_spend_tx(
                chain,
                prev_txid=genesis.txid,
                prev_index=0,
                prev_amount=1_000,
                fee=999,
                to_address=ADDR_B,
            )
            with self.assertRaisesRegex(ValidationError, "Dust output"):
                chain.validate_transaction(
                    dust_tx,
                    copy.deepcopy(chain.utxos),
                    for_mempool=True,
                    block_height=1,
                )

    def test_non_standard_pubkey_rejected_by_mempool_policy(self) -> None:
        cfg = replace(FAST_MEMPOOL_CONFIG, coinbase_maturity=0)
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            genesis = chain.chain[0].transactions[0]
            bad_pubkey = "04" + ("11" * 64)
            tx = Transaction(
                version=chain.tx_version_for_height(1),
                timestamp=chain._now(),
                nonce=secrets.randbits(64),
                inputs=[TxInput(txid=genesis.txid, index=0, pubkey=bad_pubkey)],
                outputs=[TxOutput(amount=999, address=ADDR_B)],
            )
            sighash = tx.signing_hash()
            tx.inputs[0].signature = sign_digest(PRIV_A, sighash)
            tx.txid = tx.compute_txid()

            with self.assertRaisesRegex(ValidationError, "Non-standard pubkey"):
                chain.validate_transaction(
                    tx,
                    copy.deepcopy(chain.utxos),
                    for_mempool=True,
                    block_height=1,
                )


if __name__ == "__main__":
    unittest.main()
