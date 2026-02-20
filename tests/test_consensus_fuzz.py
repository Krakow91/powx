from __future__ import annotations

import copy
import random
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
ADDR_B = address_from_public_key(private_key_to_public_key(PRIV_B).hex(), "KK91")

FAST_CONFIG = replace(
    CONFIG,
    max_target=2**255,
    initial_target=2**255,
)


def _rand_hex(rng: random.Random, length: int) -> str:
    alphabet = "0123456789abcdef"
    return "".join(rng.choice(alphabet) for _ in range(length))


class ConsensusFuzzTest(unittest.TestCase):
    def _build_fast_chain(self, td: str) -> tuple[Chain, Block, Block]:
        chain = Chain(td, config=FAST_CONFIG)
        with mock.patch("powx.chain.secrets.randbits", return_value=7):
            with mock.patch.object(chain, "_now", return_value=1700000000):
                genesis = chain.initialize(ADDR_A, genesis_supply=0)
            with mock.patch.object(chain, "_now", return_value=1700000001):
                b1 = chain.mine_block(ADDR_A, mining_backend="cpu")
        return chain, genesis, b1

    def _mutate_tx_dict(self, tx_dict: dict, rng: random.Random) -> None:
        choice = rng.randrange(12)
        if choice == 0:
            tx_dict["version"] = rng.choice([-3, -1, 0, 1, 2, "x"])
        elif choice == 1:
            tx_dict["timestamp"] = rng.choice([-1, 0, 1, 2**40, "bad"])
        elif choice == 2:
            tx_dict["nonce"] = rng.choice([-9, 0, 1, "bad"])
        elif choice == 3:
            tx_dict["txid"] = _rand_hex(rng, rng.choice([4, 8, 16, 63, 64, 65]))
        elif choice == 4:
            tx_dict["inputs"] = []
        elif choice == 5 and tx_dict.get("inputs"):
            tx_dict["inputs"][0]["signature"] = _rand_hex(rng, rng.choice([2, 64, 128, 130]))
        elif choice == 6 and tx_dict.get("inputs"):
            tx_dict["inputs"][0]["pubkey"] = _rand_hex(rng, rng.choice([2, 32, 66, 70]))
        elif choice == 7 and tx_dict.get("outputs"):
            tx_dict["outputs"][0]["amount"] = rng.choice([-5, -1, 0, 1, "bad"])
        elif choice == 8 and tx_dict.get("outputs"):
            tx_dict["outputs"][0]["address"] = rng.choice(
                [
                    "",
                    "KK91",
                    "bad",
                    "KK91" + _rand_hex(rng, 39),
                    "KK91" + _rand_hex(rng, 41),
                    "KK91" + _rand_hex(rng, 40),
                ]
            )
        elif choice == 9:
            tx_dict["outputs"] = []
        elif choice == 10:
            tx_dict["inputs"] = [{"txid": _rand_hex(rng, 64), "index": rng.choice([-1, "x"])}]
        else:
            tx_dict.pop(rng.choice(["version", "timestamp", "nonce", "inputs", "outputs", "txid"]), None)

    def _mutate_block_dict(self, block_dict: dict, rng: random.Random) -> None:
        choice = rng.randrange(14)
        if choice == 0:
            block_dict["index"] = rng.choice([-1, 0, 1, 2, "x"])
        elif choice == 1:
            block_dict["prev_hash"] = _rand_hex(rng, rng.choice([8, 16, 63, 64, 65]))
        elif choice == 2:
            block_dict["timestamp"] = rng.choice([-1, 0, 1, 2**40, "bad"])
        elif choice == 3:
            block_dict["target"] = rng.choice([0, 1, 2**255, "bad"])
        elif choice == 4:
            block_dict["nonce"] = rng.choice([-1, 0, 1, "bad"])
        elif choice == 5:
            block_dict["merkle_root"] = _rand_hex(rng, rng.choice([8, 16, 63, 64, 65]))
        elif choice == 6:
            block_dict["miner"] = rng.choice(["", "bad", "KK91" + _rand_hex(rng, 40)])
        elif choice == 7:
            block_dict["chain_work"] = rng.choice([0, 1, 2, "bad"])
        elif choice == 8:
            block_dict["block_hash"] = _rand_hex(rng, rng.choice([8, 16, 63, 64, 65]))
        elif choice == 9:
            block_dict["transactions"] = []
        elif choice == 10 and block_dict.get("transactions"):
            block_dict["transactions"][0]["txid"] = _rand_hex(rng, rng.choice([8, 32, 63, 64, 65]))
        elif choice == 11 and block_dict.get("transactions"):
            block_dict["transactions"][0]["outputs"] = []
        elif choice == 12:
            block_dict["transactions"] = [{"version": 1, "timestamp": 1, "nonce": 1}]
        else:
            block_dict.pop(
                rng.choice(
                    [
                        "index",
                        "prev_hash",
                        "timestamp",
                        "target",
                        "nonce",
                        "merkle_root",
                        "miner",
                        "transactions",
                        "chain_work",
                        "block_hash",
                    ]
                ),
                None,
            )

    def test_transaction_fuzz_no_crash(self) -> None:
        rng = random.Random(1337)
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_CONFIG)
            utxo_key = f"{'aa' * 32}:0"
            utxo_view = {utxo_key: {"amount": 1000, "address": ADDR_A}}

            base_tx = Transaction(
                version=1,
                timestamp=1700000100,
                nonce=4444444444444444,
                inputs=[TxInput(txid="aa" * 32, index=0, pubkey=PUB_A)],
                outputs=[
                    TxOutput(amount=900, address=ADDR_B),
                    TxOutput(amount=99, address=ADDR_A),
                ],
            )
            base_tx.inputs[0].signature = sign_digest(PRIV_A, base_tx.signing_hash())
            base_tx.txid = base_tx.compute_txid()

            for _ in range(300):
                candidate_dict = copy.deepcopy(base_tx.to_dict())
                self._mutate_tx_dict(candidate_dict, rng)
                try:
                    candidate_tx = Transaction.from_dict(candidate_dict)
                except (KeyError, TypeError, ValueError):
                    continue

                try:
                    _fee = chain.validate_transaction(candidate_tx, copy.deepcopy(utxo_view), block_height=1)
                except ValidationError:
                    continue
                except Exception as exc:
                    self.fail(f"Unexpected exception in tx fuzz: {type(exc).__name__}: {exc}")

    def test_candidate_block_fuzz_no_crash(self) -> None:
        rng = random.Random(7331)
        with tempfile.TemporaryDirectory() as td:
            chain, genesis, b1 = self._build_fast_chain(td)
            genesis_dict = genesis.to_dict()
            b1_dict = b1.to_dict()

            for _ in range(200):
                mut_b1 = copy.deepcopy(b1_dict)
                self._mutate_block_dict(mut_b1, rng)

                try:
                    blocks = [Block.from_dict(genesis_dict), Block.from_dict(mut_b1)]
                except (KeyError, TypeError, ValueError):
                    continue

                try:
                    _validated_chain, _validated_utxos = chain.validate_chain_blocks(blocks)
                except ValidationError:
                    continue
                except Exception as exc:
                    self.fail(f"Unexpected exception in block fuzz: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    unittest.main()
