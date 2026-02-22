from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key


PRIV_A = "1" * 64
PUB_A = private_key_to_public_key(PRIV_A).hex()
ADDR_A = address_from_public_key(PUB_A, "KK91")

PRIV_B = "2" * 64
PUB_B = private_key_to_public_key(PRIV_B).hex()
ADDR_B = address_from_public_key(PUB_B, "KK91")


FAST_CONTRACT_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-contract-test",
    coinbase_maturity=0,
    min_dust_output=1,
    initial_target=2**255,
    max_target=2**255,
    max_adjust_factor_up=1.0,
    max_adjust_factor_down=1.0,
    target_block_time=1,
    asert_half_life=60,
    max_block_timestamp_step_seconds=3600,
)


class NFTAndSmartContractTest(unittest.TestCase):
    def _mine(self, chain: Chain, count: int = 1) -> None:
        for _ in range(count):
            chain.mine_block(ADDR_A, mining_backend="cpu")

    def test_nft_marketplace_mint_list_buy_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_CONTRACT_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            fund_b = chain.create_transaction(PRIV_A, ADDR_B, amount=200, fee=1)
            chain.add_transaction(fund_b)
            self._mine(chain, 1)

            mint_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_A,
                contract_payload={
                    "kind": "nft",
                    "action": "mint",
                    "token_id": "ART-001",
                    "metadata_uri": "ipfs://art-001",
                },
                fee=1,
            )
            chain.add_transaction(mint_tx)
            self._mine(chain, 1)

            list_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_A,
                contract_payload={
                    "kind": "nft",
                    "action": "list",
                    "token_id": "ART-001",
                    "price": 50,
                },
                fee=1,
            )
            chain.add_transaction(list_tx)
            self._mine(chain, 1)

            buy_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_B,
                contract_payload={
                    "kind": "nft",
                    "action": "buy",
                    "token_id": "ART-001",
                },
                fee=1,
                to_address=ADDR_A,
                amount=50,
            )
            chain.add_transaction(buy_tx)
            self._mine(chain, 1)

            token_state = chain.nft_state("ART-001")
            self.assertTrue(token_state["exists"])
            self.assertEqual(token_state["token"]["owner"], ADDR_B)
            self.assertEqual(chain.nft_listings_state()["count"], 0)

    def test_nft_buy_without_seller_payment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_CONTRACT_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            fund_b = chain.create_transaction(PRIV_A, ADDR_B, amount=200, fee=1)
            chain.add_transaction(fund_b)
            self._mine(chain, 1)

            chain.add_transaction(
                chain.create_contract_transaction(
                    private_key_hex=PRIV_A,
                    contract_payload={
                        "kind": "nft",
                        "action": "mint",
                        "token_id": "ART-002",
                        "metadata_uri": "ipfs://art-002",
                    },
                    fee=1,
                )
            )
            self._mine(chain, 1)

            chain.add_transaction(
                chain.create_contract_transaction(
                    private_key_hex=PRIV_A,
                    contract_payload={
                        "kind": "nft",
                        "action": "list",
                        "token_id": "ART-002",
                        "price": 75,
                    },
                    fee=1,
                )
            )
            self._mine(chain, 1)

            bad_buy = chain.create_contract_transaction(
                private_key_hex=PRIV_B,
                contract_payload={
                    "kind": "nft",
                    "action": "buy",
                    "token_id": "ART-002",
                },
                fee=1,
                to_address=ADDR_B,
                amount=1,
            )
            with self.assertRaisesRegex(ValidationError, "does not pay seller"):
                chain.add_transaction(bad_buy)

    def test_smart_contract_deploy_and_owner_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_CONTRACT_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            deploy_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_A,
                contract_payload={
                    "kind": "sc",
                    "action": "deploy",
                    "contract_id": "kv-demo",
                    "template": "kv_v1",
                    "init": {"hello": "world"},
                },
                fee=1,
            )
            chain.add_transaction(deploy_tx)
            self._mine(chain, 1)

            call_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_A,
                contract_payload={
                    "kind": "sc",
                    "action": "call",
                    "contract_id": "kv-demo",
                    "method": "set",
                    "args": {"key": "color", "value": "neon"},
                },
                fee=1,
            )
            chain.add_transaction(call_tx)
            self._mine(chain, 1)

            contract_state = chain.smart_contract_state("kv-demo")
            self.assertTrue(contract_state["exists"])
            self.assertEqual(contract_state["contract"]["state"]["hello"], "world")
            self.assertEqual(contract_state["contract"]["state"]["color"], "neon")

    def test_smart_contract_non_owner_call_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=FAST_CONTRACT_CONFIG)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            fund_b = chain.create_transaction(PRIV_A, ADDR_B, amount=100, fee=1)
            chain.add_transaction(fund_b)
            self._mine(chain, 1)

            deploy_tx = chain.create_contract_transaction(
                private_key_hex=PRIV_A,
                contract_payload={
                    "kind": "sc",
                    "action": "deploy",
                    "contract_id": "kv-owner-test",
                    "template": "kv_v1",
                },
                fee=1,
            )
            chain.add_transaction(deploy_tx)
            self._mine(chain, 1)

            unauthorized_call = chain.create_contract_transaction(
                private_key_hex=PRIV_B,
                contract_payload={
                    "kind": "sc",
                    "action": "call",
                    "contract_id": "kv-owner-test",
                    "method": "set",
                    "args": {"key": "x", "value": "y"},
                },
                fee=1,
            )
            with self.assertRaisesRegex(ValidationError, "Only contract owner"):
                chain.add_transaction(unauthorized_call)


if __name__ == "__main__":
    unittest.main()
