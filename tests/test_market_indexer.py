from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from powx.market_indexer import MarketIndexer


def _fixture_payloads() -> tuple[dict, dict, dict, dict]:
    status = {"ok": True, "height": 42, "chain_id": "kk91-testnet"}
    nfts = {
        "ok": True,
        "tokens": {
            "ART-001": {
                "token_id": "ART-001",
                "creator": "KK91creator",
                "owner": "KK91ownerA",
                "metadata_uri": "ipfs://art-001",
                "mint_height": 10,
                "mint_txid": "aa",
                "updated_height": 10,
                "updated_txid": "aa",
            },
            "ART-002": {
                "token_id": "ART-002",
                "creator": "KK91creator",
                "owner": "KK91ownerB",
                "metadata_uri": "ipfs://art-002",
                "mint_height": 11,
                "mint_txid": "bb",
                "updated_height": 11,
                "updated_txid": "bb",
            },
        },
        "count": 2,
    }
    listings = {
        "ok": True,
        "listings": {
            "ART-002": {
                "token_id": "ART-002",
                "seller": "KK91ownerB",
                "price": 50,
                "listed_height": 12,
                "listed_txid": "cc",
            }
        },
        "count": 1,
    }
    contracts = {
        "ok": True,
        "contracts": {
            "kv-demo": {
                "contract_id": "kv-demo",
                "template": "kv_v1",
                "owner": "KK91ownerA",
                "state": {"hello": "world"},
                "created_height": 20,
                "created_txid": "dd",
                "updated_height": 21,
                "updated_txid": "ee",
            }
        },
        "count": 1,
    }
    return status, nfts, listings, contracts


class MarketIndexerTest(unittest.TestCase):
    def test_sync_from_payload_and_query_views(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            indexer = MarketIndexer(db_path=f"{td}/market.db", node_url="http://127.0.0.1:8844")
            status, nfts, listings, contracts = _fixture_payloads()

            result = indexer.sync_from_payload(
                status_payload=status,
                nft_payload=nfts,
                listing_payload=listings,
                contract_payload=contracts,
                source="test",
            )
            self.assertEqual(result["nft_count"], 2)
            self.assertEqual(result["listing_count"], 1)
            self.assertEqual(result["contract_count"], 1)
            self.assertEqual(result["synced_height"], 42)

            listed_rows = indexer.list_nfts(listed=True)
            self.assertEqual(len(listed_rows), 1)
            self.assertEqual(listed_rows[0]["token_id"], "ART-002")
            self.assertTrue(listed_rows[0]["listed"])

            owner_rows = indexer.list_nfts(owner="KK91ownerA")
            self.assertEqual(len(owner_rows), 1)
            self.assertEqual(owner_rows[0]["token_id"], "ART-001")

            listing_rows = indexer.list_listings(seller="KK91ownerB")
            self.assertEqual(len(listing_rows), 1)
            self.assertEqual(listing_rows[0]["price"], 50)

            contract_rows = indexer.list_contracts(contract_id="kv-demo")
            self.assertEqual(len(contract_rows), 1)
            self.assertEqual(contract_rows[0]["state"]["hello"], "world")

            nfts_update = dict(nfts)
            nfts_update["tokens"] = {"ART-001": nfts["tokens"]["ART-001"]}
            listings_update = {"ok": True, "listings": {}, "count": 0}
            contracts_update = {"ok": True, "contracts": {}, "count": 0}
            status_update = {"ok": True, "height": 43, "chain_id": "kk91-testnet"}
            indexer.sync_from_payload(
                status_payload=status_update,
                nft_payload=nfts_update,
                listing_payload=listings_update,
                contract_payload=contracts_update,
                source="test-update",
            )
            stats = indexer.stats()
            self.assertEqual(stats["nft_count"], 1)
            self.assertEqual(stats["listing_count"], 0)
            self.assertEqual(stats["contract_count"], 0)
            self.assertEqual(stats["last_sync_height"], 43)

    def test_sync_once_uses_api_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            indexer = MarketIndexer(db_path=f"{td}/market.db", node_url="http://127.0.0.1:8844")
            status, nfts, listings, contracts = _fixture_payloads()
            with mock.patch("powx.market_indexer.api_status", return_value=status), mock.patch(
                "powx.market_indexer.api_nfts", return_value=nfts
            ), mock.patch("powx.market_indexer.api_nft_listings", return_value=listings), mock.patch(
                "powx.market_indexer.api_contracts", return_value=contracts
            ):
                result = indexer.sync_once()
            self.assertEqual(result["nft_count"], 2)
            self.assertEqual(result["listing_count"], 1)
            self.assertEqual(result["contract_count"], 1)


if __name__ == "__main__":
    unittest.main()
