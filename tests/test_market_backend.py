from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest import mock

from powx.market_backend import MarketplaceBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 4.0) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url=url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        status = int(response.status)
        body = response.read().decode("utf-8")
    return status, json.loads(body)


def _fixture_payloads() -> tuple[dict, dict, dict, dict]:
    status = {"ok": True, "height": 44, "chain_id": "kk91-testnet"}
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
                "updated_height": 12,
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
                "price": 75,
                "listed_height": 13,
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


class MarketBackendTest(unittest.TestCase):
    def _start_backend(self) -> tuple[MarketplaceBackend, threading.Thread]:
        port = _free_port()
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        backend = MarketplaceBackend(
            node_url="http://127.0.0.1:8844",
            db_path=f"{tempdir.name}/market.db",
            host="127.0.0.1",
            port=port,
            sync_interval=30.0,
            request_timeout=0.5,
            auto_sync=False,
        )
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(backend.shutdown)

        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                status, payload = _http_json("GET", f"{backend.base_url}/health", timeout=0.4)
                if status == 200 and payload.get("ok"):
                    return backend, thread
            except Exception:
                time.sleep(0.05)
        raise AssertionError("Backend did not start in time")

    def test_backend_serves_status_data_and_static_ui(self) -> None:
        backend, _thread = self._start_backend()
        status, nfts, listings, contracts = _fixture_payloads()
        backend.indexer.sync_from_payload(
            status_payload=status,
            nft_payload=nfts,
            listing_payload=listings,
            contract_payload=contracts,
            source="test",
        )

        code, status_payload = _http_json("GET", f"{backend.base_url}/api/v1/market/status")
        self.assertEqual(code, 200)
        self.assertTrue(status_payload["ok"])
        self.assertEqual(status_payload["indexer"]["nft_count"], 2)
        self.assertEqual(status_payload["indexer"]["listing_count"], 1)
        self.assertEqual(status_payload["indexer"]["contract_count"], 1)

        code, nft_payload = _http_json("GET", f"{backend.base_url}/api/v1/market/nfts?listed=true")
        self.assertEqual(code, 200)
        self.assertEqual(nft_payload["count"], 1)
        self.assertEqual(nft_payload["rows"][0]["token_id"], "ART-002")

        req = Request(url=f"{backend.base_url}/", method="GET")
        with urlopen(req, timeout=3.0) as response:
            html = response.read().decode("utf-8")
        self.assertIn("KK91 Market Hub", html)

    def test_backend_build_and_submit_endpoints(self) -> None:
        backend, _thread = self._start_backend()
        with mock.patch(
            "powx.market_backend.api_build_contract_transaction",
            return_value={"ok": True, "tx": {"txid": "unsigned-demo"}},
        ) as build_mock:
            code, payload = _http_json(
                "POST",
                f"{backend.base_url}/api/v1/market/build-contract",
                payload={
                    "sender_pubkey": "a" * 66,
                    "contract": {"kind": "nft", "action": "mint", "token_id": "ART-X", "metadata_uri": "ipfs://x"},
                    "fee": 1,
                },
            )
            self.assertEqual(code, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["tx"]["txid"], "unsigned-demo")
            build_mock.assert_called_once()

        with mock.patch(
            "powx.market_backend.api_submit_transaction",
            return_value={"ok": True, "txid": "signed-demo"},
        ) as submit_mock:
            code, payload = _http_json(
                "POST",
                f"{backend.base_url}/api/v1/market/submit-signed",
                payload={
                    "tx": {
                        "version": 2,
                        "timestamp": 1_700_000_000,
                        "nonce": 1,
                        "inputs": [],
                        "outputs": [],
                        "txid": "signed-demo",
                    }
                },
            )
            self.assertEqual(code, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["txid"], "signed-demo")
            submit_mock.assert_called_once()

    def test_backend_submit_contract_endpoint(self) -> None:
        backend, _thread = self._start_backend()
        with mock.patch(
            "powx.market_backend.api_create_contract_transaction",
            return_value={"ok": True, "txid": "minted-demo", "submit": {"ok": True, "txid": "minted-demo"}},
        ) as submit_contract_mock, mock.patch.object(backend, "sync_once", return_value={"ok": True}) as sync_mock:
            code, payload = _http_json(
                "POST",
                f"{backend.base_url}/api/v1/market/submit-contract",
                payload={
                    "private_key_hex": "1" * 64,
                    "contract": {"kind": "nft", "action": "mint", "token_id": "ART-TEST", "metadata_uri": "ipfs://test"},
                    "fee": 2,
                    "broadcast_ttl": 3,
                },
            )
            self.assertEqual(code, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["txid"], "minted-demo")
            submit_contract_mock.assert_called_once()
            sync_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
