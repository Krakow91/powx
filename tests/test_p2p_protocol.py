from __future__ import annotations

import socket
import tempfile
import unittest
import json
from urllib.parse import urlparse
from unittest import mock
from dataclasses import replace

from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key
from powx.models import Block, Transaction, TxInput, TxOutput
from powx.p2p import P2PNode, api_create_contract_transaction, api_create_transaction, normalize_peer, sign_transaction_offline


PRIV_A = "1" * 64
ADDR_A = address_from_public_key(private_key_to_public_key(PRIV_A).hex(), "KK91")

FAST_P2P_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-p2p-test",
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class P2PProtocolTest(unittest.TestCase):
    @staticmethod
    def _close_node(node: P2PNode) -> None:
        try:
            node.stop_event.set()
        except Exception:
            pass
        try:
            node.server.server_close()
        except Exception:
            pass

    def _make_node(self, data_dir: str, **kwargs: object) -> P2PNode:
        port = int(kwargs.pop("port", _free_port()))
        chain_config = kwargs.pop("chain_config", FAST_P2P_CONFIG)
        node = P2PNode(
            data_dir=data_dir,
            host="127.0.0.1",
            port=port,
            sync_interval=0,
            chain_config=chain_config,
            **kwargs,
        )
        self.addCleanup(self._close_node, node)
        return node

    def _init_genesis(self, node: P2PNode, supply: int = 1000) -> Block:
        with mock.patch("powx.chain.secrets.randbits", return_value=777):
            with mock.patch.object(node.chain, "_now", return_value=1_700_000_000):
                block = node.chain.initialize(ADDR_A, genesis_supply=supply)
            if supply > 0:
                with mock.patch.object(node.chain, "_now", return_value=1_700_000_030):
                    node.chain.mine_block(ADDR_A, mining_backend="cpu")
        with node.chain_lock:
            node._refresh_chain_from_disk_unlocked()
            node._prime_seen_from_chain_unlocked()
        return block

    def test_fresh_node_uses_runtime_target_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            node = self._make_node(td)
            with node.chain_lock:
                self.assertEqual(node.chain.status()["target_schedule"], "asert-v3")

    def test_getdata_payload_returns_tx_and_block_with_auth(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            node = self._make_node(td)
            genesis = self._init_genesis(node, supply=1000)

            tx = node.chain.create_transaction(PRIV_A, ADDR_A, amount=10, fee=1)
            node.chain.add_transaction(tx)

            payload = node.getdata_payload(
                [
                    {"type": "tx", "id": tx.txid},
                    {"type": "block", "id": genesis.block_hash},
                ],
                ttl=2,
            )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["count"], 2)
            self.assertEqual(len(payload["rows"]), 2)
            for row in payload["rows"]:
                self.assertIn("auth", row)
                self.assertIsInstance(row["auth"], dict)

    def test_accept_inventory_fetches_unknown_block_via_getdata(self) -> None:
        with tempfile.TemporaryDirectory() as td_source, tempfile.TemporaryDirectory() as td_target:
            source = self._make_node(td_source)
            target = self._make_node(td_target)
            self._init_genesis(source, supply=0)
            self._init_genesis(target, supply=0)

            with mock.patch.object(source.chain, "_now", return_value=1_700_000_030):
                mined = source.chain.mine_block(ADDR_A, mining_backend="cpu")
            mined = Block.from_dict(mined.to_dict())

            target._peers.add(source.node_url)
            target._peer_pubkeys[source.node_url] = source._identity_public_key

            def fake_request_json(url: str, method: str = "GET", payload: dict | None = None, timeout: float | None = 4.0) -> dict:
                _ = method
                _ = payload
                _ = timeout
                path = urlparse(url).path
                if path == "/getdata":
                    return {
                        "ok": True,
                        "rows": [
                            {
                                "type": "block",
                                "id": mined.block_hash,
                                "block": mined.to_dict(),
                                "auth": source._build_sender_auth("block", mined.block_hash, 2),
                            }
                        ],
                        "missing": [],
                        "count": 1,
                    }
                if path == "/inv":
                    return {"ok": True}
                return {"ok": True}

            with mock.patch("powx.p2p._request_json", side_effect=fake_request_json):
                result = target.accept_inventory(
                    [{"type": "block", "id": mined.block_hash}],
                    sender=source.node_url,
                    ttl=2,
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["accepted_blocks"], 1)
            self.assertEqual(target.chain.height, 1)

    def test_orphan_block_is_queued_and_applied_after_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td_source, tempfile.TemporaryDirectory() as td_target:
            source = self._make_node(td_source)
            target = self._make_node(td_target)
            self._init_genesis(source, supply=0)
            self._init_genesis(target, supply=0)

            with mock.patch.object(source.chain, "_now", return_value=1_700_000_060):
                b1 = source.chain.mine_block(ADDR_A, mining_backend="cpu")
                b2 = source.chain.mine_block(ADDR_A, mining_backend="cpu")
            b1 = Block.from_dict(b1.to_dict())
            b2 = Block.from_dict(b2.to_dict())

            target._peers.add(source.node_url)
            target._peer_pubkeys[source.node_url] = source._identity_public_key

            orphan_result = target.accept_block(
                b2,
                sender=source.node_url,
                ttl=0,
                auth=source._build_sender_auth("block", b2.block_hash, 0),
            )
            self.assertFalse(orphan_result["accepted"])
            self.assertTrue(orphan_result.get("orphan"))
            self.assertEqual(orphan_result["reason"], "orphan-stored")

            parent_result = target.accept_block(
                b1,
                sender=source.node_url,
                ttl=0,
                auth=source._build_sender_auth("block", b1.block_hash, 0),
            )
            self.assertTrue(parent_result["accepted"])
            self.assertGreaterEqual(int(parent_result.get("adopted_orphans", 0)), 1)
            self.assertEqual(target.chain.height, 2)

            with target.chain_lock:
                self.assertEqual(len(target._orphan_blocks), 0)

    def test_relay_peer_selection_enforces_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            node = self._make_node(
                td,
                max_outbound_broadcast_peers=4,
                max_outbound_peers_per_bucket=1,
            )
            peers = [
                "http://10.1.1.1:9101",
                "http://10.1.2.2:9102",
                "http://11.2.1.1:9103",
                "http://12.2.1.1:9104",
                "http://example.com:9105",
            ]
            for peer in peers:
                node._peers.add(normalize_peer(peer))

            selected = node._get_relay_peers()
            self.assertLessEqual(len(selected), 4)

            buckets = [node._peer_diversity_bucket(peer) for peer in selected]
            self.assertEqual(len(buckets), len(set(buckets)))

    def test_api_create_transaction_uses_build_submit_without_key_leak(self) -> None:
        unsigned_tx = Transaction(
            version=1,
            timestamp=1_700_000_111,
            nonce=12345,
            inputs=[TxInput(txid="aa" * 32, index=0, pubkey=private_key_to_public_key(PRIV_A).hex())],
            outputs=[TxOutput(amount=9, address=ADDR_A)],
            txid="",
        )

        seen_paths: list[str] = []

        def fake_request_json(
            url: str,
            method: str = "GET",
            payload: dict | None = None,
            timeout: float | None = 4.0,
        ) -> dict:
            _ = method
            _ = timeout
            path = urlparse(url).path
            seen_paths.append(path)

            if path == "/api/v1/tx/build":
                self.assertIsInstance(payload, dict)
                self.assertNotIn("private_key_hex", payload)
                return {"ok": True, "tx": unsigned_tx.to_dict()}

            if path == "/api/v1/tx/submit":
                self.assertIsInstance(payload, dict)
                self.assertNotIn("private_key_hex", payload)
                tx_raw = payload.get("tx")
                self.assertIsInstance(tx_raw, dict)
                self.assertTrue(str(tx_raw.get("txid", "")))
                inputs = tx_raw.get("inputs", [])
                self.assertTrue(all(str(item.get("signature", "")) for item in inputs))
                return {"ok": True, "txid": str(tx_raw.get("txid", ""))}

            raise AssertionError(f"Unexpected API path: {path}")

        with mock.patch("powx.p2p._request_json", side_effect=fake_request_json):
            result = api_create_transaction(
                node_url="http://127.0.0.1:8844",
                private_key_hex=PRIV_A,
                to_address=ADDR_A,
                amount=9,
                fee=1,
                broadcast_ttl=2,
                timeout=5.0,
            )

        self.assertTrue(result.get("ok"))
        self.assertEqual(seen_paths, ["/api/v1/tx/build", "/api/v1/tx/submit"])

    def test_api_create_contract_transaction_uses_build_submit_without_key_leak(self) -> None:
        unsigned_tx = Transaction(
            version=2,
            timestamp=1_700_000_222,
            nonce=54321,
            inputs=[TxInput(txid="bb" * 32, index=0, pubkey=private_key_to_public_key(PRIV_A).hex())],
            outputs=[TxOutput(amount=2, address=ADDR_A)],
            contract={"kind": "nft", "action": "mint", "token_id": "ART-1", "metadata_uri": "ipfs://art-1"},
            txid="",
        )

        seen_paths: list[str] = []

        def fake_request_json(
            url: str,
            method: str = "GET",
            payload: dict | None = None,
            timeout: float | None = 4.0,
        ) -> dict:
            _ = method
            _ = timeout
            path = urlparse(url).path
            seen_paths.append(path)

            if path == "/api/v1/tx/build-contract":
                self.assertIsInstance(payload, dict)
                self.assertNotIn("private_key_hex", payload)
                return {"ok": True, "tx": unsigned_tx.to_dict()}

            if path == "/api/v1/tx/submit":
                self.assertIsInstance(payload, dict)
                self.assertNotIn("private_key_hex", payload)
                tx_raw = payload.get("tx")
                self.assertIsInstance(tx_raw, dict)
                self.assertTrue(str(tx_raw.get("txid", "")))
                inputs = tx_raw.get("inputs", [])
                self.assertTrue(all(str(item.get("signature", "")) for item in inputs))
                return {"ok": True, "txid": str(tx_raw.get("txid", ""))}

            raise AssertionError(f"Unexpected API path: {path}")

        with mock.patch("powx.p2p._request_json", side_effect=fake_request_json):
            result = api_create_contract_transaction(
                node_url="http://127.0.0.1:8844",
                private_key_hex=PRIV_A,
                contract_payload={"kind": "nft", "action": "mint", "token_id": "ART-1", "metadata_uri": "ipfs://art-1"},
                fee=1,
                broadcast_ttl=2,
                timeout=5.0,
            )

        self.assertTrue(result.get("ok"))
        self.assertEqual(seen_paths, ["/api/v1/tx/build-contract", "/api/v1/tx/submit"])

    def test_node_build_unsigned_and_submit_signed_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            node = self._make_node(td)
            self._init_genesis(node, supply=1000)

            built = node.build_unsigned_transaction(
                sender_pubkey=private_key_to_public_key(PRIV_A).hex(),
                to_address=ADDR_A,
                amount=10,
                fee=1,
            )
            self.assertTrue(built.get("ok"))
            self.assertTrue(bool(built.get("unsigned")))

            tx_raw = built.get("tx")
            self.assertIsInstance(tx_raw, dict)
            unsigned_tx = Transaction.from_dict(tx_raw)

            signed_tx = sign_transaction_offline(unsigned_tx, private_key_hex=PRIV_A)
            result = node.submit_signed_transaction(signed_tx, broadcast_ttl=0)

            self.assertTrue(result.get("ok"))
            with node.chain_lock:
                self.assertTrue(any(tx.txid == signed_tx.txid for tx in node.chain.mempool))

    def test_bootnodes_are_seeded_into_addrman(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bootnode = "http://127.0.0.1:19099"
            node = self._make_node(td, bootnodes=[bootnode])

            peer_url = normalize_peer(bootnode)
            self.assertIn(peer_url, node.get_peers())
            self.assertTrue(node.addrman_path.exists())

            raw = json.loads(node.addrman_path.read_text(encoding="utf-8"))
            self.assertIn(peer_url, raw.get("entries", {}))

    def test_sync_rejects_incompatible_chain_id_before_header_sync(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            node = self._make_node(td)
            peer_url = normalize_peer("http://127.0.0.1:19098")
            node._peers.add(peer_url)

            def fake_request_json(
                url: str,
                method: str = "GET",
                payload: dict | None = None,
                timeout: float | None = 4.0,
            ) -> dict:
                _ = method
                _ = payload
                _ = timeout
                path = urlparse(url).path
                if path == "/ping":
                    return {
                        "ok": True,
                        "chain_id": "kk91-p2p-test",
                        "height": 1,
                        "tip_hash": "00" * 32,
                    }
                if path == "/status":
                    return {
                        "ok": True,
                        "status": {
                            "chain_id": "kk91-other-net",
                            "target_schedule": "asert-v3",
                            "fixed_genesis_hash": "11" * 32,
                        },
                    }
                raise AssertionError(f"Unexpected path {path}")

            with mock.patch("powx.p2p._request_json", side_effect=fake_request_json):
                result = node.sync_from_peer(peer_url)

            self.assertFalse(result.get("updated"))
            self.assertIn("incompatible-peer:chain-id-mismatch", str(result.get("reason", "")))


if __name__ == "__main__":
    unittest.main()
