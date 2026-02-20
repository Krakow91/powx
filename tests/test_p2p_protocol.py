from __future__ import annotations

import socket
import tempfile
import unittest
from urllib.parse import urlparse
from unittest import mock

from powx.crypto import address_from_public_key, private_key_to_public_key
from powx.models import Block
from powx.p2p import P2PNode, normalize_peer


PRIV_A = "1" * 64
ADDR_A = address_from_public_key(private_key_to_public_key(PRIV_A).hex(), "KK91")


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
        node = P2PNode(
            data_dir=data_dir,
            host="127.0.0.1",
            port=port,
            sync_interval=0,
            **kwargs,
        )
        self.addCleanup(self._close_node, node)
        return node

    def _init_genesis(self, node: P2PNode, supply: int = 1000) -> Block:
        with mock.patch("powx.chain.secrets.randbits", return_value=777):
            with mock.patch.object(node.chain, "_now", return_value=1_700_000_000):
                block = node.chain.initialize(ADDR_A, genesis_supply=supply)
        with node.chain_lock:
            node._refresh_chain_from_disk_unlocked()
            node._prime_seen_from_chain_unlocked()
        return block

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


if __name__ == "__main__":
    unittest.main()
