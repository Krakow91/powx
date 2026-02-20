from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
import socket
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .chain import Chain, ValidationError
from .crypto import generate_private_key_hex, private_key_to_public_key, sign_digest, verify_signature
from .models import Block, Transaction


class NetworkError(Exception):
    pass


def normalize_peer(peer: str) -> str:
    raw = peer.strip()
    if not raw:
        raise ValueError("Peer URL must not be empty")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Peer URL scheme must be http or https")
    if not parsed.netloc:
        raise ValueError("Peer URL must include host:port")
    return f"{parsed.scheme}://{parsed.netloc}"


def _join_url(base_url: str, path: str) -> str:
    base = normalize_peer(base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"


def _request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float | None = 4.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=data, method=method.upper(), headers=headers)
    request_timeout = None if timeout is None else float(timeout)

    try:
        with urlopen(req, timeout=request_timeout) as response:
            raw = response.read()
            if not raw:
                return {}
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise NetworkError(f"Expected JSON object response from {url}")
            return decoded
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise NetworkError(f"HTTP {exc.code} calling {url}: {body}") from exc
    except URLError as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            raise NetworkError(f"Timeout calling {url}") from exc
        raise NetworkError(f"Network error calling {url}: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise NetworkError(f"Timeout calling {url}") from exc
    except json.JSONDecodeError as exc:
        raise NetworkError(f"Invalid JSON response from {url}") from exc


def get_node_status(node_url: str, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(_join_url(node_url, "/status"), method="GET", timeout=timeout)


def api_status(node_url: str, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(_join_url(node_url, "/api/v1/status"), method="GET", timeout=timeout)


def api_balance(node_url: str, address: str, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(_join_url(node_url, f"/api/v1/balance?address={address}"), method="GET", timeout=timeout)


def api_chain(node_url: str, limit: int = 20, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(_join_url(node_url, f"/api/v1/chain?limit={int(limit)}"), method="GET", timeout=timeout)


def api_mempool(node_url: str, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(_join_url(node_url, "/api/v1/mempool"), method="GET", timeout=timeout)


def api_history(node_url: str, address: str, limit: int = 120, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(
        _join_url(node_url, f"/api/v1/history?address={address}&limit={int(limit)}"),
        method="GET",
        timeout=timeout,
    )


def api_create_transaction(
    node_url: str,
    private_key_hex: str,
    to_address: str,
    amount: int,
    fee: int | None = None,
    broadcast_ttl: int = 2,
    timeout: float = 6.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "private_key_hex": private_key_hex,
        "to_address": to_address,
        "amount": int(amount),
        "broadcast_ttl": int(broadcast_ttl),
    }
    if fee is not None:
        payload["fee"] = int(fee)
    return _request_json(_join_url(node_url, "/api/v1/tx/create"), method="POST", payload=payload, timeout=timeout)


def api_mine(
    node_url: str,
    miner_address: str,
    blocks: int = 1,
    backend: str = "auto",
    broadcast_ttl: int = 2,
    timeout: float | None = None,
) -> dict[str, Any]:
    payload = {
        "miner_address": miner_address,
        "blocks": int(blocks),
        "backend": backend,
        "broadcast_ttl": int(broadcast_ttl),
    }
    return _request_json(_join_url(node_url, "/api/v1/mine"), method="POST", payload=payload, timeout=timeout)


def add_peer_to_node(node_url: str, peer: str, sync_now: bool = True, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(
        _join_url(node_url, "/peers/add"),
        method="POST",
        payload={"peer": peer, "sync": bool(sync_now)},
        timeout=timeout,
    )


def trigger_node_sync(node_url: str, peer: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if peer:
        payload["peer"] = peer
    return _request_json(_join_url(node_url, "/sync"), method="POST", payload=payload, timeout=timeout)


def submit_transaction(node_url: str, tx: Transaction, ttl: int = 2, timeout: float = 4.0) -> dict[str, Any]:
    return _request_json(
        _join_url(node_url, "/tx"),
        method="POST",
        payload={"tx": tx.to_dict(), "ttl": int(ttl)},
        timeout=timeout,
    )


def submit_block(node_url: str, block: Block, ttl: int = 2, timeout: float = 6.0) -> dict[str, Any]:
    return _request_json(
        _join_url(node_url, "/block"),
        method="POST",
        payload={"block": block.to_dict(), "ttl": int(ttl)},
        timeout=timeout,
    )


class P2PNode:
    def __init__(
        self,
        data_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 8844,
        advertise_host: str | None = None,
        peers: list[str] | None = None,
        sync_interval: float = 10.0,
        request_timeout: float = 4.0,
        seen_cache_size: int = 20_000,
        max_peer_count: int = 128,
        max_inbound_ttl: int = 6,
        max_request_body_bytes: int = 256_000,
        max_requests_per_minute: int = 360,
        sync_retry_cooldown: float = 2.0,
        peer_ban_threshold: int = 100,
        peer_ban_seconds: int = 900,
        peer_penalty_invalid_tx: int = 20,
        peer_penalty_invalid_block: int = 35,
        peer_penalty_bad_sync: int = 15,
        peer_reward_success: int = 3,
        peer_auth_max_skew_seconds: int = 120,
        peer_auth_replay_window_seconds: int = 180,
        max_inv_items: int = 256,
        max_getdata_items: int = 128,
        max_orphan_blocks: int = 2048,
        orphan_ttl_seconds: int = 1800,
        max_outbound_broadcast_peers: int = 16,
        max_outbound_sync_peers: int = 32,
        max_outbound_peers_per_bucket: int = 2,
    ) -> None:
        self.chain = Chain(data_dir)
        self.chain_lock = threading.RLock()

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = int(port)
        self.sync_interval = float(sync_interval)
        self.request_timeout = float(request_timeout)
        self.seen_cache_size = max(1000, int(seen_cache_size))
        self.max_peer_count = max(4, int(max_peer_count))
        self.max_inbound_ttl = max(0, int(max_inbound_ttl))
        self.max_request_body_bytes = max(4096, int(max_request_body_bytes))
        self.max_requests_per_minute = max(30, int(max_requests_per_minute))
        self.sync_retry_cooldown = max(0.0, float(sync_retry_cooldown))
        self.peer_ban_threshold = max(10, int(peer_ban_threshold))
        self.peer_ban_seconds = max(30, int(peer_ban_seconds))
        self.peer_penalty_invalid_tx = max(1, int(peer_penalty_invalid_tx))
        self.peer_penalty_invalid_block = max(1, int(peer_penalty_invalid_block))
        self.peer_penalty_bad_sync = max(1, int(peer_penalty_bad_sync))
        self.peer_reward_success = max(1, int(peer_reward_success))
        self.peer_auth_max_skew_seconds = max(10, int(peer_auth_max_skew_seconds))
        self.peer_auth_replay_window_seconds = max(
            self.peer_auth_max_skew_seconds,
            int(peer_auth_replay_window_seconds),
        )
        self.max_inv_items = max(16, int(max_inv_items))
        self.max_getdata_items = max(8, int(max_getdata_items))
        self.max_orphan_blocks = max(16, int(max_orphan_blocks))
        self.orphan_ttl_seconds = max(60, int(orphan_ttl_seconds))
        self.max_outbound_broadcast_peers = max(1, int(max_outbound_broadcast_peers))
        self.max_outbound_sync_peers = max(1, int(max_outbound_sync_peers))
        self.max_outbound_peers_per_bucket = max(1, int(max_outbound_peers_per_bucket))
        self.stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._rate_limit_lock = threading.Lock()
        self._sync_retry_lock = threading.Lock()
        self._peer_security_lock = threading.RLock()

        adv_host = advertise_host.strip() if advertise_host else host
        self.node_url = normalize_peer(f"http://{adv_host}:{self.port}")

        self.peers_path = self.data_dir / "peers.json"
        self.peer_security_path = self.data_dir / "peer_security.json"
        self.identity_path = self.data_dir / "node_identity.json"
        self._peers: set[str] = set()
        self._identity_private_key = ""
        self._identity_public_key = ""
        self._identity_id = ""

        self._seen_txids: set[str] = set()
        self._seen_tx_order: deque[str] = deque()
        self._seen_block_hashes: set[str] = set()
        self._seen_block_order: deque[str] = deque()
        self._orphan_blocks: dict[str, Block] = {}
        self._orphans_by_prev: dict[str, set[str]] = defaultdict(set)
        self._orphan_received_at: dict[str, float] = {}
        self._client_hits: dict[str, deque[float]] = defaultdict(deque)
        self._sync_retry_last: dict[tuple[str, str], float] = {}
        self._peer_scores: dict[str, int] = {}
        self._peer_banned_until: dict[str, float] = {}
        self._peer_last_reason: dict[str, str] = {}
        self._peer_pubkeys: dict[str, str] = {}
        self._peer_auth_seen: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        self._peer_auth_seen_set: dict[str, set[str]] = defaultdict(set)

        self._load_or_create_identity()
        self._load_peers()
        self._load_peer_security()
        for peer in peers or []:
            self.add_peer(peer, persist=False)
        self._save_peers()
        self._save_peer_security()

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self._prime_seen_from_chain_unlocked()

        self.server = ThreadingHTTPServer((self.host, self.port), self._build_handler())
        self.server.timeout = 1.0

    def _build_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> dict[str, Any]:
                length_text = self.headers.get("Content-Length", "0").strip()
                try:
                    length = int(length_text)
                except ValueError:
                    raise ValidationError("Invalid Content-Length header")
                if length <= 0:
                    return {}
                if length > node.max_request_body_bytes:
                    raise ValidationError(f"Request body too large (max {node.max_request_body_bytes} bytes)")
                body = self.rfile.read(length)
                try:
                    decoded = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise ValidationError("Request body is not valid JSON") from exc
                if not isinstance(decoded, dict):
                    raise ValidationError("Request body must be a JSON object")
                return decoded

            def _query(self) -> dict[str, list[str]]:
                parsed = urlparse(self.path)
                return parse_qs(parsed.query, keep_blank_values=False)

            @staticmethod
            def _query_first(query: dict[str, list[str]], key: str, default: str = "") -> str:
                values = query.get(key)
                if not values:
                    return default
                return str(values[0])

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    client_ip = str(self.client_address[0]) if self.client_address else "unknown"
                    if not node._allow_client_request(client_ip):
                        self._send_json(429, {"ok": False, "error": "Rate limit exceeded"})
                        return

                    if path == "/health":
                        self._send_json(200, {"ok": True})
                        return

                    if path in {"/status", "/api/v1/status"}:
                        self._send_json(200, node.status_payload())
                        return

                    if path == "/peers":
                        self._send_json(200, {"ok": True, "peers": node.get_peers()})
                        return

                    if path == "/snapshot/meta":
                        self._send_json(200, node.snapshot_meta_payload())
                        return

                    if path == "/headers/meta":
                        self._send_json(200, node.headers_meta_payload())
                        return

                    if path == "/headers":
                        query = self._query()
                        start_text = self._query_first(query, "start_height", "0").strip()
                        limit_text = self._query_first(query, "limit", "500").strip()
                        try:
                            start_height = int(start_text)
                            limit = int(limit_text)
                        except ValueError as exc:
                            raise ValidationError("Query parameters 'start_height' and 'limit' must be integers") from exc
                        self._send_json(200, node.headers_payload(start_height=start_height, limit=limit))
                        return

                    if path == "/blocks/range":
                        query = self._query()
                        start_text = self._query_first(query, "start_height", "0").strip()
                        limit_text = self._query_first(query, "limit", "200").strip()
                        try:
                            start_height = int(start_text)
                            limit = int(limit_text)
                        except ValueError as exc:
                            raise ValidationError("Query parameters 'start_height' and 'limit' must be integers") from exc
                        self._send_json(200, node.blocks_range_payload(start_height=start_height, limit=limit))
                        return

                    if path == "/block/by-hash":
                        query = self._query()
                        block_hash = self._query_first(query, "hash").strip()
                        if not block_hash:
                            raise ValidationError("Missing query parameter: hash")
                        self._send_json(200, node.block_by_hash_payload(block_hash))
                        return

                    if path == "/snapshot":
                        self._send_json(200, node.snapshot_payload())
                        return

                    if path == "/api/v1/balance":
                        query = self._query()
                        address = self._query_first(query, "address").strip()
                        if not address:
                            raise ValidationError("Missing query parameter: address")
                        self._send_json(200, node.balance_payload(address))
                        return

                    if path == "/api/v1/chain":
                        query = self._query()
                        limit_text = self._query_first(query, "limit", "20").strip()
                        try:
                            limit = int(limit_text)
                        except ValueError as exc:
                            raise ValidationError("Query parameter 'limit' must be an integer") from exc
                        self._send_json(200, node.chain_payload(limit=limit))
                        return

                    if path == "/api/v1/mempool":
                        self._send_json(200, node.mempool_payload())
                        return

                    if path == "/api/v1/history":
                        query = self._query()
                        address = self._query_first(query, "address").strip()
                        if not address:
                            raise ValidationError("Missing query parameter: address")
                        limit_text = self._query_first(query, "limit", "120").strip()
                        try:
                            limit = int(limit_text)
                        except ValueError as exc:
                            raise ValidationError("Query parameter 'limit' must be an integer") from exc
                        self._send_json(200, node.history_payload(address=address, limit=limit))
                        return

                    self._send_json(404, {"ok": False, "error": "Not found"})
                except ValidationError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:  # pragma: no cover - defensive server branch
                    self._send_json(500, {"ok": False, "error": f"Server error: {exc}"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    client_ip = str(self.client_address[0]) if self.client_address else "unknown"
                    if not node._allow_client_request(client_ip):
                        self._send_json(429, {"ok": False, "error": "Rate limit exceeded"})
                        return

                    payload = self._read_json()

                    if path == "/peers/add":
                        peer = str(payload.get("peer", "")).strip()
                        if not peer:
                            raise ValidationError("Missing peer")
                        added, reason, peer_url = node.add_peer_with_reason(peer, persist=True)
                        synced = False
                        if added and bool(payload.get("sync", True)):
                            synced = bool(node.sync_from_peer(peer).get("updated"))
                        self._send_json(
                            200,
                            {"ok": True, "added": added, "reason": reason, "peer": peer_url, "synced": synced},
                        )
                        return

                    if path == "/tx":
                        tx_obj = payload.get("tx", payload)
                        if not isinstance(tx_obj, dict):
                            raise ValidationError("Missing transaction object")
                        ttl = node._coerce_ttl(payload.get("ttl", 2), field_name="ttl")
                        sender = str(payload.get("from", "")).strip()
                        auth = payload.get("auth", None)
                        tx = Transaction.from_dict(tx_obj)
                        result = node.accept_transaction(tx, sender=sender, ttl=ttl, auth=auth)
                        status = 200 if result.get("accepted") else 409
                        self._send_json(status, result)
                        return

                    if path == "/block":
                        block_obj = payload.get("block", payload)
                        if not isinstance(block_obj, dict):
                            raise ValidationError("Missing block object")
                        ttl = node._coerce_ttl(payload.get("ttl", 2), field_name="ttl")
                        sender = str(payload.get("from", "")).strip()
                        auth = payload.get("auth", None)
                        block = Block.from_dict(block_obj)
                        result = node.accept_block(block, sender=sender, ttl=ttl, auth=auth)
                        status = 200 if result.get("accepted") else 409
                        self._send_json(status, result)
                        return

                    if path == "/inv":
                        items = payload.get("items", [])
                        ttl = node._coerce_ttl(payload.get("ttl", 2), field_name="ttl")
                        sender = str(payload.get("from", "")).strip()
                        result = node.accept_inventory(items, sender=sender, ttl=ttl)
                        status = 200 if result.get("ok") else 400
                        self._send_json(status, result)
                        return

                    if path == "/getdata":
                        items = payload.get("items", [])
                        ttl = node._coerce_ttl(payload.get("ttl", 2), field_name="ttl")
                        self._send_json(200, node.getdata_payload(items, ttl=ttl))
                        return

                    if path == "/sync":
                        peer = str(payload.get("peer", "")).strip()
                        if peer:
                            result = node.sync_from_peer(peer)
                        else:
                            result = node.sync_from_best_peer()
                        self._send_json(200, {"ok": True, **result})
                        return

                    if path == "/api/v1/tx/create":
                        private_key_hex = str(payload.get("private_key_hex", "")).strip()
                        to_address = str(payload.get("to_address", "")).strip()
                        amount_raw = payload.get("amount")
                        fee_raw = payload.get("fee", None)
                        ttl_raw = payload.get("broadcast_ttl", 2)

                        if not private_key_hex:
                            raise ValidationError("Missing field: private_key_hex")
                        if not to_address:
                            raise ValidationError("Missing field: to_address")
                        if amount_raw is None:
                            raise ValidationError("Missing field: amount")

                        try:
                            amount = int(amount_raw)
                            fee = int(fee_raw) if fee_raw is not None else None
                        except (TypeError, ValueError) as exc:
                            raise ValidationError("amount and fee must be integers") from exc

                        ttl = node._coerce_ttl(ttl_raw, field_name="broadcast_ttl")

                        result = node.create_and_broadcast_transaction(
                            private_key_hex=private_key_hex,
                            to_address=to_address,
                            amount=amount,
                            fee=fee,
                            broadcast_ttl=ttl,
                        )
                        status = 200 if result.get("ok") else 400
                        self._send_json(status, result)
                        return

                    if path == "/api/v1/mine":
                        miner_address = str(payload.get("miner_address", "")).strip()
                        blocks_raw = payload.get("blocks", 1)
                        backend = str(payload.get("backend", "auto")).strip().lower()
                        ttl_raw = payload.get("broadcast_ttl", 2)

                        if not miner_address:
                            raise ValidationError("Missing field: miner_address")
                        try:
                            blocks = int(blocks_raw)
                        except (TypeError, ValueError) as exc:
                            raise ValidationError("blocks must be an integer") from exc

                        ttl = node._coerce_ttl(ttl_raw, field_name="broadcast_ttl")

                        result = node.mine_blocks(
                            miner_address=miner_address,
                            blocks=blocks,
                            backend=backend,
                            broadcast_ttl=ttl,
                        )
                        status = 200 if result.get("ok") else 400
                        self._send_json(status, result)
                        return

                    if path == "/api/v1/mempool/prune":
                        result = node.prune_mempool()
                        self._send_json(200, result)
                        return

                    self._send_json(404, {"ok": False, "error": "Not found"})
                except ValidationError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:  # pragma: no cover - defensive server branch
                    self._send_json(500, {"ok": False, "error": f"Server error: {exc}"})

            def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - avoid noisy stdio logs
                return

        return Handler

    def _refresh_chain_from_disk_unlocked(self) -> None:
        if self.chain.exists():
            self.chain.load()

    def _remember(self, key: str, seen: set[str], order: deque[str]) -> bool:
        if key in seen:
            return False
        seen.add(key)
        order.append(key)
        if len(order) > self.seen_cache_size:
            old = order.popleft()
            seen.discard(old)
        return True

    def _remember_txid(self, txid: str) -> None:
        self._remember(txid, self._seen_txids, self._seen_tx_order)

    def _remember_block_hash(self, block_hash: str) -> None:
        self._remember(block_hash, self._seen_block_hashes, self._seen_block_order)

    def _coerce_ttl(self, raw_ttl: Any, field_name: str = "ttl") -> int:
        try:
            ttl = int(raw_ttl)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field_name} must be an integer") from exc
        if ttl <= 0:
            return 0
        return min(ttl, self.max_inbound_ttl)

    def _allow_client_request(self, client_key: str) -> bool:
        key = client_key.strip() or "unknown"
        now = time.monotonic()
        cutoff = now - 60.0

        with self._rate_limit_lock:
            bucket = self._client_hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests_per_minute:
                return False
            bucket.append(now)

            if len(self._client_hits) > 4096:
                stale = [k for k, entries in self._client_hits.items() if not entries or entries[-1] < cutoff]
                for stale_key in stale[:512]:
                    self._client_hits.pop(stale_key, None)

        return True

    def _is_known_peer(self, peer_url: str) -> bool:
        return peer_url in self._peers

    def _allow_sync_retry(self, sender_url: str, kind: str) -> bool:
        if not sender_url or not self._is_known_peer(sender_url):
            return False
        if self._is_peer_banned(sender_url):
            return False
        if self.sync_retry_cooldown <= 0:
            return True

        now = time.monotonic()
        key = (kind, sender_url)
        with self._sync_retry_lock:
            last = self._sync_retry_last.get(key, 0.0)
            if now - last < self.sync_retry_cooldown:
                return False
            self._sync_retry_last[key] = now

            if len(self._sync_retry_last) > 4096:
                stale_cutoff = now - max(self.sync_retry_cooldown * 3.0, 30.0)
                stale_keys = [item for item, ts in self._sync_retry_last.items() if ts < stale_cutoff]
                for stale_key in stale_keys[:512]:
                    self._sync_retry_last.pop(stale_key, None)

        return True

    @staticmethod
    def _is_hex_string(value: str, expected_len: int | None = None) -> bool:
        text = value.strip().lower()
        if expected_len is not None and len(text) != expected_len:
            return False
        if not text:
            return False
        try:
            int(text, 16)
        except ValueError:
            return False
        return True

    @staticmethod
    def _pubkey_fingerprint(pubkey_hex: str) -> str:
        if not P2PNode._is_hex_string(pubkey_hex):
            return ""
        digest = hashlib.sha256(bytes.fromhex(pubkey_hex)).hexdigest()
        return digest[:16]

    def _is_peer_banned(self, peer_url: str) -> bool:
        if not peer_url:
            return False
        now = time.time()
        changed = False
        is_banned = False
        with self._peer_security_lock:
            banned_until = float(self._peer_banned_until.get(peer_url, 0.0))
            if banned_until > now:
                is_banned = True
            else:
                if banned_until > 0:
                    self._peer_banned_until.pop(peer_url, None)
                    score = int(self._peer_scores.get(peer_url, 0))
                    if score > 0:
                        self._peer_scores[peer_url] = max(0, score // 2)
                    self._peer_last_reason[peer_url] = "ban-expired"
                    changed = True
        if changed:
            self._save_peer_security()
        return is_banned

    @staticmethod
    def _peer_diversity_bucket(peer_url: str) -> str:
        parsed = urlparse(peer_url)
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return "unknown"

        try:
            ip_obj = ipaddress.ip_address(host)
        except ValueError:
            labels = [label for label in host.split(".") if label]
            if len(labels) >= 2:
                return f"dns:{labels[-2]}.{labels[-1]}"
            return f"dns:{host}"

        if ip_obj.version == 4:
            octets = host.split(".")
            if len(octets) >= 2:
                return f"ipv4:{octets[0]}.{octets[1]}"
            return f"ipv4:{host}"

        hextets = ip_obj.exploded.split(":")
        return f"ipv6:{':'.join(hextets[:3])}"

    def _select_diverse_peers(self, peers: list[str], limit: int) -> list[str]:
        if limit <= 0 or not peers:
            return []

        grouped: dict[str, list[str]] = defaultdict(list)
        for peer in peers:
            grouped[self._peer_diversity_bucket(peer)].append(peer)

        with self._peer_security_lock:
            for bucket in grouped.values():
                bucket.sort(key=lambda value: (int(self._peer_scores.get(value, 0)), value))

        buckets = sorted(grouped.keys())
        selected: list[str] = []
        picked_per_bucket: dict[str, int] = defaultdict(int)

        while len(selected) < limit:
            progressed = False
            for bucket in buckets:
                picked = picked_per_bucket[bucket]
                if picked >= self.max_outbound_peers_per_bucket:
                    continue
                candidates = grouped[bucket]
                if picked >= len(candidates):
                    continue
                selected.append(candidates[picked])
                picked_per_bucket[bucket] += 1
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break

        return selected

    def _get_relay_peers(self, limit: int | None = None) -> list[str]:
        peers: list[str] = []
        for peer in self.get_peers():
            if not self._is_peer_banned(peer):
                peers.append(peer)
        if not peers:
            return []
        wanted = self.max_outbound_broadcast_peers if limit is None else max(1, int(limit))
        wanted = min(wanted, len(peers))
        return self._select_diverse_peers(peers, limit=wanted)

    def _penalize_peer(self, peer_url: str, points: int, reason: str) -> None:
        if not peer_url or not self._is_known_peer(peer_url):
            return
        penalty = max(1, int(points))
        now = time.time()
        changed = False

        with self._peer_security_lock:
            score = int(self._peer_scores.get(peer_url, 0)) + penalty
            self._peer_scores[peer_url] = score
            self._peer_last_reason[peer_url] = (reason.strip() or "penalty")[:160]
            if score >= self.peer_ban_threshold:
                self._peer_banned_until[peer_url] = now + float(self.peer_ban_seconds)
            changed = True
        if changed:
            self._save_peer_security()

    def _reward_peer(self, peer_url: str, points: int | None = None) -> None:
        if not peer_url or not self._is_known_peer(peer_url):
            return
        reward = self.peer_reward_success if points is None else max(1, int(points))
        changed = False
        with self._peer_security_lock:
            score = int(self._peer_scores.get(peer_url, 0))
            if score <= 0:
                return
            self._peer_scores[peer_url] = max(0, score - reward)
            self._peer_last_reason[peer_url] = "ok"
            changed = True
        if changed:
            self._save_peer_security()

    def _peer_security_payload(self) -> dict[str, Any]:
        now = time.time()
        rows: list[dict[str, Any]] = []

        with self._peer_security_lock:
            peers = sorted(self._peers)
            for peer in peers:
                score = int(self._peer_scores.get(peer, 0))
                banned_until = float(self._peer_banned_until.get(peer, 0.0))
                remaining = int(max(0.0, banned_until - now))
                pubkey = str(self._peer_pubkeys.get(peer, ""))
                rows.append(
                    {
                        "peer": peer,
                        "score": score,
                        "banned": remaining > 0,
                        "ban_remaining_seconds": remaining,
                        "last_reason": self._peer_last_reason.get(peer, ""),
                        "pubkey_known": bool(pubkey),
                        "pubkey_fingerprint": self._pubkey_fingerprint(pubkey),
                    }
                )

        banned_count = sum(1 for row in rows if bool(row.get("banned")))
        return {
            "ban_threshold": self.peer_ban_threshold,
            "ban_seconds": self.peer_ban_seconds,
            "penalties": {
                "invalid_tx": self.peer_penalty_invalid_tx,
                "invalid_block": self.peer_penalty_invalid_block,
                "bad_sync": self.peer_penalty_bad_sync,
            },
            "reward_success": self.peer_reward_success,
            "auth_max_skew_seconds": self.peer_auth_max_skew_seconds,
            "auth_replay_window_seconds": self.peer_auth_replay_window_seconds,
            "banned_peers": banned_count,
            "rows": rows,
        }

    def _save_identity_file(self) -> None:
        payload = {
            "private_key": self._identity_private_key,
            "public_key": self._identity_public_key,
            "node_id": self._identity_id,
        }
        temp_path = self.identity_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.identity_path)

    def _load_or_create_identity(self) -> None:
        private_key = ""
        public_key = ""

        if self.identity_path.exists():
            try:
                with self.identity_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if isinstance(raw, dict):
                    private_candidate = str(raw.get("private_key", "")).strip().lower()
                    public_candidate = str(raw.get("public_key", "")).strip().lower()
                    if self._is_hex_string(private_candidate, 64):
                        derived = private_key_to_public_key(private_candidate).hex().lower()
                        if not public_candidate or public_candidate == derived:
                            private_key = private_candidate
                            public_key = derived
            except Exception:
                private_key = ""
                public_key = ""

        if not private_key or not public_key:
            private_key = generate_private_key_hex().lower()
            public_key = private_key_to_public_key(private_key).hex().lower()

        self._identity_private_key = private_key
        self._identity_public_key = public_key
        self._identity_id = self._pubkey_fingerprint(public_key)
        self._save_identity_file()

    def _save_peer_security(self) -> None:
        with self._peer_security_lock:
            payload = {
                "version": 1,
                "updated_at": int(time.time()),
                "scores": {peer: int(score) for peer, score in self._peer_scores.items()},
                "banned_until": {peer: float(ts) for peer, ts in self._peer_banned_until.items()},
                "last_reason": {peer: str(reason) for peer, reason in self._peer_last_reason.items()},
                "peer_pubkeys": {peer: str(pubkey) for peer, pubkey in self._peer_pubkeys.items()},
            }
        temp_path = self.peer_security_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.peer_security_path)

    def _load_peer_security(self) -> None:
        if not self.peer_security_path.exists():
            return
        try:
            with self.peer_security_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                return
        except Exception:
            return

        now = time.time()
        loaded_scores: dict[str, int] = {}
        loaded_banned_until: dict[str, float] = {}
        loaded_reasons: dict[str, str] = {}
        loaded_pubkeys: dict[str, str] = {}

        scores_raw = raw.get("scores", {})
        if isinstance(scores_raw, dict):
            for key, value in scores_raw.items():
                try:
                    peer_url = normalize_peer(str(key))
                    score = max(0, int(value))
                except Exception:
                    continue
                loaded_scores[peer_url] = score

        banned_raw = raw.get("banned_until", {})
        if isinstance(banned_raw, dict):
            for key, value in banned_raw.items():
                try:
                    peer_url = normalize_peer(str(key))
                    banned_until = float(value)
                except Exception:
                    continue
                if banned_until > now:
                    loaded_banned_until[peer_url] = banned_until

        reason_raw = raw.get("last_reason", {})
        if isinstance(reason_raw, dict):
            for key, value in reason_raw.items():
                try:
                    peer_url = normalize_peer(str(key))
                except Exception:
                    continue
                loaded_reasons[peer_url] = str(value)[:160]

        pubkeys_raw = raw.get("peer_pubkeys", {})
        if isinstance(pubkeys_raw, dict):
            for key, value in pubkeys_raw.items():
                try:
                    peer_url = normalize_peer(str(key))
                except Exception:
                    continue
                pubkey = str(value).strip().lower()
                if self._is_hex_string(pubkey, 66):
                    loaded_pubkeys[peer_url] = pubkey

        with self._peer_security_lock:
            self._peer_scores.update(loaded_scores)
            self._peer_banned_until.update(loaded_banned_until)
            self._peer_last_reason.update(loaded_reasons)
            self._peer_pubkeys.update(loaded_pubkeys)

    def _auth_message_digest(
        self,
        sender_url: str,
        purpose: str,
        object_id: str,
        ttl: int,
        ts: int,
        nonce: str,
    ) -> bytes:
        canonical = {
            "v": 1,
            "chain_id": str(self.chain.config.chain_id),
            "from": sender_url,
            "purpose": purpose,
            "object_id": object_id,
            "ttl": int(ttl),
            "ts": int(ts),
            "nonce": nonce,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).digest()

    def _build_sender_auth(self, purpose: str, object_id: str, ttl: int) -> dict[str, Any]:
        now_ts = int(time.time())
        nonce = secrets.token_hex(16)
        digest = self._auth_message_digest(
            sender_url=self.node_url,
            purpose=purpose,
            object_id=object_id,
            ttl=ttl,
            ts=now_ts,
            nonce=nonce,
        )
        signature = sign_digest(self._identity_private_key, digest)
        return {
            "v": 1,
            "pubkey": self._identity_public_key,
            "ts": now_ts,
            "nonce": nonce,
            "sig": signature,
        }

    def _fetch_peer_public_key(self, peer_url: str) -> str:
        try:
            status = _request_json(_join_url(peer_url, "/status"), method="GET", timeout=max(self.request_timeout, 3.0))
        except Exception:
            return ""
        identity = status.get("node_identity")
        if not isinstance(identity, dict):
            return ""
        pubkey = str(identity.get("public_key", "")).strip().lower()
        if not self._is_hex_string(pubkey, 66):
            return ""
        return pubkey

    def _remember_auth_nonce(self, peer_url: str, ts: int, replay_key: str) -> bool:
        now = int(time.time())
        min_ts = now - self.peer_auth_replay_window_seconds
        with self._peer_security_lock:
            queue = self._peer_auth_seen[peer_url]
            seen = self._peer_auth_seen_set[peer_url]
            while queue and queue[0][0] < min_ts:
                _, old_key = queue.popleft()
                seen.discard(old_key)
            if replay_key in seen:
                return False
            queue.append((ts, replay_key))
            seen.add(replay_key)
            if len(queue) > 4096:
                _old_ts, old_key = queue.popleft()
                seen.discard(old_key)
        return True

    def _validate_sender_auth(
        self,
        sender_url: str,
        auth: Any,
        purpose: str,
        object_id: str,
        ttl: int,
    ) -> tuple[bool, str]:
        if not sender_url:
            return True, "no-sender"
        if self._is_peer_banned(sender_url):
            return False, "sender-banned"
        if not self._is_known_peer(sender_url):
            return False, "sender-not-known"
        if not isinstance(auth, dict):
            return False, "missing-sender-auth"

        pubkey = str(auth.get("pubkey", "")).strip().lower()
        signature = str(auth.get("sig", "")).strip().lower()
        nonce = str(auth.get("nonce", "")).strip().lower()
        try:
            ts = int(auth.get("ts"))
        except Exception:
            return False, "invalid-sender-auth-ts"

        if not self._is_hex_string(pubkey, 66):
            return False, "invalid-sender-auth-pubkey"
        if not self._is_hex_string(signature, 128):
            return False, "invalid-sender-auth-signature"
        if not self._is_hex_string(nonce) or len(nonce) < 8 or len(nonce) > 64:
            return False, "invalid-sender-auth-nonce"
        if not self._is_hex_string(object_id, 64):
            return False, "invalid-auth-object-id"

        now = int(time.time())
        if abs(now - ts) > self.peer_auth_max_skew_seconds:
            return False, "sender-auth-stale"

        expected_pubkey = ""
        with self._peer_security_lock:
            expected_pubkey = str(self._peer_pubkeys.get(sender_url, "")).strip().lower()
        if not expected_pubkey:
            fetched_pubkey = self._fetch_peer_public_key(sender_url)
            if not fetched_pubkey:
                return False, "sender-identity-unknown"
            expected_pubkey = fetched_pubkey
            with self._peer_security_lock:
                self._peer_pubkeys[sender_url] = expected_pubkey
                self._peer_last_reason[sender_url] = "identity-learned"
            self._save_peer_security()

        if expected_pubkey != pubkey:
            return False, "sender-pubkey-mismatch"

        digest = self._auth_message_digest(
            sender_url=sender_url,
            purpose=purpose,
            object_id=object_id,
            ttl=ttl,
            ts=ts,
            nonce=nonce,
        )
        if not verify_signature(pubkey, digest, signature):
            return False, "sender-auth-invalid-signature"

        replay_key = f"{purpose}:{object_id}:{ts}:{nonce}"
        if not self._remember_auth_nonce(sender_url, ts, replay_key):
            return False, "sender-auth-replay"

        return True, "sender-auth-ok"

    def _prime_seen_from_chain_unlocked(self) -> None:
        for block in self.chain.chain:
            self._remember_block_hash(block.block_hash)
            self._drop_orphan_unlocked(block.block_hash)
            for tx in block.transactions:
                self._remember_txid(tx.txid)
        for tx in self.chain.mempool:
            self._remember_txid(tx.txid)
        self._prune_orphans_unlocked()

    def _load_peers(self) -> None:
        if not self.peers_path.exists():
            return
        try:
            with self.peers_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                return
            for item in data:
                if not isinstance(item, str):
                    continue
                try:
                    peer_url = normalize_peer(item)
                except ValueError:
                    continue
                if peer_url != self.node_url:
                    self._peers.add(peer_url)
        except Exception:
            return

    def _save_peers(self) -> None:
        self.peers_path.parent.mkdir(parents=True, exist_ok=True)
        with self.peers_path.open("w", encoding="utf-8") as handle:
            json.dump(sorted(self._peers), handle, indent=2)

    def get_peers(self) -> list[str]:
        return sorted(self._peers)

    def add_peer_with_reason(self, peer: str, persist: bool = True) -> tuple[bool, str, str]:
        try:
            peer_url = normalize_peer(peer)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if peer_url == self.node_url:
            return False, "self-peer-not-allowed", peer_url
        if peer_url in self._peers:
            return False, "known-peer", peer_url
        if len(self._peers) >= self.max_peer_count:
            return False, "max-peers-reached", peer_url

        self._peers.add(peer_url)
        with self._peer_security_lock:
            self._peer_scores.setdefault(peer_url, 0)
            self._peer_last_reason.setdefault(peer_url, "added")
        learned_pubkey = self._fetch_peer_public_key(peer_url)
        if learned_pubkey:
            with self._peer_security_lock:
                self._peer_pubkeys[peer_url] = learned_pubkey
                self._peer_last_reason[peer_url] = "identity-verified"
        if persist:
            self._save_peers()
            self._save_peer_security()
        return True, "added-verified" if learned_pubkey else "added", peer_url

    def add_peer(self, peer: str, persist: bool = True) -> bool:
        added, _reason, _peer_url = self.add_peer_with_reason(peer, persist=persist)
        return added

    def status_payload(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self._prune_orphans_unlocked()
            status = self.chain.status()
            tip = self.chain.tip
            work = int(tip.chain_work) if tip else 0
            orphan_count = len(self._orphan_blocks)
            orphan_parent_count = len(self._orphans_by_prev)
        return {
            "ok": True,
            "node": self.node_url,
            "peers": self.get_peers(),
            "peer_count": len(self._peers),
            "node_identity": {
                "node_id": self._identity_id,
                "public_key": self._identity_public_key,
            },
            "limits": {
                "max_peers": self.max_peer_count,
                "max_inbound_ttl": self.max_inbound_ttl,
                "max_request_body_bytes": self.max_request_body_bytes,
                "max_requests_per_minute": self.max_requests_per_minute,
                "sync_retry_cooldown": self.sync_retry_cooldown,
                "peer_ban_threshold": self.peer_ban_threshold,
                "peer_ban_seconds": self.peer_ban_seconds,
                "peer_auth_max_skew_seconds": self.peer_auth_max_skew_seconds,
                "peer_auth_replay_window_seconds": self.peer_auth_replay_window_seconds,
                "max_inv_items": self.max_inv_items,
                "max_getdata_items": self.max_getdata_items,
                "max_orphan_blocks": self.max_orphan_blocks,
                "orphan_ttl_seconds": self.orphan_ttl_seconds,
                "max_outbound_broadcast_peers": self.max_outbound_broadcast_peers,
                "max_outbound_sync_peers": self.max_outbound_sync_peers,
                "max_outbound_peers_per_bucket": self.max_outbound_peers_per_bucket,
            },
            "peer_security": self._peer_security_payload(),
            "status": status,
            "chain_work": work,
            "orphans": {
                "count": orphan_count,
                "parent_keys": orphan_parent_count,
            },
            "protocols": {
                "inventory_relay": "inv/getdata",
            },
        }

    def balance_payload(self, address: str) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self.chain._validate_address_format(address)
            balance = self.chain.balance_of(address)
            status = self.chain.status()
        return {"ok": True, "address": address, "balance": balance, "status": status}

    def chain_payload(self, limit: int = 20) -> dict[str, Any]:
        if limit <= 0:
            raise ValidationError("limit must be positive")
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            start = max(0, len(self.chain.chain) - limit)
            rows = [
                {
                    "height": block.index,
                    "hash": block.block_hash,
                    "prev_hash": block.prev_hash,
                    "tx_count": len(block.transactions),
                    "target": block.target,
                    "timestamp": block.timestamp,
                }
                for block in self.chain.chain[start:]
            ]
        return {"ok": True, "rows": rows, "limit": limit}

    def mempool_payload(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            rows = [
                {
                    "txid": tx.txid,
                    "timestamp": tx.timestamp,
                    "inputs": len(tx.inputs),
                    "outputs": len(tx.outputs),
                }
                for tx in self.chain.mempool
            ]
        return {"ok": True, "rows": rows, "size": len(rows)}

    @staticmethod
    def _history_row_for_address(
        tx: Transaction,
        address: str,
        output_map: dict[str, Any],
        pending: bool,
        symbol: str,
    ) -> dict[str, Any] | None:
        ts = datetime.fromtimestamp(tx.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        status = "pending" if pending else "confirmed"

        if tx.is_coinbase():
            mined = sum(out.amount for out in tx.outputs if out.address == address)
            if mined <= 0:
                return None
            return {
                "time": ts,
                "type": "Mined",
                "amount": f"+{mined} {symbol}",
                "counterparty": "-",
                "status": status,
                "txid": tx.txid,
            }

        total_in_from_me = 0
        sender_candidates: list[str] = []
        for tx_input in tx.inputs:
            prev = output_map.get(f"{tx_input.txid}:{tx_input.index}")
            if prev is None:
                continue
            prev_addr = getattr(prev, "address", "")
            prev_amount = int(getattr(prev, "amount", 0))
            sender_candidates.append(prev_addr)
            if prev_addr == address:
                total_in_from_me += prev_amount

        received_to_me = sum(out.amount for out in tx.outputs if out.address == address)

        if total_in_from_me > 0:
            spent = max(0, total_in_from_me - received_to_me)
            counterparty = next((out.address for out in tx.outputs if out.address != address), "-")
            return {
                "time": ts,
                "type": "Sent",
                "amount": f"-{spent} {symbol}",
                "counterparty": counterparty,
                "status": status,
                "txid": tx.txid,
            }

        if received_to_me > 0:
            sender = sender_candidates[0] if sender_candidates else "-"
            return {
                "time": ts,
                "type": "Received",
                "amount": f"+{received_to_me} {symbol}",
                "counterparty": sender,
                "status": status,
                "txid": tx.txid,
            }

        return None

    def history_payload(self, address: str, limit: int = 120) -> dict[str, Any]:
        if limit <= 0:
            raise ValidationError("limit must be positive")

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self.chain._validate_address_format(address)

            output_map: dict[str, Any] = {}
            for block in self.chain.chain:
                for tx in block.transactions:
                    for index, output in enumerate(tx.outputs):
                        output_map[f"{tx.txid}:{index}"] = output

            rows: list[dict[str, Any]] = []
            for tx in reversed(self.chain.mempool):
                row = self._history_row_for_address(tx, address, output_map, pending=True, symbol=self.chain.config.symbol)
                if row is not None:
                    rows.append(row)

            for block in reversed(self.chain.chain):
                for tx in reversed(block.transactions):
                    row = self._history_row_for_address(
                        tx,
                        address,
                        output_map,
                        pending=False,
                        symbol=self.chain.config.symbol,
                    )
                    if row is not None:
                        rows.append(row)
                    if len(rows) >= limit:
                        return {"ok": True, "address": address, "rows": rows[:limit], "limit": limit}

        return {"ok": True, "address": address, "rows": rows[:limit], "limit": limit}

    def snapshot_meta_payload(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            tip = self.chain.tip
            return {
                "ok": True,
                "height": self.chain.height,
                "tip_hash": tip.block_hash if tip else None,
                "chain_work": int(tip.chain_work) if tip else 0,
            }

    def headers_meta_payload(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            tip = self.chain.tip
            status = self.chain.status()
            return {
                "ok": True,
                "height": self.chain.height,
                "tip_hash": tip.block_hash if tip else None,
                "chain_work": int(tip.chain_work) if tip else 0,
                "protocol_version": int(status.get("protocol_version", 1)),
                "next_protocol_version": int(status.get("next_protocol_version", 1)),
            }

    def headers_payload(self, start_height: int = 0, limit: int = 500) -> dict[str, Any]:
        if start_height < 0:
            raise ValidationError("start_height must be >= 0")
        if limit <= 0:
            raise ValidationError("limit must be positive")
        limit = min(limit, 2000)

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            total = len(self.chain.chain)
            if start_height >= total:
                rows: list[dict[str, Any]] = []
            else:
                selected = self.chain.chain[start_height : start_height + limit]
                rows = [
                    {
                        "height": block.index,
                        "hash": block.block_hash,
                        "prev_hash": block.prev_hash,
                        "timestamp": block.timestamp,
                        "target": block.target,
                        "chain_work": int(block.chain_work),
                        "tx_count": len(block.transactions),
                    }
                    for block in selected
                ]

        return {"ok": True, "start_height": start_height, "limit": limit, "rows": rows}

    def blocks_range_payload(self, start_height: int = 0, limit: int = 200) -> dict[str, Any]:
        if start_height < 0:
            raise ValidationError("start_height must be >= 0")
        if limit <= 0:
            raise ValidationError("limit must be positive")
        limit = min(limit, 500)

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            total = len(self.chain.chain)
            if start_height >= total:
                rows: list[dict[str, Any]] = []
            else:
                selected = self.chain.chain[start_height : start_height + limit]
                rows = [block.to_dict() for block in selected]

        return {"ok": True, "start_height": start_height, "limit": limit, "rows": rows}

    def block_by_hash_payload(self, block_hash: str) -> dict[str, Any]:
        block_hash = block_hash.strip()
        if len(block_hash) != 64:
            raise ValidationError("hash must be a 64-char hex string")
        try:
            int(block_hash, 16)
        except ValueError as exc:
            raise ValidationError("hash must be hex") from exc

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            for block in self.chain.chain:
                if block.block_hash == block_hash:
                    return {"ok": True, "block": block.to_dict()}

        raise ValidationError("Block not found")

    def snapshot_payload(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            snapshot = self.chain.snapshot()
        return {"ok": True, **snapshot}

    @staticmethod
    def _normalize_sender(sender: str) -> str:
        sender_text = sender.strip()
        if not sender_text:
            return ""
        try:
            return normalize_peer(sender_text)
        except ValueError:
            return ""

    def _normalize_inventory_items(self, items_raw: Any, limit: int) -> list[dict[str, str]]:
        if not isinstance(items_raw, list):
            raise ValidationError("items must be a list")

        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        max_items = max(1, int(limit))
        for raw in items_raw:
            if len(normalized) >= max_items:
                break
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("type", "")).strip().lower()
            object_id = str(raw.get("id", "")).strip().lower()
            if kind not in {"tx", "block"}:
                continue
            if not self._is_hex_string(object_id, 64):
                continue
            key = (kind, object_id)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"type": kind, "id": object_id})
        return normalized

    def _chain_has_block_hash_unlocked(self, block_hash: str) -> bool:
        if not block_hash:
            return False
        if block_hash in self._orphan_blocks:
            return True
        return any(block.block_hash == block_hash for block in self.chain.chain)

    def _has_txid_unlocked(self, txid: str) -> bool:
        if txid in self._seen_txids:
            return True
        return any(tx.txid == txid for tx in self.chain.mempool)

    def _drop_orphan_unlocked(self, block_hash: str) -> bool:
        block = self._orphan_blocks.pop(block_hash, None)
        self._orphan_received_at.pop(block_hash, None)
        if block is None:
            return False

        siblings = self._orphans_by_prev.get(block.prev_hash)
        if siblings is not None:
            siblings.discard(block_hash)
            if not siblings:
                self._orphans_by_prev.pop(block.prev_hash, None)
        return True

    def _prune_orphans_unlocked(self) -> int:
        removed = 0
        now = time.time()
        stale_before = now - float(self.orphan_ttl_seconds)

        stale_hashes = [block_hash for block_hash, ts in self._orphan_received_at.items() if ts < stale_before]
        for block_hash in stale_hashes:
            if self._drop_orphan_unlocked(block_hash):
                removed += 1

        if len(self._orphan_blocks) > self.max_orphan_blocks:
            overflow = len(self._orphan_blocks) - self.max_orphan_blocks
            oldest = sorted(self._orphan_received_at.items(), key=lambda item: item[1])[:overflow]
            for block_hash, _ts in oldest:
                if self._drop_orphan_unlocked(block_hash):
                    removed += 1

        return removed

    def _store_orphan_unlocked(self, block: Block) -> str:
        if block.block_hash in self._orphan_blocks:
            return "orphan-known"
        if self._chain_has_block_hash_unlocked(block.block_hash):
            return "known"

        self._prune_orphans_unlocked()
        if len(self._orphan_blocks) >= self.max_orphan_blocks:
            oldest = sorted(self._orphan_received_at.items(), key=lambda item: item[1])[:1]
            for block_hash, _ts in oldest:
                self._drop_orphan_unlocked(block_hash)

        self._orphan_blocks[block.block_hash] = Block.from_dict(block.to_dict())
        self._orphan_received_at[block.block_hash] = time.time()
        self._orphans_by_prev[block.prev_hash].add(block.block_hash)
        return "orphan-stored"

    def _collect_orphan_children_unlocked(self, parent_hash: str) -> list[Block]:
        child_hashes = sorted(self._orphans_by_prev.get(parent_hash, set()))
        if not child_hashes:
            return []

        children: list[Block] = []
        for child_hash in child_hashes:
            child_block = self._orphan_blocks.get(child_hash)
            if child_block is not None:
                children.append(child_block)
            self._drop_orphan_unlocked(child_hash)

        children.sort(key=lambda block: (block.index, block.block_hash))
        return children

    def _drain_orphans_after_parent(self, parent_hash: str, ttl: int) -> int:
        accepted = 0
        queue: deque[str] = deque([parent_hash])
        seen: set[str] = set()
        relay_ttl = max(0, self._coerce_ttl(ttl, field_name="ttl") - 1)

        while queue:
            current_parent = queue.popleft()
            if current_parent in seen:
                continue
            seen.add(current_parent)

            with self.chain_lock:
                self._refresh_chain_from_disk_unlocked()
                self._prune_orphans_unlocked()
                children = self._collect_orphan_children_unlocked(current_parent)

            for child in children:
                result = self.accept_block(child, sender="", ttl=relay_ttl, auth=None)
                if bool(result.get("accepted")):
                    accepted += 1
                    queue.append(child.block_hash)
                elif str(result.get("reason", "")).startswith("orphan"):
                    continue
                else:
                    with self.chain_lock:
                        self._drop_orphan_unlocked(child.block_hash)

        return accepted

    def getdata_payload(self, items_raw: Any, ttl: int = 2) -> dict[str, Any]:
        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")
        items = self._normalize_inventory_items(items_raw, limit=self.max_getdata_items)
        if not items:
            return {"ok": True, "rows": [], "missing": [], "count": 0}

        rows: list[dict[str, Any]] = []
        missing: list[dict[str, str]] = []

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self._prune_orphans_unlocked()
            mempool_by_id = {tx.txid: tx for tx in self.chain.mempool}
            blocks_by_hash = {block.block_hash: block for block in self.chain.chain}
            blocks_by_hash.update(self._orphan_blocks)

            for item in items:
                kind = item["type"]
                object_id = item["id"]
                if kind == "tx":
                    tx = mempool_by_id.get(object_id)
                    if tx is None:
                        missing.append(item)
                        continue
                    rows.append({"type": "tx", "id": object_id, "tx": tx.to_dict()})
                    continue

                block = blocks_by_hash.get(object_id)
                if block is None:
                    missing.append(item)
                    continue
                rows.append({"type": "block", "id": object_id, "block": block.to_dict()})

        for row in rows:
            row["auth"] = self._build_sender_auth(
                purpose=row["type"],
                object_id=row["id"],
                ttl=relay_ttl,
            )

        return {
            "ok": True,
            "rows": rows,
            "missing": missing,
            "count": len(rows),
        }

    def _request_inventory_objects(
        self,
        peer_url: str,
        items: list[dict[str, str]],
        ttl: int,
    ) -> dict[str, Any]:
        requested = self._normalize_inventory_items(items, limit=self.max_getdata_items)
        if not requested:
            return {
                "requested": 0,
                "fetched": 0,
                "accepted_tx": 0,
                "accepted_blocks": 0,
                "missing": 0,
            }

        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")
        contains_block = any(item.get("type") == "block" for item in requested)
        timeout = max(self.request_timeout, 8.0) if contains_block else self.request_timeout

        try:
            response = _request_json(
                _join_url(peer_url, "/getdata"),
                method="POST",
                payload={
                    "items": requested,
                    "ttl": relay_ttl,
                    "from": self.node_url,
                },
                timeout=timeout,
            )
        except Exception as exc:
            self._penalize_peer(
                peer_url,
                max(1, self.peer_penalty_bad_sync // 2),
                f"getdata-fetch-failed:{str(exc)[:80]}",
            )
            return {
                "requested": len(requested),
                "fetched": 0,
                "accepted_tx": 0,
                "accepted_blocks": 0,
                "missing": len(requested),
                "error": str(exc),
            }

        rows = response.get("rows")
        if not isinstance(rows, list):
            self._penalize_peer(peer_url, self.peer_penalty_bad_sync, "getdata-invalid-rows")
            return {
                "requested": len(requested),
                "fetched": 0,
                "accepted_tx": 0,
                "accepted_blocks": 0,
                "missing": len(requested),
                "error": "invalid-getdata-response",
            }

        requested_set = {(item["type"], item["id"]) for item in requested}
        fetched = 0
        accepted_tx = 0
        accepted_blocks = 0
        returned_set: set[tuple[str, str]] = set()

        for row in rows:
            if not isinstance(row, dict):
                continue

            kind = str(row.get("type", "")).strip().lower()
            object_id = str(row.get("id", "")).strip().lower()
            key = (kind, object_id)
            if key not in requested_set or key in returned_set:
                continue
            returned_set.add(key)

            auth = row.get("auth")
            if not isinstance(auth, dict):
                self._penalize_peer(peer_url, max(1, self.peer_penalty_bad_sync // 2), "getdata-missing-auth")
                continue

            if kind == "tx":
                tx_raw = row.get("tx")
                if not isinstance(tx_raw, dict):
                    continue
                try:
                    tx = Transaction.from_dict(tx_raw)
                except Exception:
                    self._penalize_peer(peer_url, max(1, self.peer_penalty_bad_sync // 2), "getdata-invalid-tx")
                    continue
                if tx.txid != object_id:
                    self._penalize_peer(peer_url, self.peer_penalty_invalid_tx, "getdata-txid-mismatch")
                    continue
                fetched += 1
                result = self.accept_transaction(tx, sender=peer_url, ttl=relay_ttl, auth=auth)
                if bool(result.get("accepted")):
                    accepted_tx += 1
                continue

            if kind == "block":
                block_raw = row.get("block")
                if not isinstance(block_raw, dict):
                    continue
                try:
                    block = Block.from_dict(block_raw)
                except Exception:
                    self._penalize_peer(peer_url, max(1, self.peer_penalty_bad_sync // 2), "getdata-invalid-block")
                    continue
                if block.block_hash != object_id:
                    self._penalize_peer(peer_url, self.peer_penalty_invalid_block, "getdata-block-hash-mismatch")
                    continue
                fetched += 1
                result = self.accept_block(block, sender=peer_url, ttl=relay_ttl, auth=auth)
                if bool(result.get("accepted")):
                    accepted_blocks += 1

        missing_count = len(requested_set - returned_set)
        return {
            "requested": len(requested),
            "fetched": fetched,
            "accepted_tx": accepted_tx,
            "accepted_blocks": accepted_blocks,
            "missing": missing_count,
        }

    def accept_inventory(self, items_raw: Any, sender: str = "", ttl: int = 2) -> dict[str, Any]:
        sender_url = self._normalize_sender(sender)
        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")

        if sender_url:
            if self._is_peer_banned(sender_url):
                return {"ok": False, "accepted": False, "reason": "sender-banned", "requested": 0}
            if not self._is_known_peer(sender_url):
                return {"ok": False, "accepted": False, "reason": "sender-not-known", "requested": 0}

        items = self._normalize_inventory_items(items_raw, limit=self.max_inv_items)
        if not items:
            return {"ok": True, "accepted": False, "reason": "empty-inventory", "requested": 0}

        requested: list[dict[str, str]] = []
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self._prune_orphans_unlocked()

            for item in items:
                kind = item["type"]
                object_id = item["id"]
                if kind == "tx":
                    if self._has_txid_unlocked(object_id):
                        continue
                    requested.append(item)
                    continue

                if self._chain_has_block_hash_unlocked(object_id) or object_id in self._seen_block_hashes:
                    continue
                requested.append(item)

        if not requested:
            return {
                "ok": True,
                "accepted": True,
                "reason": "inventory-known",
                "requested": 0,
            }

        if not sender_url:
            return {
                "ok": True,
                "accepted": False,
                "reason": "inventory-sender-missing",
                "requested": len(requested),
            }

        fetch = self._request_inventory_objects(sender_url, requested, relay_ttl)
        return {
            "ok": True,
            "accepted": True,
            "reason": "inventory-requested",
            **fetch,
        }

    def _broadcast_inventory(
        self,
        items: list[dict[str, str]],
        ttl: int,
        exclude: set[str] | None = None,
    ) -> list[str]:
        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")
        if relay_ttl <= 0:
            return []

        normalized_items = self._normalize_inventory_items(items, limit=self.max_inv_items)
        if not normalized_items:
            return []

        skip = set(exclude or set())
        failed: list[str] = []
        for peer in self._get_relay_peers(limit=self.max_outbound_broadcast_peers):
            if peer in skip:
                continue
            try:
                _request_json(
                    _join_url(peer, "/inv"),
                    method="POST",
                    payload={
                        "items": normalized_items,
                        "ttl": relay_ttl,
                        "from": self.node_url,
                    },
                    timeout=self.request_timeout,
                )
            except NetworkError:
                failed.append(peer)
        return failed

    def _broadcast_tx(self, tx: Transaction, ttl: int, exclude: set[str] | None = None) -> None:
        bounded_ttl = self._coerce_ttl(ttl, field_name="ttl")
        if bounded_ttl <= 0:
            return
        skip = set(exclude or set())
        failed_peers = self._broadcast_inventory(
            [{"type": "tx", "id": tx.txid}],
            ttl=bounded_ttl,
            exclude=skip,
        )
        if not failed_peers:
            return

        auth = self._build_sender_auth(purpose="tx", object_id=tx.txid, ttl=bounded_ttl)
        for peer in failed_peers:
            try:
                _request_json(
                    _join_url(peer, "/tx"),
                    method="POST",
                    payload={
                        "tx": tx.to_dict(),
                        "ttl": bounded_ttl,
                        "from": self.node_url,
                        "auth": auth,
                    },
                    timeout=self.request_timeout,
                )
            except NetworkError:
                continue

    def _broadcast_block(self, block: Block, ttl: int, exclude: set[str] | None = None) -> None:
        bounded_ttl = self._coerce_ttl(ttl, field_name="ttl")
        if bounded_ttl <= 0:
            return
        skip = set(exclude or set())
        failed_peers = self._broadcast_inventory(
            [{"type": "block", "id": block.block_hash}],
            ttl=bounded_ttl,
            exclude=skip,
        )
        if not failed_peers:
            return

        auth = self._build_sender_auth(purpose="block", object_id=block.block_hash, ttl=bounded_ttl)
        for peer in failed_peers:
            try:
                _request_json(
                    _join_url(peer, "/block"),
                    method="POST",
                    payload={
                        "block": block.to_dict(),
                        "ttl": bounded_ttl,
                        "from": self.node_url,
                        "auth": auth,
                    },
                    timeout=max(self.request_timeout, 6.0),
                )
            except NetworkError:
                continue

    def create_and_broadcast_transaction(
        self,
        private_key_hex: str,
        to_address: str,
        amount: int,
        fee: int | None = None,
        broadcast_ttl: int = 2,
    ) -> dict[str, Any]:
        if amount <= 0:
            raise ValidationError("amount must be positive")
        ttl = self._coerce_ttl(broadcast_ttl, field_name="broadcast_ttl")

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            if not self.chain.chain:
                raise ValidationError("Initialize the chain first")
            tx = self.chain.create_transaction(
                private_key_hex=private_key_hex,
                to_address=to_address,
                amount=amount,
                fee=fee,
            )
            self.chain.add_transaction(tx)
            self._remember_txid(tx.txid)

        if ttl > 0:
            self._broadcast_tx(tx, ttl=ttl)

        return {
            "ok": True,
            "txid": tx.txid,
            "inputs": len(tx.inputs),
            "outputs": len(tx.outputs),
            "broadcast_ttl": ttl,
        }

    def mine_blocks(
        self,
        miner_address: str,
        blocks: int = 1,
        backend: str = "auto",
        broadcast_ttl: int = 2,
    ) -> dict[str, Any]:
        if blocks <= 0:
            raise ValidationError("blocks must be positive")
        if backend not in {"auto", "gpu", "cpu"}:
            raise ValidationError("backend must be one of: auto, gpu, cpu")
        ttl = self._coerce_ttl(broadcast_ttl, field_name="broadcast_ttl")

        mined_rows: list[dict[str, Any]] = []
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            if not self.chain.chain:
                raise ValidationError("Initialize the chain first")

            for _ in range(blocks):
                block = self.chain.mine_block(miner_address, mining_backend=backend)
                self._remember_block_hash(block.block_hash)
                for tx in block.transactions:
                    self._remember_txid(tx.txid)
                mined_rows.append(
                    {
                        "height": block.index,
                        "hash": block.block_hash,
                        "reward": sum(out.amount for out in block.transactions[0].outputs),
                        "tx_count": len(block.transactions),
                    }
                )

        if ttl > 0:
            with self.chain_lock:
                self._refresh_chain_from_disk_unlocked()
                # Broadcast only the newly mined blocks by hash lookup.
                mined_hashes = {row["hash"] for row in mined_rows}
                blocks_to_send = [block for block in self.chain.chain if block.block_hash in mined_hashes]
            for block in blocks_to_send:
                self._broadcast_block(block, ttl=ttl)

        return {
            "ok": True,
            "blocks": mined_rows,
            "count": len(mined_rows),
            "backend": backend,
            "broadcast_ttl": ttl,
        }

    def prune_mempool(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            removed = self.chain.prune_mempool(save=True)
            size = len(self.chain.mempool)
        return {"ok": True, "removed": removed, "size": size}

    def accept_transaction(self, tx: Transaction, sender: str = "", ttl: int = 2, auth: Any = None) -> dict[str, Any]:
        sender_url = self._normalize_sender(sender)
        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")

        if sender_url:
            auth_ok, auth_reason = self._validate_sender_auth(
                sender_url=sender_url,
                auth=auth,
                purpose="tx",
                object_id=tx.txid,
                ttl=relay_ttl,
            )
            if not auth_ok:
                if auth_reason != "sender-banned":
                    self._penalize_peer(
                        sender_url,
                        max(1, self.peer_penalty_invalid_tx // 2),
                        f"auth-tx:{auth_reason}",
                    )
                return {"ok": False, "accepted": False, "reason": auth_reason, "txid": tx.txid}

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()

            if tx.txid in self._seen_txids:
                known_result = {"ok": True, "accepted": True, "reason": "known", "txid": tx.txid}
                # Allow local/manual re-broadcast of known transactions.
                if relay_ttl > 0:
                    self._broadcast_tx(tx, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
                return known_result

            if not self.chain.chain:
                return {"ok": False, "accepted": False, "reason": "chain-not-initialized", "txid": tx.txid}

            try:
                self.chain.add_transaction(tx)
            except ValidationError as exc:
                err = str(exc)
                if "Transaction already exists in mempool" in err:
                    self._remember_txid(tx.txid)
                    if relay_ttl > 0:
                        self._broadcast_tx(tx, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
                    return {"ok": True, "accepted": True, "reason": "known", "txid": tx.txid}
                # A sync-retry can recover from lagging state between peers.
                if self._allow_sync_retry(sender_url, "tx") and ("Missing UTXO" in err or "Transaction ID mismatch" in err):
                    sync = self.sync_from_peer(sender_url)
                    if sync.get("updated"):
                        with self.chain_lock:
                            self._refresh_chain_from_disk_unlocked()
                            try:
                                self.chain.add_transaction(tx)
                            except ValidationError as inner_exc:
                                self._penalize_peer(
                                    sender_url,
                                    self.peer_penalty_invalid_tx,
                                    f"invalid-tx-after-sync:{str(inner_exc)[:80]}",
                                )
                                return {"ok": False, "accepted": False, "reason": str(inner_exc), "txid": tx.txid}
                    else:
                        self._penalize_peer(
                            sender_url,
                            max(1, self.peer_penalty_invalid_tx // 2),
                            f"tx-sync-not-updated:{err[:80]}",
                        )
                        return {"ok": False, "accepted": False, "reason": err, "txid": tx.txid}
                else:
                    self._penalize_peer(sender_url, self.peer_penalty_invalid_tx, f"invalid-tx:{err[:80]}")
                    return {"ok": False, "accepted": False, "reason": err, "txid": tx.txid}

            self._remember_txid(tx.txid)
            self._reward_peer(sender_url)

        self._broadcast_tx(tx, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
        return {"ok": True, "accepted": True, "reason": "accepted", "txid": tx.txid}

    def accept_block(self, block: Block, sender: str = "", ttl: int = 2, auth: Any = None) -> dict[str, Any]:
        sender_url = self._normalize_sender(sender)
        relay_ttl = self._coerce_ttl(ttl, field_name="ttl")

        if sender_url:
            auth_ok, auth_reason = self._validate_sender_auth(
                sender_url=sender_url,
                auth=auth,
                purpose="block",
                object_id=block.block_hash,
                ttl=relay_ttl,
            )
            if not auth_ok:
                if auth_reason != "sender-banned":
                    self._penalize_peer(
                        sender_url,
                        max(1, self.peer_penalty_invalid_block // 2),
                        f"auth-block:{auth_reason}",
                    )
                return {"ok": False, "accepted": False, "reason": auth_reason, "hash": block.block_hash}

        orphan_pool_size = 0
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            self._prune_orphans_unlocked()

            if block.block_hash in self._seen_block_hashes:
                known_result = {"ok": True, "accepted": True, "reason": "known", "hash": block.block_hash}
                if relay_ttl > 0:
                    self._broadcast_block(block, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
                return known_result

            if any(existing.block_hash == block.block_hash for existing in self.chain.chain):
                self._remember_block_hash(block.block_hash)
                self._drop_orphan_unlocked(block.block_hash)
                if relay_ttl > 0:
                    self._broadcast_block(block, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
                return {"ok": True, "accepted": True, "reason": "known", "hash": block.block_hash}

            parent_on_main_chain = any(existing.block_hash == block.prev_hash for existing in self.chain.chain)
            if block.index > 0 and not parent_on_main_chain:
                orphan_reason = self._store_orphan_unlocked(block)
                return {
                    "ok": True,
                    "accepted": False,
                    "reason": orphan_reason,
                    "hash": block.block_hash,
                    "orphan": True,
                    "orphan_pool_size": len(self._orphan_blocks),
                }

            try:
                if not self.chain.chain and block.index == 0:
                    self.chain.replace_chain([block], require_better=False)
                else:
                    self.chain.add_block(block)
            except ValidationError as exc:
                err = str(exc)
                # A lagging node can sync and then retry once.
                if self._allow_sync_retry(sender_url, "block") and ("Wrong block index" in err or "Wrong previous block hash" in err):
                    sync = self.sync_from_peer(sender_url)
                    if sync.get("updated"):
                        with self.chain_lock:
                            self._refresh_chain_from_disk_unlocked()
                            try:
                                if any(existing.block_hash == block.block_hash for existing in self.chain.chain):
                                    self._remember_block_hash(block.block_hash)
                                    return {"ok": True, "accepted": True, "reason": "known", "hash": block.block_hash}
                                self.chain.add_block(block)
                            except ValidationError as inner_exc:
                                self._penalize_peer(
                                    sender_url,
                                    self.peer_penalty_invalid_block,
                                    f"invalid-block-after-sync:{str(inner_exc)[:80]}",
                                )
                                return {
                                    "ok": False,
                                    "accepted": False,
                                    "reason": str(inner_exc),
                                    "hash": block.block_hash,
                                }
                    else:
                        self._penalize_peer(
                            sender_url,
                            max(1, self.peer_penalty_invalid_block // 2),
                            f"block-sync-not-updated:{err[:80]}",
                        )
                        return {"ok": False, "accepted": False, "reason": err, "hash": block.block_hash}
                else:
                    self._penalize_peer(sender_url, self.peer_penalty_invalid_block, f"invalid-block:{err[:80]}")
                    return {"ok": False, "accepted": False, "reason": err, "hash": block.block_hash}

            self._drop_orphan_unlocked(block.block_hash)
            self._remember_block_hash(block.block_hash)
            for tx in block.transactions:
                self._remember_txid(tx.txid)
            self._reward_peer(sender_url)
            orphan_pool_size = len(self._orphan_blocks)

        self._broadcast_block(block, ttl=relay_ttl - 1, exclude={sender_url} if sender_url else set())
        adopted_orphans = self._drain_orphans_after_parent(block.block_hash, ttl=relay_ttl)
        return {
            "ok": True,
            "accepted": True,
            "reason": "accepted",
            "hash": block.block_hash,
            "adopted_orphans": adopted_orphans,
            "orphan_pool_size": orphan_pool_size,
        }

    def _candidate_peer_metas(self) -> list[tuple[str, int, int]]:
        metas: list[tuple[str, int, int]] = []
        for peer in self._get_relay_peers(limit=self.max_outbound_sync_peers):
            try:
                try:
                    meta = _request_json(_join_url(peer, "/headers/meta"), method="GET", timeout=self.request_timeout)
                except Exception:
                    meta = _request_json(_join_url(peer, "/snapshot/meta"), method="GET", timeout=self.request_timeout)
                chain_work = int(meta.get("chain_work", 0))
                height = int(meta.get("height", -1))
                metas.append((peer, chain_work, height))
            except Exception:
                continue
        return metas

    def sync_from_best_peer(self) -> dict[str, Any]:
        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            local_work = int(self.chain.tip.chain_work) if self.chain.tip else 0
            local_height = self.chain.height

        best_peer = ""
        best_work = local_work
        best_height = local_height
        for peer, work, height in self._candidate_peer_metas():
            if work > best_work or (work == best_work and height > best_height):
                best_peer = peer
                best_work = work
                best_height = height

        if not best_peer:
            return {"updated": False, "reason": "no-better-peer", "peer": None}

        return self.sync_from_peer(best_peer)

    @staticmethod
    def _is_penalizable_sync_reason(reason: str) -> bool:
        normalized = reason.strip().lower()
        if not normalized or normalized in {"not-better", "no-better-peer"}:
            return False
        if normalized.startswith("snapshot-fetch-failed: timeout"):
            return False
        if normalized.startswith("snapshot-fetch-failed: network error"):
            return False
        markers = (
            "invalid-headers",
            "header-",
            "missing-headers",
            "invalid-block-range",
            "missing-block-range",
            "block-decode-failed",
            "empty-block-range-chunk",
            "unexpected-block-height",
            "header-block-hash-mismatch",
            "invalid-snapshot",
            "decode-failed",
            "validation-failed",
            "no-common-ancestor",
            "reorg-depth-exceeded",
            "broken-prev-hash-link",
            "non-contiguous-height",
        )
        return any(marker in normalized for marker in markers)

    def sync_from_peer(self, peer: str) -> dict[str, Any]:
        try:
            peer_url = normalize_peer(peer)
        except ValueError as exc:
            return {"updated": False, "reason": str(exc), "peer": peer}

        if self._is_peer_banned(peer_url):
            return {"updated": False, "reason": "peer-banned", "peer": peer_url}

        try:
            header_result = self._sync_from_peer_headers(peer_url)
            if header_result.get("mode") == "headers":
                if bool(header_result.get("updated")):
                    self._reward_peer(peer_url)
                else:
                    reason = str(header_result.get("reason", ""))
                    if self._is_penalizable_sync_reason(reason):
                        self._penalize_peer(
                            peer_url,
                            self.peer_penalty_bad_sync,
                            f"bad-sync-headers:{reason[:80]}",
                        )
                return header_result
        except Exception as exc:
            self._penalize_peer(
                peer_url,
                max(1, self.peer_penalty_bad_sync // 2),
                f"headers-sync-exception:{str(exc)[:80]}",
            )
            # Compatibility fallback for older peers that do not provide header endpoints.
            pass

        try:
            snapshot = _request_json(_join_url(peer_url, "/snapshot"), method="GET", timeout=max(self.request_timeout, 8.0))
        except Exception as exc:
            return {"updated": False, "reason": f"snapshot-fetch-failed: {exc}", "peer": peer_url}

        chain_raw = snapshot.get("chain")
        if not isinstance(chain_raw, list):
            self._penalize_peer(peer_url, self.peer_penalty_bad_sync, "bad-sync-snapshot:invalid-snapshot")
            return {"updated": False, "reason": "invalid-snapshot", "peer": peer_url}

        mempool_raw = snapshot.get("mempool")
        if not isinstance(mempool_raw, list):
            mempool_raw = []

        try:
            blocks = [Block.from_dict(item) for item in chain_raw if isinstance(item, dict)]
            incoming_mempool = [Transaction.from_dict(item) for item in mempool_raw if isinstance(item, dict)]
        except Exception as exc:
            self._penalize_peer(peer_url, self.peer_penalty_bad_sync, "bad-sync-snapshot:decode-failed")
            return {"updated": False, "reason": f"decode-failed: {exc}", "peer": peer_url}

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            require_better = bool(self.chain.chain)
            try:
                updated = self.chain.replace_chain(blocks, incoming_mempool=incoming_mempool, require_better=require_better)
            except ValidationError as exc:
                self._penalize_peer(peer_url, self.peer_penalty_bad_sync, f"bad-sync-snapshot:validation:{str(exc)[:80]}")
                return {"updated": False, "reason": f"validation-failed: {exc}", "peer": peer_url}
            self._prime_seen_from_chain_unlocked()

        if updated:
            self._reward_peer(peer_url)

        return {
            "updated": updated,
            "reason": "applied" if updated else "not-better",
            "peer": peer_url,
            "mode": "snapshot",
        }

    @staticmethod
    def _validate_header_rows(rows: list[dict[str, Any]]) -> tuple[bool, str]:
        if not rows:
            return False, "empty-headers"

        prev_height = -1
        prev_hash = ""
        for row in rows:
            if not isinstance(row, dict):
                return False, "invalid-header-row"
            try:
                height = int(row.get("height", -1))
                block_hash = str(row.get("hash", ""))
                row_prev = str(row.get("prev_hash", ""))
                int(row.get("timestamp", 0))
                int(row.get("target", 0))
                int(row.get("chain_work", 0))
            except Exception:
                return False, "invalid-header-types"

            if len(block_hash) != 64 or len(row_prev) != 64:
                return False, "invalid-header-hash-format"
            try:
                int(block_hash, 16)
                int(row_prev, 16)
            except ValueError:
                return False, "invalid-header-hash-hex"

            if prev_height >= 0:
                if height != prev_height + 1:
                    return False, "non-contiguous-height"
                if row_prev != prev_hash:
                    return False, "broken-prev-hash-link"

            prev_height = height
            prev_hash = block_hash

        return True, ""

    def _fetch_peer_headers(self, peer_url: str, tip_height: int) -> tuple[list[dict[str, Any]], str]:
        headers: list[dict[str, Any]] = []
        cursor = 0
        max_chunk = 2000

        while cursor <= tip_height:
            path = f"/headers?start_height={cursor}&limit={max_chunk}"
            response = _request_json(_join_url(peer_url, path), method="GET", timeout=max(self.request_timeout, 8.0))
            rows = response.get("rows")
            if not isinstance(rows, list):
                return [], "invalid-headers-response"
            if not rows:
                break

            parsed_rows: list[dict[str, Any]] = [row for row in rows if isinstance(row, dict)]
            headers.extend(parsed_rows)
            cursor += len(parsed_rows)

            if len(parsed_rows) < max_chunk:
                break

        if not headers:
            return [], "missing-headers"
        return headers, ""

    def _fetch_peer_blocks_range(
        self,
        peer_url: str,
        start_height: int,
        end_height: int,
    ) -> tuple[list[Block], str]:
        blocks: list[Block] = []
        cursor = start_height
        max_chunk = 250

        while cursor <= end_height:
            wanted = min(max_chunk, end_height - cursor + 1)
            path = f"/blocks/range?start_height={cursor}&limit={wanted}"
            response = _request_json(_join_url(peer_url, path), method="GET", timeout=max(self.request_timeout, 8.0))
            rows = response.get("rows")
            if not isinstance(rows, list):
                return [], "invalid-block-range-response"
            if not rows:
                return [], "missing-block-range-rows"

            parsed_chunk: list[Block] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    parsed_chunk.append(Block.from_dict(row))
                except Exception:
                    return [], "block-decode-failed"

            if not parsed_chunk:
                return [], "empty-block-range-chunk"

            for idx, block in enumerate(parsed_chunk):
                expected_height = cursor + idx
                if block.index != expected_height:
                    return [], "unexpected-block-height"

            blocks.extend(parsed_chunk)
            cursor += len(parsed_chunk)

        return blocks, ""

    def _sync_from_peer_headers(self, peer_url: str) -> dict[str, Any]:
        meta = _request_json(_join_url(peer_url, "/headers/meta"), method="GET", timeout=max(self.request_timeout, 6.0))
        remote_height = int(meta.get("height", -1))
        remote_work = int(meta.get("chain_work", 0))
        if remote_height < 0:
            return {"updated": False, "reason": "invalid-headers-meta", "peer": peer_url, "mode": "headers"}

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            local_work = int(self.chain.tip.chain_work) if self.chain.tip else 0
            local_height = self.chain.height
            local_chain = [Block.from_dict(block.to_dict()) for block in self.chain.chain]

        if remote_work < local_work or (remote_work == local_work and remote_height <= local_height):
            return {"updated": False, "reason": "not-better", "peer": peer_url, "mode": "headers"}

        headers, headers_err = self._fetch_peer_headers(peer_url, tip_height=remote_height)
        if headers_err:
            return {"updated": False, "reason": headers_err, "peer": peer_url, "mode": "headers"}

        valid_headers, reason = self._validate_header_rows(headers)
        if not valid_headers:
            return {"updated": False, "reason": reason, "peer": peer_url, "mode": "headers"}

        header_tip = headers[-1]
        if int(header_tip.get("height", -1)) != remote_height:
            return {"updated": False, "reason": "header-tip-height-mismatch", "peer": peer_url, "mode": "headers"}

        header_work = int(header_tip.get("chain_work", 0))
        if header_work != remote_work:
            return {"updated": False, "reason": "header-tip-work-mismatch", "peer": peer_url, "mode": "headers"}

        header_hash_by_height: dict[int, str] = {
            int(row["height"]): str(row["hash"])
            for row in headers
        }

        shared_height = -1
        probe = min(local_height, remote_height)
        while probe >= 0:
            local_hash = local_chain[probe].block_hash if probe < len(local_chain) else ""
            remote_hash = header_hash_by_height.get(probe, "")
            if local_hash and remote_hash and local_hash == remote_hash:
                shared_height = probe
                break
            probe -= 1

        if shared_height < 0:
            if local_height >= 0:
                return {"updated": False, "reason": "no-common-ancestor", "peer": peer_url, "mode": "headers"}
            # Empty local chain: full bootstrap from height 0.
            shared_height = -1

        reorg_depth = max(0, local_height - shared_height)
        if reorg_depth > self.chain.config.max_reorg_depth:
            return {
                "updated": False,
                "reason": f"reorg-depth-exceeded:{reorg_depth}",
                "peer": peer_url,
                "mode": "headers",
            }

        if shared_height == remote_height:
            return {"updated": False, "reason": "not-better", "peer": peer_url, "mode": "headers"}

        fetch_from = shared_height + 1
        fetched_blocks, block_err = self._fetch_peer_blocks_range(peer_url, start_height=fetch_from, end_height=remote_height)
        if block_err:
            return {"updated": False, "reason": block_err, "peer": peer_url, "mode": "headers"}

        for block in fetched_blocks:
            expected_hash = header_hash_by_height.get(block.index, "")
            if not expected_hash or block.block_hash != expected_hash:
                return {"updated": False, "reason": "header-block-hash-mismatch", "peer": peer_url, "mode": "headers"}

        candidate_prefix = [Block.from_dict(block.to_dict()) for block in local_chain[: shared_height + 1]]
        candidate_chain = candidate_prefix + fetched_blocks

        with self.chain_lock:
            self._refresh_chain_from_disk_unlocked()
            try:
                updated = self.chain.replace_chain(candidate_chain, incoming_mempool=None, require_better=True)
            except ValidationError as exc:
                return {
                    "updated": False,
                    "reason": f"validation-failed: {exc}",
                    "peer": peer_url,
                    "mode": "headers",
                }
            self._prime_seen_from_chain_unlocked()

        return {
            "updated": updated,
            "reason": "applied" if updated else "not-better",
            "peer": peer_url,
            "mode": "headers",
            "shared_height": shared_height,
            "fetched_blocks": len(fetched_blocks),
        }

    def _sync_loop(self) -> None:
        # Keep node state close to the strongest known peer.
        while not self.stop_event.wait(self.sync_interval):
            try:
                self.sync_from_best_peer()
            except Exception:
                continue

    def serve_forever(self) -> None:
        if self.sync_interval > 0:
            self._sync_thread = threading.Thread(target=self._sync_loop, name="powx-sync-loop", daemon=True)
            self._sync_thread.start()

        try:
            self.server.serve_forever(poll_interval=0.5)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
