from __future__ import annotations

import json
import mimetypes
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .market_indexer import MarketIndexer, MarketIndexerError
from .models import Transaction
from .p2p import (
    NetworkError,
    api_build_contract_transaction,
    api_create_contract_transaction,
    api_submit_transaction,
    normalize_peer,
)


def _query_first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key, [])
    if not values:
        return default
    return str(values[0])


def _parse_int(raw: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _parse_bool(raw: str) -> bool | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


class MarketplaceBackend:
    """Marketplace web layer: indexer + REST API + static frontend hosting."""

    def __init__(
        self,
        *,
        node_url: str,
        db_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8950,
        sync_interval: float = 5.0,
        request_timeout: float = 4.0,
        static_dir: str | Path | None = None,
        auto_sync: bool = True,
    ) -> None:
        self.node_url = normalize_peer(node_url)
        self.host = str(host).strip() or "127.0.0.1"
        self.port = int(port)
        self.sync_interval = max(1.0, float(sync_interval))
        self.request_timeout = max(0.5, float(request_timeout))
        self.auto_sync = bool(auto_sync)
        self.stop_event = threading.Event()
        self._shutdown_started = False
        self._sync_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._last_sync_result: dict[str, Any] = {}
        self._last_sync_error = ""
        self._last_sync_epoch = 0

        if static_dir is None:
            self.static_dir = (Path(__file__).resolve().parent / "market_ui").resolve()
        else:
            self.static_dir = Path(static_dir).resolve()

        self.indexer = MarketIndexer(db_path=db_path, node_url=self.node_url, timeout=self.request_timeout)

        class QuietThreadingHTTPServer(ThreadingHTTPServer):
            def handle_error(self, request: object, client_address: tuple[str, int]) -> None:  # type: ignore[override]
                _ = request
                _ = client_address
                _exc_type, exc_value, _tb = sys.exc_info()
                if exc_value is None:
                    return
                if isinstance(exc_value, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
                    return
                if isinstance(exc_value, OSError):
                    win_err = int(getattr(exc_value, "winerror", 0) or 0)
                    if win_err in {10053, 10054, 10038}:
                        return
                    if exc_value.errno in {socket.ECONNRESET if hasattr(socket, "ECONNRESET") else 104}:
                        return
                super().handle_error(request, client_address)

        handler = self._build_handler()
        self.server = QuietThreadingHTTPServer((self.host, self.port), handler)
        self.server.daemon_threads = True

    @property
    def base_url(self) -> str:
        bind_host, bind_port = self.server.server_address
        return f"http://{bind_host}:{bind_port}"

    def _record_sync_success(self, result: dict[str, Any]) -> None:
        self._last_sync_result = dict(result)
        self._last_sync_error = ""
        self._last_sync_epoch = int(time.time())

    def _record_sync_error(self, exc: Exception) -> None:
        self._last_sync_error = str(exc)
        self._last_sync_epoch = int(time.time())

    def sync_once(self, *, source: str = "manual") -> dict[str, Any]:
        with self._sync_lock:
            try:
                result = self.indexer.sync_once()
            except (NetworkError, MarketIndexerError, ValueError) as exc:
                self._record_sync_error(exc)
                raise
            except Exception as exc:
                self._record_sync_error(exc)
                raise MarketIndexerError(f"Sync failed: {exc}") from exc
            result["source"] = source
            self._record_sync_success(result)
            return result

    def _sync_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.sync_once(source="auto")
            except Exception:
                # Sync status is tracked via _last_sync_error.
                pass
            self.stop_event.wait(self.sync_interval)

    def _start_background_sync(self) -> None:
        if not self.auto_sync:
            return
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return
        self._sync_thread = threading.Thread(target=self._sync_loop, name="kk91-market-sync", daemon=True)
        self._sync_thread.start()

    def status_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "kk91-market-web",
            "node_url": self.node_url,
            "base_url": self.base_url,
            "auto_sync": self.auto_sync,
            "sync_interval_seconds": self.sync_interval,
            "request_timeout_seconds": self.request_timeout,
            "indexer": self.indexer.stats(),
            "last_sync_epoch": self._last_sync_epoch,
            "last_sync_result": dict(self._last_sync_result),
            "last_sync_error": self._last_sync_error,
            "static_dir": str(self.static_dir),
        }

    def serve_forever(self) -> None:
        try:
            try:
                self.sync_once(source="startup")
            except Exception:
                # Startup can continue even if node is temporarily unavailable.
                pass
            self._start_background_sync()
            self.server.serve_forever(poll_interval=0.25)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
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

    def _resolve_static_path(self, path: str) -> Path | None:
        relative = path.lstrip("/")
        if not relative:
            relative = "index.html"
        candidate = (self.static_dir / relative).resolve()
        try:
            candidate.relative_to(self.static_dir)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                _ = format
                _ = args
                return

            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                try:
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return

            def _read_json(self, max_bytes: int = 512000) -> dict[str, Any]:
                raw_len = self.headers.get("Content-Length", "0").strip()
                try:
                    length = int(raw_len)
                except Exception as exc:
                    raise ValueError("Invalid Content-Length") from exc
                if length <= 0:
                    raise ValueError("Missing request body")
                if length > max_bytes:
                    raise ValueError(f"Request body exceeds limit ({max_bytes} bytes)")
                data = self.rfile.read(length)
                try:
                    decoded = json.loads(data.decode("utf-8"))
                except Exception as exc:
                    raise ValueError("Invalid JSON body") from exc
                if not isinstance(decoded, dict):
                    raise ValueError("JSON body must be an object")
                return decoded

            def _send_file(self, file_path: Path) -> None:
                raw = file_path.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(file_path))
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", (mime_type or "application/octet-stream") + "; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                try:
                    if path in {"/health", "/api/v1/market/health"}:
                        self._send_json(200, {"ok": True, "service": "kk91-market-web"})
                        return

                    if path in {"/api/v1/market/status", "/api/v1/market/stats"}:
                        self._send_json(200, backend.status_payload())
                        return

                    if path == "/api/v1/market/nfts":
                        listed = _parse_bool(_query_first(query, "listed", ""))
                        rows = backend.indexer.list_nfts(
                            owner=_query_first(query, "owner", "").strip() or None,
                            listed=listed,
                            search=_query_first(query, "search", "").strip() or None,
                            limit=_parse_int(_query_first(query, "limit", "100"), 100, minimum=1, maximum=1000),
                            offset=_parse_int(_query_first(query, "offset", "0"), 0, minimum=0, maximum=2_000_000),
                        )
                        self._send_json(200, {"ok": True, "count": len(rows), "rows": rows})
                        return

                    if path == "/api/v1/market/listings":
                        rows = backend.indexer.list_listings(
                            seller=_query_first(query, "seller", "").strip() or None,
                            limit=_parse_int(_query_first(query, "limit", "100"), 100, minimum=1, maximum=1000),
                            offset=_parse_int(_query_first(query, "offset", "0"), 0, minimum=0, maximum=2_000_000),
                        )
                        self._send_json(200, {"ok": True, "count": len(rows), "rows": rows})
                        return

                    if path == "/api/v1/market/contracts":
                        rows = backend.indexer.list_contracts(
                            owner=_query_first(query, "owner", "").strip() or None,
                            template=_query_first(query, "template", "").strip() or None,
                            contract_id=_query_first(query, "contract_id", "").strip() or None,
                            limit=_parse_int(_query_first(query, "limit", "100"), 100, minimum=1, maximum=1000),
                            offset=_parse_int(_query_first(query, "offset", "0"), 0, minimum=0, maximum=2_000_000),
                        )
                        self._send_json(200, {"ok": True, "count": len(rows), "rows": rows})
                        return

                    static_path = backend._resolve_static_path(path)
                    if static_path is not None:
                        self._send_file(static_path)
                        return

                    self._send_json(404, {"ok": False, "error": "Not found"})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    if path == "/api/v1/market/sync":
                        result = backend.sync_once(source="manual")
                        self._send_json(200, {"ok": True, "result": result})
                        return

                    if path == "/api/v1/market/build-contract":
                        payload = self._read_json()
                        sender_pubkey = str(payload.get("sender_pubkey", "")).strip().lower()
                        contract_payload = payload.get("contract")
                        if not sender_pubkey:
                            raise ValueError("Missing sender_pubkey")
                        if not isinstance(contract_payload, dict):
                            raise ValueError("Missing or invalid contract payload")
                        fee = payload.get("fee")
                        to_address = payload.get("to_address")
                        amount = payload.get("amount")
                        timeout = float(payload.get("timeout", backend.request_timeout))
                        result = api_build_contract_transaction(
                            backend.node_url,
                            sender_pubkey=sender_pubkey,
                            contract_payload=contract_payload,
                            fee=int(fee) if fee is not None else None,
                            to_address=str(to_address).strip() if to_address is not None else None,
                            amount=int(amount) if amount is not None else None,
                            timeout=max(0.5, timeout),
                        )
                        self._send_json(200, {"ok": True, "result": result})
                        return

                    if path == "/api/v1/market/submit-signed":
                        payload = self._read_json()
                        tx_raw = payload.get("tx")
                        if not isinstance(tx_raw, dict):
                            raise ValueError("Missing signed tx object")
                        tx = Transaction.from_dict(tx_raw)
                        ttl = _parse_int(payload.get("broadcast_ttl", 2), 2, minimum=0, maximum=16)
                        timeout = max(0.5, float(payload.get("timeout", backend.request_timeout)))
                        result = api_submit_transaction(
                            backend.node_url,
                            tx=tx,
                            broadcast_ttl=ttl,
                            timeout=timeout,
                        )
                        self._send_json(200, {"ok": True, "result": result})
                        return

                    if path == "/api/v1/market/submit-contract":
                        payload = self._read_json()
                        private_key_hex = str(payload.get("private_key_hex", "")).strip().lower()
                        contract_payload = payload.get("contract")
                        if len(private_key_hex) != 64:
                            raise ValueError("private_key_hex must be a 64-char hex string")
                        try:
                            int(private_key_hex, 16)
                        except Exception as exc:
                            raise ValueError("private_key_hex must be hex") from exc
                        if not isinstance(contract_payload, dict):
                            raise ValueError("Missing or invalid contract payload")

                        fee = payload.get("fee")
                        to_address = payload.get("to_address")
                        amount = payload.get("amount")
                        ttl = _parse_int(payload.get("broadcast_ttl", 2), 2, minimum=0, maximum=16)
                        timeout = max(0.5, float(payload.get("timeout", backend.request_timeout)))

                        result = api_create_contract_transaction(
                            backend.node_url,
                            private_key_hex=private_key_hex,
                            contract_payload=contract_payload,
                            fee=int(fee) if fee is not None else None,
                            to_address=str(to_address).strip() if to_address is not None else None,
                            amount=int(amount) if amount is not None else None,
                            broadcast_ttl=ttl,
                            timeout=timeout,
                        )
                        # Best effort: keep read model fresh after successful write.
                        try:
                            backend.sync_once(source="post-submit-contract")
                        except Exception:
                            pass
                        self._send_json(200, {"ok": True, "result": result})
                        return

                    self._send_json(404, {"ok": False, "error": "Not found"})
                except (ValueError, NetworkError, MarketIndexerError) as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})

        return Handler
