from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .p2p import api_contracts, api_nft_listings, api_nfts, api_status


class MarketIndexerError(RuntimeError):
    pass


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class MarketIndexer:
    """SQLite-backed read model for NFT + smart-contract marketplace data."""

    def __init__(self, db_path: str | Path, node_url: str, timeout: float = 4.0) -> None:
        self.db_path = Path(db_path)
        self.node_url = str(node_url).strip()
        self.timeout = max(0.5, float(timeout))
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nfts (
            token_id TEXT PRIMARY KEY,
            creator TEXT NOT NULL,
            owner TEXT NOT NULL,
            metadata_uri TEXT NOT NULL,
            mint_height INTEGER NOT NULL,
            mint_txid TEXT NOT NULL,
            updated_height INTEGER NOT NULL,
            updated_txid TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS listings (
            token_id TEXT PRIMARY KEY,
            seller TEXT NOT NULL,
            price INTEGER NOT NULL,
            listed_height INTEGER NOT NULL,
            listed_txid TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contracts (
            contract_id TEXT PRIMARY KEY,
            template TEXT NOT NULL,
            owner TEXT NOT NULL,
            created_height INTEGER NOT NULL,
            created_txid TEXT NOT NULL,
            updated_height INTEGER NOT NULL,
            updated_txid TEXT NOT NULL,
            state_json TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_nfts_owner ON nfts(owner);
        CREATE INDEX IF NOT EXISTS idx_listings_seller ON listings(seller);
        CREATE INDEX IF NOT EXISTS idx_contracts_owner ON contracts(owner);
        """
        with self._lock:
            with closing(self._connect()) as conn:
                conn.executescript(schema)
                conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)", ("schema_version", "1"))
                conn.commit()

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    @staticmethod
    def _meta_set(cursor: sqlite3.Cursor, key: str, value: str) -> None:
        cursor.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @staticmethod
    def _purge_missing(cursor: sqlite3.Cursor, table: str, key_field: str, keys: list[str]) -> None:
        if keys:
            placeholders = ",".join("?" for _ in keys)
            cursor.execute(f"DELETE FROM {table} WHERE {key_field} NOT IN ({placeholders})", keys)
            return
        cursor.execute(f"DELETE FROM {table}")

    @staticmethod
    def _to_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def sync_once(self) -> dict[str, Any]:
        status_payload = api_status(self.node_url, timeout=self.timeout)
        nft_payload = api_nfts(self.node_url, timeout=self.timeout)
        listing_payload = api_nft_listings(self.node_url, timeout=self.timeout)
        contract_payload = api_contracts(self.node_url, timeout=self.timeout)
        return self.sync_from_payload(
            status_payload=status_payload,
            nft_payload=nft_payload,
            listing_payload=listing_payload,
            contract_payload=contract_payload,
            source="node",
        )

    def sync_from_payload(
        self,
        *,
        status_payload: dict[str, Any],
        nft_payload: dict[str, Any],
        listing_payload: dict[str, Any],
        contract_payload: dict[str, Any],
        source: str = "manual",
    ) -> dict[str, Any]:
        if not isinstance(status_payload, dict):
            raise MarketIndexerError("status payload must be an object")
        if not isinstance(nft_payload, dict):
            raise MarketIndexerError("nft payload must be an object")
        if not isinstance(listing_payload, dict):
            raise MarketIndexerError("listing payload must be an object")
        if not isinstance(contract_payload, dict):
            raise MarketIndexerError("contract payload must be an object")

        tokens_raw = nft_payload.get("tokens", {})
        listings_raw = listing_payload.get("listings", {})
        contracts_raw = contract_payload.get("contracts", {})
        if not isinstance(tokens_raw, dict):
            raise MarketIndexerError("nft payload has invalid tokens map")
        if not isinstance(listings_raw, dict):
            raise MarketIndexerError("listing payload has invalid listings map")
        if not isinstance(contracts_raw, dict):
            raise MarketIndexerError("contract payload has invalid contracts map")

        now_epoch = int(time.time())
        remote_height = _safe_int(status_payload.get("height"), default=-1)
        remote_chain_id = str(status_payload.get("chain_id", "")).strip()
        source_name = str(source).strip() or "manual"

        nft_rows: list[dict[str, Any]] = []
        for token_id, raw_entry in sorted(tokens_raw.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_entry, dict):
                raise MarketIndexerError("invalid nft entry")
            token = str(token_id).strip()
            if not token:
                continue
            nft_rows.append(
                {
                    "token_id": token,
                    "creator": str(raw_entry.get("creator", "")).strip(),
                    "owner": str(raw_entry.get("owner", "")).strip(),
                    "metadata_uri": str(raw_entry.get("metadata_uri", "")).strip(),
                    "mint_height": _safe_int(raw_entry.get("mint_height")),
                    "mint_txid": str(raw_entry.get("mint_txid", "")).strip(),
                    "updated_height": _safe_int(raw_entry.get("updated_height")),
                    "updated_txid": str(raw_entry.get("updated_txid", "")).strip(),
                    "raw_json": self._to_json(raw_entry),
                }
            )

        listing_rows: list[dict[str, Any]] = []
        for token_id, raw_entry in sorted(listings_raw.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_entry, dict):
                raise MarketIndexerError("invalid listing entry")
            token = str(token_id).strip()
            if not token:
                continue
            listing_rows.append(
                {
                    "token_id": token,
                    "seller": str(raw_entry.get("seller", "")).strip(),
                    "price": _safe_int(raw_entry.get("price")),
                    "listed_height": _safe_int(raw_entry.get("listed_height")),
                    "listed_txid": str(raw_entry.get("listed_txid", "")).strip(),
                    "raw_json": self._to_json(raw_entry),
                }
            )

        contract_rows: list[dict[str, Any]] = []
        for contract_id, raw_entry in sorted(contracts_raw.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_entry, dict):
                raise MarketIndexerError("invalid contract entry")
            cid = str(contract_id).strip()
            if not cid:
                continue
            raw_state = raw_entry.get("state", {})
            if not isinstance(raw_state, dict):
                raw_state = {}
            contract_rows.append(
                {
                    "contract_id": cid,
                    "template": str(raw_entry.get("template", "")).strip(),
                    "owner": str(raw_entry.get("owner", "")).strip(),
                    "created_height": _safe_int(raw_entry.get("created_height")),
                    "created_txid": str(raw_entry.get("created_txid", "")).strip(),
                    "updated_height": _safe_int(raw_entry.get("updated_height")),
                    "updated_txid": str(raw_entry.get("updated_txid", "")).strip(),
                    "state_json": self._to_json(raw_state),
                    "raw_json": self._to_json(raw_entry),
                }
            )

        with self._lock:
            with closing(self._connect()) as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN")

                nft_keys: list[str] = []
                for row in nft_rows:
                    nft_keys.append(row["token_id"])
                    cursor.execute(
                        """
                        INSERT INTO nfts(
                            token_id, creator, owner, metadata_uri, mint_height, mint_txid,
                            updated_height, updated_txid, raw_json, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(token_id) DO UPDATE SET
                            creator=excluded.creator,
                            owner=excluded.owner,
                            metadata_uri=excluded.metadata_uri,
                            mint_height=excluded.mint_height,
                            mint_txid=excluded.mint_txid,
                            updated_height=excluded.updated_height,
                            updated_txid=excluded.updated_txid,
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at
                        """,
                        (
                            row["token_id"],
                            row["creator"],
                            row["owner"],
                            row["metadata_uri"],
                            row["mint_height"],
                            row["mint_txid"],
                            row["updated_height"],
                            row["updated_txid"],
                            row["raw_json"],
                            now_epoch,
                        ),
                    )
                self._purge_missing(cursor, "nfts", "token_id", nft_keys)

                listing_keys: list[str] = []
                for row in listing_rows:
                    listing_keys.append(row["token_id"])
                    cursor.execute(
                        """
                        INSERT INTO listings(
                            token_id, seller, price, listed_height, listed_txid, raw_json, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(token_id) DO UPDATE SET
                            seller=excluded.seller,
                            price=excluded.price,
                            listed_height=excluded.listed_height,
                            listed_txid=excluded.listed_txid,
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at
                        """,
                        (
                            row["token_id"],
                            row["seller"],
                            row["price"],
                            row["listed_height"],
                            row["listed_txid"],
                            row["raw_json"],
                            now_epoch,
                        ),
                    )
                self._purge_missing(cursor, "listings", "token_id", listing_keys)

                contract_keys: list[str] = []
                for row in contract_rows:
                    contract_keys.append(row["contract_id"])
                    cursor.execute(
                        """
                        INSERT INTO contracts(
                            contract_id, template, owner, created_height, created_txid,
                            updated_height, updated_txid, state_json, raw_json, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(contract_id) DO UPDATE SET
                            template=excluded.template,
                            owner=excluded.owner,
                            created_height=excluded.created_height,
                            created_txid=excluded.created_txid,
                            updated_height=excluded.updated_height,
                            updated_txid=excluded.updated_txid,
                            state_json=excluded.state_json,
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at
                        """,
                        (
                            row["contract_id"],
                            row["template"],
                            row["owner"],
                            row["created_height"],
                            row["created_txid"],
                            row["updated_height"],
                            row["updated_txid"],
                            row["state_json"],
                            row["raw_json"],
                            now_epoch,
                        ),
                    )
                self._purge_missing(cursor, "contracts", "contract_id", contract_keys)

                self._meta_set(cursor, "last_sync_epoch", str(now_epoch))
                self._meta_set(cursor, "last_sync_height", str(remote_height))
                self._meta_set(cursor, "last_sync_chain_id", remote_chain_id)
                self._meta_set(cursor, "last_sync_source", source_name)
                self._meta_set(cursor, "last_sync_node", self.node_url)
                conn.commit()

        stats = self.stats()
        stats["synced_height"] = remote_height
        stats["synced_chain_id"] = remote_chain_id
        stats["source"] = source_name
        return stats

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with closing(self._connect()) as conn:
                nft_count = int(conn.execute("SELECT COUNT(*) AS c FROM nfts").fetchone()["c"])
                listing_count = int(conn.execute("SELECT COUNT(*) AS c FROM listings").fetchone()["c"])
                contract_count = int(conn.execute("SELECT COUNT(*) AS c FROM contracts").fetchone()["c"])
                return {
                    "db_path": str(self.db_path),
                    "nft_count": nft_count,
                    "listing_count": listing_count,
                    "contract_count": contract_count,
                    "last_sync_epoch": _safe_int(self._meta_get(conn, "last_sync_epoch", "0")),
                    "last_sync_height": _safe_int(self._meta_get(conn, "last_sync_height", "-1"), default=-1),
                    "last_sync_chain_id": self._meta_get(conn, "last_sync_chain_id", ""),
                    "last_sync_source": self._meta_get(conn, "last_sync_source", ""),
                    "last_sync_node": self._meta_get(conn, "last_sync_node", ""),
                }

    def list_nfts(
        self,
        *,
        owner: str | None = None,
        listed: bool | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), 1000))
        start = max(0, int(offset))
        clauses: list[str] = []
        args: list[Any] = []
        if owner:
            clauses.append("n.owner = ?")
            args.append(str(owner).strip())
        if listed is True:
            clauses.append("l.token_id IS NOT NULL")
        elif listed is False:
            clauses.append("l.token_id IS NULL")
        if search:
            pattern = f"%{str(search).strip()}%"
            clauses.append("(n.token_id LIKE ? OR n.metadata_uri LIKE ?)")
            args.extend([pattern, pattern])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                n.token_id,
                n.creator,
                n.owner,
                n.metadata_uri,
                n.mint_height,
                n.mint_txid,
                n.updated_height,
                n.updated_txid,
                l.token_id AS listing_token_id,
                l.seller AS listing_seller,
                l.price AS listing_price,
                l.listed_height,
                l.listed_txid
            FROM nfts n
            LEFT JOIN listings l ON l.token_id = n.token_id
            {where}
            ORDER BY n.updated_height DESC, n.token_id ASC
            LIMIT ? OFFSET ?
        """
        args.extend([cap, start])
        rows: list[dict[str, Any]] = []
        with self._lock:
            with closing(self._connect()) as conn:
                for row in conn.execute(sql, args):
                    listing = None
                    if row["listing_token_id"] is not None:
                        listing = {
                            "token_id": str(row["listing_token_id"]),
                            "seller": str(row["listing_seller"]),
                            "price": int(row["listing_price"]),
                            "listed_height": int(row["listed_height"]),
                            "listed_txid": str(row["listed_txid"]),
                        }
                    rows.append(
                        {
                            "token_id": str(row["token_id"]),
                            "creator": str(row["creator"]),
                            "owner": str(row["owner"]),
                            "metadata_uri": str(row["metadata_uri"]),
                            "mint_height": int(row["mint_height"]),
                            "mint_txid": str(row["mint_txid"]),
                            "updated_height": int(row["updated_height"]),
                            "updated_txid": str(row["updated_txid"]),
                            "listed": listing is not None,
                            "listing": listing,
                        }
                    )
        return rows

    def list_listings(
        self,
        *,
        seller: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), 1000))
        start = max(0, int(offset))
        clauses: list[str] = []
        args: list[Any] = []
        if seller:
            clauses.append("l.seller = ?")
            args.append(str(seller).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                l.token_id,
                l.seller,
                l.price,
                l.listed_height,
                l.listed_txid,
                n.owner AS current_owner,
                n.metadata_uri
            FROM listings l
            LEFT JOIN nfts n ON n.token_id = l.token_id
            {where}
            ORDER BY l.listed_height DESC, l.token_id ASC
            LIMIT ? OFFSET ?
        """
        args.extend([cap, start])
        rows: list[dict[str, Any]] = []
        with self._lock:
            with closing(self._connect()) as conn:
                for row in conn.execute(sql, args):
                    rows.append(
                        {
                            "token_id": str(row["token_id"]),
                            "seller": str(row["seller"]),
                            "price": int(row["price"]),
                            "listed_height": int(row["listed_height"]),
                            "listed_txid": str(row["listed_txid"]),
                            "current_owner": str(row["current_owner"] or ""),
                            "metadata_uri": str(row["metadata_uri"] or ""),
                        }
                    )
        return rows

    def list_contracts(
        self,
        *,
        owner: str | None = None,
        template: str | None = None,
        contract_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), 1000))
        start = max(0, int(offset))
        clauses: list[str] = []
        args: list[Any] = []
        if owner:
            clauses.append("owner = ?")
            args.append(str(owner).strip())
        if template:
            clauses.append("template = ?")
            args.append(str(template).strip())
        if contract_id:
            clauses.append("contract_id = ?")
            args.append(str(contract_id).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                contract_id,
                template,
                owner,
                created_height,
                created_txid,
                updated_height,
                updated_txid,
                state_json
            FROM contracts
            {where}
            ORDER BY updated_height DESC, contract_id ASC
            LIMIT ? OFFSET ?
        """
        args.extend([cap, start])
        rows: list[dict[str, Any]] = []
        with self._lock:
            with closing(self._connect()) as conn:
                for row in conn.execute(sql, args):
                    state: dict[str, Any]
                    try:
                        decoded = json.loads(str(row["state_json"]))
                    except Exception:
                        decoded = {}
                    if isinstance(decoded, dict):
                        state = decoded
                    else:
                        state = {}
                    rows.append(
                        {
                            "contract_id": str(row["contract_id"]),
                            "template": str(row["template"]),
                            "owner": str(row["owner"]),
                            "created_height": int(row["created_height"]),
                            "created_txid": str(row["created_txid"]),
                            "updated_height": int(row["updated_height"]),
                            "updated_txid": str(row["updated_txid"]),
                            "state": state,
                        }
                    )
        return rows
