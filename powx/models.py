from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class TxInput:
    txid: str
    index: int
    signature: str = ""
    pubkey: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "txid": self.txid,
            "index": self.index,
            "signature": self.signature,
            "pubkey": self.pubkey,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TxInput":
        return cls(
            txid=data["txid"],
            index=int(data["index"]),
            signature=data.get("signature", ""),
            pubkey=data.get("pubkey", ""),
        )


@dataclass
class TxOutput:
    amount: int
    address: str

    def to_dict(self) -> dict[str, Any]:
        return {"amount": self.amount, "address": self.address}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TxOutput":
        return cls(amount=int(data["amount"]), address=data["address"])


@dataclass
class Transaction:
    version: int
    timestamp: int
    nonce: int
    inputs: list[TxInput] = field(default_factory=list)
    outputs: list[TxOutput] = field(default_factory=list)
    txid: str = ""

    def to_dict(self, include_txid: bool = True) -> dict[str, Any]:
        data = {
            "version": self.version,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "inputs": [tx_in.to_dict() for tx_in in self.inputs],
            "outputs": [tx_out.to_dict() for tx_out in self.outputs],
        }
        if include_txid:
            data["txid"] = self.txid
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transaction":
        return cls(
            version=int(data["version"]),
            timestamp=int(data["timestamp"]),
            nonce=int(data["nonce"]),
            inputs=[TxInput.from_dict(item) for item in data.get("inputs", [])],
            outputs=[TxOutput.from_dict(item) for item in data.get("outputs", [])],
            txid=data.get("txid", ""),
        )

    def is_coinbase(self) -> bool:
        return len(self.inputs) == 0

    def signing_payload(self) -> bytes:
        payload = {
            "version": self.version,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "inputs": [
                {
                    "txid": tx_in.txid,
                    "index": tx_in.index,
                    "pubkey": tx_in.pubkey,
                }
                for tx_in in self.inputs
            ],
            "outputs": [tx_out.to_dict() for tx_out in self.outputs],
        }
        return canonical_json(payload).encode("utf-8")

    def signing_hash(self) -> bytes:
        return hashlib.sha256(self.signing_payload()).digest()

    def compute_txid(self) -> str:
        payload = self.to_dict(include_txid=False)
        return sha256_hex(canonical_json(payload).encode("utf-8"))


@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: int
    target: int
    nonce: int
    merkle_root: str
    miner: str
    transactions: list[Transaction]
    chain_work: int
    block_hash: str = ""

    def header_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "target": str(self.target),
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "miner": self.miner,
        }

    def header_bytes(self) -> bytes:
        return canonical_json(self.header_dict()).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "target": str(self.target),
            "nonce": self.nonce,
            "merkle_root": self.merkle_root,
            "miner": self.miner,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "chain_work": str(self.chain_work),
            "block_hash": self.block_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        return cls(
            index=int(data["index"]),
            prev_hash=data["prev_hash"],
            timestamp=int(data["timestamp"]),
            target=int(data["target"]),
            nonce=int(data["nonce"]),
            merkle_root=data["merkle_root"],
            miner=data["miner"],
            transactions=[Transaction.from_dict(item) for item in data.get("transactions", [])],
            chain_work=int(data.get("chain_work", "0")),
            block_hash=data.get("block_hash", ""),
        )


def merkle_root(txids: list[str]) -> str:
    if not txids:
        return sha256_hex(b"")

    nodes = txids[:]
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        next_level: list[str] = []
        for i in range(0, len(nodes), 2):
            pair = bytes.fromhex(nodes[i]) + bytes.fromhex(nodes[i + 1])
            next_level.append(sha256_hex(pair))
        nodes = next_level
    return nodes[0]


def block_work(target: int) -> int:
    if target <= 0:
        return 0
    return (1 << 256) // (target + 1)
