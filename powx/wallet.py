from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG
from .crypto import address_from_public_key, generate_private_key_hex, private_key_to_public_key
from .mnemonic import generate_mnemonic, normalize_mnemonic, private_key_from_mnemonic


@dataclass
class Wallet:
    private_key: str
    public_key: str
    address: str
    mnemonic: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {
            "private_key": self.private_key,
            "public_key": self.public_key,
            "address": self.address,
        }
        if self.mnemonic:
            data["mnemonic"] = self.mnemonic
        return data

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Wallet":
        return cls(
            private_key=data["private_key"],
            public_key=data["public_key"],
            address=data["address"],
            mnemonic=data.get("mnemonic", ""),
        )


def wallet_from_private_key(private_key: str, symbol: str = CONFIG.symbol, mnemonic: str = "") -> Wallet:
    public_key = private_key_to_public_key(private_key).hex()
    address = address_from_public_key(public_key, symbol)
    return Wallet(private_key=private_key, public_key=public_key, address=address, mnemonic=mnemonic)


def create_wallet(symbol: str = CONFIG.symbol) -> Wallet:
    private_key = generate_private_key_hex()
    return wallet_from_private_key(private_key, symbol=symbol)


def create_seed_wallet(symbol: str = CONFIG.symbol) -> Wallet:
    mnemonic = generate_mnemonic()
    private_key = private_key_from_mnemonic(mnemonic)
    return wallet_from_private_key(private_key, symbol=symbol, mnemonic=mnemonic)


def recover_wallet_from_seed(mnemonic: str, symbol: str = CONFIG.symbol) -> Wallet:
    normalized = normalize_mnemonic(mnemonic)
    private_key = private_key_from_mnemonic(normalized)
    return wallet_from_private_key(private_key, symbol=symbol, mnemonic=normalized)


def save_wallet(wallet: Wallet, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(wallet.to_dict(), handle, indent=2)


def load_wallet(path: str | Path) -> Wallet:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Wallet.from_dict(data)


def address_from_private_key(private_key: str, symbol: str = CONFIG.symbol) -> str:
    public_key = private_key_to_public_key(private_key).hex()
    return address_from_public_key(public_key, symbol)
