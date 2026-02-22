from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONFIG
from .crypto import address_from_public_key, generate_private_key_hex, private_key_to_public_key
from .mnemonic import generate_mnemonic, normalize_mnemonic, private_key_from_mnemonic

try:  # pragma: no cover - optional dependency
    from argon2.low_level import Type as Argon2Type
    from argon2.low_level import hash_secret_raw as argon2_hash_secret_raw
except Exception:  # pragma: no cover - optional dependency
    Argon2Type = None
    argon2_hash_secret_raw = None


ENCRYPTED_WALLET_FORMAT = "kk91-wallet-encrypted-v2"
ENCRYPTED_WALLET_CIPHER = "xor-hmac-sha256"

_KDF_DKLEN = 64
_NONCE_BYTES = 16
_SALT_BYTES = 16
_DEFAULT_SCRYPT_N = 1 << 14
_DEFAULT_SCRYPT_R = 8
_DEFAULT_SCRYPT_P = 1
_DEFAULT_ARGON2_TIME_COST = 3
_DEFAULT_ARGON2_MEMORY_COST = 65536
_DEFAULT_ARGON2_PARALLELISM = 1


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


def _normalize_kdf_name(kdf: str) -> str:
    normalized = str(kdf).strip().lower()
    if normalized in {"scrypt"}:
        return "scrypt"
    if normalized in {"argon2", "argon2id"}:
        return "argon2id"
    raise ValueError("Unsupported wallet KDF. Use 'scrypt' or 'argon2id'")


def is_argon2_available() -> bool:
    return argon2_hash_secret_raw is not None and Argon2Type is not None


def _default_kdf_params(kdf_name: str) -> dict[str, int]:
    if kdf_name == "scrypt":
        return {
            "n": _DEFAULT_SCRYPT_N,
            "r": _DEFAULT_SCRYPT_R,
            "p": _DEFAULT_SCRYPT_P,
        }
    if kdf_name == "argon2id":
        return {
            "time_cost": _DEFAULT_ARGON2_TIME_COST,
            "memory_cost": _DEFAULT_ARGON2_MEMORY_COST,
            "parallelism": _DEFAULT_ARGON2_PARALLELISM,
        }
    raise ValueError(f"Unsupported wallet KDF: {kdf_name}")


def _derive_key(password: str, salt: bytes, kdf_name: str, params: dict[str, Any]) -> bytes:
    if not password:
        raise ValueError("Wallet password must not be empty")

    if kdf_name == "scrypt":
        try:
            n = int(params.get("n", _DEFAULT_SCRYPT_N))
            r = int(params.get("r", _DEFAULT_SCRYPT_R))
            p = int(params.get("p", _DEFAULT_SCRYPT_P))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid scrypt wallet KDF parameters") from exc

        if n <= 1 or r <= 0 or p <= 0:
            raise ValueError("Invalid scrypt wallet KDF parameters")

        return hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=_KDF_DKLEN,
        )

    if kdf_name == "argon2id":
        if not is_argon2_available():
            raise ValueError("Argon2 wallet encryption is unavailable. Install 'argon2-cffi'")

        try:
            time_cost = int(params.get("time_cost", _DEFAULT_ARGON2_TIME_COST))
            memory_cost = int(params.get("memory_cost", _DEFAULT_ARGON2_MEMORY_COST))
            parallelism = int(params.get("parallelism", _DEFAULT_ARGON2_PARALLELISM))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Argon2 wallet KDF parameters") from exc

        if time_cost <= 0 or memory_cost <= 0 or parallelism <= 0:
            raise ValueError("Invalid Argon2 wallet KDF parameters")

        return argon2_hash_secret_raw(
            secret=password.encode("utf-8"),
            salt=salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=_KDF_DKLEN,
            type=Argon2Type.ID,
        )

    raise ValueError(f"Unsupported wallet KDF: {kdf_name}")


def _xor_keystream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    output = bytearray(len(data))
    cursor = 0
    counter = 0
    while cursor < len(data):
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        chunk_len = min(len(block), len(data) - cursor)
        for offset in range(chunk_len):
            output[cursor + offset] = data[cursor + offset] ^ block[offset]
        cursor += chunk_len
        counter += 1
    return bytes(output)


def _wallet_plaintext(wallet: Wallet) -> bytes:
    return json.dumps(wallet.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encrypt_wallet_payload(wallet: Wallet, password: str, kdf: str) -> dict[str, Any]:
    kdf_name = _normalize_kdf_name(kdf)
    params = _default_kdf_params(kdf_name)
    salt = secrets.token_bytes(_SALT_BYTES)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    key_material = _derive_key(password, salt=salt, kdf_name=kdf_name, params=params)
    enc_key = key_material[:32]
    mac_key = key_material[32:]

    plaintext = _wallet_plaintext(wallet)
    ciphertext = _xor_keystream(plaintext, key=enc_key, nonce=nonce)
    mac_hex = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).hexdigest()

    return {
        "wallet_format": ENCRYPTED_WALLET_FORMAT,
        "encrypted": True,
        "kdf": {
            "name": kdf_name,
            "salt": salt.hex(),
            "params": params,
        },
        "cipher": {
            "name": ENCRYPTED_WALLET_CIPHER,
            "nonce": nonce.hex(),
        },
        "ciphertext": ciphertext.hex(),
        "mac": mac_hex,
    }


def _is_encrypted_wallet_payload(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and str(data.get("wallet_format", "")).strip().lower() == ENCRYPTED_WALLET_FORMAT
        and bool(data.get("encrypted", False))
    )


def _decrypt_wallet_payload(payload: dict[str, Any], password: str) -> Wallet:
    if not password:
        raise ValueError("Encrypted wallet requires password")

    cipher = payload.get("cipher")
    if not isinstance(cipher, dict):
        raise ValueError("Invalid encrypted wallet: missing cipher section")

    kdf_data = payload.get("kdf")
    if not isinstance(kdf_data, dict):
        raise ValueError("Invalid encrypted wallet: missing kdf section")

    kdf_name = _normalize_kdf_name(str(kdf_data.get("name", "scrypt")))
    params = kdf_data.get("params")
    if not isinstance(params, dict):
        raise ValueError("Invalid encrypted wallet: invalid kdf params")

    try:
        salt = bytes.fromhex(str(kdf_data.get("salt", "")))
        nonce = bytes.fromhex(str(cipher.get("nonce", "")))
        ciphertext = bytes.fromhex(str(payload.get("ciphertext", "")))
    except ValueError as exc:
        raise ValueError("Invalid encrypted wallet: non-hex payload fields") from exc

    if len(salt) < 8:
        raise ValueError("Invalid encrypted wallet: bad salt")
    if len(nonce) != _NONCE_BYTES:
        raise ValueError("Invalid encrypted wallet: bad nonce")
    if str(cipher.get("name", "")).strip().lower() != ENCRYPTED_WALLET_CIPHER:
        raise ValueError("Unsupported encrypted wallet cipher")

    key_material = _derive_key(password, salt=salt, kdf_name=kdf_name, params=params)
    enc_key = key_material[:32]
    mac_key = key_material[32:]

    expected_mac = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).hexdigest()
    supplied_mac = str(payload.get("mac", "")).strip().lower()
    if not hmac.compare_digest(expected_mac, supplied_mac):
        raise ValueError("Wallet password is incorrect or wallet file was modified")

    plaintext = _xor_keystream(ciphertext, key=enc_key, nonce=nonce)
    try:
        decoded = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Encrypted wallet payload could not be decoded") from exc

    if not isinstance(decoded, dict):
        raise ValueError("Encrypted wallet payload is invalid")
    return Wallet.from_dict(decoded)


def wallet_file_requires_password(path: str | Path) -> bool:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _is_encrypted_wallet_payload(data)


def save_wallet(wallet: Wallet, path: str | Path, password: str | None = None, kdf: str = "scrypt") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if password is None:
        payload: dict[str, Any] = wallet.to_dict()
    else:
        payload = _encrypt_wallet_payload(wallet, password=password, kdf=kdf)

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_wallet(path: str | Path, password: str | None = None) -> Wallet:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if _is_encrypted_wallet_payload(data):
        return _decrypt_wallet_payload(data, password=password or "")

    if not isinstance(data, dict):
        raise ValueError("Invalid wallet file format")
    return Wallet.from_dict(data)


def address_from_private_key(private_key: str, symbol: str = CONFIG.symbol) -> str:
    public_key = private_key_to_public_key(private_key).hex()
    return address_from_public_key(public_key, symbol)
