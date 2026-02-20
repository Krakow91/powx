from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
G = (
    55066263022277343669578718895168534326250603453777594175500187360389116729240,
    32670510020758816978083085130507043184471273380659243275938904335757337482424,
)

Point = Optional[tuple[int, int]]


def _mod_inv(value: int, modulus: int) -> int:
    return pow(value, -1, modulus)


def _is_on_curve(point: Point) -> bool:
    if point is None:
        return True
    x, y = point
    return (y * y - (x * x * x + A * x + B)) % P == 0


def _point_add(p1: Point, p2: Point) -> Point:
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % P == 0:
        return None

    if p1 == p2:
        slope = ((3 * x1 * x1 + A) * _mod_inv((2 * y1) % P, P)) % P
    else:
        slope = ((y2 - y1) * _mod_inv((x2 - x1) % P, P)) % P

    x3 = (slope * slope - x1 - x2) % P
    y3 = (slope * (x1 - x3) - y1) % P
    point = (x3, y3)

    if not _is_on_curve(point):
        raise ValueError("Point operation produced invalid curve point")
    return point


def _point_mul(scalar: int, point: Point = G) -> Point:
    if scalar % N == 0 or point is None:
        return None

    scalar = scalar % N
    result: Point = None
    addend: Point = point

    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1

    return result


def _deterministic_k(private_key: int, message_hash: bytes) -> int:
    x = private_key.to_bytes(32, "big")
    h1 = message_hash
    v = b"\x01" * 32
    k = b"\x00" * 32

    k = hmac.new(k, v + b"\x00" + x + h1, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()
    k = hmac.new(k, v + b"\x01" + x + h1, hashlib.sha256).digest()
    v = hmac.new(k, v, hashlib.sha256).digest()

    while True:
        t = b""
        while len(t) < 32:
            v = hmac.new(k, v, hashlib.sha256).digest()
            t += v

        candidate = int.from_bytes(t[:32], "big")
        if 1 <= candidate < N:
            return candidate

        k = hmac.new(k, v + b"\x00", hashlib.sha256).digest()
        v = hmac.new(k, v, hashlib.sha256).digest()


def generate_private_key_hex() -> str:
    value = secrets.randbelow(N - 1) + 1
    return f"{value:064x}"


def private_key_to_public_key(private_key_hex: str) -> bytes:
    private_key = int(private_key_hex, 16)
    if not 1 <= private_key < N:
        raise ValueError("Invalid private key")

    point = _point_mul(private_key, G)
    if point is None:
        raise ValueError("Could not derive public key")
    return compress_public_key(point)


def compress_public_key(point: tuple[int, int]) -> bytes:
    x, y = point
    prefix = 0x02 if y % 2 == 0 else 0x03
    return bytes([prefix]) + x.to_bytes(32, "big")


def decompress_public_key(public_key_bytes: bytes) -> tuple[int, int]:
    if len(public_key_bytes) != 33:
        raise ValueError("Compressed public key must be 33 bytes")

    prefix = public_key_bytes[0]
    if prefix not in (0x02, 0x03):
        raise ValueError("Invalid compressed public key prefix")

    x = int.from_bytes(public_key_bytes[1:], "big")
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)

    if (y % 2 == 0 and prefix == 0x03) or (y % 2 == 1 and prefix == 0x02):
        y = P - y

    point = (x, y)
    if not _is_on_curve(point):
        raise ValueError("Public key is not on secp256k1")
    return point


def sign_digest(private_key_hex: str, message_hash: bytes) -> str:
    if len(message_hash) != 32:
        raise ValueError("Message hash must be 32 bytes")

    private_key = int(private_key_hex, 16)
    if not 1 <= private_key < N:
        raise ValueError("Invalid private key")

    z = int.from_bytes(message_hash, "big")
    k = _deterministic_k(private_key, message_hash)

    while True:
        point = _point_mul(k, G)
        if point is None:
            k = (k + 1) % N
            continue

        r = point[0] % N
        if r == 0:
            k = (k + 1) % N
            continue

        s = (_mod_inv(k, N) * (z + r * private_key)) % N
        if s == 0:
            k = (k + 1) % N
            continue

        # Low-S normalization reduces signature malleability.
        if s > N // 2:
            s = N - s

        return f"{r:064x}{s:064x}"


def verify_signature(public_key_hex: str, message_hash: bytes, signature_hex: str) -> bool:
    try:
        if len(message_hash) != 32 or len(signature_hex) != 128:
            return False

        r = int(signature_hex[:64], 16)
        s = int(signature_hex[64:], 16)
        if not (1 <= r < N and 1 <= s < N):
            return False

        public_key = bytes.fromhex(public_key_hex)
        public_point = decompress_public_key(public_key)

        z = int.from_bytes(message_hash, "big")
        w = _mod_inv(s, N)
        u1 = (z * w) % N
        u2 = (r * w) % N

        point = _point_add(_point_mul(u1, G), _point_mul(u2, public_point))
        if point is None:
            return False

        return point[0] % N == r
    except Exception:
        return False


def pubkey_hash_hex(public_key_hex: str) -> str:
    public_key_bytes = bytes.fromhex(public_key_hex)
    return hashlib.sha256(public_key_bytes).hexdigest()[:40]


def address_from_public_key(public_key_hex: str, symbol: str = "KK91") -> str:
    return f"{symbol}{pubkey_hash_hex(public_key_hex)}"
