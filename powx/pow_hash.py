from __future__ import annotations

import hashlib
from typing import Iterable

from .models import Block, canonical_json


MASK64 = (1 << 64) - 1

CONST_A = 0x9E3779B185EBCA87
CONST_B = 0xC2B2AE3D27D4EB4F
CONST_C = 0x165667B19E3779F9
CONST_D = 0x85EBCA77C2B2AE63


def _rotl64(value: int, shift: int) -> int:
    shift &= 63
    return ((value << shift) & MASK64) | (value >> (64 - shift))


def seed_words_from_block(block: Block, chain_id: str) -> tuple[int, int, int, int, int, int, int, int]:
    payload = canonical_json(
        {
            "chain_id": chain_id,
            "index": block.index,
            "prev_hash": block.prev_hash,
            "timestamp": block.timestamp,
            "target": str(block.target),
            "merkle_root": block.merkle_root,
            "miner": block.miner,
        }
    ).encode("utf-8")
    seed = hashlib.sha512(payload).digest()
    words = tuple(int.from_bytes(seed[i : i + 8], "big") for i in range(0, 64, 8))
    return words  # type: ignore[return-value]


def kkhash_words(
    seed_words: tuple[int, int, int, int, int, int, int, int],
    nonce: int,
) -> tuple[int, int, int, int]:
    x = (nonce & MASK64) ^ seed_words[4]
    y = seed_words[5] ^ CONST_A
    z = seed_words[6] ^ CONST_B
    w = seed_words[7] ^ CONST_C

    for round_index in range(32):
        key = seed_words[round_index & 7]
        x = (x + _rotl64(y ^ key, (round_index % 23) + 5)) & MASK64
        y = (y ^ _rotl64((z + key + CONST_A) & MASK64, (round_index % 19) + 7)) & MASK64
        z = (z + (w ^ key ^ CONST_B) + ((x * CONST_C) & MASK64)) & MASK64
        w = _rotl64((w + (x ^ y) + CONST_D) & MASK64, (round_index % 17) + 11)

        x ^= x >> 29
        y ^= y >> 31
        z ^= z >> 33
        w ^= w >> 27

    h0 = (x ^ seed_words[0] ^ _rotl64(z, 17)) & MASK64
    h1 = (y ^ seed_words[1] ^ _rotl64(w, 29)) & MASK64
    h2 = (z ^ seed_words[2] ^ _rotl64(x, 41)) & MASK64
    h3 = (w ^ seed_words[3] ^ _rotl64(y, 53)) & MASK64
    return h0, h1, h2, h3


def words_to_bytes(words: Iterable[int]) -> bytes:
    return b"".join((word & MASK64).to_bytes(8, "big") for word in words)


def hash_from_seed_words(
    seed_words: tuple[int, int, int, int, int, int, int, int],
    nonce: int,
) -> bytes:
    return words_to_bytes(kkhash_words(seed_words, nonce))


def hash_int_from_seed_words(
    seed_words: tuple[int, int, int, int, int, int, int, int],
    nonce: int,
) -> int:
    return int.from_bytes(hash_from_seed_words(seed_words, nonce), "big")


def target_to_words(target: int) -> tuple[int, int, int, int]:
    if target < 0:
        raise ValueError("Target must be non-negative")
    target = target & ((1 << 256) - 1)
    raw = target.to_bytes(32, "big")
    words = tuple(int.from_bytes(raw[i : i + 8], "big") for i in range(0, 32, 8))
    return words  # type: ignore[return-value]
