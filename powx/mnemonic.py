from __future__ import annotations

import hashlib
import secrets

from .crypto import N


_ONSETS = ("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v")
_VOWELS = ("a", "e", "i", "o")
_MIDDLES = ("b", "d", "f", "g", "k", "l", "m", "n")
_TAILS = ("na", "ra", "ta", "ko")


def _build_wordlist() -> tuple[str, ...]:
    words: list[str] = []
    for onset in _ONSETS:
        for vowel in _VOWELS:
            for middle in _MIDDLES:
                for tail in _TAILS:
                    words.append(f"{onset}{vowel}{middle}{tail}")
    return tuple(words)


WORDLIST = _build_wordlist()
WORD_TO_INDEX = {word: idx for idx, word in enumerate(WORDLIST)}

WORDS_PER_MNEMONIC = 12
ENTROPY_BYTES = 16
BITS_PER_WORD = 11
BACKUP_WORDS_PER_MNEMONIC = 24
PRIVATE_KEY_BYTES = 32
PRIVATE_KEY_CHECKSUM_BITS = 8


def normalize_mnemonic(mnemonic: str) -> str:
    return " ".join(mnemonic.strip().lower().split())


def _bytes_to_bits(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def _bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit string length must be a multiple of 8")
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def _word_indexes_to_bits(words: list[str]) -> str:
    bits = ""
    for word in words:
        if word not in WORD_TO_INDEX:
            raise ValueError(f"Unknown mnemonic word: {word}")
        bits += f"{WORD_TO_INDEX[word]:011b}"
    return bits


def entropy_to_mnemonic(entropy: bytes) -> str:
    if len(entropy) != ENTROPY_BYTES:
        raise ValueError(f"Entropy must be {ENTROPY_BYTES} bytes")

    checksum_bits_count = len(entropy) * 8 // 32
    checksum = _bytes_to_bits(hashlib.sha256(entropy).digest())[:checksum_bits_count]
    payload = _bytes_to_bits(entropy) + checksum

    words: list[str] = []
    for offset in range(0, len(payload), BITS_PER_WORD):
        index = int(payload[offset : offset + BITS_PER_WORD], 2)
        words.append(WORDLIST[index])
    return " ".join(words)


def mnemonic_to_entropy(mnemonic: str) -> bytes:
    normalized = normalize_mnemonic(mnemonic)
    words = normalized.split(" ")
    if len(words) != WORDS_PER_MNEMONIC:
        raise ValueError(f"Mnemonic must contain {WORDS_PER_MNEMONIC} words")

    bits = _word_indexes_to_bits(words)

    checksum_bits_count = ENTROPY_BYTES * 8 // 32
    entropy_bits = bits[: ENTROPY_BYTES * 8]
    checksum_bits = bits[ENTROPY_BYTES * 8 :]

    if len(checksum_bits) != checksum_bits_count:
        raise ValueError("Invalid mnemonic payload size")

    entropy = _bits_to_bytes(entropy_bits)
    expected_checksum = _bytes_to_bits(hashlib.sha256(entropy).digest())[:checksum_bits_count]
    if checksum_bits != expected_checksum:
        raise ValueError("Mnemonic checksum mismatch")

    return entropy


def generate_mnemonic() -> str:
    return entropy_to_mnemonic(secrets.token_bytes(ENTROPY_BYTES))


def validate_mnemonic(mnemonic: str) -> bool:
    try:
        _ = private_key_from_mnemonic(mnemonic)
        return True
    except Exception:
        return False


def private_key_from_mnemonic(mnemonic: str, passphrase: str = "") -> str:
    normalized = normalize_mnemonic(mnemonic)
    words = normalized.split(" ")

    if len(words) == WORDS_PER_MNEMONIC:
        _ = mnemonic_to_entropy(normalized)
        salt = f"kk91-mnemonic-{passphrase}".encode("utf-8")
        seed = hashlib.pbkdf2_hmac("sha512", normalized.encode("utf-8"), salt, 4096, dklen=64)
        value = int.from_bytes(seed[:32], "big")
        key = (value % (N - 1)) + 1
        return f"{key:064x}"

    if len(words) == BACKUP_WORDS_PER_MNEMONIC:
        return private_key_from_backup_mnemonic(normalized)

    raise ValueError(
        f"Mnemonic must contain {WORDS_PER_MNEMONIC} words (new wallet) or "
        f"{BACKUP_WORDS_PER_MNEMONIC} words (wallet backup)"
    )


def private_key_to_backup_mnemonic(private_key_hex: str) -> str:
    key_hex = private_key_hex.strip().lower()
    if len(key_hex) != PRIVATE_KEY_BYTES * 2:
        raise ValueError("Private key must be 64 hex chars")
    try:
        key_bytes = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise ValueError("Private key contains invalid hex characters") from exc

    checksum = _bytes_to_bits(hashlib.sha256(key_bytes).digest())[:PRIVATE_KEY_CHECKSUM_BITS]
    payload = _bytes_to_bits(key_bytes) + checksum

    words: list[str] = []
    for offset in range(0, len(payload), BITS_PER_WORD):
        index = int(payload[offset : offset + BITS_PER_WORD], 2)
        words.append(WORDLIST[index])

    if len(words) != BACKUP_WORDS_PER_MNEMONIC:
        raise ValueError("Backup mnemonic generation failed")

    return " ".join(words)


def private_key_from_backup_mnemonic(mnemonic: str) -> str:
    normalized = normalize_mnemonic(mnemonic)
    words = normalized.split(" ")
    if len(words) != BACKUP_WORDS_PER_MNEMONIC:
        raise ValueError(f"Backup mnemonic must contain {BACKUP_WORDS_PER_MNEMONIC} words")

    bits = _word_indexes_to_bits(words)
    key_bits = bits[: PRIVATE_KEY_BYTES * 8]
    checksum_bits = bits[PRIVATE_KEY_BYTES * 8 :]

    if len(checksum_bits) != PRIVATE_KEY_CHECKSUM_BITS:
        raise ValueError("Backup mnemonic payload size mismatch")

    key_bytes = _bits_to_bytes(key_bits)
    expected_checksum = _bytes_to_bits(hashlib.sha256(key_bytes).digest())[:PRIVATE_KEY_CHECKSUM_BITS]
    if checksum_bits != expected_checksum:
        raise ValueError("Backup mnemonic checksum mismatch")

    return key_bytes.hex()


def mnemonic_words(mnemonic: str) -> list[str]:
    normalized = normalize_mnemonic(mnemonic)
    words = normalized.split(" ")
    if len(words) not in {WORDS_PER_MNEMONIC, BACKUP_WORDS_PER_MNEMONIC}:
        raise ValueError(
            f"Mnemonic must contain {WORDS_PER_MNEMONIC} words (new wallet) or "
            f"{BACKUP_WORDS_PER_MNEMONIC} words (wallet backup)"
        )
    return words


def backup_challenge_positions(mnemonic: str, count: int = 3) -> list[int]:
    words = mnemonic_words(mnemonic)
    if count <= 0:
        raise ValueError("Backup challenge count must be positive")
    if count > len(words):
        raise ValueError("Backup challenge count exceeds mnemonic length")

    indexes = list(range(1, len(words) + 1))
    chosen = secrets.SystemRandom().sample(indexes, count)
    chosen.sort()
    return chosen


def mnemonic_words_for_positions(mnemonic: str, positions: list[int]) -> list[str]:
    words = mnemonic_words(mnemonic)
    expected: list[str] = []
    for position in positions:
        if position <= 0 or position > len(words):
            raise ValueError(f"Mnemonic position out of range: {position}")
        expected.append(words[position - 1])
    return expected


def verify_backup_challenge(mnemonic: str, positions: list[int], provided_words: list[str]) -> bool:
    if len(positions) != len(provided_words):
        return False
    try:
        expected = mnemonic_words_for_positions(mnemonic, positions)
    except ValueError:
        return False

    normalized_provided: list[str] = []
    for word in provided_words:
        normalized = normalize_mnemonic(word)
        parts = normalized.split(" ") if normalized else []
        if len(parts) != 1:
            return False
        normalized_provided.append(parts[0])

    return expected == normalized_provided
