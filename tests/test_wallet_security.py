from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key
from powx.mnemonic import (
    backup_challenge_positions,
    mnemonic_words_for_positions,
    private_key_to_backup_mnemonic,
    verify_backup_challenge,
)
from powx.wallet import (
    create_seed_wallet,
    is_argon2_available,
    load_wallet,
    save_wallet,
    wallet_file_requires_password,
)


PRIV_A = "1" * 64
PUB_A = private_key_to_public_key(PRIV_A).hex()
ADDR_A = address_from_public_key(PUB_A, "KK91")

PRIV_B = "2" * 64
PUB_B = private_key_to_public_key(PRIV_B).hex()
ADDR_B = address_from_public_key(PUB_B, "KK91")


class WalletSecurityTest(unittest.TestCase):
    def test_encrypted_wallet_roundtrip_and_wrong_password(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wallet = create_seed_wallet()
            wallet_path = Path(td) / "wallet_encrypted.json"
            save_wallet(wallet, wallet_path, password="test-password", kdf="scrypt")

            raw = wallet_path.read_text(encoding="utf-8")
            self.assertIn("wallet_format", raw)
            self.assertNotIn(wallet.private_key, raw)
            self.assertTrue(wallet_file_requires_password(wallet_path))

            loaded = load_wallet(wallet_path, password="test-password")
            self.assertEqual(loaded.address, wallet.address)
            self.assertEqual(loaded.public_key, wallet.public_key)
            self.assertEqual(loaded.private_key, wallet.private_key)

            with self.assertRaisesRegex(ValueError, "requires password"):
                load_wallet(wallet_path)
            with self.assertRaisesRegex(ValueError, "incorrect|modified"):
                load_wallet(wallet_path, password="wrong-password")

    def test_plaintext_wallet_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wallet = create_seed_wallet()
            wallet_path = Path(td) / "wallet_plain.json"
            save_wallet(wallet, wallet_path)

            loaded = load_wallet(wallet_path)
            self.assertEqual(loaded.address, wallet.address)
            self.assertFalse(wallet_file_requires_password(wallet_path))

    def test_argon2_wallet_support_or_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wallet = create_seed_wallet()
            wallet_path = Path(td) / "wallet_argon2.json"
            if is_argon2_available():
                save_wallet(wallet, wallet_path, password="pw", kdf="argon2id")
                loaded = load_wallet(wallet_path, password="pw")
                self.assertEqual(loaded.address, wallet.address)
            else:
                with self.assertRaisesRegex(ValueError, "Argon2"):
                    save_wallet(wallet, wallet_path, password="pw", kdf="argon2id")

    def test_seed_backup_challenge_verification(self) -> None:
        wallet = create_seed_wallet()
        positions = backup_challenge_positions(wallet.mnemonic, count=3)
        words = mnemonic_words_for_positions(wallet.mnemonic, positions)

        self.assertTrue(verify_backup_challenge(wallet.mnemonic, positions=positions, provided_words=words))
        wrong_words = list(words)
        wrong_words[0] = "invalidword"
        self.assertFalse(verify_backup_challenge(wallet.mnemonic, positions=positions, provided_words=wrong_words))

    def test_seed_backup_challenge_verification_for_24_word_backup(self) -> None:
        backup_mnemonic = private_key_to_backup_mnemonic(PRIV_A)
        positions = backup_challenge_positions(backup_mnemonic, count=4)
        words = mnemonic_words_for_positions(backup_mnemonic, positions)
        self.assertTrue(verify_backup_challenge(backup_mnemonic, positions=positions, provided_words=words))

    def test_offline_build_sign_submit_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(CONFIG, consensus_lock_enabled=False, chain_id="kk91-wallet-test")
            chain = Chain(td, config=cfg)
            chain.initialize(ADDR_A, genesis_supply=1_000)

            unsigned = chain.create_unsigned_transaction(
                sender_pubkey=PUB_A,
                to_address=ADDR_B,
                amount=120,
                fee=5,
            )
            self.assertTrue(all(not tx_input.signature for tx_input in unsigned.inputs))
            self.assertEqual(unsigned.txid, "")

            with self.assertRaisesRegex(ValidationError, "does not match"):
                _ = chain.sign_transaction(unsigned, private_key_hex=PRIV_B)

            signed = chain.sign_transaction(unsigned, private_key_hex=PRIV_A)
            self.assertTrue(signed.txid)
            self.assertTrue(all(tx_input.signature for tx_input in signed.inputs))

            chain.add_transaction(signed)
            self.assertTrue(any(tx.txid == signed.txid for tx in chain.mempool))


if __name__ == "__main__":
    unittest.main()
