from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.crypto import address_from_public_key, private_key_to_public_key


PRIV_A = "1" * 64
ADDR_A = address_from_public_key(private_key_to_public_key(PRIV_A).hex(), "KK91")


ASERT_TEST_CONFIG = replace(
    CONFIG,
    consensus_lock_enabled=False,
    chain_id="kk91-difficulty-test",
    target_schedule="asert-v3",
    initial_target=2**240,
    max_target=2**252,
    target_block_time=30,
    asert_half_life=300,
    max_adjust_factor_up=4.0,
    max_adjust_factor_down=0.25,
    max_block_timestamp_step_seconds=3_600,
)


class DifficultyTimeRulesTest(unittest.TestCase):
    def test_empty_chain_uses_runtime_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            self.assertEqual(chain.status()["target_schedule"], "asert-v3")

    def test_default_schedule_is_asert_for_new_chains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)
            self.assertEqual(chain.status()["target_schedule"], "asert-v3")

    def test_asert_target_adjusts_smoothly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)

            prev = chain.tip
            self.assertIsNotNone(prev)
            assert prev is not None

            on_time = chain.next_target(prev.timestamp + 30)
            faster = chain.next_target(prev.timestamp + 15)
            slower = chain.next_target(prev.timestamp + 60)

            self.assertLess(faster, on_time)
            self.assertLess(on_time, slower)
            self.assertGreaterEqual(faster, int(prev.target * ASERT_TEST_CONFIG.max_adjust_factor_down))
            self.assertLessEqual(slower, int(prev.target * ASERT_TEST_CONFIG.max_adjust_factor_up))

    def test_timestamp_regression_and_jump_bounds_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)
            with mock.patch.object(chain, "_now", return_value=1_700_000_030):
                chain.mine_block(ADDR_A, mining_backend="cpu")
            with mock.patch.object(chain, "_now", return_value=1_700_000_060):
                chain.mine_block(ADDR_A, mining_backend="cpu")

            prev = chain.tip
            self.assertIsNotNone(prev)
            assert prev is not None
            mtp = chain.median_time_past(count=chain._mtp_window_for_schedule())

            with self.assertRaisesRegex(ValidationError, "timestamp regresses"):
                chain._validate_block_timestamp(
                    prev.timestamp - 1,
                    median_past=mtp,
                    context="Block",
                    prev_timestamp=prev.timestamp,
                )

            with mock.patch.object(
                chain,
                "_now",
                return_value=prev.timestamp,
            ):
                with self.assertRaisesRegex(ValidationError, "timestamp jump is too large"):
                    chain._validate_block_timestamp(
                        prev.timestamp + chain.config.max_block_timestamp_step_seconds + 1,
                        median_past=mtp,
                        context="Block",
                        prev_timestamp=prev.timestamp,
                    )

    def test_window_schedule_chain_remains_loadable(self) -> None:
        window_cfg = replace(ASERT_TEST_CONFIG, target_schedule="window-v2")
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=window_cfg)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)
            with mock.patch.object(chain, "_now", return_value=1_700_000_030):
                chain.mine_block(ADDR_A, mining_backend="cpu")

            reloaded = Chain(td, config=ASERT_TEST_CONFIG)
            reloaded.load()
            self.assertEqual(reloaded.status()["target_schedule"], "window-v2")
            self.assertEqual(reloaded.height, 1)

    def test_load_migrates_bugged_genesis_only_window_state_to_asert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)

            state_path = chain.state_path
            with state_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            raw["config"]["target_schedule"] = "window-v2"
            with state_path.open("w", encoding="utf-8") as handle:
                json.dump(raw, handle, indent=2)

            reloaded = Chain(td, config=ASERT_TEST_CONFIG)
            reloaded.load()
            self.assertEqual(reloaded.status()["target_schedule"], "asert-v3")

            with state_path.open("r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(str(persisted.get("config", {}).get("target_schedule")), "asert-v3")

    def test_load_infers_asert_when_state_omits_schedule_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            chain = Chain(td, config=ASERT_TEST_CONFIG)
            with mock.patch.object(chain, "_now", return_value=1_700_000_000):
                chain.initialize(ADDR_A, genesis_supply=0)
            for timestamp in (1_700_000_015, 1_700_000_045, 1_700_000_105, 1_700_000_120):
                with mock.patch.object(chain, "_now", return_value=timestamp):
                    chain.mine_block(ADDR_A, mining_backend="cpu")

            state_path = chain.state_path
            with state_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            raw_config = raw.get("config", {})
            if isinstance(raw_config, dict):
                raw_config.pop("target_schedule", None)
                raw_config.pop("difficulty_window", None)
            with state_path.open("w", encoding="utf-8") as handle:
                json.dump(raw, handle, indent=2)

            reloaded = Chain(td, config=ASERT_TEST_CONFIG)
            reloaded.load()
            self.assertEqual(reloaded.height, 4)
            self.assertEqual(reloaded.status()["target_schedule"], "asert-v3")


if __name__ == "__main__":
    unittest.main()
