from dataclasses import dataclass


@dataclass(frozen=True)
class ChainConfig:
    symbol: str = "KK91"
    chain_id: str = "kk91-gpu-main"
    protocol_version: int = 1
    protocol_upgrade_v2_height: int = 2_147_483_647
    target_schedule: str = "asert-v3"
    target_block_time: int = 30
    difficulty_window: int = 30
    asert_half_life: int = 30 * 60
    mtp_window: int = 17
    halving_interval: int = 210_000
    initial_block_reward: int = 50
    max_total_supply: int = 911_000_000
    min_tx_fee: int = 1
    # Easier-than-Bitcoin target, but not instant on most CPUs.
    max_target: int = 2**252
    initial_target: int = 2**240
    max_adjust_factor_up: float = 2.0
    max_adjust_factor_down: float = 0.2
    max_transactions_per_block: int = 500
    max_mempool_transactions: int = 4000
    max_mempool_virtual_bytes: int = 8_000_000
    min_mempool_fee_rate: float = 0.0
    mempool_ancestor_limit: int = 25
    mempool_descendant_limit: int = 25
    mempool_rbf_enabled: bool = True
    mempool_cpfp_enabled: bool = True
    max_rbf_replacements: int = 100
    min_rbf_fee_delta: int = 1
    min_rbf_feerate_delta: float = 0.0
    max_tx_inputs: int = 64
    max_tx_outputs: int = 64
    max_future_block_seconds: int = 2 * 60 * 60
    max_block_timestamp_step_seconds: int = 24 * 60 * 60
    max_mempool_tx_age_seconds: int = 48 * 60 * 60
    max_future_tx_seconds: int = 2 * 60
    max_reorg_depth: int = 12
    pow_algorithm: str = "kkhash-v1"
    gpu_batch_size: int = 131072


CONFIG = ChainConfig()
