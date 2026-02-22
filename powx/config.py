from dataclasses import dataclass


@dataclass(frozen=True)
class ChainConfig:
    symbol: str = "KK91"
    chain_id: str = "kk91-gpu-main"
    protocol_version: int = 1
    # Mainnet consensus lock:
    # - fixed chain_id + fixed genesis template
    # - fixed target schedule (asert-v3)
    # Set to False for local dev/test chains.
    consensus_lock_enabled: bool = True
    # First planned hard-fork activation height (explicit, not "infinite").
    protocol_upgrade_v2_height: int = 840_000
    target_schedule: str = "asert-v3"
    target_block_time: int = 30
    difficulty_window: int = 30
    asert_half_life: int = 30 * 60
    mtp_window: int = 17
    halving_interval: int = 210_000
    # Calibrated for 911,000,000 cap with 210,000-block halvings.
    # Theoretical uncapped subsidy is 911,190,000 and is clipped by max_total_supply.
    initial_block_reward: int = 2_173
    max_total_supply: int = 911_000_000
    # Consensus rule: coinbase outputs require maturity confirmations before spending.
    coinbase_maturity: int = 100
    min_tx_fee: int = 1
    # Easier-than-Bitcoin target, but not instant on most CPUs.
    max_target: int = 2**252
    initial_target: int = 2**240
    max_adjust_factor_up: float = 2.0
    max_adjust_factor_down: float = 0.2
    max_transactions_per_block: int = 500
    max_mempool_transactions: int = 4000
    max_mempool_virtual_bytes: int = 8_000_000
    min_mempool_fee_rate: float = 0.001
    min_standard_tx_fee_rate: float = 0.002
    min_dust_output: int = 2
    max_standard_tx_virtual_bytes: int = 100_000
    smart_contracts_enabled: bool = True
    nft_market_enabled: bool = True
    max_contract_payload_bytes: int = 4_096
    max_contract_kv_entries: int = 1_024
    max_contract_key_bytes: int = 64
    max_contract_value_bytes: int = 1_024
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
    # Locked canonical genesis for mainnet consensus.
    fixed_genesis_hash: str = "00000330c8fe2065d8f358842cee842734c98e6ce46d302440db095297252160"
    fixed_genesis_txid: str = "90ed3f83e3e35b517af2ec9116bba8296d51be31b8a57462cf92e23de2a8c008"
    fixed_genesis_timestamp: int = 1_700_000_000
    fixed_genesis_address: str = "KK915b6b92b37b765963ab61d52a3171a54da33778c1"
    fixed_genesis_supply: int = 0
    fixed_genesis_tx_nonce: int = 1_111_111_111_111_111
    fixed_genesis_block_nonce: int = 15_267


CONFIG = ChainConfig()
