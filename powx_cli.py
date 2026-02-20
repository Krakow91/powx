from __future__ import annotations

import argparse
import json
from pathlib import Path

from powx.chain import Chain, ValidationError
from powx.p2p import (
    NetworkError,
    P2PNode,
    add_peer_to_node,
    api_balance,
    api_chain,
    api_create_transaction,
    api_history,
    api_mempool,
    api_mine,
    api_status,
    get_node_status,
    submit_block,
    submit_transaction,
    trigger_node_sync,
)
from powx.wallet import create_wallet, load_wallet, save_wallet


def _load_chain(data_dir: str, must_exist: bool = True) -> Chain:
    chain = Chain(data_dir)
    if chain.exists():
        chain.load()
        return chain

    if must_exist:
        raise ValidationError(
            f"Chain state not found in '{data_dir}'. Run 'init' first."
        )

    return chain


def cmd_wallet_new(args: argparse.Namespace) -> None:
    wallet = create_wallet()
    save_wallet(wallet, args.out)
    print(f"Wallet created: {args.out}")
    print(json.dumps(wallet.to_dict(), indent=2))


def cmd_wallet_address(args: argparse.Namespace) -> None:
    wallet = load_wallet(args.wallet)
    print(wallet.address)


def cmd_init(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir, must_exist=False)
    if chain.chain:
        raise ValidationError("Chain is already initialized")

    if args.genesis_wallet:
        wallet = load_wallet(args.genesis_wallet)
    elif args.genesis_address:
        wallet = None
    else:
        auto_wallet_path = Path(args.data_dir) / "genesis_wallet.json"
        wallet = create_wallet()
        save_wallet(wallet, auto_wallet_path)
        print(f"Generated genesis wallet at: {auto_wallet_path}")

    genesis_address = wallet.address if wallet else args.genesis_address
    block = chain.initialize(genesis_address, genesis_supply=args.genesis_supply)

    print("Chain initialized")
    print(json.dumps({"height": block.index, "hash": block.block_hash, "address": genesis_address}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)
    print(json.dumps(chain.status(), indent=2))


def cmd_balance(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)

    if args.wallet:
        address = load_wallet(args.wallet).address
    else:
        address = args.address

    print(chain.balance_of(address))


def _miner_address(args: argparse.Namespace) -> str:
    if args.wallet:
        return load_wallet(args.wallet).address
    return args.address


def cmd_mine(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)
    miner_address = _miner_address(args)

    for _ in range(args.blocks):
        block = chain.mine_block(miner_address, mining_backend=args.backend)
        if args.broadcast_node:
            try:
                submit_block(args.broadcast_node, block)
            except NetworkError as exc:
                raise ValidationError(f"Block broadcast failed: {exc}") from exc
        print(
            json.dumps(
                {
                    "height": block.index,
                    "hash": block.block_hash,
                    "reward": sum(out.amount for out in block.transactions[0].outputs),
                    "tx_count": len(block.transactions),
                    "backend": args.backend,
                    "broadcast_node": args.broadcast_node or "",
                }
            )
        )


def cmd_send(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)
    wallet = load_wallet(args.wallet)

    tx = chain.create_transaction(
        private_key_hex=wallet.private_key,
        to_address=args.to,
        amount=args.amount,
        fee=args.fee,
    )
    chain.add_transaction(tx)
    if args.broadcast_node:
        try:
            submit_transaction(args.broadcast_node, tx)
        except NetworkError as exc:
            raise ValidationError(f"Transaction broadcast failed: {exc}") from exc

    print(
        json.dumps(
            {
                "txid": tx.txid,
                "inputs": len(tx.inputs),
                "outputs": len(tx.outputs),
                "broadcast_node": args.broadcast_node or "",
            },
            indent=2,
        )
    )


def cmd_chain(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)

    start = max(0, len(chain.chain) - args.limit)
    rows = []
    for block in chain.chain[start:]:
        rows.append(
            {
                "height": block.index,
                "hash": block.block_hash,
                "prev_hash": block.prev_hash,
                "tx_count": len(block.transactions),
                "target": block.target,
                "timestamp": block.timestamp,
            }
        )
    print(json.dumps(rows, indent=2))


def cmd_mempool(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)
    rows = [
        {
            "txid": tx.txid,
            "inputs": len(tx.inputs),
            "outputs": len(tx.outputs),
        }
        for tx in chain.mempool
    ]
    print(json.dumps(rows, indent=2))


def cmd_node_run(args: argparse.Namespace) -> None:
    node = P2PNode(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        advertise_host=args.advertise_host,
        peers=args.peer,
        sync_interval=args.sync_interval,
        request_timeout=args.request_timeout,
        max_peer_count=args.max_peers,
        max_inbound_ttl=args.max_inbound_ttl,
        max_request_body_bytes=args.max_request_body_bytes,
        max_requests_per_minute=args.max_requests_per_minute,
        sync_retry_cooldown=args.sync_retry_cooldown,
        peer_ban_threshold=args.peer_ban_threshold,
        peer_ban_seconds=args.peer_ban_seconds,
        peer_penalty_invalid_tx=args.peer_penalty_invalid_tx,
        peer_penalty_invalid_block=args.peer_penalty_invalid_block,
        peer_penalty_bad_sync=args.peer_penalty_bad_sync,
        peer_reward_success=args.peer_reward_success,
        peer_auth_max_skew_seconds=args.peer_auth_max_skew_seconds,
        peer_auth_replay_window_seconds=args.peer_auth_replay_window_seconds,
    )
    print(json.dumps({"node": node.node_url, "data_dir": str(Path(args.data_dir)), "peers": node.get_peers()}, indent=2))
    print("Node running. Press Ctrl+C to stop.")

    try:
        node.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


def cmd_node_status(args: argparse.Namespace) -> None:
    try:
        status = get_node_status(args.node, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(status, indent=2))


def cmd_node_add_peer(args: argparse.Namespace) -> None:
    try:
        result = add_peer_to_node(args.node, args.peer, sync_now=not args.no_sync, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_node_sync(args: argparse.Namespace) -> None:
    try:
        result = trigger_node_sync(args.node, peer=args.peer, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_status(args: argparse.Namespace) -> None:
    try:
        result = api_status(args.node, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_balance(args: argparse.Namespace) -> None:
    if args.wallet:
        address = load_wallet(args.wallet).address
    else:
        address = args.address

    try:
        result = api_balance(args.node, address=address, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_chain(args: argparse.Namespace) -> None:
    try:
        result = api_chain(args.node, limit=args.limit, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_mempool(args: argparse.Namespace) -> None:
    try:
        result = api_mempool(args.node, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_history(args: argparse.Namespace) -> None:
    if args.wallet:
        address = load_wallet(args.wallet).address
    else:
        address = args.address

    try:
        result = api_history(args.node, address=address, limit=args.limit, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_send(args: argparse.Namespace) -> None:
    wallet = load_wallet(args.wallet)
    try:
        result = api_create_transaction(
            args.node,
            private_key_hex=wallet.private_key,
            to_address=args.to,
            amount=args.amount,
            fee=args.fee,
            broadcast_ttl=args.broadcast_ttl,
            timeout=args.timeout,
        )
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_mine(args: argparse.Namespace) -> None:
    miner_address = _miner_address(args)
    timeout = None if args.timeout <= 0 else float(args.timeout)
    try:
        result = api_mine(
            args.node,
            miner_address=miner_address,
            blocks=args.blocks,
            backend=args.backend,
            broadcast_ttl=args.broadcast_ttl,
            timeout=timeout,
        )
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powx",
        description="Fast PoW cryptocurrency prototype (Bitcoin-like, educational).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    wallet_new = subparsers.add_parser("wallet-new", help="Create a new wallet")
    wallet_new.add_argument("--out", default="wallet.json", help="Wallet output file")
    wallet_new.set_defaults(func=cmd_wallet_new)

    wallet_address = subparsers.add_parser("wallet-address", help="Print wallet address")
    wallet_address.add_argument("--wallet", required=True, help="Wallet file path")
    wallet_address.set_defaults(func=cmd_wallet_address)

    init = subparsers.add_parser("init", help="Initialize chain state")
    init.add_argument("--data-dir", default="./data", help="Data directory")
    init.add_argument("--genesis-wallet", help="Wallet file for genesis funds")
    init.add_argument("--genesis-address", help="Address for genesis funds")
    init.add_argument("--genesis-supply", type=int, default=0, help="Genesis coin supply (default: 0 for no premine)")
    init.set_defaults(func=cmd_init)

    status = subparsers.add_parser("status", help="Show chain status")
    status.add_argument("--data-dir", default="./data", help="Data directory")
    status.set_defaults(func=cmd_status)

    balance = subparsers.add_parser("balance", help="Show address balance")
    balance.add_argument("--data-dir", default="./data", help="Data directory")
    source = balance.add_mutually_exclusive_group(required=True)
    source.add_argument("--address", help="Address")
    source.add_argument("--wallet", help="Wallet file")
    balance.set_defaults(func=cmd_balance)

    mine = subparsers.add_parser("mine", help="Mine new blocks")
    mine.add_argument("--data-dir", default="./data", help="Data directory")
    mine.add_argument("--blocks", type=int, default=1, help="Number of blocks to mine")
    mine.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto", help="Mining backend")
    mine.add_argument("--broadcast-node", help="Optional node URL to broadcast mined blocks (e.g. http://127.0.0.1:8844)")
    miner = mine.add_mutually_exclusive_group(required=True)
    miner.add_argument("--address", help="Miner address")
    miner.add_argument("--wallet", help="Miner wallet file")
    mine.set_defaults(func=cmd_mine)

    send = subparsers.add_parser("send", help="Create and broadcast a transaction")
    send.add_argument("--data-dir", default="./data", help="Data directory")
    send.add_argument("--wallet", required=True, help="Sender wallet file")
    send.add_argument("--to", required=True, help="Recipient address")
    send.add_argument("--amount", type=int, required=True, help="Amount")
    send.add_argument("--fee", type=int, help="Optional fee, default is minimum fee")
    send.add_argument("--broadcast-node", help="Optional node URL to broadcast transaction")
    send.set_defaults(func=cmd_send)

    chain_cmd = subparsers.add_parser("chain", help="Show blocks")
    chain_cmd.add_argument("--data-dir", default="./data", help="Data directory")
    chain_cmd.add_argument("--limit", type=int, default=20, help="How many latest blocks")
    chain_cmd.set_defaults(func=cmd_chain)

    mempool = subparsers.add_parser("mempool", help="Show mempool")
    mempool.add_argument("--data-dir", default="./data", help="Data directory")
    mempool.set_defaults(func=cmd_mempool)

    node_run = subparsers.add_parser("node-run", help="Run a P2P node server")
    node_run.add_argument("--data-dir", default="./data", help="Node data directory")
    node_run.add_argument("--host", default="127.0.0.1", help="Bind host")
    node_run.add_argument("--port", type=int, default=8844, help="Bind port")
    node_run.add_argument("--advertise-host", help="Advertised host for peers")
    node_run.add_argument("--peer", action="append", default=[], help="Bootstrap peer URL (repeatable)")
    node_run.add_argument("--sync-interval", type=float, default=10.0, help="Automatic sync interval in seconds")
    node_run.add_argument("--request-timeout", type=float, default=4.0, help="Peer request timeout in seconds")
    node_run.add_argument("--max-peers", type=int, default=128, help="Maximum number of remembered peers")
    node_run.add_argument("--max-inbound-ttl", type=int, default=6, help="Maximum accepted inbound broadcast TTL")
    node_run.add_argument("--max-request-body-bytes", type=int, default=256000, help="Maximum request body size in bytes")
    node_run.add_argument("--max-requests-per-minute", type=int, default=360, help="Per-client request cap (HTTP 429 on exceed)")
    node_run.add_argument("--sync-retry-cooldown", type=float, default=2.0, help="Minimum seconds between auto sync-retries per sender")
    node_run.add_argument("--peer-ban-threshold", type=int, default=100, help="Score threshold that triggers temporary peer ban")
    node_run.add_argument("--peer-ban-seconds", type=int, default=900, help="Temporary peer ban duration in seconds")
    node_run.add_argument("--peer-penalty-invalid-tx", type=int, default=20, help="Penalty points for invalid tx from peer")
    node_run.add_argument("--peer-penalty-invalid-block", type=int, default=35, help="Penalty points for invalid block from peer")
    node_run.add_argument("--peer-penalty-bad-sync", type=int, default=15, help="Penalty points for invalid sync data from peer")
    node_run.add_argument("--peer-reward-success", type=int, default=3, help="Score reduction points on successful peer data")
    node_run.add_argument("--peer-auth-max-skew-seconds", type=int, default=120, help="Maximum allowed clock skew for signed peer messages")
    node_run.add_argument("--peer-auth-replay-window-seconds", type=int, default=180, help="Replay window for signed peer message nonces")
    node_run.set_defaults(func=cmd_node_run)

    node_status = subparsers.add_parser("node-status", help="Read status from a running node")
    node_status.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    node_status.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    node_status.set_defaults(func=cmd_node_status)

    node_add_peer = subparsers.add_parser("node-add-peer", help="Add a peer to a running node")
    node_add_peer.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    node_add_peer.add_argument("--peer", required=True, help="Peer URL")
    node_add_peer.add_argument("--no-sync", action="store_true", help="Do not trigger immediate sync after adding")
    node_add_peer.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds")
    node_add_peer.set_defaults(func=cmd_node_add_peer)

    node_sync = subparsers.add_parser("node-sync", help="Trigger node synchronization")
    node_sync.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    node_sync.add_argument("--peer", help="Specific peer URL to sync from")
    node_sync.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    node_sync.set_defaults(func=cmd_node_sync)

    api_status_cmd = subparsers.add_parser("api-status", help="Read REST API status from node")
    api_status_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_status_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_status_cmd.set_defaults(func=cmd_api_status)

    api_balance_cmd = subparsers.add_parser("api-balance", help="Read balance via node REST API")
    api_balance_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_balance_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_balance_source = api_balance_cmd.add_mutually_exclusive_group(required=True)
    api_balance_source.add_argument("--address", help="Address")
    api_balance_source.add_argument("--wallet", help="Wallet file")
    api_balance_cmd.set_defaults(func=cmd_api_balance)

    api_chain_cmd = subparsers.add_parser("api-chain", help="Read chain rows via node REST API")
    api_chain_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_chain_cmd.add_argument("--limit", type=int, default=20, help="How many latest blocks")
    api_chain_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_chain_cmd.set_defaults(func=cmd_api_chain)

    api_mempool_cmd = subparsers.add_parser("api-mempool", help="Read mempool via node REST API")
    api_mempool_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_mempool_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_mempool_cmd.set_defaults(func=cmd_api_mempool)

    api_history_cmd = subparsers.add_parser("api-history", help="Read address history via node REST API")
    api_history_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_history_cmd.add_argument("--limit", type=int, default=120, help="Max history rows")
    api_history_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_history_source = api_history_cmd.add_mutually_exclusive_group(required=True)
    api_history_source.add_argument("--address", help="Address")
    api_history_source.add_argument("--wallet", help="Wallet file")
    api_history_cmd.set_defaults(func=cmd_api_history)

    api_send_cmd = subparsers.add_parser("api-send", help="Create and broadcast tx through node REST API")
    api_send_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_send_cmd.add_argument("--wallet", required=True, help="Sender wallet file")
    api_send_cmd.add_argument("--to", required=True, help="Recipient address")
    api_send_cmd.add_argument("--amount", type=int, required=True, help="Amount")
    api_send_cmd.add_argument("--fee", type=int, help="Optional fee")
    api_send_cmd.add_argument("--broadcast-ttl", type=int, default=2, help="Forwarding hops for peers")
    api_send_cmd.add_argument("--timeout", type=float, default=6.0, help="Request timeout in seconds")
    api_send_cmd.set_defaults(func=cmd_api_send)

    api_mine_cmd = subparsers.add_parser("api-mine", help="Mine blocks through node REST API")
    api_mine_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_mine_cmd.add_argument("--blocks", type=int, default=1, help="Number of blocks to mine")
    api_mine_cmd.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto", help="Mining backend")
    api_mine_cmd.add_argument("--broadcast-ttl", type=int, default=2, help="Forwarding hops for peers")
    api_mine_cmd.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Request timeout in seconds (<=0 disables timeout for long mining jobs)",
    )
    api_miner = api_mine_cmd.add_mutually_exclusive_group(required=True)
    api_miner.add_argument("--address", help="Miner address")
    api_miner.add_argument("--wallet", help="Miner wallet file")
    api_mine_cmd.set_defaults(func=cmd_api_mine)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except ValidationError as exc:
        print(f"Validation error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
