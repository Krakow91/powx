from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from powx.chain import Chain, ValidationError
from powx.market_backend import MarketplaceBackend
from powx.market_indexer import MarketIndexer, MarketIndexerError
from powx.mnemonic import backup_challenge_positions, verify_backup_challenge
from powx.models import Transaction
from powx.p2p import (
    NetworkError,
    P2PNode,
    add_peer_to_node,
    api_balance,
    api_chain,
    api_contracts,
    api_create_transaction,
    api_history,
    api_nft_listings,
    api_nfts,
    api_mempool,
    api_mine,
    api_status,
    get_node_status,
    submit_block,
    submit_transaction,
    trigger_node_sync,
)
from powx.wallet import Wallet, create_seed_wallet, create_wallet, load_wallet, save_wallet


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


def _resolve_wallet_password(args: argparse.Namespace, prompt_if_missing: bool = False) -> str | None:
    direct = str(getattr(args, "wallet_password", "") or "").strip()
    if direct:
        return direct

    env_name = str(getattr(args, "wallet_password_env", "") or "").strip()
    if env_name:
        value = os.environ.get(env_name)
        if value is None:
            raise ValidationError(f"Wallet password environment variable '{env_name}' is not set")
        if not value:
            raise ValidationError(f"Wallet password environment variable '{env_name}' is empty")
        return value

    wants_prompt = bool(getattr(args, "wallet_password_prompt", False))
    if wants_prompt or prompt_if_missing:
        try:
            entered = getpass.getpass("Wallet password: ")
        except Exception as exc:
            raise ValidationError(f"Failed to read wallet password: {exc}") from exc
        if not entered:
            raise ValidationError("Wallet password must not be empty")
        return entered

    return None


def _load_wallet_with_args(
    path: str | Path,
    args: argparse.Namespace,
    prompt_on_encrypted: bool = True,
) -> Wallet:
    password = _resolve_wallet_password(args, prompt_if_missing=False)
    try:
        return load_wallet(path, password=password)
    except ValueError as exc:
        msg = str(exc)
        if prompt_on_encrypted and (not password) and "Encrypted wallet requires password" in msg:
            prompted = _resolve_wallet_password(args, prompt_if_missing=True)
            return load_wallet(path, password=prompted)
        raise ValidationError(msg) from exc


def _add_wallet_password_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wallet-password", help="Wallet password for encrypted wallet files")
    parser.add_argument("--wallet-password-env", help="Environment variable containing wallet password")
    parser.add_argument("--wallet-password-prompt", action="store_true", help="Prompt for wallet password")


def _read_tx_file(path: str | Path) -> Transaction:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ValidationError(f"Failed to read transaction file '{source}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValidationError("Transaction file must contain a JSON object")
    try:
        return Transaction.from_dict(data)
    except Exception as exc:
        raise ValidationError(f"Invalid transaction JSON: {exc}") from exc


def _write_json_file(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _parse_positions_csv(raw: str) -> list[int]:
    text = raw.strip()
    if not text:
        raise ValidationError("Positions cannot be empty")
    values: list[int] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            raise ValidationError("Positions list contains an empty value")
        try:
            value = int(token)
        except ValueError as exc:
            raise ValidationError(f"Invalid position value: '{token}'") from exc
        values.append(value)
    return values


def _parse_words_csv(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        raise ValidationError("Words cannot be empty")
    values: list[str] = []
    for chunk in text.split(","):
        token = chunk.strip().lower()
        if not token:
            raise ValidationError("Words list contains an empty value")
        values.append(token)
    return values


def cmd_wallet_new(args: argparse.Namespace) -> None:
    wallet = create_seed_wallet() if bool(args.seed) else create_wallet()
    password: str | None = None
    if bool(args.encrypt):
        password = _resolve_wallet_password(args, prompt_if_missing=True)

    try:
        save_wallet(wallet, args.out, password=password, kdf=args.kdf)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    print(f"Wallet created: {args.out}")
    summary = {
        "address": wallet.address,
        "public_key": wallet.public_key,
        "mnemonic_words": len(wallet.mnemonic.split()) if wallet.mnemonic else 0,
        "encrypted": bool(password),
        "kdf": args.kdf if password else "",
    }
    print(json.dumps(summary, indent=2))


def cmd_wallet_address(args: argparse.Namespace) -> None:
    wallet = _load_wallet_with_args(args.wallet, args)
    print(wallet.address)


def cmd_init(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir, must_exist=False)
    if chain.chain:
        raise ValidationError("Chain is already initialized")

    if args.genesis_wallet:
        wallet = _load_wallet_with_args(args.genesis_wallet, args)
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
        address = _load_wallet_with_args(args.wallet, args).address
    else:
        address = args.address

    print(chain.balance_of(address))


def _miner_address(args: argparse.Namespace) -> str:
    if args.wallet:
        return _load_wallet_with_args(args.wallet, args).address
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
    wallet = _load_wallet_with_args(args.wallet, args)

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


def cmd_tx_build_offline(args: argparse.Namespace) -> None:
    chain = _load_chain(args.data_dir)

    if args.from_wallet:
        sender_pubkey = _load_wallet_with_args(args.from_wallet, args).public_key
    else:
        sender_pubkey = str(args.from_pubkey).strip().lower()
        if not sender_pubkey:
            raise ValidationError("Sender pubkey is required")

    tx = chain.create_unsigned_transaction(
        sender_pubkey=sender_pubkey,
        to_address=args.to,
        amount=args.amount,
        fee=args.fee,
    )
    _write_json_file(args.out, tx.to_dict())
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(Path(args.out)),
                "inputs": len(tx.inputs),
                "outputs": len(tx.outputs),
                "unsigned": True,
            },
            indent=2,
        )
    )


def cmd_tx_sign_offline(args: argparse.Namespace) -> None:
    unsigned_tx = _read_tx_file(args.tx_in)
    wallet = _load_wallet_with_args(args.wallet, args)

    chain = Chain(Path("."))
    signed_tx = chain.sign_transaction(unsigned_tx, private_key_hex=wallet.private_key)
    _write_json_file(args.tx_out, signed_tx.to_dict())

    print(
        json.dumps(
            {
                "ok": True,
                "txid": signed_tx.txid,
                "out": str(Path(args.tx_out)),
                "inputs": len(signed_tx.inputs),
                "outputs": len(signed_tx.outputs),
            },
            indent=2,
        )
    )


def cmd_tx_send_signed(args: argparse.Namespace) -> None:
    tx = _read_tx_file(args.tx)
    ttl = int(args.broadcast_ttl)

    if args.node:
        try:
            result = submit_transaction(args.node, tx, ttl=ttl)
        except NetworkError as exc:
            raise ValidationError(f"Transaction broadcast failed: {exc}") from exc
        print(json.dumps({"ok": True, "mode": "node", "result": result}, indent=2))
        return

    chain = _load_chain(args.data_dir)
    chain.add_transaction(tx)
    result: dict[str, object] = {
        "ok": True,
        "mode": "local-chain",
        "txid": tx.txid,
    }
    if args.broadcast_node:
        try:
            broadcast = submit_transaction(args.broadcast_node, tx, ttl=ttl)
        except NetworkError as exc:
            raise ValidationError(f"Transaction broadcast failed: {exc}") from exc
        result["broadcast_result"] = broadcast

    print(json.dumps(result, indent=2))


def cmd_seed_backup_challenge(args: argparse.Namespace) -> None:
    wallet = _load_wallet_with_args(args.wallet, args)
    mnemonic = wallet.mnemonic.strip()
    if not mnemonic:
        raise ValidationError("Wallet has no seed phrase saved")

    try:
        positions = backup_challenge_positions(mnemonic, count=args.count)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps({"ok": True, "positions": positions}, indent=2))


def cmd_seed_backup_verify(args: argparse.Namespace) -> None:
    wallet = _load_wallet_with_args(args.wallet, args)
    mnemonic = wallet.mnemonic.strip()
    if not mnemonic:
        raise ValidationError("Wallet has no seed phrase saved")

    positions = _parse_positions_csv(args.positions)
    words = _parse_words_csv(args.words)
    ok = verify_backup_challenge(mnemonic, positions=positions, provided_words=words)
    print(json.dumps({"ok": ok, "positions": positions}, indent=2))


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
        address = _load_wallet_with_args(args.wallet, args).address
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
        address = _load_wallet_with_args(args.wallet, args).address
    else:
        address = args.address

    try:
        result = api_history(args.node, address=address, limit=args.limit, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_nfts(args: argparse.Namespace) -> None:
    token_id = str(args.token_id).strip() if args.token_id else None
    try:
        result = api_nfts(args.node, token_id=token_id, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_nft_listings(args: argparse.Namespace) -> None:
    try:
        result = api_nft_listings(args.node, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_contracts(args: argparse.Namespace) -> None:
    contract_id = str(args.contract_id).strip() if args.contract_id else None
    try:
        result = api_contracts(args.node, contract_id=contract_id, timeout=args.timeout)
    except NetworkError as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps(result, indent=2))


def cmd_api_send(args: argparse.Namespace) -> None:
    wallet = _load_wallet_with_args(args.wallet, args)
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


def cmd_market_sync(args: argparse.Namespace) -> None:
    try:
        indexer = MarketIndexer(
            db_path=args.db,
            node_url=args.node,
            timeout=args.timeout,
        )
        result = indexer.sync_once()
    except (NetworkError, MarketIndexerError) as exc:
        raise ValidationError(str(exc)) from exc
    print(json.dumps({"ok": True, "db": str(Path(args.db)), "result": result}, indent=2))


def cmd_market_run(args: argparse.Namespace) -> None:
    backend = MarketplaceBackend(
        node_url=args.node,
        db_path=args.db,
        host=args.host,
        port=args.port,
        sync_interval=args.sync_interval,
        request_timeout=args.timeout,
        static_dir=args.static_dir,
        auto_sync=not bool(args.no_auto_sync),
    )
    print(
        json.dumps(
            {
                "service_url": backend.base_url,
                "node_url": backend.node_url,
                "db": str(Path(args.db)),
                "sync_interval_seconds": args.sync_interval,
                "auto_sync": not bool(args.no_auto_sync),
                "static_dir": str(backend.static_dir),
            },
            indent=2,
        )
    )
    print("Market web backend running. Press Ctrl+C to stop.")

    try:
        backend.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="powx",
        description="Fast PoW cryptocurrency prototype (Bitcoin-like, educational).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    wallet_new = subparsers.add_parser("wallet-new", help="Create a new wallet")
    wallet_new.add_argument("--out", default="wallet.json", help="Wallet output file")
    wallet_new.add_argument("--seed", action="store_true", help="Create mnemonic-backed wallet")
    wallet_new.add_argument("--kdf", choices=["scrypt", "argon2id"], default="scrypt", help="Wallet file KDF")
    wallet_new.add_argument("--encrypt", dest="encrypt", action="store_true", default=True, help="Encrypt wallet file")
    wallet_new.add_argument("--no-encrypt", dest="encrypt", action="store_false", help="Store wallet file in plaintext")
    _add_wallet_password_args(wallet_new)
    wallet_new.set_defaults(func=cmd_wallet_new)

    wallet_address = subparsers.add_parser("wallet-address", help="Print wallet address")
    wallet_address.add_argument("--wallet", required=True, help="Wallet file path")
    _add_wallet_password_args(wallet_address)
    wallet_address.set_defaults(func=cmd_wallet_address)

    init = subparsers.add_parser("init", help="Initialize chain state")
    init.add_argument("--data-dir", default="./data", help="Data directory")
    init.add_argument("--genesis-wallet", help="Wallet file for genesis funds")
    init.add_argument("--genesis-address", help="Address for genesis funds")
    init.add_argument("--genesis-supply", type=int, default=0, help="Genesis coin supply (default: 0 for no premine)")
    _add_wallet_password_args(init)
    init.set_defaults(func=cmd_init)

    status = subparsers.add_parser("status", help="Show chain status")
    status.add_argument("--data-dir", default="./data", help="Data directory")
    status.set_defaults(func=cmd_status)

    balance = subparsers.add_parser("balance", help="Show address balance")
    balance.add_argument("--data-dir", default="./data", help="Data directory")
    source = balance.add_mutually_exclusive_group(required=True)
    source.add_argument("--address", help="Address")
    source.add_argument("--wallet", help="Wallet file")
    _add_wallet_password_args(balance)
    balance.set_defaults(func=cmd_balance)

    mine = subparsers.add_parser("mine", help="Mine new blocks")
    mine.add_argument("--data-dir", default="./data", help="Data directory")
    mine.add_argument("--blocks", type=int, default=1, help="Number of blocks to mine")
    mine.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto", help="Mining backend")
    mine.add_argument("--broadcast-node", help="Optional node URL to broadcast mined blocks (e.g. http://127.0.0.1:8844)")
    miner = mine.add_mutually_exclusive_group(required=True)
    miner.add_argument("--address", help="Miner address")
    miner.add_argument("--wallet", help="Miner wallet file")
    _add_wallet_password_args(mine)
    mine.set_defaults(func=cmd_mine)

    send = subparsers.add_parser("send", help="Create and broadcast a transaction")
    send.add_argument("--data-dir", default="./data", help="Data directory")
    send.add_argument("--wallet", required=True, help="Sender wallet file")
    send.add_argument("--to", required=True, help="Recipient address")
    send.add_argument("--amount", type=int, required=True, help="Amount")
    send.add_argument("--fee", type=int, help="Optional fee, default is minimum fee")
    send.add_argument("--broadcast-node", help="Optional node URL to broadcast transaction")
    _add_wallet_password_args(send)
    send.set_defaults(func=cmd_send)

    tx_build_offline = subparsers.add_parser("tx-build-offline", help="Build unsigned transaction for offline signing")
    tx_build_offline.add_argument("--data-dir", default="./data", help="Data directory")
    tx_build_source = tx_build_offline.add_mutually_exclusive_group(required=True)
    tx_build_source.add_argument("--from-wallet", help="Wallet file to derive sender public key")
    tx_build_source.add_argument("--from-pubkey", help="Sender compressed public key (hex)")
    tx_build_offline.add_argument("--to", required=True, help="Recipient address")
    tx_build_offline.add_argument("--amount", type=int, required=True, help="Amount")
    tx_build_offline.add_argument("--fee", type=int, help="Optional fee")
    tx_build_offline.add_argument("--out", default="unsigned_tx.json", help="Unsigned transaction output file")
    _add_wallet_password_args(tx_build_offline)
    tx_build_offline.set_defaults(func=cmd_tx_build_offline)

    tx_sign_offline = subparsers.add_parser("tx-sign-offline", help="Sign unsigned transaction using local wallet key")
    tx_sign_offline.add_argument("--tx-in", required=True, help="Unsigned transaction JSON")
    tx_sign_offline.add_argument("--wallet", required=True, help="Signer wallet file")
    tx_sign_offline.add_argument("--tx-out", default="signed_tx.json", help="Signed transaction output file")
    _add_wallet_password_args(tx_sign_offline)
    tx_sign_offline.set_defaults(func=cmd_tx_sign_offline)

    tx_send_signed = subparsers.add_parser("tx-send-signed", help="Submit a pre-signed transaction")
    tx_send_signed.add_argument("--tx", required=True, help="Signed transaction JSON")
    tx_send_signed.add_argument("--data-dir", default="./data", help="Data directory for local mempool submission")
    tx_send_signed.add_argument("--node", help="Optional node URL for direct network submission")
    tx_send_signed.add_argument("--broadcast-node", help="Optional node URL to broadcast after local add")
    tx_send_signed.add_argument("--broadcast-ttl", type=int, default=2, help="Forwarding hops for peers")
    tx_send_signed.set_defaults(func=cmd_tx_send_signed)

    seed_backup_challenge = subparsers.add_parser(
        "seed-backup-challenge",
        help="Generate random seed-word positions for backup checks",
    )
    seed_backup_challenge.add_argument("--wallet", required=True, help="Wallet file path")
    seed_backup_challenge.add_argument("--count", type=int, default=3, help="Number of challenge positions")
    _add_wallet_password_args(seed_backup_challenge)
    seed_backup_challenge.set_defaults(func=cmd_seed_backup_challenge)

    seed_backup_verify = subparsers.add_parser(
        "seed-backup-verify",
        help="Verify selected seed words against stored wallet mnemonic",
    )
    seed_backup_verify.add_argument("--wallet", required=True, help="Wallet file path")
    seed_backup_verify.add_argument("--positions", required=True, help="Comma-separated 1-based positions, e.g. 2,7,11")
    seed_backup_verify.add_argument("--words", required=True, help="Comma-separated seed words matching positions")
    _add_wallet_password_args(seed_backup_verify)
    seed_backup_verify.set_defaults(func=cmd_seed_backup_verify)

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
    _add_wallet_password_args(api_balance_cmd)
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
    _add_wallet_password_args(api_history_cmd)
    api_history_cmd.set_defaults(func=cmd_api_history)

    api_nfts_cmd = subparsers.add_parser("api-nfts", help="Read NFT state via node REST API")
    api_nfts_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_nfts_cmd.add_argument("--token-id", help="Optional token id filter")
    api_nfts_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_nfts_cmd.set_defaults(func=cmd_api_nfts)

    api_nft_listings_cmd = subparsers.add_parser("api-nft-listings", help="Read NFT listings via node REST API")
    api_nft_listings_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_nft_listings_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_nft_listings_cmd.set_defaults(func=cmd_api_nft_listings)

    api_contracts_cmd = subparsers.add_parser("api-contracts", help="Read smart contract state via node REST API")
    api_contracts_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_contracts_cmd.add_argument("--contract-id", help="Optional contract id filter")
    api_contracts_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    api_contracts_cmd.set_defaults(func=cmd_api_contracts)

    api_send_cmd = subparsers.add_parser("api-send", help="Create and broadcast tx through node REST API")
    api_send_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    api_send_cmd.add_argument("--wallet", required=True, help="Sender wallet file")
    api_send_cmd.add_argument("--to", required=True, help="Recipient address")
    api_send_cmd.add_argument("--amount", type=int, required=True, help="Amount")
    api_send_cmd.add_argument("--fee", type=int, help="Optional fee")
    api_send_cmd.add_argument("--broadcast-ttl", type=int, default=2, help="Forwarding hops for peers")
    api_send_cmd.add_argument("--timeout", type=float, default=6.0, help="Request timeout in seconds")
    _add_wallet_password_args(api_send_cmd)
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
    _add_wallet_password_args(api_mine_cmd)
    api_mine_cmd.set_defaults(func=cmd_api_mine)

    market_sync_cmd = subparsers.add_parser("market-sync", help="Sync marketplace index from node API")
    market_sync_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    market_sync_cmd.add_argument("--db", default="./market/market_index.db", help="SQLite index database path")
    market_sync_cmd.add_argument("--timeout", type=float, default=4.0, help="Request timeout in seconds")
    market_sync_cmd.set_defaults(func=cmd_market_sync)

    market_run_cmd = subparsers.add_parser("market-run", help="Run marketplace web backend + frontend")
    market_run_cmd.add_argument("--node", default="http://127.0.0.1:8844", help="Node URL")
    market_run_cmd.add_argument("--db", default="./market/market_index.db", help="SQLite index database path")
    market_run_cmd.add_argument("--host", default="127.0.0.1", help="Bind host")
    market_run_cmd.add_argument("--port", type=int, default=8950, help="Bind port")
    market_run_cmd.add_argument("--sync-interval", type=float, default=5.0, help="Background sync interval in seconds")
    market_run_cmd.add_argument("--timeout", type=float, default=4.0, help="Node request timeout in seconds")
    market_run_cmd.add_argument("--static-dir", help="Optional custom directory for web frontend assets")
    market_run_cmd.add_argument("--no-auto-sync", action="store_true", help="Disable background auto-sync thread")
    market_run_cmd.set_defaults(func=cmd_market_run)

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
