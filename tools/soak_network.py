from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from powx.chain import Chain
from powx.config import CONFIG
from powx.p2p import P2PNode, get_node_status
from powx.wallet import create_wallet


def _start_node(node: P2PNode) -> threading.Thread:
    thread = threading.Thread(target=node.serve_forever, name=f"node-{node.port}", daemon=True)
    thread.start()
    return thread


def _ensure_genesis(node_dir: Path, miner_address: str, genesis_supply: int) -> None:
    chain = Chain(node_dir)
    if chain.exists():
        chain.load()
        if chain.chain:
            return
    chain.initialize(miner_address, genesis_supply=genesis_supply)


def _node_heights(node_urls: list[str]) -> list[int]:
    heights: list[int] = []
    for url in node_urls:
        status = get_node_status(url, timeout=3.0)
        nested = status.get("status")
        if isinstance(nested, dict):
            heights.append(int(nested.get("height", -1)))
        else:
            heights.append(int(status.get("height", -1)))
    return heights


def run_soak(args: argparse.Namespace) -> int:
    root_dir = Path(args.root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    node_count = max(2, int(args.nodes))
    base_port = int(args.base_port)

    miner_wallet = create_wallet(CONFIG.symbol)
    miner_address = miner_wallet.address

    node_dirs = [root_dir / f"node{i + 1}" for i in range(node_count)]
    _ensure_genesis(node_dirs[0], miner_address=miner_address, genesis_supply=int(args.genesis_supply))

    nodes: list[P2PNode] = []
    threads: list[threading.Thread] = []

    try:
        node0_url = f"http://127.0.0.1:{base_port}"
        for idx, node_dir in enumerate(node_dirs):
            port = base_port + idx
            peers: list[str] = []
            if idx > 0:
                peers.append(node0_url)

            node = P2PNode(
                data_dir=node_dir,
                host="127.0.0.1",
                port=port,
                peers=peers,
                sync_interval=float(args.sync_interval),
                request_timeout=4.0,
                peer_ban_threshold=1_000_000,
                peer_ban_seconds=60,
                peer_penalty_invalid_tx=1,
                peer_penalty_invalid_block=1,
                peer_penalty_bad_sync=1,
                peer_reward_success=1,
            )
            nodes.append(node)
            threads.append(_start_node(node))

        # Let node 0 know all other peers for better relay coverage.
        for peer in nodes[1:]:
            try:
                nodes[0].add_peer(peer.node_url, persist=True)
            except Exception:
                continue

        # Bootstrap synchronization.
        time.sleep(1.0)
        for node in nodes[1:]:
            try:
                node.sync_from_peer(nodes[0].node_url)
            except Exception:
                continue

        node_urls = [node.node_url for node in nodes]
        started = time.time()
        next_mine = started
        next_status = started
        heights_seen: list[list[int]] = []
        mined_blocks = 0

        print(f"[soak] running {node_count} nodes for {int(args.duration_seconds)}s under {root_dir}", flush=True)
        print(f"[soak] miner address: {miner_address}", flush=True)

        while True:
            now = time.time()
            if now - started >= float(args.duration_seconds):
                break

            if now >= next_mine:
                result = nodes[0].mine_blocks(
                    miner_address=miner_address,
                    blocks=1,
                    backend=str(args.backend),
                    broadcast_ttl=2,
                )
                mined_blocks += int(result.get("count", 0))
                for follower in nodes[1:]:
                    try:
                        follower.sync_from_peer(nodes[0].node_url)
                    except Exception:
                        continue
                next_mine = now + float(args.mine_every_seconds)

            if now >= next_status:
                heights = _node_heights(node_urls)
                heights_seen.append(heights)
                drift = max(heights) - min(heights)
                print(f"[soak] t+{int(now - started):4d}s heights={heights} drift={drift}", flush=True)
                next_status = now + float(args.status_every_seconds)

            time.sleep(0.2)

        final_heights = _node_heights(node_urls)
        max_drift = max(max(row) - min(row) for row in heights_seen) if heights_seen else 0
        avg_height = mean(final_heights) if final_heights else 0.0

        print("[soak] summary", flush=True)
        print(f"  mined_blocks: {mined_blocks}", flush=True)
        print(f"  final_heights: {final_heights}", flush=True)
        print(f"  max_drift: {max_drift}", flush=True)
        print(f"  avg_height: {avg_height:.2f}", flush=True)
        return 0
    finally:
        for node in nodes:
            try:
                node.shutdown()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local KK91 multi-node soak test.")
    parser.add_argument("--root-dir", default="./soak_run", help="Directory containing node data folders")
    parser.add_argument("--nodes", type=int, default=3, help="Number of local nodes")
    parser.add_argument("--base-port", type=int, default=8944, help="Base TCP port for first node")
    parser.add_argument("--duration-seconds", type=int, default=600, help="Soak test duration")
    parser.add_argument("--mine-every-seconds", type=float, default=20.0, help="Mine cadence on node 1")
    parser.add_argument("--status-every-seconds", type=float, default=10.0, help="Status print cadence")
    parser.add_argument("--sync-interval", type=float, default=3.0, help="Node auto-sync interval")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="cpu", help="Mining backend for soak")
    parser.add_argument("--genesis-supply", type=int, default=0, help="Genesis supply for node 1 bootstrap")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = run_soak(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
