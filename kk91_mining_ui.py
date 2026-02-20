from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from powx.chain import Chain, MiningInterruptedError, ValidationError
from powx.config import CONFIG


class KK91MiningUI(tk.Tk):
    """The-Graph-inspired mining dashboard with robust worker/event handling."""

    def __init__(self) -> None:
        super().__init__()

        self.title("KK91 Mining Nexus")
        self.geometry("1280x820")
        self.minsize(1060, 700)
        self.configure(bg="#0C0B1E")
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

        self.settings_path = Path(__file__).with_name("mining_ui_settings.json")
        settings = self._load_settings()
        wallet_settings = self._load_wallet_ui_settings()

        saved_data_dir = settings.get("data_dir", "").strip()
        wallet_data_dir = wallet_settings.get("data_dir", "").strip()
        if (not saved_data_dir) or (saved_data_dir in {"./kk91_ui_data", "./kk91_data"}):
            default_data_dir = wallet_data_dir or saved_data_dir or "./kk91_data"
        else:
            default_data_dir = saved_data_dir

        self.wallet_var = tk.StringVar(value=settings.get("wallet_address", ""))
        self.data_dir_var = tk.StringVar(value=default_data_dir)
        self.backend_var = tk.StringVar(value=settings.get("backend", "gpu"))

        self.run_state_var = tk.StringVar(value="Stopped")
        self.height_var = tk.StringVar(value="-")
        self.balance_var = tk.StringVar(value="-")
        self.mempool_var = tk.StringVar(value="-")
        self.utxo_var = tk.StringVar(value="-")
        self.target_var = tk.StringVar(value="-")
        self.tip_var = tk.StringVar(value="-")

        self.nonce_var = tk.StringVar(value="0")
        self.attempts_var = tk.StringVar(value="0")
        self.rate_var = tk.StringVar(value="0.0 H/s")
        self.elapsed_var = tk.StringVar(value="0.0s")
        self.preview_var = tk.StringVar(value="-")

        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None

        self.event_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=600)
        self._progress_lock = threading.Lock()
        self._latest_progress: dict[str, object] | None = None
        self._recovery_attempted = False

        self.hashrate_history: list[float] = []

        self._setup_style()
        self._build_layout()
        self._refresh_snapshot()

        self.after(80, self._process_events)
        self.after(2800, self._periodic_refresh)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        midnight = "#0C0B1E"
        panes = "#1A172F"
        tooltip = "#222037"
        purple = "#6F4CFF"
        blue = "#4C66FF"
        turquoise = "#66D8FF"
        red = "#ED4A6D"
        white = "#FFFFFF"
        text_dim = "#AFAEC2"
        line = "#2E2A48"

        style.configure("App.TFrame", background=midnight)
        style.configure("Card.TFrame", background=panes, relief="solid", borderwidth=1)
        style.configure("Panel.TFrame", background=tooltip, relief="solid", borderwidth=1)
        style.configure("Divider.TFrame", background=line)

        style.configure("Title.TLabel", background=midnight, foreground=white, font=("Segoe UI Semibold", 26))
        style.configure("Subtitle.TLabel", background=midnight, foreground=text_dim, font=("Segoe UI", 10))

        style.configure("CardTitle.TLabel", background=panes, foreground=white, font=("Segoe UI Semibold", 13))
        style.configure("Meta.TLabel", background=panes, foreground=text_dim, font=("Segoe UI", 9))
        style.configure("Value.TLabel", background=panes, foreground=turquoise, font=("Consolas", 12, "bold"))
        style.configure("PanelMeta.TLabel", background=tooltip, foreground=text_dim, font=("Segoe UI", 9))
        style.configure("PanelValue.TLabel", background=tooltip, foreground=white, font=("Consolas", 12, "bold"))

        style.configure(
            "Action.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(12, 9),
            foreground=white,
            background=tooltip,
            borderwidth=1,
            focusthickness=1,
            focuscolor=blue,
            bordercolor=line,
            lightcolor=line,
            darkcolor=line,
        )
        style.map(
            "Action.TButton",
            background=[("active", "#2A2742"), ("pressed", "#1A182D"), ("disabled", "#151429")],
            foreground=[("disabled", "#78758F")],
            bordercolor=[("focus", blue)],
            lightcolor=[("focus", blue)],
            darkcolor=[("focus", blue)],
        )

        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(12, 9),
            foreground=white,
            background=purple,
            borderwidth=1,
            focusthickness=1,
            focuscolor=blue,
            bordercolor=purple,
            lightcolor=purple,
            darkcolor=purple,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#7B62FF"), ("pressed", "#5E42E2"), ("disabled", "#2C2750")],
            foreground=[("disabled", "#8681A4")],
            bordercolor=[("focus", blue)],
            lightcolor=[("focus", blue)],
            darkcolor=[("focus", blue)],
        )

        style.configure(
            "Danger.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(12, 9),
            foreground=white,
            background=red,
            borderwidth=1,
            bordercolor=red,
            lightcolor=red,
            darkcolor=red,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#F75E80"), ("pressed", "#D73D61"), ("disabled", "#42263A")],
            foreground=[("disabled", "#9D8292")],
            bordercolor=[("focus", blue)],
            lightcolor=[("focus", blue)],
            darkcolor=[("focus", blue)],
        )

        style.configure(
            "Field.TEntry",
            fieldbackground=tooltip,
            background=tooltip,
            foreground=white,
            bordercolor=line,
            lightcolor=line,
            darkcolor=line,
            insertcolor=turquoise,
            padding=(8, 7),
        )
        style.map(
            "Field.TEntry",
            bordercolor=[("focus", blue)],
            lightcolor=[("focus", blue)],
            darkcolor=[("focus", blue)],
        )

        style.configure(
            "Field.TCombobox",
            fieldbackground=tooltip,
            background=tooltip,
            foreground=white,
            bordercolor=line,
            lightcolor=line,
            darkcolor=line,
            arrowcolor=turquoise,
            padding=(8, 7),
        )
        style.map(
            "Field.TCombobox",
            bordercolor=[("focus", blue)],
            lightcolor=[("focus", blue)],
            darkcolor=[("focus", blue)],
        )

        style.configure(
            "Mining.Horizontal.TProgressbar",
            troughcolor="#2A2742",
            bordercolor="#2A2742",
            background=purple,
            lightcolor=purple,
            darkcolor="#5E42E2",
        )

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        ttk.Label(header, text="KK91 Mining Nexus", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Fast, easy mining controls with live onchain telemetry.",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        ttk.Frame(header, style="Divider.TFrame", height=1).pack(fill="x", pady=(10, 0))

        left = ttk.Frame(root, style="App.TFrame")
        left.grid(row=1, column=0, sticky="ns", padx=(0, 12))

        right = ttk.Frame(root, style="App.TFrame")
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_control_card(left)
        self._build_chain_card(left)

        self._build_live_card(right)
        self._build_chart_and_log_card(right)

    def _build_control_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.pack(fill="x", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Control", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(card, text="Wallet Address", style="Meta.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.wallet_var, style="Field.TEntry").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 9))
        ttk.Label(card, text="Rewards are paid to this address.", style="Meta.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(card, text="Data Directory", style="Meta.TLabel").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.data_dir_var, style="Field.TEntry").grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 9))

        ttk.Label(card, text="Mining Backend", style="Meta.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")
        backend_box = ttk.Combobox(
            card,
            textvariable=self.backend_var,
            values=("gpu", "auto", "cpu"),
            state="readonly",
            style="Field.TCombobox",
        )
        backend_box.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 12))

        self.save_button = ttk.Button(card, text="Save Profile", style="Action.TButton", command=self._save_settings)
        self.save_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        self.refresh_button = ttk.Button(card, text="Refresh", style="Action.TButton", command=self._refresh_snapshot)
        self.refresh_button.grid(row=9, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        self.mine_one_button = ttk.Button(card, text="Mine 1 Block", style="Action.TButton", command=self._mine_one_block)
        self.mine_one_button.grid(row=9, column=1, sticky="ew", padx=(6, 0), pady=(0, 8))

        self.start_button = ttk.Button(card, text="Start Mining", style="Primary.TButton", command=self._start_mining)
        self.start_button.grid(row=10, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ttk.Button(card, text="Stop Mining", style="Danger.TButton", command=self._stop_mining)
        self.stop_button.grid(row=10, column=1, sticky="ew", padx=(6, 0))
        self.stop_button.state(["disabled"])

    def _build_chain_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Chain State", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self._stat_row(card, 1, "State", self.run_state_var)
        self._stat_row(card, 2, "Height", self.height_var)
        self._stat_row(card, 3, "Balance", self.balance_var)
        self._stat_row(card, 4, "Mempool", self.mempool_var)
        self._stat_row(card, 5, "UTXO", self.utxo_var)
        self._stat_row(card, 6, "Target", self.target_var)
        self._stat_row(card, 7, "Tip", self.tip_var)

    def _stat_row(self, parent: ttk.Frame, row: int, label: str, value_var: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="Meta.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=value_var, style="Value.TLabel").grid(row=row, column=1, sticky="e", pady=2)

    def _build_live_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=1)

        ttk.Label(card, text="Live Telemetry", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 9))

        self._live_metric(card, 1, 0, "Nonce", self.nonce_var)
        self._live_metric(card, 1, 1, "Attempts", self.attempts_var)
        self._live_metric(card, 1, 2, "Hashrate", self.rate_var)
        self._live_metric(card, 2, 0, "Elapsed", self.elapsed_var)
        self._live_metric(card, 2, 1, "Hash Preview", self.preview_var)

        self.activity_bar = ttk.Progressbar(card, style="Mining.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.activity_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _live_metric(self, parent: ttk.Frame, row: int, col: int, label: str, var: tk.StringVar) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 8))
        frame.grid(row=row, column=col, sticky="ew", padx=4, pady=3)
        ttk.Label(frame, text=label, style="PanelMeta.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=var, style="PanelValue.TLabel").pack(anchor="w")

    def _build_chart_and_log_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.grid(row=1, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=1)

        ttk.Label(card, text="Hashrate Graph", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, text="Real-time performance stream", style="Meta.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.chart = tk.Canvas(
            card,
            bg="#222037",
            highlightthickness=1,
            highlightbackground="#2E2A48",
            bd=0,
            height=250,
        )
        self.chart.grid(row=2, column=0, sticky="nsew")

        ttk.Label(card, text="Event Log", style="CardTitle.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 6))

        self.log_text = tk.Text(
            card,
            height=10,
            bg="#222037",
            fg="#FFFFFF",
            insertbackground="#66D8FF",
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
            highlightthickness=1,
            highlightbackground="#2E2A48",
            highlightcolor="#4C66FF",
            padx=8,
            pady=6,
        )
        self.log_text.grid(row=4, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _load_settings(self) -> dict[str, str]:
        return self._read_settings_file(self.settings_path)

    def _load_wallet_ui_settings(self) -> dict[str, str]:
        wallet_settings_path = Path(__file__).with_name("wallet_ui_settings.json")
        return self._read_settings_file(wallet_settings_path)

    def _save_settings(self) -> None:
        data = {
            "wallet_address": self.wallet_var.get().strip(),
            "data_dir": self.data_dir_var.get().strip() or "./kk91_data",
            "backend": self.backend_var.get().strip() or "gpu",
        }
        with self.settings_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        self._log("Profile saved")

    @staticmethod
    def _read_settings_file(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
        return {}

    def _validate_address(self, address: str) -> None:
        prefix = CONFIG.symbol
        if not address.startswith(prefix):
            raise ValidationError(f"Address must start with '{prefix}'")
        body = address[len(prefix) :]
        if len(body) != 40:
            raise ValidationError("Address must contain 40 hex chars after prefix")
        try:
            int(body, 16)
        except ValueError as exc:
            raise ValidationError("Address contains invalid characters") from exc

    def _is_mining(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _set_running_state(self, running: bool, backend_label: str = "") -> None:
        if running:
            label = f"Mining ({backend_label})" if backend_label else "Mining"
            self.run_state_var.set(label)
            self.start_button.state(["disabled"])
            self.mine_one_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
        else:
            self.run_state_var.set("Stopped")
            self.start_button.state(["!disabled"])
            self.mine_one_button.state(["!disabled"])
            self.stop_button.state(["disabled"])

    def _start_mining(self) -> None:
        self._start_mining_mode(single_block=False)

    def _mine_one_block(self) -> None:
        self._start_mining_mode(single_block=True)

    def _start_mining_mode(self, single_block: bool) -> None:
        if self._is_mining():
            return

        address = self.wallet_var.get().strip()
        data_dir = Path(self.data_dir_var.get().strip() or "./kk91_data")
        backend = (self.backend_var.get().strip() or "gpu").lower()

        try:
            self._validate_address(address)
        except ValidationError as exc:
            messagebox.showerror("Invalid address", str(exc))
            return

        if backend not in {"gpu", "auto", "cpu"}:
            messagebox.showerror("Invalid backend", "Backend must be gpu, auto or cpu")
            return

        with self._progress_lock:
            self._latest_progress = None

        self._save_settings()
        self.stop_event = threading.Event()
        self._set_running_state(True, backend.upper())

        self.worker_thread = threading.Thread(
            target=self._mining_loop,
            args=(address, data_dir, backend, single_block, self.stop_event),
            daemon=True,
        )
        self.worker_thread.start()

        if single_block:
            self._log(f"Mining started ({backend.upper()}) - single block")
        else:
            self._log(f"Mining started ({backend.upper()})")

    def _stop_mining(self) -> None:
        if self.stop_event is None:
            return
        self.stop_event.set()
        self.run_state_var.set("Stopping...")
        self._log("Stop requested")

    def _enqueue_event(self, event: dict[str, object]) -> None:
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            # Preserve critical events over progress bursts.
            if str(event.get("type", "")) in {"error", "block", "stopped", "genesis"}:
                try:
                    _ = self.event_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.event_queue.put_nowait(event)
                except queue.Full:
                    pass

    def _on_progress(self, stats: dict[str, object]) -> None:
        with self._progress_lock:
            self._latest_progress = stats

    def _mining_loop(
        self,
        address: str,
        data_dir: Path,
        backend: str,
        single_block: bool,
        stop_event: threading.Event,
    ) -> None:
        try:
            chain = Chain(data_dir)
            if chain.exists():
                chain.load()
            else:
                genesis = chain.initialize(address, genesis_supply=1_000_000)
                self._enqueue_event(
                    {
                        "type": "genesis",
                        "height": genesis.index,
                        "hash": genesis.block_hash,
                    }
                )

            while not stop_event.is_set():
                block = chain.mine_block(
                    address,
                    progress_callback=self._on_progress,
                    stop_requested=stop_event.is_set,
                    mining_backend=backend,
                )
                self._enqueue_event(
                    {
                        "type": "block",
                        "height": block.index,
                        "hash": block.block_hash,
                        "reward": sum(out.amount for out in block.transactions[0].outputs),
                    }
                )
                if single_block:
                    break

        except MiningInterruptedError:
            pass
        except ValidationError as exc:
            self._enqueue_event({"type": "error", "message": str(exc)})
        except Exception as exc:
            self._enqueue_event({"type": "error", "message": f"Unexpected error: {exc}"})
        finally:
            self._enqueue_event({"type": "stopped"})

    def _refresh_snapshot(self) -> None:
        data_dir = Path(self.data_dir_var.get().strip() or "./kk91_data")
        address = self.wallet_var.get().strip()

        chain = Chain(data_dir)
        if not chain.exists():
            self.height_var.set("-")
            self.balance_var.set("-")
            self.mempool_var.set("-")
            self.utxo_var.set("-")
            self.target_var.set("-")
            self.tip_var.set("-")
            return

        try:
            chain.load()
            status = chain.status()

            self.height_var.set(str(status["height"]))
            self.mempool_var.set(str(status["mempool_size"]))
            self.utxo_var.set(str(status["utxo_count"]))

            target = status.get("target")
            self.target_var.set(f"{int(target):.3e}" if target is not None else "-")

            tip = status.get("tip_hash")
            self.tip_var.set((str(tip)[:16] + "...") if tip else "-")

            if address:
                try:
                    self.balance_var.set(str(chain.balance_of(address)))
                except Exception:
                    self.balance_var.set("-")
            else:
                self.balance_var.set("-")
            self._recovery_attempted = False
        except Exception as exc:
            if self._try_recover_data_dir(exc):
                self._refresh_snapshot()
                return
            self._log(f"Snapshot refresh failed: {exc}")

    def _find_recoverable_data_dir(self) -> Path | None:
        root = Path(__file__).resolve().parent
        current = Path(self.data_dir_var.get().strip() or "./kk91_data")
        preferred = [
            current,
            root / "kk91_ui_data",
            root / "final-mining-ui-20260219-215832" / "data",
        ]

        candidates: list[Path] = []
        seen: set[Path] = set()

        for item in preferred:
            candidate = item.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

        for state_path in root.rglob("chain_state.json"):
            candidate = state_path.parent.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

        for candidate in candidates:
            try:
                chain = Chain(candidate)
                if not chain.exists():
                    continue
                chain.load()
                return candidate
            except Exception:
                continue
        return None

    def _try_recover_data_dir(self, source_error: Exception) -> bool:
        if self._recovery_attempted:
            return False
        self._recovery_attempted = True

        recovered = self._find_recoverable_data_dir()
        if recovered is None:
            return False

        self.data_dir_var.set(str(recovered))
        try:
            self._save_settings()
        except Exception:
            pass
        self._log(f"Auto-recovered data dir: {recovered} (after: {source_error})")
        return True

    def _periodic_refresh(self) -> None:
        if not self._is_mining():
            self._refresh_snapshot()
        self.after(2800, self._periodic_refresh)

    def _process_events(self) -> None:
        # Process latest progress sample only (throttled) for UI stability.
        latest_progress: dict[str, object] | None = None
        with self._progress_lock:
            if self._latest_progress is not None:
                latest_progress = self._latest_progress
                self._latest_progress = None

        if latest_progress is not None:
            self.nonce_var.set(str(latest_progress.get("nonce", 0)))
            self.attempts_var.set(str(latest_progress.get("attempts", 0)))
            self.rate_var.set(f"{float(latest_progress.get('hash_rate', 0.0)):.1f} H/s")
            self.elapsed_var.set(f"{float(latest_progress.get('elapsed', 0.0)):.1f}s")
            self.preview_var.set(str(latest_progress.get("hash_preview", "-")))

            backend = str(latest_progress.get("backend", "")).upper()
            if backend and self._is_mining():
                self.run_state_var.set(f"Mining ({backend})")

            self.hashrate_history.append(float(latest_progress.get("hash_rate", 0.0)))
            self.hashrate_history = self.hashrate_history[-180:]
            self._draw_hashrate_chart()
            self.activity_bar["value"] = (float(self.activity_bar["value"]) + 8.0) % 100.0

        # Drain finite event count per tick, keeping the main loop responsive.
        for _ in range(90):
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            try:
                etype = str(event.get("type", ""))
                if etype == "genesis":
                    self._log(
                        f"Genesis created: height={event.get('height')} hash={str(event.get('hash'))[:16]}..."
                    )
                    self._refresh_snapshot()
                elif etype == "block":
                    self._log(
                        f"Block mined: height={event.get('height')} reward={event.get('reward')} hash={str(event.get('hash'))[:18]}..."
                    )
                    self._refresh_snapshot()
                elif etype == "error":
                    msg = str(event.get("message", "Unknown mining error"))
                    self._log(msg)
                    self.worker_thread = None
                    self.stop_event = None
                    self._set_running_state(False)
                    messagebox.showerror("Mining error", msg)
                elif etype == "stopped":
                    self.worker_thread = None
                    self.stop_event = None
                    self._set_running_state(False)
                    self._refresh_snapshot()
                    self._log("Mining stopped")
            except Exception as exc:
                self._log(f"UI event handling error: {exc}")

        self.after(80, self._process_events)

    def _draw_hashrate_chart(self) -> None:
        self.chart.update_idletasks()
        width = max(40, self.chart.winfo_width())
        height = max(40, self.chart.winfo_height())

        self.chart.delete("all")

        for y in range(0, height, 36):
            self.chart.create_line(0, y, width, y, fill="#2E2A48", width=1)
        for x in range(0, width, 48):
            self.chart.create_line(x, 0, x, height, fill="#292642", width=1)

        values = self.hashrate_history
        if len(values) < 2:
            self.chart.create_text(
                width // 2,
                height // 2,
                text="Hashrate stream appears while mining",
                fill="#AFAEC2",
                font=("Segoe UI", 10),
            )
            return

        min_rate = min(values)
        max_rate = max(values)
        span = max(1.0, max_rate - min_rate)

        points: list[float] = []
        step = width / (len(values) - 1)
        for idx, rate in enumerate(values):
            x = idx * step
            y = height - (((rate - min_rate) / span) * (height - 20)) - 10
            points.extend([x, y])

        self.chart.create_line(*points, fill="#4C66FF", width=1, smooth=True)
        self.chart.create_line(*points, fill="#66D8FF", width=2, smooth=True)
        self.chart.create_text(8, 8, text=f"max {max_rate:.1f} H/s", anchor="nw", fill="#BDAEFF", font=("Consolas", 9))
        self.chart.create_text(8, 24, text=f"min {min_rate:.1f} H/s", anchor="nw", fill="#A7D4FF", font=("Consolas", 9))

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        try:
            self._save_settings()
        except Exception:
            pass

        self.destroy()


if __name__ == "__main__":
    app = KK91MiningUI()
    app.mainloop()
