from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from powx.chain import Chain, ValidationError
from powx.config import CONFIG
from powx.mnemonic import private_key_to_backup_mnemonic
from powx.wallet import (
    Wallet,
    create_seed_wallet,
    create_wallet,
    load_wallet,
    recover_wallet_from_seed,
    save_wallet,
)

try:
    import qrcode
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency
    qrcode = None
    Image = None
    ImageTk = None


class KK91WalletUI(tk.Tk):
    """The-Graph-inspired wallet UI for storage, backup, send and receive."""

    def __init__(self) -> None:
        super().__init__()

        self.title("KK91 Wallet")
        self.geometry("1400x900")
        self.minsize(1160, 760)
        self.configure(bg="#0C0B1E")
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

        self.settings_path = Path(__file__).with_name("wallet_ui_settings.json")
        settings = self._load_settings()

        self.data_dir_var = tk.StringVar(value=settings.get("data_dir", "./kk91_data"))
        self.wallet_path_var = tk.StringVar(value=settings.get("wallet_path", ""))
        self.genesis_supply_var = tk.StringVar(value=settings.get("genesis_supply", "0"))

        self.status_var = tk.StringVar(value="Idle")
        self.balance_var = tk.StringVar(value="-")
        self.height_var = tk.StringVar(value="-")
        self.mempool_var = tk.StringVar(value="-")
        self.utxo_var = tk.StringVar(value="-")

        self.send_to_var = tk.StringVar(value="")
        self.send_amount_var = tk.StringVar(value="")
        self.send_fee_var = tk.StringVar(value="1")

        self.current_wallet: Wallet | None = None
        self._last_history_rows: list[tuple[str, str, str, str, str, str]] = []
        self.qr_preview_image = None
        self._recovery_attempted = False

        self._setup_style()
        self._build_layout()
        self._try_autoload_wallet()
        self._refresh_all()

        self.after(2600, self._periodic_refresh)
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
        white = "#FFFFFF"
        text_dim = "#AFAEC2"
        line = "#2E2A48"

        style.configure("App.TFrame", background=midnight)
        style.configure("Card.TFrame", background=panes, relief="solid", borderwidth=1)
        style.configure("Panel.TFrame", background=tooltip, relief="solid", borderwidth=1)

        style.configure("Title.TLabel", background=midnight, foreground=white, font=("Segoe UI Semibold", 27))
        style.configure("Subtitle.TLabel", background=midnight, foreground=text_dim, font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=panes, foreground=white, font=("Segoe UI Semibold", 13))
        style.configure("Meta.TLabel", background=panes, foreground=text_dim, font=("Segoe UI", 9))
        style.configure("Value.TLabel", background=panes, foreground=turquoise, font=("Consolas", 12, "bold"))
        style.configure("SeedHint.TLabel", background=panes, foreground="#BDAEFF", font=("Segoe UI", 9))
        style.configure("Divider.TFrame", background=line)

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
            "History.Treeview",
            rowheight=27,
            background=tooltip,
            fieldbackground=tooltip,
            foreground=white,
            borderwidth=1,
            font=("Segoe UI", 10),
        )
        style.configure(
            "History.Treeview.Heading",
            background=panes,
            foreground=white,
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "History.Treeview",
            background=[("selected", blue)],
            foreground=[("selected", white)],
        )
        style.map("History.Treeview.Heading", background=[("active", "#27224A")])

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        ttk.Label(header, text="KK91 Wallet Dashboard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Store, backup, receive and send KK91. Mining lives only in the Mining Dashboard.",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        ttk.Frame(header, style="Divider.TFrame", height=1).pack(fill="x", pady=(10, 0))

        left = ttk.Frame(root, style="App.TFrame")
        left.grid(row=1, column=0, sticky="ns", padx=(0, 12))

        right = ttk.Frame(root, style="App.TFrame")
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self._build_wallet_card(left)
        self._build_seed_card(left)
        self._build_receive_card(left)

        self._build_top_card(right)
        self._build_send_card(right)
        self._build_history_card(right)

    def _build_wallet_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.pack(fill="x", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Wallet Setup", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(card, text="Data Directory", style="Meta.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.data_dir_var, style="Field.TEntry").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Label(card, text="Wallet File", style="Meta.TLabel").grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.wallet_path_var, style="Field.TEntry").grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Button(card, text="Browse Wallet", style="Action.TButton", command=self._browse_wallet).grid(
            row=5, column=0, sticky="ew", padx=(0, 5), pady=(0, 7)
        )
        ttk.Button(card, text="Load Wallet", style="Action.TButton", command=self._load_selected_wallet).grid(
            row=5, column=1, sticky="ew", padx=(5, 0), pady=(0, 7)
        )

        ttk.Button(card, text="Create Random Wallet", style="Action.TButton", command=self._create_random_wallet).grid(
            row=6, column=0, sticky="ew", padx=(0, 5), pady=(0, 7)
        )
        ttk.Button(card, text="Create Seed Wallet", style="Primary.TButton", command=self._create_seed_wallet_action).grid(
            row=6, column=1, sticky="ew", padx=(5, 0), pady=(0, 7)
        )

        ttk.Button(card, text="Recover From Seed", style="Primary.TButton", command=self._open_seed_recovery).grid(
            row=7, column=0, sticky="ew", padx=(0, 5), pady=(0, 7)
        )
        ttk.Button(card, text="Save Settings", style="Action.TButton", command=self._save_settings).grid(
            row=7, column=1, sticky="ew", padx=(5, 0), pady=(0, 7)
        )

        ttk.Label(card, text="Genesis Supply", style="Meta.TLabel").grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.genesis_supply_var, style="Field.TEntry").grid(row=9, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Button(card, text="Init Local Chain", style="Primary.TButton", command=self._init_chain).grid(
            row=10, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(card, text="Refresh", style="Action.TButton", command=self._refresh_all).grid(
            row=10, column=1, sticky="ew", padx=(5, 0)
        )

    def _build_seed_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.pack(fill="x", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=1)

        ttk.Label(card, text="Seed Phrase Backup", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(card, text="12 words (new) or 24 words (backup) - keep offline and private", style="SeedHint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )

        self.seed_box = tk.Text(
            card,
            height=3,
            wrap="word",
            bg="#222037",
            fg="#BDAEFF",
            relief="flat",
            font=("Consolas", 10),
            insertbackground="#BDAEFF",
            highlightthickness=1,
            highlightbackground="#2E2A48",
            highlightcolor="#4C66FF",
            padx=8,
            pady=6,
        )
        self.seed_box.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.seed_box.configure(state="disabled")

        ttk.Button(card, text="Create Backup Seed", style="Primary.TButton", command=self._generate_backup_seed).grid(
            row=3, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(card, text="Copy Seed Phrase", style="Action.TButton", command=self._copy_seed_phrase).grid(
            row=3, column=1, sticky="ew", padx=4
        )
        ttk.Button(card, text="Recover Wallet", style="Primary.TButton", command=self._open_seed_recovery).grid(
            row=3, column=2, sticky="ew", padx=(4, 0)
        )

    def _build_receive_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.pack(fill="x")
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=1)

        ttk.Label(card, text="Receive", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(card, text="Wallet address", style="Meta.TLabel").grid(row=1, column=0, columnspan=3, sticky="w")

        self.address_box = tk.Text(
            card,
            height=3,
            wrap="word",
            bg="#222037",
            fg="#FFFFFF",
            relief="flat",
            font=("Consolas", 10),
            insertbackground="#66D8FF",
            highlightthickness=1,
            highlightbackground="#2E2A48",
            highlightcolor="#4C66FF",
            padx=8,
            pady=6,
        )
        self.address_box.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 8))
        self.address_box.configure(state="disabled")

        self.qr_preview_label = ttk.Label(card, text="QR preview unavailable", style="Meta.TLabel")
        self.qr_preview_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Button(card, text="Copy Address", style="Action.TButton", command=self._copy_address).grid(
            row=4, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(card, text="Show QR", style="Primary.TButton", command=self._show_qr_popup).grid(
            row=4, column=1, sticky="ew", padx=4
        )
        ttk.Button(card, text="Save QR PNG", style="Action.TButton", command=self._save_qr_png).grid(
            row=4, column=2, sticky="ew", padx=(4, 0)
        )

    def _build_top_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Overview", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._stat_row(card, 1, "Status", self.status_var)
        self._stat_row(card, 2, "Balance", self.balance_var)
        self._stat_row(card, 3, "Height", self.height_var)
        self._stat_row(card, 4, "Mempool", self.mempool_var)
        self._stat_row(card, 5, "UTXO", self.utxo_var)

    def _build_send_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Send KK91", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(card, text="Recipient address", style="Meta.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Entry(card, textvariable=self.send_to_var, style="Field.TEntry").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Label(card, text="Amount", style="Meta.TLabel").grid(row=3, column=0, sticky="w")
        ttk.Label(card, text="Fee", style="Meta.TLabel").grid(row=3, column=1, sticky="w")

        ttk.Entry(card, textvariable=self.send_amount_var, style="Field.TEntry").grid(row=4, column=0, sticky="ew", padx=(0, 5), pady=(2, 8))
        ttk.Entry(card, textvariable=self.send_fee_var, style="Field.TEntry").grid(row=4, column=1, sticky="ew", padx=(5, 0), pady=(2, 8))

        ttk.Button(card, text="Send Transaction", style="Primary.TButton", command=self._send_transaction).grid(
            row=5, column=0, columnspan=2, sticky="ew"
        )

    def _build_history_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(15, 14))
        card.grid(row=2, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=0)
        card.rowconfigure(1, weight=1)

        ttk.Label(card, text="History", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(card, text="Export CSV", style="Action.TButton", command=self._export_history_csv).grid(
            row=0, column=1, sticky="e", pady=(0, 8)
        )

        columns = ("time", "type", "amount", "counterparty", "status", "txid")
        self.history = ttk.Treeview(card, columns=columns, show="headings", style="History.Treeview")
        self.history.grid(row=1, column=0, columnspan=2, sticky="nsew")

        headings = {
            "time": "Time",
            "type": "Type",
            "amount": "Amount",
            "counterparty": "Counterparty",
            "status": "Status",
            "txid": "TXID",
        }
        widths = {
            "time": 152,
            "type": 100,
            "amount": 118,
            "counterparty": 320,
            "status": 96,
            "txid": 170,
        }
        for col in columns:
            self.history.heading(col, text=headings[col])
            self.history.column(col, width=widths[col], anchor="w")

        ttk.Label(card, text="Wallet Log", style="CardTitle.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 6))

        self.log_text = tk.Text(
            card,
            height=8,
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
        self.log_text.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _stat_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="Meta.TLabel").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=var, style="Value.TLabel").grid(row=row, column=1, sticky="e", pady=2)

    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _load_settings(self) -> dict[str, str]:
        if not self.settings_path.exists():
            return {}
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
        return {}

    def _save_settings(self) -> None:
        data = {
            "data_dir": self.data_dir_var.get().strip() or "./kk91_data",
            "wallet_path": self.wallet_path_var.get().strip(),
            "genesis_supply": self.genesis_supply_var.get().strip() or "0",
        }
        with self.settings_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        self._log("Settings saved")

    def _browse_wallet(self) -> None:
        path = filedialog.askopenfilename(
            title="Select wallet file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if path:
            self.wallet_path_var.set(path)

    def _set_address(self, address: str) -> None:
        self.address_box.configure(state="normal")
        self.address_box.delete("1.0", "end")
        self.address_box.insert("1.0", address)
        self.address_box.configure(state="disabled")
        self._update_qr_preview(address)

    def _set_seed_phrase(self, mnemonic: str) -> None:
        display = mnemonic.strip() if mnemonic.strip() else "No seed phrase stored in this wallet file."
        self.seed_box.configure(state="normal")
        self.seed_box.delete("1.0", "end")
        self.seed_box.insert("1.0", display)
        self.seed_box.configure(state="disabled")

    def _wallets_dir(self) -> Path:
        data_dir = Path(self.data_dir_var.get().strip() or "./kk91_data")
        wallets_dir = data_dir / "wallets"
        wallets_dir.mkdir(parents=True, exist_ok=True)
        return wallets_dir

    @staticmethod
    def _unique_wallet_path(wallets_dir: Path, stem: str) -> Path:
        base = wallets_dir / f"{stem}.json"
        if not base.exists():
            return base
        for idx in range(2, 10000):
            path = wallets_dir / f"{stem}_{idx}.json"
            if not path.exists():
                return path
        raise RuntimeError("Could not create a unique wallet filename")

    def _activate_wallet(self, wallet: Wallet, path: Path, status_text: str) -> None:
        self.current_wallet = wallet
        self.wallet_path_var.set(str(path))
        self._set_address(wallet.address)
        self._set_seed_phrase(wallet.mnemonic)
        self.status_var.set(status_text)
        self._save_settings()
        self._refresh_all()

    def _try_autoload_wallet(self) -> None:
        wallet_path = self.wallet_path_var.get().strip()
        if not wallet_path:
            self._set_address("No wallet loaded")
            self._set_seed_phrase("")
            self.status_var.set("No wallet loaded")
            return

        path_obj = Path(wallet_path)
        if not path_obj.exists():
            self._set_address("No wallet loaded")
            self._set_seed_phrase("")
            self.status_var.set("No wallet loaded")
            return

        try:
            wallet = load_wallet(path_obj)
            self.current_wallet = wallet
            self._set_address(wallet.address)
            self._set_seed_phrase(wallet.mnemonic)
            self.status_var.set("Wallet loaded")
        except Exception:
            self.current_wallet = None
            self._set_address("No wallet loaded")
            self._set_seed_phrase("")
            self.status_var.set("No wallet loaded")

    def _load_selected_wallet(self) -> None:
        wallet_path = self.wallet_path_var.get().strip()
        if not wallet_path:
            messagebox.showerror("Wallet", "Please select a wallet file")
            return

        try:
            wallet = load_wallet(wallet_path)
            self.current_wallet = wallet
            self._set_address(wallet.address)
            self._set_seed_phrase(wallet.mnemonic)
            self.status_var.set("Wallet loaded")
            self._save_settings()
            self._log(f"Wallet loaded: {wallet_path}")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Wallet", f"Failed to load wallet:\n{exc}")

    def _create_random_wallet(self) -> None:
        wallets_dir = self._wallets_dir()
        wallet = create_wallet(CONFIG.symbol)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._unique_wallet_path(wallets_dir, f"wallet_{stamp}")
        save_wallet(wallet, path)

        self._activate_wallet(wallet, path, "Wallet created")
        self._log(f"Wallet created: {path}")

    def _create_seed_wallet_action(self) -> None:
        wallets_dir = self._wallets_dir()
        wallet = create_seed_wallet(CONFIG.symbol)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._unique_wallet_path(wallets_dir, f"wallet_seed_{stamp}")
        save_wallet(wallet, path)

        self._activate_wallet(wallet, path, "Seed wallet created")
        self._log(f"Seed wallet created: {path}")
        messagebox.showinfo(
            "Seed phrase",
            "New wallet with seed phrase created.\nWrite the phrase down offline before closing.",
        )

    def _open_seed_recovery(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Recover Wallet From Seed")
        dialog.configure(bg="#0C0B1E")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, style="Card.TFrame", padding=(14, 12))
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Seed Recovery", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Paste your 12-word or 24-word seed phrase", style="Meta.TLabel").pack(anchor="w", pady=(0, 8))

        seed_input = tk.Text(
            frame,
            height=4,
            width=58,
            wrap="word",
            bg="#222037",
            fg="#BDAEFF",
            relief="flat",
            font=("Consolas", 10),
            insertbackground="#BDAEFF",
            highlightthickness=1,
            highlightbackground="#2E2A48",
            highlightcolor="#4C66FF",
            padx=8,
            pady=6,
        )
        seed_input.pack(fill="x", pady=(0, 10))

        buttons = ttk.Frame(frame, style="Card.TFrame")
        buttons.pack(fill="x")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        def recover_now() -> None:
            phrase = seed_input.get("1.0", "end").strip()
            if not phrase:
                messagebox.showerror("Recover", "Please enter the seed phrase", parent=dialog)
                return
            try:
                wallet = recover_wallet_from_seed(phrase, CONFIG.symbol)
                wallets_dir = self._wallets_dir()
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = self._unique_wallet_path(wallets_dir, f"wallet_recovered_{stamp}")
                save_wallet(wallet, path)
                self._activate_wallet(wallet, path, "Wallet recovered")
                self._log(f"Wallet recovered from seed: {path}")
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("Recover", f"Failed to recover wallet:\n{exc}", parent=dialog)

        ttk.Button(buttons, text="Recover Wallet", style="Primary.TButton", command=recover_now).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(buttons, text="Cancel", style="Action.TButton", command=dialog.destroy).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

    def _copy_address(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Copy address", "Load or create a wallet first")
            return
        self.clipboard_clear()
        self.clipboard_append(self.current_wallet.address)
        self._log("Address copied to clipboard")

    def _persist_current_wallet(self) -> Path:
        if self.current_wallet is None:
            raise RuntimeError("No wallet loaded")

        wallet_path = self.wallet_path_var.get().strip()
        if wallet_path:
            path = Path(wallet_path)
        else:
            wallets_dir = self._wallets_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._unique_wallet_path(wallets_dir, f"wallet_{stamp}")
            self.wallet_path_var.set(str(path))

        save_wallet(self.current_wallet, path)
        self._save_settings()
        return path

    def _generate_backup_seed(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Backup seed", "Load or create a wallet first")
            return

        if self.current_wallet.mnemonic.strip():
            messagebox.showinfo("Backup seed", "This wallet already has a saved seed phrase.")
            return

        try:
            backup = private_key_to_backup_mnemonic(self.current_wallet.private_key)
            self.current_wallet.mnemonic = backup
            path = self._persist_current_wallet()
            self._set_seed_phrase(backup)
            self.status_var.set("Backup seed created")
            self._log(f"Backup seed created and saved: {path}")
            messagebox.showinfo(
                "Backup seed created",
                "A backup seed phrase was generated for this wallet.\nWrite it down offline and keep it private.",
            )
        except Exception as exc:
            messagebox.showerror("Backup seed", f"Failed to create backup seed:\n{exc}")

    def _copy_seed_phrase(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Copy seed", "Load or create a wallet first")
            return
        mnemonic = self.current_wallet.mnemonic.strip()
        if not mnemonic:
            messagebox.showerror("Copy seed", "Current wallet has no seed phrase")
            return
        self.clipboard_clear()
        self.clipboard_append(mnemonic)
        self._log("Seed phrase copied to clipboard")

    def _can_render_qr(self) -> bool:
        return qrcode is not None and Image is not None and ImageTk is not None

    def _build_qr_image(self, address: str, pixel_size: int) -> Image.Image:
        if not self._can_render_qr():
            raise RuntimeError("QR rendering requires 'qrcode' and 'Pillow' packages")

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=2,
        )
        qr.add_data(address)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        return img.resize((pixel_size, pixel_size), Image.Resampling.NEAREST)

    def _update_qr_preview(self, address: str) -> None:
        if not address or address == "No wallet loaded":
            self.qr_preview_label.configure(text="QR preview unavailable", image="")
            self.qr_preview_image = None
            return

        if not self._can_render_qr():
            self.qr_preview_label.configure(text="Install qrcode + Pillow for QR preview", image="")
            self.qr_preview_image = None
            return

        try:
            img = self._build_qr_image(address, 118)
            self.qr_preview_image = ImageTk.PhotoImage(img)
            self.qr_preview_label.configure(text="", image=self.qr_preview_image)
        except Exception as exc:
            self.qr_preview_label.configure(text=f"QR preview failed: {exc}", image="")
            self.qr_preview_image = None

    def _show_qr_popup(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Show QR", "Load or create a wallet first")
            return

        if not self._can_render_qr():
            messagebox.showerror("Show QR", "Install dependencies: python -m pip install qrcode Pillow")
            return

        try:
            img = self._build_qr_image(self.current_wallet.address, 320)
            img_tk = ImageTk.PhotoImage(img)
        except Exception as exc:
            messagebox.showerror("Show QR", f"QR generation failed:\n{exc}")
            return

        win = tk.Toplevel(self)
        win.title("Receive QR")
        win.configure(bg="#0C0B1E")
        win.resizable(False, False)

        frame = ttk.Frame(win, style="Card.TFrame", padding=(12, 12))
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Wallet Receive QR", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

        image_label = ttk.Label(frame, image=img_tk)
        image_label.image = img_tk
        image_label.pack(pady=(0, 10))

        addr_label = tk.Text(
            frame,
            height=2,
            width=44,
            bg="#222037",
            fg="#FFFFFF",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            insertbackground="#66D8FF",
            highlightthickness=1,
            highlightbackground="#2E2A48",
            highlightcolor="#4C66FF",
            padx=8,
            pady=6,
        )
        addr_label.insert("1.0", self.current_wallet.address)
        addr_label.configure(state="disabled")
        addr_label.pack(fill="x")

    def _save_qr_png(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Save QR", "Load or create a wallet first")
            return

        if not self._can_render_qr():
            messagebox.showerror("Save QR", "Install dependencies: python -m pip install qrcode Pillow")
            return

        data_dir = Path(self.data_dir_var.get().strip() or ".")
        default_name = f"kk91_receive_qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        target = filedialog.asksaveasfilename(
            title="Save receive QR",
            defaultextension=".png",
            initialdir=str(data_dir),
            initialfile=default_name,
            filetypes=[("PNG image", "*.png")],
        )
        if not target:
            return

        try:
            img = self._build_qr_image(self.current_wallet.address, 640)
            img.save(target, format="PNG")
            self._log(f"QR saved: {target}")
        except Exception as exc:
            messagebox.showerror("Save QR", f"QR export failed:\n{exc}")

    def _write_history_csv(self, rows: list[tuple[str, str, str, str, str, str]], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "type", "amount", "counterparty", "status", "txid"])
            for row in rows:
                writer.writerow(list(row))

    def _export_history_csv(self) -> None:
        if not self._last_history_rows:
            messagebox.showerror("Export CSV", "No history rows to export")
            return

        data_dir = Path(self.data_dir_var.get().strip() or ".")
        default_name = f"kk91_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        target = filedialog.asksaveasfilename(
            title="Export transaction history",
            defaultextension=".csv",
            initialdir=str(data_dir),
            initialfile=default_name,
            filetypes=[("CSV file", "*.csv")],
        )
        if not target:
            return

        try:
            self._write_history_csv(self._last_history_rows, Path(target))
            self._log(f"History exported: {target}")
        except Exception as exc:
            messagebox.showerror("Export CSV", f"Export failed:\n{exc}")

    def _validate_address(self, address: str) -> None:
        if not address.startswith(CONFIG.symbol):
            raise ValidationError(f"Address must start with '{CONFIG.symbol}'")
        body = address[len(CONFIG.symbol) :]
        if len(body) != 40:
            raise ValidationError("Address must contain 40 hex chars after prefix")
        try:
            int(body, 16)
        except ValueError as exc:
            raise ValidationError("Address contains invalid characters") from exc

    def _load_chain(self) -> Chain | None:
        data_dir = Path(self.data_dir_var.get().strip() or "./kk91_data")
        chain = Chain(data_dir)
        if not chain.exists():
            return None
        chain.load()
        return chain

    def _find_recoverable_data_dir(self) -> Path | None:
        root = Path(__file__).resolve().parent
        current = Path(self.data_dir_var.get().strip() or "./kk91_data")
        preferred = [
            current,
            root / "kk91_ui_data",
            root / "kk91_wallet_data",
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

    def _init_chain(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Init chain", "Load or create a wallet first")
            return

        try:
            genesis_supply = int(self.genesis_supply_var.get().strip())
            if genesis_supply < 0:
                raise ValueError("Genesis supply must be >= 0")
        except Exception as exc:
            messagebox.showerror("Init chain", f"Invalid genesis supply:\n{exc}")
            return

        data_dir = Path(self.data_dir_var.get().strip() or "./kk91_data")
        chain = Chain(data_dir)

        if chain.exists():
            chain.load()
            if chain.chain:
                self._log("Chain already initialized")
                self.status_var.set("Chain ready")
                self._refresh_all()
                return

        try:
            block = chain.initialize(self.current_wallet.address, genesis_supply=genesis_supply)
            self.status_var.set("Chain initialized")
            self._log(f"Genesis block created: {block.block_hash[:18]}...")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Init chain", f"Failed to initialize chain:\n{exc}")

    def _send_transaction(self) -> None:
        if self.current_wallet is None:
            messagebox.showerror("Send", "Load or create a wallet first")
            return

        to_address = self.send_to_var.get().strip()
        amount_text = self.send_amount_var.get().strip()
        fee_text = self.send_fee_var.get().strip() or "1"

        try:
            self._validate_address(to_address)
            amount = int(amount_text)
            fee = int(fee_text)
            if amount <= 0:
                raise ValidationError("Amount must be positive")
            if fee <= 0:
                raise ValidationError("Fee must be positive")
        except Exception as exc:
            messagebox.showerror("Send", f"Invalid input:\n{exc}")
            return

        chain = self._load_chain()
        if chain is None:
            messagebox.showerror("Send", "Chain not initialized. Click 'Init Local Chain' first.")
            return

        try:
            tx = chain.create_transaction(
                private_key_hex=self.current_wallet.private_key,
                to_address=to_address,
                amount=amount,
                fee=fee,
            )
            chain.add_transaction(tx)
            self.send_amount_var.set("")
            self.status_var.set("Transaction broadcast")
            self._log(f"Transaction sent: {tx.txid[:18]}...")
            self._refresh_all()
        except Exception as exc:
            messagebox.showerror("Send", f"Transaction failed:\n{exc}")

    def _collect_history(self, chain: Chain, address: str, limit: int = 180) -> list[tuple[str, str, str, str, str, str]]:
        output_map: dict[str, object] = {}
        for block in chain.chain:
            for tx in block.transactions:
                for index, output in enumerate(tx.outputs):
                    output_map[f"{tx.txid}:{index}"] = output

        rows: list[tuple[str, str, str, str, str, str]] = []

        for tx in reversed(chain.mempool):
            row = self._tx_row_for_address(tx, address, output_map, pending=True)
            if row is not None:
                rows.append(row)

        for block in reversed(chain.chain):
            for tx in reversed(block.transactions):
                row = self._tx_row_for_address(tx, address, output_map, pending=False)
                if row is not None:
                    rows.append(row)
                if len(rows) >= limit:
                    return rows[:limit]

        return rows[:limit]

    def _tx_row_for_address(
        self,
        tx,
        address: str,
        output_map: dict[str, object],
        pending: bool,
    ) -> tuple[str, str, str, str, str, str] | None:
        ts = datetime.fromtimestamp(tx.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        status = "pending" if pending else "confirmed"

        if tx.is_coinbase():
            mined = sum(out.amount for out in tx.outputs if out.address == address)
            if mined <= 0:
                return None
            return (ts, "Mined", f"+{mined} {CONFIG.symbol}", "-", status, tx.txid[:14])

        total_in_from_me = 0
        sender_candidates: list[str] = []
        for tx_input in tx.inputs:
            prev = output_map.get(f"{tx_input.txid}:{tx_input.index}")
            if prev is None:
                continue
            prev_addr = getattr(prev, "address", "")
            prev_amount = int(getattr(prev, "amount", 0))
            sender_candidates.append(prev_addr)
            if prev_addr == address:
                total_in_from_me += prev_amount

        received_to_me = sum(out.amount for out in tx.outputs if out.address == address)

        if total_in_from_me > 0:
            spent = max(0, total_in_from_me - received_to_me)
            counterparty = next((out.address for out in tx.outputs if out.address != address), "-")
            return (ts, "Sent", f"-{spent} {CONFIG.symbol}", counterparty, status, tx.txid[:14])

        if received_to_me > 0:
            sender = sender_candidates[0] if sender_candidates else "-"
            return (ts, "Received", f"+{received_to_me} {CONFIG.symbol}", sender, status, tx.txid[:14])

        return None

    def _fill_history(self, rows: list[tuple[str, str, str, str, str, str]]) -> None:
        self._last_history_rows = list(rows)
        for item in self.history.get_children():
            self.history.delete(item)
        for row in rows:
            self.history.insert("", "end", values=row)

    def _refresh_all(self) -> None:
        if self.current_wallet is not None:
            self._set_address(self.current_wallet.address)
            self._set_seed_phrase(self.current_wallet.mnemonic)
        else:
            self._set_address("No wallet loaded")
            self._set_seed_phrase("")

        try:
            chain = self._load_chain()
        except Exception as exc:
            if self._try_recover_data_dir(exc):
                self._refresh_all()
                return
            self.height_var.set("-")
            self.balance_var.set("-")
            self.mempool_var.set("-")
            self.utxo_var.set("-")
            self._fill_history([])
            self.status_var.set("Chain error")
            self._log(f"Chain load failed: {exc}")
            return

        if chain is None:
            self._recovery_attempted = False
            self.height_var.set("-")
            self.balance_var.set("-")
            self.mempool_var.set("-")
            self.utxo_var.set("-")
            self._fill_history([])
            if self.current_wallet is None:
                self.status_var.set("No wallet loaded")
            else:
                self.status_var.set("Chain missing")
            return

        status = chain.status()
        self._recovery_attempted = False
        self.height_var.set(str(status["height"]))
        self.mempool_var.set(str(status["mempool_size"]))
        self.utxo_var.set(str(status["utxo_count"]))

        if self.current_wallet is not None:
            balance = chain.balance_of(self.current_wallet.address)
            self.balance_var.set(f"{balance} {CONFIG.symbol}")
            rows = self._collect_history(chain, self.current_wallet.address)
            self._fill_history(rows)
            if not self.status_var.get().startswith("Transaction"):
                self.status_var.set("Ready")
        else:
            self.balance_var.set("-")
            self._fill_history([])
            self.status_var.set("Wallet not loaded")

    def _periodic_refresh(self) -> None:
        try:
            self._refresh_all()
        except Exception:
            pass
        self.after(2600, self._periodic_refresh)

    def _on_close(self) -> None:
        try:
            self._save_settings()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = KK91WalletUI()
    app.mainloop()
