# KK91 - Fast Proof-of-Work Crypto Prototype

KK91 ist ein **Bitcoin-inspirierter PoW-Prototyp** mit Fokus auf schnellere Bestätigungen und zusätzliche Sicherheitsregeln.

Wichtig: Das ist ein Lern-/Demo-Projekt, **kein production-ready Mainnet**.

## Was ist schneller/sicherer als Bitcoin (im Prototyp)

- Schnellere Ziel-Blockzeit: `30s` statt ~10 Minuten
- Dynamische Difficulty pro Block mit begrenzten Anpassungen
- Eigenes PoW (`kkhash-v1`) mit CPU-Validierung und optionalem GPU-Mining (OpenCL)
- Signierte UTXO-Transaktionen (secp256k1, deterministische Nonces)
- Upgrade-Framework: Protokollversion + Aktivierungshoehe fuer v2 (`protocol_upgrade_v2_height`)
- Chain-Work-Validierung + strenge Block/Tx-Checks (Merkle, Rewards, Double-Spend)
- Reorg-Schutz per `max_reorg_depth` + Genesis-Mismatch-Abwehr
- Mempool-Hardening: Tx-Limits, Age/Future-Checks und Kapazitaetslimit

## Voraussetzungen

- Python 3.11+ (3.10 sollte ebenfalls funktionieren)
- Fuer GPU-Mining: OpenCL-Treiber + `pyopencl` + `numpy`
- Fuer QR in Wallet-UI: `qrcode` + `Pillow`

Installation (optional fuer GPU):

```bash
python -m pip install numpy pyopencl
python -m pip install qrcode Pillow
```

## Schnellstart

```bash
cd "d:\Projekt X\powx"
python powx_cli.py wallet-new --out alice.json
python powx_cli.py init --data-dir ./data --genesis-wallet alice.json --genesis-supply 0
python powx_cli.py status --data-dir ./data
python powx_cli.py mine --data-dir ./data --wallet alice.json --blocks 2 --backend auto
python powx_cli.py balance --data-dir ./data --wallet alice.json
```

## Konsens-Tests (Schritt 1)

Feste Testvektoren + Fuzz-Tests fuer Konsensregeln liegen unter `tests/`:
- `tests/test_consensus_vectors.py`
- `tests/test_consensus_fuzz.py`

Ausfuehren:

```bash
cd "d:\Projekt X\powx"
python -m unittest discover -s tests -p "test_*.py" -v
```

Schritt 2 erweitert die Vektor-Tests um:
- Halving-Reward-Schedule
- Supply-Cap-Verhalten (inkl. abgeschnittener letzter Subsidy bis genau `max_total_supply`)
- Reorg-Depth-Grenze bei `replace_chain`

Schritt 3 erweitert um Upgrade-/Kompatibilitaetstests:
- Protokoll-Upgrade-Transition (Tx-Version v1 -> v2 per Height)
- Tx-Version-Gate nach Upgrade (legacy v1 Tx wird ab Upgrade-Hoehe abgelehnt)
- State-Kompatibilitaet fuer aeltere Dateien ohne Protokollfelder
- Persistente Uebernahme von `protocol_upgrade_v2_height` aus State-Datei

## Mining-Oberflaeche (GUI)

```bash
cd "d:\Projekt X\powx"
python kk91_mining_ui.py
```

Was die GUI kann:
- Wallet-Adresse eingeben und lokal speichern (`mining_ui_settings.json`)
- Rewards gehen immer an die eingegebene Wallet-Adresse
- Start/Stop-Mining mit stabiler Worker-Logik (kein UI-Freeze bei Stop)
- Einzelblock-Mining per `Mine 1 Block`
- Backend-Wahl: `gpu`, `auto`, `cpu`
- Live-Ansicht: Nonce, Attempts, Hashrate, Hash-Preview, Event-Log
- Optischer Hashrate-Chart waehrend des Minings
- Bei leerem Datenordner wird die Chain automatisch initialisiert
- Fuer Wallet/Mining-Sync denselben `data_dir` in beiden UIs verwenden

## Wallet-Oberflaeche (GUI)

```bash
cd "d:\Projekt X\powx"
python kk91_wallet_ui.py
```

Was die Wallet-GUI kann:
- Wallet erstellen/laden und Einstellungen speichern (`wallet_ui_settings.json`)
- Seed-Wallet erstellen (12-Wort Seed Phrase) und Wallet aus Seed wiederherstellen
- Backup-Seed fuer bestehende Wallet erzeugen (24-Wort) und daraus wiederherstellen
- Seed Phrase anzeigen/kopieren fuer Backup
- Kontostand, Blockhoehe und Mempool live anzeigen
- Coins senden (Empfaenger, Betrag, Fee)
- Empfangsadresse kopieren
- Empfangs-QR anzeigen und als PNG speichern
- Historie mit Pending/Confirmed-Status
- Historie als CSV exportieren
- Lokale Chain initialisieren (Genesis)
- Kein Mining in der Wallet (Mining nur im separaten Mining-Dashboard)

## Transaktion senden

```bash
python powx_cli.py wallet-new --out bob.json
python powx_cli.py wallet-address --wallet bob.json
# Ausgabe-Adresse als <BOB_ADDR> einsetzen
python powx_cli.py send --data-dir ./data --wallet alice.json --to <BOB_ADDR> --amount 1000 --fee 2
python powx_cli.py mempool --data-dir ./data
python powx_cli.py mine --data-dir ./data --wallet alice.json --blocks 1 --backend auto
python powx_cli.py balance --data-dir ./data --wallet bob.json
```

## P2P-Netzwerk (Node zu Node)

Node A starten:

```bash
python powx_cli.py node-run --data-dir ./nodeA --host 127.0.0.1 --port 8844
```

Node B starten und Node A als Peer setzen:

```bash
python powx_cli.py node-run --data-dir ./nodeB --host 127.0.0.1 --port 8845 --peer http://127.0.0.1:8844
```

Peer nachtraeglich hinzufuegen + sofort sync:

```bash
python powx_cli.py node-add-peer --node http://127.0.0.1:8845 --peer http://127.0.0.1:8844
```

Sync manuell triggern:

```bash
python powx_cli.py node-sync --node http://127.0.0.1:8845
```

Block/Tx aus CLI an Node broadcasten:

```bash
python powx_cli.py send --data-dir ./nodeA --wallet alice.json --to <ADDR> --amount 5 --broadcast-node http://127.0.0.1:8844
python powx_cli.py mine --data-dir ./nodeA --wallet alice.json --blocks 1 --backend auto --broadcast-node http://127.0.0.1:8844
```

## Node REST API + Hardening (Schritt 3)

Die Node stellt REST-Endpunkte bereit, damit Wallet/Mining nicht nur dateibasiert arbeiten:

- `GET /api/v1/status`
- `GET /api/v1/balance?address=<ADDR>`
- `GET /api/v1/chain?limit=20`
- `GET /api/v1/mempool`
- `GET /api/v1/history?address=<ADDR>&limit=120`
- `POST /api/v1/tx/create` (signiert + broadcastet Tx auf Node)
- `POST /api/v1/mine` (mine auf Node)
- `POST /api/v1/mempool/prune`

Zusatz in Schritt 3 (Härtung):
- HTTP Rate-Limit pro Client (429 bei Überschreitung)
- Request-Body-Limit fuer POST-Requests
- Inbound TTL wird auf ein Maximal-Limit gecappt (Schutz gegen Relay-Stuermung)
- Maximale Peer-Anzahl (`max_peers`)
- Sync-Retry nach invaliden Tx/Blocks nur von bekannten Peers + Cooldown

Neue `node-run` Optionen:
- `--max-peers`
- `--max-inbound-ttl`
- `--max-request-body-bytes`
- `--max-requests-per-minute`
- `--sync-retry-cooldown`

## Peer Security (Schritt 4)

Peer-Scoring + Temp-Ban ist aktiv:
- Ungueltige Transaktionen von einem Peer erhoehen den Score
- Ungueltige Bloecke von einem Peer erhoehen den Score staerker
- Ungueltige Sync-Daten erhoehen den Score
- Erfolgreiche Datenannahme reduziert den Score wieder
- Ab Schwellwert wird der Peer temporaer gebannt

Zusatzoptionen fuer `node-run`:
- `--peer-ban-threshold`
- `--peer-ban-seconds`
- `--peer-penalty-invalid-tx`
- `--peer-penalty-invalid-block`
- `--peer-penalty-bad-sync`
- `--peer-reward-success`

Der Node-Status (`/status` oder `api-status`) enthaelt jetzt `peer_security` mit Score/Ban-Zustand pro Peer.

## Peer Identity + Persistence (Schritt 5)

Neue Haertung fuer Peer-Kommunikation:
- Gossip-Nachrichten (`/tx`, `/block`) werden zwischen Nodes signiert (`auth`-Envelope mit secp256k1)
- `from` wird nicht mehr blind vertraut: bekannte Peers brauchen gueltige Signatur + passendes Pubkey-Binding
- Replay-Schutz fuer Peer-Signaturen (Nonce + Zeitfenster)

Persistenz:
- Jede Node hat eine eigene Identitaet in `node_identity.json` (priv/pub key)
- Peer-Reputation/Ban/Pubkey-Bindings werden in `peer_security.json` gespeichert
- Nach Neustart bleiben Score/Ban-Zustaende und bekannte Peer-Pubkeys erhalten

Weitere `node-run` Optionen:
- `--peer-auth-max-skew-seconds`
- `--peer-auth-replay-window-seconds`

## P2P Relay + Robustness (Schritt 6)

Neue Gossip-Semantik (Bitcoin-aehnlich):
- `POST /inv` kuendigt nur Inventar (`txid`/`block hash`) an
- `POST /getdata` liefert Full-Payload nur auf Anforderung
- Fallback auf `/tx` und `/block` bleibt fuer Kompatibilitaet aktiv

Robustheit:
- Orphan-Block-Pool mit TTL + Groessenlimit (fehlender Parent wird zwischengespeichert)
- Automatisches Nachziehen von Orphans, sobald der Parent angekommen ist
- Outbound-Peer-Diversitaet fuer Broadcast/Sync (Bucket nach /16 bei IPv4, /48 bei IPv6, Domain-Suffix bei DNS)

## P2P Sync (Schritt 2)

Sync nutzt jetzt zuerst einen Header-First-Pfad und faellt bei alten Peers auf Snapshot-Sync zurueck:

- `GET /headers/meta`
- `GET /headers?start_height=<H>&limit=<N>`
- `GET /blocks/range?start_height=<H>&limit=<N>`
- `GET /block/by-hash?hash=<BLOCK_HASH>`

`node-sync`/`/sync` liefert den verwendeten Modus im Ergebnis:
- `mode: "headers"` (bevorzugt)
- `mode: "snapshot"` (Fallback fuer Legacy-Peers)

CLI-Wrapper fuer die REST API:

```bash
python powx_cli.py api-status --node http://127.0.0.1:8844
python powx_cli.py api-balance --node http://127.0.0.1:8844 --address <ADDR>
python powx_cli.py api-chain --node http://127.0.0.1:8844 --limit 20
python powx_cli.py api-mempool --node http://127.0.0.1:8844
python powx_cli.py api-history --node http://127.0.0.1:8844 --address <ADDR> --limit 50
python powx_cli.py api-send --node http://127.0.0.1:8844 --wallet alice.json --to <ADDR> --amount 7 --fee 1
python powx_cli.py api-mine --node http://127.0.0.1:8844 --wallet alice.json --blocks 1 --backend auto
# optional bei sehr langsamem Mining: --timeout 180
# oder ohne Limit (Standard): --timeout 0
```

## Difficulty + Time Hardening (Schritt 7)

- Standard-Schedule fuer neue Chains: `asert-v3` (stabileres exponentielles Retargeting)
- Legacy/Window-Chains bleiben kompatibel (`legacy-v1`, `window-v2` werden beim Laden weiter erkannt)
- Strengere Timestamp-Regeln:
- `timestamp > Median-Time-Past` (bei `asert-v3` mit groesserem MTP-Fenster)
- Keine Zeit-Rueckspruenge (`timestamp` darf nicht kleiner als der Parent-Timestamp sein)
- Klare DoS-Grenze fuer Zeit-Spruenge pro Block (`max_block_timestamp_step_seconds`)
- Harte Future-Grenze bleibt aktiv (`max_future_block_seconds`)

Neue relevante Config-Felder:
- `target_schedule` (z.B. `asert-v3`)
- `asert_half_life`
- `mtp_window`
- `max_block_timestamp_step_seconds`

## Mempool Policy (Schritt 8)

- Fee-Rate-basierte Admission/Selektion (`fee / vsize`)
- Eviction bei vollem Mempool nach niedrigster Package-Fee-Rate
- Ancestor-/Descendant-Limits gegen unendliche Unconfirmed-Chains
- Optionales RBF (Ersatz konkurrierender Tx nur mit hoehrem Fee/Fee-Rate)
- Optionales CPFP-Package-Prioritizing beim Mining

Neue relevante Config-Felder:
- `max_mempool_virtual_bytes`
- `min_mempool_fee_rate`
- `mempool_ancestor_limit`
- `mempool_descendant_limit`
- `mempool_rbf_enabled`
- `mempool_cpfp_enabled`
- `max_rbf_replacements`
- `min_rbf_fee_delta`
- `min_rbf_feerate_delta`

## Befehle

- `wallet-new --out <file>`
- `wallet-address --wallet <file>`
- `init --data-dir <dir> [--genesis-wallet <file> | --genesis-address <addr>] [--genesis-supply N]`
- `status --data-dir <dir>`
- `balance --data-dir <dir> (--wallet <file> | --address <addr>)`
- `send --data-dir <dir> --wallet <file> --to <addr> --amount N [--fee N] [--broadcast-node <url>]`
- `mine --data-dir <dir> (--wallet <file> | --address <addr>) [--blocks N] [--backend auto|gpu|cpu] [--broadcast-node <url>]`
- `mempool --data-dir <dir>`
- `chain --data-dir <dir> [--limit N]`
- `node-run --data-dir <dir> --host <host> --port <port> [--peer <url> ...] [--max-peers N] [--max-inbound-ttl N] [--max-request-body-bytes N] [--max-requests-per-minute N] [--sync-retry-cooldown S] [--peer-ban-threshold N] [--peer-ban-seconds N] [--peer-penalty-invalid-tx N] [--peer-penalty-invalid-block N] [--peer-penalty-bad-sync N] [--peer-reward-success N] [--peer-auth-max-skew-seconds N] [--peer-auth-replay-window-seconds N]`
- `node-status --node <url>`
- `node-add-peer --node <url> --peer <url>`
- `node-sync --node <url> [--peer <url>]`
- `api-status --node <url>`
- `api-balance --node <url> (--address <addr> | --wallet <file>)`
- `api-chain --node <url> [--limit N]`
- `api-mempool --node <url>`
- `api-history --node <url> (--address <addr> | --wallet <file>) [--limit N]`
- `api-send --node <url> --wallet <file> --to <addr> --amount N [--fee N]`
- `api-mine --node <url> (--wallet <file> | --address <addr>) [--blocks N] [--backend auto|gpu|cpu] [--timeout S]`

## Hinweise

- Keys liegen unverschlüsselt in JSON-Dateien. Für echte Sicherheit: Hardware Wallet/HSM, Verschlüsselung, Key-Rotation.
- P2P bleibt HTTP-basiert, nutzt aber jetzt Header-First-Sync plus `inv/getdata`, Orphan-Handling und diversifizierte Outbound-Peer-Auswahl.
- Schritt 3 reduziert DoS-Risiken deutlich (Rate-Limit, Request-Size-Limit, TTL-Cap, Peer-Limit, Sync-Retry-Cooldown).
- Schritt 4 + 5 bringen Peer-Scoring/Temp-Ban, signierte Peer-Messages und persistente Reputationsdaten; fuer ein echtes Mainnet fehlen trotzdem noch robustere Anti-Sybil-/Peer-Discovery-/Reputation-Mechanismen.
- Die Chain validiert beim Laden den gesamten Blockverlauf neu und rekonstruiert UTXOs aus den Blöcken (State-Härtung gegen manipulierte Dateien).
- Aeltere `chain_state.json`-Dateien werden kompatibel erkannt (Legacy-Target-Schedule), damit bestehende lokale KK91-Daten weiter nutzbar bleiben.
- Bitcoin-aehnliche Geldpolitik ist aktiv: Halving (`halving_interval=210000`) + feste Obergrenze (`max_total_supply=911000000`).
- Standard fuer `init` ist `--genesis-supply 0` (kein Premine). Wenn du einen Premine setzt, wird der Rest fuer Mining entsprechend kleiner.
- Schritt 1 umgesetzt: Upgrade-Framework mit `protocol_version` und `protocol_upgrade_v2_height` (Status zeigt aktive/naechste Protokollversion).
- Neue Blöcke mit zu weit in der Zukunft liegenden Timestamps werden abgelehnt (`max_future_block_seconds`).
- Difficulty ist standardmaessig auf ASERT (`asert-v3`) umgestellt; `window-v2` wird fuer bestehende Chains weiterhin unterstuetzt.
- Difficulty/PoW-Parameter sind lokal auf praktikables Mining eingestellt.
- Wenn Mining zu schnell ist, nutze einen neuen Data-Ordner (`--data-dir`), damit die aktuelle Difficulty-Konfiguration greift.
- Sehr schnelles Mining ist in einem lokalen Testnetz normal und beabsichtigt; es ist kein Mainnet-Difficulty-Profil.
