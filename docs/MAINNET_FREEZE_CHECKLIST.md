# KK91 Mainnet Freeze Checklist

Diese Checkliste friert Konsens und Release-Artefakte vor Mainnet ein.

## 1. Konsens-Parameter final setzen

- `powx/config.py` final pruefen:
- `symbol`, `chain_id`, `pow_algorithm`
- `consensus_lock_enabled=true`
- `max_total_supply`, `initial_block_reward`, `halving_interval`
- `target_schedule=asert-v3`, `target_block_time`, `asert_half_life`
- `protocol_upgrade_v2_height` explizit (kein "infinite" Platzhalter)
- `fixed_genesis_hash`, `fixed_genesis_txid`, `fixed_genesis_timestamp`
- `fixed_genesis_address`, `fixed_genesis_supply`, `fixed_genesis_tx_nonce`, `fixed_genesis_block_nonce`
- `max_target`, `initial_target`, Zeitregeln und Reorg-Limits
- Keine offenen TODOs in Konsens-/Validierungslogik.

## 2. Wallet-/Key-Security erzwingen

- Wallet-Dateien nur verschluesselt erzeugen (`scrypt`, optional `argon2id`).
- Seed-Backup-Check mindestens einmal pro Seed-Wallet durchfuehren.
- Private Keys niemals an Nodes uebertragen.
- Fuer Node-API nur `tx/build` + lokal signieren + `tx/submit` verwenden.

## 3. Soak-Test im lokalen Cluster

Mehrstundigen Lauf mit mehreren Nodes durchfuehren:

```bash
cd "d:\Projekt X\powx"
python tools/soak_network.py --root-dir ./soak_run --nodes 3 --duration-seconds 7200 --backend cpu
```

Erwartung:
- Keine Abstuerze
- Regelmaessige Blockproduktion
- Kleine Height-Drift zwischen Nodes
- Keine dauerhaften Sync-Fehler

## 4. Freeze-Snapshot erzeugen

Snapshot fuer Konsens-Konfiguration + aktuelle Chain schreiben:

```bash
cd "d:\Projekt X\powx"
python tools/mainnet_freeze_snapshot.py --data-dir ./data --out ./docs/mainnet_freeze_snapshot.json
```

Ergebnis:
- `consensus_config_hash`
- `freeze_snapshot_hash`
- Chain-Metadaten (genesis/tip/height)

Diese Hashes im Release-Tag und in Release-Notes dokumentieren.

## 5. Release/Tag

- Version taggen (z. B. `v1.0.0-mainnet-candidate`).
- Binary/Source-Artefakte reproduzierbar bauen.
- README + API-Doku auf finalen Stand bringen.

## 6. Go/No-Go Kriterien

- Voller Testlauf gruen (`python -m unittest discover tests -v`).
- Soak-Test abgeschlossen ohne kritische Fehler.
- Freeze-Snapshot-Hash final dokumentiert.
- Kein geplanter Konsens-Breaking Change mehr offen.

Nach Go-Live sind Konsens-aendernde Updates nur noch via klar dokumentiertem Hardfork-Plan moeglich.
