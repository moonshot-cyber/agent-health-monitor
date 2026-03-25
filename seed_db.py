#!/usr/bin/env python3
"""One-shot seed script: imports scan data from db_export.json into the production DB.

Append-only — skips rows that already exist (by checking scan count).
Run via: python seed_db.py

Expects db_export.json in the same directory.
"""

import json
import os
import sqlite3
import sys

DB_PATH = os.getenv("DB_PATH", "./ahm_history.db")
EXPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_export.json")


def main():
    if not os.path.isfile(EXPORT_FILE):
        print(f"ERROR: {EXPORT_FILE} not found")
        sys.exit(1)

    print(f"DB_PATH: {DB_PATH}")
    print(f"Export file: {EXPORT_FILE}")

    # Ensure DB and tables exist
    import db
    db.init_db()

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Check existing row counts
    existing_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    existing_patterns = conn.execute("SELECT COUNT(*) FROM scan_patterns").fetchone()[0]
    existing_wallets = conn.execute("SELECT COUNT(*) FROM known_wallets").fetchone()[0]
    print(f"\nExisting data: {existing_scans} scans, {existing_patterns} patterns, {existing_wallets} wallets")

    with open(EXPORT_FILE, "r") as f:
        data = json.load(f)

    import_scans = data["scans"]
    import_patterns = data["scan_patterns"]
    import_wallets = data["known_wallets"]
    print(f"Import file:  {len(import_scans)} scans, {len(import_patterns)} patterns, {len(import_wallets)} wallets")

    if existing_scans >= len(import_scans):
        print("\nProduction DB already has >= import data. Nothing to do.")
        conn.close()
        return

    # Build set of existing addresses for wallets (to avoid duplicates)
    existing_addrs = set()
    for row in conn.execute("SELECT address FROM known_wallets"):
        existing_addrs.add(row[0])

    # Build map: old scan ID -> new scan ID (for pattern foreign keys)
    old_to_new_scan_id = {}

    inserted_scans = 0
    inserted_patterns = 0
    inserted_wallets = 0

    # Insert scans (append only — use INSERT OR IGNORE won't work since id is autoincrement)
    # We'll skip if a scan with same address+endpoint+scan_timestamp exists
    for scan in import_scans:
        existing = conn.execute(
            "SELECT id FROM scans WHERE address = ? AND endpoint = ? AND scan_timestamp = ?",
            (scan["address"], scan["endpoint"], scan["scan_timestamp"]),
        ).fetchone()

        if existing:
            old_to_new_scan_id[scan["id"]] = existing[0]
            continue

        cols = [k for k in scan.keys() if k != "id"]
        vals = [scan[k] for k in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        cur = conn.execute(f"INSERT INTO scans ({col_names}) VALUES ({placeholders})", vals)
        old_to_new_scan_id[scan["id"]] = cur.lastrowid
        inserted_scans += 1

    # Insert patterns (mapped to new scan IDs)
    for pat in import_patterns:
        new_scan_id = old_to_new_scan_id.get(pat["scan_id"])
        if new_scan_id is None:
            continue

        existing = conn.execute(
            "SELECT id FROM scan_patterns WHERE scan_id = ? AND pattern_name = ?",
            (new_scan_id, pat["pattern_name"]),
        ).fetchone()

        if existing:
            continue

        conn.execute(
            "INSERT INTO scan_patterns (scan_id, pattern_name, severity, description, modifier) VALUES (?, ?, ?, ?, ?)",
            (new_scan_id, pat["pattern_name"], pat["severity"], pat["description"], pat["modifier"]),
        )
        inserted_patterns += 1

    # Upsert known_wallets
    for w in import_wallets:
        if w["address"] in existing_addrs:
            continue

        cols = list(w.keys())
        vals = [w[k] for k in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        conn.execute(f"INSERT OR IGNORE INTO known_wallets ({col_names}) VALUES ({placeholders})", vals)
        inserted_wallets += 1

    conn.commit()
    conn.close()

    print(f"\nInserted: {inserted_scans} scans, {inserted_patterns} patterns, {inserted_wallets} wallets")
    print("Done.")


if __name__ == "__main__":
    main()
