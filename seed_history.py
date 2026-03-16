#!/usr/bin/env python3
"""Seed the scan history database with existing ecosystem scan results."""

import csv
import db


ECOSYSTEM_CSV = "ecosystem_scan_results.csv"
ECOSYSTEM_TIMESTAMP = "2026-03-16T15:00:00Z"

# Outreach scan results from outreach_scan.py run on Mar 16 2026
OUTREACH_RESULTS = [
    # (label, address, ahs, grade, grade_label, confidence, d1, d2, patterns, tx_count, history_days)
    ("PayAI (signer 1)", "0xc6699d2aada6c36dfea5c248dd70f9cb0235cb63", 51, "D", "Degraded", "LOW", 63, 46, None, 0, 0),
    ("PayAI (signer 2)", "0xb2bd29925cbbcea7628279c91945ca5b98bf371b", 61, "C", "Needs Attention", "LOW", 62, 61, None, 0, 0),
    ("PayAI (example)", "0x209693Bc6afc0C5328bA36FaF03C514EF312287C", 58, "D", "Degraded", "LOW", 75, 50, None, 0, 0),
    ("Daydreams (EVM signer)", "0x1363C7Ff51CcCE10258A7F7bddd63bAaB6aAf678", 56, "D", "Degraded", "LOW", 64, 52, None, 0, 0),
    ("Daydreams (facilitator)", "0x279e08f711182c79Ba6d09669127a426228a4653", 38, "E", "Critical", "LOW", 67, 40,
     [{"name": "Stale Strategy", "severity": "warning", "description": "Agent is repeatedly failing on the same contract interaction without adapting.", "modifier": -10}], 0, 0),
    ("Daydreams (payTo example)", "0xb308ed39d67D0d4BAe5BC2FAEF60c66BBb6AE429", 82, "B", "Good", "LOW", 89, 79, None, 0, 0),
]

OUTREACH_TIMESTAMP = "2026-03-16T15:37:00Z"


def main():
    db.init_db()
    conn = db.get_connection()

    # Check if seed data already exists
    row = conn.execute("SELECT COUNT(*) FROM scans WHERE source IN ('ecosystem_scan', 'outreach_scan')").fetchone()
    if row[0] > 0:
        print(f"[!] Seed data already present ({row[0]} rows). Skipping.")
        conn.close()
        return

    conn.close()

    count = 0

    # Import ecosystem CSV
    with open(ECOSYSTEM_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ahs"] == "ERROR" or not row["ahs"].isdigit():
                continue
            db.log_scan(
                address=row["address"],
                endpoint="ahs",
                scan_timestamp=ECOSYSTEM_TIMESTAMP,
                source="ecosystem_scan",
                label=row["source"],
                ahs_score=int(row["ahs"]),
                grade=row["grade"],
                grade_label=row["grade_label"],
                confidence=row["confidence"],
                mode="2D",
                d1_score=int(row["d1_wallet"]) if row["d1_wallet"] else None,
                d2_score=int(row["d2_behaviour"]) if row["d2_behaviour"] else None,
                tx_count=int(row["tx_count"]) if row["tx_count"] else None,
                history_days=int(row["history_days"]) if row["history_days"] else None,
                response_data=dict(row),
            )
            count += 1
            print(f"  [+] {row['source']}: AHS {row['ahs']}/{row['grade']}")

    # Import outreach results
    for label, addr, ahs, grade, grade_label, confidence, d1, d2, patterns, tx_count, history_days in OUTREACH_RESULTS:
        db.log_scan(
            address=addr,
            endpoint="ahs",
            scan_timestamp=OUTREACH_TIMESTAMP,
            source="outreach_scan",
            label=label,
            ahs_score=ahs,
            grade=grade,
            grade_label=grade_label,
            confidence=confidence,
            mode="2D",
            d1_score=d1,
            d2_score=d2,
            patterns=patterns,
            tx_count=tx_count,
            history_days=history_days,
            response_data={"ahs": ahs, "grade": grade, "d1": d1, "d2": d2},
        )
        count += 1
        print(f"  [+] {label}: AHS {ahs}/{grade}")

    print(f"\n[+] Seeded {count} scan records.")

    # Show stats
    conn = db.get_connection()
    row = conn.execute("SELECT COUNT(*) FROM scans").fetchone()
    wallets = conn.execute("SELECT COUNT(*) FROM known_wallets").fetchone()
    print(f"[+] Total scans: {row[0]}, Known wallets: {wallets[0]}")
    conn.close()


if __name__ == "__main__":
    main()
