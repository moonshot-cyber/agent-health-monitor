"""Generate a JSON snapshot for the AHM Intelligence dashboard.

Connects directly to the AHM SQLite database and produces a JSON file
containing ecosystem KPIs, per-registry breakdowns, and daily trend data.

Usage:
    python scripts/generate_intelligence_snapshot.py --db data/ahm.db --output snapshot.json
    python scripts/generate_intelligence_snapshot.py --db data/ahm.db  # stdout
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def generate_snapshot(db_path: str) -> dict:
    conn = _connect(db_path)
    try:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Ecosystem stats ---
        row = conn.execute(
            "SELECT COUNT(DISTINCT address) FROM scans "
            "WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL"
        ).fetchone()
        total_addresses = row[0]

        row = conn.execute("SELECT COUNT(*) FROM scans").fetchone()
        total_scans = row[0]

        avg_row = conn.execute(
            """SELECT AVG(ahs_score) FROM (
                SELECT address, ahs_score,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            ) WHERE rn = 1"""
        ).fetchone()
        avg_score = round(avg_row[0], 1) if avg_row[0] is not None else 0

        grade_rows = conn.execute(
            """SELECT grade, COUNT(*) as cnt FROM (
                SELECT address, grade,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND grade IS NOT NULL
            ) WHERE rn = 1 GROUP BY grade ORDER BY grade"""
        ).fetchall()
        grade_distribution = {r["grade"]: r["cnt"] for r in grade_rows}

        ecosystem = {
            "total_addresses": total_addresses,
            "total_scans": total_scans,
            "avg_score": avg_score,
            "grade_distribution": grade_distribution,
        }

        # --- Per-registry breakdowns ---
        # known_wallets.registries is a comma-separated string like "acp,erc8004,virtuals"
        wallet_rows = conn.execute(
            "SELECT address, registries, latest_ahs, latest_grade "
            "FROM known_wallets WHERE latest_ahs IS NOT NULL AND registries IS NOT NULL AND registries != ''"
        ).fetchall()

        registry_data = {}  # name -> {scores: [], grades: []}
        for w in wallet_rows:
            regs = [r.strip().lower() for r in w["registries"].split(",") if r.strip()]
            for reg in regs:
                if reg not in registry_data:
                    registry_data[reg] = {"scores": [], "grades": []}
                registry_data[reg]["scores"].append(w["latest_ahs"])
                if w["latest_grade"]:
                    registry_data[reg]["grades"].append(w["latest_grade"])

        # Pretty display names — match against substring in the raw registry key
        def _display_name(key: str) -> str:
            k = key.lower()
            if "acp" in k:
                return "ACP"
            if "erc8004" in k or "erc-8004" in k:
                return "ERC-8004"
            if "erc8183" in k or "erc-8183" in k:
                return "ERC-8183"
            if "virtual" in k:
                return "Virtuals"
            if "olas" in k:
                return "Olas"
            if "celo" in k:
                return "Celo"
            if "arc" in k:
                return "Arc"
            return key

        # Consolidate raw keys into display names
        consolidated = {}  # display_name -> {scores: [], grades: []}
        for reg_key, data in registry_data.items():
            name = _display_name(reg_key)
            if name not in consolidated:
                consolidated[name] = {"scores": [], "grades": []}
            consolidated[name]["scores"].extend(data["scores"])
            consolidated[name]["grades"].extend(data["grades"])

        registries = []
        for name, data in sorted(consolidated.items(), key=lambda x: -len(x[1]["scores"])):
            scores = data["scores"]
            grades = data["grades"]
            grade_dist = {}
            for g in grades:
                grade_dist[g] = grade_dist.get(g, 0) + 1
            top_count = grade_dist.get("A", 0) + grade_dist.get("B", 0)
            total_graded = len(grades)
            registries.append({
                "name": name,
                "address_count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "grade_distribution": grade_dist,
                "top_grade_pct": round(top_count / total_graded * 100, 1) if total_graded > 0 else 0,
            })

        # --- Daily stats (last 30 days) ---
        daily_rows = conn.execute(
            """SELECT DATE(scan_timestamp) as day, COUNT(*) as scan_count, AVG(ahs_score) as avg_score
            FROM scans
            WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
              AND scan_timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(scan_timestamp)
            ORDER BY day"""
        ).fetchall()
        daily_stats = [
            {
                "date": r["day"],
                "scan_count": r["scan_count"],
                "avg_score": round(r["avg_score"], 1) if r["avg_score"] is not None else 0,
            }
            for r in daily_rows
        ]

        return {
            "generated_at": now_iso,
            "ecosystem": ecosystem,
            "registries": registries,
            "daily_stats": daily_stats,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate AHM Intelligence snapshot")
    parser.add_argument("--db", default="data/ahm.db", help="Path to AHM SQLite database")
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    snapshot = generate_snapshot(args.db)
    output = json.dumps(snapshot, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            f.write("\n")
        print(f"Snapshot written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
