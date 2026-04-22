"""Generate a JSON snapshot for the AHM Intelligence dashboard.

Two modes:
  --db <path>    Direct SQLite access (for use on Railway or with a local copy)
  --api <url>    Fetch from public AHM API endpoints (no auth required)

Usage:
    python scripts/generate_intelligence_snapshot.py --db /data/ahm.db --output snapshot.json
    python scripts/generate_intelligence_snapshot.py --api https://agenthealthmonitor.xyz --output snapshot.json
"""

import argparse
import json
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone


def _display_name(key: str) -> str:
    """Map raw registry/source keys to clean display names."""
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


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "AHM-Intelligence-Snapshot/1.0",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def generate_snapshot_from_api(base_url: str) -> dict:
    """Build snapshot from public AHM API endpoints (no auth required)."""
    base = base_url.rstrip("/")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # /api/ecosystem-stats — public
    eco = _fetch_json(f"{base}/api/ecosystem-stats")
    ecosystem = {
        "total_addresses": eco["total_scanned"],
        "total_scans": eco["total_scanned"],  # best available from public API
        "avg_score": eco["avg_ahs"],
        "grade_distribution": eco["grade_distribution"],
    }

    # Per-registry: build from data_sources counts + batch quality for scores
    batches = _fetch_json(f"{base}/scan/quality").get("batches", [])

    # Aggregate batch data per registry
    reg_agg = {}  # name -> {total_score_sum, total_count, grade_dist}
    for b in batches:
        src = b.get("source", "").lower()
        if not src:
            continue
        name = _display_name(src)
        if name not in reg_agg:
            reg_agg[name] = {"score_sum": 0, "count": 0, "grade_dist": {}}
        reg_agg[name]["score_sum"] += b["average_ahs"] * b["wallets_scanned"]
        reg_agg[name]["count"] += b["wallets_scanned"]
        for g, cnt in b.get("grade_distribution", {}).items():
            reg_agg[name]["grade_dist"][g] = reg_agg[name]["grade_dist"].get(g, 0) + cnt

    # Merge with data_sources for address counts
    registries = []
    for name, addr_count in sorted(eco.get("data_sources", {}).items(), key=lambda x: -x[1]):
        agg = reg_agg.get(name, {})
        grade_dist = agg.get("grade_dist", {})
        avg = round(agg["score_sum"] / agg["count"], 1) if agg.get("count") else 0
        top_count = grade_dist.get("A", 0) + grade_dist.get("B", 0)
        total_graded = sum(grade_dist.values())
        registries.append({
            "name": name,
            "address_count": addr_count,
            "avg_score": avg,
            "grade_distribution": grade_dist,
            "top_grade_pct": round(top_count / total_graded * 100, 1) if total_graded > 0 else 0,
        })

    # Daily stats from batch quality (aggregate per day)
    daily_agg = {}  # date -> {scan_count, score_sum}
    for b in batches:
        day = b["batch_date"][:10]
        if day not in daily_agg:
            daily_agg[day] = {"scan_count": 0, "score_sum": 0, "count": 0}
        daily_agg[day]["scan_count"] += b["wallets_scanned"]
        daily_agg[day]["score_sum"] += b["average_ahs"] * b["wallets_scanned"]
        daily_agg[day]["count"] += b["wallets_scanned"]

    daily_stats = [
        {
            "date": day,
            "scan_count": d["scan_count"],
            "avg_score": round(d["score_sum"] / d["count"], 1) if d["count"] else 0,
        }
        for day, d in sorted(daily_agg.items())
    ]

    return {
        "generated_at": now_iso,
        "ecosystem": ecosystem,
        "registries": registries,
        "daily_stats": daily_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate AHM Intelligence snapshot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--db", help="Path to AHM SQLite database")
    group.add_argument("--api", help="AHM API base URL (e.g. https://agenthealthmonitor.xyz)")
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    if args.db:
        snapshot = generate_snapshot(args.db)
    else:
        snapshot = generate_snapshot_from_api(args.api)

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
