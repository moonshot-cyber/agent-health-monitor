#!/usr/bin/env python3
"""Assess and reverse the 7 October 2026 chain-data outage.

Between 2026-10-07 and 2026-10-27 the Base chain data API sat behind a bot
challenge. monitor.fetch_transactions returned [] instead of raising, so every
wallet scanned in that window was scored from no data at all: tx_count 0,
D1 75, D2 50, composite 58, Grade D, written with full confidence.

The scans table is append-only, so the pre-outage scores still exist. Only
known_wallets.latest_* was overwritten, and that is derivable from scans. This
script quantifies what is recoverable and, with --apply, restores it.

    python scripts/recover_outage_scans.py --db /data/ahm_history.db
    python scripts/recover_outage_scans.py --db /data/ahm_history.db --apply

Poisoned scans are MOVED to scans_quarantine, never deleted, so --apply is
reversible with --rollback.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUTAGE_START = "2026-10-07"

# A poisoned scan is one produced from no transaction data at all. tx_count 0
# together with the D2 no-data baseline is the signature; the 58 and the Grade
# D follow arithmetically from 0.30*75 + 0.70*50.
POISON_WHERE = """
    endpoint = 'ahs'
    AND scan_timestamp >= :since
    AND COALESCE(tx_count, 0) = 0
    AND d2_score = 50
    AND ahs_score = 58
"""


def connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        sys.exit(
            f"No database at {path}.\n"
            f"Pass --db, and run this where the Railway volume is mounted."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_quarantine(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scans_quarantine AS "
        "SELECT *, '' AS quarantined_at, '' AS quarantine_reason FROM scans WHERE 0"
    )


def assess(conn: sqlite3.Connection, since: str) -> dict:
    def q(sql):
        return conn.execute(sql, {"since": since}).fetchall()

    total_window = q(
        "SELECT COUNT(*) n FROM scans "
        "WHERE endpoint='ahs' AND scan_timestamp >= :since"
    )[0]["n"]
    poisoned = q(f"SELECT COUNT(*) n FROM scans WHERE {POISON_WHERE}")[0]["n"]
    addrs = q(f"SELECT COUNT(DISTINCT address) n FROM scans WHERE {POISON_WHERE}")[0]["n"]

    # Provably wrong per wallet: the same address has an earlier scan that DID
    # see transactions. An outage cannot erase a wallet's history, so a later
    # tx_count of 0 is the scanner failing, not the wallet changing.
    recoverable = q(f"""
        SELECT COUNT(DISTINCT p.address) n FROM scans p
        WHERE {POISON_WHERE}
          AND EXISTS (SELECT 1 FROM scans g
                      WHERE g.address = p.address AND g.endpoint = 'ahs'
                        AND g.scan_timestamp < :since
                        AND COALESCE(g.tx_count, 0) > 0
                        AND g.ahs_score IS NOT NULL)
    """)[0]["n"]

    orphans = q(f"""
        SELECT COUNT(DISTINCT p.address) n FROM scans p
        WHERE {POISON_WHERE}
          AND NOT EXISTS (SELECT 1 FROM scans g
                          WHERE g.address = p.address AND g.endpoint = 'ahs'
                            AND g.scan_timestamp < :since
                            AND g.ahs_score IS NOT NULL)
    """)[0]["n"]

    by_source = q(
        f"SELECT source, COUNT(*) n FROM scans WHERE {POISON_WHERE} "
        f"GROUP BY source ORDER BY n DESC"
    )

    now_grades = Counter({r["grade"]: r["n"] for r in q("""
        SELECT grade, COUNT(*) n FROM (
            SELECT address, grade, ROW_NUMBER() OVER
                (PARTITION BY address ORDER BY scan_timestamp DESC) rn
            FROM scans WHERE endpoint='ahs' AND ahs_score IS NOT NULL
        ) WHERE rn=1 AND grade IS NOT NULL GROUP BY grade
    """)})

    after_grades = Counter({r["grade"]: r["n"] for r in q(f"""
        SELECT grade, COUNT(*) n FROM (
            SELECT address, grade, ROW_NUMBER() OVER
                (PARTITION BY address ORDER BY scan_timestamp DESC) rn
            FROM scans
            WHERE endpoint='ahs' AND ahs_score IS NOT NULL
              AND id NOT IN (SELECT id FROM scans WHERE {POISON_WHERE})
        ) WHERE rn=1 AND grade IS NOT NULL GROUP BY grade
    """)})

    return {
        "total_window": total_window, "poisoned": poisoned, "addrs": addrs,
        "recoverable": recoverable, "orphans": orphans, "by_source": by_source,
        "now": now_grades, "after": after_grades,
    }


def report(a: dict, since: str) -> None:
    pct = f" ({a['poisoned'] / a['total_window'] * 100:.1f}% of the window)" if a["total_window"] else ""
    print(f"  outage window       : {since} -> now")
    print(f"  AHS scans in window : {a['total_window']:,}")
    print(f"  matching poison     : {a['poisoned']:,}{pct}")
    print(f"  distinct wallets    : {a['addrs']:,}")
    print()
    print(f"  RECOVERABLE         : {a['recoverable']:,} wallets have an earlier scan "
          f"that saw real transactions")
    print(f"  no prior good scan  : {a['orphans']:,} wallets were only ever scored "
          f"during the outage")

    if a["by_source"]:
        print("\n  poisoned scans by source:")
        for r in a["by_source"]:
            print(f"    {(r['source'] or '(none)'):<18} {r['n']:>8,}")

    print("\n  grade distribution, latest scan per wallet:")
    print(f"    {'grade':<7} {'now':>9} {'after recovery':>16} {'change':>10}")
    for g in "ABCDEF":
        n, af = a["now"].get(g, 0), a["after"].get(g, 0)
        print(f"    {g:<7} {n:>9,} {af:>16,} {af - n:>+10,}")
    tn, ta = sum(a["now"].values()), sum(a["after"].values())
    print(f"    {'TOTAL':<7} {tn:>9,} {ta:>16,} {ta - tn:>+10,}")
    if ta < tn:
        print(f"\n    ({tn - ta:,} wallets drop out entirely - they have no scan "
              f"outside the outage window)")


def apply_recovery(conn: sqlite3.Connection, since: str) -> None:
    ensure_quarantine(conn)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = f"chain-data outage from {since}"

    with conn:
        cur = conn.execute(
            f"INSERT INTO scans_quarantine SELECT *, :stamp, :reason "
            f"FROM scans WHERE {POISON_WHERE}",
            {"since": since, "stamp": stamp, "reason": reason},
        )
        moved = cur.rowcount
        conn.execute(f"DELETE FROM scans WHERE {POISON_WHERE}", {"since": since})

        # Rebuild the denormalised cache from whatever scans survive. The
        # dashboard reads scans directly, but /api/agent and the leaderboard
        # read known_wallets, so both have to move together.
        conn.execute("""
            UPDATE known_wallets SET
                latest_ahs = (SELECT s.ahs_score FROM scans s
                    WHERE s.address = known_wallets.address AND s.endpoint='ahs'
                      AND s.ahs_score IS NOT NULL
                    ORDER BY s.scan_timestamp DESC LIMIT 1),
                latest_grade = (SELECT s.grade FROM scans s
                    WHERE s.address = known_wallets.address AND s.endpoint='ahs'
                      AND s.ahs_score IS NOT NULL
                    ORDER BY s.scan_timestamp DESC LIMIT 1),
                latest_d1 = (SELECT s.d1_score FROM scans s
                    WHERE s.address = known_wallets.address AND s.endpoint='ahs'
                      AND s.ahs_score IS NOT NULL
                    ORDER BY s.scan_timestamp DESC LIMIT 1),
                latest_d2 = (SELECT s.d2_score FROM scans s
                    WHERE s.address = known_wallets.address AND s.endpoint='ahs'
                      AND s.ahs_score IS NOT NULL
                    ORDER BY s.scan_timestamp DESC LIMIT 1),
                last_scanned_at = COALESCE(
                    (SELECT s.scan_timestamp FROM scans s
                     WHERE s.address = known_wallets.address AND s.endpoint='ahs'
                     ORDER BY s.scan_timestamp DESC LIMIT 1),
                    known_wallets.last_scanned_at)
        """)
    print(f"  quarantined {moved:,} scans and rebuilt known_wallets")


def rollback(conn: sqlite3.Connection) -> None:
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(scans)")]
        cl = ", ".join(cols)
        with conn:
            n = conn.execute(
                f"INSERT INTO scans ({cl}) SELECT {cl} FROM scans_quarantine"
            ).rowcount
            conn.execute("DELETE FROM scans_quarantine")
        print(f"  restored {n:,} scans from quarantine")
    except sqlite3.OperationalError as e:
        sys.exit(f"  nothing to roll back ({e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./ahm_history.db", type=Path)
    ap.add_argument("--since", default=OUTAGE_START)
    ap.add_argument("--apply", action="store_true",
                    help="quarantine poisoned scans and rebuild known_wallets")
    ap.add_argument("--rollback", action="store_true", help="undo --apply")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    conn = connect(args.db)

    if args.rollback:
        rollback(conn)
        return 0

    a = assess(conn, args.since)
    report(a, args.since)

    if not args.apply:
        print("\n  Read-only. Re-run with --apply to quarantine and rebuild.")
        return 0
    if not a["poisoned"]:
        print("\n  Nothing to do.")
        return 0

    if not args.no_backup:
        backup = args.db.with_suffix(
            f".pre-recovery-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.db")
        shutil.copy2(args.db, backup)
        print(f"\n  backup written to {backup}")

    apply_recovery(conn, args.since)
    print("\n  after:")
    report(assess(conn, args.since), args.since)
    print("\n  Reverse with --rollback if this looks wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
