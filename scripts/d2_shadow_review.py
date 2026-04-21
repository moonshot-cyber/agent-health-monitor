#!/usr/bin/env python3
"""D2 Shadow Score Review -- Gate Analysis for D2 Promotion to Live Weighting.

Queries the production AHM database and produces a readout covering:
  1. D2 shadow score distribution (bucketed)
  2. D2 vs current AHS grade correlation
  3. Simulated AHS with D2 at live 70% weight -- grade migration
  4. Coverage -- % of agents with a D2 shadow score
  5. Outliers -- agents where D2 diverges >20 points from current AHS

Output goes to stdout and is saved to docs/d2-shadow-review-YYYYMMDD.txt.

Usage:
    python scripts/d2_shadow_review.py                     # uses default ./ahm_history.db
    python scripts/d2_shadow_review.py --db /path/to/db    # explicit DB path
    DB_PATH=/path/to/db python scripts/d2_shadow_review.py # via env var
"""

import json
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("DB_PATH", "./ahm_history.db")

GRADE_BOUNDARIES = [
    ("A", 90, 100),
    ("B", 75, 89),
    ("C", 60, 74),
    ("D", 40, 59),
    ("E", 20, 39),
    ("F", 0, 19),
]

D2_BUCKETS = [
    ("0-20", 0, 20),
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
]

# Current production weights (2D mode -- the dominant mode)
D1_WEIGHT_2D = 0.30
D2_WEIGHT_2D = 0.70

# 3D mode weights
D1_WEIGHT_3D = 0.25
D2_WEIGHT_3D = 0.45
D3_WEIGHT_3D = 0.30

DIVERGENCE_THRESHOLD = 20  # points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grade_from_score(score: int) -> str:
    """Map AHS score to letter grade -- mirrors monitor._ahs_grade()."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    if score >= 20:
        return "E"
    return "F"


def median(values: list[int | float]) -> float:
    """Median that handles empty lists."""
    if not values:
        return 0.0
    return statistics.median(values)


def bar(count: int, total: int, width: int = 40) -> str:
    """ASCII bar for histogram."""
    if total == 0:
        return ""
    filled = round(count / total * width)
    return "#" * filled + "." * (width - filled)


def pct(count: int, total: int) -> str:
    """Percentage string."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_latest_scans(db_path: str) -> list[dict]:
    """Fetch the latest AHS scan per address with D1, D2, D3, AHS scores.

    Returns list of dicts with keys:
        address, ahs_score, grade, d1_score, d2_score, d3_score, mode,
        cdp_modifier, shadow_signals_json
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT address, ahs_score, grade, d1_score, d2_score, d3_score,
                   mode, cdp_modifier, shadow_signals_json
            FROM (
                SELECT address, ahs_score, grade, d1_score, d2_score, d3_score,
                       mode, cdp_modifier, shadow_signals_json,
                       ROW_NUMBER() OVER (
                           PARTITION BY address
                           ORDER BY scan_timestamp DESC
                       ) AS rn
                FROM scans
                WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            )
            WHERE rn = 1
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analysis Sections
# ---------------------------------------------------------------------------

def section_coverage(scans: list[dict]) -> list[str]:
    """Section 1: Coverage -- what % of agents have a D2 score."""
    total = len(scans)
    with_d2 = sum(1 for s in scans if s["d2_score"] is not None)
    without_d2 = total - with_d2

    # Shadow signals coverage
    with_shadow = sum(
        1 for s in scans
        if s.get("shadow_signals_json") and s["shadow_signals_json"] != "null"
    )
    with_session = 0
    for s in scans:
        raw = s.get("shadow_signals_json")
        if raw and raw != "null":
            try:
                ss = json.loads(raw)
                if ss.get("session_continuity_score") is not None:
                    with_session += 1
            except (json.JSONDecodeError, TypeError):
                pass

    lines = [
        "=" * 72,
        "1. COVERAGE",
        "=" * 72,
        "",
        f"  Total unique agents scanned (AHS):  {total}",
        f"  Agents with D2 score:               {with_d2}  ({pct(with_d2, total)})",
        f"  Agents without D2 score:            {without_d2}  ({pct(without_d2, total)})",
        "",
        f"  Agents with shadow_signals data:    {with_shadow}  ({pct(with_shadow, total)})",
        f"  Agents with session_continuity:     {with_session}  ({pct(with_session, total)})",
        "",
    ]
    return lines


def section_distribution(scans: list[dict]) -> list[str]:
    """Section 2: D2 score distribution (bucketed histogram)."""
    d2_scores = [s["d2_score"] for s in scans if s["d2_score"] is not None]
    total = len(d2_scores)

    if total == 0:
        return [
            "=" * 72,
            "2. D2 SCORE DISTRIBUTION",
            "=" * 72,
            "",
            "  No D2 scores found in database.",
            "",
        ]

    avg = statistics.mean(d2_scores)
    med = statistics.median(d2_scores)
    std = statistics.stdev(d2_scores) if len(d2_scores) >= 2 else 0.0

    lines = [
        "=" * 72,
        "2. D2 SCORE DISTRIBUTION",
        "=" * 72,
        "",
        f"  Total agents with D2 score:  {total}",
        f"  Mean:    {avg:.1f}",
        f"  Median:  {med:.1f}",
        f"  Std Dev: {std:.1f}",
        f"  Min:     {min(d2_scores)}",
        f"  Max:     {max(d2_scores)}",
        "",
        f"  {'Bucket':<10} {'Count':>6} {'Pct':>8}   Distribution",
        f"  {'-' * 10} {'-' * 6} {'-' * 8}   {'-' * 40}",
    ]

    for label, lo, hi in D2_BUCKETS:
        count = sum(1 for s in d2_scores if lo <= s <= hi)
        lines.append(
            f"  {label:<10} {count:>6} {pct(count, total):>8}   {bar(count, total)}"
        )

    lines.append("")
    return lines


def section_grade_correlation(scans: list[dict]) -> list[str]:
    """Section 3: D2 vs current AHS grade correlation."""
    grade_buckets: dict[str, list[int]] = {g: [] for g, _, _ in GRADE_BOUNDARIES}

    for s in scans:
        if s["d2_score"] is None or s["grade"] is None:
            continue
        grade = s["grade"]
        if grade in grade_buckets:
            grade_buckets[grade].append(s["d2_score"])

    lines = [
        "=" * 72,
        "3. D2 vs CURRENT AHS GRADE CORRELATION",
        "=" * 72,
        "",
        f"  {'Grade':<7} {'Count':>6} {'Avg D2':>8} {'Med D2':>8} {'Min':>5} {'Max':>5} {'StdDev':>8}",
        f"  {'-' * 7} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 5} {'-' * 8}",
    ]

    for grade, _, _ in GRADE_BOUNDARIES:
        scores = grade_buckets[grade]
        if not scores:
            lines.append(f"  {grade:<7} {0:>6} {'--':>8} {'--':>8} {'--':>5} {'--':>5} {'--':>8}")
            continue
        avg = statistics.mean(scores)
        med = statistics.median(scores)
        sd = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        lines.append(
            f"  {grade:<7} {len(scores):>6} {avg:>8.1f} {med:>8.1f} "
            f"{min(scores):>5} {max(scores):>5} {sd:>8.1f}"
        )

    # Directional consistency check
    grade_avgs = []
    for grade, _, _ in GRADE_BOUNDARIES:
        scores = grade_buckets[grade]
        if scores:
            grade_avgs.append((grade, statistics.mean(scores)))

    monotonic = True
    for i in range(1, len(grade_avgs)):
        if grade_avgs[i][1] > grade_avgs[i - 1][1]:
            monotonic = False
            break

    lines.append("")
    if monotonic and len(grade_avgs) >= 3:
        lines.append("  VERDICT: D2 scores are monotonically decreasing across grades")
        lines.append("           (higher grade => higher D2). This is CONSISTENT.")
    elif len(grade_avgs) >= 3:
        lines.append("  VERDICT: D2 scores are NOT monotonically decreasing across grades.")
        lines.append("           There may be inversions. Review per-grade averages above.")
        # Show inversions
        for i in range(1, len(grade_avgs)):
            if grade_avgs[i][1] > grade_avgs[i - 1][1]:
                lines.append(
                    f"           Inversion: Grade {grade_avgs[i][0]} "
                    f"(avg D2={grade_avgs[i][1]:.1f}) > "
                    f"Grade {grade_avgs[i - 1][0]} "
                    f"(avg D2={grade_avgs[i - 1][1]:.1f})"
                )
    else:
        lines.append("  VERDICT: Insufficient grade buckets for monotonicity check.")

    lines.append("")
    return lines


def section_simulated_ahs(scans: list[dict]) -> list[str]:
    """Section 4: Simulated AHS with D2 at live 70% weight.

    Since D2 is ALREADY at 70% weight in the production composite, this
    section computes a pure D1+D2 composite WITHOUT the CDP modifier to
    show the raw effect of D2 on grading, then compares to current grades.
    """
    # For agents with both D1 and D2, simulate: composite = 0.30*D1 + 0.70*D2
    # (no CDP modifier, no temporal smoothing -- raw composite)
    migrations: dict[str, int] = {"upgrade": 0, "downgrade": 0, "unchanged": 0}
    grade_changes: list[tuple[str, str, str, int, int, int]] = []  # addr, old_grade, new_grade, ahs, d1, d2

    grade_migration_matrix: dict[str, dict[str, int]] = {}
    for g, _, _ in GRADE_BOUNDARIES:
        grade_migration_matrix[g] = {gg: 0 for gg, _, _ in GRADE_BOUNDARIES}

    eligible = 0
    for s in scans:
        d1 = s["d1_score"]
        d2 = s["d2_score"]
        current_grade = s["grade"]
        if d1 is None or d2 is None or current_grade is None:
            continue
        eligible += 1

        d3 = s["d3_score"]
        if d3 is not None:
            sim_score = round(D1_WEIGHT_3D * d1 + D2_WEIGHT_3D * d2 + D3_WEIGHT_3D * d3)
        else:
            sim_score = round(D1_WEIGHT_2D * d1 + D2_WEIGHT_2D * d2)

        sim_score = max(0, min(100, sim_score))
        sim_grade = grade_from_score(sim_score)

        if current_grade in grade_migration_matrix and sim_grade in grade_migration_matrix[current_grade]:
            grade_migration_matrix[current_grade][sim_grade] += 1

        grade_order = [g for g, _, _ in GRADE_BOUNDARIES]
        old_idx = grade_order.index(current_grade) if current_grade in grade_order else -1
        new_idx = grade_order.index(sim_grade) if sim_grade in grade_order else -1

        if old_idx < 0 or new_idx < 0:
            continue

        if new_idx < old_idx:
            migrations["upgrade"] += 1
            grade_changes.append((s["address"], current_grade, sim_grade, s["ahs_score"], d1, d2))
        elif new_idx > old_idx:
            migrations["downgrade"] += 1
            grade_changes.append((s["address"], current_grade, sim_grade, s["ahs_score"], d1, d2))
        else:
            migrations["unchanged"] += 1

    lines = [
        "=" * 72,
        "4. SIMULATED AHS -- D2 AT LIVE WEIGHT (raw composite, no CDP modifier)",
        "=" * 72,
        "",
        f"  Eligible agents (have D1 + D2):  {eligible}",
        "",
        f"  Grade unchanged:   {migrations['unchanged']:>5}  ({pct(migrations['unchanged'], eligible)})",
        f"  Grade upgraded:    {migrations['upgrade']:>5}  ({pct(migrations['upgrade'], eligible)})",
        f"  Grade downgraded:  {migrations['downgrade']:>5}  ({pct(migrations['downgrade'], eligible)})",
        "",
    ]

    # Migration matrix
    grades = [g for g, _, _ in GRADE_BOUNDARIES]
    lines.append("  Grade Migration Matrix (rows = current, cols = simulated):")
    lines.append("")
    header = "          " + "".join(f"{g:>6}" for g in grades)
    lines.append(header)
    lines.append("  " + "-" * (8 + 6 * len(grades)))
    for g in grades:
        row_data = grade_migration_matrix.get(g, {})
        row_vals = "".join(f"{row_data.get(gg, 0):>6}" for gg in grades)
        lines.append(f"  {g:>6} |{row_vals}")

    lines.append("")

    # Top movers (sample)
    if grade_changes:
        grade_changes.sort(key=lambda x: abs(
            [g for g, _, _ in GRADE_BOUNDARIES].index(x[1]) -
            [g for g, _, _ in GRADE_BOUNDARIES].index(x[2])
        ), reverse=True)
        lines.append("  Largest grade changes (top 15):")
        lines.append(f"  {'Address':<44} {'Current':>8} {'Simulated':>10} {'AHS':>5} {'D1':>5} {'D2':>5}")
        lines.append(f"  {'-' * 44} {'-' * 8} {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5}")
        for addr, old_g, new_g, ahs, d1, d2 in grade_changes[:15]:
            arrow = "^" if [g for g, _, _ in GRADE_BOUNDARIES].index(new_g) < [g for g, _, _ in GRADE_BOUNDARIES].index(old_g) else "v"
            lines.append(
                f"  {addr:<44} {old_g:>8} {arrow} {new_g:<8} {ahs:>5} {d1:>5} {d2:>5}"
            )
        lines.append("")

    return lines


def section_outliers(scans: list[dict]) -> list[str]:
    """Section 5: Agents where D2 diverges >20 points from AHS."""
    outliers = []
    for s in scans:
        d2 = s["d2_score"]
        ahs = s["ahs_score"]
        if d2 is None or ahs is None:
            continue
        delta = d2 - ahs
        if abs(delta) > DIVERGENCE_THRESHOLD:
            outliers.append((s["address"], ahs, d2, delta, s["d1_score"], s["grade"]))

    outliers.sort(key=lambda x: abs(x[3]), reverse=True)

    total_with_d2 = sum(1 for s in scans if s["d2_score"] is not None)

    lines = [
        "=" * 72,
        f"5. OUTLIERS -- D2 DIVERGES >{DIVERGENCE_THRESHOLD} POINTS FROM CURRENT AHS",
        "=" * 72,
        "",
        f"  Total outliers:  {len(outliers)} / {total_with_d2}  ({pct(len(outliers), total_with_d2)})",
        "",
    ]

    # Breakdown: D2 >> AHS vs D2 << AHS
    d2_above = [o for o in outliers if o[3] > 0]
    d2_below = [o for o in outliers if o[3] < 0]
    lines.append(f"  D2 > AHS by >{DIVERGENCE_THRESHOLD}pts (D2 is more generous):  {len(d2_above)}")
    lines.append(f"  D2 < AHS by >{DIVERGENCE_THRESHOLD}pts (D2 is harsher):       {len(d2_below)}")
    lines.append("")

    if outliers:
        lines.append(f"  {'Address':<44} {'AHS':>5} {'D2':>5} {'Delta':>7} {'D1':>5} {'Grade':>6}")
        lines.append(f"  {'-' * 44} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 5} {'-' * 6}")
        for addr, ahs, d2, delta, d1, grade in outliers[:30]:
            sign = "+" if delta > 0 else ""
            lines.append(
                f"  {addr:<44} {ahs:>5} {d2:>5} {sign}{delta:>6} {d1 or 0:>5} {grade or '?':>6}"
            )
        if len(outliers) > 30:
            lines.append(f"  ... and {len(outliers) - 30} more outliers")
        lines.append("")

    return lines


def section_session_continuity(scans: list[dict]) -> list[str]:
    """Section 6: Session continuity shadow signal analysis."""
    session_scores: list[int] = []
    abrupt_counts: list[int] = []
    budget_exhaust_counts: list[int] = []
    total_sessions_list: list[int] = []

    for s in scans:
        raw = s.get("shadow_signals_json")
        if not raw or raw == "null":
            continue
        try:
            ss = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        sc = ss.get("session_continuity_score")
        if sc is not None:
            session_scores.append(sc)
            abrupt_counts.append(ss.get("abrupt_sessions", 0))
            budget_exhaust_counts.append(ss.get("budget_exhaustion_count", 0))
            total_sessions_list.append(ss.get("total_sessions", 0))

    total = len(session_scores)

    lines = [
        "=" * 72,
        "6. SESSION CONTINUITY SHADOW SIGNAL (D2 sub-signal, not in composite)",
        "=" * 72,
        "",
        f"  Agents with session_continuity_score:  {total}",
    ]

    if total == 0:
        lines.append("  No session continuity data found.")
        lines.append("")
        return lines

    avg = statistics.mean(session_scores)
    med = statistics.median(session_scores)
    std = statistics.stdev(session_scores) if total >= 2 else 0.0

    lines.extend([
        f"  Mean score:        {avg:.1f}",
        f"  Median score:      {med:.1f}",
        f"  Std Dev:           {std:.1f}",
        "",
        f"  Avg abrupt sessions:       {statistics.mean(abrupt_counts):.1f}",
        f"  Avg budget exhaustions:     {statistics.mean(budget_exhaust_counts):.1f}",
        f"  Avg total sessions:         {statistics.mean(total_sessions_list):.1f}",
        "",
    ])

    # Bucket session scores
    lines.append(f"  {'Bucket':<10} {'Count':>6} {'Pct':>8}   Distribution")
    lines.append(f"  {'-' * 10} {'-' * 6} {'-' * 8}   {'-' * 40}")
    for label, lo, hi in D2_BUCKETS:
        count = sum(1 for sc in session_scores if lo <= sc <= hi)
        lines.append(
            f"  {label:<10} {count:>6} {pct(count, total):>8}   {bar(count, total)}"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="D2 Shadow Score Review Analysis")
    parser.add_argument("--db", default=DB_PATH, help="Path to AHM SQLite database")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        print("Set DB_PATH env var or use --db flag.")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_slug = now.strftime("%Y%m%d")

    # Fetch data
    scans = fetch_latest_scans(db_path)
    if not scans:
        print("ERROR: No AHS scans found in database.")
        sys.exit(1)

    # Build report
    header = [
        "=" * 72,
        "D2 SHADOW SCORE REVIEW -- GATE ANALYSIS FOR LIVE PROMOTION",
        "=" * 72,
        "",
        f"  Generated:  {timestamp}",
        f"  Database:   {os.path.abspath(db_path)}",
        f"  Agents:     {len(scans)}",
        "",
        "  Current production weights:",
        f"    2D mode:  D1={D1_WEIGHT_2D:.0%}  D2={D2_WEIGHT_2D:.0%}",
        f"    3D mode:  D1={D1_WEIGHT_3D:.0%}  D2={D2_WEIGHT_3D:.0%}  D3={D3_WEIGHT_3D:.0%}",
        "",
        "  Note: D2 is already live at these weights in the composite score.",
        "  The 'session_continuity' sub-signal within D2 is in SHADOW MODE",
        "  (computed and stored but NOT included in D2's composite).",
        "",
    ]

    sections = (
        header
        + section_coverage(scans)
        + section_distribution(scans)
        + section_grade_correlation(scans)
        + section_simulated_ahs(scans)
        + section_outliers(scans)
        + section_session_continuity(scans)
    )

    # Footer
    sections.extend([
        "=" * 72,
        "END OF REPORT",
        "=" * 72,
    ])

    report = "\n".join(sections)

    # Output to stdout
    print(report)

    # Save to docs/
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    output_path = docs_dir / f"d2-shadow-review-{date_slug}.txt"
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
