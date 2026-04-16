#!/usr/bin/env python3
"""Cron wrapper for ACP proactive scanning.

Designed to run as a Railway cron service or local scheduled task.
Handles logging, clean exit codes, and runtime guards.

Railway cron schedule: 0 2 * * * (daily at 02:00 UTC)

Usage:
    python cron_acp_scan.py                    # Default: 500 agents, 100 scans
    python cron_acp_scan.py --max-agents 1000  # Override agent limit
    python cron_acp_scan.py --max-scans 200    # Override scan limit
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable via environment variables (Railway service variables)
# ---------------------------------------------------------------------------
MAX_AGENTS = int(os.getenv("ACP_MAX_AGENTS", "500"))
MAX_SCANS = int(os.getenv("ACP_MAX_SCANS", "100"))
SORT_ORDER = os.getenv("ACP_SORT", "successfulJobCount:desc")
MAX_RUNTIME_SECONDS = int(os.getenv("ACP_MAX_RUNTIME", "3600"))  # 1 hour guard


def log(msg: str):
    """Print with UTC timestamp for Railway log ingestion."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


AHM_VP_PATH = Path(__file__).parent / "AHM_VALUE_PROPOSITIONS.md"


def _update_value_propositions():
    """Rewrite AHM_VALUE_PROPOSITIONS.md with latest stats from the DB."""
    try:
        import db
        stats = db.get_ecosystem_dashboard_stats()
        if not stats or not stats.get("total_scanned"):
            log("Skipping AHM_VALUE_PROPOSITIONS.md update — no stats available")
            return

        total = stats["total_scanned"]
        avg_ahs = stats.get("avg_ahs", 0)
        avg_d1 = stats.get("avg_d1", 0)
        avg_d2 = stats.get("avg_d2", 0)

        grades = stats.get("grade_distribution", {})
        total_graded = sum(grades.values()) or 1
        grade_a_pct = round(grades.get("A", 0) / total_graded * 100, 1)

        patterns = stats.get("pattern_distribution", {})
        zombie_count = patterns.get("Zombie Agent", 0)
        zombie_pct = round(zombie_count / total_graded * 100)

        sources = stats.get("data_sources", {})
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        grade_lines = "\n".join(
            f"| {g} | {c} |" for g, c in sorted(grades.items())
        )

        source_lines = "\n".join(
            f"| {s} | {c:,} |" for s, c in sorted(sources.items(), key=lambda x: -x[1])
        )

        content = f"""# AHM Value Propositions — Live Ecosystem Stats

> **Auto-updated by the nightly scan pipeline.** Do not edit manually — values are
> overwritten after each `cron_acp_scan.py` run via `db.get_ecosystem_dashboard_stats()`.

## Ecosystem Health (Source of Truth)

| Metric | Value |
|--------|-------|
| Total agents scored | {total:,} |
| Ecosystem avg AHS | {avg_ahs} |
| D1 avg (Wallet Hygiene) | {avg_d1} |
| D2 avg (Behavioural Patterns) | {avg_d2} |
| Zombie Agent rate | {zombie_pct}% |
| Grade A rate | {grade_a_pct}% |

## Registry Breakdown

| Source | Agents |
|--------|--------|
{source_lines}

## Grade Distribution

| Grade | Count |
|-------|-------|
{grade_lines}

---

*Last updated: {ts} by nightly scan pipeline.*
"""
        AHM_VP_PATH.write_text(content, encoding="utf-8")
        log(f"AHM_VALUE_PROPOSITIONS.md updated ({total:,} agents, avg AHS {avg_ahs})")
    except Exception as e:
        log(f"Warning: failed to update AHM_VALUE_PROPOSITIONS.md — {e}")


def main():
    parser = argparse.ArgumentParser(description="Cron wrapper for ACP scan")
    parser.add_argument("--max-agents", type=int, default=MAX_AGENTS)
    parser.add_argument("--max-scans", type=int, default=MAX_SCANS)
    parser.add_argument("--sort", type=str, default=SORT_ORDER)
    args = parser.parse_args()

    log("=" * 50)
    log("ACP NIGHTLY SCAN — START")
    log(f"  max-agents: {args.max_agents}")
    log(f"  max-scans:  {args.max_scans}")
    log(f"  sort:       {args.sort}")
    log(f"  max-runtime: {MAX_RUNTIME_SECONDS}s")
    log("=" * 50)

    start = time.time()

    try:
        from acp_proactive_scan import discover_agents, deduplicate_wallets, scan_wallets, generate_report

        # Phase 1: Discovery
        agents, api_stats = discover_agents(
            max_agents=args.max_agents, sort=args.sort,
        )
        if not agents:
            log("No agents discovered. Exiting cleanly.")
            sys.exit(0)

        # Phase 2: Deduplication
        dedup_stats = deduplicate_wallets(agents)

        # Runtime guard — abort before scanning if discovery took too long
        elapsed = time.time() - start
        if elapsed > MAX_RUNTIME_SECONDS * 0.8:
            log(f"Runtime guard: discovery took {elapsed:.0f}s, skipping scan phase")
            generate_report(agents, api_stats, dedup_stats, {})
            sys.exit(0)

        # Phase 3: AHS Scanning (stale-first rotation)
        scan_results = scan_wallets(agents, max_scans=args.max_scans)

        # Phase 4: Report
        generate_report(agents, api_stats, dedup_stats, scan_results)

        # Phase 5: Update AHM_VALUE_PROPOSITIONS.md with latest DB stats
        _update_value_propositions()

        elapsed = time.time() - start
        log(f"ACP NIGHTLY SCAN — COMPLETE ({elapsed:.0f}s)")
        sys.exit(0)

    except KeyboardInterrupt:
        log("Interrupted by signal. Exiting.")
        sys.exit(0)

    except Exception as e:
        elapsed = time.time() - start
        log(f"ACP NIGHTLY SCAN — FAILED after {elapsed:.0f}s")
        log(f"Error: {e}")
        # Exit 1 so Railway marks the run as failed
        sys.exit(1)


if __name__ == "__main__":
    main()
