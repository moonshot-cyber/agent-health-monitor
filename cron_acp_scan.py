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
        agents, api_stats = discover_agents(max_agents=args.max_agents, sort=args.sort)
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

        # Phase 3: AHS Scanning (skips already-scanned wallets by default)
        scan_results = scan_wallets(agents, max_scans=args.max_scans, force_rescan=False)

        # Phase 4: Report
        generate_report(agents, api_stats, dedup_stats, scan_results)

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
