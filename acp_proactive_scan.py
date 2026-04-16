#!/usr/bin/env python3
"""ACP (Agent Commerce Protocol) proactive ecosystem scanner.

Discovers agents registered on the Virtuals ACP platform (agdp.io),
extracts wallet addresses, runs lightweight D1+D2 AHS scans (free APIs only),
and stores results in the scan history database.

Data source: https://acpx.virtuals.io/api/agents (public, no auth required)

Usage:
    python acp_proactive_scan.py                          # Default: 500 agents, scan all
    python acp_proactive_scan.py --max-agents 1000 --max-scans 200
    python acp_proactive_scan.py --skip-scan              # Discovery + dedup only
    python acp_proactive_scan.py --sort revenue:desc      # Sort by revenue
    python acp_proactive_scan.py --force-rescan           # Ignore already-scanned wallets
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACP_API_BASE = "https://acpx.virtuals.io/api/agents"
PAGE_SIZE = 100          # Max per Strapi page
BLOCKSCOUT_DELAY = 2.0   # Rate limit: ~50 req/min without API key
API_TIMEOUT = 15          # HTTP timeout for ACP API
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N scans


def _safe_str(s: str) -> str:
    """Strip non-ASCII characters for safe console output on Windows."""
    return s.encode("ascii", "replace").decode("ascii")

# Output
CSV_PATH = "acp_scan_results.csv"
MD_PATH = "acp_batch_scan_results.md"
CHECKPOINT_PATH = "acp_scan_checkpoint.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ACPAgent:
    acp_id: int
    name: str = ""
    wallet_address: str = ""
    owner_address: str = ""
    success_rate: float = 0.0
    successful_job_count: int = 0
    revenue: float = 0.0
    last_active_at: str = ""
    cluster: str = ""
    role: str = ""
    is_high_risk: bool = False
    # AHS results
    ahs_score: int | None = None
    grade: str = ""
    grade_label: str = ""
    confidence: str = ""
    d1_score: int | None = None
    d2_score: int | None = None
    patterns: str = ""
    tx_count: int = 0
    history_days: int = 0
    scan_error: str = ""


# ---------------------------------------------------------------------------
# Phase 1: ACP Agent Discovery
# ---------------------------------------------------------------------------

def discover_agents(max_agents: int = 50, sort: str = "successfulJobCount:desc") -> tuple[list[ACPAgent], dict]:
    """Fetch agents from the ACP API.

    Returns (agents_list, api_stats).
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: ACP Agent Discovery")
    print("=" * 60)
    print(f"[*] API: {ACP_API_BASE}")
    print(f"[*] Sort: {sort} | Max agents: {max_agents}")

    agents = []
    page = 1
    total_available = None
    # Hard ceiling: don't paginate forever (max 5000 agents or end of registry)
    hard_max = max(max_agents, 5000)

    while len(agents) < hard_max:
        if len(agents) >= max_agents:
            break

        url = (
            f"{ACP_API_BASE}"
            f"?pagination[page]={page}"
            f"&pagination[pageSize]={PAGE_SIZE}"
            f"&sort={sort}"
        )

        try:
            resp = requests.get(url, timeout=API_TIMEOUT, headers={
                "Accept": "application/json",
                "User-Agent": "AHM-ACP-Scanner/1.0",
            })
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [!] API error on page {page}: {e}")
            break
        except json.JSONDecodeError:
            print(f"  [!] Invalid JSON on page {page}")
            break

        # Parse pagination metadata (Strapi: meta.pagination)
        pagination = data.get("meta", {}).get("pagination", {})
        if total_available is None:
            total_available = pagination.get("total", 0)
            print(f"[+] Total agents in ACP registry: {total_available:,}")

        items = data.get("data", [])
        if not items:
            break

        for item in items:
            if len(agents) >= max_agents:
                break

            wallet = (item.get("walletAddress") or "").strip()
            if not wallet or not wallet.startswith("0x") or len(wallet) != 42:
                continue

            addr = wallet.lower()
            agent = ACPAgent(
                acp_id=item.get("id", 0),
                name=(item.get("name") or "")[:80],
                wallet_address=addr,
                owner_address=(item.get("ownerAddress") or "").lower(),
                success_rate=float(item.get("successRate") or 0),
                successful_job_count=int(item.get("successfulJobCount") or 0),
                revenue=float(item.get("revenue") or 0),
                last_active_at=item.get("lastActiveAt") or "",
                cluster=item.get("cluster") or "",
                role=item.get("role") or "",
                is_high_risk=bool(item.get("isHighRisk")),
            )
            agents.append(agent)

        print(f"  [page {page}] fetched {len(items)} agents ({len(agents)} total)")
        page += 1

        # Stop if we've exhausted the registry
        total_pages = pagination.get("pageCount", 999)
        if page > total_pages:
            print(f"[+] Reached end of ACP registry ({total_pages} pages)")
            break

        # Don't hammer the API
        time.sleep(0.5)

    api_stats = {
        "total_available": total_available or 0,
        "fetched": len(agents),
        "pages_read": page - 1,
        "sort": sort,
    }

    print(f"[+] Discovered {len(agents)} agents")
    return agents, api_stats


# ---------------------------------------------------------------------------
# Phase 2: Wallet Deduplication
# ---------------------------------------------------------------------------

def deduplicate_wallets(agents: list[ACPAgent]) -> dict:
    """Analyse wallet address sharing and return dedup stats.

    Returns dict with sharing analysis.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Wallet Deduplication")
    print("=" * 60)

    wallet_to_agents: dict[str, list[ACPAgent]] = {}
    for agent in agents:
        wallet_to_agents.setdefault(agent.wallet_address, []).append(agent)

    unique_wallets = len(wallet_to_agents)
    shared_wallets = sum(1 for v in wallet_to_agents.values() if len(v) > 1)
    max_sharing = max(len(v) for v in wallet_to_agents.values()) if wallet_to_agents else 0

    print(f"[+] Total agents: {len(agents)}")
    print(f"[+] Unique wallets: {unique_wallets}")
    print(f"[+] Shared wallets (>1 agent): {shared_wallets}")
    print(f"[+] Max agents per wallet: {max_sharing}")

    # Show top shared wallets
    if shared_wallets > 0:
        sorted_shared = sorted(wallet_to_agents.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\n  Top shared wallets:")
        for wallet, ags in sorted_shared[:5]:
            if len(ags) > 1:
                names = ", ".join(_safe_str(a.name[:20]) for a in ags[:3])
                print(f"    {wallet[:10]}... -- {len(ags)} agents ({names})")

    # Check overlap with owner addresses
    owner_wallets = {a.owner_address for a in agents if a.owner_address}
    agent_wallets = {a.wallet_address for a in agents}
    overlap = owner_wallets & agent_wallets
    print(f"[+] Owner addresses that are also agent wallets: {len(overlap)}")

    return {
        "unique_wallets": unique_wallets,
        "shared_wallets": shared_wallets,
        "max_sharing": max_sharing,
        "owner_wallet_overlap": len(overlap),
    }


# ---------------------------------------------------------------------------
# Phase 3: AHS Scanning
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Load checkpoint from previous interrupted run."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"scanned_addresses": [], "results": {}, "scan_count": 0}


def save_checkpoint(scanned: list[str], results: dict, scan_count: int):
    """Persist checkpoint so restarts don't rescan."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "scanned_addresses": scanned,
            "results": results,
            "scan_count": scan_count,
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f)


def _ingest_discovered_wallets_to_db(agents: list[ACPAgent]) -> int:
    """Persist newly-discovered ACP wallets to known_wallets.

    Each discovered wallet is INSERT OR IGNORE'd so existing rows (with
    their scan history and last_scanned_at) are left untouched. New rows
    are inserted with last_scanned_at=NULL and scan_count=0, which makes
    them bubble to the top of the stale-first candidate query below.

    Returns the number of rows actually inserted (excludes addresses
    that were already in the table).
    """
    if not agents:
        return 0

    import db

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Deduplicate by wallet address — multiple agents may share a wallet
    seen: set[str] = set()
    unique: list[ACPAgent] = []
    for agent in agents:
        if agent.wallet_address not in seen:
            seen.add(agent.wallet_address)
            unique.append(agent)

    conn = db.get_connection()
    inserted = 0
    try:
        for agent in unique:
            label = f"ACP #{agent.acp_id}"
            if agent.name:
                label += f" \u2014 {agent.name}"
            cur = conn.execute(
                """INSERT OR IGNORE INTO known_wallets
                       (address, label, source, first_seen_at, scan_count, registries)
                   VALUES (?, ?, ?, ?, 0, ?)""",
                (agent.wallet_address, label, "acp_proactive_scan",
                 now_iso, "acp_proactive_scan"),
            )
            if cur.rowcount:
                inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_acp_scan_candidates(limit: int) -> list[dict]:
    """Return ACP wallets to AHS-score this run, ordered stale-first.

    Uses ``ORDER BY last_scanned_at ASC NULLS FIRST LIMIT ?`` so:
      1. Never-scanned wallets (NULL last_scanned_at) are prioritised
         — discovered-but-not-yet-scored agents are picked up first.
      2. Among scanned wallets, the oldest scan comes next — coverage
         rotates through the full registry instead of re-scoring the
         same batch every night.
    """
    import db
    conn = db.get_connection()
    try:
        rows = conn.execute(
            """SELECT address, label, source, last_scanned_at
               FROM known_wallets
               WHERE source = 'acp_proactive_scan'
               ORDER BY last_scanned_at ASC NULLS FIRST
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def scan_wallets(agents: list[ACPAgent], max_scans: int = 500) -> dict:
    """Run AHS 2D scans on ACP wallets, stale-first rotation.

    Coverage strategy (same pattern as celo_scan.scan_celo_agents):
      1. Persist every discovered wallet to known_wallets via INSERT OR
         IGNORE so agents aren't dropped when new discoveries arrive.
      2. Pull the ``max_scans`` stalest wallets from known_wallets
         (NULL-scanned first, then oldest-scanned) and AHS-score those.

    This replaces the older "scan new discoveries, skip everything
    already in the DB" flow, which permanently excluded every scored
    wallet and never rescanned as on-chain activity changed.

    Returns dict mapping address -> scan result.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: AHS Scanning (D1+D2, free APIs only)")
    print("=" * 60)

    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price
    import db

    db.init_db()

    # Phase 3a: Persist discoveries so they survive across runs.
    inserted = _ingest_discovered_wallets_to_db(agents)

    # Build address -> agent(s) lookup for metadata / report cross-reference.
    wallet_to_agents: dict[str, list[ACPAgent]] = {}
    for agent in agents:
        wallet_to_agents.setdefault(agent.wallet_address, []).append(agent)

    total_known = 0
    try:
        conn = db.get_connection()
        total_known = conn.execute(
            "SELECT COUNT(*) FROM known_wallets WHERE source = 'acp_proactive_scan'"
        ).fetchone()[0]
        conn.close()
    except Exception:
        pass

    print(f"[+] Ingested {inserted} new wallets into known_wallets "
          f"({total_known} total ACP wallets tracked)")

    # Phase 3b: Select stalest candidates.
    candidates = get_acp_scan_candidates(limit=max_scans)
    if not candidates:
        print("[+] No ACP candidates available")
        return {}

    never_scanned = sum(1 for c in candidates if c.get("last_scanned_at") is None)
    print(f"[+] Selected {len(candidates)} candidates for scoring "
          f"({never_scanned} never-scanned, {len(candidates) - never_scanned} rescans)")

    # Resume from checkpoint (addresses scored in THIS run)
    checkpoint = load_checkpoint()
    checkpoint_addrs = set(checkpoint.get("scanned_addresses", []))
    if checkpoint_addrs:
        print(f"[+] Resuming from checkpoint: {len(checkpoint_addrs)} addresses to skip")

    results: dict = checkpoint.get("results", {})
    scanned_this_run: list[str] = checkpoint.get("scanned_addresses", [])

    # Fetch ETH price once
    eth_price = get_eth_price()
    print(f"[+] ETH price: ${eth_price:,.2f}")

    scan_count = 0
    errors = 0
    start_time = time.time()

    for wallet in candidates:
        if scan_count >= max_scans:
            print(f"  [stop] Reached max_scans={max_scans}")
            break

        address = wallet["address"]

        # Skip if already scored in this run (checkpoint resume)
        if address in checkpoint_addrs:
            continue

        scan_count += 1
        source_agents = wallet_to_agents.get(address, [])
        agent_name = source_agents[0].name if source_agents else ""
        acp_id = source_agents[0].acp_id if source_agents else 0
        display_name = _safe_str(agent_name[:30]) if agent_name else (wallet.get("label") or "?")
        rescan_tag = "  (rescan)" if wallet.get("last_scanned_at") else ""

        elapsed = time.time() - start_time
        rate = scan_count / elapsed * 3600 if elapsed > 0 else 0
        print(f"  [{scan_count}/{max_scans}] {address[:12]}... ({display_name})  [{rate:.0f}/hr]{rescan_tag}")

        try:
            txs = fetch_transactions(address)
            time.sleep(BLOCKSCOUT_DELAY)
            tokens = fetch_tokens_v2(address, max_pages=3)
            time.sleep(BLOCKSCOUT_DELAY)

            ahs = calculate_ahs(
                address=address,
                tokens=tokens,
                transactions=txs,
                eth_price=eth_price,
            )

            # Build patterns list for db.log_scan
            patterns_list = None
            if ahs.patterns_detected:
                patterns_list = [
                    {
                        "name": p.get("name", "") if isinstance(p, dict) else str(p),
                        "severity": p.get("severity", "") if isinstance(p, dict) else "",
                        "description": p.get("description", "") if isinstance(p, dict) else "",
                        "modifier": p.get("modifier") if isinstance(p, dict) else None,
                    }
                    for p in ahs.patterns_detected
                ]

            # Prefer fresh metadata from this run's discovery; fall back
            # to the label already stored in known_wallets.
            if source_agents:
                label = f"ACP #{acp_id}"
                if agent_name:
                    label += f" \u2014 {agent_name}"
            else:
                label = wallet.get("label") or "ACP Agent"

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.log_scan(
                address=address,
                endpoint="ahs",
                scan_timestamp=now_iso,
                source="acp_proactive_scan",
                label=label,
                ahs_score=ahs.agent_health_score,
                grade=ahs.grade,
                grade_label=ahs.grade_label,
                confidence=ahs.confidence,
                mode="2D",
                d1_score=ahs.d1_score,
                d2_score=ahs.d2_score,
                patterns=patterns_list,
                tx_count=ahs.tx_count,
                history_days=ahs.history_days,
                response_data={
                    "ahs": ahs.agent_health_score,
                    "grade": ahs.grade,
                    "d1": ahs.d1_score,
                    "d2": ahs.d2_score,
                    "d2_data_source": ahs.d2_data_source,
                },
            )

            result = {
                "ahs_score": ahs.agent_health_score,
                "grade": ahs.grade,
                "grade_label": ahs.grade_label,
                "confidence": ahs.confidence,
                "d1_score": ahs.d1_score,
                "d2_score": ahs.d2_score,
                "d2_data_source": ahs.d2_data_source,
                "tx_count": ahs.tx_count,
                "history_days": ahs.history_days,
                "patterns": "; ".join(
                    p.get("name", str(p)) if isinstance(p, dict) else str(p)
                    for p in ahs.patterns_detected
                ) if ahs.patterns_detected else "none",
            }
            results[address] = result

            # Map back to agent records
            for agent in source_agents:
                if agent.ahs_score is None:
                    agent.ahs_score = result["ahs_score"]
                    agent.grade = result["grade"]
                    agent.grade_label = result["grade_label"]
                    agent.confidence = result["confidence"]
                    agent.d1_score = result["d1_score"]
                    agent.d2_score = result["d2_score"]
                    agent.patterns = result["patterns"]
                    agent.tx_count = result["tx_count"]
                    agent.history_days = result["history_days"]

            print(f"           AHS {ahs.agent_health_score}/{ahs.grade} | D1={ahs.d1_score} D2={ahs.d2_score} | {ahs.tx_count} txs, {ahs.history_days}d | src={ahs.d2_data_source}")

        except Exception as e:
            err_msg = str(e)[:80]
            print(f"           [!] Error: {err_msg}")
            errors += 1
            for agent in source_agents:
                if not agent.scan_error:
                    agent.scan_error = err_msg

        # Track and checkpoint
        scanned_this_run.append(address)
        if scan_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(scanned_this_run, results, scan_count)
            print(f"  [checkpoint] Saved at {scan_count} scans")

    # Final checkpoint
    save_checkpoint(scanned_this_run, results, scan_count)

    elapsed_total = time.time() - start_time
    print(f"\n[+] Scanned {scan_count} wallets in {elapsed_total/60:.1f} min ({errors} errors)")
    if scan_count > 0:
        print(f"[+] Average: {elapsed_total/scan_count:.1f}s per wallet")

    # Log batch quality stats
    try:
        scored = [r for r in results.values() if r.get("ahs_score") is not None]
        if scored:
            scores = [r["ahs_score"] for r in scored]
            grades = {}
            for r in scored:
                g = r.get("grade", "?")
                grades[g] = grades.get(g, 0) + 1
            d1_scores = [r["d1_score"] for r in scored if r.get("d1_score") is not None]
            d2_scores = [r["d2_score"] for r in scored if r.get("d2_score") is not None]

            db.log_batch_quality(
                source="acp",
                wallets_scanned=len(scored),
                average_ahs=round(sum(scores) / len(scores), 1),
                min_ahs=min(scores),
                max_ahs=max(scores),
                grade_distribution=grades,
                avg_d1=round(sum(d1_scores) / len(d1_scores), 1) if d1_scores else None,
                avg_d2=round(sum(d2_scores) / len(d2_scores), 1) if d2_scores else None,
            )
            print(f"[+] Batch quality logged: {len(scored)} wallets, avg AHS {sum(scores)/len(scores):.1f}")
    except Exception as e:
        print(f"[!] Batch quality logging failed (non-fatal): {e}")

    return results


# ---------------------------------------------------------------------------
# Phase 4: Report
# ---------------------------------------------------------------------------

def generate_report(
    agents: list[ACPAgent],
    api_stats: dict,
    dedup_stats: dict,
    scan_results: dict,
) -> None:
    """Generate console summary, CSV, and Markdown report."""
    print("\n" + "=" * 60)
    print("  PHASE 4: Report")
    print("=" * 60)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    scored = [r for r in scan_results.values() if r.get("ahs_score") is not None]
    grade_dist = {}
    for r in scored:
        g = r["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    avg_ahs = sum(r["ahs_score"] for r in scored) / len(scored) if scored else 0
    scores = [r["ahs_score"] for r in scored]
    min_ahs = min(scores) if scores else 0
    max_ahs = max(scores) if scores else 0

    # Console summary
    print(f"\n  ACP agents in registry:    {api_stats['total_available']:,}")
    print(f"  Agents fetched:            {api_stats['fetched']}")
    print(f"  Unique wallets:            {dedup_stats['unique_wallets']}")
    print(f"  Shared wallets:            {dedup_stats['shared_wallets']}")
    print(f"  Wallets scanned (AHS):     {len(scan_results)}")
    if scored:
        print(f"  Average AHS:               {avg_ahs:.1f}")
        print(f"  AHS range:                 {min_ahs}-{max_ahs}")
        print(f"  Grade distribution:        {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}")

    # -- CSV --
    fieldnames = [
        "acp_id", "name", "wallet_address", "owner_address",
        "successful_job_count", "revenue", "success_rate",
        "cluster", "role", "last_active_at",
        "ahs", "grade", "grade_label", "d1", "d2",
        "confidence", "patterns", "tx_count", "history_days", "scan_error",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in agents:
            writer.writerow({
                "acp_id": a.acp_id,
                "name": a.name,
                "wallet_address": a.wallet_address,
                "owner_address": a.owner_address,
                "successful_job_count": a.successful_job_count,
                "revenue": f"{a.revenue:.2f}",
                "success_rate": a.success_rate,
                "cluster": a.cluster,
                "role": a.role,
                "last_active_at": a.last_active_at,
                "ahs": a.ahs_score if a.ahs_score is not None else "",
                "grade": a.grade,
                "grade_label": a.grade_label,
                "d1": a.d1_score if a.d1_score is not None else "",
                "d2": a.d2_score if a.d2_score is not None else "",
                "confidence": a.confidence,
                "patterns": a.patterns,
                "tx_count": a.tx_count,
                "history_days": a.history_days,
                "scan_error": a.scan_error,
            })
    print(f"\n[+] CSV written: {CSV_PATH}")

    # -- Markdown --
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Proactive Ecosystem Scan Spike — ACP (agdp.io)\n\n")
        f.write(f"> Scanned: {now_str}\n")
        f.write(f"> Source: ACP API (`acpx.virtuals.io/api/agents`)\n")
        f.write(f"> Scan mode: 2D (D1+D2 only, free APIs, no Nansen)\n\n")

        f.write("## Spike Objective\n\n")
        f.write("Prove the end-to-end proactive scanning flow: discover agent wallets from an ")
        f.write("external registry, run AHS scans, store results in the database. This is the ")
        f.write("first step toward Priority 4 (Proactive Ecosystem Scanning) in the AHM backlog.\n\n")

        f.write("## Source Assessment\n\n")
        f.write("| Source | Status | Notes |\n")
        f.write("|--------|--------|-------|\n")
        f.write("| Virtuals ACP (agdp.io) | **Selected** | Free API, no auth, 40K+ agents with `walletAddress` field |\n")
        f.write("| x402scan | Blocked | All endpoints paywalled ($0.01/call via x402) |\n")
        f.write("| 402index.io | Viable (backup) | Free API but no payTo addresses — requires 2-step probe |\n")
        f.write("| Virtuals Protocol API | Blocked | Agents share TBA shards — per-agent AHS not meaningful |\n\n")

        f.write("## Discovery Summary\n\n")
        f.write(f"- **Total agents in ACP registry:** {api_stats['total_available']:,}\n")
        f.write(f"- **Agents fetched (sorted by {api_stats['sort']}):** {api_stats['fetched']}\n")
        f.write(f"- **Unique wallet addresses:** {dedup_stats['unique_wallets']}\n")
        f.write(f"- **Shared wallets (>1 agent per wallet):** {dedup_stats['shared_wallets']}\n")
        f.write(f"- **Max agents sharing one wallet:** {dedup_stats['max_sharing']}\n")
        f.write(f"- **Owner/agent wallet overlap:** {dedup_stats['owner_wallet_overlap']}\n\n")

        # Wallet sharing assessment
        sharing_pct = dedup_stats['shared_wallets'] / dedup_stats['unique_wallets'] * 100 if dedup_stats['unique_wallets'] else 0
        if dedup_stats['max_sharing'] > 10:
            f.write("**Wallet sharing assessment:** HIGH sharing detected. Like the Virtuals Protocol ")
            f.write("main API, many ACP agents share wallet infrastructure. Per-agent AHS scoring ")
            f.write(f"reflects shared wallet health, not individual agent behaviour. ({sharing_pct:.0f}% of wallets shared.)\n\n")
        elif dedup_stats['shared_wallets'] > 0:
            f.write(f"**Wallet sharing assessment:** MODERATE sharing detected ({sharing_pct:.0f}% of wallets shared, ")
            f.write(f"max {dedup_stats['max_sharing']} agents per wallet). Most ACP agent wallets are ")
            f.write("sufficiently independent for per-agent health scoring.\n\n")
        else:
            f.write("**Wallet sharing assessment:** No wallet sharing detected. ACP agent wallets are ")
            f.write("independent — ideal for per-agent health scoring.\n\n")

        # AHS results
        if scored:
            f.write("## AHS Scan Results\n\n")
            f.write(f"- **Wallets scanned:** {len(scan_results)}\n")
            f.write(f"- **Average AHS:** {avg_ahs:.1f}\n")
            f.write(f"- **AHS range:** {min_ahs}-{max_ahs}\n")
            f.write(f"- **Grade distribution:** {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}\n\n")

            # Grade breakdown table
            f.write("### Grade Distribution\n\n")
            f.write("| Grade | Count | % | Score Range |\n")
            f.write("|-------|-------|---|-------------|\n")
            boundaries = [("A", "90-100"), ("B", "75-89"), ("C", "60-74"),
                          ("D", "40-59"), ("E", "20-39"), ("F", "0-19")]
            for g, rng in boundaries:
                cnt = grade_dist.get(g, 0)
                pct = cnt / len(scored) * 100 if scored else 0
                f.write(f"| {g} | {cnt} | {pct:.1f}% | {rng} |\n")

            # Individual results table
            scored_agents = [a for a in agents if a.ahs_score is not None]
            if scored_agents:
                scored_agents.sort(key=lambda a: a.ahs_score or 0, reverse=True)
                f.write("\n### Scanned Agents (by AHS)\n\n")
                f.write("| ACP ID | Name | Wallet | Jobs | Revenue | AHS | Grade | D1 | D2 | Patterns |\n")
                f.write("|--------|------|--------|------|---------|-----|-------|----|----|----------|\n")
                for a in scored_agents:
                    wallet_short = f"`{a.wallet_address[:8]}...{a.wallet_address[-4:]}`"
                    name = a.name[:25] if a.name else f"Agent #{a.acp_id}"
                    pats = a.patterns if a.patterns and a.patterns != "none" else ""
                    rev = f"${a.revenue:,.0f}" if a.revenue else "$0"
                    f.write(f"| {a.acp_id} | {name} | {wallet_short} | {a.successful_job_count:,} | {rev} | {a.ahs_score} | {a.grade} {a.grade_label} | {a.d1_score} | {a.d2_score} | {pats} |\n")

            # Pattern frequency
            pattern_counts = {}
            for a in scored_agents:
                if a.patterns and a.patterns != "none":
                    for p in a.patterns.split("; "):
                        pattern_counts[p] = pattern_counts.get(p, 0) + 1
            if pattern_counts:
                f.write("\n### Pattern Frequency\n\n")
                f.write("| Pattern | Count |\n")
                f.write("|---------|-------|\n")
                for p, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
                    f.write(f"| {p} | {cnt} |\n")

        f.write("\n## Database Storage\n\n")
        f.write(f"- All {len(scan_results)} scan results persisted to `ahm_history.db` via `db.log_scan()`\n")
        f.write(f"- Source: `acp_proactive_scan`\n")
        f.write(f"- Registry tracking: `registries` column updated with `acp_proactive_scan`\n")
        f.write(f"- Labels format: `ACP #<id> — <name>`\n\n")

        f.write("## Spike Conclusions\n\n")
        f.write("### What worked\n\n")
        f.write("1. **ACP API is the best discovery source** — free, no auth, returns wallet addresses directly\n")
        f.write("2. **End-to-end flow proven** — discovery → dedup → scan → store → report\n")
        f.write("3. **Existing AHS engine works unchanged** — `calculate_ahs()` handles ACP wallets identically to ERC-8004\n")
        f.write("4. **`db.log_scan()` cross-registry tracking** — ACP scans integrate cleanly with existing schema\n\n")

        f.write("### Next steps to full pipeline\n\n")
        f.write("1. **Scheduled scanning** — cron/scheduler to run ACP discovery + AHS scans periodically\n")
        f.write("2. **402index.io integration** — second discovery source via payTo address extraction\n")
        f.write("3. **Dedup across registries** — merge ACP + ERC-8004 + 402index wallet sets\n")
        f.write("4. **Public Health Dashboard** — aggregate stats from all scanned wallets (backlog P4 deliverable)\n")

    print(f"[+] Markdown written: {MD_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ACP Proactive Ecosystem Scanner")
    parser.add_argument("--max-agents", type=int, default=500,
                        help="Max agents to fetch from ACP API (default: 500)")
    parser.add_argument("--max-scans", type=int, default=100,
                        help="Max AHS scans to perform (default: 100)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Discovery + dedup only, no AHS scanning")
    parser.add_argument("--force-rescan", action="store_true",
                        help="(Deprecated — rotation handles this) Ignored.")
    parser.add_argument("--sort", type=str, default="successfulJobCount:desc",
                        help="Sort order for ACP API (default: successfulJobCount:desc)")
    parser.add_argument("--clean-checkpoint", action="store_true",
                        help="Delete checkpoint file and start fresh")
    args = parser.parse_args()

    if args.clean_checkpoint and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[*] Checkpoint cleared")

    print("[*] ACP Proactive Ecosystem Scanner")
    print(f"[*] Source: {ACP_API_BASE}")
    print(f"[*] Max agents: {args.max_agents} | Max scans: {args.max_scans} | Sort: {args.sort}")

    # Phase 1: Discovery
    agents, api_stats = discover_agents(
        max_agents=args.max_agents, sort=args.sort,
    )

    if not agents:
        print("[!] No agents discovered. Exiting.")
        sys.exit(1)

    # Phase 2: Deduplication
    dedup_stats = deduplicate_wallets(agents)

    # Phase 3: AHS Scanning (stale-first rotation)
    scan_results = {}
    if not args.skip_scan:
        scan_results = scan_wallets(agents, max_scans=args.max_scans)
    else:
        print("\n[*] Skipping AHS scanning (--skip-scan)")

    # Phase 4: Report
    generate_report(agents, api_stats, dedup_stats, scan_results)

    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[+] Checkpoint cleaned up (scan complete)")

    print("\n" + "=" * 60)
    print("  SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
