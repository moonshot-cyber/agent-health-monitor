#!/usr/bin/env python3
"""Virtuals ACP agent wallet scan.

Queries the Virtuals Protocol API to discover agents, extract wallet addresses,
run AHS scans, and store results in the scan history database.

Usage:
    python virtuals_scan.py                              # Default (30 pages, 100 scans)
    python virtuals_scan.py --max-pages 50 --max-scans 200
    python virtuals_scan.py --skip-scan                  # Enumerate only
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

VIRTUALS_API = "https://api.virtuals.io/api/virtuals"
PAGE_SIZE = 10  # API max
API_DELAY = 0.5  # seconds between page fetches
BLOCKSCOUT_DELAY = 2.0  # Blockscout API rate limit

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

CSV_PATH = "virtuals_scan_results.csv"
MD_PATH = "virtuals_scan_results.md"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VirtualAgent:
    virtuals_id: int = 0
    acp_agent_id: int | None = None
    name: str = ""
    symbol: str = ""
    chain: str = ""
    status: str = ""
    holder_count: int = 0
    volume_24h: float = 0.0
    tvl: str = ""
    # Wallet addresses
    wallet_address: str = ""      # Creator wallet
    tba_address: str = ""         # Token Bound Account (agent operational wallet)
    sentient_address: str = ""    # Sentient wallet
    token_address: str = ""       # Token contract
    # AHS results (filled for the primary scanned wallet)
    scanned_wallet: str = ""
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
# Phase 1: API Discovery
# ---------------------------------------------------------------------------

def discover_api():
    """Validate Virtuals API connectivity and get total agent count."""
    print("\n" + "=" * 60)
    print("  PHASE 1: API Discovery")
    print("=" * 60)

    try:
        resp = requests.get(
            VIRTUALS_API,
            params={"page": 1, "limit": PAGE_SIZE},
            timeout=15,
            headers={"Accept": "application/json", "User-Agent": "AHM-VirtualsScan/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[!] API connection failed: {e}")
        sys.exit(1)

    meta = data.get("meta", {}).get("pagination", {})
    total = meta.get("total", 0)
    page_count = meta.get("pageCount", 0)
    page_size = meta.get("pageSize", PAGE_SIZE)

    print(f"[+] API connected: {VIRTUALS_API}")
    print(f"[+] Total agents: {total:,}")
    print(f"[+] Pages: {page_count:,} (page size: {page_size})")

    # Sample first agent
    agents = data.get("data", [])
    if agents:
        a = agents[0]
        print(f"[+] Sample: {a.get('name', '?')} (#{a.get('id', '?')})")
        print(f"    walletAddress: {a.get('walletAddress', 'null')}")
        print(f"    tbaAddress: {a.get('tbaAddress', 'null')}")
        print(f"    sentientWalletAddress: {a.get('sentientWalletAddress', 'null')}")
        print(f"    acpAgentId: {a.get('acpAgentId', 'null')}")

    return {"total": total, "page_count": page_count, "page_size": page_size}


# ---------------------------------------------------------------------------
# Phase 2: Agent Enumeration
# ---------------------------------------------------------------------------

def enumerate_agents(max_pages=30) -> list[VirtualAgent]:
    """Paginate through the Virtuals API and collect agent records."""
    print("\n" + "=" * 60)
    print("  PHASE 2: Agent Enumeration")
    print("=" * 60)
    print(f"[*] Fetching up to {max_pages} pages ({max_pages * PAGE_SIZE} agents)...")

    agents = []
    for page in range(1, max_pages + 1):
        if page % 5 == 1:
            print(f"  [page {page}/{max_pages}] {len(agents)} agents collected...")

        try:
            resp = requests.get(
                VIRTUALS_API,
                params={"page": page, "limit": PAGE_SIZE},
                timeout=15,
                headers={"Accept": "application/json", "User-Agent": "AHM-VirtualsScan/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [!] Page {page} failed: {e}")
            time.sleep(API_DELAY * 2)
            continue

        items = data.get("data", [])
        if not items:
            print(f"  [!] No data on page {page}, stopping")
            break

        for item in items:
            rec = VirtualAgent(
                virtuals_id=item.get("id", 0),
                acp_agent_id=item.get("acpAgentId"),
                name=(item.get("name") or "")[:80],
                symbol=(item.get("symbol") or "")[:20],
                chain=item.get("chain", ""),
                status=item.get("status", ""),
                holder_count=item.get("holderCount") or 0,
                volume_24h=item.get("volume24h") or 0.0,
                tvl=str(item.get("totalValueLocked") or ""),
                wallet_address=(item.get("walletAddress") or "").lower(),
                tba_address=(item.get("tbaAddress") or "").lower(),
                sentient_address=(item.get("sentientWalletAddress") or "").lower(),
                token_address=(item.get("tokenAddress") or "").lower(),
            )
            agents.append(rec)

        time.sleep(API_DELAY)

    # Stats
    with_wallet = sum(1 for a in agents if a.wallet_address)
    with_tba = sum(1 for a in agents if a.tba_address)
    with_sentient = sum(1 for a in agents if a.sentient_address)
    with_acp = sum(1 for a in agents if a.acp_agent_id)

    print(f"[+] Enumerated {len(agents)} agents")
    print(f"    With walletAddress: {with_wallet}")
    print(f"    With tbaAddress: {with_tba}")
    print(f"    With sentientWalletAddress: {with_sentient}")
    print(f"    With acpAgentId: {with_acp}")

    return agents


# ---------------------------------------------------------------------------
# Phase 3: Deduplication
# ---------------------------------------------------------------------------

def deduplicate_wallets(agents: list[VirtualAgent]) -> tuple[dict[str, list[VirtualAgent]], set[str]]:
    """Collect unique wallets, check against DB for already-scanned ones.

    Returns (wallet_to_agents_map, already_scanned_set).
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: Deduplication")
    print("=" * 60)

    # Collect unique wallets, preferring tba > sentient > wallet
    wallet_to_agents: dict[str, list[VirtualAgent]] = {}
    for agent in agents:
        # Pick the best wallet to scan for this agent
        for addr in [agent.tba_address, agent.sentient_address, agent.wallet_address]:
            if addr and addr != ZERO_ADDRESS and len(addr) == 42:
                wallet_to_agents.setdefault(addr, []).append(agent)
                if not agent.scanned_wallet:
                    agent.scanned_wallet = addr

    unique_wallets = set(wallet_to_agents.keys())
    print(f"[+] Unique wallets extracted: {len(unique_wallets)}")

    # Check DB for already-scanned wallets
    import db
    db.init_db()

    already_scanned = set()
    cross_registry_hits = []
    conn = db.get_connection()
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT address, source, registries FROM known_wallets"
        ).fetchall()
        db_wallets = {r[0]: (r[1], r[2]) for r in rows}

        for addr in unique_wallets:
            if addr in db_wallets:
                src, regs = db_wallets[addr]
                cross_registry_hits.append((addr, src, regs))

        # Only skip if scanned today
        today_rows = conn.execute(
            "SELECT address FROM known_wallets WHERE last_scanned_at >= ?",
            (today + "T00:00:00Z",),
        ).fetchall()
        already_scanned = {r[0] for r in today_rows} & unique_wallets
    finally:
        conn.close()

    new_wallets = unique_wallets - already_scanned
    print(f"[+] Already in DB: {len(cross_registry_hits)} (cross-registry overlap)")
    print(f"[+] Scanned today (skip): {len(already_scanned)}")
    print(f"[+] New wallets to scan: {len(new_wallets)}")

    if cross_registry_hits:
        print(f"[+] Cross-registry examples:")
        for addr, src, regs in cross_registry_hits[:5]:
            print(f"    {addr[:14]}... source={src} registries={regs}")

    return wallet_to_agents, already_scanned


# ---------------------------------------------------------------------------
# Phase 4: AHS Scanning
# ---------------------------------------------------------------------------

def scan_wallets(
    agents: list[VirtualAgent],
    wallet_to_agents: dict[str, list[VirtualAgent]],
    already_scanned: set[str],
    max_scans: int = 100,
) -> dict:
    """Run AHS 2D scans on unique wallet addresses."""
    print("\n" + "=" * 60)
    print("  PHASE 4: AHS Scanning")
    print("=" * 60)

    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price
    import db

    db.init_db()

    eth_price = get_eth_price()
    print(f"[+] ETH price: ${eth_price:,.2f}")

    unique_wallets = [a for a in wallet_to_agents.keys() if a not in already_scanned]
    print(f"[+] Wallets to scan: {len(unique_wallets)} (max {max_scans})")

    results = {}
    scan_count = 0

    for address in unique_wallets:
        if scan_count >= max_scans:
            print(f"  [stop] Reached max_scans={max_scans}")
            break

        if (scan_count + 1) % 10 == 0:
            print(f"  [{scan_count + 1}/{min(max_scans, len(unique_wallets))}] scanning {address[:10]}...")

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

            # Build label from agent info
            source_agents = wallet_to_agents.get(address, [])
            agent_name = next((a.name for a in source_agents if a.name), "")
            vid = source_agents[0].virtuals_id if source_agents else 0
            label = f"Virtuals #{vid}"
            if agent_name:
                label += f" — {agent_name}"

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.log_scan(
                address=address,
                endpoint="ahs",
                scan_timestamp=now_iso,
                source="virtuals_scan",
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
                },
                shadow_signals={
                    k: ahs._signals.get(k, d)
                    for k, d in [
                        ("session_continuity_score", None),
                        ("abrupt_sessions", 0),
                        ("budget_exhaustion_count", 0),
                        ("total_sessions", 0),
                        ("avg_session_length", 0.0),
                        ("shadow_patterns", []),
                    ]
                },
            )

            result = {
                "ahs_score": ahs.agent_health_score,
                "grade": ahs.grade,
                "grade_label": ahs.grade_label,
                "confidence": ahs.confidence,
                "d1_score": ahs.d1_score,
                "d2_score": ahs.d2_score,
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

            scan_count += 1

        except Exception as e:
            err_msg = str(e)[:80]
            print(f"  [!] Error scanning {address[:10]}...: {err_msg}")
            for agent in wallet_to_agents.get(address, []):
                if not agent.scan_error:
                    agent.scan_error = err_msg

    print(f"[+] Scanned {scan_count} wallets")
    return results


# ---------------------------------------------------------------------------
# Phase 5: Report
# ---------------------------------------------------------------------------

def generate_report(
    agents: list[VirtualAgent],
    api_info: dict,
    scan_results: dict,
    cross_registry_count: int,
) -> None:
    """Print summary, write CSV and Markdown."""
    print("\n" + "=" * 60)
    print("  PHASE 5: Report")
    print("=" * 60)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    enumerated = len(agents)
    with_acp = sum(1 for a in agents if a.acp_agent_id)
    scanned = len(scan_results)
    scored = [r for r in scan_results.values() if r.get("ahs_score") is not None]

    grade_dist = {}
    for r in scored:
        g = r["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    avg_ahs = sum(r["ahs_score"] for r in scored) / len(scored) if scored else 0
    scores = [r["ahs_score"] for r in scored]
    min_ahs = min(scores) if scores else 0
    max_ahs = max(scores) if scores else 0

    # Console
    print(f"\n  Total Virtuals agents:        {api_info.get('total', 0):,}")
    print(f"  Agents enumerated:            {enumerated}")
    print(f"  With ACP ID:                  {with_acp}")
    print(f"  Cross-registry overlap:       {cross_registry_count}")
    print(f"  Unique wallets scanned (AHS): {scanned}")
    if scored:
        print(f"  Average AHS:                  {avg_ahs:.1f}")
        print(f"  AHS range:                    {min_ahs}-{max_ahs}")
        print(f"  Grade distribution:           {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}")

    # -- CSV --
    fieldnames = [
        "virtuals_id", "acp_agent_id", "name", "symbol", "chain", "status",
        "holder_count", "volume_24h", "wallet_address", "tba_address",
        "sentient_address", "scanned_wallet", "ahs", "grade", "grade_label",
        "d1", "d2", "confidence", "patterns", "tx_count", "history_days", "scan_error",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in agents:
            writer.writerow({
                "virtuals_id": a.virtuals_id,
                "acp_agent_id": a.acp_agent_id or "",
                "name": a.name,
                "symbol": a.symbol,
                "chain": a.chain,
                "status": a.status,
                "holder_count": a.holder_count,
                "volume_24h": round(a.volume_24h, 2) if a.volume_24h else "",
                "wallet_address": a.wallet_address,
                "tba_address": a.tba_address,
                "sentient_address": a.sentient_address,
                "scanned_wallet": a.scanned_wallet,
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
        f.write("# Virtuals ACP Agent Scan Results\n\n")
        f.write(f"> Scanned: {now_str}\n")
        f.write(f"> API: `{VIRTUALS_API}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Virtuals agents:** {api_info.get('total', 0):,}\n")
        f.write(f"- **Agents enumerated:** {enumerated}\n")
        f.write(f"- **With ACP ID:** {with_acp}\n")
        f.write(f"- **Cross-registry overlap (already in DB):** {cross_registry_count}\n")
        f.write(f"- **Unique wallets scanned (AHS):** {scanned}\n")
        if scored:
            f.write(f"- **Average AHS:** {avg_ahs:.1f}\n")
            f.write(f"- **AHS range:** {min_ahs}-{max_ahs}\n")
            f.write(f"- **Grade distribution:** {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}\n")

        if grade_dist:
            f.write("\n## Grade Distribution\n\n")
            f.write("| Grade | Count | % | Score Range |\n")
            f.write("|-------|-------|---|-------------|\n")
            boundaries = [("A", "90-100"), ("B", "75-89"), ("C", "60-74"),
                          ("D", "40-59"), ("E", "20-39"), ("F", "0-19")]
            for g, rng in boundaries:
                cnt = grade_dist.get(g, 0)
                pct = cnt / len(scored) * 100 if scored else 0
                f.write(f"| {g} | {cnt} | {pct:.1f}% | {rng} |\n")

        # Build unique wallet list with representative agent (highest holders)
        wallet_reps = {}  # addr -> best VirtualAgent for that wallet
        for a in agents:
            if a.ahs_score is not None and a.scanned_wallet:
                prev = wallet_reps.get(a.scanned_wallet)
                if prev is None or a.holder_count > prev.holder_count:
                    wallet_reps[a.scanned_wallet] = a
        deduped = list(wallet_reps.values())

        # Also count how many agents share each wallet
        wallet_agent_counts = {}
        for a in agents:
            if a.scanned_wallet:
                wallet_agent_counts[a.scanned_wallet] = wallet_agent_counts.get(a.scanned_wallet, 0) + 1

        # E/F grade wallets (outreach targets)
        critical = [a for a in deduped if a.grade in ("E", "F")]
        if critical:
            critical.sort(key=lambda a: a.ahs_score or 0)
            f.write("\n## Critical Wallets (E/F Grade) — Outreach Targets\n\n")
            f.write("| Virtuals ID | Name | Symbol | Wallet | AHS | Grade | D1 | D2 | Patterns | Agents Sharing |\n")
            f.write("|-------------|------|--------|--------|-----|-------|----|----|----------|----------------|\n")
            for a in critical:
                pats = a.patterns if a.patterns and a.patterns != "none" else ""
                shared = wallet_agent_counts.get(a.scanned_wallet, 1)
                f.write(f"| {a.virtuals_id} | {a.name[:30]} | {a.symbol} | `{a.scanned_wallet[:10]}...` | {a.ahs_score} | {a.grade} {a.grade_label} | {a.d1_score} | {a.d2_score} | {pats} | {shared} |\n")

        # B+ grade wallets (healthy case studies)
        healthy = [a for a in deduped if a.grade in ("A", "B")]
        if healthy:
            healthy.sort(key=lambda a: a.ahs_score or 0, reverse=True)
            f.write("\n## Healthy Wallets (B+ Grade) — Case Studies\n\n")
            f.write("| Virtuals ID | Name | Symbol | Wallet | AHS | Grade | D1 | D2 | Holders | Agents Sharing |\n")
            f.write("|-------------|------|--------|--------|-----|-------|----|----|---------|----------------|\n")
            for a in healthy:
                shared = wallet_agent_counts.get(a.scanned_wallet, 1)
                f.write(f"| {a.virtuals_id} | {a.name[:30]} | {a.symbol} | `{a.scanned_wallet[:10]}...` | {a.ahs_score} | {a.grade} {a.grade_label} | {a.d1_score} | {a.d2_score} | {a.holder_count:,} | {shared} |\n")

        # All scanned wallets (deduplicated, 1 row per unique wallet)
        if deduped:
            deduped.sort(key=lambda a: a.ahs_score or 0, reverse=True)
            f.write("\n## All Scanned Wallets (by AHS, deduplicated)\n\n")
            f.write("| Virtuals ID | Name | Symbol | Wallet | AHS | Grade | D1 | D2 | Patterns | Agents Sharing |\n")
            f.write("|-------------|------|--------|--------|-----|-------|----|----|----------|----------------|\n")
            for a in deduped:
                pats = a.patterns if a.patterns and a.patterns != "none" else ""
                shared = wallet_agent_counts.get(a.scanned_wallet, 1)
                f.write(f"| {a.virtuals_id} | {a.name[:30]} | {a.symbol} | `{a.scanned_wallet[:10]}...` | {a.ahs_score} | {a.grade} {a.grade_label} | {a.d1_score} | {a.d2_score} | {pats} | {shared} |\n")

        # Wallet sharing stats
        if wallet_agent_counts:
            f.write("\n## Wallet Sharing Analysis\n\n")
            total_wallets = len(wallet_agent_counts)
            shared_wallets = sum(1 for v in wallet_agent_counts.values() if v > 1)
            max_sharing = max(wallet_agent_counts.values())
            f.write(f"- **Unique wallets:** {total_wallets}\n")
            f.write(f"- **Shared by 2+ agents:** {shared_wallets}\n")
            f.write(f"- **Max agents per wallet:** {max_sharing}\n")
            f.write(f"- **Total agents enumerated:** {enumerated}\n")
            f.write(f"- **Avg agents per wallet:** {enumerated / total_wallets:.1f}\n")

        # Pattern frequency (from deduped)
        pattern_counts = {}
        for a in deduped:
            if a.patterns and a.patterns != "none":
                for p in a.patterns.split("; "):
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
        if pattern_counts:
            f.write("\n## Pattern Frequency\n\n")
            f.write("| Pattern | Count |\n")
            f.write("|---------|-------|\n")
            for p, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
                f.write(f"| {p} | {cnt} |\n")

    print(f"[+] Markdown written: {MD_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Virtuals ACP Agent Wallet Scan")
    parser.add_argument("--max-pages", type=int, default=30, help="Max API pages to fetch (default: 30)")
    parser.add_argument("--max-scans", type=int, default=100, help="Max AHS scans to perform (default: 100)")
    parser.add_argument("--skip-scan", action="store_true", help="Enumerate only, no AHS scanning")
    args = parser.parse_args()

    print("[*] Virtuals ACP Agent Wallet Scan")
    print(f"[*] API: {VIRTUALS_API}")
    print(f"[*] Max pages: {args.max_pages} | Max scans: {args.max_scans}")

    # Phase 1
    api_info = discover_api()

    # Phase 2
    agents = enumerate_agents(max_pages=args.max_pages)

    # Phase 3
    wallet_to_agents, already_scanned = deduplicate_wallets(agents)
    # Count actual cross-registry overlap (wallets in DB from other non-virtuals sources)
    import db
    db.init_db()
    conn = db.get_connection()
    try:
        existing = conn.execute(
            "SELECT address FROM known_wallets WHERE source != 'virtuals_scan'"
        ).fetchall()
        existing_set = {r[0] for r in existing}
        cross_registry_count = len(set(wallet_to_agents.keys()) & existing_set)
    finally:
        conn.close()

    # Phase 4
    scan_results = {}
    if not args.skip_scan:
        scan_results = scan_wallets(agents, wallet_to_agents, already_scanned, max_scans=args.max_scans)
    else:
        print("\n[*] Skipping AHS scanning (--skip-scan)")

    # Phase 5
    generate_report(agents, api_info, scan_results, cross_registry_count)

    print("\n" + "=" * 60)
    print("  SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
