#!/usr/bin/env python3
"""
AHM Agent Taxonomy Classification POC

Classifies AHM-scanned agents into taxonomy categories based on their
on-chain contract interaction patterns fetched from Basescan.

Usage:
    python scripts/classify_agents_taxonomy.py --db /data/ahm_history.db [--sample 500] [--api-key KEY]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASESCAN_API = "https://api.basescan.org/api"
RATE_LIMIT_SLEEP = 0.22  # ~4.5 req/s to stay under 5/s limit
TAXONOMY_JSON = Path(__file__).parent / "taxonomy_contracts.json"


def load_lookup_table(path: Path) -> tuple[dict, dict]:
    """Load contract→category lookup and subcategory signal sets."""
    with open(path) as f:
        data = json.load(f)
    contracts = {addr.lower(): info for addr, info in data["contracts"].items()}
    signals = {k: set(v) for k, v in data["subcategory_signals"].items()}
    return contracts, signals


def query_agents(db_path: str, sample: int) -> list[dict]:
    """Get top agents by tx_count from the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT s.address, s.ahs_score, s.grade, s.tx_count,
               json_extract(s.shadow_signals_json, '$.unique_contracts') as unique_contracts,
               w.registries, w.source
        FROM scans s
        JOIN known_wallets w ON w.address = s.address
        WHERE s.endpoint = 'ahs' AND s.ahs_score IS NOT NULL
          AND s.tx_count >= 5
        GROUP BY s.address
        HAVING MAX(s.scan_timestamp)
        ORDER BY s.tx_count DESC
        LIMIT ?
    """, (sample,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_transactions(address: str, api_key: str) -> list[dict]:
    """Fetch transaction list from Basescan for a given address."""
    params = (
        f"?module=account&action=txlist&address={address}"
        f"&sort=desc&offset=200&page=1&apikey={api_key}"
    )
    url = BASESCAN_API + params
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AHM-Taxonomy-POC/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "1" and isinstance(data.get("result"), list):
            return data["result"]
        return []
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return []


def classify_agent(
    agent: dict,
    txs: list[dict],
    contracts: dict,
    signals: dict,
) -> dict:
    """Classify a single agent based on its transaction targets."""
    address = agent["address"]
    registries = (agent.get("registries") or "").lower()
    source = (agent.get("source") or "").lower()
    is_acp = "acp" in registries
    is_olas = source == "olas"

    # Count interactions per contract (outgoing txs only)
    contract_counts = Counter()
    for tx in txs:
        if tx.get("from", "").lower() == address.lower() and tx.get("to"):
            contract_counts[tx["to"].lower()] += 1

    # Map to categories
    category_scores = defaultdict(int)
    matched_interactions = 0
    top_contracts = []

    for contract_addr, count in contract_counts.most_common():
        if contract_addr in contracts:
            cat = contracts[contract_addr]["category"]
            label = contracts[contract_addr]["label"]
            category_scores[cat] += count
            matched_interactions += count
            top_contracts.append((contract_addr, count, label))

    total_interactions = sum(contract_counts.values())

    # Two-stage ACP classification
    if is_acp and not category_scores:
        category_scores["Orchestration Agents"] = 1  # default for ACP agents

    if is_acp and category_scores:
        # Check if non-orchestration categories dominate via subcategory signals
        acp_sub_counts = {"dex": 0, "oracle": 0, "nft_media": 0}
        for contract_addr, count in contract_counts.items():
            for sig_name, sig_addrs in signals.items():
                if contract_addr in sig_addrs:
                    acp_sub_counts[sig_name] += count

        dominant_sub = max(acp_sub_counts, key=acp_sub_counts.get)
        dominant_count = acp_sub_counts[dominant_sub]

        if dominant_count > 0:
            reclassify_map = {
                "dex": "Financial Agents",
                "oracle": "Intelligence & Analytics",
                "nft_media": "Creative Agents",
            }
            # Only reclassify if the sub-signal is substantial
            orch_score = category_scores.get("Orchestration Agents", 0)
            if dominant_count > orch_score:
                category_scores[reclassify_map[dominant_sub]] += dominant_count

    # Olas boost
    if is_olas:
        category_scores["Orchestration Agents"] += max(1, total_interactions // 10)

    # Determine primary category and confidence
    if not category_scores:
        return {
            "address": address,
            "tx_count": agent.get("tx_count", 0),
            "unique_contracts": agent.get("unique_contracts"),
            "primary_category": None,
            "confidence": "UNCLASSIFIABLE",
            "category_scores": {},
            "top_contracts": [],
            "top_unmatched": [
                (addr, cnt) for addr, cnt in contract_counts.most_common(5)
                if addr not in contracts
            ],
            "registries": agent.get("registries", ""),
            "source": agent.get("source", ""),
            "total_interactions": total_interactions,
            "matched_interactions": 0,
        }

    primary = max(category_scores, key=category_scores.get)
    primary_count = category_scores[primary]
    total_matched = sum(category_scores.values())

    if total_matched > 0:
        ratio = primary_count / total_matched
        if ratio > 0.5:
            confidence = "HIGH"
        elif ratio > 0.25:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        confidence = "LOW"

    return {
        "address": address,
        "tx_count": agent.get("tx_count", 0),
        "unique_contracts": agent.get("unique_contracts"),
        "primary_category": primary,
        "confidence": confidence,
        "category_scores": dict(category_scores),
        "top_contracts": top_contracts[:5],
        "top_unmatched": [
            (addr, cnt) for addr, cnt in contract_counts.most_common(10)
            if addr not in contracts
        ][:5],
        "registries": agent.get("registries", ""),
        "source": agent.get("source", ""),
        "total_interactions": total_interactions,
        "matched_interactions": matched_interactions,
    }


def generate_report(results: list[dict], sample_size: int, elapsed: float) -> str:
    """Generate the gap analysis report text."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("=" * 70)
    lines.append("AHM AGENT TAXONOMY CLASSIFICATION — POC REPORT")
    lines.append(f"Generated: {now}")
    lines.append(f"Sample size: {sample_size} agents | Elapsed: {elapsed:.0f}s")
    lines.append("=" * 70)

    # 1. Coverage
    classified = [r for r in results if r["primary_category"]]
    unclassifiable = [r for r in results if not r["primary_category"]]
    total = len(results)

    lines.append("")
    lines.append("1. COVERAGE")
    lines.append("-" * 40)
    lines.append(f"   Classified:      {len(classified):>5}  ({len(classified)/total*100:.1f}%)" if total else "   No agents")
    lines.append(f"   Unclassifiable:  {len(unclassifiable):>5}  ({len(unclassifiable)/total*100:.1f}%)" if total else "")
    lines.append(f"   Total scanned:   {total:>5}")

    # 2. Category Distribution
    cat_counter = Counter(r["primary_category"] for r in classified)
    lines.append("")
    lines.append("2. CATEGORY DISTRIBUTION")
    lines.append("-" * 40)
    for cat, count in cat_counter.most_common():
        pct = count / len(classified) * 100 if classified else 0
        lines.append(f"   {cat:<30} {count:>5}  ({pct:.1f}%)")

    # 3. Confidence Distribution
    conf_counter = Counter(r["confidence"] for r in results)
    lines.append("")
    lines.append("3. CONFIDENCE DISTRIBUTION")
    lines.append("-" * 40)
    for level in ["HIGH", "MEDIUM", "LOW", "UNCLASSIFIABLE"]:
        count = conf_counter.get(level, 0)
        pct = count / total * 100 if total else 0
        lines.append(f"   {level:<20} {count:>5}  ({pct:.1f}%)")

    # 4. Top Unclassified Contracts
    unmatched_global = Counter()
    for r in results:
        for addr, cnt in r.get("top_unmatched", []):
            unmatched_global[addr] += cnt

    lines.append("")
    lines.append("4. TOP UNCLASSIFIED CONTRACTS (enrichment gaps)")
    lines.append("-" * 40)
    for addr, cnt in unmatched_global.most_common(30):
        lines.append(f"   {addr}  interactions={cnt}")

    # 5. Spot-Check Sample
    lines.append("")
    lines.append("5. SPOT-CHECK SAMPLE (10 agents)")
    lines.append("-" * 40)
    sample_agents = (classified[:7] + unclassifiable[:3]) if len(classified) >= 7 else results[:10]
    for r in sample_agents[:10]:
        lines.append(f"   Address:    {r['address']}")
        lines.append(f"   Category:   {r['primary_category'] or 'UNCLASSIFIABLE'}")
        lines.append(f"   Confidence: {r['confidence']}")
        lines.append(f"   TX count:   {r['tx_count']}")
        lines.append(f"   Registries: {r['registries'] or 'none'}")
        if r["top_contracts"]:
            lines.append(f"   Top contracts:")
            for addr, cnt, label in r["top_contracts"][:3]:
                lines.append(f"     - {label} ({addr[:10]}…) × {cnt}")
        if r["top_unmatched"]:
            lines.append(f"   Top unmatched:")
            for addr, cnt in r["top_unmatched"][:3]:
                lines.append(f"     - {addr} × {cnt}")
        lines.append("")

    # 6. Recommendations
    lines.append("6. RECOMMENDATIONS")
    lines.append("-" * 40)
    if unmatched_global:
        lines.append("   - The top unclassified contracts above are candidates for")
        lines.append("     addition to taxonomy_contracts.json to improve coverage.")
    if len(unclassifiable) > total * 0.3:
        lines.append("   - >30% unclassifiable rate suggests the lookup table needs")
        lines.append("     significant expansion. Consider batch-labelling the top 10")
        lines.append("     unmatched contracts.")
    lines.append("   - Consider adding ERC-20 token transfer analysis (tokentx)")
    lines.append("     for agents whose normal txlist has few matches.")
    lines.append("   - Internal transaction traces (txlistinternal) may reveal")
    lines.append("     additional contract interactions for proxy patterns.")

    # 7. ACP Enrichment Note
    lines.append("")
    lines.append("7. ACP ENRICHMENT NOTE")
    lines.append("-" * 40)
    lines.append("   Virtuals ACP v2 deploys per-job contracts dynamically.")
    lines.append("   The Virtuals API (app.virtuals.io/acp) exposes agent")
    lines.append("   self-described categories that could be used as a future")
    lines.append("   enrichment source for Orchestration sub-classification.")
    lines.append("   This is out of scope for the POC but recommended for v2.")

    acp_agents = [r for r in results if "acp" in (r.get("registries") or "").lower()]
    if acp_agents:
        lines.append(f"   ACP-registry agents in sample: {len(acp_agents)}")
        acp_cats = Counter(r["primary_category"] for r in acp_agents if r["primary_category"])
        for cat, cnt in acp_cats.most_common():
            lines.append(f"     - {cat}: {cnt}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AHM Agent Taxonomy Classification POC")
    parser.add_argument("--db", required=True, help="Path to AHM SQLite database")
    parser.add_argument("--sample", type=int, default=500, help="Max agents to classify")
    parser.add_argument("--api-key", default=None, help="Basescan API key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("BASESCAN_API_KEY", "")
    if not api_key:
        print("ERROR: No Basescan API key. Use --api-key or set BASESCAN_API_KEY", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading lookup table from {TAXONOMY_JSON}")
    contracts, signals = load_lookup_table(TAXONOMY_JSON)
    print(f"  {len(contracts)} contracts across categories")

    print(f"Querying database for top {args.sample} agents...")
    agents = query_agents(args.db, args.sample)
    print(f"  Found {len(agents)} agents with tx_count >= 5")

    if not agents:
        print("No agents found. Exiting.")
        sys.exit(0)

    start = time.time()
    results = []
    errors = 0

    for i, agent in enumerate(agents):
        addr = agent["address"]
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(agents)}] Processing {addr[:10]}… ({elapsed:.0f}s elapsed)")

        txs = fetch_transactions(addr, api_key)
        if not txs and agent.get("tx_count", 0) > 0:
            errors += 1

        result = classify_agent(agent, txs, contracts, signals)
        results.append(result)
        time.sleep(RATE_LIMIT_SLEEP)

    elapsed = time.time() - start
    print(f"\nClassification complete: {len(results)} agents in {elapsed:.0f}s ({errors} API errors)")

    report = generate_report(results, len(agents), elapsed)

    # Write report
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    report_path = docs_dir / f"taxonomy-poc-report-{date_str}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")

    # Also print to stdout
    print("\n" + report)


if __name__ == "__main__":
    main()
