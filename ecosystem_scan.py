#!/usr/bin/env python3
"""
Ecosystem Scan — Phase 1
Proactive scanning of real agent wallet addresses from x402 registries.
Runs lightweight AHS (2D mode, no Nansen) against discovered addresses.
"""

import csv
import time
from dataclasses import asdict
from datetime import datetime, timezone

from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price

DELAY = 2.0  # seconds between addresses to respect rate limits

# -- Real x402 agent payment addresses (harvested from .well-known/x402 endpoints) --
X402_AGENTS = [
    ("BlockRun (LLM/Search/Images)", "0xe9030014F5DAe217d0A152f02A043567b16c1aBf"),
    ("BlockRun (X/Twitter)", "0x1b092B21BC07076E3adEDe7bAc8216daEa115e99"),
    ("zeroreader", "0xCa99149c1A5959F7E5968259178f974aACC70F55"),
    ("httpay.xyz", "0x5f5d6FcB315871c26F720dc6fEf17052dD984359"),
    ("Hugen.tokyo (10 services)", "0x29322Ea7EcB34aA6164cb2ddeB9CE650902E4f60"),
]

# -- Top discovered on-chain agent wallets (from discover.py, score >= 50) --
ONCHAIN_AGENTS = [
    ("Onchain-75-contract", "0x464fc339add314932920d3e060745bd7ea3e92ad"),
    ("Onchain-75-contract", "0x622661ab4b6ab93c659e751f47ebb0c6e6ad9f48"),
    ("Onchain-75-contract", "0x8678f58ac6c4748b5289d0db70e627eef395dead"),
    ("Onchain-75-contract", "0xf5c299316699131d29adcb7ef87af8e97bbc7ead"),
    ("Onchain-70-contract", "0x560a9ccd6b43ea4ff7c12c5a48dcf5d2eafa018d"),
    ("Onchain-68-contract", "0x0071c69c37a4410ebafc501d841c8c992c0deb9a"),
    ("Onchain-60-EOA", "0xd13da05b9288ba4961973110594bd0fe3428791f"),
    ("Onchain-55-EOA", "0x1dc89ab25ab5d8714fcf9ee4bd9c9a58debeb4d8"),
    ("Onchain-55-EOA", "0xa73072adc6c34859426fcc29bc6ca2cac07c93c3"),
    ("Onchain-55-EOA", "0xc2d2ad92f7786786d978c45d7181af2c1f461007"),
    ("Onchain-55-EOA", "0xf9b6a1eb0190bf76274b0876957ee9f4f508af41"),
    ("Onchain-50-EOA", "0x15eb4202553d4795d3a3f4e927c1dfd6e9721ba9"),
    ("Onchain-50-EOA", "0xfb1c505c975e229a7c9746141971abdab71f46c4"),
]

ALL_AGENTS = X402_AGENTS + ONCHAIN_AGENTS

CSV_PATH = "ecosystem_scan_results.csv"
MD_PATH = "ecosystem_scan_results.md"


def main():
    print(f"[*] Ecosystem Scan — Phase 1")
    print(f"[*] {len(ALL_AGENTS)} addresses to scan")
    print(f"[*] Fetching ETH price...")

    eth_price = get_eth_price()
    print(f"[+] ETH price: ${eth_price:,.2f}")

    results = []

    for i, (label, address) in enumerate(ALL_AGENTS, 1):
        print(f"\n[{i}/{len(ALL_AGENTS)}] {label}: {address[:10]}...{address[-6:]}")

        try:
            txs = fetch_transactions(address)
            print(f"  Transactions: {len(txs)}")
            time.sleep(DELAY)

            tokens = fetch_tokens_v2(address, max_pages=3)
            print(f"  Tokens: {len(tokens)}")
            time.sleep(DELAY)

            ahs = calculate_ahs(
                address=address.lower(),
                tokens=tokens,
                transactions=txs,
                eth_price=eth_price,
            )

            patterns = "; ".join(
                p.get("pattern", str(p)) if isinstance(p, dict) else str(p)
                for p in ahs.patterns_detected
            ) if ahs.patterns_detected else "none"

            row = {
                "source": label,
                "address": address,
                "ahs": ahs.agent_health_score,
                "grade": ahs.grade,
                "grade_label": ahs.grade_label,
                "confidence": ahs.confidence,
                "d1_wallet": ahs.d1_score,
                "d2_behaviour": ahs.d2_score,
                "patterns": patterns,
                "tx_count": ahs.tx_count,
                "history_days": ahs.history_days,
            }
            results.append(row)
            print(f"  AHS: {ahs.agent_health_score} ({ahs.grade} — {ahs.grade_label}) | D1={ahs.d1_score} D2={ahs.d2_score} | Patterns: {patterns}")

        except Exception as e:
            print(f"  [!] Error: {e}")
            results.append({
                "source": label,
                "address": address,
                "ahs": "ERROR",
                "grade": "",
                "grade_label": "",
                "confidence": "",
                "d1_wallet": "",
                "d2_behaviour": "",
                "patterns": str(e),
                "tx_count": "",
                "history_days": "",
            })

    # -- Write CSV --
    if results:
        fieldnames = list(results[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[+] CSV written: {CSV_PATH}")

    # -- Write Markdown --
    with open(MD_PATH, "w") as f:
        f.write("# Ecosystem Scan Results — Phase 1\n\n")
        f.write(f"> Scanned: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"> Addresses: {len(ALL_AGENTS)} | ETH price: ${eth_price:,.2f}\n\n")

        f.write("## x402 Registry Agents\n\n")
        f.write("| Service | Address | AHS | Grade | D1 | D2 | Patterns |\n")
        f.write("|---------|---------|-----|-------|----|----|----------|\n")
        for r in results[:len(X402_AGENTS)]:
            addr_short = f"{r['address'][:8]}...{r['address'][-4:]}"
            f.write(f"| {r['source']} | `{addr_short}` | {r['ahs']} | {r['grade']} {r['grade_label']} | {r['d1_wallet']} | {r['d2_behaviour']} | {r['patterns']} |\n")

        f.write("\n## On-Chain Discovered Agents\n\n")
        f.write("| Source | Address | AHS | Grade | D1 | D2 | Patterns |\n")
        f.write("|--------|---------|-----|-------|----|----|----------|\n")
        for r in results[len(X402_AGENTS):]:
            addr_short = f"{r['address'][:8]}...{r['address'][-4:]}"
            f.write(f"| {r['source']} | `{addr_short}` | {r['ahs']} | {r['grade']} {r['grade_label']} | {r['d1_wallet']} | {r['d2_behaviour']} | {r['patterns']} |\n")

        # Summary stats
        scored = [r for r in results if isinstance(r.get("ahs"), int)]
        if scored:
            scores = [r["ahs"] for r in scored]
            avg = sum(scores) / len(scores)
            f.write(f"\n## Summary\n\n")
            f.write(f"- **Scanned:** {len(ALL_AGENTS)} addresses\n")
            f.write(f"- **Scored:** {len(scored)}\n")
            f.write(f"- **Average AHS:** {avg:.0f}\n")
            f.write(f"- **Range:** {min(scores)}-{max(scores)}\n")

            grade_dist = {}
            for r in scored:
                g = r["grade"]
                grade_dist[g] = grade_dist.get(g, 0) + 1
            f.write(f"- **Grade distribution:** {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}\n")

            patterns_seen = {}
            for r in scored:
                if r["patterns"] != "none":
                    for p in r["patterns"].split("; "):
                        patterns_seen[p] = patterns_seen.get(p, 0) + 1
            if patterns_seen:
                f.write(f"- **Patterns detected:** {', '.join(f'{p} ({n}x)' for p, n in sorted(patterns_seen.items(), key=lambda x: -x[1]))}\n")

    print(f"[+] Markdown written: {MD_PATH}")
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE")
    if scored:
        print(f"  Avg AHS: {avg:.0f} | Range: {min(scores)}-{max(scores)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
