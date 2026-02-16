#!/usr/bin/env python3
"""
Agent Address Discovery Tool
Automatically discovers AI agent wallets on Base L2 by querying on-chain
sources, scores them for agent-like behavior, and feeds qualifying addresses
into the health monitor.

Usage:
    python discover.py
    python discover.py --min-score 50
    python discover.py --skip-monitor
    python discover.py --top 30
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

# -- Configuration ----------------------------------------------------------

BASESCAN_API_URL = "https://base.blockscout.com/api"
BLOCKSCOUT_V2_URL = "https://base.blockscout.com/api/v2"
DELAY_BETWEEN_REQUESTS = 1.5

# Seed sources: name, address, API action, description
SEED_SOURCES = [
    {
        "name": "Coinbase Smart Wallet Factory",
        "address": "0x0ba5ed0c6aa8c49038f819e587e2633c4a9f428a",
        "action": "txlistinternal",
        "extract": "created_wallets",
    },
    {
        "name": "ERC-4337 EntryPoint",
        "address": "0x5FF137D4b0FDCD49DcA30c7CF57E578a026d2789",
        "action": "txlist",
        "extract": "callers",
    },
    {
        "name": "Uniswap V3 Router",
        "address": "0x2626664c2603336E57B271c5C0b26F421741e481",
        "action": "txlist",
        "extract": "callers",
    },
    {
        "name": "Aerodrome Router",
        "address": "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
        "action": "txlist",
        "extract": "callers",
    },
    {
        "name": "Uniswap Universal Router",
        "address": "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",
        "action": "txlist",
        "extract": "callers",
    },
]

ADDRESSES_FILE = "addresses.txt"
REPORT_FILE = "agent_health_report.csv"


# -- Data Model -------------------------------------------------------------

@dataclass
class Candidate:
    address: str
    sources: list[str] = field(default_factory=list)
    tx_count: int = 0
    token_transfers_count: int = 0
    is_contract: bool = False
    creation_timestamp: str = ""
    score: int = 0


# -- API Layer --------------------------------------------------------------

def basescan_get(params: dict) -> dict:
    """Make a Blockscout etherscan-compatible API request."""
    try:
        resp = requests.get(BASESCAN_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [!] API error: {e}")
        return {}


def blockscout_v2_get(path: str) -> dict:
    """Make a Blockscout v2 API request."""
    try:
        resp = requests.get(f"{BLOCKSCOUT_V2_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [!] V2 API error: {e}")
        return {}


# -- Phase 1: Seed Discovery -----------------------------------------------

def extract_created_wallets(transactions: list[dict]) -> list[str]:
    """Extract contract-creation target addresses from internal txns."""
    addrs = []
    for tx in transactions:
        # Internal txns from factory: 'contractAddress' or 'to' is the new wallet
        addr = tx.get("contractAddress", "")
        if not addr or addr == "0x" + "0" * 40:
            addr = tx.get("to", "")
        if addr and addr.startswith("0x") and len(addr) == 42:
            addrs.append(addr.lower())
    return addrs


def extract_callers(transactions: list[dict]) -> list[str]:
    """Extract unique 'from' addresses (callers) from txlist."""
    addrs = []
    for tx in transactions:
        addr = tx.get("from", "")
        if addr and addr.startswith("0x") and len(addr) == 42:
            addrs.append(addr.lower())
    return addrs


def discover_seeds() -> dict[str, list[str]]:
    """Phase 1: Query all seed sources and return {address: [source_names]}."""
    addr_sources: dict[str, list[str]] = {}

    for src in SEED_SOURCES:
        print(f"\n  [{src['name']}]")
        print(f"    Querying {src['action']} for {src['address'][:10]}...")

        params = {
            "module": "account",
            "action": src["action"],
            "address": src["address"],
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 500,
            "sort": "desc",
        }
        data = basescan_get(params)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        results = data.get("result", [])
        if not isinstance(results, list):
            print(f"    No results (status: {data.get('message', 'unknown')})")
            continue

        if src["extract"] == "created_wallets":
            found = extract_created_wallets(results)
        else:
            found = extract_callers(results)

        unique = list(dict.fromkeys(found))
        print(f"    Found {len(unique)} unique addresses")

        for addr in unique:
            if addr not in addr_sources:
                addr_sources[addr] = []
            if src["name"] not in addr_sources[addr]:
                addr_sources[addr].append(src["name"])

    return addr_sources


# -- Phase 2: Behavioral Scoring -------------------------------------------

def fetch_address_counters(address: str) -> dict:
    """Fetch tx_count and token_transfers_count via v2 API."""
    return blockscout_v2_get(f"/addresses/{address}/counters")


def fetch_address_info(address: str) -> dict:
    """Fetch address info (is_contract, creation_tx) via v2 API."""
    return blockscout_v2_get(f"/addresses/{address}")


def score_candidate(candidate: Candidate) -> int:
    """Score an address 0-100 for agent-like behavior."""
    score = 0

    # Transaction volume (0-30 pts)
    if candidate.tx_count > 100:
        score += 30
    elif candidate.tx_count > 50:
        score += 20
    elif candidate.tx_count > 20:
        score += 10

    # Is smart contract (0-20 pts)
    if candidate.is_contract:
        score += 20

    # DeFi activity ratio (0-20 pts): token_transfers / tx_count
    if candidate.tx_count > 0:
        ratio = candidate.token_transfers_count / candidate.tx_count
        if ratio > 2.0:
            score += 20
        elif ratio > 1.0:
            score += 15
        elif ratio > 0.5:
            score += 10
        elif ratio > 0.2:
            score += 5

    # Multi-source discovery (0-15 pts)
    n_sources = len(candidate.sources)
    if n_sources >= 3:
        score += 15
    elif n_sources == 2:
        score += 10
    elif n_sources == 1:
        score += 3

    # Age + activity profile (0-15 pts): young account with high activity
    if candidate.creation_timestamp:
        try:
            created = datetime.fromisoformat(
                candidate.creation_timestamp.replace("Z", "+00:00")
            )
            age_days = (datetime.now(timezone.utc) - created).days
            if age_days > 0:
                tx_per_day = candidate.tx_count / age_days
                if tx_per_day > 10:
                    score += 15
                elif tx_per_day > 5:
                    score += 12
                elif tx_per_day > 1:
                    score += 8
                elif tx_per_day > 0.3:
                    score += 4
        except (ValueError, TypeError):
            pass

    return score


def score_candidates(
    addr_sources: dict[str, list[str]], top_n: int
) -> list[Candidate]:
    """Phase 2: Fetch metrics and score top candidates."""
    # Rank by number of sources (descending), then alphabetically for stability
    ranked = sorted(
        addr_sources.items(), key=lambda x: (-len(x[1]), x[0])
    )[:top_n]

    candidates = []
    total = len(ranked)

    for i, (addr, sources) in enumerate(ranked, 1):
        print(f"\n  [{i}/{total}] Scoring {addr[:10]}...{addr[-6:]}")
        candidate = Candidate(address=addr, sources=sources)

        # Fetch counters
        counters = fetch_address_counters(addr)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        candidate.tx_count = _safe_int(counters.get("transactions_count", 0))
        candidate.token_transfers_count = _safe_int(
            counters.get("token_transfers_count", 0)
        )
        print(f"    Txns: {candidate.tx_count}, Token transfers: {candidate.token_transfers_count}")

        # Fetch address info
        info = fetch_address_info(addr)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        candidate.is_contract = info.get("is_contract", False)
        candidate.creation_timestamp = info.get("creation_tx_hash", "")
        # Try to get creation timestamp from the block if available
        if info.get("creator_address_hash"):
            candidate.is_contract = True
        if info.get("block_number_balance_updated_at"):
            # Use creation_transaction timestamp if present
            creation_tx = info.get("creation_transaction")
            if isinstance(creation_tx, dict):
                candidate.creation_timestamp = creation_tx.get("timestamp", "")

        type_label = "contract" if candidate.is_contract else "EOA"
        src_label = ", ".join(sources)
        print(f"    Type: {type_label} | Sources: {src_label}")

        # Score
        candidate.score = score_candidate(candidate)
        print(f"    Score: {candidate.score}/100")

        candidates.append(candidate)

    return candidates


def _safe_int(value) -> int:
    """Convert a value to int, handling strings and None."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# -- Phase 3: Output & Monitor Run -----------------------------------------

def write_addresses_file(
    candidates: list[Candidate], min_score: int, filepath: str
) -> list[Candidate]:
    """Write qualifying addresses to file with metadata comments."""
    qualified = [c for c in candidates if c.score >= min_score]
    qualified.sort(key=lambda c: -c.score)

    with open(filepath, "w") as f:
        f.write(f"# Agent Address Discovery — Auto-generated\n")
        f.write(
            f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        )
        f.write(f"# Candidates scored: {len(candidates)}\n")
        f.write(f"# Qualifying (score >= {min_score}): {len(qualified)}\n")
        f.write(f"# Score range: {qualified[-1].score}-{qualified[0].score}\n" if qualified else "")
        f.write(f"#\n")

        for c in qualified:
            src_str = "+".join(s.split()[0] for s in c.sources)
            type_str = "contract" if c.is_contract else "EOA"
            f.write(
                f"# score={c.score} | {type_str} | txns={c.tx_count} "
                f"| sources={src_str}\n"
            )
            f.write(f"{c.address}\n")

    print(f"\n[+] Wrote {len(qualified)} addresses to {filepath}")
    return qualified


def run_monitor(filepath: str, report: str) -> bool:
    """Run monitor.py on the discovered addresses."""
    cmd = [
        sys.executable, "monitor.py",
        "-f", filepath,
        "--check-contracts",
        "-o", report,
    ]
    print(f"\n[*] Running: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        print("[!] Could not find monitor.py — skipping monitor run")
        return False


def print_discovery_summary(
    total_seeds: int,
    scored: int,
    qualified: list[Candidate],
):
    """Print discovery results summary."""
    print(f"\n{'='*70}")
    print(f"  AGENT DISCOVERY SUMMARY — Base L2")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}")
    print(f"\n  Seed addresses found:    {total_seeds}")
    print(f"  Candidates scored:       {scored}")
    print(f"  Qualified agents:        {len(qualified)}")

    if qualified:
        contracts = sum(1 for c in qualified if c.is_contract)
        eoas = len(qualified) - contracts
        avg_score = sum(c.score for c in qualified) / len(qualified)
        print(f"  Average score:           {avg_score:.0f}/100")
        print(f"  Smart contracts:         {contracts}")
        print(f"  EOAs:                    {eoas}")

        print(f"\n  {'—'*66}")
        print(f"  TOP DISCOVERED AGENTS")
        print(f"  {'—'*66}")
        print(
            f"  {'Address':<44} {'Score':>5} {'Txns':>7} {'Type':<9} {'Sources':>3}"
        )
        print(f"  {'—'*66}")
        for c in qualified[:15]:
            type_str = "contract" if c.is_contract else "EOA"
            print(
                f"  {c.address:<44} {c.score:>5} {c.tx_count:>7} "
                f"{type_str:<9} {len(c.sources):>3}"
            )
        print(f"  {'—'*66}")


# -- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agent Address Discovery Tool — Base L2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discover.py                      # Full discovery + monitor run
  python discover.py --min-score 50       # Stricter filtering
  python discover.py --top 30             # Score top 30 candidates
  python discover.py --skip-monitor       # Discovery only, no monitor
  python discover.py -o my_agents.txt     # Custom output file
        """,
    )
    parser.add_argument(
        "--min-score", type=int, default=40,
        help="minimum agent score to qualify (default: 40)",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="number of top seed addresses to score (default: 50)",
    )
    parser.add_argument(
        "--skip-monitor", action="store_true",
        help="skip running monitor.py after discovery",
    )
    parser.add_argument(
        "-o", "--output", default=ADDRESSES_FILE,
        help=f"output addresses file (default: {ADDRESSES_FILE})",
    )
    parser.add_argument(
        "--report", default=REPORT_FILE,
        help=f"monitor report CSV path (default: {REPORT_FILE})",
    )
    args = parser.parse_args()

    print(f"[*] Agent Address Discovery Tool")
    print(f"[*] Network: Base L2 (Mainnet)")
    print(f"[*] API: Blockscout (free, {DELAY_BETWEEN_REQUESTS}s between requests)")
    print(f"[*] Scoring top {args.top} candidates, min score {args.min_score}")

    # -- Phase 1: Seed Discovery --
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Seed Discovery")
    print(f"{'='*70}")

    addr_sources = discover_seeds()
    total_seeds = len(addr_sources)
    print(f"\n[+] Phase 1 complete: {total_seeds} unique addresses from {len(SEED_SOURCES)} sources")

    if total_seeds == 0:
        print("[!] No addresses discovered. Check network connectivity.")
        sys.exit(1)

    # -- Phase 2: Behavioral Scoring --
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Behavioral Scoring")
    print(f"{'='*70}")

    candidates = score_candidates(addr_sources, args.top)
    qualified = write_addresses_file(candidates, args.min_score, args.output)

    # Summary
    print_discovery_summary(total_seeds, len(candidates), qualified)

    if not qualified:
        print("\n[!] No addresses met the minimum score threshold.")
        print(f"[*] Try lowering --min-score (currently {args.min_score})")
        sys.exit(0)

    # -- Phase 3: Monitor Run --
    if args.skip_monitor:
        print(f"\n[*] Skipping monitor run (--skip-monitor)")
        print(f"[*] Run manually: python monitor.py -f {args.output} --check-contracts")
    else:
        print(f"\n{'='*70}")
        print(f"  PHASE 3: Health Monitor")
        print(f"{'='*70}")
        success = run_monitor(args.output, args.report)
        if success:
            print(f"\n[+] Discovery + monitoring complete!")
            print(f"    Addresses: {args.output}")
            print(f"    Report:    {args.report}")
        else:
            print(f"\n[!] Monitor run had issues. Check output above.")
            print(f"    Addresses file is still available: {args.output}")


if __name__ == "__main__":
    main()
