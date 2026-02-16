#!/usr/bin/env python3
"""
Agent Transaction Health Monitor
Monitors AI agent wallets on Base L2, detects transaction failures,
calculates wasted gas, and generates health reports for optimization outreach.

Usage:
    python monitor.py 0xADDRESS1 0xADDRESS2 ...
    python monitor.py -f addresses.txt
    python monitor.py -f addresses.txt -k YOUR_BASESCAN_API_KEY
"""

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

# -- Configuration ------------------------------------------------------

# Blockscout etherscan-compatible API (free, no key required)
BASESCAN_API_URL = "https://base.blockscout.com/api"
BASE_RPC_URL = "https://mainnet.base.org"

# Blockscout is free but has rate limits (~50 req/min)
DELAY_BETWEEN_REQUESTS = 1.5

# Health score weights (sum to 1.0)
W_SUCCESS_RATE = 0.50
W_GAS_EFFICIENCY = 0.25
W_COST_EFFICIENCY = 0.15
W_NONCE_HEALTH = 0.10

ETH_PRICE_FALLBACK = 2500.0


# -- Data Model ---------------------------------------------------------

@dataclass
class AgentHealth:
    address: str
    is_contract: bool = False
    total_transactions: int = 0
    successful: int = 0
    failed: int = 0
    success_rate_pct: float = 0.0
    total_gas_spent_eth: float = 0.0
    wasted_gas_eth: float = 0.0
    estimated_monthly_waste_usd: float = 0.0
    avg_gas_efficiency_pct: float = 0.0
    out_of_gas_count: int = 0
    reverted_count: int = 0
    nonce_gap_count: int = 0
    retry_count: int = 0
    health_score: float = 0.0
    optimization_priority: str = "LOW"
    top_failure_type: str = "none"
    first_seen: str = ""
    last_seen: str = ""


# -- API Layer ----------------------------------------------------------

def basescan_get(params: dict, api_key: str = "") -> dict:
    """Make a Blockscout/Basescan API request with error handling."""
    if api_key:
        params["apikey"] = api_key
    try:
        resp = requests.get(BASESCAN_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [!] API error: {e}")
        return {}


def fetch_transactions(address: str, api_key: str = "") -> list[dict]:
    """Fetch outgoing transaction history from Basescan."""
    data = basescan_get({
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": 1000,
        "sort": "asc",
    }, api_key)

    if data.get("status") == "1" and isinstance(data.get("result"), list):
        return data["result"]
    return []


def get_eth_price(api_key: str = "") -> float:
    """Fetch current ETH/USD price."""
    # Try Blockscout stats endpoint
    data = basescan_get({"module": "stats", "action": "ethprice"}, api_key)
    try:
        return float(data["result"]["ethusd"])
    except (KeyError, TypeError, ValueError):
        pass
    # Fallback: CoinGecko free API
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "ethereum", "vs_currencies": "usd"},
            timeout=10,
        )
        return float(resp.json()["ethereum"]["usd"])
    except Exception:
        return ETH_PRICE_FALLBACK


def is_contract_address(address: str) -> bool:
    """Check if address is a contract (smart wallet) via Base RPC."""
    try:
        resp = requests.post(BASE_RPC_URL, json={
            "jsonrpc": "2.0",
            "method": "eth_getCode",
            "params": [address, "latest"],
            "id": 1,
        }, timeout=10)
        code = resp.json().get("result", "0x")
        return code != "0x" and len(code) > 2
    except requests.RequestException:
        return False


# -- Analysis Engine ----------------------------------------------------

def classify_failure(tx: dict) -> str:
    """Classify why a transaction failed."""
    gas_used = int(tx.get("gasUsed", 0))
    gas_limit = int(tx.get("gas", 0))

    if gas_limit > 0 and gas_used >= gas_limit * 0.95:
        return "out_of_gas"
    return "reverted"


def detect_nonce_issues(transactions: list[dict]) -> tuple[int, int]:
    """Detect nonce gaps and retry attempts."""
    nonces = [int(tx.get("nonce", 0)) for tx in transactions]
    if not nonces:
        return 0, 0

    # Gaps in nonce sequence
    gaps = 0
    unique_sorted = sorted(set(nonces))
    for i in range(1, len(unique_sorted)):
        gap = unique_sorted[i] - unique_sorted[i - 1] - 1
        if gap > 0:
            gaps += gap

    # Retries: same nonce used multiple times
    counts = Counter(nonces)
    retries = sum(c - 1 for c in counts.values() if c > 1)

    return gaps, retries


def compute_health_score(
    success_rate: float,
    successful_txns: list[dict],
    total_gas_wei: int,
    wasted_gas_wei: int,
    nonce_gaps: int,
    retries: int,
) -> float:
    """Compute composite health score 0-100."""
    # 1. Success rate (50%) - direct percentage
    sr = success_rate * 100

    # 2. Gas efficiency (25%) - how well gas limits match actual usage
    effs = []
    for tx in successful_txns:
        gl = int(tx.get("gas", 0))
        gu = int(tx.get("gasUsed", 0))
        if gl > 0:
            effs.append(gu / gl)

    if effs:
        avg = sum(effs) / len(effs)
        if 0.4 <= avg <= 0.85:
            ge = 100.0
        elif avg < 0.4:
            ge = (avg / 0.4) * 100
        else:
            ge = max(0.0, 100 - (avg - 0.85) * 500)
    else:
        ge = 0.0

    # 3. Cost efficiency (15%) - fraction of gas NOT wasted
    if total_gas_wei > 0:
        ce = (1 - wasted_gas_wei / total_gas_wei) * 100
    else:
        ce = 100.0

    # 4. Nonce health (10%) - penalize each issue
    nh = max(0.0, 100 - (nonce_gaps + retries) * 10)

    score = sr * W_SUCCESS_RATE + ge * W_GAS_EFFICIENCY + ce * W_COST_EFFICIENCY + nh * W_NONCE_HEALTH
    return round(min(score, 100.0), 1)


def analyze_address(address: str, transactions: list[dict], eth_price: float) -> AgentHealth:
    """Full health analysis for one agent address."""
    health = AgentHealth(address=address)

    if not transactions:
        return health

    # Only outgoing transactions (sent by this address)
    outgoing = [tx for tx in transactions if tx.get("from", "").lower() == address.lower()]
    if not outgoing:
        return health

    health.total_transactions = len(outgoing)

    # Time range
    first_ts = int(outgoing[0].get("timeStamp", 0))
    last_ts = int(outgoing[-1].get("timeStamp", 0))
    if first_ts:
        health.first_seen = datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    if last_ts:
        health.last_seen = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    total_gas_wei = 0
    wasted_gas_wei = 0
    gas_efficiencies = []
    successful_txns = []
    failure_counts = {"out_of_gas": 0, "reverted": 0}

    for tx in outgoing:
        gas_used = int(tx.get("gasUsed") or 0)
        gas_price = int(tx.get("gasPrice") or 0)
        gas_limit = int(tx.get("gas") or 0)
        is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"

        cost_wei = gas_used * gas_price
        total_gas_wei += cost_wei

        if gas_limit > 0:
            gas_efficiencies.append(gas_used / gas_limit)

        if is_error:
            health.failed += 1
            wasted_gas_wei += cost_wei
            ftype = classify_failure(tx)
            failure_counts[ftype] += 1
        else:
            health.successful += 1
            successful_txns.append(tx)

    # Metrics
    health.success_rate_pct = round(health.successful / health.total_transactions * 100, 2)
    health.total_gas_spent_eth = round(total_gas_wei / 1e18, 8)
    health.wasted_gas_eth = round(wasted_gas_wei / 1e18, 8)
    health.avg_gas_efficiency_pct = round(
        (sum(gas_efficiencies) / len(gas_efficiencies) * 100) if gas_efficiencies else 0, 2
    )

    # Failure breakdown
    health.out_of_gas_count = failure_counts["out_of_gas"]
    health.reverted_count = failure_counts["reverted"]
    if health.failed > 0:
        health.top_failure_type = max(failure_counts, key=failure_counts.get)

    # Nonce issues
    health.nonce_gap_count, health.retry_count = detect_nonce_issues(outgoing)

    # Monthly waste projection
    period_days = max((last_ts - first_ts) / 86400, 1) if last_ts > first_ts else 1
    daily_waste_eth = (wasted_gas_wei / 1e18) / period_days
    health.estimated_monthly_waste_usd = round(daily_waste_eth * 30 * eth_price, 2)

    # Health score
    success_rate = health.successful / health.total_transactions if health.total_transactions else 0
    health.health_score = compute_health_score(
        success_rate, successful_txns, total_gas_wei, wasted_gas_wei,
        health.nonce_gap_count, health.retry_count,
    )

    # Priority bucket
    if health.health_score < 50:
        health.optimization_priority = "CRITICAL"
    elif health.health_score < 70:
        health.optimization_priority = "HIGH"
    elif health.health_score < 85:
        health.optimization_priority = "MEDIUM"
    else:
        health.optimization_priority = "LOW"

    return health


# -- Output -------------------------------------------------------------

CSV_FIELDS = [
    "address", "is_contract", "health_score", "optimization_priority",
    "total_transactions", "successful", "failed", "success_rate_pct",
    "total_gas_spent_eth", "wasted_gas_eth", "estimated_monthly_waste_usd",
    "avg_gas_efficiency_pct", "out_of_gas_count", "reverted_count",
    "nonce_gap_count", "retry_count", "top_failure_type",
    "first_seen", "last_seen",
]


def generate_csv(results: list[AgentHealth], filename: str):
    """Write health report CSV sorted by score ascending (worst first)."""
    results.sort(key=lambda r: r.health_score)

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in CSV_FIELDS})

    print(f"\n[+] Report saved: {filename}")
    print(f"    {len(results)} agents analyzed")

    critical = sum(1 for r in results if r.optimization_priority == "CRITICAL")
    high = sum(1 for r in results if r.optimization_priority == "HIGH")
    if critical or high:
        print(f"    {critical} CRITICAL + {high} HIGH priority = outreach targets")


def print_summary(results: list[AgentHealth], eth_price: float):
    """Print console summary."""
    active = [r for r in results if r.total_transactions > 0]
    if not active:
        print("\nNo transaction data found for any address.")
        return

    total_waste_eth = sum(r.wasted_gas_eth for r in active)
    total_monthly_usd = sum(r.estimated_monthly_waste_usd for r in active)
    avg_score = sum(r.health_score for r in active) / len(active)
    total_failed = sum(r.failed for r in active)
    total_txns = sum(r.total_transactions for r in active)

    print(f"\n{'='*78}")
    print(f"  AGENT TRANSACTION HEALTH REPORT — Base L2")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*78}")
    print(f"\n  Agents analyzed:       {len(active)}")
    print(f"  Total transactions:    {total_txns:,}")
    print(f"  Total failures:        {total_failed:,} ({total_failed/total_txns*100:.1f}% failure rate)" if total_txns else "")
    print(f"  Average health score:  {avg_score:.1f} / 100")
    print(f"  Total gas wasted:      {total_waste_eth:.6f} ETH (${total_waste_eth * eth_price:.2f})")
    print(f"  Est. monthly waste:    ${total_monthly_usd:.2f} USD across all agents")

    # Worst agents table
    worst = sorted(active, key=lambda r: r.health_score)[:10]
    print(f"\n  {'-'*74}")
    print(f"  TOP AGENTS NEEDING OPTIMIZATION")
    print(f"  {'-'*74}")
    print(f"  {'Address':<44} {'Score':>5} {'Fail%':>6} {'Waste ETH':>12} {'Priority':<9}")
    print(f"  {'-'*74}")
    for r in worst:
        fail_pct = f"{100 - r.success_rate_pct:.1f}%"
        waste = f"{r.wasted_gas_eth:.6f}"
        addr = r.address
        print(f"  {addr:<44} {r.health_score:>5.1f} {fail_pct:>6} {waste:>12} {r.optimization_priority:<9}")
    print(f"  {'-'*74}")


# -- CLI ----------------------------------------------------------------

def load_addresses(args) -> list[str]:
    """Collect and deduplicate addresses from all input sources."""
    addresses = []

    # From positional args
    if args.addresses:
        addresses.extend(args.addresses)

    # From file
    if args.file:
        try:
            with open(args.file) as f:
                for line in f:
                    addr = line.strip()
                    if addr and not addr.startswith("#"):
                        addresses.append(addr)
        except FileNotFoundError:
            print(f"[!] File not found: {args.file}")
            sys.exit(1)

    # Validate format
    valid = []
    for addr in addresses:
        if addr.startswith("0x") and len(addr) == 42:
            valid.append(addr.lower())
        else:
            print(f"[!] Skipping invalid address: {addr}")

    # Deduplicate preserving order
    return list(dict.fromkeys(valid))


def main():
    parser = argparse.ArgumentParser(
        description="Agent Transaction Health Monitor — Base L2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor.py 0x1234...abcd 0x5678...efgh
  python monitor.py -f addresses.txt -k YOUR_BASESCAN_KEY
  python monitor.py -f addresses.txt -o report.csv

Get a free Basescan API key at https://basescan.org/apis
Without a key, requests are rate-limited to 1 every 5 seconds.
        """,
    )
    parser.add_argument("addresses", nargs="*", help="wallet addresses to analyze")
    parser.add_argument("-f", "--file", help="file with addresses, one per line")
    parser.add_argument("-k", "--api-key", default="", help="Basescan API key (free tier)")
    parser.add_argument("-o", "--output", default="agent_health_report.csv", help="output CSV path")
    parser.add_argument("--check-contracts", action="store_true",
                        help="check if addresses are contracts (smart wallets) via RPC")

    args = parser.parse_args()
    addresses = load_addresses(args)

    if not addresses:
        parser.print_help()
        print("\n[!] No valid addresses provided.")
        sys.exit(1)

    delay = DELAY_BETWEEN_REQUESTS

    print(f"[*] Agent Transaction Health Monitor")
    print(f"[*] Network: Base L2 (Mainnet)")
    print(f"[*] Addresses: {len(addresses)}")
    print(f"[*] API: Blockscout (free, {delay}s between requests)")

    # ETH price
    eth_price = get_eth_price(args.api_key)
    print(f"[*] ETH/USD: ${eth_price:,.2f}")

    # Analyze each address
    results = []
    for i, addr in enumerate(addresses, 1):
        print(f"\n[{i}/{len(addresses)}] {addr}")

        # Optional contract check
        if args.check_contracts:
            is_contract = is_contract_address(addr)
            if is_contract:
                print(f"  Type: Smart contract (smart wallet / agent contract)")
            else:
                print(f"  Type: EOA (externally owned account)")
        else:
            is_contract = False

        # Fetch transactions
        txns = fetch_transactions(addr, api_key=args.api_key)
        print(f"  Transactions found: {len(txns)}")

        # Analyze
        health = analyze_address(addr, txns, eth_price)
        health.is_contract = is_contract
        results.append(health)

        if health.total_transactions > 0:
            status = (
                f"  Score: {health.health_score}/100 [{health.optimization_priority}] | "
                f"Success: {health.success_rate_pct}% | "
                f"Failed: {health.failed} | "
                f"Wasted: {health.wasted_gas_eth:.6f} ETH"
            )
            print(status)
        else:
            print(f"  No outgoing transactions found")

        # Rate limit
        if i < len(addresses):
            time.sleep(delay)

    # Report
    print_summary(results, eth_price)
    generate_csv(results, args.output)

    # Outreach summary
    targets = [r for r in results if r.optimization_priority in ("CRITICAL", "HIGH")]
    if targets:
        print(f"\n[*] {len(targets)} outreach targets identified in {args.output}")
        print(f"[*] Filter CSV by optimization_priority = CRITICAL or HIGH")


if __name__ == "__main__":
    main()
