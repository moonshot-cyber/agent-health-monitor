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
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
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


def classify_retry_failure(tx: dict, all_txs: list[dict]) -> str:
    """Detailed failure classification for retry analysis."""
    gas_used = int(tx.get("gasUsed", 0))
    gas_limit = int(tx.get("gas", 0))
    nonce = int(tx.get("nonce", 0))

    # Out of gas: used >= 95% of limit
    if gas_limit > 0 and gas_used >= gas_limit * 0.95:
        return "out_of_gas"

    # Nonce conflict: another tx with the same nonce succeeded
    same_nonce = [
        t for t in all_txs
        if int(t.get("nonce", -1)) == nonce
        and t.get("hash") != tx.get("hash")
        and t.get("isError") != "1"
        and t.get("txreceipt_status") != "0"
    ]
    if same_nonce:
        return "nonce_conflict"

    # Slippage: common DEX method selectors with low gas usage (reverted early)
    inp = tx.get("input", "0x")
    selector = inp[:10] if len(inp) >= 10 else "0x"
    dex_selectors = {
        "0x5ae401dc", "0x3593564c", "0x38ed1739", "0x7ff36ab5",
        "0x18cbafe5", "0xb6f9de95", "0x791ac947",
    }
    if selector in dex_selectors and gas_limit > 0 and gas_used < gas_limit * 0.5:
        return "slippage"

    return "reverted"


def is_retryable(failure_reason: str, tx: dict, all_txs: list[dict]) -> bool:
    """Determine if a failed transaction is worth retrying."""
    # Nonce conflicts: another tx already succeeded with that nonce
    if failure_reason == "nonce_conflict":
        return False

    # Pure reverts with very low gas usage are likely contract-level logic failures
    # that would fail again (e.g., insufficient balance, unauthorized)
    if failure_reason == "reverted":
        gas_used = int(tx.get("gasUsed", 0))
        gas_limit = int(tx.get("gas", 0))
        # If reverted using < 10% of gas, it's likely a require() check
        # that would fail again with the same calldata
        if gas_limit > 0 and gas_used < gas_limit * 0.10:
            return False

    # out_of_gas and slippage are retryable with corrected parameters
    return True


def get_current_gas_params() -> dict:
    """Fetch current base fee and estimate priority fee from Base RPC."""
    result = {"base_fee_gwei": 0.1, "priority_fee_gwei": 0.05}
    try:
        # Get latest block for base fee
        resp = requests.post(BASE_RPC_URL, json={
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": ["latest", False],
            "id": 1,
        }, timeout=10)
        block = resp.json().get("result", {})
        base_fee_hex = block.get("baseFeePerGas", "0x0")
        base_fee_wei = int(base_fee_hex, 16)
        result["base_fee_gwei"] = base_fee_wei / 1e9

        # Priority fee estimate
        resp2 = requests.post(BASE_RPC_URL, json={
            "jsonrpc": "2.0",
            "method": "eth_maxPriorityFeePerGas",
            "params": [],
            "id": 2,
        }, timeout=10)
        prio_hex = resp2.json().get("result", "0x0")
        prio_wei = int(prio_hex, 16)
        result["priority_fee_gwei"] = prio_wei / 1e9
    except (requests.RequestException, ValueError, KeyError):
        pass
    return result


@dataclass
class RetryTransaction:
    """An optimized retry transaction ready for signing."""
    original_tx_hash: str
    failure_reason: str
    optimized_transaction: dict  # to, data, value, gas_limit, max_fee_per_gas, max_priority_fee_per_gas
    estimated_gas_cost_usd: float = 0.0
    confidence: str = "medium"  # "high", "medium", "low"


@dataclass
class RetryAnalysis:
    """Full retry analysis for a wallet."""
    address: str
    failed_transactions_analyzed: int = 0
    retryable_count: int = 0
    retry_transactions: list[RetryTransaction] = field(default_factory=list)
    total_estimated_retry_cost_usd: float = 0.0
    potential_value_recovered_usd: float = 0.0


def analyze_retryable_transactions(
    address: str, transactions: list[dict], eth_price: float,
) -> RetryAnalysis:
    """Analyze failed transactions and build optimized retry transactions."""
    result = RetryAnalysis(address=address)

    if not transactions:
        return result

    outgoing = [
        tx for tx in transactions if tx.get("from", "").lower() == address.lower()
    ]
    if not outgoing:
        return result

    # Identify failed transactions
    failed_txs = [
        tx for tx in outgoing
        if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"
    ]
    result.failed_transactions_analyzed = len(failed_txs)

    if not failed_txs:
        return result

    # Get current gas parameters from the network
    gas_params = get_current_gas_params()
    base_fee_wei = int(gas_params["base_fee_gwei"] * 1e9)
    priority_fee_wei = int(gas_params["priority_fee_gwei"] * 1e9)
    # max_fee = 2x base fee + priority fee (EIP-1559 best practice)
    max_fee_wei = base_fee_wei * 2 + priority_fee_wei

    # Build gas optimization data for p95 gas limits per tx type
    optimization = optimize_gas(address, transactions, eth_price)
    # Map contract:method -> optimal gas limit from GasOptimizer
    optimal_gas_map: dict[str, int] = {}
    for tx_type in optimization.tx_types:
        key = f"{tx_type.contract}:{tx_type.method_id}"
        optimal_gas_map[key] = tx_type.optimal_gas_limit

    # Compute total value of failed txs (ETH value + gas cost paid)
    total_value_wei = 0
    total_retry_cost = 0.0

    for tx in failed_txs:
        reason = classify_retry_failure(tx, outgoing)

        if not is_retryable(reason, tx, outgoing):
            continue

        # Build the optimized replacement transaction
        to_addr = tx.get("to") or ""
        inp = tx.get("input", "0x")
        value_hex = tx.get("value", "0")
        value_wei = int(value_hex)

        # Determine optimal gas limit
        selector = inp[:10] if len(inp) >= 10 else "0x"
        type_key = f"{to_addr.lower()}:{selector}"
        if type_key in optimal_gas_map and optimal_gas_map[type_key] > 0:
            # Use p95 * 1.2 safety margin (optimizer already uses 1.15, bump to 1.2)
            opt_limit = optimal_gas_map[type_key]
            gas_limit = int(opt_limit * (1.2 / 1.15))
        else:
            # Fallback: original gas limit * 1.2 for out_of_gas, or same for others
            original_limit = int(tx.get("gas", 0))
            if reason == "out_of_gas":
                gas_limit = int(original_limit * 1.5)
            else:
                gas_limit = int(original_limit * 1.2) if original_limit > 0 else 200000

        # Estimate cost
        estimated_gas_cost_wei = gas_limit * max_fee_wei
        estimated_gas_cost_usd = round((estimated_gas_cost_wei / 1e18) * eth_price, 4)
        total_retry_cost += estimated_gas_cost_usd

        # Track value that could be recovered
        total_value_wei += value_wei
        # Also count the gas already wasted on the failed tx
        wasted_gas_wei = int(tx.get("gasUsed", 0)) * int(tx.get("gasPrice", 0))
        total_value_wei += wasted_gas_wei

        # Confidence level
        if reason == "out_of_gas":
            confidence = "high"
        elif reason == "slippage":
            confidence = "medium"
        else:
            confidence = "low"

        retry_tx = RetryTransaction(
            original_tx_hash=tx.get("hash", ""),
            failure_reason=reason,
            optimized_transaction={
                "to": to_addr,
                "data": inp,
                "value": hex(value_wei),
                "gas_limit": hex(gas_limit),
                "max_fee_per_gas": hex(max_fee_wei),
                "max_priority_fee_per_gas": hex(priority_fee_wei),
            },
            estimated_gas_cost_usd=estimated_gas_cost_usd,
            confidence=confidence,
        )
        result.retry_transactions.append(retry_tx)

    result.retryable_count = len(result.retry_transactions)
    result.total_estimated_retry_cost_usd = round(total_retry_cost, 2)
    result.potential_value_recovered_usd = round(
        (total_value_wei / 1e18) * eth_price, 2
    )

    return result


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


# -- Gas Optimization Engine --------------------------------------------

KNOWN_SELECTORS = {
    "0x5ae401dc": "multicall (Uniswap V3)",
    "0x3593564c": "execute (Universal Router)",
    "0xa9059cbb": "transfer (ERC-20)",
    "0x095ea7b3": "approve (ERC-20)",
    "0x38ed1739": "swapExactTokensForTokens",
    "0x7ff36ab5": "swapExactETHForTokens",
    "0x18cbafe5": "swapExactTokensForETH",
    "0x23b872dd": "transferFrom (ERC-20)",
    "0xb6f9de95": "swapExactETHForTokensSupportingFee",
    "0x791ac947": "swapExactTokensForETHSupportingFee",
    "0x": "ETH transfer",
}


def _percentile(values: list[int], p: float) -> int:
    """Compute the p-th percentile of a list of integers."""
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return int(s[f] * (c - k) + s[c] * (k - f))


def _method_label(selector: str) -> str:
    """Resolve a 4-byte method selector to a human label."""
    return KNOWN_SELECTORS.get(selector, selector)


@dataclass
class TxTypeOptimization:
    """Gas optimization for one transaction type (contract + method)."""
    contract: str
    method_id: str
    method_label: str
    tx_count: int = 0
    failed_count: int = 0
    failure_rate_pct: float = 0.0
    current_avg_gas_limit: int = 0
    current_p50_gas_used: int = 0
    current_p95_gas_used: int = 0
    optimal_gas_limit: int = 0
    gas_limit_reduction_pct: float = 0.0
    wasted_gas_eth: float = 0.0
    wasted_gas_usd: float = 0.0


@dataclass
class GasOptimization:
    """Full gas optimization report for a wallet."""
    address: str
    total_transactions_analyzed: int = 0
    current_monthly_gas_usd: float = 0.0
    optimized_monthly_gas_usd: float = 0.0
    estimated_monthly_savings_usd: float = 0.0
    total_wasted_gas_eth: float = 0.0
    total_wasted_gas_usd: float = 0.0
    tx_types: list[TxTypeOptimization] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def optimize_gas(
    address: str, transactions: list[dict], eth_price: float,
) -> GasOptimization:
    """Analyze transactions and generate per-type gas optimization report."""
    result = GasOptimization(address=address)

    if not transactions:
        return result

    outgoing = [
        tx for tx in transactions if tx.get("from", "").lower() == address.lower()
    ]
    if not outgoing:
        return result

    result.total_transactions_analyzed = len(outgoing)

    # Group by contract:methodId
    groups: dict[str, list[dict]] = {}
    for tx in outgoing:
        to_addr = (tx.get("to") or "0x0000000000000000000000000000000000000000").lower()
        inp = tx.get("input", "0x")
        selector = inp[:10] if len(inp) >= 10 else "0x"
        key = f"{to_addr}:{selector}"
        groups.setdefault(key, []).append(tx)

    # Time range for monthly projection
    timestamps = [int(tx.get("timeStamp", 0)) for tx in outgoing]
    first_ts, last_ts = min(timestamps), max(timestamps)
    period_days = max((last_ts - first_ts) / 86400, 1)

    total_gas_cost_wei = 0
    total_wasted_wei = 0
    total_optimized_cost_wei = 0

    type_results = []

    for key, txs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        to_addr, selector = key.rsplit(":", 1)
        label = _method_label(selector)

        gas_limits = []
        gas_used_list = []
        failed = 0
        group_wasted_wei = 0
        group_cost_wei = 0

        for tx in txs:
            g_used = int(tx.get("gasUsed") or 0)
            g_limit = int(tx.get("gas") or 0)
            g_price = int(tx.get("gasPrice") or 0)
            is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"

            cost_wei = g_used * g_price
            group_cost_wei += cost_wei
            total_gas_cost_wei += cost_wei

            if g_limit > 0:
                gas_limits.append(g_limit)
            if g_used > 0:
                gas_used_list.append(g_used)

            if is_error:
                failed += 1
                group_wasted_wei += cost_wei
                total_wasted_wei += cost_wei

        avg_limit = int(sum(gas_limits) / len(gas_limits)) if gas_limits else 0
        p50_used = _percentile(gas_used_list, 50)
        p95_used = _percentile(gas_used_list, 95)
        optimal = int(p95_used * 1.15) if p95_used > 0 else avg_limit

        # Project optimized cost: for successful txs, cost stays same (gas_used * price).
        # Savings come from eliminating failed txs via better limits + simulation.
        total_optimized_cost_wei += (group_cost_wei - group_wasted_wei)

        reduction_pct = (
            round((1 - optimal / avg_limit) * 100, 1) if avg_limit > 0 and optimal < avg_limit else 0.0
        )

        opt = TxTypeOptimization(
            contract=to_addr,
            method_id=selector,
            method_label=label,
            tx_count=len(txs),
            failed_count=failed,
            failure_rate_pct=round(failed / len(txs) * 100, 1) if txs else 0.0,
            current_avg_gas_limit=avg_limit,
            current_p50_gas_used=p50_used,
            current_p95_gas_used=p95_used,
            optimal_gas_limit=optimal,
            gas_limit_reduction_pct=reduction_pct,
            wasted_gas_eth=round(group_wasted_wei / 1e18, 8),
            wasted_gas_usd=round(group_wasted_wei / 1e18 * eth_price, 2),
        )
        type_results.append(opt)

    # Monthly projections
    daily_total = (total_gas_cost_wei / 1e18) / period_days
    daily_optimized = (total_optimized_cost_wei / 1e18) / period_days

    result.current_monthly_gas_usd = round(daily_total * 30 * eth_price, 2)
    result.optimized_monthly_gas_usd = round(daily_optimized * 30 * eth_price, 2)
    result.estimated_monthly_savings_usd = round(
        result.current_monthly_gas_usd - result.optimized_monthly_gas_usd, 2
    )
    result.total_wasted_gas_eth = round(total_wasted_wei / 1e18, 8)
    result.total_wasted_gas_usd = round(total_wasted_wei / 1e18 * eth_price, 2)
    result.tx_types = type_results

    # Generate recommendations
    recs = []
    high_fail_types = [t for t in type_results if t.failure_rate_pct > 10 and t.tx_count >= 3]
    if high_fail_types:
        worst = max(high_fail_types, key=lambda t: t.failure_rate_pct)
        recs.append(
            f"High failure rate ({worst.failure_rate_pct}%) on {worst.method_label} "
            f"to {worst.contract[:10]}...{worst.contract[-4:]}. "
            f"Add eth_call simulation before submitting these transactions."
        )

    over_limit = [t for t in type_results if t.gas_limit_reduction_pct > 30 and t.tx_count >= 3]
    if over_limit:
        worst = max(over_limit, key=lambda t: t.gas_limit_reduction_pct)
        recs.append(
            f"Gas limits are {worst.gas_limit_reduction_pct}% too high for {worst.method_label}. "
            f"Current avg: {worst.current_avg_gas_limit:,}, optimal: {worst.optimal_gas_limit:,}. "
            f"Use eth_estimateGas with a 15% buffer."
        )

    if result.estimated_monthly_savings_usd > 0:
        recs.append(
            f"Eliminating failed transactions would save ~${result.estimated_monthly_savings_usd:.2f}/month "
            f"({result.total_wasted_gas_eth:.6f} ETH wasted so far)."
        )

    tight_limit = [
        t for t in type_results
        if t.current_avg_gas_limit > 0
        and t.current_p95_gas_used > t.current_avg_gas_limit * 0.9
        and t.tx_count >= 3
    ]
    if tight_limit:
        recs.append(
            f"{len(tight_limit)} transaction type(s) have dangerously tight gas limits "
            f"(p95 usage >90% of limit). Increase buffer to avoid out-of-gas failures."
        )

    if not recs:
        recs.append(
            "Gas usage looks well-optimized. Continue monitoring for regressions."
        )

    result.recommendations = recs
    return result


# -- Protection Agent Orchestrator --------------------------------------

@dataclass
class RecommendedAction:
    """A prioritized action from the protection agent."""
    priority: int
    action: str
    description: str
    potential_value_usd: float = 0.0
    potential_savings_monthly_usd: float = 0.0


@dataclass
class ProtectionSummary:
    """Summary of the protection agent analysis."""
    total_issues_found: int = 0
    total_potential_savings_usd: float = 0.0
    retry_transactions_ready: int = 0
    estimated_retry_cost_usd: float = 0.0


@dataclass
class ProtectionResult:
    """Full result from the protection agent orchestrator."""
    address: str
    risk_level: str = "low"
    health_score: float = 0.0
    services_run: list[str] = field(default_factory=list)
    summary: ProtectionSummary = field(default_factory=ProtectionSummary)
    health: AgentHealth = field(default_factory=lambda: AgentHealth(address=""))
    gas_optimization: GasOptimization | None = None
    retry_analysis: RetryAnalysis | None = None
    recommended_actions: list[RecommendedAction] = field(default_factory=list)


def run_protection_agent(
    address: str, transactions: list[dict], eth_price: float,
) -> ProtectionResult:
    """
    Autonomous protection agent orchestrator.

    Triages based on health score and runs the appropriate services:
      90-100: Low risk — health report only
      70-89:  Medium risk — health + gas optimization
      50-69:  High risk — health + gas optimization + retry bot
      0-49:   Critical — all services, urgent flagging
    """
    result = ProtectionResult(address=address)

    # Step 1: Always run health analysis
    health = analyze_address(address, transactions, eth_price)
    result.health = health
    result.health_score = health.health_score
    result.services_run.append("health_check")

    # Determine risk level from health score
    score = health.health_score
    if score >= 90:
        result.risk_level = "low"
    elif score >= 70:
        result.risk_level = "medium"
    elif score >= 50:
        result.risk_level = "high"
    else:
        result.risk_level = "critical"

    total_savings = 0.0
    total_issues = health.failed + health.nonce_gap_count + health.retry_count
    actions: list[RecommendedAction] = []

    # Step 2: Run GasOptimizer for medium risk and above (score < 90)
    gas_opt = None
    if score < 90:
        gas_opt = optimize_gas(address, transactions, eth_price)
        result.gas_optimization = gas_opt
        result.services_run.append("gas_optimizer")

        if gas_opt.estimated_monthly_savings_usd > 0:
            over_limit_types = [
                t for t in gas_opt.tx_types
                if t.gas_limit_reduction_pct > 10 and t.tx_count >= 3
            ]
            total_savings += gas_opt.estimated_monthly_savings_usd
            actions.append(RecommendedAction(
                priority=0,  # ranked later
                action="Apply gas limit optimizations",
                description=(
                    f"Reduce gas limits on {len(over_limit_types)} transaction type(s). "
                    f"Current monthly spend: ${gas_opt.current_monthly_gas_usd:.2f}, "
                    f"optimized: ${gas_opt.optimized_monthly_gas_usd:.2f}."
                ),
                potential_savings_monthly_usd=gas_opt.estimated_monthly_savings_usd,
            ))

    # Step 3: Run RetryBot for high risk and above (score < 70)
    retry = None
    if score < 70:
        retry = analyze_retryable_transactions(address, transactions, eth_price)
        result.retry_analysis = retry
        result.services_run.append("retry_bot")

        if retry.retryable_count > 0:
            total_savings += retry.potential_value_recovered_usd
            actions.append(RecommendedAction(
                priority=0,
                action="Execute retry transactions",
                description=(
                    f"{retry.retryable_count} failed transactions can be retried "
                    f"with optimized gas parameters."
                ),
                potential_value_usd=retry.potential_value_recovered_usd,
            ))

    # Step 4: Always recommend monitoring for anything below low risk
    if score < 90:
        actions.append(RecommendedAction(
            priority=0,
            action="Set up monitoring alerts",
            description=(
                f"Subscribe to automated health monitoring at /alerts/subscribe/{address} "
                f"to get webhook alerts when health score drops below thresholds."
            ),
        ))

    # Step 5: Add urgent flag for critical wallets
    if result.risk_level == "critical":
        actions.insert(0, RecommendedAction(
            priority=0,
            action="URGENT: Investigate transaction failures immediately",
            description=(
                f"Health score is {score}/100 (CRITICAL). "
                f"{health.failed} failed transactions, "
                f"{health.out_of_gas_count} out-of-gas, "
                f"{health.reverted_count} reverted. "
                f"Estimated ${health.estimated_monthly_waste_usd:.2f}/month wasted."
            ),
            potential_value_usd=health.estimated_monthly_waste_usd,
        ))

    # Rank actions by potential value (highest first)
    actions.sort(
        key=lambda a: max(a.potential_value_usd, a.potential_savings_monthly_usd),
        reverse=True,
    )
    for i, action in enumerate(actions, 1):
        action.priority = i
    result.recommended_actions = actions

    # Build summary
    result.summary = ProtectionSummary(
        total_issues_found=total_issues,
        total_potential_savings_usd=round(total_savings, 2),
        retry_transactions_ready=retry.retryable_count if retry else 0,
        estimated_retry_cost_usd=retry.total_estimated_retry_cost_usd if retry else 0.0,
    )

    return result


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
