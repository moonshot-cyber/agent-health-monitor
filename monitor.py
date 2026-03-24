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
import re
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


def fetch_token_transfers(address: str, max_pages: int = 5) -> list[dict]:
    """Fetch ERC-20 token transfer history via Blockscout V2 API.

    Used as a D2 fallback for smart contract wallets that have zero
    regular transactions but active token transfer activity (e.g. ACP agents).

    Returns list of transfer dicts normalized to etherscan-compat format:
    {from, to, timeStamp, contractAddress, tokenName, hash, value, ...}
    """
    transfers: list[dict] = []
    url = f"{BLOCKSCOUT_V2_URL}/addresses/{address}/token-transfers"
    params: dict = {"type": "ERC-20"}

    for _ in range(max_pages):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            break

        items = data.get("items", [])
        for item in items:
            # Normalize V2 nested format to flat dict for D2 sub-signals
            ts_str = item.get("timestamp", "")
            unix_ts = 0
            if ts_str:
                try:
                    from datetime import datetime as _dt
                    unix_ts = int(_dt.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
                except (ValueError, TypeError):
                    pass
            transfers.append({
                "from": (item.get("from") or {}).get("hash", ""),
                "to": (item.get("to") or {}).get("hash", ""),
                "timeStamp": str(unix_ts),
                "contractAddress": (item.get("token") or {}).get("address_hash", ""),
                "tokenName": (item.get("token") or {}).get("name", ""),
                "hash": item.get("transaction_hash", ""),
                "value": (item.get("total") or {}).get("value", "0"),
            })

        next_page = data.get("next_page_params")
        if not next_page:
            break
        params = {"type": "ERC-20", **next_page}

    return transfers


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


# -- Agent Wash Analysis Engine -------------------------------------------

BLOCKSCOUT_V2_URL = "https://base.blockscout.com/api/v2"


def fetch_tokens_v2(address: str, max_pages: int = 10) -> list[dict]:
    """Fetch ERC-20 token holdings via Blockscout V2 API with pagination.

    Returns list of token items, each containing 'token' (metadata) and 'value' (balance).
    The V2 API returns 50 items per page sorted by fiat value descending.
    """
    tokens: list[dict] = []
    url = f"{BLOCKSCOUT_V2_URL}/addresses/{address}/tokens"
    params = {"type": "ERC-20"}

    for _ in range(max_pages):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            break

        items = data.get("items", [])
        tokens.extend(items)

        # Blockscout V2 pagination uses next_page_params
        next_page = data.get("next_page_params")
        if not next_page:
            break
        params = {"type": "ERC-20", **next_page}

    return tokens


_URL_PATTERN = re.compile(r"(https?://|\.com|\.io|\.us|\.xyz|t\.ly|claim|visit)", re.IGNORECASE)


def is_spam_token(token: dict) -> tuple[bool, str | None]:
    """Determine if a token is spam using composite heuristics.

    Args:
        token: A token dict from Blockscout V2 API (the 'token' sub-object).

    Returns:
        (is_spam, reason) — True + reason string if spam, False + None otherwise.
    """
    name = (token.get("name") or "").lower()
    symbol = (token.get("symbol") or "").lower()

    # Strong signals — any one is enough
    if _URL_PATTERN.search(name) or _URL_PATTERN.search(symbol):
        return True, "URL in token name"

    if len(token.get("name") or "") > 50:
        return True, "Suspiciously long token name"

    # Medium signals — need 2+ to flag
    spam_signals = 0
    reasons: list[str] = []

    holders = token.get("holders_count") or 0
    if isinstance(holders, str):
        holders = int(holders) if holders.isdigit() else 0
    if holders < 100:
        spam_signals += 1
        reasons.append(f"Low holder count ({holders})")

    volume = float(token.get("volume_24h") or 0)
    if volume == 0:
        spam_signals += 1
        reasons.append("Zero 24h volume")

    exchange_rate = token.get("exchange_rate")
    if exchange_rate is None:
        spam_signals += 1
        reasons.append("No exchange rate (unlisted)")

    market_cap = float(token.get("circulating_market_cap") or 0)
    if market_cap == 0:
        spam_signals += 1
        reasons.append("Zero market cap")

    if spam_signals >= 2:
        return True, "; ".join(reasons)

    return False, None


@dataclass
class WashResult:
    """Full wash scan result for a wallet."""
    address: str
    cleanliness_score: int = 100
    cleanliness_grade: str = "Spotless"
    total_issues: int = 0
    issues_by_severity: dict = field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})
    dust_tokens: int = 0
    dust_total_usd: float = 0.0
    spam_tokens: int = 0
    spam_token_list: list = field(default_factory=list)
    gas_efficiency_pct: float = 0.0
    gas_efficiency_grade: str = "N/A"
    wasted_gas_usd: float = 0.0
    failed_tx_count_24hr: int = 0
    failed_tx_patterns: list = field(default_factory=list)
    nonce_gaps: int = 0
    issues: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    scan_timestamp: str = ""
    next_wash_recommended: str = "30 days"


def _cleanliness_grade(score: int) -> str:
    """Map cleanliness score to human-readable grade."""
    if score >= 90:
        return "Spotless"
    if score >= 70:
        return "Clean"
    if score >= 50:
        return "Needs Attention"
    if score >= 30:
        return "Dirty"
    return "Critical"


def _gas_efficiency_grade(pct: float) -> str:
    """Map gas efficiency percentage to grade."""
    if pct > 80:
        return "Excellent"
    if pct > 60:
        return "Good"
    if pct > 40:
        return "Fair"
    return "Poor"


def analyze_wash(
    address: str,
    tokens: list[dict],
    transactions: list[dict],
    eth_price: float,
) -> WashResult:
    """Run full wash hygiene analysis on a wallet.

    Args:
        address: Wallet address (lowercase).
        tokens: Token list from fetch_tokens_v2().
        transactions: Transaction list from fetch_transactions().
        eth_price: Current ETH/USD price.

    Returns:
        WashResult with cleanliness score, issues, and recommendations.
    """
    result = WashResult(
        address=address,
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    issues: list[dict] = []

    # ── A) Dust Detection ─────────────────────────────────────────────────
    dust_count = 0
    dust_total_usd = 0.0
    dust_names: list[str] = []

    for item in tokens:
        tok = item.get("token", {})
        raw_value = item.get("value", "0")
        decimals = int(tok.get("decimals") or 18)
        exchange_rate_str = tok.get("exchange_rate")

        if exchange_rate_str is None:
            continue  # no price data — skip for dust (handled in spam)

        try:
            exchange_rate = float(exchange_rate_str)
        except (ValueError, TypeError):
            continue

        try:
            balance = int(raw_value) / (10 ** decimals)
        except (ValueError, TypeError, OverflowError):
            continue

        usd_value = balance * exchange_rate
        if usd_value < 0.01 and usd_value >= 0:
            dust_count += 1
            dust_total_usd += usd_value
            dust_names.append(tok.get("name") or tok.get("symbol") or "Unknown")

    result.dust_tokens = dust_count
    result.dust_total_usd = round(dust_total_usd, 4)

    if dust_count > 0:
        severity = "high" if dust_count > 15 else "medium" if dust_count > 5 else "low"
        issues.append({
            "category": "dust",
            "severity": severity,
            "description": f"{dust_count} dust tokens worth ${dust_total_usd:.4f} total cluttering wallet",
            "action": f"Clear {dust_count} dust tokens to declutter wallet",
            "estimated_savings": f"${dust_total_usd:.4f} recoverable" if dust_total_usd > 0.001 else None,
        })

    # ── B) Spam Token Detection ───────────────────────────────────────────
    spam_count = 0
    spam_list: list[dict] = []

    for item in tokens:
        tok = item.get("token", {})
        is_spam, reason = is_spam_token(tok)
        if is_spam:
            spam_count += 1
            spam_list.append({
                "name": tok.get("name") or "Unknown",
                "symbol": tok.get("symbol") or "???",
                "reason": reason,
            })

    result.spam_tokens = spam_count
    result.spam_token_list = spam_list

    if spam_count > 0:
        severity = "high" if spam_count > 10 else "medium" if spam_count > 3 else "low"
        issues.append({
            "category": "spam",
            "severity": severity,
            "description": f"{spam_count} spam tokens detected in wallet",
            "action": f"{spam_count} spam tokens detected — consider blocking future airdrops",
            "estimated_savings": None,
        })

    # ── C) Gas Efficiency Analysis ────────────────────────────────────────
    outgoing = [
        tx for tx in transactions
        if tx.get("from", "").lower() == address.lower()
    ]

    gas_efficiencies: list[float] = []
    wasted_gas_wei = 0
    over_limit_count = 0

    for tx in outgoing:
        gas_used = int(tx.get("gasUsed") or 0)
        gas_limit = int(tx.get("gas") or 0)
        gas_price = int(tx.get("gasPrice") or 0)
        is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"

        if gas_limit > 0:
            eff = gas_used / gas_limit * 100
            gas_efficiencies.append(eff)

            if gas_limit > 2 * gas_used and gas_used > 0:
                over_limit_count += 1

        if is_error:
            wasted_gas_wei += (gas_limit - gas_used) * gas_price

    avg_efficiency = sum(gas_efficiencies) / len(gas_efficiencies) if gas_efficiencies else 0.0
    result.gas_efficiency_pct = round(avg_efficiency, 1)
    result.gas_efficiency_grade = _gas_efficiency_grade(avg_efficiency)
    result.wasted_gas_usd = round(wasted_gas_wei / 1e18 * eth_price, 2)

    if avg_efficiency > 0 and avg_efficiency < 60:
        severity = "high" if avg_efficiency < 40 else "medium"
        issues.append({
            "category": "gas",
            "severity": severity,
            "description": f"Gas efficiency at {avg_efficiency:.1f}% — gas limits set too high",
            "action": f"Reduce gas limits by ~{int(100 - avg_efficiency)}% to save on transaction costs",
            "estimated_savings": f"~{int(100 - avg_efficiency)}% gas savings" if avg_efficiency < 60 else None,
        })

    if over_limit_count > 3:
        issues.append({
            "category": "gas",
            "severity": "medium",
            "description": f"{over_limit_count} transactions used gas limits >2x actual usage",
            "action": "Use eth_estimateGas with a 15% buffer instead of hardcoded limits",
            "estimated_savings": None,
        })

    if result.wasted_gas_usd > 0.01:
        issues.append({
            "category": "gas",
            "severity": "high" if result.wasted_gas_usd > 1.0 else "medium",
            "description": f"${result.wasted_gas_usd:.2f} wasted on failed transactions",
            "action": "Add pre-flight simulation (eth_call) to avoid paying for failures",
            "estimated_savings": f"${result.wasted_gas_usd:.2f} recoverable",
        })

    # ── D) Failed Transaction Patterns ────────────────────────────────────
    now_ts = datetime.now(timezone.utc).timestamp()
    twenty_four_hours_ago = now_ts - 86400

    failed_24hr: list[dict] = []
    for tx in outgoing:
        is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"
        if not is_error:
            continue
        try:
            tx_ts = int(tx.get("timeStamp", 0))
        except (ValueError, TypeError):
            continue
        if tx_ts >= twenty_four_hours_ago:
            failed_24hr.append(tx)

    result.failed_tx_count_24hr = len(failed_24hr)

    # Group failed txs by target contract
    contract_failures: dict[str, int] = {}
    for tx in failed_24hr:
        to_addr = (tx.get("to") or "").lower()
        if to_addr:
            contract_failures[to_addr] = contract_failures.get(to_addr, 0) + 1

    patterns: list[dict] = []
    for contract, count in contract_failures.items():
        if count >= 3:
            patterns.append({
                "contract": contract,
                "failure_count": count,
                "pattern": "Repeated failures",
            })

    # Method-specific failure rates
    method_stats: dict[str, dict] = {}  # method_id -> {total, failed}
    for tx in outgoing:
        inp = tx.get("input", "0x")
        method_id = inp[:10] if len(inp) >= 10 else "0x"
        if method_id not in method_stats:
            method_stats[method_id] = {"total": 0, "failed": 0}
        method_stats[method_id]["total"] += 1
        is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"
        if is_error:
            method_stats[method_id]["failed"] += 1

    for method_id, stats in method_stats.items():
        if stats["total"] >= 5:
            fail_rate = stats["failed"] / stats["total"] * 100
            if fail_rate > 20:
                label = KNOWN_SELECTORS.get(method_id, method_id)
                patterns.append({
                    "contract": f"method:{method_id}",
                    "failure_count": stats["failed"],
                    "pattern": f"Method {label} has {fail_rate:.0f}% failure rate",
                })

    # Retry storm detection: same to + input within 5 minute window
    retry_storms = 0
    for i, tx in enumerate(outgoing):
        if tx.get("isError") != "1" and tx.get("txreceipt_status") != "0":
            continue
        to_addr = (tx.get("to") or "").lower()
        inp = tx.get("input", "0x")
        try:
            tx_ts = int(tx.get("timeStamp", 0))
        except (ValueError, TypeError):
            continue
        for j in range(i + 1, min(i + 20, len(outgoing))):
            other = outgoing[j]
            try:
                other_ts = int(other.get("timeStamp", 0))
            except (ValueError, TypeError):
                continue
            if other_ts - tx_ts > 300:  # 5 minutes
                break
            if (other.get("to") or "").lower() == to_addr and other.get("input", "0x") == inp:
                retry_storms += 1
                break

    if retry_storms > 0:
        patterns.append({
            "contract": "multiple",
            "failure_count": retry_storms,
            "pattern": f"Retry storm: {retry_storms} duplicate submissions within 5 minutes",
        })

    result.failed_tx_patterns = patterns

    if len(failed_24hr) > 0:
        severity = "high" if len(failed_24hr) > 5 else "medium" if len(failed_24hr) > 2 else "low"
        issues.append({
            "category": "failed_tx",
            "severity": severity,
            "description": f"{len(failed_24hr)} failed transactions in the last 24 hours",
            "action": "Review failure reasons and add pre-flight checks",
            "estimated_savings": None,
        })

    for p in patterns:
        if p["pattern"] == "Repeated failures":
            issues.append({
                "category": "failed_tx",
                "severity": "high",
                "description": f"{p['failure_count']} repeated failures to contract {p['contract'][:10]}...{p['contract'][-4:]}",
                "action": f"Check integration status with contract {p['contract'][:10]}...{p['contract'][-4:]}",
                "estimated_savings": None,
            })

    # ── E) Nonce Gap Detection ────────────────────────────────────────────
    nonce_gaps, _ = detect_nonce_issues(outgoing)
    result.nonce_gaps = nonce_gaps

    if nonce_gaps > 0:
        severity = "high" if nonce_gaps > 3 else "medium"
        issues.append({
            "category": "nonce",
            "severity": severity,
            "description": f"{nonce_gaps} nonce gaps detected in transaction sequence",
            "action": "Implement proper nonce tracking with pending transaction awareness",
            "estimated_savings": None,
        })

    # ── Calculate Cleanliness Score ───────────────────────────────────────
    failed_tx_pct = (
        len([tx for tx in outgoing if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"])
        / len(outgoing) * 100
    ) if outgoing else 0

    score = 100
    score -= min(30, dust_count * 2)
    score -= min(20, spam_count * 3)
    score -= min(25, (100 - avg_efficiency) * 0.5) if avg_efficiency > 0 else 0
    score -= min(15, failed_tx_pct * 0.5)
    score -= min(10, nonce_gaps * 5)
    score = max(0, int(score))

    result.cleanliness_score = score
    result.cleanliness_grade = _cleanliness_grade(score)

    # ── Sort Issues by Severity ──────────────────────────────────────────
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

    result.issues = issues
    result.total_issues = len(issues)
    result.issues_by_severity = {
        "high": sum(1 for i in issues if i["severity"] == "high"),
        "medium": sum(1 for i in issues if i["severity"] == "medium"),
        "low": sum(1 for i in issues if i["severity"] == "low"),
    }

    # ── Generate Recommendations ─────────────────────────────────────────
    recs: list[str] = []

    # High severity issues first
    high_issues = [i for i in issues if i["severity"] == "high"]
    for hi in high_issues[:2]:
        recs.append(hi["action"])

    # Fill remaining slots from medium issues
    if len(recs) < 3:
        medium_issues = [i for i in issues if i["severity"] == "medium"]
        for mi in medium_issues[:3 - len(recs)]:
            recs.append(mi["action"])

    # Fill remaining from low
    if len(recs) < 3:
        low_issues = [i for i in issues if i["severity"] == "low"]
        for li in low_issues[:3 - len(recs)]:
            recs.append(li["action"])

    if not recs:
        if not outgoing:
            recs.append("New wallet — insufficient history for full assessment")
        else:
            recs.append("Looking good — schedule your next wash in 30 days")

    result.recommendations = recs[:3]

    # ── Next Wash Recommendation ─────────────────────────────────────────
    if score >= 70:
        result.next_wash_recommended = "30 days"
    elif score >= 50:
        result.next_wash_recommended = "14 days"
    else:
        result.next_wash_recommended = "7 days"

    return result


# -- Agent Health Score (AHS) Scoring Engine ---------------------------------

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AHSResult:
    """Full AHS scan result."""
    address: str
    agent_health_score: int = 50
    grade: str = "Fair"
    grade_label: str = "Degraded"
    confidence: str = "LOW"
    mode: str = "2D"
    d1_score: int = 50
    d2_score: int = 50
    d3_score: Optional[int] = None
    d1_weight: float = 0.30
    d2_weight: float = 0.70
    d3_weight: float = 0.0
    cdp_modifier: int = 0
    patterns_detected: list = field(default_factory=list)
    d1_top_factors: list = field(default_factory=list)
    d2_top_factors: list = field(default_factory=list)
    d3_top_factors: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    trend: Optional[str] = None
    temporal_score: Optional[int] = None
    model_version: str = "AHS-v1"
    scan_timestamp: str = ""
    next_scan_recommended: str = "14 days"
    tx_count: int = 0
    history_days: int = 0
    d2_data_source: str = "txlist"
    # Internal signal values (NOT exposed in API)
    _signals: dict = field(default_factory=dict)


def _ahs_grade(score: int) -> tuple[str, str]:
    """Map AHS score to grade and label."""
    if score >= 90:
        return "A", "Excellent"
    if score >= 75:
        return "B", "Good"
    if score >= 60:
        return "C", "Needs Attention"
    if score >= 40:
        return "D", "Degraded"
    if score >= 20:
        return "E", "Critical"
    return "F", "Failing"


def _ahs_confidence(tx_count: int, history_days: int, has_d3: bool, has_previous: bool) -> str:
    """Determine confidence level based on data availability."""
    level = 0
    if tx_count >= 100 and history_days >= 7:
        level = 2  # base HIGH
    elif tx_count >= 50 and history_days >= 3:
        level = 1  # base MEDIUM
    elif tx_count >= 10:
        level = 0  # base LOW
    else:
        return "INSUFFICIENT"

    if has_d3:
        level = min(level + 1, 2)
    if has_previous:
        level = min(level + 1, 2)

    return ["LOW", "MEDIUM", "HIGH"][level]


def _next_scan_recommendation(score: int, confidence: str) -> str:
    if score >= 80 and confidence == "HIGH":
        return "30 days"
    if score >= 60:
        return "14 days"
    return "7 days"


# -- D1: Wallet Hygiene Scoring --

def calculate_d1_score(
    dust_count: int,
    dust_total_usd: float,
    spam_count: int,
    avg_gas_efficiency: float,
    failed_pct_24h: float,
    nonce_gaps: int,
) -> tuple[int, list]:
    """Calculate Dimension 1 (Wallet Hygiene) score. Returns (score, top_factors)."""
    # Signal scores
    d1_dust = max(0, 100 - dust_count * 1.5)
    d1_dust_val = 100 if dust_total_usd < 0.10 else max(0, 100 - (dust_total_usd - 0.10) * 50)
    d1_spam = max(0, 100 - spam_count * 2)

    # Gas efficiency
    avg_eff = avg_gas_efficiency / 100.0 if avg_gas_efficiency > 1 else avg_gas_efficiency
    if 0.40 <= avg_eff <= 0.85:
        d1_gas_eff = 100
    elif avg_eff < 0.40:
        d1_gas_eff = (avg_eff / 0.40) * 100 if avg_eff > 0 else 0
    else:
        d1_gas_eff = max(0, 100 - (avg_eff - 0.85) * 500)

    d1_fail_rate = max(0, 100 - failed_pct_24h * 3)
    d1_nonce = max(0, 100 - nonce_gaps * 15)

    score = (
        d1_dust * 0.15
        + d1_dust_val * 0.05
        + d1_spam * 0.20
        + d1_gas_eff * 0.25
        + d1_fail_rate * 0.20
        + d1_nonce * 0.15
    )
    score = max(0, min(100, int(round(score))))

    # Top factors (only report significant detractors)
    factors = []
    if d1_dust < 70:
        factors.append(f"{dust_count} dust tokens cluttering wallet")
    if d1_spam < 70:
        factors.append(f"{spam_count} spam tokens detected")
    if d1_gas_eff < 70:
        factors.append(f"Gas efficiency suboptimal ({avg_gas_efficiency:.1f}%)")
    if d1_fail_rate < 70:
        factors.append(f"Failed transaction rate elevated ({failed_pct_24h:.1f}%)")
    if d1_nonce < 70:
        factors.append(f"{nonce_gaps} nonce gaps in sequence")
    if not factors:
        factors.append("Wallet hygiene is healthy")

    return score, factors[:3]


# -- D2: Behavioural Patterns Scoring --

def _calc_repeated_failure_score(transactions: list[dict]) -> tuple[int, int, bool]:
    """Repeated failure patterns. Returns (score, max_consecutive, has_recovery)."""
    groups: dict[str, list[dict]] = {}
    for tx in transactions:
        to_addr = (tx.get("to") or "").lower()
        inp = tx.get("input", "0x")
        method_id = inp[:10] if len(inp) >= 10 else "0x"
        key = f"{to_addr}:{method_id}"
        groups.setdefault(key, []).append(tx)

    max_consec = 0
    has_recovery = False

    for key, txs in groups.items():
        sorted_txs = sorted(txs, key=lambda t: int(t.get("timeStamp", 0)))
        consecutive = 0
        group_max = 0
        seen_failure = False

        for tx in sorted_txs:
            is_error = tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"
            if is_error:
                consecutive += 1
                seen_failure = True
                group_max = max(group_max, consecutive)
            else:
                if seen_failure and consecutive > 0:
                    has_recovery = True
                consecutive = 0

        max_consec = max(max_consec, group_max)

    if max_consec <= 2:
        score = 100
    elif max_consec <= 5:
        score = 80 - (max_consec - 2) * 10
    elif max_consec <= 10:
        score = 50 - (max_consec - 5) * 8
    else:
        score = 0

    if has_recovery and score < 80:
        score = min(80, score + 15)

    return max(0, score), max_consec, has_recovery


def _calc_gas_adaptation_score(transactions: list[dict]) -> tuple[int, float]:
    """Gas adaptation index. Returns (score, cv)."""
    now_ts = datetime.now(timezone.utc).timestamp()
    twenty_four_hours_ago = now_ts - 86400

    gas_prices = []
    for tx in transactions:
        try:
            tx_ts = int(tx.get("timeStamp", 0))
            if tx_ts >= twenty_four_hours_ago:
                gp = int(tx.get("gasPrice", 0))
                if gp > 0:
                    gas_prices.append(gp)
        except (ValueError, TypeError):
            continue

    if len(gas_prices) < 5:
        return 50, 0.0  # insufficient data

    mean_gp = statistics.mean(gas_prices)
    std_gp = statistics.stdev(gas_prices)
    cv = std_gp / mean_gp if mean_gp > 0 else 0

    if cv >= 0.15:
        score = 100
    elif cv >= 0.05:
        score = int(60 + (cv - 0.05) * 400)
    elif cv >= 0.01:
        score = int(30 + (cv - 0.01) * 750)
    else:
        score = int(max(0, cv * 3000))

    return max(0, min(100, score)), cv


def _calc_nonce_management_score(transactions: list[dict]) -> tuple[int, int]:
    """Nonce management quality. Returns (score, persistent_gaps)."""
    nonce_gaps, _ = detect_nonce_issues(transactions)

    # Check for persistent gaps (gaps in txs > 48h apart)
    nonces_with_ts = []
    for tx in transactions:
        try:
            nonces_with_ts.append((int(tx.get("nonce", 0)), int(tx.get("timeStamp", 0))))
        except (ValueError, TypeError):
            continue

    if not nonces_with_ts:
        return 100, 0

    nonces_with_ts.sort(key=lambda x: x[1])
    unique_sorted = sorted(set(n for n, _ in nonces_with_ts))

    persistent_gaps = 0
    gap_nonces = set()
    for i in range(1, len(unique_sorted)):
        for missing in range(unique_sorted[i - 1] + 1, unique_sorted[i]):
            gap_nonces.add(missing)

    if gap_nonces:
        # Find time span of gap: earliest tx after gap vs latest tx before gap
        nonce_to_ts = {}
        for n, ts in nonces_with_ts:
            nonce_to_ts[n] = ts

        for gap_n in gap_nonces:
            before_ts = max((ts for n, ts in nonces_with_ts if n < gap_n), default=0)
            after_ts = min((ts for n, ts in nonces_with_ts if n > gap_n), default=0)
            if before_ts > 0 and after_ts > 0 and (after_ts - before_ts) > 48 * 3600:
                persistent_gaps += 1

    if persistent_gaps == 0 and nonce_gaps <= 1:
        score = 100
    elif persistent_gaps == 0:
        score = max(60, 100 - nonce_gaps * 10)
    elif persistent_gaps <= 2:
        score = max(20, 60 - persistent_gaps * 20)
    else:
        score = 0

    return max(0, min(100, score)), persistent_gaps


def _calc_timing_regularity_score(transactions: list[dict]) -> tuple[int, float, float, int]:
    """Timing regularity. Returns (score, cv, gap_ratio, burst_count)."""
    timestamps = sorted(int(tx.get("timeStamp", 0)) for tx in transactions)
    timestamps = [t for t in timestamps if t > 0]

    if len(timestamps) < 6:
        return 50, 0.0, 1.0, 0  # insufficient data

    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    intervals = [i for i in intervals if i > 0]

    if not intervals:
        return 50, 0.0, 1.0, 0

    mean_int = statistics.mean(intervals)
    median_int = statistics.median(intervals)
    max_int = max(intervals)

    cv = statistics.stdev(intervals) / mean_int if mean_int > 0 else 0
    gap_ratio = max_int / median_int if median_int > 0 else 1.0
    burst_count = sum(1 for i in intervals if median_int > 0 and i < median_int / 10)

    if 0.3 <= cv <= 2.0:
        timing_base = 100
    elif cv < 0.3:
        timing_base = 75
    else:
        timing_base = max(0, int(100 - (cv - 2.0) * 15))

    if gap_ratio > 20:
        timing_base -= 30
    elif gap_ratio > 10:
        timing_base -= 15

    if burst_count > 5:
        timing_base -= 20

    return max(0, min(100, timing_base)), cv, gap_ratio, burst_count


def _calc_tx_diversity_score(transactions: list[dict]) -> tuple[int, float, int]:
    """Transaction diversity. Returns (score, diversity_ratio, unique_pairs)."""
    now_ts = datetime.now(timezone.utc).timestamp()
    seven_days_ago = now_ts - 7 * 86400

    recent_txs = []
    for tx in transactions:
        try:
            if int(tx.get("timeStamp", 0)) >= seven_days_ago:
                recent_txs.append(tx)
        except (ValueError, TypeError):
            continue

    if not recent_txs:
        recent_txs = transactions[-100:] if len(transactions) > 100 else transactions

    pairs = set()
    for tx in recent_txs:
        to_addr = (tx.get("to") or "").lower()
        inp = tx.get("input", "0x")
        method_id = inp[:10] if len(inp) >= 10 else "0x"
        if inp != "0x":
            pairs.add((to_addr, method_id))

    diversity_ratio = len(pairs) / len(recent_txs) if recent_txs else 0
    unique_pairs = len(pairs)

    if unique_pairs >= 10 or diversity_ratio >= 0.10:
        score = 100
    elif unique_pairs >= 5 or diversity_ratio >= 0.05:
        score = 75
    elif unique_pairs >= 2 or diversity_ratio >= 0.02:
        score = 50
    elif unique_pairs == 1:
        score = 25
    else:
        score = 0

    return score, diversity_ratio, unique_pairs


def _calc_token_transfer_diversity_score(transfers: list[dict]) -> tuple[int, float, int]:
    """Token transfer diversity. Adapted from _calc_tx_diversity_score.

    Uses (to, contractAddress) pairs instead of (to, methodId) since token
    transfers don't have an input field.
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    seven_days_ago = now_ts - 7 * 86400

    recent = []
    for tx in transfers:
        try:
            if int(tx.get("timeStamp", 0)) >= seven_days_ago:
                recent.append(tx)
        except (ValueError, TypeError):
            continue

    if not recent:
        recent = transfers[-100:] if len(transfers) > 100 else transfers

    pairs = set()
    for tx in recent:
        to_addr = (tx.get("to") or "").lower()
        token_addr = (tx.get("contractAddress") or tx.get("tokenAddress") or "").lower()
        if to_addr and token_addr:
            pairs.add((to_addr, token_addr))

    diversity_ratio = len(pairs) / len(recent) if recent else 0
    unique_pairs = len(pairs)

    if unique_pairs >= 10 or diversity_ratio >= 0.10:
        score = 100
    elif unique_pairs >= 5 or diversity_ratio >= 0.05:
        score = 75
    elif unique_pairs >= 2 or diversity_ratio >= 0.02:
        score = 50
    elif unique_pairs == 1:
        score = 25
    else:
        score = 0

    return score, diversity_ratio, unique_pairs


def _calc_retry_storm_score(transactions: list[dict]) -> tuple[int, int]:
    """Retry storm frequency. Returns (score, storm_events)."""
    groups: dict[str, list[int]] = {}
    for tx in transactions:
        to_addr = (tx.get("to") or "").lower()
        inp = tx.get("input", "0x")
        key = f"{to_addr}:{inp}"
        try:
            ts = int(tx.get("timeStamp", 0))
            groups.setdefault(key, []).append(ts)
        except (ValueError, TypeError):
            continue

    storm_events = 0
    for key, timestamps in groups.items():
        if len(timestamps) < 3:
            continue
        sorted_ts = sorted(timestamps)
        window_start = sorted_ts[0]
        window_count = 1
        for ts in sorted_ts[1:]:
            if ts - window_start <= 300:
                window_count += 1
            else:
                if window_count >= 3:
                    storm_events += 1
                window_start = ts
                window_count = 1
        if window_count >= 3:
            storm_events += 1

    if storm_events == 0:
        score = 100
    elif storm_events <= 2:
        score = 70 - storm_events * 10
    elif storm_events <= 5:
        score = 50 - (storm_events - 2) * 10
    else:
        score = max(0, 20 - (storm_events - 5) * 5)

    return score, storm_events


def _calc_contract_breadth_score(transactions: list[dict]) -> tuple[int, float, int]:
    """Contract interaction breadth. Returns (score, breadth_ratio, unique_contracts)."""
    now_ts = datetime.now(timezone.utc).timestamp()
    seven_days_ago = now_ts - 7 * 86400

    recent_txs = []
    for tx in transactions:
        try:
            if int(tx.get("timeStamp", 0)) >= seven_days_ago:
                recent_txs.append(tx)
        except (ValueError, TypeError):
            continue

    if not recent_txs:
        recent_txs = transactions[-100:] if len(transactions) > 100 else transactions

    unique_contracts = set()
    for tx in recent_txs:
        to_addr = (tx.get("to") or "").lower()
        if to_addr:
            unique_contracts.add(to_addr)

    breadth_ratio = len(unique_contracts) / len(recent_txs) if recent_txs else 0

    if breadth_ratio >= 0.15 or len(unique_contracts) >= 8:
        score = 100
    elif breadth_ratio >= 0.05 or len(unique_contracts) >= 4:
        score = 70
    elif len(unique_contracts) >= 2:
        score = 40
    else:
        score = 10

    return score, breadth_ratio, len(unique_contracts)


def _calc_activity_gap_score(transactions: list[dict]) -> tuple[int, float]:
    """Activity gap detection. Returns (score, gap_ratio)."""
    now_ts = datetime.now(timezone.utc).timestamp()
    seven_days_ago = now_ts - 7 * 86400

    recent = []
    for tx in transactions:
        try:
            ts = int(tx.get("timeStamp", 0))
            if ts >= seven_days_ago:
                recent.append(ts)
        except (ValueError, TypeError):
            continue

    if not recent:
        recent = sorted(int(tx.get("timeStamp", 0)) for tx in transactions[-50:])

    if len(recent) < 3:
        return 50, 1.0

    recent.sort()
    intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    intervals = [i for i in intervals if i > 0]

    if not intervals:
        return 50, 1.0

    max_gap = max(intervals)
    median_gap = statistics.median(intervals)
    gap_ratio = max_gap / median_gap if median_gap > 0 else 1.0

    if gap_ratio < 5:
        score = 100
    elif gap_ratio < 10:
        score = 70
    elif gap_ratio < 20:
        score = 40
    else:
        score = 10

    return score, gap_ratio


def calculate_d2_score(transactions: list[dict]) -> tuple[int, list, dict]:
    """Calculate Dimension 2 (Behavioural Patterns) score.

    Returns (score, top_factors, raw_signals dict).
    """
    if len(transactions) < 10:
        return 50, ["Insufficient transaction history for behavioural analysis"], {}

    rep_score, max_consec, has_recovery = _calc_repeated_failure_score(transactions)
    gas_adapt_score, gas_cv = _calc_gas_adaptation_score(transactions)
    nonce_score, persistent_gaps = _calc_nonce_management_score(transactions)
    timing_score, timing_cv, gap_ratio, burst_count = _calc_timing_regularity_score(transactions)
    diversity_score, diversity_ratio, unique_pairs = _calc_tx_diversity_score(transactions)
    storm_score, storm_events = _calc_retry_storm_score(transactions)
    breadth_score, breadth_ratio, unique_contracts = _calc_contract_breadth_score(transactions)
    gap_score, activity_gap_ratio = _calc_activity_gap_score(transactions)

    # Weighted average per design doc
    score = (
        rep_score * 0.20
        + gas_adapt_score * 0.15
        + nonce_score * 0.10
        + timing_score * 0.15
        + diversity_score * 0.10
        + storm_score * 0.15
        + breadth_score * 0.10
        + gap_score * 0.05
    )
    score = max(0, min(100, int(round(score))))

    # Top factors
    factors = []
    signal_scores = [
        (rep_score, f"Repeated failures: {max_consec} consecutive to same contract"),
        (gas_adapt_score, f"Gas adaptation index: {gas_cv:.3f} (CV)"),
        (nonce_score, f"Nonce management: {persistent_gaps} persistent gaps"),
        (timing_score, f"Timing regularity: CV={timing_cv:.2f}, gap ratio={gap_ratio:.1f}"),
        (diversity_score, f"Transaction diversity: {unique_pairs} unique pairs ({diversity_ratio:.3f})"),
        (storm_score, f"Retry storms: {storm_events} events detected"),
        (breadth_score, f"Contract breadth: {unique_contracts} unique contracts ({breadth_ratio:.3f})"),
        (gap_score, f"Activity gaps: ratio {activity_gap_ratio:.1f}"),
    ]

    # Report worst signals
    signal_scores.sort(key=lambda x: x[0])
    for s, desc in signal_scores[:3]:
        if s < 70:
            factors.append(desc)

    if not factors:
        factors.append("Behavioural patterns are healthy")

    raw_signals = {
        "repeated_failure_score": rep_score,
        "max_consecutive_failures": max_consec,
        "has_recovery": has_recovery,
        "gas_adaptation_score": gas_adapt_score,
        "gas_adaptation_cv": gas_cv,
        "nonce_management_score": nonce_score,
        "persistent_nonce_gaps": persistent_gaps,
        "timing_score": timing_score,
        "timing_cv": timing_cv,
        "gap_ratio": gap_ratio,
        "burst_count": burst_count,
        "tx_diversity_score": diversity_score,
        "tx_diversity_ratio": diversity_ratio,
        "unique_pairs": unique_pairs,
        "retry_storm_score": storm_score,
        "storm_events": storm_events,
        "contract_breadth_score": breadth_score,
        "breadth_ratio": breadth_ratio,
        "unique_contracts": unique_contracts,
        "activity_gap_score": gap_score,
        "activity_gap_ratio": activity_gap_ratio,
    }

    return score, factors[:3], raw_signals


def calculate_d2_score_from_transfers(transfers: list[dict]) -> tuple[int, list, dict]:
    """Calculate D2 (Behavioural Patterns) from token transfers.

    Fallback for smart contract wallets with no regular txlist data.
    Uses 4 of 8 signals (those that don't require nonce, gasPrice, input, isError).
    Weights redistributed proportionally from the 4 usable signals.
    """
    if len(transfers) < 10:
        return 50, ["Insufficient token transfer history for behavioural analysis"], {}

    timing_score, timing_cv, gap_ratio, burst_count = _calc_timing_regularity_score(transfers)
    breadth_score, breadth_ratio, unique_contracts = _calc_contract_breadth_score(transfers)
    gap_score, activity_gap_ratio = _calc_activity_gap_score(transfers)
    diversity_score, diversity_ratio, unique_pairs = _calc_token_transfer_diversity_score(transfers)

    # Original weights for these 4 signals sum to 0.40; normalize to 1.0
    score = (
        timing_score * 0.375
        + diversity_score * 0.250
        + breadth_score * 0.250
        + gap_score * 0.125
    )
    score = max(0, min(100, int(round(score))))

    factors = []
    signal_scores = [
        (timing_score, f"Timing regularity: CV={timing_cv:.2f}, gap ratio={gap_ratio:.1f}"),
        (diversity_score, f"Transfer diversity: {unique_pairs} unique pairs ({diversity_ratio:.3f})"),
        (breadth_score, f"Counterparty breadth: {unique_contracts} unique recipients ({breadth_ratio:.3f})"),
        (gap_score, f"Activity gaps: ratio {activity_gap_ratio:.1f}"),
    ]
    signal_scores.sort(key=lambda x: x[0])
    for s, desc in signal_scores[:3]:
        if s < 70:
            factors.append(desc)
    if not factors:
        factors.append("Token transfer behavioural patterns are healthy")
    factors.append("(scored from token transfers \u2014 4/8 signals used)")

    raw_signals = {
        "repeated_failure_score": None,
        "max_consecutive_failures": 0,
        "has_recovery": False,
        "gas_adaptation_score": None,
        "gas_adaptation_cv": 0.0,
        "nonce_management_score": None,
        "persistent_nonce_gaps": 0,
        "retry_storm_score": None,
        "storm_events": 0,
        "timing_score": timing_score,
        "timing_cv": timing_cv,
        "gap_ratio": gap_ratio,
        "burst_count": burst_count,
        "tx_diversity_score": diversity_score,
        "tx_diversity_ratio": diversity_ratio,
        "unique_pairs": unique_pairs,
        "contract_breadth_score": breadth_score,
        "breadth_ratio": breadth_ratio,
        "unique_contracts": unique_contracts,
        "activity_gap_score": gap_score,
        "activity_gap_ratio": activity_gap_ratio,
        "d2_data_source": "tokentx",
        "d2_signals_used": 4,
    }

    return score, factors[:4], raw_signals


# -- D3: Infrastructure Health Scoring (sync probing) --

def probe_infrastructure_sync(agent_url: str) -> tuple[int, list]:
    """Probe agent infrastructure (synchronous). Returns (score, top_factors)."""
    import time as _time

    scores = {}
    factors = []

    # Probe 1: Availability
    try:
        start = _time.monotonic()
        resp = requests.get(agent_url, timeout=10, allow_redirects=True)
        latency_ms = (_time.monotonic() - start) * 1000

        if resp.status_code < 500:
            scores["availability"] = 100
        else:
            scores["availability"] = 20
            factors.append(f"Agent returned {resp.status_code}")
    except Exception:
        scores["availability"] = 0
        scores["latency"] = 0
        factors.append("Agent unreachable — endpoint timed out or refused")
        # Can't probe further if unreachable
        d3 = int(scores["availability"] * 0.30 + scores.get("latency", 0) * 0.20)
        return max(0, min(100, d3)), factors[:3] if factors else ["Agent completely unreachable"]

    # Probe 2: Latency
    if latency_ms < 200:
        scores["latency"] = 100
    elif latency_ms < 500:
        scores["latency"] = 85
    elif latency_ms < 1000:
        scores["latency"] = 65
    elif latency_ms < 3000:
        scores["latency"] = 35
    else:
        scores["latency"] = 10
        factors.append(f"Response latency {latency_ms:.0f}ms (>3s)")

    # Probe 3: x402 header correctness
    if resp.status_code == 402:
        try:
            body = resp.json()
            x402_score = 0
            if "payTo" in body:
                x402_score += 25
            if "maxAmountRequired" in body:
                x402_score += 25
            if "network" in body:
                x402_score += 25
            if "resource" in body:
                x402_score += 25
            scores["x402"] = x402_score
            if x402_score < 75:
                factors.append("x402 payment headers incomplete")
        except Exception:
            scores["x402"] = 50
    else:
        scores["x402"] = 50  # can't assess

    # Probe 4: API metadata
    scores["metadata"] = 0
    try:
        parsed = requests.compat.urlparse(agent_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        r1 = requests.get(f"{base}/.well-known/x402", timeout=5)
        if r1.status_code == 200:
            try:
                data = r1.json()
                if "endpoints" in data or "routes" in data:
                    scores["metadata"] += 50
            except Exception:
                pass

        r2 = requests.get(f"{base}/api/info", timeout=5)
        if r2.status_code == 200:
            try:
                data = r2.json()
                if "version" in data:
                    scores["metadata"] += 25
                if "endpoints" in data:
                    scores["metadata"] += 25
            except Exception:
                pass
    except Exception:
        pass

    if scores["metadata"] == 0:
        factors.append("No API metadata endpoints found")

    # Probe 5: Data freshness
    scores["freshness"] = 50  # default neutral
    try:
        parsed = requests.compat.urlparse(agent_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        r3 = requests.get(f"{base}/api/info", timeout=5)
        if r3.status_code == 200:
            data = r3.json()
            for field_name in ["last_updated", "analyzed_at", "scan_timestamp"]:
                if field_name in data:
                    try:
                        ts_val = datetime.fromisoformat(data[field_name].replace("Z", "+00:00"))
                        age_seconds = (datetime.now(timezone.utc) - ts_val).total_seconds()
                        if age_seconds < 300:
                            scores["freshness"] = 100
                        elif age_seconds < 3600:
                            scores["freshness"] = 70
                        elif age_seconds < 86400:
                            scores["freshness"] = 40
                        else:
                            scores["freshness"] = 10
                    except Exception:
                        pass
                    break
    except Exception:
        pass

    # Weighted D3 score
    d3 = (
        scores.get("availability", 0) * 0.30
        + scores.get("latency", 0) * 0.20
        + scores.get("x402", 50) * 0.15
        + scores.get("metadata", 0) * 0.15
        + scores.get("freshness", 50) * 0.20
    )

    if not factors:
        factors.append("Infrastructure is responding well")

    return max(0, min(100, int(round(d3)))), factors[:3]


# -- Cross-Dimensional Pattern Detection --

def detect_cdp_patterns(
    d1_score: int,
    d2_score: int,
    d3_score: Optional[int],
    signals: dict,
    transactions: list[dict],
) -> tuple[int, list]:
    """Detect cross-dimensional patterns. Returns (modifier, patterns_list)."""
    modifier = 0
    patterns = []

    # Extract signal values safely. Use _sig() to handle None from token
    # transfer fallback (where some D2 signals are unavailable). Defaults
    # are "neutral" values that prevent false-triggering on missing data.
    def _sig(key, default):
        val = signals.get(key, default)
        return default if val is None else val

    failed_pct = _sig("failed_pct_24h", 0)
    d1_gas_eff_raw = _sig("d1_gas_eff_score", 100)
    tx_div_ratio = _sig("tx_diversity_ratio", 1.0)
    unique_contracts = _sig("unique_contracts", 10)
    max_consec = _sig("max_consecutive_failures", 0)
    gas_cv = _sig("gas_adaptation_cv", 0.5)
    persistent_gaps = _sig("persistent_nonce_gaps", 0)
    gap_ratio = _sig("gap_ratio", 1.0)
    storm_events = _sig("storm_events", 0)
    gas_adapt_score = _sig("gas_adaptation_score", 50)

    # Pattern 1: Zombie Agent
    if (d1_gas_eff_raw > 90 and failed_pct == 0 and
            tx_div_ratio < 0.02 and unique_contracts <= 1):
        modifier -= 15
        patterns.append({
            "name": "Zombie Agent",
            "detected": True,
            "severity": "critical",
            "description": (
                "Agent appears technically functional but brain-dead — executing "
                "the same single operation on repeat with zero diversity. Possible "
                "crashed strategy module with transaction sender still running."
            ),
        })

    # Pattern 2: Cascading Infrastructure Failure
    now_ts = datetime.now(timezone.utc).timestamp()
    seven_days_ago = now_ts - 7 * 86400
    recent = [tx for tx in transactions if int(tx.get("timeStamp", 0)) >= seven_days_ago]
    if len(recent) >= 10:
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        first_fail = sum(1 for tx in first_half if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0") / max(len(first_half), 1)
        second_fail = sum(1 for tx in second_half if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0") / max(len(second_half), 1)
        failure_rising = second_fail > first_fail * 1.5 and second_fail > 0.05

        if failure_rising and persistent_gaps > 0 and gap_ratio > 10:
            modifier -= 15
            patterns.append({
                "name": "Cascading Infrastructure Failure",
                "detected": True,
                "severity": "critical",
                "description": (
                    "Active degradation detected — failure rate is rising, nonce gaps "
                    "are widening, and timing shows crash-restart patterns. This agent "
                    "is in an infrastructure death spiral."
                ),
            })

    # Pattern 3: Stale Strategy
    if max_consec > 5 and tx_div_ratio < 0.03 and gas_cv < 0.05:
        modifier -= 10
        patterns.append({
            "name": "Stale Strategy",
            "detected": True,
            "severity": "warning",
            "description": (
                "Agent is repeatedly failing on the same contract interaction without "
                "adapting. Possible causes: revoked approval, removed liquidity, "
                "contract upgrade. Gas price is hardcoded (no adaptation)."
            ),
        })

    # Pattern 4: Healthy Operator
    if (d1_score >= 80 and gas_cv > 0.15 and tx_div_ratio > 0.05 and
            storm_events == 0 and (d3_score is None or d3_score >= 80)):
        modifier += 5
        patterns.append({
            "name": "Healthy Operator",
            "detected": True,
            "severity": "info",
            "description": (
                "All dimensions performing well — clean wallet, adaptive gas "
                "pricing, diverse interactions, and no retry storms."
            ),
        })

    # Pattern 5: Gas War Casualty
    if gas_cv > 0.40 and failed_pct > 15 and storm_events > 2 and gas_adapt_score < 60:
        modifier -= 10
        patterns.append({
            "name": "Gas War Casualty",
            "detected": True,
            "severity": "warning",
            "description": (
                "Agent is adapting gas prices but losing gas wars — high variance "
                "with high failure rate and retry storms. Needs better MEV protection "
                "or gas bidding strategy."
            ),
        })

    # Pattern 6: Phantom Activity (requires D3)
    if d3_score is not None and d3_score >= 70 and len(recent) < 3 and len(transactions) > 20:
        modifier -= 8
        patterns.append({
            "name": "Phantom Activity",
            "detected": True,
            "severity": "warning",
            "description": (
                "Agent server is running and responding to health checks, but "
                "near-zero on-chain activity in the last 7 days despite historical "
                "activity. Service is up but agent is idle."
            ),
        })

    # Pattern 7: Recovery in Progress
    twenty_four_hours_ago = now_ts - 86400
    recent_24h = [tx for tx in transactions if int(tx.get("timeStamp", 0)) >= twenty_four_hours_ago]
    older = [tx for tx in transactions if seven_days_ago <= int(tx.get("timeStamp", 0)) < twenty_four_hours_ago]
    if len(recent_24h) > 5 and len(older) > 10:
        recent_fail_rate = sum(1 for tx in recent_24h if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0") / len(recent_24h)
        older_fail_rate = sum(1 for tx in older if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0") / len(older)
        nonce_gaps_recent, _ = detect_nonce_issues(recent_24h)
        nonce_gaps_older, _ = detect_nonce_issues(older)
        if recent_fail_rate < older_fail_rate * 0.5 and nonce_gaps_recent <= nonce_gaps_older:
            modifier += 3
            patterns.append({
                "name": "Recovery in Progress",
                "detected": True,
                "severity": "info",
                "description": (
                    "Recent failure rate has dropped significantly compared to the "
                    "previous period. Agent appears to be recovering from a prior issue."
                ),
            })

    # Clamp modifier
    modifier = max(-15, min(5, modifier))

    return modifier, patterns


# -- Main AHS Calculation --

def calculate_ahs(
    address: str,
    tokens: list[dict],
    transactions: list[dict],
    eth_price: float,
    agent_url: Optional[str] = None,
    previous_score: Optional[int] = None,
    previous_ema: Optional[float] = None,
    scan_count: int = 1,
) -> AHSResult:
    """Calculate the full Agent Health Score.

    Args:
        address: Wallet address (lowercase).
        tokens: Token list from fetch_tokens_v2().
        transactions: Transaction list from fetch_transactions().
        eth_price: Current ETH/USD price.
        agent_url: Optional agent service URL for D3 probing.
        previous_score: Previous AHS score (from JWT token).
        previous_ema: Previous EMA score (from JWT token).
        scan_count: Number of scans (from JWT token).

    Returns:
        AHSResult with composite score and all details.
    """
    result = AHSResult(
        address=address,
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Filter to outgoing transactions
    outgoing = [
        tx for tx in transactions
        if tx.get("from", "").lower() == address.lower()
    ]

    result.tx_count = len(outgoing)

    # Calculate history span
    if outgoing:
        timestamps = [int(tx.get("timeStamp", 0)) for tx in outgoing]
        first_ts = min(t for t in timestamps if t > 0) if any(t > 0 for t in timestamps) else 0
        last_ts = max(timestamps)
        result.history_days = max(1, int((last_ts - first_ts) / 86400)) if first_ts > 0 else 0

    # -- D1: Wallet Hygiene --
    # Reuse wash analysis data extraction
    dust_count = 0
    dust_total_usd = 0.0
    for item in tokens:
        tok = item.get("token", {})
        exchange_rate_str = tok.get("exchange_rate")
        if exchange_rate_str is None:
            continue
        try:
            exchange_rate = float(exchange_rate_str)
            raw_value = item.get("value", "0")
            decimals = int(tok.get("decimals") or 18)
            balance = int(raw_value) / (10 ** decimals)
            usd_value = balance * exchange_rate
            if 0 <= usd_value < 0.01:
                dust_count += 1
                dust_total_usd += usd_value
        except (ValueError, TypeError, OverflowError):
            continue

    spam_count = sum(1 for item in tokens if is_spam_token(item.get("token", {}))[0])

    # Gas efficiency
    gas_efficiencies = []
    for tx in outgoing:
        gas_used = int(tx.get("gasUsed") or 0)
        gas_limit = int(tx.get("gas") or 0)
        if gas_limit > 0:
            gas_efficiencies.append(gas_used / gas_limit * 100)
    avg_gas_eff = statistics.mean(gas_efficiencies) if gas_efficiencies else 0

    # Failed tx rate 24h
    now_ts = datetime.now(timezone.utc).timestamp()
    twenty_four_hours_ago = now_ts - 86400
    txs_24h = [tx for tx in outgoing if int(tx.get("timeStamp", 0)) >= twenty_four_hours_ago]
    failed_24h = sum(1 for tx in txs_24h if tx.get("isError") == "1" or tx.get("txreceipt_status") == "0")
    failed_pct_24h = (failed_24h / len(txs_24h) * 100) if txs_24h else 0

    # Nonce gaps
    nonce_gaps, _ = detect_nonce_issues(outgoing)

    d1_score, d1_factors = calculate_d1_score(
        dust_count, dust_total_usd, spam_count,
        avg_gas_eff, failed_pct_24h, nonce_gaps,
    )
    result.d1_score = d1_score
    result.d1_top_factors = d1_factors

    # -- D2: Behavioural Patterns --
    if len(outgoing) >= 10:
        # Normal path: sufficient txlist data
        d2_score, d2_factors, d2_signals = calculate_d2_score(outgoing)
        result.d2_data_source = "txlist"
    else:
        # Fallback: try token transfers for smart contract wallets.
        # Use ALL transfers (not just outgoing) because ACP agents and
        # similar smart contract wallets primarily receive payments.
        # Normalize so "to" always = counterparty (the other party in
        # the transfer), matching how D2 sub-signals use the "to" field.
        token_transfers = fetch_token_transfers(address)
        addr_lower = address.lower()
        for tt in token_transfers:
            if tt.get("to", "").lower() == addr_lower:
                # Incoming: swap so "to" = sender (counterparty)
                tt["to"], tt["from"] = tt["from"], tt["to"]
        if len(token_transfers) >= 10:
            d2_score, d2_factors, d2_signals = calculate_d2_score_from_transfers(token_transfers)
            result.d2_data_source = "tokentx"
            result.tx_count = len(token_transfers)
            tt_timestamps = [int(tx.get("timeStamp", 0)) for tx in token_transfers]
            tt_valid = [t for t in tt_timestamps if t > 0]
            if tt_valid:
                tt_first = min(tt_valid)
                tt_last = max(tt_valid)
                result.history_days = max(result.history_days, max(1, int((tt_last - tt_first) / 86400)))
        else:
            # Neither txlist nor tokentx has enough data — baseline
            d2_score, d2_factors, d2_signals = calculate_d2_score(outgoing)
            result.d2_data_source = "txlist"
    result.d2_score = d2_score
    result.d2_top_factors = d2_factors
    d2_signals["d2_data_source"] = result.d2_data_source

    # -- D3: Infrastructure Health --
    d3_score = None
    d3_factors = []
    if agent_url:
        d3_score, d3_factors = probe_infrastructure_sync(agent_url)
        result.d3_score = d3_score
        result.d3_top_factors = d3_factors
        result.mode = "3D"
        result.d1_weight = 0.25
        result.d2_weight = 0.45
        result.d3_weight = 0.30
    else:
        result.mode = "2D"
        result.d1_weight = 0.30
        result.d2_weight = 0.70
        result.d3_weight = 0.0

    # -- Composite Score --
    if d3_score is not None:
        composite = 0.25 * d1_score + 0.45 * d2_score + 0.30 * d3_score
    else:
        composite = 0.30 * d1_score + 0.70 * d2_score

    # -- CDP Patterns --
    # Build signals dict for CDP detection
    d1_gas_eff_val = avg_gas_eff
    if 40 <= avg_gas_eff <= 85:
        d1_gas_eff_score = 100
    elif avg_gas_eff < 40:
        d1_gas_eff_score = (avg_gas_eff / 40) * 100
    else:
        d1_gas_eff_score = max(0, 100 - (avg_gas_eff - 85) * 5)

    cdp_signals = {
        **d2_signals,
        "failed_pct_24h": failed_pct_24h,
        "d1_gas_eff_score": d1_gas_eff_score,
    }

    cdp_modifier, patterns = detect_cdp_patterns(
        d1_score, d2_score, d3_score, cdp_signals, outgoing,
    )
    result.cdp_modifier = cdp_modifier
    result.patterns_detected = patterns

    composite = max(0, min(100, int(round(composite + cdp_modifier))))

    # -- Temporal Scoring --
    has_previous = previous_score is not None
    if has_previous:
        if scan_count == 2:
            temporal = composite * 0.8 + previous_score * 0.2
        else:
            alpha = 0.6
            prev_ema = previous_ema if previous_ema is not None else previous_score
            temporal = composite * alpha + prev_ema * (1 - alpha)

        result.temporal_score = int(round(temporal))

        delta = composite - previous_score
        if delta > 5:
            result.trend = "improving"
        elif delta < -5:
            result.trend = "declining"
        else:
            result.trend = "stable"

        composite = int(round(temporal))

    result.agent_health_score = max(0, min(100, composite))

    # Grade
    grade_letter, grade_label = _ahs_grade(result.agent_health_score)
    result.grade = grade_letter
    result.grade_label = grade_label

    # Confidence
    result.confidence = _ahs_confidence(
        result.tx_count, result.history_days,
        has_d3=d3_score is not None, has_previous=has_previous,
    )

    # Next scan recommendation
    result.next_scan_recommended = _next_scan_recommendation(
        result.agent_health_score, result.confidence,
    )

    # -- Recommendations --
    recs = []
    # Based on worst patterns
    for p in patterns:
        if p["severity"] == "critical":
            recs.append(p["description"][:120])
    # Based on worst D2 signals
    for factor in d2_factors:
        if "healthy" not in factor.lower() and len(recs) < 3:
            recs.append(factor)
    # Based on D1 issues
    for factor in d1_factors:
        if "healthy" not in factor.lower() and len(recs) < 3:
            recs.append(factor)
    # Based on D3 issues
    for factor in d3_factors:
        if "well" not in factor.lower() and len(recs) < 3:
            recs.append(factor)

    if not recs:
        if result.agent_health_score >= 80:
            recs.append(f"Maintaining excellent health — scan again in {result.next_scan_recommended}")
        elif result.tx_count < 20:
            recs.append(f"Insufficient history for full assessment — scan again in 7 days to build baseline")
        else:
            recs.append(f"Score {result.agent_health_score}/100 — next scan recommended in {result.next_scan_recommended}")

    result.recommendations = recs[:3]

    # Store internal signals
    result._signals = {
        "d1_dust_count": dust_count,
        "d1_spam_count": spam_count,
        "d1_gas_eff": avg_gas_eff,
        "d1_failed_pct_24h": failed_pct_24h,
        "d1_nonce_gaps": nonce_gaps,
        **d2_signals,
    }

    return result


if __name__ == "__main__":
    main()
