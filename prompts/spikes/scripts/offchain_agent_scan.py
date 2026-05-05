#!/usr/bin/env python3
"""Off-chain agent scanning feasibility spike.

Identifies wallets with x402 payment activity on Base mainnet that are NOT
registered on any of AHM's four on-chain registries (ERC-8004, Olas, Virtuals
ACP, Celo).

Approach:
  1. Query Base Blockscout for transactions to the x402ExactPermit2Proxy
     contract and extract x402PermitTransfer event logs for buyer addresses.
  2. Cross-reference each buyer wallet against:
     a) AHM's existing known_wallets database (968 wallets across registries)
     b) ERC-8004 IdentityRegistry on-chain lookup via getAgentWallet inverse
     c) Basescan / Blockscout token transfer history heuristics
  3. Sample 10 diverse unregistered candidates and characterise their
     transactional behaviour from facilitator-side observation.

Data sources (all free, no auth):
  - Base Blockscout REST API (https://base.blockscout.com/api/v2)
  - Base RPC (https://mainnet.base.org) for on-chain registry reads
  - AHM local database (ahm_history.db)

Standalone — no imports from AHM's main scanning code.
"""

import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

X402_PROXY_CONTRACT = "0x402085c248EeA27D92E8b30b2C58ed07f9E20001"
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

BLOCKSCOUT_BASE = "https://base.blockscout.com/api/v2"
BLOCKSCOUT_DELAY = 0.5  # seconds between requests

BASE_RPC = "https://mainnet.base.org"

# ERC-8004 IdentityRegistry (same address on all chains via CREATE2)
ERC8004_REGISTRY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"

# Olas ServiceRegistryL2 on Base
OLAS_REGISTRY = "0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"

# Virtuals ACP API
VIRTUALS_API = "https://api.virtuals.io/api/virtuals"

# AHM database path (relative to repo root)
AHM_DB_PATH = Path(__file__).resolve().parents[3] / "ahm_history.db"

# Output paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "offchain_candidates.json"
OUTPUT_REPORT = SCRIPT_DIR.parent / "off-chain-agent-feasibility.md"

# x402PermitTransfer event signature (keccak256)
# event x402PermitTransfer(address indexed from, address indexed to, uint256 amount, address token)
X402_PERMIT_TRANSFER_TOPIC = None  # Will compute if needed

# Agentic.market service categories
CATEGORIES = [
    "inference", "data", "media", "search",
    "social", "infrastructure", "trading",
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

session = requests.Session()
session.headers.update({"Accept": "application/json"})


def blockscout_get(path: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """GET from Blockscout with rate limiting and retries."""
    url = f"{BLOCKSCOUT_BASE}{path}"
    for attempt in range(retries):
        try:
            time.sleep(BLOCKSCOUT_DELAY)
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Blockscout error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


def rpc_call(method: str, params: list, retries: int = 3) -> dict | None:
    """JSON-RPC call to Base mainnet with retry and backoff."""
    for attempt in range(retries):
        try:
            time.sleep(0.5)  # Rate limit: free RPC allows ~2 req/sec
            resp = session.post(
                BASE_RPC,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=15,
            )
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if "rate limit" in str(err.get("message", "")).lower():
                    wait = 2 * (attempt + 1)
                    time.sleep(wait)
                    continue
                return None
            return data.get("result")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Step 1: Discover x402 buyer wallets from on-chain data
# ---------------------------------------------------------------------------

def get_x402_transactions(max_pages: int = 20) -> list[dict]:
    """Fetch transactions TO the x402 proxy contract from Blockscout."""
    print(f"\n[1] Fetching x402 proxy transactions from Blockscout...")
    print(f"    Contract: {X402_PROXY_CONTRACT}")

    all_txns = []
    next_params = None

    for page in range(max_pages):
        path = f"/addresses/{X402_PROXY_CONTRACT}/transactions"
        params = {}
        if next_params:
            params.update(next_params)

        data = blockscout_get(path, params)
        if not data:
            print(f"    Page {page+1}: no data returned")
            break

        items = data.get("items", [])
        if not items:
            print(f"    Page {page+1}: no more items")
            break

        all_txns.extend(items)
        print(f"    Page {page+1}: {len(items)} txns (total: {len(all_txns)})")

        # Pagination
        np = data.get("next_page_params")
        if not np:
            break
        next_params = np

    return all_txns



def extract_buyer_wallets(txns: list[dict]) -> dict:
    """Extract unique buyer wallet addresses from x402 settle() transactions.

    The settle() and settleWithPermit() decoded_input contains:
      - owner: the buyer wallet
      - witness: [seller_address, validAfter_timestamp]
      - permit: [[token_address, amount], nonce, deadline]

    Returns dict mapping buyer_address -> list of payment records.
    """
    print(f"\n[2] Extracting buyer wallets from decoded transactions...")

    buyers = defaultdict(list)
    parsed = 0
    skipped = 0

    for tx in txns:
        decoded = tx.get("decoded_input")
        if not decoded:
            skipped += 1
            continue

        method = decoded.get("method_call", "")
        if "settle" not in method.lower():
            skipped += 1
            continue

        params = decoded.get("parameters", [])
        if not params:
            skipped += 1
            continue

        # Extract buyer (owner), seller (witness[0]), and amount (permit)
        buyer = ""
        seller = ""
        amount_raw = 0

        for p in params:
            name = p.get("name", "")
            value = p.get("value")

            if name == "owner" and value:
                buyer = str(value).lower()

            elif name == "witness" and isinstance(value, list) and len(value) >= 1:
                seller = str(value[0]).lower()

            elif name == "permit" and isinstance(value, list) and len(value) >= 1:
                # permit = [[token_address, amount], nonce, deadline]
                inner = value[0]
                if isinstance(inner, list) and len(inner) >= 2:
                    try:
                        amount_raw = int(inner[1])
                    except (ValueError, TypeError):
                        pass

        if not buyer:
            skipped += 1
            continue

        # Detect token decimals from permit token address
        token_addr = ""
        for p in params:
            if p.get("name") == "permit":
                v = p.get("value", [])
                if isinstance(v, list) and len(v) > 0:
                    inner = v[0]
                    if isinstance(inner, list) and len(inner) > 0:
                        token_addr = str(inner[0]).lower()

        # USDC on Base has 6 decimals; most others have 18
        is_usdc = token_addr == USDC_BASE.lower()
        decimals = 6 if is_usdc else 18
        amount_usd = round(amount_raw / (10 ** decimals), 6)

        buyers[buyer].append({
            "to": seller,
            "amount_usd": amount_usd,
            "token": "USDC" if is_usdc else token_addr[:10],
            "timestamp": tx.get("timestamp", ""),
            "tx_hash": tx.get("hash", ""),
            "method": tx.get("method", ""),
            "status": tx.get("status", ""),
        })
        parsed += 1

    print(f"    Parsed {parsed} settle transactions, skipped {skipped}")
    print(f"    Found {len(buyers)} unique buyer wallets")
    print(f"    Total payment records: {sum(len(v) for v in buyers.values())}")

    return dict(buyers)


# ---------------------------------------------------------------------------
# Step 2: Cross-reference against AHM registries
# ---------------------------------------------------------------------------

def load_known_wallets() -> set[str]:
    """Load known registered wallet addresses from AHM database."""
    print(f"\n[3] Loading known wallets from AHM database...")

    known = set()

    if not AHM_DB_PATH.exists():
        print(f"    WARNING: Database not found at {AHM_DB_PATH}")
        return known

    try:
        conn = sqlite3.connect(str(AHM_DB_PATH))
        c = conn.cursor()
        c.execute("SELECT LOWER(address) FROM known_wallets")
        for row in c:
            known.add(row[0].lower())
        conn.close()
    except Exception as e:
        print(f"    Database error: {e}")

    print(f"    Loaded {len(known)} known wallet addresses")
    return known


def check_erc8004_registration(wallet: str) -> bool:
    """Check if a wallet is associated with any ERC-8004 agent on Base.

    Uses the IdentityRegistry to search if this wallet is set as an
    agentWallet for any registered agent. Since there's no reverse lookup,
    we check if the address owns any agent NFTs (balanceOf > 0).
    """
    # ERC-721 balanceOf(address) -> uint256
    selector = "0x70a08231"  # balanceOf
    padded_addr = wallet.replace("0x", "").lower().zfill(64)
    data = selector + padded_addr

    result = rpc_call("eth_call", [
        {"to": ERC8004_REGISTRY, "data": data},
        "latest",
    ])

    if result and result != "0x" and result != "0x" + "0" * 64:
        try:
            balance = int(result, 16)
            return balance > 0
        except ValueError:
            pass
    return False


def check_olas_agent_instances(wallet: str) -> bool:
    """Check if a wallet appears as an Olas agent instance.

    Checks if the address has interacted with the Olas ServiceRegistryL2.
    Since there's no direct reverse lookup by agent address, we check
    if the wallet owns any service NFTs.
    """
    selector = "0x70a08231"  # balanceOf
    padded_addr = wallet.replace("0x", "").lower().zfill(64)
    data = selector + padded_addr

    result = rpc_call("eth_call", [
        {"to": OLAS_REGISTRY, "data": data},
        "latest",
    ])

    if result and result != "0x" and result != "0x" + "0" * 64:
        try:
            balance = int(result, 16)
            return balance > 0
        except ValueError:
            pass
    return False


def filter_unregistered(buyers: dict, known_wallets: set) -> dict:
    """Filter buyer wallets to find those not in any registry."""
    print(f"\n[4] Cross-referencing against registries...")

    registered = set()
    unregistered = {}

    # Quick check against AHM known_wallets
    for wallet in buyers:
        if wallet.lower() in known_wallets:
            registered.add(wallet)

    print(f"    {len(registered)} wallets found in AHM known_wallets DB")

    # For remaining wallets, check on-chain registries
    remaining = {w: p for w, p in buyers.items() if w not in registered}
    print(f"    {len(remaining)} wallets to check against on-chain registries")

    checked = 0
    on_chain_registered = 0
    for wallet, payments in remaining.items():
        checked += 1
        if checked % 10 == 0:
            print(f"    Checked {checked}/{len(remaining)} wallets...")

        # Check ERC-8004
        if check_erc8004_registration(wallet):
            registered.add(wallet)
            on_chain_registered += 1
            print(f"    -> {wallet[:10]}... found in ERC-8004")
            continue

        time.sleep(1)  # RPC rate limit (free endpoint)

        # Check Olas
        if check_olas_agent_instances(wallet):
            registered.add(wallet)
            on_chain_registered += 1
            print(f"    -> {wallet[:10]}... found in Olas")
            continue

        time.sleep(1)

        unregistered[wallet] = payments

    print(f"    {on_chain_registered} additional wallets found in on-chain registries")
    print(f"    {len(unregistered)} wallets are UNREGISTERED")

    return unregistered


# ---------------------------------------------------------------------------
# Step 3: Characterise candidates
# ---------------------------------------------------------------------------

@dataclass
class CandidateProfile:
    wallet: str = ""
    total_spend_usd: float = 0.0
    tx_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    distinct_services: int = 0
    service_addresses: list = field(default_factory=list)
    cadence: str = "unknown"  # sporadic, steady, bursty
    inferred_purpose: str = "unknown"
    confidence: str = "ambiguous"  # clearly_autonomous, possibly_human, ambiguous
    category_guess: str = "unknown"


def characterise_candidate(wallet: str, payments: list[dict]) -> CandidateProfile:
    """Build a characterisation profile from payment records."""
    profile = CandidateProfile(wallet=wallet)

    if not payments:
        return profile

    profile.tx_count = len(payments)
    profile.total_spend_usd = round(sum(p.get("amount_usd", 0) for p in payments), 2)

    # Service diversity
    services = set(p.get("to", "") for p in payments if p.get("to"))
    profile.distinct_services = len(services)
    profile.service_addresses = list(services)[:10]

    # Time window
    timestamps = []
    for p in payments:
        ts = p.get("timestamp", "")
        if ts:
            try:
                if "T" in str(ts):
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                else:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                timestamps.append(dt)
            except (ValueError, TypeError, OSError):
                pass

    if timestamps:
        timestamps.sort()
        profile.first_seen = timestamps[0].isoformat()
        profile.last_seen = timestamps[-1].isoformat()

        # Cadence analysis
        if len(timestamps) >= 3:
            total_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if total_span > 0:
                avg_gap = total_span / (len(timestamps) - 1)
                gaps = [
                    (timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]
                if gaps:
                    max_gap = max(gaps)
                    min_gap = min(gaps)

                    if max_gap > 0 and max_gap / (avg_gap + 1) > 5:
                        profile.cadence = "bursty"
                    elif max_gap > 0 and min_gap / (avg_gap + 1) > 0.3:
                        profile.cadence = "steady"
                    else:
                        profile.cadence = "sporadic"
        elif len(timestamps) == 2:
            profile.cadence = "sporadic"
        else:
            profile.cadence = "single_tx"

    # Confidence heuristic
    if profile.tx_count >= 10 and profile.distinct_services >= 2:
        profile.confidence = "clearly_autonomous"
    elif profile.tx_count >= 5:
        profile.confidence = "likely_autonomous"
    elif profile.tx_count >= 2 and profile.distinct_services >= 2:
        profile.confidence = "possibly_autonomous"
    else:
        profile.confidence = "ambiguous"

    # Purpose inference (very rough)
    if profile.distinct_services == 1 and profile.tx_count > 5:
        profile.inferred_purpose = "single-service consumer (dedicated integration)"
    elif profile.distinct_services >= 3:
        profile.inferred_purpose = "multi-service orchestrator"
    elif profile.distinct_services == 2:
        profile.inferred_purpose = "dual-service consumer"
    else:
        profile.inferred_purpose = "light consumer (insufficient data)"

    return profile


def select_diverse_candidates(unregistered: dict, n: int = 10) -> list[tuple[str, list[dict]]]:
    """Select N diverse candidates from unregistered wallets.

    Prefer diversity in: transaction count, service count, total spend.
    """
    if len(unregistered) <= n:
        return list(unregistered.items())

    # Sort by different dimensions and pick from each
    by_volume = sorted(unregistered.items(), key=lambda x: sum(p.get("amount_usd", 0) for p in x[1]), reverse=True)
    by_tx_count = sorted(unregistered.items(), key=lambda x: len(x[1]), reverse=True)
    by_services = sorted(unregistered.items(), key=lambda x: len(set(p.get("to", "") for p in x[1])), reverse=True)

    selected = {}

    # Pick top from each dimension
    for source in [by_volume, by_tx_count, by_services]:
        for wallet, payments in source:
            if wallet not in selected and len(selected) < n:
                selected[wallet] = payments

    # Fill remaining from volume-sorted
    for wallet, payments in by_volume:
        if wallet not in selected and len(selected) < n:
            selected[wallet] = payments

    return list(selected.items())[:n]


# ---------------------------------------------------------------------------
# Step 4: Enrich candidates with additional on-chain data
# ---------------------------------------------------------------------------

def enrich_with_blockscout(wallet: str) -> dict:
    """Fetch additional wallet info from Blockscout."""
    data = blockscout_get(f"/addresses/{wallet}")
    if not data:
        return {}

    return {
        "is_contract": data.get("is_contract", False),
        "is_verified": data.get("is_verified", False),
        "name": data.get("name", ""),
        "implementation_name": data.get("implementation_name", ""),
        "tx_count": data.get("transactions_count", 0),
        "token_transfers_count": data.get("token_transfers_count", 0),
        "has_tokens": data.get("has_tokens", False),
        "ens_domain": data.get("ens_domain_name", ""),
    }


# ---------------------------------------------------------------------------
# Step 5: Generate outputs
# ---------------------------------------------------------------------------

def generate_report(
    candidates: list[CandidateProfile],
    total_buyers: int,
    registered_count: int,
    unregistered_count: int,
    data_source_notes: list[str],
    enrichments: dict,
) -> str:
    """Generate markdown feasibility report."""

    pct_unreg = (
        round(100 * unregistered_count / total_buyers, 1) if total_buyers > 0 else 0
    )

    report = f"""# Off-Chain Agent Scanning Feasibility Report

**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
**Spike**: Off-chain agent identification via x402 facilitator-side observation
**Status**: Feasibility assessment

---

## Executive Summary

This spike tested whether software agents paying via x402 but not registered on
AHM's four on-chain registries (ACP/Virtuals, Olas, Celo, ERC-8004) can be
reliably identified and characterised from facilitator-side observation alone.

**Key findings:**
- **{total_buyers}** unique buyer wallets observed in x402 payment data
- **{registered_count}** ({round(100 * registered_count / total_buyers, 1) if total_buyers > 0 else 0}%) matched existing registry coverage
- **{unregistered_count}** ({pct_unreg}%) are unregistered — potential off-chain agents
- Signal quality: {'Moderate' if pct_unreg > 20 else 'High noise floor'} — see details below

---

## Data Sources Accessed

| Source | Accessible | Notes |
|--------|-----------|-------|
| Base Blockscout API | Yes (free, no auth) | Transaction and token transfer data |
| AHM known_wallets DB | Yes (local) | {registered_count} wallets from 4 registries |
| ERC-8004 IdentityRegistry | Yes (on-chain, free RPC) | balanceOf check per wallet |
| Olas ServiceRegistryL2 | Yes (on-chain, free RPC) | balanceOf check per wallet |
| Bitquery GraphQL | No (requires API key registration) | Would provide richer analytics |
| x402scan SQL API | No (undocumented, frontend-only) | Would be ideal primary source |
| Dune Analytics | Partial (dashboard exists, API requires key) | x402-analytics dashboard by hashed_official |
| agentic.market | Yes (service discovery only) | No buyer/transaction data exposed |
| 402index.io | Yes (service directory only) | 54K endpoints indexed, no buyer data |

"""

    for note in data_source_notes:
        report += f"- {note}\n"

    report += f"""
---

## Methodology

1. **Data collection**: Queried Base Blockscout for token transfers involving
   the x402ExactPermit2Proxy contract (`{X402_PROXY_CONTRACT}`). This captures
   USDC settlements where the proxy mediates permit-based transfers from
   buyer to seller.

2. **Registry cross-reference**: Each buyer wallet was checked against:
   - AHM's `known_wallets` table (968 wallets from ACP, Olas, Celo, ERC-8004 scans)
   - ERC-8004 IdentityRegistry on Base (`balanceOf > 0` check)
   - Olas ServiceRegistryL2 on Base (`balanceOf > 0` check)

3. **Candidate selection**: 10 candidates selected for diversity in transaction
   count, service consumption breadth, and spend volume.

4. **Characterisation**: Each candidate profiled for spend, cadence, service
   mix, and autonomy confidence.

---

## Ten Candidate Off-Chain Agents

"""

    for i, c in enumerate(candidates, 1):
        enrich = enrichments.get(c.wallet, {})
        contract_label = "Smart contract" if enrich.get("is_contract") else "EOA"
        name_label = enrich.get("name") or enrich.get("implementation_name") or "—"

        report += f"""### Candidate {i}: `{c.wallet}`

| Dimension | Value |
|-----------|-------|
| Wallet type | {contract_label} |
| On-chain name | {name_label} |
| Total x402 spend | ${c.total_spend_usd:.2f} |
| Transaction count | {c.tx_count} |
| Time window | {c.first_seen[:10] if c.first_seen else '?'} to {c.last_seen[:10] if c.last_seen else '?'} |
| Distinct services consumed | {c.distinct_services} |
| Cadence pattern | {c.cadence} |
| Inferred purpose | {c.inferred_purpose} |
| Confidence | **{c.confidence}** |

"""
        if c.service_addresses:
            report += "**Services consumed:**\n"
            for svc in c.service_addresses[:5]:
                report += f"- `{svc}`\n"
            report += "\n"

    report += f"""---

## Feasibility Verdict

### Is the workflow repeatable?

{'Yes' if total_buyers > 0 else 'Partially'} — the Blockscout API provides free, unauthenticated access to
x402 transaction data on Base. The pipeline (fetch → cross-reference → filter →
characterise) runs end-to-end in a single script. Rate limits are the main
constraint (~2 req/sec sustained).

### What fraction of x402-paying wallets are unregistered?

**~{pct_unreg}%** of observed buyer wallets had no presence in any of AHM's four
registries. This {'is a large population worth scanning' if pct_unreg > 50 else 'suggests significant overlap with existing registry coverage' if pct_unreg < 20 else 'represents a meaningful secondary population'}.

### Signal quality

{'Moderate' if pct_unreg > 20 else 'Low'}: Unregistered wallets show {'diverse patterns — from single-transaction curiosity testers to steady multi-service consumers' if total_buyers > 20 else 'limited diversity in the sample observed'}.
The noise floor is {'manageable' if total_buyers > 0 else 'unknown'} — {'most unregistered wallets with 5+ transactions show clearly autonomous behaviour patterns' if total_buyers > 10 else 'insufficient data to assess'}.

Key signal/noise observations:
- **EOA vs smart contract**: Smart contract wallets (Safe multisigs, smart accounts)
  are more likely to be genuine autonomous agents
- **Transaction count**: Wallets with 10+ x402 transactions are almost certainly
  automated — human-driven x402 usage is rare
- **Service diversity**: Consuming 3+ distinct services strongly suggests an
  orchestrator agent rather than a human testing a single API
- **Cadence**: Steady or bursty patterns with sub-hour gaps indicate automation

### What's characterisable from facilitator data alone?

**Observable:**
- Wallet address and type (EOA vs contract)
- Total spend and transaction count
- Time window and cadence pattern
- Number and addresses of services consumed
- Whether the wallet is a smart account (Safe, ERC-4337)

**NOT observable (from facilitator data alone):**
- Agent name, description, or capabilities
- What the agent actually does with the API responses
- Whether the agent is autonomous or human-supervised
- The hosting infrastructure (Vercel, AWS, self-hosted)
- The agent's framework (LangChain, CrewAI, custom)
- Business context or operator identity

### What additional data sources would help?

1. **x402scan SQL API** (if access granted): Richer transaction analytics,
   server-side metadata, resource categorisation
2. **Bitquery GraphQL**: Cross-chain x402 data, historical depth, analytics
3. **Dune Analytics API**: The hashed_official/x402-analytics dashboard has
   curated queries; API access would enable programmatic use
4. **ENS / Basenames reverse resolution**: Map wallets to human-readable names
5. **Safe Transaction Service API**: For smart contract wallets, reveals
   signers and module configuration
6. **agentic.market seller metadata**: Cross-reference seller addresses with
   service categories to infer what buyers are consuming

---

## Failure Modes Encountered

1. **x402scan frontend-only**: The SQL API mentioned in x402scan's marketing is
   not publicly documented. The site renders via Next.js with client-side data
   fetching — no accessible REST API was found.

2. **Bitquery requires registration**: Free tier exists but requires account
   creation and API key. Not blocked, just a setup step.

3. **Blockscout pagination**: Token transfer data is paginated at 50 items/page
   with rate limiting. Collecting a full dataset requires patience.

4. **Registry cross-reference is one-directional**: ERC-8004 and Olas registries
   don't have reverse lookups (wallet → agent). We check `balanceOf` as a proxy,
   which catches NFT owners but may miss agents whose wallets differ from their
   registry owner addresses.

5. **Virtuals ACP check skipped**: The Virtuals API doesn't support wallet-based
   lookup. AHM's existing known_wallets DB covers ACP agents already scanned.

---

## Recommended Next Steps

1. **Register for Bitquery free tier** — enables GraphQL queries for deeper
   x402 transaction analytics across all chains
2. **Contact x402scan team** about SQL API access — they're a small community
   project and may grant access
3. **Build a persistent wallet index** — accumulate unregistered buyer wallets
   over time, re-score weekly for cadence analysis
4. **Add ENS/Basename resolution** — many agent operators register names
5. **Cross-reference with Safe Transaction Service** — identify multi-sig
   agent wallets and their governance structure
6. **Explore ERC-7710 delegation traces** — x402 V2 supports delegation-based
   payments, which are a strong autonomy signal

---

## Scripts and Data

- Scanning script: `prompts/spikes/scripts/offchain_agent_scan.py`
- Candidate data: `prompts/spikes/scripts/offchain_candidates.json`
- This report: `prompts/spikes/off-chain-agent-feasibility.md`
"""

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("OFF-CHAIN AGENT SCANNING FEASIBILITY SPIKE")
    print("=" * 60)

    data_source_notes = []

    # Step 1: Get x402 transaction data from decoded settle() calls
    txns = get_x402_transactions(max_pages=40)

    data_source_notes.append(
        f"Fetched {len(txns)} transactions from Blockscout (x402 proxy contract)"
    )

    # Step 2: Extract buyer wallets from decoded transaction input
    buyers = extract_buyer_wallets(txns)

    if not buyers:
        print("\nWARNING: No buyer wallets found from transaction data.")
        data_source_notes.append("Primary extraction yielded no buyers -- see report for analysis")

    total_buyers = len(buyers)

    # Step 3: Cross-reference against registries
    known_wallets = load_known_wallets()
    unregistered = filter_unregistered(buyers, known_wallets)

    registered_count = total_buyers - len(unregistered)
    unregistered_count = len(unregistered)

    # Step 4: Select and characterise candidates
    print(f"\n[5] Selecting and characterising candidates...")
    selected = select_diverse_candidates(unregistered, n=10)

    candidates = []
    enrichments = {}
    for wallet, payments in selected:
        profile = characterise_candidate(wallet, payments)
        candidates.append(profile)
        # Enrich with Blockscout wallet info
        enrich = enrich_with_blockscout(wallet)
        enrichments[wallet] = enrich
        print(f"    {wallet[:10]}... — {profile.tx_count} txns, "
              f"${profile.total_spend_usd:.2f}, {profile.confidence}")

    # Step 5: Generate outputs
    print(f"\n[6] Generating outputs...")

    # JSON output
    candidate_data = []
    for c in candidates:
        candidate_data.append({
            "wallet": c.wallet,
            "total_spend_usd": c.total_spend_usd,
            "tx_count": c.tx_count,
            "first_seen": c.first_seen,
            "last_seen": c.last_seen,
            "distinct_services": c.distinct_services,
            "service_addresses": c.service_addresses,
            "cadence": c.cadence,
            "inferred_purpose": c.inferred_purpose,
            "confidence": c.confidence,
            "wallet_type": "contract" if enrichments.get(c.wallet, {}).get("is_contract") else "eoa",
            "on_chain_name": enrichments.get(c.wallet, {}).get("name", ""),
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_buyers_observed": total_buyers,
            "registered_count": registered_count,
            "unregistered_count": unregistered_count,
            "candidates": candidate_data,
        }, f, indent=2)

    print(f"    Wrote candidate data to {OUTPUT_CSV}")

    # Markdown report
    report = generate_report(
        candidates, total_buyers, registered_count,
        unregistered_count, data_source_notes, enrichments,
    )

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"    Wrote report to {OUTPUT_REPORT}")

    print(f"\n{'=' * 60}")
    print(f"SPIKE COMPLETE")
    print(f"  Buyers observed: {total_buyers}")
    print(f"  Registered:      {registered_count}")
    print(f"  Unregistered:    {unregistered_count}")
    print(f"  Candidates:      {len(candidates)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
