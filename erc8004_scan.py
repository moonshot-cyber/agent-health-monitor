#!/usr/bin/env python3
"""ERC-8004 Identity Registry scan spike.

Discovers agents registered on the ERC-8004 Identity Registry (Base mainnet),
resolves their registration URIs, extracts wallet addresses, runs AHS scans,
and stores results in the scan history database.

Usage:
    python erc8004_scan.py                    # Full scan (200 agents, 100 AHS scans)
    python erc8004_scan.py --max-agents 50 --max-scans 20   # Smaller run
    python erc8004_scan.py --skip-scan        # Enumerate + resolve only
    python erc8004_scan.py --start-id 100     # Resume from agent ID 100
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from web3 import Web3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_RPC = "https://mainnet.base.org"
REGISTRY_ADDRESS = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
ABI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_erc8004_abi.json")
AHM_AGENT_ID = 32328  # Our own agentId — known lower bound

# Rate limiting
RPC_DELAY = 0.1         # RPC calls (free endpoint, minimal delay)
BLOCKSCOUT_DELAY = 2.0  # Blockscout API (strict rate limits)
URI_FETCH_DELAY = 0.5   # HTTP fetches for agentURI docs
URI_FETCH_TIMEOUT = 10  # seconds

# IPFS gateway
IPFS_GATEWAY = "https://ipfs.io/ipfs/"

# Output
CSV_PATH = "erc8004_scan_results.csv"
MD_PATH = "erc8004_scan_results.md"

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

ETH_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

WALLET_KEYS = {
    "wallet", "address", "payment_address", "payto", "pay_to",
    "signer", "operator", "owner", "recipient", "treasury",
    "evm_address", "eth_address", "base_address",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    agent_id: int
    owner: str = ""
    agent_wallet: str = ""
    token_uri: str = ""
    # URI resolution
    uri_resolved: bool = False
    uri_error: str = ""
    agent_name: str = ""
    registration_json: dict | None = None
    extracted_wallets: list = field(default_factory=list)
    # AHS results (filled per-wallet, stored on the record for the primary wallet)
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
# Phase 1: Registry Discovery
# ---------------------------------------------------------------------------

def discover_registry():
    """Connect to Base mainnet, load ABI, validate registry contract.

    Returns (w3, registry_contract, info_dict) or exits on failure.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Registry Discovery")
    print("=" * 60)

    # Load ABI
    if not os.path.exists(ABI_PATH):
        print(f"[!] ABI file not found: {ABI_PATH}")
        sys.exit(1)
    with open(ABI_PATH) as f:
        abi = json.load(f)
    print(f"[+] ABI loaded: {len(abi)} entries")

    # Connect
    w3 = Web3(Web3.HTTPProvider(BASE_RPC))
    if not w3.is_connected():
        print(f"[!] Cannot connect to {BASE_RPC}")
        sys.exit(1)
    chain_id = w3.eth.chain_id
    block = w3.eth.block_number
    print(f"[+] Connected to Base mainnet (chain {chain_id}, block {block:,})")

    registry = w3.eth.contract(
        address=Web3.to_checksum_address(REGISTRY_ADDRESS), abi=abi
    )

    # Validate
    info = {"chain_id": chain_id, "block_number": block}

    try:
        info["name"] = registry.functions.name().call()
        print(f"[+] Contract name: {info['name']}")
    except Exception as e:
        info["name"] = f"error: {e}"

    try:
        info["symbol"] = registry.functions.symbol().call()
        print(f"[+] Contract symbol: {info['symbol']}")
    except Exception as e:
        info["symbol"] = f"error: {e}"

    try:
        info["version"] = registry.functions.getVersion().call()
        print(f"[+] Contract version: {info['version']}")
    except Exception as e:
        info["version"] = f"error: {e}"

    # Validate with our known agentId
    try:
        ahm_owner = registry.functions.ownerOf(AHM_AGENT_ID).call()
        print(f"[+] ownerOf({AHM_AGENT_ID}) = {ahm_owner} (AHM)")
        info["ahm_owner"] = ahm_owner
    except Exception as e:
        print(f"[!] ownerOf({AHM_AGENT_ID}) failed: {e}")
        info["ahm_owner"] = None

    # List available view functions
    view_fns = [
        item["name"] for item in abi
        if item.get("type") == "function"
        and item.get("stateMutability") in ("view", "pure")
    ]
    print(f"[+] View functions: {', '.join(sorted(view_fns))}")
    info["view_functions"] = view_fns

    # Check for totalSupply (not in ABI, confirm)
    has_total_supply = "totalSupply" in view_fns
    print(f"[+] Has totalSupply: {has_total_supply}")
    info["has_total_supply"] = has_total_supply

    return w3, registry, info


# ---------------------------------------------------------------------------
# Phase 2: Agent Enumeration
# ---------------------------------------------------------------------------

def find_max_agent_id(registry) -> int:
    """Binary search on ownerOf() to find the highest valid agentId."""
    print("\n[*] Finding max agentId via binary search...")

    # Exponential probe upward from known ID
    probe = AHM_AGENT_ID
    last_good = probe
    while True:
        try:
            registry.functions.ownerOf(probe).call()
            last_good = probe
            probe *= 2
            time.sleep(RPC_DELAY)
        except Exception:
            break

    # Binary search between last_good and probe
    lo, hi = last_good, probe
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            registry.functions.ownerOf(mid).call()
            lo = mid
        except Exception:
            hi = mid - 1
        time.sleep(RPC_DELAY)

    print(f"[+] Max agentId: {lo:,}")
    return lo


def enumerate_agents(registry, max_agents=200, start_id=1) -> list[AgentRecord]:
    """Iterate agent IDs, fetching owner, tokenURI, and agentWallet for each."""
    print(f"\n[*] Enumerating agents from ID {start_id} (max {max_agents})...")

    agents = []
    skipped = 0
    agent_id = start_id

    while len(agents) < max_agents:
        if agent_id % 20 == 0 or len(agents) % 20 == 0 and len(agents) > 0:
            print(f"  [{len(agents)}/{max_agents}] probing ID {agent_id}...")

        # ownerOf
        try:
            owner = registry.functions.ownerOf(agent_id).call()
        except Exception:
            skipped += 1
            agent_id += 1
            time.sleep(RPC_DELAY)
            # If we've skipped too many in a row, we've likely hit the end
            if skipped > 50:
                print(f"  [!] 50+ consecutive skips at ID {agent_id}, stopping enumeration")
                break
            continue

        skipped = 0  # reset consecutive skip counter
        rec = AgentRecord(agent_id=agent_id, owner=owner.lower())

        # tokenURI
        try:
            rec.token_uri = registry.functions.tokenURI(agent_id).call()
        except Exception:
            pass
        time.sleep(RPC_DELAY)

        # getAgentWallet
        try:
            wallet = registry.functions.getAgentWallet(agent_id).call()
            if wallet != ZERO_ADDRESS:
                rec.agent_wallet = wallet.lower()
        except Exception:
            pass
        time.sleep(RPC_DELAY)

        agents.append(rec)
        agent_id += 1

    print(f"[+] Enumerated {len(agents)} agents (skipped {skipped} empty IDs)")
    return agents


# ---------------------------------------------------------------------------
# Phase 3: URI Resolution
# ---------------------------------------------------------------------------

def _resolve_one_uri(uri: str) -> tuple[dict | None, str]:
    """Fetch a single URI and return (json_data, error_string)."""
    if not uri or not uri.strip():
        return None, "empty"

    url = uri.strip()

    # data: URI (base64 JSON, common in NFTs)
    if url.startswith("data:"):
        try:
            _, encoded = url.split(",", 1)
            decoded = base64.b64decode(encoded).decode("utf-8")
            return json.loads(decoded), ""
        except Exception as e:
            return None, f"data_uri_error: {str(e)[:60]}"

    # IPFS
    if url.startswith("ipfs://"):
        cid = url[7:]
        url = f"{IPFS_GATEWAY}{cid}"

    # Must be HTTP(S) at this point
    if not url.startswith("http"):
        return None, f"unsupported_scheme: {url[:40]}"

    try:
        resp = requests.get(
            url,
            timeout=URI_FETCH_TIMEOUT,
            headers={"Accept": "application/json", "User-Agent": "AHM-ERC8004-Scanner/1.0"},
        )
        resp.raise_for_status()
        return resp.json(), ""
    except requests.Timeout:
        return None, "timeout"
    except requests.HTTPError as e:
        return None, f"http_{e.response.status_code}"
    except requests.ConnectionError:
        return None, "connection_error"
    except json.JSONDecodeError:
        return None, "invalid_json"
    except Exception as e:
        return None, str(e)[:60]


def _extract_wallets_from_json(data: dict) -> list[str]:
    """Recursively search JSON for Ethereum addresses."""
    wallets = set()

    def _recurse(obj, depth=0):
        if depth > 10:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, str) and ETH_ADDR_RE.match(val):
                    wallets.add(val.lower())
                elif key.lower() in WALLET_KEYS and isinstance(val, str) and val.startswith("0x") and len(val) == 42:
                    wallets.add(val.lower())
                else:
                    _recurse(val, depth + 1)
            return
        if isinstance(obj, list):
            for item in obj:
                _recurse(item, depth + 1)

    _recurse(data)
    return list(wallets)


def _extract_name_from_json(data: dict) -> str:
    """Pull agent name from registration JSON."""
    for key in ("name", "title", "agent_name", "agentName"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:80]
    return ""


def resolve_uris(agents: list[AgentRecord]) -> None:
    """Fetch and parse agentURI registration documents."""
    print("\n" + "=" * 60)
    print("  PHASE 3: URI Resolution")
    print("=" * 60)

    resolved = 0
    errors_by_type = {}

    for i, agent in enumerate(agents):
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(agents)}] resolving URIs...")

        if not agent.token_uri:
            agent.uri_error = "no_uri"
            errors_by_type["no_uri"] = errors_by_type.get("no_uri", 0) + 1
            continue

        data, err = _resolve_one_uri(agent.token_uri)

        if err:
            agent.uri_error = err
            errors_by_type[err.split(":")[0]] = errors_by_type.get(err.split(":")[0], 0) + 1
        else:
            agent.uri_resolved = True
            agent.registration_json = data
            agent.agent_name = _extract_name_from_json(data)
            agent.extracted_wallets = _extract_wallets_from_json(data)
            resolved += 1

        time.sleep(URI_FETCH_DELAY)

    print(f"[+] Resolved: {resolved}/{len(agents)}")
    if errors_by_type:
        print(f"[+] Errors: {', '.join(f'{k}={v}' for k, v in sorted(errors_by_type.items(), key=lambda x: -x[1]))}")


# ---------------------------------------------------------------------------
# Phase 4: AHS Scanning
# ---------------------------------------------------------------------------

def scan_wallets(agents: list[AgentRecord], max_scans=100) -> dict:
    """Run AHS 2D scans on unique wallet addresses from the enumeration.

    Returns dict mapping address -> {ahs_score, grade, ...} for all scanned wallets.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: AHS Scanning")
    print("=" * 60)

    # Import AHM scanning functions
    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price
    import db

    db.init_db()

    # Collect unique wallet addresses with their source agent
    wallet_to_agents: dict[str, list[AgentRecord]] = {}
    for agent in agents:
        for addr in [agent.owner, agent.agent_wallet] + agent.extracted_wallets:
            if addr and addr != ZERO_ADDRESS:
                wallet_to_agents.setdefault(addr.lower(), []).append(agent)

    unique_wallets = list(wallet_to_agents.keys())
    print(f"[+] Unique wallets to scan: {len(unique_wallets)} (max {max_scans})")

    # Check which are already scanned today
    already_scanned = set()
    try:
        conn = db.get_connection()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT address FROM known_wallets WHERE last_scanned_at >= ?",
            (today + "T00:00:00Z",),
        ).fetchall()
        already_scanned = {row[0] for row in rows}
        conn.close()
    except Exception:
        pass

    if already_scanned:
        print(f"[+] Skipping {len(already_scanned)} wallets already scanned today")

    # Fetch ETH price once
    eth_price = get_eth_price()
    print(f"[+] ETH price: ${eth_price:,.2f}")

    results = {}
    scan_count = 0

    for i, address in enumerate(unique_wallets):
        if scan_count >= max_scans:
            print(f"  [stop] Reached max_scans={max_scans}")
            break

        if address in already_scanned:
            continue

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

            # Find a label from the agent records
            source_agents = wallet_to_agents.get(address, [])
            agent_id = source_agents[0].agent_id if source_agents else 0
            agent_name = next((a.agent_name for a in source_agents if a.agent_name), "")
            label = f"ERC-8004 #{agent_id}"
            if agent_name:
                label += f" — {agent_name}"

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.log_scan(
                address=address,
                endpoint="ahs",
                scan_timestamp=now_iso,
                source="erc8004_scan",
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

            # Map results back to agent records
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

def generate_report(agents: list[AgentRecord], max_id: int, scan_results: dict) -> None:
    """Print summary, write CSV and Markdown."""
    print("\n" + "=" * 60)
    print("  PHASE 5: Report")
    print("=" * 60)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Stats
    total_registered = max_id
    enumerated = len(agents)
    with_uri = sum(1 for a in agents if a.token_uri)
    resolved = sum(1 for a in agents if a.uri_resolved)
    with_agent_wallet = sum(1 for a in agents if a.agent_wallet)
    scanned = len(scan_results)
    scored = [r for r in scan_results.values() if r.get("ahs_score") is not None]

    # Grade distribution
    grade_dist = {}
    for r in scored:
        g = r["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    avg_ahs = sum(r["ahs_score"] for r in scored) / len(scored) if scored else 0
    scores = [r["ahs_score"] for r in scored]
    min_ahs = min(scores) if scores else 0
    max_ahs = max(scores) if scores else 0

    # Console summary
    print(f"\n  Total agents registered (max ID): {total_registered:,}")
    print(f"  Agents enumerated:                {enumerated}")
    print(f"  With tokenURI:                    {with_uri}")
    print(f"  URIs resolved:                    {resolved}")
    print(f"  With agent wallet (getAgentWallet): {with_agent_wallet}")
    print(f"  Unique wallets scanned (AHS):     {scanned}")
    if scored:
        print(f"  Average AHS:                      {avg_ahs:.1f}")
        print(f"  AHS range:                        {min_ahs}-{max_ahs}")
        print(f"  Grade distribution:               {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}")

    # -- CSV --
    fieldnames = [
        "agent_id", "owner", "agent_wallet", "token_uri", "uri_resolved",
        "name", "ahs", "grade", "grade_label", "d1", "d2",
        "confidence", "patterns", "tx_count", "history_days", "scan_error",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in agents:
            writer.writerow({
                "agent_id": a.agent_id,
                "owner": a.owner,
                "agent_wallet": a.agent_wallet,
                "token_uri": a.token_uri[:120] if a.token_uri else "",
                "uri_resolved": a.uri_resolved,
                "name": a.agent_name,
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
        f.write("# ERC-8004 Registry Scan Results\n\n")
        f.write(f"> Scanned: {now_str}\n")
        f.write(f"> Registry: `{REGISTRY_ADDRESS}` (Base mainnet)\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total agents registered (max ID):** {total_registered:,}\n")
        f.write(f"- **Agents enumerated:** {enumerated}\n")
        f.write(f"- **With tokenURI:** {with_uri}\n")
        f.write(f"- **URIs resolved:** {resolved}/{with_uri}\n")
        f.write(f"- **With agent wallet (getAgentWallet):** {with_agent_wallet}\n")
        f.write(f"- **Unique wallets scanned (AHS):** {scanned}\n")
        if scored:
            f.write(f"- **Average AHS:** {avg_ahs:.1f}\n")
            f.write(f"- **AHS range:** {min_ahs}-{max_ahs}\n")
            f.write(f"- **Grade distribution:** {', '.join(f'{g}={n}' for g, n in sorted(grade_dist.items()))}\n")

        # Grade breakdown table
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

        # Scored agents table
        scored_agents = [a for a in agents if a.ahs_score is not None]
        if scored_agents:
            scored_agents.sort(key=lambda a: a.ahs_score or 0, reverse=True)
            f.write("\n## Scanned Agents (by AHS)\n\n")
            f.write("| ID | Name | Owner | AHS | Grade | D1 | D2 | Patterns |\n")
            f.write("|----|------|-------|-----|-------|----|----|----------|\n")
            for a in scored_agents:
                owner_short = f"`{a.owner[:8]}...{a.owner[-4:]}`" if a.owner else ""
                name = a.agent_name[:30] if a.agent_name else f"Agent #{a.agent_id}"
                pats = a.patterns if a.patterns and a.patterns != "none" else ""
                f.write(f"| {a.agent_id} | {name} | {owner_short} | {a.ahs_score} | {a.grade} {a.grade_label} | {a.d1_score} | {a.d2_score} | {pats} |\n")

        # URI resolution stats
        uri_errors = {}
        for a in agents:
            if a.uri_error:
                key = a.uri_error.split(":")[0]
                uri_errors[key] = uri_errors.get(key, 0) + 1
        if uri_errors:
            f.write("\n## URI Resolution Errors\n\n")
            f.write("| Error | Count |\n")
            f.write("|-------|-------|\n")
            for err, cnt in sorted(uri_errors.items(), key=lambda x: -x[1]):
                f.write(f"| {err} | {cnt} |\n")

        # Pattern frequency
        pattern_counts = {}
        for a in scored_agents:
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
    parser = argparse.ArgumentParser(description="ERC-8004 Identity Registry Scan Spike")
    parser.add_argument("--max-agents", type=int, default=200, help="Max agents to enumerate (default: 200)")
    parser.add_argument("--max-scans", type=int, default=100, help="Max AHS scans to perform (default: 100)")
    parser.add_argument("--skip-scan", action="store_true", help="Enumerate + resolve only, no AHS scanning")
    parser.add_argument("--start-id", type=int, default=1, help="Start agent ID for enumeration (default: 1)")
    args = parser.parse_args()

    print("[*] ERC-8004 Identity Registry Scan Spike")
    print(f"[*] Registry: {REGISTRY_ADDRESS}")
    print(f"[*] RPC: {BASE_RPC}")
    print(f"[*] Max agents: {args.max_agents} | Max scans: {args.max_scans} | Start ID: {args.start_id}")

    # Phase 1
    w3, registry, info = discover_registry()

    # Phase 2
    print("\n" + "=" * 60)
    print("  PHASE 2: Agent Enumeration")
    print("=" * 60)
    max_id = find_max_agent_id(registry)
    agents = enumerate_agents(registry, max_agents=args.max_agents, start_id=args.start_id)

    # Phase 3
    resolve_uris(agents)

    # Phase 4
    scan_results = {}
    if not args.skip_scan:
        scan_results = scan_wallets(agents, max_scans=args.max_scans)
    else:
        print("\n[*] Skipping AHS scanning (--skip-scan)")

    # Phase 5
    generate_report(agents, max_id, scan_results)

    print("\n" + "=" * 60)
    print("  SCAN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
