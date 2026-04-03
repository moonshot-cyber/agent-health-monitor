#!/usr/bin/env python3
"""Arc Protocol (ERC-8004) agent registry scanner.

Discovers autonomous agent wallets from the ERC-8004 IdentityRegistry contract
on Arc testnet. Enumerates agents via Registered events, resolves agent wallet
addresses via getAgentWallet(), then runs AHS scoring on discovered wallets.

Data source: ERC-8004 IdentityRegistry on Arc testnet
  (0x8004A818BFB912233c491871b3d84c89A494BD9e)
Chain: Arc testnet (5042002)
RPC: https://rpc.testnet.arc.network

Usage:
    python arc_scan.py                    # Default: scan up to 200 wallets
    python arc_scan.py --max-scans 50     # Limit wallet scans
    python arc_scan.py --skip-scan        # Discovery only, no AHS scoring
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime, timezone
from functools import partial

logger = logging.getLogger("ahm.arc_scan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARC_RPC_URL = os.getenv("ARC_RPC_URL", "https://rpc.testnet.arc.network")

# ERC-8004 IdentityRegistry on Arc testnet (chain 5042002)
IDENTITY_REGISTRY_ADDRESS = "0x8004A818BFB912233c491871b3d84c89A494BD9e"

# Minimal ABI — only the entries we need for discovery + wallet resolution.
# The upgradeable IdentityRegistry does NOT expose totalSupply(); we discover
# agents by indexing Registered events instead.
IDENTITY_REGISTRY_ABI = [
    {
        "name": "ownerOf",
        "type": "function",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "name": "tokenURI",
        "type": "function",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
    },
    {
        "name": "getAgentWallet",
        "type": "function",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    {
        "anonymous": False,
        "name": "Registered",
        "type": "event",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "agentURI", "type": "string", "indexed": False},
            {"name": "owner", "type": "address", "indexed": True},
        ],
    },
]

# RPC rate limiting and retry
RPC_DELAY = 0.3  # seconds between calls — Arc testnet can be slow
RPC_MAX_RETRIES = 4
RPC_BACKOFF_BASE = 2.0

ZERO_ADDRESS = "0x" + "0" * 40


# ---------------------------------------------------------------------------
# Discovery: scan the on-chain registry via Registered events
# ---------------------------------------------------------------------------

def _get_contract():
    """Lazy-initialise Web3 and return the IdentityRegistry contract."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(
        ARC_RPC_URL,
        request_kwargs={"timeout": 30},
    ))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Arc RPC at {ARC_RPC_URL}")
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(IDENTITY_REGISTRY_ADDRESS),
        abi=IDENTITY_REGISTRY_ABI,
    )
    return w3, contract


def _rpc_call_with_retry(fn, *args, label: str = "RPC call"):
    """Execute a Web3 contract call with exponential backoff on 429/timeouts."""
    last_exc = None
    for attempt in range(RPC_MAX_RETRIES + 1):
        try:
            return fn.call()
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "too many" in err_str or "rate" in err_str
            is_timeout = "timeout" in err_str or "timed out" in err_str

            if not (is_rate_limit or is_timeout) or attempt == RPC_MAX_RETRIES:
                raise

            delay = RPC_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "%s attempt %d/%d failed (429/timeout), retrying in %.1fs: %s",
                label, attempt + 1, RPC_MAX_RETRIES + 1, delay, str(e)[:80],
            )
            time.sleep(delay)

    raise last_exc  # unreachable, but keeps type checkers happy


def _fetch_registered_events(w3, contract, from_block: int = 0) -> list[dict]:
    """Fetch all Registered events from the IdentityRegistry.

    Uses batch log queries. Falls back to incremental probing if
    the RPC doesn't support large block ranges.
    """
    latest_block = w3.eth.block_number
    logger.info("Arc discovery: fetching Registered events from block %d to %d", from_block, latest_block)

    all_events = []
    # Try full range first; some RPCs limit range to ~10k blocks
    chunk_size = latest_block - from_block + 1
    start = from_block

    while start <= latest_block:
        end = min(start + chunk_size - 1, latest_block)
        try:
            event_filter = contract.events.Registered.create_filter(
                from_block=start, to_block=end,
            )
            events = event_filter.get_all_entries()
            all_events.extend(events)
            start = end + 1
        except Exception as e:
            err_str = str(e).lower()
            if "range" in err_str or "too many" in err_str or "limit" in err_str:
                # Reduce chunk size and retry
                chunk_size = max(1000, chunk_size // 4)
                logger.info("Arc discovery: reducing chunk size to %d blocks", chunk_size)
                continue
            raise

    logger.info("Arc discovery: found %d Registered events", len(all_events))
    return all_events


def discover_arc_wallets(max_agents: int = 500) -> list[dict]:
    """Discover wallet addresses from the ERC-8004 IdentityRegistry on Arc testnet.

    Enumerates agents via Registered events, then calls getAgentWallet()
    for each to resolve operational wallet addresses. Falls back to the
    owner address if no agent wallet is set.

    Returns list of dicts:
        {address, source ("arc_agent_wallet" | "arc_owner"), metadata}
    """
    logger.info("Arc discovery — connecting to Arc testnet RPC")
    w3, contract = _get_contract()

    chain_id = w3.eth.chain_id
    logger.info("Arc IdentityRegistry connected (chain=%d, rpc=%s)", chain_id, ARC_RPC_URL)

    # Discover agents via Registered events
    events = _fetch_registered_events(w3, contract)

    if not events:
        logger.warning("No Registered events found on Arc IdentityRegistry")
        return []

    wallets: list[dict] = []
    seen_addresses: set[str] = set()
    error_count = 0

    # Process events, newest first (highest agentId = most recent)
    events_sorted = sorted(events, key=lambda e: e["args"]["agentId"], reverse=True)

    for i, event in enumerate(events_sorted):
        if len(wallets) >= max_agents:
            break

        agent_id = event["args"]["agentId"]
        owner = event["args"]["owner"]
        agent_uri = event["args"].get("agentURI", "")

        try:
            # Try to get the dedicated agent wallet
            agent_wallet = _rpc_call_with_retry(
                contract.functions.getAgentWallet(agent_id),
                label=f"getAgentWallet({agent_id})",
            )
            agent_wallet_lower = agent_wallet.lower() if agent_wallet else ""

            if agent_wallet_lower and agent_wallet_lower != ZERO_ADDRESS:
                # Agent has a dedicated operational wallet
                if agent_wallet_lower not in seen_addresses:
                    seen_addresses.add(agent_wallet_lower)
                    wallets.append({
                        "address": agent_wallet_lower,
                        "source": "arc_agent_wallet",
                        "metadata": {
                            "agent_id": agent_id,
                            "type": "agent_wallet",
                            "owner": owner.lower(),
                            "agent_uri": agent_uri[:200] if agent_uri else "",
                        },
                    })
            else:
                # No agent wallet set — use owner address
                owner_lower = owner.lower()
                if owner_lower and owner_lower != ZERO_ADDRESS and owner_lower not in seen_addresses:
                    seen_addresses.add(owner_lower)
                    wallets.append({
                        "address": owner_lower,
                        "source": "arc_owner",
                        "metadata": {
                            "agent_id": agent_id,
                            "type": "owner",
                            "agent_uri": agent_uri[:200] if agent_uri else "",
                        },
                    })

            if (i + 1) % 20 == 0:
                logger.info(
                    "  progress: processed %d/%d agents, %d wallets found",
                    i + 1, len(events_sorted), len(wallets),
                )

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                logger.warning("Error reading agent %d: %s", agent_id, e)
            elif error_count == 6:
                logger.warning("Suppressing further RPC errors...")
            if error_count > 20:
                logger.error("Too many RPC errors (%d), stopping discovery", error_count)
                break

        time.sleep(RPC_DELAY)

    logger.info(
        "Arc discovery complete: %d Registered events, %d unique wallets, %d errors",
        len(events), len(wallets), error_count,
    )
    return wallets


# ---------------------------------------------------------------------------
# Scan: run AHS scoring on discovered wallets
# ---------------------------------------------------------------------------

def get_already_scanned_arc_addresses() -> set[str]:
    """Get wallet addresses already in the database from previous Arc scans."""
    import db
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT address FROM known_wallets WHERE source IN ('arc_agent_wallet', 'arc_owner')"
        ).fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        return set()


def scan_arc_agents(max_scans: int = 200) -> list[dict]:
    """Discover Arc agents and run AHS scans on them.

    This is the main entry point called by the nightly pipeline.

    Args:
        max_scans: Maximum number of wallets to AHS-score in this run.

    Returns:
        List of wallet dicts that were discovered (all of them, not just scored).
    """
    import db

    logger.info(
        "=== Arc Protocol Scan — ENTERED (max_scans=%d, rpc=%s) ===",
        max_scans, ARC_RPC_URL,
    )
    start_time = time.time()

    # Phase 1: Discovery
    wallets = discover_arc_wallets(max_agents=max_scans * 3)

    if not wallets:
        logger.info("Arc scan: no wallets discovered")
        return []

    logger.info("Discovered %d Arc wallets (%d agent_wallet, %d owner)",
                len(wallets),
                sum(1 for w in wallets if w["source"] == "arc_agent_wallet"),
                sum(1 for w in wallets if w["source"] == "arc_owner"))

    # Phase 2: Dedup against already-scanned
    already_scanned = get_already_scanned_arc_addresses()
    new_wallets = [w for w in wallets if w["address"] not in already_scanned]
    logger.info("New wallets to scan: %d (skipping %d already scanned)",
                len(new_wallets), len(wallets) - len(new_wallets))

    # Phase 3: AHS scoring
    db.init_db()

    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price

    eth_price = get_eth_price()
    scan_count = 0
    error_count = 0
    scan_results: list[dict] = []

    for wallet in new_wallets:
        if scan_count >= max_scans:
            logger.info("Reached max_scans=%d, stopping", max_scans)
            break

        address = wallet["address"]
        source = wallet["source"]
        meta = wallet["metadata"]
        scan_count += 1

        logger.info("  [%d/%d] Scanning %s (source=%s, agent_id=%s)",
                     scan_count, min(max_scans, len(new_wallets)),
                     address[:12] + "...", source, meta.get("agent_id"))

        try:
            txs = fetch_transactions(address)
            time.sleep(2.0)  # rate limit for Blockscout
            tokens = fetch_tokens_v2(address, max_pages=3)
            time.sleep(2.0)

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

            label_type = "Wallet" if source == "arc_agent_wallet" else "Owner"
            label = f"Arc Agent #{meta.get('agent_id', '?')} ({label_type})"

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.log_scan(
                address=address,
                endpoint="ahs",
                scan_timestamp=now_iso,
                source=source,
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

            scan_results.append({
                "ahs_score": ahs.agent_health_score,
                "grade": ahs.grade,
                "d1_score": ahs.d1_score,
                "d2_score": ahs.d2_score,
            })

            logger.info("           AHS %d/%s | D1=%s D2=%s | %d txs",
                         ahs.agent_health_score, ahs.grade,
                         ahs.d1_score, ahs.d2_score, ahs.tx_count)

        except Exception as e:
            error_count += 1
            logger.warning("           Error scanning %s: %s", address[:12], str(e)[:80])

    elapsed = time.time() - start_time
    logger.info(
        "=== Arc Protocol Scan — COMPLETE (%.0fs) ===\n"
        "  Wallets discovered: %d | Scanned: %d | Errors: %d",
        elapsed, len(wallets), scan_count, error_count,
    )

    # Phase 4: Batch quality tracking
    try:
        scored = [r for r in scan_results if r.get("ahs_score") is not None]
        if scored:
            scores = [r["ahs_score"] for r in scored]
            grades: dict[str, int] = {}
            for r in scored:
                g = r.get("grade", "?")
                grades[g] = grades.get(g, 0) + 1
            d1_scores = [r["d1_score"] for r in scored if r.get("d1_score") is not None]
            d2_scores = [r["d2_score"] for r in scored if r.get("d2_score") is not None]

            db.log_batch_quality(
                source="arc",
                wallets_scanned=len(scored),
                average_ahs=round(sum(scores) / len(scores), 1),
                min_ahs=min(scores),
                max_ahs=max(scores),
                grade_distribution=grades,
                avg_d1=round(sum(d1_scores) / len(d1_scores), 1) if d1_scores else None,
                avg_d2=round(sum(d2_scores) / len(d2_scores), 1) if d2_scores else None,
            )
            logger.info("Batch quality logged: %d wallets, avg AHS %.1f",
                         len(scored), sum(scores) / len(scores))
    except Exception:
        logger.exception("Failed to log batch quality (non-fatal)")

    return wallets


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Arc Protocol (ERC-8004) Agent Scanner")
    parser.add_argument("--max-scans", type=int, default=200,
                        help="Max wallets to AHS-score (default: 200)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Discovery only, no AHS scoring")
    args = parser.parse_args()

    if args.skip_scan:
        wallets = discover_arc_wallets()
        print(f"\nDiscovered {len(wallets)} Arc agent wallets")
        for w in wallets[:20]:
            print(f"  {w['address'][:12]}... source={w['source']} "
                  f"agent_id={w['metadata'].get('agent_id')}")
        if len(wallets) > 20:
            print(f"  ... and {len(wallets) - 20} more")
    else:
        wallets = scan_arc_agents(max_scans=args.max_scans)
        print(f"\nArc scan complete: {len(wallets)} wallets discovered")


if __name__ == "__main__":
    main()
