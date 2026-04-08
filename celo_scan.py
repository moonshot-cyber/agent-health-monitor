#!/usr/bin/env python3
"""Celo ERC-8004 agent registry scanner.

Discovers autonomous agent wallets from the ERC-8004 IdentityRegistry contract
on Celo mainnet. Enumerates agents via Registered events, resolves agent wallet
addresses via getAgentWallet(), then runs AHS scoring on discovered wallets.

Data source: ERC-8004 IdentityRegistry on Celo mainnet
  (0x8004A169FB4a3325136EB29fA0ceB6D2e539a432)
Chain: Celo mainnet (42220)
RPC: https://forno.celo.org

The Celo deployment is the same canonical ERC-8004 IdentityRegistry as Base
mainnet — the contract is deployed via deterministic CREATE2 to the same
address across chains. The implementation is an upgradeable ERC1967 proxy
("8004: Identity Registry" / AgentIdentity AGENT) verified on Celoscan.

Usage:
    python celo_scan.py                    # Default: scan up to 200 wallets
    python celo_scan.py --max-scans 50     # Limit wallet scans
    python celo_scan.py --skip-scan        # Discovery only, no AHS scoring
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("ahm.celo_scan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CELO_RPC_URL = os.getenv("CELO_RPC_URL", "https://forno.celo.org")

# ERC-8004 IdentityRegistry on Celo mainnet (chain 42220).
# Verified on Celoscan as "8004: Identity Registry" — ERC1967Proxy delegating to
# IdentityRegistryUpgradeable. Address confirmed against the official
# erc-8004/erc-8004-contracts repository README.
IDENTITY_REGISTRY_ADDRESS = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"

CELO_CHAIN_ID = 42220

# Minimal ABI — only the entries we need for discovery + wallet resolution.
# Identical to the Arc testnet deployment because both chains share the same
# canonical ERC-8004 IdentityRegistry implementation.
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

# RPC rate limiting and retry — Forno is generally responsive but we keep
# the same conservative defaults as Arc to avoid 429s on shared public nodes.
RPC_DELAY = 0.3
RPC_MAX_RETRIES = 4
RPC_BACKOFF_BASE = 2.0

ZERO_ADDRESS = "0x" + "0" * 40

# Block checkpoint — avoids re-scanning from block 0 on every run.
CHECKPOINT_PATH = Path(os.getenv("CELO_CHECKPOINT_PATH", "celo_scan_checkpoint.json"))

# Blocks per eth_getLogs request. forno.celo.org is a shared public RPC and
# rejects wide ranges much sooner than Base or Arc — observed failures at
# 10,000 blocks. 500 keeps us comfortably below any documented limit while
# still scanning ~300k blocks of nightly delta in a few minutes. Override
# via CELO_EVENT_CHUNK_SIZE if forno relaxes its limits.
EVENT_CHUNK_SIZE = int(os.getenv("CELO_EVENT_CHUNK_SIZE", "500"))

# Floor for adaptive chunk-halving. Any chunk request that fails while
# chunk_size is still above this floor is halved and retried. Once we reach
# the floor and still fail, the exception propagates so the scan doesn't
# silently hang.
MIN_EVENT_CHUNK_SIZE = 50


# ---------------------------------------------------------------------------
# Block checkpoint persistence
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Load the last-scanned-block checkpoint from disk."""
    try:
        if CHECKPOINT_PATH.exists():
            data = json.loads(CHECKPOINT_PATH.read_text())
            logger.info("Loaded Celo checkpoint: last_scanned_block=%s", data.get("last_scanned_block"))
            return data
    except Exception:
        logger.warning("Failed to read Celo checkpoint, starting from block 0")
    return {}


def save_checkpoint(last_scanned_block: int, events_found: int) -> None:
    """Persist the highest block we have fetched events through."""
    data = {
        "last_scanned_block": last_scanned_block,
        "events_found": events_found,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        CHECKPOINT_PATH.write_text(json.dumps(data))
        logger.info("Saved Celo checkpoint: block %d", last_scanned_block)
    except Exception:
        logger.warning("Failed to save Celo checkpoint (non-fatal)")


# ---------------------------------------------------------------------------
# Discovery: scan the on-chain registry via Registered events
# ---------------------------------------------------------------------------

def _get_contract():
    """Lazy-initialise Web3 and return the IdentityRegistry contract."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(
        CELO_RPC_URL,
        request_kwargs={"timeout": 30},
    ))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Celo RPC at {CELO_RPC_URL}")
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
    """Fetch Registered events from the IdentityRegistry in adaptive chunks.

    Uses the stateless ``eth_getLogs`` JSON-RPC method via
    ``w3.eth.get_logs({...})`` rather than ``contract.events.X.create_filter``
    (which routes through the stateful ``eth_newFilter`` /
    ``eth_getFilterLogs`` pair). Many public RPCs — notably
    ``forno.celo.org`` — either don't persist filter state at all or expire
    it aggressively, returning ``-32000 filter not found`` on the follow-up
    lookup. ``eth_getLogs`` is a single stateless call that is widely
    supported and has no such pitfall.

    Chunks start at EVENT_CHUNK_SIZE blocks (default 500 — conservative for
    forno.celo.org's rate-limited public RPC). On any exception the current
    chunk is halved and retried down to MIN_EVENT_CHUNK_SIZE. Once the floor
    is reached, the underlying exception propagates so we don't silently
    mask genuine bugs.

    Each successful chunk logs its event count so future failures can be
    pinpointed from the logs alone.
    """
    from web3 import Web3

    latest_block = w3.eth.block_number
    total_range = latest_block - from_block
    logger.info(
        "Celo discovery: fetching Registered events blocks %d→%d (%d blocks, chunk=%d)",
        from_block, latest_block, total_range, EVENT_CHUNK_SIZE,
    )

    # Topic0 for Registered(uint256,string,address) — computed once per
    # invocation. Using Web3.to_hex ensures the "0x"-prefixed lowercase
    # string form that eth_getLogs expects, independent of hexbytes version.
    registered_topic = Web3.to_hex(
        Web3.keccak(text="Registered(uint256,string,address)")
    )
    registered_event = contract.events.Registered()

    all_events: list[dict] = []
    chunk_size = EVENT_CHUNK_SIZE
    start = from_block
    chunks_queried = 0

    while start <= latest_block:
        end = min(start + chunk_size - 1, latest_block)
        try:
            raw_logs = w3.eth.get_logs({
                "fromBlock": start,
                "toBlock": end,
                "address": contract.address,
                "topics": [registered_topic],
            })
            # Decode raw logs into the same AttributeDict shape that
            # get_all_entries() used to return (with .args.agentId etc.)
            # so every caller downstream keeps working unchanged.
            decoded = [registered_event.process_log(log) for log in raw_logs]
        except Exception as e:
            # Log the raw error verbosely — forno's error format is opaque
            # and the only way to diagnose future regressions is to see
            # the exact message that came back over the wire.
            logger.warning(
                "Celo discovery: chunk %d-%d (size=%d) failed: %s: %s",
                start, end, chunk_size, type(e).__name__, str(e)[:200],
            )
            if chunk_size > MIN_EVENT_CHUNK_SIZE:
                chunk_size = max(MIN_EVENT_CHUNK_SIZE, chunk_size // 2)
                logger.info(
                    "Celo discovery: halving chunk size to %d blocks and retrying",
                    chunk_size,
                )
                continue
            # At the floor and still failing — let the caller decide.
            logger.error(
                "Celo discovery: chunk size already at floor (%d), propagating error",
                MIN_EVENT_CHUNK_SIZE,
            )
            raise

        all_events.extend(decoded)
        logger.info(
            "Celo discovery: chunk %d-%d → %d events (total=%d)",
            start, end, len(decoded), len(all_events),
        )
        start = end + 1
        chunks_queried += 1

        if chunks_queried % 100 == 0:
            pct = (start - from_block) / max(total_range, 1) * 100
            logger.info(
                "  progress: %d/%d blocks (%.0f%%), %d events so far",
                start - from_block, total_range, pct, len(all_events),
            )

    logger.info(
        "Celo discovery: found %d Registered events in %d chunks",
        len(all_events), chunks_queried,
    )
    return all_events


def discover_celo_wallets(max_agents: int = 500) -> list[dict]:
    """Discover wallet addresses from the ERC-8004 IdentityRegistry on Celo mainnet.

    Enumerates agents via Registered events, then calls getAgentWallet()
    for each to resolve operational wallet addresses. Falls back to the
    owner address if no agent wallet is set.

    Returns list of dicts:
        {address, source ("celo_agent_wallet" | "celo_owner"), metadata}
    """
    logger.info("Celo discovery — connecting to Celo mainnet RPC")
    w3, contract = _get_contract()

    chain_id = w3.eth.chain_id
    logger.info("Celo IdentityRegistry connected (chain=%d, rpc=%s)", chain_id, CELO_RPC_URL)

    # Resume from last checkpoint (or block 0 on first run)
    checkpoint = load_checkpoint()
    from_block = checkpoint.get("last_scanned_block", 0)
    if from_block > 0:
        # Start one block after the last fully-scanned block
        from_block += 1
        logger.info("Resuming from checkpoint block %d", from_block)

    # Discover agents via Registered events
    latest_block = w3.eth.block_number
    events = _fetch_registered_events(w3, contract, from_block=from_block)

    # Persist the block we scanned through, even if 0 events found
    save_checkpoint(latest_block, len(events))

    if not events:
        logger.warning("No new Registered events found since block %d", from_block)
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
                        "source": "celo_agent_wallet",
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
                        "source": "celo_owner",
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
        "Celo discovery complete: %d Registered events, %d unique wallets, %d errors",
        len(events), len(wallets), error_count,
    )
    return wallets


# ---------------------------------------------------------------------------
# Scan: run AHS scoring on discovered wallets
# ---------------------------------------------------------------------------

def get_already_scanned_celo_addresses() -> set[str]:
    """Get wallet addresses already in the database from previous Celo scans."""
    import db
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT address FROM known_wallets WHERE source IN ('celo_agent_wallet', 'celo_owner')"
        ).fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        return set()


def scan_celo_agents(max_scans: int = 200) -> list[dict]:
    """Discover Celo agents and run AHS scans on them.

    This is the main entry point called by the nightly pipeline.

    Args:
        max_scans: Maximum number of wallets to AHS-score in this run.

    Returns:
        List of wallet dicts that were discovered (all of them, not just scored).
    """
    import db

    logger.info(
        "=== Celo ERC-8004 Scan — ENTERED (max_scans=%d, rpc=%s) ===",
        max_scans, CELO_RPC_URL,
    )
    start_time = time.time()

    # Phase 1: Discovery
    wallets = discover_celo_wallets(max_agents=max_scans * 3)

    if not wallets:
        logger.info("Celo scan: no wallets discovered")
        return []

    logger.info("Discovered %d Celo wallets (%d agent_wallet, %d owner)",
                len(wallets),
                sum(1 for w in wallets if w["source"] == "celo_agent_wallet"),
                sum(1 for w in wallets if w["source"] == "celo_owner"))

    # Phase 2: Dedup against already-scanned
    already_scanned = get_already_scanned_celo_addresses()
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

            label_type = "Wallet" if source == "celo_agent_wallet" else "Owner"
            label = f"Celo Agent #{meta.get('agent_id', '?')} ({label_type})"

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
        "=== Celo ERC-8004 Scan — COMPLETE (%.0fs) ===\n"
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
                source="celo",
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

    parser = argparse.ArgumentParser(description="Celo ERC-8004 Agent Scanner")
    parser.add_argument("--max-scans", type=int, default=200,
                        help="Max wallets to AHS-score (default: 200)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Discovery only, no AHS scoring")
    args = parser.parse_args()

    if args.skip_scan:
        wallets = discover_celo_wallets()
        print(f"\nDiscovered {len(wallets)} Celo agent wallets")
        for w in wallets[:20]:
            print(f"  {w['address'][:12]}... source={w['source']} "
                  f"agent_id={w['metadata'].get('agent_id')}")
        if len(wallets) > 20:
            print(f"  ... and {len(wallets) - 20} more")
    else:
        wallets = scan_celo_agents(max_scans=args.max_scans)
        print(f"\nCelo scan complete: {len(wallets)} wallets discovered")


if __name__ == "__main__":
    main()
