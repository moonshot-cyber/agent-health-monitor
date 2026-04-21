#!/usr/bin/env python3
"""Olas Protocol (Autonolas) service registry scanner.

Discovers autonomous agent wallets from the Olas ServiceRegistryL2 contract
on Base mainnet. Extracts multisig Safe addresses and agent instance EOAs
from deployed services, then stores them in the AHM scan history database.

Data source: ServiceRegistryL2 on Base (0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE)
Chain: Base mainnet (8453)
RPC: https://mainnet.base.org (public, no key needed)

Usage:
    python olas_scan.py                    # Default: scan up to 200 wallets
    python olas_scan.py --max-scans 50     # Limit wallet scans
    python olas_scan.py --skip-scan        # Discovery only, no AHS scoring
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from functools import partial

logger = logging.getLogger("ahm.olas_scan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_RPC_URL = os.getenv("OLAS_RPC_URL", os.getenv("BASE_RPC_URL", "https://mainnet.base.org"))

# ServiceRegistryL2 on Base mainnet (chain 8453)
# Source: valory-xyz/autonolas-registries configuration.json
SERVICE_REGISTRY_ADDRESS = "0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"

# Service states in the Olas registry
SERVICE_STATE_DEPLOYED = 4

# Minimal ABI for read-only calls.
# Source: Basescan verified contract ABI for 0x3C1fF68f...on Base.
# getService returns a Service struct with 8 fields (including agentIds[]).
# getAgentInstances returns (uint256, address[]).
SERVICE_REGISTRY_ABI = [
    {
        "name": "totalSupply",
        "type": "function",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "name": "getService",
        "type": "function",
        "inputs": [{"name": "serviceId", "type": "uint256"}],
        "outputs": [
            {
                "name": "service",
                "type": "tuple",
                "components": [
                    {"name": "securityDeposit", "type": "uint96"},
                    {"name": "multisig", "type": "address"},
                    {"name": "configHash", "type": "bytes32"},
                    {"name": "threshold", "type": "uint32"},
                    {"name": "maxNumAgentInstances", "type": "uint32"},
                    {"name": "numAgentInstances", "type": "uint32"},
                    {"name": "state", "type": "uint8"},
                    {"name": "agentIds", "type": "uint32[]"},
                ],
            }
        ],
        "stateMutability": "view",
    },
    {
        "name": "getAgentInstances",
        "type": "function",
        "inputs": [{"name": "serviceId", "type": "uint256"}],
        "outputs": [
            {"name": "numAgentInstances", "type": "uint256"},
            {"name": "agentInstances", "type": "address[]"},
        ],
        "stateMutability": "view",
    },
]

# RPC rate limiting and retry
RPC_DELAY = 0.25  # seconds between calls to avoid rate limits
RPC_MAX_RETRIES = 4  # max retries per RPC call on 429/timeout
RPC_BACKOFF_BASE = 2.0  # exponential backoff base (seconds)


# ---------------------------------------------------------------------------
# Discovery: scan the on-chain registry
# ---------------------------------------------------------------------------

def _get_contract():
    """Lazy-initialise Web3 and return the ServiceRegistryL2 contract."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(
        BASE_RPC_URL,
        request_kwargs={"timeout": 30},
    ))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Base RPC at {BASE_RPC_URL}")
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(SERVICE_REGISTRY_ADDRESS),
        abi=SERVICE_REGISTRY_ABI,
    )
    return w3, contract


def _rpc_call_with_retry(fn, *args, label: str = "RPC call"):
    """Execute a Web3 contract call with exponential backoff on 429/timeouts.

    Args:
        fn: Bound contract function (e.g. contract.functions.getService(1))
        label: Human-readable label for log messages.

    Returns:
        The call result.
    """
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

            # Exponential backoff with jitter
            delay = RPC_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "%s attempt %d/%d failed (429/timeout), retrying in %.1fs: %s",
                label, attempt + 1, RPC_MAX_RETRIES + 1, delay, str(e)[:80],
            )
            time.sleep(delay)

    raise last_exc  # unreachable, but keeps type checkers happy


def fetch_registry_total_supply() -> int:
    """Return the live service count from the Olas ServiceRegistryL2.

    Thin read-only wrapper around ``totalSupply()`` on the registry
    contract — isolated from ``discover_olas_wallets`` so callers that
    only need the count (e.g. the /olas-scan/status endpoint for
    dashboard saturation metrics) don't pay the cost of a full scan.

    Returns:
        Current total number of services registered (includes every
        lifecycle state, not just DEPLOYED).

    Raises:
        ConnectionError: if the Base RPC is unreachable.
        Exception: propagates any non-retryable RPC error from the
        underlying ``_rpc_call_with_retry`` (callers should catch and
        treat as unavailability rather than crashing the status
        endpoint).
    """
    _, contract = _get_contract()
    return int(_rpc_call_with_retry(
        contract.functions.totalSupply(),
        label="totalSupply",
    ))


def discover_olas_wallets(max_services: int | None = None) -> list[dict]:
    """Discover wallet addresses from Olas ServiceRegistryL2 on Base.

    Iterates from the highest service ID downward (newest first) and scans
    the entire registry from totalSupply down to service ID 1 by default,
    so coverage tracks registry growth automatically and no service is ever
    silently dropped.

    Args:
        max_services: Optional cap on the discovery window for testing or
            partial scans. If None (the default), the full registry is
            walked end-to-end. Production callers should leave this None.

    Only includes services in DEPLOYED state (state == 4).

    Returns list of dicts:
        {address, source ("olas_service" | "olas_instance"), metadata}
    """
    logger.info("Olas discovery — connecting to Base RPC")
    w3, contract = _get_contract()

    total_supply = _rpc_call_with_retry(
        contract.functions.totalSupply(), label="totalSupply",
    )
    logger.info("Olas ServiceRegistryL2 totalSupply: %d", total_supply)

    if total_supply == 0:
        logger.warning("No services found in the registry")
        return []

    wallets: list[dict] = []
    seen_addresses: set[str] = set()
    deployed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate from highest ID (newest) downward. Default: full registry walk
    # (totalSupply → 1) so coverage scales with registry growth. Optional
    # max_services caps the window for tests or partial scans.
    start_id = total_supply
    if max_services is None:
        end_id = 1
    else:
        end_id = max(1, total_supply - max_services + 1)

    for service_id in range(start_id, end_id - 1, -1):
        try:
            service = _rpc_call_with_retry(
                contract.functions.getService(service_id),
                label=f"getService({service_id})",
            )
            state = service[6]  # state field (index 6 in the 8-field struct)

            if state != SERVICE_STATE_DEPLOYED:
                skipped_count += 1
                continue

            deployed_count += 1
            multisig = service[1].lower()  # multisig address
            num_instances = service[5]

            # Add the multisig Safe address
            if multisig and multisig != "0x" + "0" * 40 and multisig not in seen_addresses:
                seen_addresses.add(multisig)
                wallets.append({
                    "address": multisig,
                    "source": "olas_service",
                    "metadata": {
                        "service_id": service_id,
                        "type": "multisig",
                        "num_agent_instances": num_instances,
                    },
                })

            # Get agent instance EOA addresses
            time.sleep(RPC_DELAY)
            num_agents, agent_instances = _rpc_call_with_retry(
                contract.functions.getAgentInstances(service_id),
                label=f"getAgentInstances({service_id})",
            )

            for agent_addr in agent_instances:
                addr = agent_addr.lower()
                if addr and addr != "0x" + "0" * 40 and addr not in seen_addresses:
                    seen_addresses.add(addr)
                    wallets.append({
                        "address": addr,
                        "source": "olas_instance",
                        "metadata": {
                            "service_id": service_id,
                            "type": "agent_instance",
                            "parent_multisig": multisig,
                        },
                    })

            if deployed_count % 20 == 0:
                logger.info(
                    "  progress: checked %d services, %d deployed, %d wallets found",
                    start_id - service_id + 1, deployed_count, len(wallets),
                )

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                logger.warning("Error reading service %d: %s", service_id, e)
            elif error_count == 6:
                logger.warning("Suppressing further RPC errors...")
            if error_count > 20:
                logger.error("Too many RPC errors (%d), stopping discovery", error_count)
                break

        time.sleep(RPC_DELAY)

    logger.info(
        "Olas discovery complete: %d total services checked (IDs %d→%d), "
        "%d deployed, %d not-deployed (skipped), %d unique wallets, %d errors",
        start_id - end_id + 1, start_id, end_id,
        deployed_count, skipped_count, len(wallets), error_count,
    )
    return wallets


# ---------------------------------------------------------------------------
# Scan: run AHS scoring on discovered wallets
# ---------------------------------------------------------------------------

def get_already_scanned_olas_addresses() -> set[str]:
    """Get wallet addresses already in the database from previous Olas scans."""
    import db
    try:
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT address FROM known_wallets WHERE source IN ('olas_service', 'olas_instance')"
        ).fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        return set()


def scan_olas_services(max_scans: int = 200) -> list[dict]:
    """Discover Olas wallets and run AHS scans on them.

    This is the main entry point called by the nightly pipeline.

    Args:
        max_scans: Maximum number of wallets to AHS-score in this run.

    Returns:
        List of wallet dicts that were discovered (all of them, not just scored).
    """
    import db

    logger.info(
        "=== Olas Protocol Scan — ENTERED (max_scans=%d, rpc=%s) ===",
        max_scans, BASE_RPC_URL,
    )
    start_time = time.time()

    # Phase 1: Discovery — always walk the full registry so coverage tracks
    # registry growth. max_scans only caps how many *new* wallets get scored.
    wallets = discover_olas_wallets()

    if not wallets:
        logger.info("Olas scan: no wallets discovered")
        return []

    logger.info("Discovered %d Olas wallets (%d service, %d instance)",
                len(wallets),
                sum(1 for w in wallets if w["source"] == "olas_service"),
                sum(1 for w in wallets if w["source"] == "olas_instance"))

    # Phase 2: Dedup against already-scanned
    already_scanned = get_already_scanned_olas_addresses()
    new_wallets = [w for w in wallets if w["address"] not in already_scanned]
    skipped_already_scanned = len(wallets) - len(new_wallets)
    logger.info(
        "Coverage: %d wallets discovered | %d skipped (already scanned) | "
        "%d new to score (cap=%d)",
        len(wallets), skipped_already_scanned, len(new_wallets), max_scans,
    )

    # Phase 3: AHS scoring
    db.init_db()

    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price

    eth_price = get_eth_price()
    scan_count = 0
    error_count = 0
    scan_results: list[dict] = []  # collect per-wallet scores for batch quality

    for wallet in new_wallets:
        if scan_count >= max_scans:
            logger.info("Reached max_scans=%d, stopping", max_scans)
            break

        address = wallet["address"]
        source = wallet["source"]
        meta = wallet["metadata"]
        scan_count += 1

        logger.info("  [%d/%d] Scanning %s (source=%s, service_id=%s)",
                     scan_count, min(max_scans, len(new_wallets)),
                     address[:12] + "...", source, meta.get("service_id"))

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

            # Build patterns list
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

            label_type = "Service" if source == "olas_service" else "Instance"
            label = f"Olas {label_type} #{meta.get('service_id', '?')}"

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
                shadow_signals={
                    k: ahs._signals.get(k, d)
                    for k, d in [
                        ("session_continuity_score", None),
                        ("abrupt_sessions", 0),
                        ("budget_exhaustion_count", 0),
                        ("total_sessions", 0),
                        ("avg_session_length", 0.0),
                        ("shadow_patterns", []),
                    ]
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
        "=== Olas Protocol Scan — COMPLETE (%.0fs) ===\n"
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
                source="olas",
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

    parser = argparse.ArgumentParser(description="Olas Protocol Service Scanner")
    parser.add_argument("--max-scans", type=int, default=200,
                        help="Max wallets to AHS-score (default: 200)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Discovery only, no AHS scoring")
    args = parser.parse_args()

    if args.skip_scan:
        wallets = discover_olas_wallets()
        print(f"\nDiscovered {len(wallets)} Olas wallets")
        for w in wallets[:20]:
            print(f"  {w['address'][:12]}... source={w['source']} "
                  f"service_id={w['metadata'].get('service_id')}")
        if len(wallets) > 20:
            print(f"  ... and {len(wallets) - 20} more")
    else:
        wallets = scan_olas_services(max_scans=args.max_scans)
        print(f"\nOlas scan complete: {len(wallets)} wallets discovered")


if __name__ == "__main__":
    main()
