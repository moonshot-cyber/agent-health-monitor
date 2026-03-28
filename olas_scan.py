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
import sys
import time
from datetime import datetime, timezone
from functools import partial

logger = logging.getLogger("ahm.olas_scan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")

# ServiceRegistryL2 on Base mainnet (chain 8453)
# Source: valory-xyz/autonolas-registries configuration.json
SERVICE_REGISTRY_ADDRESS = "0x3C1fF68f5aa342D296d4DEe4Bb1cACCA912D95fE"

# Service states in the Olas registry
SERVICE_STATE_DEPLOYED = 4

# Minimal ABI for read-only calls
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
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "securityDeposit", "type": "uint96"},
                    {"name": "multisig", "type": "address"},
                    {"name": "configHash", "type": "bytes32"},
                    {"name": "threshold", "type": "uint32"},
                    {"name": "maxNumAgentInstances", "type": "uint32"},
                    {"name": "numAgentInstances", "type": "uint32"},
                    {"name": "state", "type": "uint8"},
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
            {"name": "numAgentInstances", "type": "uint32"},
            {"name": "agentInstances", "type": "address[]"},
        ],
        "stateMutability": "view",
    },
]

# Rate limiting for RPC calls
RPC_DELAY = 0.15  # seconds between calls to avoid rate limits


# ---------------------------------------------------------------------------
# Discovery: scan the on-chain registry
# ---------------------------------------------------------------------------

def _get_contract():
    """Lazy-initialise Web3 and return the ServiceRegistryL2 contract."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Base RPC at {BASE_RPC_URL}")
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(SERVICE_REGISTRY_ADDRESS),
        abi=SERVICE_REGISTRY_ABI,
    )
    return w3, contract


def discover_olas_wallets(max_services: int = 500) -> list[dict]:
    """Discover wallet addresses from Olas ServiceRegistryL2 on Base.

    Iterates from the highest service ID downward (newest first).
    Only includes services in DEPLOYED state (state == 4).

    Returns list of dicts:
        {address, source ("olas_service" | "olas_instance"), metadata}
    """
    logger.info("Olas discovery — connecting to Base RPC")
    w3, contract = _get_contract()

    total_supply = contract.functions.totalSupply().call()
    logger.info("Olas ServiceRegistryL2 totalSupply: %d", total_supply)

    if total_supply == 0:
        logger.warning("No services found in the registry")
        return []

    wallets: list[dict] = []
    seen_addresses: set[str] = set()
    deployed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate from highest ID (most recent) to 1
    start_id = total_supply
    end_id = max(1, total_supply - max_services * 3)  # overshoot to find enough deployed

    for service_id in range(start_id, end_id - 1, -1):
        try:
            service = contract.functions.getService(service_id).call()
            state = service[6]  # state is the last tuple element

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
            num_agents, agent_instances = contract.functions.getAgentInstances(service_id).call()

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
        "Olas discovery complete: %d total services checked, %d deployed, "
        "%d unique wallets, %d errors",
        start_id - end_id + 1, deployed_count, len(wallets), error_count,
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

    logger.info("=== Olas Protocol Scan — START ===")
    start_time = time.time()

    # Phase 1: Discovery
    wallets = discover_olas_wallets(max_services=max_scans * 3)

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
    logger.info("New wallets to scan: %d (skipping %d already scanned)",
                len(new_wallets), len(wallets) - len(new_wallets))

    # Phase 3: AHS scoring
    db.init_db()

    from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price

    eth_price = get_eth_price()
    scan_count = 0
    error_count = 0

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
            )

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
