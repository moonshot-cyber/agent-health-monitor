#!/usr/bin/env python3
"""
Standalone test: call Nansen's API DIRECTLY (api.nansen.ai) vs. the Corbits
proxy (nansen.api.corbits.dev) for three profiler endpoints.

Goal: determine whether Nansen exposes x402 payment support on their own
domain, or if x402 is only available through the Corbits gateway.

Endpoints tested:
  1. /profiler/address/labels        — works via Corbits (baseline)
  2. /profiler/address/counterparties — 500s via Corbits (key test)
  3. /profiler/address/pnl-summary    — untested / future

For each endpoint the script tries:
  A. Corbits proxy  (https://nansen.api.corbits.dev/api/beta/...)
  B. Nansen direct  (https://api.nansen.ai/api/v1/...)

All calls use the same x402 payer wallet (NANSEN_PAYER_PRIVATE_KEY env var,
USDC on Base) and are made sequentially to avoid nonce conflicts.

Usage:
    export NANSEN_PAYER_PRIVATE_KEY="0x..."
    python test_nansen_direct.py
"""

import asyncio
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("nansen_direct_test")

# ── URLs ────────────────────────────────────────────────────────────────────

CORBITS_BASE = "https://nansen.api.corbits.dev/api/beta"
NANSEN_DIRECT_BASE = "https://api.nansen.ai/api/v1"

ENDPOINTS: list[dict] = [
    {
        "name": "labels",
        "path": "/profiler/address/labels",
        # Corbits uses "parameters" wrapper; direct Nansen uses flat body
        "corbits_body": {
            "parameters": {"chain": "all", "address": "ADDR"},
            "pagination": {"page": 1, "recordsPerPage": 100},
        },
        "direct_body": {
            "address": "ADDR",
            "chain": "all",
            "pagination": {"page": 1, "per_page": 100},
        },
    },
    {
        "name": "counterparties",
        "path": "/profiler/address/counterparties",
        "corbits_body": {
            "parameters": {
                "chain": "all",
                "address": "ADDR",
                "date": {
                    "from": "2025-08-28T00:00:00Z",
                    "to": "2026-02-28T00:00:00Z",
                },
            },
            "pagination": {"page": 1, "recordsPerPage": 25},
        },
        "direct_body": {
            "address": "ADDR",
            "chain": "all",
            "date": {
                "from": "2025-08-28T00:00:00Z",
                "to": "2026-02-28T00:00:00Z",
            },
            "pagination": {"page": 1, "per_page": 25},
        },
    },
    {
        "name": "pnl-summary",
        "path": "/profiler/address/pnl-summary",
        "corbits_body": {
            "parameters": {
                "chain": "all",
                "address": "ADDR",
                "date": {
                    "from": "2025-08-28T00:00:00Z",
                    "to": "2026-02-28T00:00:00Z",
                },
            },
        },
        "direct_body": {
            "address": "ADDR",
            "chain": "all",
            "date": {
                "from": "2025-08-28T00:00:00Z",
                "to": "2026-02-28T00:00:00Z",
            },
        },
    },
]

TEST_ADDRESS = "0x464fc339add314932920d3e060745bd7ea3e92ad"
NETWORK = os.getenv("NETWORK", "eip155:8453")


def _inject_address(body: dict, address: str) -> dict:
    """Deep-copy body dict and replace 'ADDR' placeholder with real address."""
    raw = json.dumps(body).replace("ADDR", address)
    return json.loads(raw)


def _dump_response(resp) -> None:
    """Log full response: status, headers, body."""
    log.info("  Status:  %d", resp.status_code)
    log.info("  Headers:")
    for k, v in resp.headers.items():
        log.info("    %s: %s", k, v)
    try:
        data = resp.json()
        pretty = json.dumps(data, indent=2)
        # Truncate very long bodies but show enough to be useful
        if len(pretty) > 3000:
            log.info("  Body (first 3000 chars):\n%s\n  ... [truncated, %d total chars]",
                     pretty[:3000], len(pretty))
        else:
            log.info("  Body:\n%s", pretty)
    except Exception:
        text = resp.text[:3000]
        log.info("  Body (raw):\n%s", text)


async def main() -> None:
    private_key = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
    if not private_key:
        log.error("Set NANSEN_PAYER_PRIVATE_KEY env var to a Base-funded wallet private key")
        sys.exit(1)

    # ── Setup x402 payer ────────────────────────────────────────────────────
    from eth_account import Account
    from x402 import x402Client as x402PayerClient
    from x402.http.clients.httpx import x402HttpxClient
    from x402.mechanisms.evm.exact import register_exact_evm_client
    from x402.mechanisms.evm.signers import EthAccountSigner

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    client = x402PayerClient()
    register_exact_evm_client(client, signer, networks=[NETWORK])
    log.info("x402 payer initialized — address: %s, network: %s", account.address, NETWORK)

    import httpx

    results: list[dict] = []

    for ep in ENDPOINTS:
        ep_name = ep["name"]
        ep_path = ep["path"]

        # ── A. Corbits proxy ────────────────────────────────────────────────
        corbits_url = CORBITS_BASE + ep_path
        corbits_body = _inject_address(ep["corbits_body"], TEST_ADDRESS)

        log.info("")
        log.info("=" * 78)
        log.info("ENDPOINT: %s  |  TARGET: Corbits proxy", ep_name)
        log.info("  URL:  %s", corbits_url)
        log.info("  Body: %s", json.dumps(corbits_body, indent=2))
        log.info("-" * 78)

        t0 = time.monotonic()
        try:
            async with x402HttpxClient(client, timeout=httpx.Timeout(45.0)) as http:
                resp = await http.post(corbits_url, json=corbits_body)
            elapsed = time.monotonic() - t0
            log.info("  Elapsed: %.2fs", elapsed)
            _dump_response(resp)
            results.append({
                "endpoint": ep_name,
                "target": "corbits",
                "status": resp.status_code,
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  EXCEPTION after %.2fs: %s", elapsed, e, exc_info=True)
            results.append({
                "endpoint": ep_name,
                "target": "corbits",
                "status": "exception",
                "error": str(e),
                "elapsed": round(elapsed, 2),
            })

        # ── B. Nansen direct ────────────────────────────────────────────────
        direct_url = NANSEN_DIRECT_BASE + ep_path
        direct_body = _inject_address(ep["direct_body"], TEST_ADDRESS)

        log.info("")
        log.info("=" * 78)
        log.info("ENDPOINT: %s  |  TARGET: Nansen DIRECT", ep_name)
        log.info("  URL:  %s", direct_url)
        log.info("  Body: %s", json.dumps(direct_body, indent=2))
        log.info("-" * 78)

        # First try with x402 client (in case Nansen serves 402 directly)
        t0 = time.monotonic()
        try:
            async with x402HttpxClient(client, timeout=httpx.Timeout(45.0)) as http:
                resp = await http.post(direct_url, json=direct_body)
            elapsed = time.monotonic() - t0
            log.info("  [x402 client] Elapsed: %.2fs", elapsed)
            _dump_response(resp)
            results.append({
                "endpoint": ep_name,
                "target": "nansen_direct_x402",
                "status": resp.status_code,
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  [x402 client] EXCEPTION after %.2fs: %s", elapsed, e, exc_info=True)
            results.append({
                "endpoint": ep_name,
                "target": "nansen_direct_x402",
                "status": "exception",
                "error": str(e),
                "elapsed": round(elapsed, 2),
            })

        # Also try a raw request (no x402) to see what Nansen returns
        # (e.g. 401/403 asking for API key, or 402 for x402)
        log.info("")
        log.info("  --- Raw request (no x402, no API key) ---")
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as raw_http:
                raw_resp = await raw_http.post(direct_url, json=direct_body)
            elapsed = time.monotonic() - t0
            log.info("  [raw] Elapsed: %.2fs", elapsed)
            _dump_response(raw_resp)
            results.append({
                "endpoint": ep_name,
                "target": "nansen_direct_raw",
                "status": raw_resp.status_code,
                "elapsed": round(elapsed, 2),
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  [raw] EXCEPTION after %.2fs: %s", elapsed, e, exc_info=True)
            results.append({
                "endpoint": ep_name,
                "target": "nansen_direct_raw",
                "status": "exception",
                "error": str(e),
                "elapsed": round(elapsed, 2),
            })

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("SUMMARY")
    log.info("=" * 78)
    log.info("%-20s %-25s %-10s %s", "Endpoint", "Target", "Status", "Elapsed")
    log.info("-" * 78)
    for r in results:
        log.info(
            "%-20s %-25s %-10s %.2fs",
            r["endpoint"],
            r["target"],
            r["status"],
            r["elapsed"],
        )
    log.info("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
