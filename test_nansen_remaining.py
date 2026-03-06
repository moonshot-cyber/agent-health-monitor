#!/usr/bin/env python3
"""
Test script: check if related-wallets and smart-money endpoints work
via direct Nansen API (api.nansen.ai) with x402 payment.

For each endpoint, tries:
  1. Raw request (no x402, no API key) — reveals 402 vs 401
  2. x402 client WITHOUT date field
  3. x402 client WITH date field (180-day lookback, YYYY-MM-DD format)

Sequential calls to avoid payer nonce conflicts.

Usage:
    export NANSEN_PAYER_PRIVATE_KEY="0x..."
    python test_nansen_remaining.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger("nansen_remaining")

NANSEN_DIRECT = "https://api.nansen.ai/api/v1"
TEST_ADDRESS = "0x464fc339add314932920d3e060745bd7ea3e92ad"
NETWORK = os.getenv("NETWORK", "eip155:8453")

now = datetime.now(timezone.utc)
DATE_FIELD = {
    "from": (now - timedelta(days=180)).strftime("%Y-%m-%d"),
    "to": now.strftime("%Y-%m-%d"),
}

ENDPOINTS = [
    {
        "name": "related-wallets",
        "path": "/profiler/address/related-wallets",
        "body_no_date": {
            "address": TEST_ADDRESS,
            "chain": "all",
        },
        "body_with_date": {
            "address": TEST_ADDRESS,
            "chain": "all",
            "date": DATE_FIELD,
        },
    },
    {
        "name": "smart-money",
        "path": "/profiler/address/smart-money",
        "body_no_date": {
            "address": TEST_ADDRESS,
            "chain": "all",
        },
        "body_with_date": {
            "address": TEST_ADDRESS,
            "chain": "all",
            "date": DATE_FIELD,
        },
    },
]


def dump_response(resp, prefix="") -> None:
    log.info("%sStatus: %d", prefix, resp.status_code)
    log.info("%sHeaders:", prefix)
    for k, v in resp.headers.items():
        if k.lower() in ("payment-required", "payment-response", "set-cookie"):
            log.info("%s  %s: %s...", prefix, k, v[:80])
        else:
            log.info("%s  %s: %s", prefix, k, v)
    try:
        data = resp.json()
        pretty = json.dumps(data, indent=2, ensure_ascii=True)
        if len(pretty) > 3000:
            log.info("%sBody (first 3000 chars):\n%s\n%s... [%d total]",
                     prefix, pretty[:3000], prefix, len(pretty))
        else:
            log.info("%sBody:\n%s", prefix, pretty)
    except Exception:
        log.info("%sBody (raw): %s", prefix, resp.text[:2000])


async def main() -> None:
    private_key = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
    if not private_key:
        log.error("Set NANSEN_PAYER_PRIVATE_KEY")
        sys.exit(1)

    import httpx
    from eth_account import Account
    from x402 import x402Client as x402PayerClient
    from x402.http.clients.httpx import x402HttpxClient
    from x402.mechanisms.evm.exact import register_exact_evm_client
    from x402.mechanisms.evm.signers import EthAccountSigner

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    client = x402PayerClient()
    register_exact_evm_client(client, signer, networks=[NETWORK])
    log.info("Payer: %s  Network: %s", account.address, NETWORK)

    results = []

    for ep in ENDPOINTS:
        url = NANSEN_DIRECT + ep["path"]

        # --- 1. Raw request (no x402) ---
        log.info("")
        log.info("=" * 78)
        log.info("%s — RAW (no x402, no API key)", ep["name"])
        log.info("  POST %s", url)
        log.info("  Body: %s", json.dumps(ep["body_no_date"]))
        log.info("-" * 78)
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as raw:
                resp = await raw.post(url, json=ep["body_no_date"])
            elapsed = time.monotonic() - t0
            dump_response(resp, "  ")
            results.append({"ep": ep["name"], "variant": "raw", "status": resp.status_code, "time": f"{elapsed:.2f}s"})
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  EXCEPTION: %s", e)
            results.append({"ep": ep["name"], "variant": "raw", "status": "error", "time": f"{elapsed:.2f}s"})

        # --- 2. x402 WITHOUT date ---
        log.info("")
        log.info("=" * 78)
        log.info("%s — x402 WITHOUT date", ep["name"])
        log.info("  POST %s", url)
        log.info("  Body: %s", json.dumps(ep["body_no_date"]))
        log.info("-" * 78)
        t0 = time.monotonic()
        try:
            async with x402HttpxClient(client, timeout=httpx.Timeout(45.0)) as http:
                resp = await http.post(url, json=ep["body_no_date"])
            elapsed = time.monotonic() - t0
            dump_response(resp, "  ")
            results.append({"ep": ep["name"], "variant": "x402_no_date", "status": resp.status_code, "time": f"{elapsed:.2f}s"})
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  EXCEPTION: %s", e)
            results.append({"ep": ep["name"], "variant": "x402_no_date", "status": "error", "time": f"{elapsed:.2f}s"})

        # --- 3. x402 WITH date ---
        log.info("")
        log.info("=" * 78)
        log.info("%s — x402 WITH date", ep["name"])
        log.info("  POST %s", url)
        log.info("  Body: %s", json.dumps(ep["body_with_date"]))
        log.info("-" * 78)
        t0 = time.monotonic()
        try:
            async with x402HttpxClient(client, timeout=httpx.Timeout(45.0)) as http:
                resp = await http.post(url, json=ep["body_with_date"])
            elapsed = time.monotonic() - t0
            dump_response(resp, "  ")
            results.append({"ep": ep["name"], "variant": "x402_with_date", "status": resp.status_code, "time": f"{elapsed:.2f}s"})
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error("  EXCEPTION: %s", e)
            results.append({"ep": ep["name"], "variant": "x402_with_date", "status": "error", "time": f"{elapsed:.2f}s"})

    # --- Summary ---
    log.info("")
    log.info("=" * 78)
    log.info("SUMMARY")
    log.info("=" * 78)
    log.info("%-20s %-20s %-10s %s", "Endpoint", "Variant", "Status", "Time")
    log.info("-" * 78)
    for r in results:
        log.info("%-20s %-20s %-10s %s", r["ep"], r["variant"], r["status"], r["time"])
    log.info("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
