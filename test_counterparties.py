"""
Standalone test: call the Corbits Nansen counterparties proxy directly
using the same x402 client setup as the main app.

Tries multiple request body variations and logs the full response for each.

Usage:
    python test_counterparties.py
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

import httpx
from eth_account import Account as EthAccount
from x402 import x402Client as x402PayerClient
from x402.http.clients.httpx import x402HttpxClient
from x402.mechanisms.evm.exact import register_exact_evm_client
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

NANSEN_COUNTERPARTIES_URL = (
    "https://nansen.api.corbits.dev/api/beta/profiler/address/counterparties"
)
NETWORK = os.getenv("NETWORK", "eip155:8453")
NANSEN_PAYER_PRIVATE_KEY = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")

TEST_ADDRESS = "0x464fc339add314932920d3e060745bd7ea3e92ad"

# ── Request body variations to test ──────────────────────────────────────────

VARIATIONS: list[tuple[str, dict]] = [
    (
        "1) date INSIDE parameters (current app format)",
        {
            "parameters": {
                "chain": "all",
                "address": TEST_ADDRESS,
                "date": {
                    "from": "2025-08-28T00:00:00Z",
                    "to": "2026-02-28T00:00:00Z",
                },
            },
            "pagination": {"page": 1, "recordsPerPage": 25},
        },
    ),
    (
        "2) Flat body — date at top level (native Nansen format)",
        {
            "address": TEST_ADDRESS,
            "chain": "all",
            "date": {
                "from": "2025-08-28T00:00:00Z",
                "to": "2026-02-28T00:00:00Z",
            },
            "pagination": {"page": 1, "per_page": 25},
        },
    ),
    (
        "3) Flat body with recordsPerPage instead of per_page",
        {
            "address": TEST_ADDRESS,
            "chain": "all",
            "date": {
                "from": "2025-08-28T00:00:00Z",
                "to": "2026-02-28T00:00:00Z",
            },
            "pagination": {"page": 1, "recordsPerPage": 25},
        },
    ),
    (
        "4) parameters wrapper, NO date (original broken format)",
        {
            "parameters": {"chain": "all", "address": TEST_ADDRESS},
            "pagination": {"page": 1, "recordsPerPage": 25},
        },
    ),
    (
        "5) Flat body, NO date",
        {
            "address": TEST_ADDRESS,
            "chain": "all",
            "pagination": {"page": 1, "per_page": 25},
        },
    ),
]


async def run_test(client: x402PayerClient) -> None:
    for label, body in VARIATIONS:
        print("\n" + "=" * 78)
        print(f"TEST: {label}")
        print(f"POST {NANSEN_COUNTERPARTIES_URL}")
        print(f"Body: {json.dumps(body, indent=2)}")
        print("-" * 78)

        try:
            async with x402HttpxClient(
                client, timeout=httpx.Timeout(30.0)
            ) as http:
                resp = await http.post(NANSEN_COUNTERPARTIES_URL, json=body)

            print(f"Status:  {resp.status_code}")
            print(f"Headers: {dict(resp.headers)}")
            try:
                data = resp.json()
                print(f"Body:    {json.dumps(data, indent=2)[:2000]}")
            except Exception:
                print(f"Body (raw): {resp.text[:2000]}")

            if resp.status_code == 200:
                print(">>> SUCCESS")
            else:
                print(f">>> FAILED ({resp.status_code})")

        except Exception as e:
            print(f">>> EXCEPTION: {e}")

        print("=" * 78)


async def main() -> None:
    if not NANSEN_PAYER_PRIVATE_KEY:
        print("ERROR: NANSEN_PAYER_PRIVATE_KEY not set in .env")
        return

    account = EthAccount.from_key(NANSEN_PAYER_PRIVATE_KEY)
    signer = EthAccountSigner(account)
    client = x402PayerClient()
    register_exact_evm_client(client, signer, networks=[NETWORK])
    log.info("x402 client initialized (payer: %s)", account.address)

    await run_test(client)


if __name__ == "__main__":
    asyncio.run(main())
