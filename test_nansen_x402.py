#!/usr/bin/env python3
"""
Standalone test script: Prove the Nansen x402 payment flow works.

Loads a Base-funded wallet private key, creates an x402 client,
and makes a paid request to Nansen's address labels API.

Includes full debug logging at every step of the 402 → sign → retry flow.

Usage:
    export NANSEN_PAYER_PRIVATE_KEY="0x..."
    python test_nansen_x402.py
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("nansen_x402_test")

NANSEN_API_URL = "https://nansen.api.corbits.dev/api/beta/profiler/address/labels"
TEST_ADDRESS = "0x464fc339add314932920d3e060745bd7ea3e92ad"


async def main():
    private_key = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
    if not private_key:
        log.error("Set NANSEN_PAYER_PRIVATE_KEY env var to a Base-funded wallet private key")
        sys.exit(1)

    # ── Step 1: Set up signer ──────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 1: Setting up wallet and signer")
    log.info("=" * 70)

    from eth_account import Account

    account = Account.from_key(private_key)
    log.info("Payer address: %s", account.address)

    from x402.mechanisms.evm.signers import EthAccountSigner

    signer = EthAccountSigner(account)
    log.info("EthAccountSigner created")

    # ── Step 2: Create x402 client and register schemes ────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 2: Creating x402 client and registering schemes")
    log.info("=" * 70)

    from x402 import x402Client
    from x402.mechanisms.evm.exact import register_exact_evm_client

    client = x402Client()
    register_exact_evm_client(client, signer)  # wildcard eip155:* + all V1 networks
    log.info("Registered schemes:")
    for version, entries in client.get_registered_schemes().items():
        for e in entries:
            log.info("  V%d: network=%s scheme=%s", version, e["network"], e["scheme"])

    # ── Step 3: Raw 402 request to see what Nansen sends ───────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 3: Making initial request to Nansen (expect 402)")
    log.info("=" * 70)

    import httpx

    request_body = {
        "parameters": {"chain": "all", "address": TEST_ADDRESS},
        "pagination": {"page": 1, "recordsPerPage": 100},
    }
    log.info("URL: %s", NANSEN_API_URL)
    log.info("Body: %s", json.dumps(request_body))

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as raw_http:
        raw_response = await raw_http.post(NANSEN_API_URL, json=request_body)

    log.info("Response status: %d", raw_response.status_code)
    log.info("Response headers:")
    for k, v in raw_response.headers.items():
        log.info("  %s: %s", k, v)

    if raw_response.status_code != 402:
        log.error("Expected 402, got %d. Body: %s", raw_response.status_code, raw_response.text)
        sys.exit(1)

    raw_body = raw_response.json()
    log.info("402 body x402Version: %s", raw_body.get("x402Version"))
    log.info("Number of accepts: %d", len(raw_body.get("accepts", [])))

    for i, opt in enumerate(raw_body.get("accepts", [])):
        log.info(
            "  [%d] network=%s scheme=%s asset=%s payTo=%s amount=%s",
            i,
            opt.get("network"),
            opt.get("scheme"),
            opt.get("asset", "")[:20] + "...",
            opt.get("payTo", "")[:20] + "...",
            opt.get("maxAmountRequired"),
        )
        if opt.get("extra"):
            log.info("      extra=%s", json.dumps(opt["extra"]))

    # ── Step 4: Parse PaymentRequired using SDK ────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 4: Parsing 402 response with SDK")
    log.info("=" * 70)

    from x402.schemas.v1 import PaymentRequiredV1

    try:
        payment_required = PaymentRequiredV1.model_validate(raw_body)
        log.info("Parsed PaymentRequiredV1 with %d accepts", len(payment_required.accepts))
        for i, req in enumerate(payment_required.accepts):
            log.info(
                "  [%d] network=%s scheme=%s asset=%s payTo=%s amount=%s",
                i, req.network, req.scheme, req.asset[:20] + "...", req.pay_to[:20] + "...", req.max_amount_required,
            )
    except Exception as e:
        log.error("Failed to parse PaymentRequiredV1: %s", e)
        traceback.print_exc()
        sys.exit(1)

    # ── Step 5: Select payment option ──────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 5: Selecting payment option (SDK scheme matching)")
    log.info("=" * 70)

    from x402.schemas import find_schemes_by_network

    for req in payment_required.accepts:
        schemes = find_schemes_by_network(client._schemes_v1, req.network)
        match = "MATCH" if (schemes and req.scheme in schemes) else "no match"
        log.info("  network=%s scheme=%s → %s", req.network, req.scheme, match)

    # ── Step 6: Create payment payload ─────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 6: Creating payment payload (EIP-712 signing)")
    log.info("=" * 70)

    try:
        payment_payload = await client.create_payment_payload(payment_required)
        log.info("Payment payload created successfully!")
        log.info("  x402_version: %d", payment_payload.x402_version)
        log.info("  scheme: %s", payment_payload.scheme)
        log.info("  network: %s", payment_payload.network)

        inner = payment_payload.payload
        log.info("  authorization.from: %s", inner.get("authorization", {}).get("from"))
        log.info("  authorization.to: %s", inner.get("authorization", {}).get("to"))
        log.info("  authorization.value: %s", inner.get("authorization", {}).get("value"))
        log.info("  authorization.validAfter: %s", inner.get("authorization", {}).get("validAfter"))
        log.info("  authorization.validBefore: %s", inner.get("authorization", {}).get("validBefore"))
        log.info("  authorization.nonce: %s", inner.get("authorization", {}).get("nonce", "")[:20] + "...")
        sig = inner.get("signature", "")
        log.info("  signature: %s...%s (%d chars)", sig[:10], sig[-6:], len(sig))
    except Exception as e:
        log.error("Failed to create payment payload: %s", e)
        traceback.print_exc()
        sys.exit(1)

    # ── Step 7: Encode payment header ──────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 7: Encoding payment header")
    log.info("=" * 70)

    from x402.http.x402_http_client import x402HTTPClient

    http_client = x402HTTPClient(client)
    payment_headers = http_client.encode_payment_signature_header(payment_payload)
    for k, v in payment_headers.items():
        decoded = base64.b64decode(v).decode("utf-8")
        log.info("  Header: %s", k)
        log.info("  Value (first 200 chars): %s...", v[:200])
        log.info("  Decoded JSON (first 500 chars): %s...", decoded[:500])

    # ── Step 8: Retry request with payment ─────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 8: Retrying request with payment header")
    log.info("=" * 70)

    headers = {"Content-Type": "application/json"}
    headers.update(payment_headers)
    headers["Access-Control-Expose-Headers"] = "PAYMENT-RESPONSE,X-PAYMENT-RESPONSE"

    log.info("Request headers:")
    for k, v in headers.items():
        if k in ("X-PAYMENT", "PAYMENT-SIGNATURE"):
            log.info("  %s: %s... (%d chars)", k, v[:60], len(v))
        else:
            log.info("  %s: %s", k, v)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as retry_http:
        retry_response = await retry_http.post(
            NANSEN_API_URL,
            json=request_body,
            headers=headers,
        )

    log.info("Retry response status: %d", retry_response.status_code)
    log.info("Retry response headers:")
    for k, v in retry_response.headers.items():
        log.info("  %s: %s", k, v)

    if retry_response.status_code == 200:
        data = retry_response.json()
        log.info("")
        log.info("=" * 70)
        log.info("SUCCESS! Nansen returned data:")
        log.info("=" * 70)
        log.info("%s", json.dumps(data, indent=2)[:2000])
    else:
        log.error("Payment rejected! Status: %d", retry_response.status_code)
        log.error("Response body: %s", retry_response.text[:2000])

        # If still 402, decode and show what Nansen says
        if retry_response.status_code == 402:
            try:
                err_body = retry_response.json()
                log.error("Nansen 402 error details: %s", json.dumps(err_body, indent=2)[:1000])
            except Exception:
                pass

    # ── Step 9: Also test the SDK auto-flow end-to-end ─────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STEP 9: Testing SDK automatic 402 flow (x402HttpxClient)")
    log.info("=" * 70)

    from x402.http.clients.httpx import x402HttpxClient

    try:
        async with x402HttpxClient(client, timeout=httpx.Timeout(30.0)) as auto_http:
            auto_response = await auto_http.post(NANSEN_API_URL, json=request_body)

        log.info("Auto-flow response status: %d", auto_response.status_code)
        if auto_response.status_code == 200:
            log.info("Auto-flow SUCCESS!")
            log.info("Response: %s", auto_response.text[:500])
        else:
            log.error("Auto-flow returned %d: %s", auto_response.status_code, auto_response.text[:500])
    except Exception as e:
        log.error("Auto-flow exception: %s", e)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
