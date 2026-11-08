#!/usr/bin/env python3
"""Prove the chain data source works before re-enabling scanning.

Run this after configuring BLOCKSCOUT_API_KEY and before setting
SCANS_ENABLED=true. It makes real calls and checks the things that silently
failed last time:

  - does the upstream answer at all, with JSON
  - does every field the scorers read actually arrive
  - do transactions actually come back for a wallet known to have them
  - does a real score come out, rather than the 58 / 75 / 50 no-data default
  - how many API credits a scan costs, against the daily budget

Exit code 0 means it is safe to resume scanning.

    python scripts/verify_chain_source.py
    python scripts/verify_chain_source.py --address 0x...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

import monitor  # noqa: E402

# Wallets AHM itself ranked highly before the outage, so they are known to have
# substantial transaction history. If these come back empty, the source is
# wrong - not the wallets.
KNOWN_ACTIVE = [
    "0x9c49c6d25fa949e2b4f3fabe287a6dbcc5310e80",
    "0xeaa0866bd5eea2ebf992be81fba399df469e6160",
    "0x81fabbc954e089c7b423a80f83715313f85dd939",
]

# Every transaction key the scoring code reads.
REQUIRED = ["hash", "from", "to", "timeStamp", "gas", "gasUsed", "gasPrice",
            "nonce", "isError", "txreceipt_status", "input", "value"]

NO_DATA_DEFAULT = (58, 75, 50)  # the signature of the October outage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", action="append", default=None)
    args = ap.parse_args()
    addresses = args.address or KNOWN_ACTIVE

    print("Chain data source verification\n")
    print(f"  endpoint : {monitor.BLOCKSCOUT_API_URL}/{monitor.BLOCKSCOUT_CHAIN_ID}"
          f"/api/v2/addresses/.../transactions")
    print(f"  api key  : {'set (' + str(len(monitor.BLOCKSCOUT_API_KEY)) + ' chars)' if monitor.BLOCKSCOUT_API_KEY else 'NOT SET'}")
    print(f"  page size: {monitor.BLOCKSCOUT_TX_PAGE_SIZE}\n")

    if not monitor.BLOCKSCOUT_API_KEY:
        print("  FAIL - BLOCKSCOUT_API_KEY is not set in this environment.")
        return 1

    failures: list[str] = []

    # -- credits, read from the response headers of one raw call -------------
    url = (f"{monitor.BLOCKSCOUT_API_URL.rstrip('/')}/{monitor.BLOCKSCOUT_CHAIN_ID}"
           f"/api/v2/addresses/{addresses[0]}/transactions")
    try:
        r = requests.get(
            url,
            params={"filter": "from", "items_count": monitor.BLOCKSCOUT_TX_PAGE_SIZE},
            headers={"accept": "application/json",
                     "authorization": f"Bearer {monitor.BLOCKSCOUT_API_KEY}"},
            timeout=30,
        )
        print(f"  HTTP {r.status_code}")
        for h in ("x-credits-remaining", "x-ratelimit-limit", "x-ratelimit-remaining"):
            if h in r.headers:
                print(f"  {h}: {r.headers[h]}")
        print()
    except requests.RequestException as e:
        print(f"  FAIL - could not reach the API: {e}")
        return 1

    # -- per-wallet checks ---------------------------------------------------
    for addr in addresses:
        short = addr[:12] + "..."
        try:
            txs = monitor.fetch_transactions(addr)
        except monitor.ChainDataUnavailable as e:
            print(f"  {short}  FAIL - {e}")
            failures.append(f"{short}: unavailable")
            continue

        if not txs:
            print(f"  {short}  FAIL - 0 transactions for a known-active wallet")
            failures.append(f"{short}: empty")
            continue

        missing = [k for k in REQUIRED if k not in txs[0]]
        blank = [k for k in ("hash", "from", "timeStamp") if not txs[0].get(k)]

        result = monitor.calculate_ahs(address=addr, tokens=[], transactions=txs,
                                       eth_price=2000.0)
        sig = (result.agent_health_score, result.d1_score, result.d2_score)
        is_default = sig == NO_DATA_DEFAULT

        status = "ok"
        if missing:
            status = f"MISSING FIELDS {missing}"
            failures.append(f"{short}: missing {missing}")
        elif blank:
            status = f"BLANK FIELDS {blank}"
            failures.append(f"{short}: blank {blank}")
        elif is_default:
            status = "SCORED THE NO-DATA DEFAULT (58/75/50)"
            failures.append(f"{short}: no-data default")

        print(f"  {short}  {len(txs):>4} txs   AHS {result.agent_health_score:>3} "
              f"({result.grade})  D1 {result.d1_score:>3}  D2 {result.d2_score:>3}   {status}")

    print()
    if failures:
        print(f"  FAIL - {len(failures)} problem(s):")
        for f in failures:
            print(f"    - {f}")
        print("\n  Do NOT set SCANS_ENABLED=true yet.")
        return 1

    print("  PASS - real transactions, all fields present, real scores.")
    print("  Safe to set SCANS_ENABLED=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
