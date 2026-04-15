#!/usr/bin/env python3
"""Generate a named design-partner API key with expiry.

Usage:
    python scripts/create_partner_key.py

The script creates an enterprise-tier key with unlimited calls and a
fixed expiry date.  The raw key is printed ONCE — store it securely.

Environment:
    Set DB_PATH if the production database is not at the default location.
"""

import os
import sys

# Allow importing db module from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db as _db

# -- Configuration -----------------------------------------------------------

PARTNER_ID = "nevermined"
CUSTOMER_EMAIL = "don@nevermined.io"
TIER = "enterprise"
KEY_TYPE = "partner"
CALLS_TOTAL = None  # unlimited
EXPIRES_AT = "2026-07-15T00:00:00Z"  # ~3 months from 2026-04-15
IS_RESELLER = False
WHOLESALE_RATE = 0.0  # design partner — no wholesale billing

# -- Main --------------------------------------------------------------------


def main() -> None:
    _db.init_db()

    raw_key = _db.create_api_key(
        customer_email=CUSTOMER_EMAIL,
        key_type=KEY_TYPE,
        tier=TIER,
        calls_total=CALLS_TOTAL,
        partner_id=PARTNER_ID,
        is_reseller=IS_RESELLER,
        wholesale_rate=WHOLESALE_RATE,
        expires_at=EXPIRES_AT,
    )

    # Verify round-trip
    record = _db.validate_api_key(raw_key)
    assert record is not None, "Key validation failed immediately after creation"
    assert record["partner_id"] == PARTNER_ID
    assert record["expires_at"] == EXPIRES_AT
    assert record["calls_remaining"] is None  # unlimited

    print()
    print("=" * 64)
    print("  Nevermined Design Partner Key Created")
    print("=" * 64)
    print()
    print(f"  Partner:     {PARTNER_ID}")
    print(f"  Email:       {CUSTOMER_EMAIL}")
    print(f"  Tier:        {TIER}")
    print(f"  Calls:       unlimited")
    print(f"  Expires:     {EXPIRES_AT}")
    print(f"  Key prefix:  {record['key_prefix']}")
    print()
    print(f"  RAW KEY (store securely — shown only once):")
    print(f"  {raw_key}")
    print()
    print("  SDK integration:")
    print(f'  shield = AHMShield(api_key="{raw_key}", partner_id="{PARTNER_ID}")')
    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
