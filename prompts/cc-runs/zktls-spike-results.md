# zkTLS Spike Session Results

**Date:** 2026-06-03
**Branch:** `spike/zktls-verification`
**PR:** https://github.com/moonshot-cyber/agent-health-monitor/pull/189
**Status:** Spike complete. PR open for review. DO NOT MERGE without review.

---

## Outcome

**Feasible.** AHM can verify Reclaim Protocol zkTLS proofs server-side
using existing dependencies (`web3`, `eth-account`) plus one new
pure-Python dep (`json-canonical`). No SDK needed — the underlying
crypto is standard EIP-191 signature recovery.

## What was built

| File | Lines | Purpose |
|------|-------|---------|
| `zktls.py` | ~200 | `verify_reclaim_proof()` + `apply_attestation_confidence()` |
| `tests/test_zktls.py` | ~280 | 32 tests: valid proofs, 5 tamper vectors, confidence rules |
| `docs/zktls_spike.md` | ~200 | Full spike findings document |

**Zero production changes.** No endpoints, no DB schema, no import
changes to existing modules.

## Key findings

1. **Verification path:** Manual EIP-191 recovery (Path 2). Official
   Reclaim Python SDK (v2.0.0) exists but fails to install due to
   `safe-pysha3` C extension. Node subprocess (Path 3) not needed.

2. **Attestor set:** 3 addresses, fetched from centralised Reclaim API.
   No on-chain governance. Acceptable for v1 since attestation is an
   *additional* signal, not a replacement for D1-D4.

3. **Confidence rule:** Verified solvency attestation lifts
   INSUFFICIENT → LOW only. Never touches dimension scores. Never
   lifts above LOW in v1.

4. **New dependency:** `json-canonical>=2.0.0` — pure Python, RFC 8785
   canonical JSON serialization. Required for identifier computation.

## Open questions for production

1. **Proof-to-wallet binding** — `contextAddress` in signed envelope
   (v1) vs wallet signature over proof hash (v2 hardening)
2. **Freshness** — enforce max proof age (24h suggested for solvency)
3. **Replay protection** — store proof hashes in DB
4. **Attestor governance** — cache with TTL, log rotations
5. **Reclaim developer app** — Pablo needs to register app ID/secret
   to define custom providers (CEX balance). **Pablo action item.**

## Competitive note

zkPass is repositioning as "AI Agent Trust Layer" with agent credit
scoring — direct competitive overlap. Worth tracking as both threat
and potential provider #2 alongside Reclaim.

## Test results

```
32 passed in 1.04s
```

All tamper vectors confirmed:
- Tampered identifier → rejected
- Tampered owner → rejected
- Tampered timestamp → rejected
- Tampered epoch → rejected
- Unknown attestor → rejected
- Malformed signature → rejected (graceful)
- Wrong-length signature → rejected (graceful)
