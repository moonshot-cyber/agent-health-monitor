# zkTLS Feasibility Spike — Reclaim Proof Verification as Confidence Input

**Date:** 2026-06-03
**Branch:** `spike/zktls-verification`
**Status:** Spike complete. Not production. Do not merge without review.

---

## Summary

This spike proves that AHM can verify a Reclaim Protocol zkTLS proof
server-side and fold it into scan output as a confidence modifier.
AHM acts as the **verifier only** — agent operators generate proofs
against their own accounts (e.g. CEX balance).

**Result:** Feasible. Manual EIP-191 signature verification works with
existing repo dependencies (`web3`, `eth-account`). 32 tests passing
including valid proof verification, tamper detection (5 tamper vectors),
and confidence-modification rule enforcement.

---

## Phase 1 — Verification Path Findings

Three paths evaluated, in order of preference:

### Path 1: Official Reclaim Python SDK (reclaimprotocol-python-sdk)

- **Version:** 2.0.0 on PyPI (setup.py) / 1.0.3 (PyPI listing — stale)
- **Package name:** `reclaimprotocol-python-sdk`
- **Status:** EXISTS but FAILS TO INSTALL on Windows (and likely any
  environment without a C compiler) due to `safe-pysha3` dependency
  requiring MSVC 14.0+.
- **Dependencies:** `web3>=6.0.0`, `eth-account>=0.8.0`, `json-canonical>=2.0.0`,
  `canonicaljson>=1.0.0`, `aiohttp>=3.8.0`, `httpx>=0.24.0`,
  `safe-pysha3>=1.0.2`, `requests>=2.32.3`, `python-dotenv>=1.0.0`
- **Key export:** `verify_proof(proof, config)` — async function that
  fetches attestor list from Reclaim backend, recovers signers, checks
  against attestor set.
- **Verdict:** Not chosen. The `safe-pysha3` install failure is specific
  to the Windows dev box (no MSVC); Railway runs Linux and would likely
  build it. The real reasons to avoid the SDK are: (1) it is async-only,
  adding complexity to a sync FastAPI codebase; (2) it pulls 9
  dependencies where we need only the signature-recovery path; (3) it
  fetches the attestor list from Reclaim's backend at verify time,
  adding a runtime network dependency that manual verification avoids.

### Path 2: Manual verification ← CHOSEN

- **How it works:** Parse proof JSON → build sign-data string from claim
  fields → `encode_defunct()` (EIP-191) → `recover_message()` (ECDSA
  recovery) → check recovered address against known attestor set.
- **Dependencies:** `web3` and `eth-account` (both already in
  requirements.txt), plus `json-canonical` for RFC 8785 context
  canonicalization.
- **New dependency:** `json-canonical>=2.0.0` (pure Python, no C
  extensions, 2.0.0 installs cleanly).
- **Verdict:** Works. All test fixtures verify. Tamper detection
  confirmed on 5 vectors (identifier, owner, timestamp, epoch, unknown
  attestor). This is the approach used in the spike.

### Path 3: Node subprocess wrapping JS SDK

- **Not attempted.** Path 2 works cleanly with zero operational overhead.
  A Node subprocess would add: Node runtime dependency on Railway,
  IPC serialization, subprocess lifecycle management, cold-start latency.
  Not justified.

---

## Proof JSON Anatomy

A Reclaim proof has this structure:

```json
{
  "identifier": "0x<keccak256 of provider|parameters|context>",
  "claimData": {
    "identifier": "0x...",
    "provider": "http",
    "parameters": "{\"url\":\"...\",\"method\":\"GET\",...}",
    "owner": "0x<agent wallet address>",
    "timestampS": 1774346626,
    "context": "{\"contextAddress\":\"0x0\",\"contextMessage\":\"...\",\"extractedParameters\":{...},\"providerHash\":\"0x...\"}",
    "epoch": 1
  },
  "signatures": ["0x<65-byte EIP-191 signature hex>"],
  "witnesses": [
    {
      "id": "0x<attestor ethereum address>",
      "url": "wss://attestor.reclaimprotocol.org:444/ws"
    }
  ]
}
```

### Field semantics

| Field | Role |
|-------|------|
| `identifier` | `keccak256(provider + "\n" + parameters + "\n" + canonicalJSON(context))` — deterministic claim fingerprint |
| `claimData.provider` | Always `"http"` for HTTP-based providers; actual provider identity lives in parameters URL and context.providerHash |
| `claimData.parameters` | JSON string defining the HTTP request: URL template, method, response matching rules, redaction paths |
| `claimData.owner` | Ethereum address of the proof requester (the agent operator's wallet) |
| `claimData.timestampS` | Unix timestamp when the attestor witnessed the claim |
| `claimData.context` | JSON string with `contextAddress`, `contextMessage`, `extractedParameters`, `providerHash` |
| `claimData.epoch` | Beacon epoch number (attestor set rotation epoch) |
| `signatures` | EIP-191 signatures over `identifier\nowner\ntimestampS\nepoch` |
| `witnesses` | Attestor addresses and WebSocket URLs that produced the signatures |

### Signature verification flow

1. Build sign-data string: `"{identifier}\n{owner}\n{timestampS}\n{epoch}"`
2. Wrap with EIP-191 prefix: `encode_defunct(text=sign_data)`
3. Recover signer: `w3.eth.account.recover_message(message, signature=sig_bytes)`
4. Check: `recovered_address.lower() in trusted_attestor_set`

---

## Attestor Trust Assumptions (Proxy Model)

Reclaim uses an **opaque proxy** attestor model:

- The attestor (witness server) sits in the TLS session between the
  user's browser and the target website (e.g. Binance).
- It does NOT terminate TLS or see plaintext credentials — it validates
  the TLS certificate chain and signs a claim that the user accessed a
  specific resource and received a response matching defined patterns.
- The attestor signs over: the claim identifier (hash of
  provider+parameters+context), the owner address, the timestamp, and
  the epoch.
- Trust chain: **AHM trusts the attestor's signature** → the attestor
  attests that the HTTP response contained specific content → the
  content is the claim (e.g. "balance > $X").

### What AHM is trusting

1. **Reclaim's attestor infrastructure** — that the attestor correctly
   validates TLS and does not collude with the proof requester.
2. **The attestor set governance** — currently 3 attestor addresses
   fetched from `https://api.reclaimprotocol.org/api/attestors`.
   Reclaim controls this set. No on-chain governance.
3. **The provider definition** — that the HTTP request template
   accurately queries the intended data source (e.g. Binance balance
   API) and that response matching rules extract the correct value.

### What AHM is NOT trusting

- The agent operator (they provide the proof; AHM verifies it).
- The content of the response beyond what the attestor signed over.
- Any claim about the agent's on-chain behaviour (that remains D1-D4).

---

## Open Questions

### 1. Proof-to-wallet binding

The `claimData.owner` field carries an Ethereum address — this is the
address that requested the proof, not necessarily the agent wallet being
scored by AHM. Two binding approaches:

- **Context field binding:** The `contextAddress` field inside the
  signed context could carry the agent wallet address. Since context is
  part of the signed claim (hashed into the identifier), the attestor
  signs over it — any tampering invalidates the proof. This requires
  the proof-generation step to embed the agent wallet address in
  `contextAddress`.
- **Wallet signature over proof hash:** The agent wallet signs the
  proof hash after generation, proving the wallet holder produced or
  endorsed the proof. More robust (works even if context wasn't
  configured at proof time) but requires a second signature.

**Recommendation:** Context field binding for v1 (simpler, already in
the signed envelope). Wallet-signature binding as v2 hardening.

### 2. Proof freshness / replay protection

A proof is timestamped (`timestampS`) by the attestor. Without a
freshness check, an old proof could be replayed indefinitely.

**Recommendation:** Enforce a maximum proof age (e.g. 24 hours for
solvency claims). Store proof hashes in the scan database to detect
replay of the same proof across scans.

### 3. Attestor set governance

The attestor set is currently 3 addresses fetched from Reclaim's
centralised API. No on-chain registry. If Reclaim rotates attestors,
AHM needs to update. If Reclaim is compromised, all proofs become
untrustworthy.

**Recommendation:** Cache attestors with a 1-hour TTL. Log attestor
set changes. Monitor for Reclaim moving to on-chain governance. In v1,
this is acceptable — the attestor is an additional signal, not a
replacement for on-chain D1-D4 scoring.

### 4. Identifier recomputation — HARD REQUIREMENT

The attestor signature covers only `identifier|owner|timestampS|epoch`.
The context bytes (including `contextAddress`, `extractedParameters`,
and `providerHash`) are bound to the signature **only via the identifier
hash** — `keccak256(provider + "\n" + parameters + "\n" +
canonicalJSON(context))`. Without hard identifier recomputation, a
validly-signed proof can have its context swapped: an attacker takes
a proof signed for wallet A, replaces `contextAddress` with wallet B,
and the signature still verifies because the sign-data string uses
the original identifier. This makes proof-to-wallet binding
(production item 4) and provider registry mapping (production item 9)
entirely bypassable without this check.

`_compute_identifier()` already exists in `zktls.py` (implemented
during the spike but not wired into `verify_reclaim_proof()`).
**Spike finding:** identifier recomputation works correctly against
both real SDK test fixtures using `json-canonical` (RFC 8785). The
earlier canonicalization mismatch was caused by corrupted test fixture
data, not by a fundamental RFC 8785 incompatibility. Two passing tests
(`test_compute_identifier_fixture_1`, `test_compute_identifier_fixture_2`)
confirm this.

**Requirement for production:** `verify_reclaim_proof()` must call
`_compute_identifier()` and hard-fail on mismatch. Canonicalization
edge cases are to be solved (match the attestor's byte-level RFC 8785
output), not bypassed. The Reclaim SDK's
`dangerouslyDisableContentValidation` flag exists for development
convenience; AHM must not use it — our entire trust model for context
fields depends on this check. Production items 4 (proof-to-wallet
binding) and 9 (providerHash mapping) are security-dependent on this
being a hard check.

---

## What a Production Integration Would Require

1. **New dependency:** `json-canonical>=2.0.0` added to requirements.txt.
2. **Attestor caching:** Fetch from `https://api.reclaimprotocol.org/api/attestors`
   with 1-hour TTL cache instead of hardcoded set.
3. **New endpoint:** `POST /ahs/attestation` accepting a proof JSON
   and an agent wallet address. Returns verification result.
4. **Proof-to-wallet binding:** Validate that `contextAddress` matches
   the wallet being scored, or require a wallet signature over the
   proof hash. **Depends on identifier recomputation (OQ4)** — without
   it, `contextAddress` can be swapped on a validly-signed proof.
5. **Freshness enforcement:** Reject proofs older than configurable
   max age (24h default for solvency claims).
6. **Replay protection:** Store proof hashes in DB; reject duplicates.
7. **Confidence integration:** Wire `apply_attestation_confidence()`
   into the `calculate_ahs()` return path. The scan result would carry
   an `attestations` field listing any verified proofs that modified
   confidence.
8. **Reclaim developer app:** Pablo needs to register a Reclaim app
   (app ID + secret) to define custom providers (e.g. "Binance balance
   > $1000"). This controls what proofs agents can generate against
   AHM. **This is a Pablo action item — not automatable.**
9. **Provider registry mapping:** Map `providerHash` values to
   human-readable claim types (binance_balance, coinbase_balance, etc.)
   so `apply_attestation_confidence()` can distinguish solvency
   attestations from other proof types. **Depends on identifier
   recomputation (OQ4)** — without it, `providerHash` in the context
   can be swapped to impersonate a different provider type.

---

## zkPass — Provider #2 Note

[zkPass](https://zkpass.org/) is a competing zkTLS protocol with a
similar attestor-based architecture. Their recent repositioning toward
**agent credit scoring** (TransGate SDK for verifiable credentials,
"AI Agent Trust Layer" messaging) creates direct competitive overlap
with AHM's trust-scoring positioning.

Key differences from Reclaim:
- zkPass uses a three-party TLS (3P-TLS) model vs Reclaim's proxy model.
- zkPass has an on-chain verification contract (Solidity SDK).
- zkPass is positioning their attestations as credit inputs — if they
  gain traction, AHM consuming zkPass proofs alongside Reclaim would
  be strategically valuable (provider-neutral verification layer).

**Recommendation:** Track zkPass as both a competitive-overlap watch
item and a potential provider #2. The `verify_reclaim_proof()` function
is provider-specific; a production design would abstract behind a
`verify_attestation(proof, provider="reclaim"|"zkpass")` interface.

---

## Files in This Spike

| File | Purpose |
|------|---------|
| `zktls.py` | Verification module: `verify_reclaim_proof()`, `apply_attestation_confidence()` |
| `tests/test_zktls.py` | 32 tests: valid proofs, tamper detection, confidence rules, helpers |
| `docs/zktls_spike.md` | This document |
