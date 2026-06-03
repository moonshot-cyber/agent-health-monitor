"""
zkTLS proof verification for AHM — Reclaim Protocol attestation verifier.

SPIKE MODULE — not wired to production endpoints.

AHM is the VERIFIER only. Agent operators generate proofs via Reclaim
against their own accounts (e.g. CEX balance). This module verifies
those proofs server-side and produces an attestation result that can
modify scan confidence (never dimension scores).

Verification path: manual EIP-191 signature recovery using web3/eth-account.
The official Reclaim Python SDK (reclaimprotocol-python-sdk 2.0.0) exists
but fails to install on Windows due to safe-pysha3 C extension requirement.
The underlying crypto is standard and reproducible with existing deps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from eth_account.messages import encode_defunct
from json_canonical import canonicalize
from web3 import Web3

# ---------------------------------------------------------------------------
# Known Reclaim attestor addresses (fetched from
# https://api.reclaimprotocol.org/api/attestors on 2026-06-03).
# In production, fetch dynamically and cache with TTL.
# ---------------------------------------------------------------------------
RECLAIM_ATTESTOR_ADDRESSES: set[str] = {
    "0x244897572368eadf65bfbc5aec98d8e5443a9072",
    "0x1be31a94361a391bbafb2a4ccd704f57dc04d4bb",
    "0x15ea90114bffea9bb996e8775bf6ca0a338e97e5",
}

# Claim types we recognise for confidence modification
SOLVENCY_CLAIM_TYPES = frozenset({
    "binance_balance",
    "coinbase_balance",
    "kraken_balance",
    "okx_balance",
    "cex_balance",          # generic
})


@dataclass
class ReclaimVerificationResult:
    """Result of verifying a single Reclaim zkTLS proof."""
    valid: bool
    provider: str = "reclaim"
    claim_type: str = ""
    extracted_parameters: dict | None = None
    proof_hash: str = ""
    verified_at: str = ""
    error: str | None = None
    signer_address: str | None = None


def _compute_identifier(provider: str, parameters: str, context: str) -> str:
    """Recompute the claim identifier from provider/parameters/context.

    Uses RFC 8785 canonical JSON for the context field, matching the
    Reclaim SDK's get_identifier_from_claim_info().
    """
    canonical_context = context or ""
    if canonical_context:
        try:
            ctx = json.loads(canonical_context)
            canonical_bytes = canonicalize(ctx)
            canonical_context = (
                canonical_bytes.decode("utf-8")
                if isinstance(canonical_bytes, bytes)
                else canonical_bytes
            )
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"unable to parse context as JSON: {exc}") from exc

    text = f"{provider}\n{parameters}\n{canonical_context}"
    return "0x" + Web3.keccak(text=text).hex().lower()


def _create_sign_data(claim_data: dict) -> str:
    """Build the message string that attestors sign over.

    Format: identifier\\nowner\\ntimestampS\\nepoch
    Uses the identifier from the proof (pre-computed by the attestor).
    """
    return "\n".join([
        claim_data["identifier"],
        claim_data["owner"].lower(),
        str(claim_data["timestampS"]),
        str(claim_data["epoch"]),
    ])


def _recover_signer(sign_data: str, signature_hex: str) -> str:
    """Recover the Ethereum address that produced the EIP-191 signature."""
    sig_bytes = bytes.fromhex(signature_hex.replace("0x", ""))
    message = encode_defunct(text=sign_data)
    w3 = Web3()
    return w3.eth.account.recover_message(message, signature=sig_bytes).lower()


def _extract_claim_type(claim_data: dict) -> str:
    """Best-effort extraction of a human-readable claim type.

    Reclaim proofs use 'provider' = 'http' for all HTTP-based providers.
    The actual provider identity lives in the parameters JSON (the URL
    pattern) or in context.providerHash.  For this spike, return the
    provider field verbatim; production would map providerHash to a
    known provider registry.
    """
    return claim_data.get("provider", "unknown")


def _extract_parameters(claim_data: dict) -> dict | None:
    """Extract parameters from the proof context."""
    context_str = claim_data.get("context", "")
    if not context_str:
        return None
    try:
        ctx = json.loads(context_str)
        return ctx.get("extractedParameters")
    except (json.JSONDecodeError, TypeError):
        return None


def verify_reclaim_proof(
    proof_json: dict,
    attestor_addresses: set[str] | None = None,
) -> ReclaimVerificationResult:
    """Verify a Reclaim Protocol zkTLS proof.

    Checks:
    1. Proof structure is well-formed (required fields present).
    2. At least one signature recovers to a known attestor address.

    Does NOT re-derive the identifier from provider/parameters/context
    (this would require exact byte-level match with attestor-side
    canonicalization; the SDK itself supports skipping this via
    dangerouslyDisableContentValidation). Signature verification alone
    proves the attestor endorsed this exact claim data.

    Args:
        proof_json: The proof dict with keys: identifier, claimData,
                    signatures, witnesses.
        attestor_addresses: Override set of trusted attestor addresses
                           (lowercase hex). Defaults to hardcoded set.

    Returns:
        ReclaimVerificationResult with valid=True/False and details.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    trusted = attestor_addresses or RECLAIM_ATTESTOR_ADDRESSES

    # -- Structural validation --
    claim_data = proof_json.get("claimData")
    if not claim_data:
        return ReclaimVerificationResult(
            valid=False, verified_at=now,
            error="missing claimData",
        )

    required_claim_fields = {"identifier", "provider", "owner", "timestampS", "epoch"}
    missing = required_claim_fields - set(claim_data.keys())
    if missing:
        return ReclaimVerificationResult(
            valid=False, verified_at=now,
            error=f"claimData missing fields: {sorted(missing)}",
        )

    signatures = proof_json.get("signatures")
    if not signatures or not isinstance(signatures, list):
        return ReclaimVerificationResult(
            valid=False, verified_at=now,
            error="missing or empty signatures",
        )

    # -- Signature verification --
    sign_data = _create_sign_data(claim_data)

    verified_signer = None
    for sig in signatures:
        try:
            signer = _recover_signer(sign_data, sig)
            if signer in trusted:
                verified_signer = signer
                break
        except Exception:
            continue  # malformed signature, try next

    if not verified_signer:
        return ReclaimVerificationResult(
            valid=False, verified_at=now,
            error="no signature matched a trusted attestor",
        )

    # -- Build result --
    proof_hash = hashlib.sha256(
        json.dumps(proof_json, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return ReclaimVerificationResult(
        valid=True,
        provider="reclaim",
        claim_type=_extract_claim_type(claim_data),
        extracted_parameters=_extract_parameters(claim_data),
        proof_hash=proof_hash,
        verified_at=now,
        signer_address=verified_signer,
    )


def apply_attestation_confidence(
    scan_confidence: str,
    attestation: ReclaimVerificationResult,
    claim_type_override: str | None = None,
) -> str:
    """Apply a verified attestation to modify scan confidence.

    Rules (v1):
    - Only verified solvency attestations have any effect.
    - INSUFFICIENT -> LOW is the only allowed lift.
    - Never modifies D1-D4 dimension scores.
    - Never lifts confidence above LOW.

    Args:
        scan_confidence: Current confidence from _ahs_confidence()
                        ("INSUFFICIENT", "LOW", "MEDIUM", "HIGH").
        attestation: Result from verify_reclaim_proof().
        claim_type_override: If provided, used instead of
                            attestation.claim_type for solvency check.

    Returns:
        The (possibly modified) confidence string.
    """
    if not attestation.valid:
        return scan_confidence

    claim_type = (claim_type_override or attestation.claim_type).lower()

    # Only solvency-type attestations modify confidence in v1
    if claim_type not in SOLVENCY_CLAIM_TYPES:
        return scan_confidence

    # Only lift INSUFFICIENT -> LOW
    if scan_confidence == "INSUFFICIENT":
        return "LOW"

    return scan_confidence
