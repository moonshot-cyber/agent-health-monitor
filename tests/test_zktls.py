"""Tests for zkTLS proof verification (Reclaim Protocol).

Uses real test fixture proofs from the Reclaim Python SDK test suite.
Signature recovery and attestor matching verified against known data.
"""

import copy
import json

import pytest

from zktls import (
    RECLAIM_ATTESTOR_ADDRESSES,
    ReclaimVerificationResult,
    _compute_identifier,
    _create_sign_data,
    _extract_parameters,
    _recover_signer,
    apply_attestation_confidence,
    verify_reclaim_proof,
)


# ---------------------------------------------------------------------------
# Fixtures — real proofs from reclaimprotocol/reclaim-python-sdk conftest.py
# ---------------------------------------------------------------------------

FIXTURE_PROOF_1 = {
    "identifier": "0xbb5c63656a650276728d3cb9ce3f90361223c7814fd94f6582b682dfc96e4ba8",
    "claimData": {
        "identifier": "0xbb5c63656a650276728d3cb9ce3f90361223c7814fd94f6582b682dfc96e4ba8",
        "provider": "http",
        "parameters": json.dumps({
            "body": "",
            "geoLocation": "in",
            "method": "GET",
            "paramValues": {"DYNAMIC_GEO": "IN", "username": "srivatsanqb"},
            "responseMatches": [{"type": "contains", "value": "{{username}}"}],
            "responseRedactions": [{"jsonPath": "$.username", "xPath": ""}],
            "url": "https://www.kaggle.com/api/i/users.UserService/GetUser?username={{username}}",
        }, separators=(",", ":")),
        "owner": "0x9c3dcb81fe10f6e494bfaa0220ea0ba7bcf3ad94",
        "timestampS": 1774346626,
        "context": json.dumps({
            "contextAddress": "0x0",
            "contextMessage": "for trust scoring",
            "extractedParameters": {"DYNAMIC_GEO": "IN", "username": "srivatsanqb"},
            "providerHash": "0xf908ede49tried2f85f8bb32a5fa57cd1968455d7ecddae7700e9bad37fc23ab4",
        }, separators=(",", ":")),
        "epoch": 1,
    },
    "signatures": [
        "0x379b164165e005d75be4ec7854d745d68ad56d738a08da3a4c30eb071948bf5d"
        "0c7262bb8c46189e0cadb583dbb00917b73fbdbf74b5914eb69774ce97196a911c"
    ],
    "witnesses": [
        {
            "id": "0x244897572368eadf65bfbc5aec98d8e5443a9072",
            "url": "wss://attestor.reclaimprotocol.org:444/ws",
        }
    ],
}

FIXTURE_PROOF_2 = {
    "identifier": "0x51c192777d45010e9318c0e1eb2fefc0bc5a444f59e3d3e5a11e9a3d1b98e10c",
    "claimData": {
        "identifier": "0x51c192777d45010e9318c0e1eb2fefc0bc5a444f59e3d3e5a11e9a3d1b98e10c",
        "provider": "http",
        "parameters": json.dumps({
            "body": "",
            "geoLocation": "in",
            "method": "GET",
            "paramValues": {"DYNAMIC_GEO": "IN", "username": "mushaheedsyed"},
            "responseMatches": [{"type": "contains", "value": "{{username}}"}],
            "responseRedactions": [{"jsonPath": "$.username", "xPath": ""}],
            "url": "https://www.kaggle.com/api/i/users.UserService/GetUser?username={{username}}",
        }, separators=(",", ":")),
        "owner": "0x2967c5e6b3c4f179699bcc6e45bbe13b2203818e",
        "timestampS": 1773163350,
        "context": json.dumps({
            "contextAddress": "0x0",
            "contextMessage": "for verification",
            "extractedParameters": {"DYNAMIC_GEO": "IN", "username": "mushaheedsyed"},
            "providerHash": "0xf908ede49tried2f85f8bb32a5fa57cd1968455d7ecddae7700e9bad37fc23ab4",
        }, separators=(",", ":")),
        "epoch": 1,
    },
    "signatures": [
        "0x561d209c999536ad0c6b5834bb5416963a3d61b3045e621d99ba5e0a07aa1a7b"
        "0707a4e8f4a218c5dd13f9e470d3c7023b7ddeda5463069eb08c231dbb0ab63c1b"
    ],
    "witnesses": [
        {
            "id": "0x244897572368eadf65bfbc5aec98d8e5443a9072",
            "url": "wss://attestor.reclaimprotocol.org:444/ws",
        }
    ],
}


# ===================================================================
# verify_reclaim_proof — valid proofs
# ===================================================================

class TestVerifyReclaimProofValid:
    """Proofs signed by a known attestor pass verification."""

    def test_fixture_1_valid(self):
        result = verify_reclaim_proof(FIXTURE_PROOF_1)
        assert result.valid is True
        assert result.provider == "reclaim"
        assert result.signer_address == "0x244897572368eadf65bfbc5aec98d8e5443a9072"
        assert result.proof_hash  # non-empty
        assert result.verified_at  # non-empty
        assert result.error is None

    def test_fixture_2_valid(self):
        result = verify_reclaim_proof(FIXTURE_PROOF_2)
        assert result.valid is True
        assert result.signer_address == "0x244897572368eadf65bfbc5aec98d8e5443a9072"

    def test_extracted_parameters(self):
        result = verify_reclaim_proof(FIXTURE_PROOF_1)
        assert result.extracted_parameters == {
            "DYNAMIC_GEO": "IN",
            "username": "srivatsanqb",
        }

    def test_claim_type_from_provider(self):
        result = verify_reclaim_proof(FIXTURE_PROOF_1)
        assert result.claim_type == "http"


# ===================================================================
# verify_reclaim_proof — invalid / tampered proofs
# ===================================================================

class TestVerifyReclaimProofInvalid:
    """Tampered or malformed proofs are rejected."""

    def test_missing_claim_data(self):
        result = verify_reclaim_proof({"signatures": ["0xabc"]})
        assert result.valid is False
        assert "missing claimData" in result.error

    def test_missing_claim_fields(self):
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        del proof["claimData"]["owner"]
        del proof["claimData"]["epoch"]
        result = verify_reclaim_proof(proof)
        assert result.valid is False
        assert "epoch" in result.error
        assert "owner" in result.error

    def test_missing_signatures(self):
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["signatures"] = []
        result = verify_reclaim_proof(proof)
        assert result.valid is False
        assert "empty signatures" in result.error

    def test_tampered_identifier(self):
        """Changing the identifier invalidates the signature."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["claimData"]["identifier"] = "0xdeadbeefdeadbeefdeadbeefdeadbeef" + "00" * 16
        result = verify_reclaim_proof(proof)
        assert result.valid is False
        assert "no signature matched" in result.error

    def test_tampered_owner(self):
        """Changing the owner invalidates the signature."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["claimData"]["owner"] = "0x0000000000000000000000000000000000000001"
        result = verify_reclaim_proof(proof)
        assert result.valid is False

    def test_tampered_timestamp(self):
        """Changing the timestamp invalidates the signature."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["claimData"]["timestampS"] = 9999999999
        result = verify_reclaim_proof(proof)
        assert result.valid is False

    def test_tampered_epoch(self):
        """Changing the epoch invalidates the signature."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["claimData"]["epoch"] = 999
        result = verify_reclaim_proof(proof)
        assert result.valid is False

    def test_unknown_attestor(self):
        """Valid signature but from an address not in our trusted set."""
        result = verify_reclaim_proof(
            FIXTURE_PROOF_1,
            attestor_addresses={"0x0000000000000000000000000000000000000001"},
        )
        assert result.valid is False
        assert "no signature matched" in result.error

    def test_malformed_signature_hex(self):
        """Garbage signature hex is handled gracefully."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["signatures"] = ["0xnotavalidhex"]
        result = verify_reclaim_proof(proof)
        assert result.valid is False

    def test_wrong_length_signature(self):
        """Signature with wrong byte length is handled gracefully."""
        proof = copy.deepcopy(FIXTURE_PROOF_1)
        proof["signatures"] = ["0x" + "ab" * 32]  # 32 bytes, needs 65
        result = verify_reclaim_proof(proof)
        assert result.valid is False


# ===================================================================
# apply_attestation_confidence
# ===================================================================

class TestApplyAttestationConfidence:
    """Confidence modification rules for v1."""

    def _make_valid_attestation(self, claim_type: str = "http") -> ReclaimVerificationResult:
        return ReclaimVerificationResult(
            valid=True,
            provider="reclaim",
            claim_type=claim_type,
            proof_hash="abc123",
            verified_at="2026-06-03T00:00:00Z",
        )

    def _make_invalid_attestation(self) -> ReclaimVerificationResult:
        return ReclaimVerificationResult(
            valid=False,
            error="test failure",
        )

    # -- INSUFFICIENT -> LOW with solvency claim --

    def test_insufficient_to_low_with_solvency(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "INSUFFICIENT", att, claim_type_override="binance_balance",
        )
        assert result == "LOW"

    def test_insufficient_to_low_coinbase(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "INSUFFICIENT", att, claim_type_override="coinbase_balance",
        )
        assert result == "LOW"

    def test_insufficient_to_low_generic_cex(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "INSUFFICIENT", att, claim_type_override="cex_balance",
        )
        assert result == "LOW"

    # -- No lift above LOW --

    def test_low_stays_low(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "LOW", att, claim_type_override="binance_balance",
        )
        assert result == "LOW"

    def test_medium_stays_medium(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "MEDIUM", att, claim_type_override="binance_balance",
        )
        assert result == "MEDIUM"

    def test_high_stays_high(self):
        att = self._make_valid_attestation()
        result = apply_attestation_confidence(
            "HIGH", att, claim_type_override="binance_balance",
        )
        assert result == "HIGH"

    # -- Non-solvency claims have no effect --

    def test_non_solvency_no_effect(self):
        att = self._make_valid_attestation(claim_type="http")
        result = apply_attestation_confidence("INSUFFICIENT", att)
        assert result == "INSUFFICIENT"

    def test_unknown_claim_type_no_effect(self):
        att = self._make_valid_attestation(claim_type="github_stars")
        result = apply_attestation_confidence("INSUFFICIENT", att)
        assert result == "INSUFFICIENT"

    # -- Invalid attestation has no effect --

    def test_invalid_attestation_no_effect(self):
        att = self._make_invalid_attestation()
        result = apply_attestation_confidence("INSUFFICIENT", att)
        assert result == "INSUFFICIENT"

    def test_invalid_attestation_no_effect_on_low(self):
        att = self._make_invalid_attestation()
        result = apply_attestation_confidence("LOW", att)
        assert result == "LOW"


# ===================================================================
# Internal helpers
# ===================================================================

class TestInternalHelpers:
    """Unit tests for internal functions."""

    def test_recover_signer_fixture_1(self):
        sign_data = _create_sign_data(FIXTURE_PROOF_1["claimData"])
        sig = FIXTURE_PROOF_1["signatures"][0]
        signer = _recover_signer(sign_data, sig)
        assert signer == "0x244897572368eadf65bfbc5aec98d8e5443a9072"

    def test_recover_signer_fixture_2(self):
        sign_data = _create_sign_data(FIXTURE_PROOF_2["claimData"])
        sig = FIXTURE_PROOF_2["signatures"][0]
        signer = _recover_signer(sign_data, sig)
        assert signer == "0x244897572368eadf65bfbc5aec98d8e5443a9072"

    def test_create_sign_data_format(self):
        """Sign data is identifier\\nowner\\ntimestamp\\nepoch."""
        data = _create_sign_data(FIXTURE_PROOF_1["claimData"])
        lines = data.split("\n")
        assert len(lines) == 4
        assert lines[0] == FIXTURE_PROOF_1["claimData"]["identifier"]
        assert lines[1] == FIXTURE_PROOF_1["claimData"]["owner"].lower()
        assert lines[2] == str(FIXTURE_PROOF_1["claimData"]["timestampS"])
        assert lines[3] == str(FIXTURE_PROOF_1["claimData"]["epoch"])

    def test_extract_parameters(self):
        params = _extract_parameters(FIXTURE_PROOF_1["claimData"])
        assert params == {"DYNAMIC_GEO": "IN", "username": "srivatsanqb"}

    def test_extract_parameters_empty_context(self):
        claim = {"context": ""}
        assert _extract_parameters(claim) is None

    def test_extract_parameters_no_context(self):
        claim = {}
        assert _extract_parameters(claim) is None

    def test_proof_hash_deterministic(self):
        """Same proof produces same hash."""
        r1 = verify_reclaim_proof(FIXTURE_PROOF_1)
        r2 = verify_reclaim_proof(FIXTURE_PROOF_1)
        assert r1.proof_hash == r2.proof_hash
        assert r1.proof_hash != ""

    def test_proof_hash_changes_on_tamper(self):
        """Different proof content produces different hash."""
        r1 = verify_reclaim_proof(FIXTURE_PROOF_1)
        r2 = verify_reclaim_proof(FIXTURE_PROOF_2)
        assert r1.proof_hash != r2.proof_hash
