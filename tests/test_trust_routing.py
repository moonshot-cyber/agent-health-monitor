"""Tests for the Tiered Trust Routing feature.

Covers:
- _trust_routing() helper mappings
- _trust_routing_with_policy() policy-aware routing
- GET /ahs/route/{address} endpoint: success, stale flag, 404, policy integration
- GET/PUT /ahs/route/policy endpoints
- Routing policy validation
- Routing policy DB CRUD
"""

import os
import sys

os.environ.setdefault("PAYMENT_ADDRESS", "0x" + "a" * 40)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Unit tests for _trust_routing helper ─────────────────────────────────


class TestTrustRoutingHelper:
    """Verify grade → routing recommendation mapping."""

    def test_grade_a_instant_settle(self):
        from api import _trust_routing
        assert _trust_routing("A") == "instant_settle"

    def test_grade_b_instant_settle(self):
        from api import _trust_routing
        assert _trust_routing("B") == "instant_settle"

    def test_grade_c_escrow(self):
        from api import _trust_routing
        assert _trust_routing("C") == "escrow"

    def test_grade_d_reject(self):
        from api import _trust_routing
        assert _trust_routing("D") == "reject"

    def test_grade_e_reject(self):
        from api import _trust_routing
        assert _trust_routing("E") == "reject"

    def test_grade_f_reject(self):
        from api import _trust_routing
        assert _trust_routing("F") == "reject"


# ── Unit tests for _trust_routing_with_policy ────────────────────────────


class TestTrustRoutingWithPolicy:
    """Verify policy-aware routing logic."""

    def test_none_policy_uses_defaults(self):
        """No policy → falls back to default _trust_routing behavior."""
        from api import _trust_routing_with_policy
        assert _trust_routing_with_policy("A", None) == "instant_settle"
        assert _trust_routing_with_policy("C", None) == "escrow"
        assert _trust_routing_with_policy("D", None) == "reject"

    def test_custom_instant_grades(self):
        """Custom policy: A,B,C → instant_settle."""
        from api import _trust_routing_with_policy
        policy = {"instant_grades": "A,B,C", "escrow_grades": "D", "reject_grades": "E,F"}
        assert _trust_routing_with_policy("C", policy) == "instant_settle"
        assert _trust_routing_with_policy("D", policy) == "escrow"
        assert _trust_routing_with_policy("E", policy) == "reject"

    def test_escrow_disabled_binary(self):
        """escrow_disabled=True → C falls through to reject."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "",
            "reject_grades": "C,D,E,F",
            "escrow_disabled": True,
        }
        assert _trust_routing_with_policy("A", policy) == "instant_settle"
        assert _trust_routing_with_policy("C", policy) == "reject"

    def test_allowlisted_always_instant(self):
        """Allowlisted address → instant_settle regardless of grade."""
        from api import _trust_routing_with_policy
        # Even with a policy that would reject grade F
        policy = {"instant_grades": "A", "escrow_grades": "B", "reject_grades": "C,D,E,F"}
        assert _trust_routing_with_policy("F", policy, is_allowlisted=True) == "instant_settle"

    def test_allowlisted_no_policy(self):
        """Allowlisted + no policy → still instant_settle."""
        from api import _trust_routing_with_policy
        assert _trust_routing_with_policy("F", None, is_allowlisted=True) == "instant_settle"

    # ── Confidence override unit tests ────────────────────────────────────

    def test_confidence_override_promotes_grade(self):
        """Grade C (normally escrow) + HIGH confidence → instant_settle."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="HIGH") == "instant_settle"

    def test_confidence_override_demotes_grade(self):
        """Grade B (normally instant) + LOW confidence → escrow."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"B": {"LOW": "escrow"}},
        }
        assert _trust_routing_with_policy("B", policy, confidence="LOW") == "escrow"

    def test_confidence_override_to_reject(self):
        """Grade C + INSUFFICIENT → reject."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"INSUFFICIENT": "reject"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="INSUFFICIENT") == "reject"

    def test_confidence_override_no_match_uses_default(self):
        """Override exists for C+HIGH, query is C+MEDIUM → grade-based default."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="MEDIUM") == "escrow"

    def test_confidence_override_empty_dict_no_effect(self):
        """Empty overrides → same as no overrides."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {},
        }
        assert _trust_routing_with_policy("C", policy, confidence="HIGH") == "escrow"

    def test_confidence_override_null_no_effect(self):
        """None overrides → same as no overrides."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": None,
        }
        assert _trust_routing_with_policy("C", policy, confidence="HIGH") == "escrow"

    def test_confidence_override_case_insensitive(self):
        """Lowercase 'high' matches 'HIGH' entry via normalisation."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="high") == "instant_settle"

    def test_confidence_override_unknown_confidence_ignored(self):
        """'unknown' confidence → no override applied."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="unknown") == "escrow"

    def test_confidence_override_with_escrow_disabled_guard(self):
        """Override maps to escrow + escrow_disabled → downgraded to reject."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "",
            "reject_grades": "C,D,E,F",
            "escrow_disabled": True,
            "confidence_overrides": {"C": {"HIGH": "escrow"}},
        }
        assert _trust_routing_with_policy("C", policy, confidence="HIGH") == "reject"

    def test_confidence_override_allowlist_takes_precedence(self):
        """Allowlisted address ignores all overrides → instant_settle."""
        from api import _trust_routing_with_policy
        policy = {
            "instant_grades": "A,B",
            "escrow_grades": "C",
            "reject_grades": "D,E,F",
            "confidence_overrides": {"C": {"HIGH": "reject"}},
        }
        assert _trust_routing_with_policy(
            "C", policy, is_allowlisted=True, confidence="HIGH",
        ) == "instant_settle"


# ── Integration tests for GET /ahs/route/{address} ──────────────────────


VALID_ADDR = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"


@pytest.fixture
def client():
    import db as _db
    _db.init_db()
    from api import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    """X-Internal-Key header to bypass x402 payment middleware."""
    from api import INTERNAL_API_KEY
    return {"X-Internal-Key": INTERNAL_API_KEY}


def _mock_record(grade="B", ahs=82, confidence="high", hours_ago=2, address=VALID_ADDR):
    """Build a mock known_wallets row for get_latest_ahs_for_address."""
    scored_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "address": address,
        "latest_ahs": ahs,
        "latest_grade": grade,
        "confidence": confidence,
        "last_scanned_at": scored_at,
    }


class TestTrustRouteEndpoint:
    """GET /ahs/route/{address} integration tests."""

    @patch("db.get_latest_ahs_for_address")
    def test_instant_settle(self, mock_db, client, auth_headers):
        """Grade A/B → instant_settle."""
        mock_db.return_value = _mock_record(grade="A", ahs=92)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "instant_settle"
        assert data["agent_health_score"] == 92
        assert data["stale"] is False

    @patch("db.get_latest_ahs_for_address")
    def test_escrow(self, mock_db, client, auth_headers):
        """Grade C → escrow."""
        mock_db.return_value = _mock_record(grade="C", ahs=65)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "escrow"

    @patch("db.get_latest_ahs_for_address")
    def test_reject(self, mock_db, client, auth_headers):
        """Grade D/E/F → reject."""
        mock_db.return_value = _mock_record(grade="D", ahs=35)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "reject"

    @patch("db.get_latest_ahs_for_address")
    def test_stale_flag(self, mock_db, client, auth_headers):
        """Score older than 24h → stale: true."""
        mock_db.return_value = _mock_record(grade="B", ahs=80, hours_ago=30)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["stale"] is True

    @patch("db.get_latest_ahs_for_address")
    def test_404_unknown_address(self, mock_db, client, auth_headers):
        """No cached score → 404."""
        mock_db.return_value = None
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 404
        assert "No score available" in resp.json()["detail"]

    @patch("db.get_latest_ahs_for_address")
    def test_default_response_has_policy_fields(self, mock_db, client, auth_headers):
        """Default response includes policy_applied=False, allowlisted=False."""
        mock_db.return_value = _mock_record(grade="B", ahs=80)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy_applied"] is False
        assert data["allowlisted"] is False


# ── Routing policy validation tests ──────────────────────────────────────


class TestRoutingPolicyValidation:
    """Validate _validate_routing_policy edge cases."""

    def _make_body(self, **overrides):
        from api import RoutingPolicyRequest
        defaults = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
            "allowlist": None,
        }
        defaults.update(overrides)
        return RoutingPolicyRequest(**defaults)

    def test_valid_default_policy(self):
        """Default grade assignments pass validation."""
        from api import _validate_routing_policy
        body = self._make_body()
        _validate_routing_policy(body, None)  # should not raise

    def test_invalid_grade_letter(self):
        """Grade letter 'X' is rejected."""
        from api import _validate_routing_policy
        body = self._make_body(instant_grades=["A", "X"], reject_grades=["D", "E", "F"])
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Invalid grade" in str(exc_info.value.detail)

    def test_duplicate_grade(self):
        """Same grade in two categories → error."""
        from api import _validate_routing_policy
        body = self._make_body(
            instant_grades=["A", "B", "C"],
            escrow_grades=["C"],
            reject_grades=["D", "E", "F"],
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "multiple categories" in str(exc_info.value.detail)

    def test_missing_grades(self):
        """Not all 6 grades covered → error."""
        from api import _validate_routing_policy
        body = self._make_body(
            instant_grades=["A", "B"],
            escrow_grades=["C"],
            reject_grades=["D", "E"],  # Missing F
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Missing" in str(exc_info.value.detail)

    def test_escrow_disabled_with_escrow_grades(self):
        """escrow_disabled=True but escrow_grades non-empty → error."""
        from api import _validate_routing_policy
        body = self._make_body(escrow_disabled=True)  # escrow_grades=["C"] by default
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "escrow_disabled" in str(exc_info.value.detail)

    def test_escrow_enabled_no_escrow_grades(self):
        """escrow_disabled=False but escrow_grades empty → error."""
        from api import _validate_routing_policy
        body = self._make_body(
            escrow_grades=[],
            reject_grades=["C", "D", "E", "F"],
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "escrow_disabled" in str(exc_info.value.detail)

    def test_allowlist_too_large(self):
        """Allowlist > 1000 → error."""
        from api import _validate_routing_policy
        body = self._make_body(allowlist=["0x" + "a" * 40] * 1001)
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "1000" in str(exc_info.value.detail)

    def test_allowlist_invalid_address(self):
        """Invalid Ethereum address → error."""
        from api import _validate_routing_policy
        body = self._make_body(allowlist=["not-an-address"])
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Invalid Ethereum address" in str(exc_info.value.detail)

    def test_self_allowlist_rejected(self):
        """Caller's own address in allowlist → error."""
        from api import _validate_routing_policy
        caller = "0x" + "ab" * 20
        body = self._make_body(allowlist=[caller])
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, caller)
        assert "own wallet" in str(exc_info.value.detail)

    @patch("db.get_latest_ahs_for_address")
    def test_allowlist_grade_floor_below_c_rejected(self, mock_db):
        """Address with Grade D → rejected from allowlist."""
        from api import _validate_routing_policy
        addr = "0x" + "cc" * 20
        mock_db.return_value = _mock_record(grade="D", ahs=35, address=addr)
        body = self._make_body(allowlist=[addr])
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Grade D" in str(exc_info.value.detail)

    @patch("db.get_latest_ahs_for_address")
    def test_allowlist_no_score_rejected(self, mock_db):
        """Address with no AHS score → rejected from allowlist."""
        from api import _validate_routing_policy
        addr = "0x" + "dd" * 20
        mock_db.return_value = None
        body = self._make_body(allowlist=[addr])
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "no AHS score" in str(exc_info.value.detail)

    @patch("db.get_latest_ahs_for_address")
    def test_allowlist_grade_c_accepted(self, mock_db):
        """Address with Grade C → accepted in allowlist."""
        from api import _validate_routing_policy
        addr = "0x" + "ee" * 20
        mock_db.return_value = _mock_record(grade="C", ahs=65, address=addr)
        body = self._make_body(allowlist=[addr])
        _validate_routing_policy(body, None)  # should not raise

    # ── Confidence overrides validation tests ─────────────────────────────

    def test_confidence_overrides_valid(self):
        """Valid overrides accepted (no error raised)."""
        from api import _validate_routing_policy
        body = self._make_body(
            confidence_overrides={"C": {"HIGH": "instant_settle"}, "D": {"HIGH": "escrow"}},
        )
        _validate_routing_policy(body, None)  # should not raise

    def test_confidence_overrides_invalid_confidence_key(self):
        """Key 'SUPER_HIGH' → 400."""
        from api import _validate_routing_policy
        body = self._make_body(
            confidence_overrides={"C": {"SUPER_HIGH": "instant_settle"}},
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Invalid confidence level" in str(exc_info.value.detail)

    def test_confidence_overrides_invalid_action_value(self):
        """Value 'hold' → 400."""
        from api import _validate_routing_policy
        body = self._make_body(
            confidence_overrides={"C": {"HIGH": "hold"}},
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Invalid routing action" in str(exc_info.value.detail)

    def test_confidence_overrides_invalid_grade_key(self):
        """Outer key 'X' → 400."""
        from api import _validate_routing_policy
        body = self._make_body(
            confidence_overrides={"X": {"HIGH": "instant_settle"}},
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "Invalid grade" in str(exc_info.value.detail)

    def test_confidence_overrides_escrow_action_with_escrow_disabled(self):
        """Override maps to 'escrow' + escrow_disabled=true → 400."""
        from api import _validate_routing_policy
        body = self._make_body(
            escrow_disabled=True,
            escrow_grades=[],
            reject_grades=["C", "D", "E", "F"],
            confidence_overrides={"C": {"HIGH": "escrow"}},
        )
        with pytest.raises(Exception) as exc_info:
            _validate_routing_policy(body, None)
        assert "escrow_disabled" in str(exc_info.value.detail)

    def test_confidence_overrides_empty_dict_accepted(self):
        """{} is valid (no-op)."""
        from api import _validate_routing_policy
        body = self._make_body(confidence_overrides={})
        _validate_routing_policy(body, None)  # should not raise

    def test_confidence_overrides_null_accepted(self):
        """null / omitted is valid."""
        from api import _validate_routing_policy
        body = self._make_body(confidence_overrides=None)
        _validate_routing_policy(body, None)  # should not raise

    def test_confidence_overrides_insufficient_accepted(self):
        """INSUFFICIENT is a valid confidence-level key."""
        from api import _validate_routing_policy
        body = self._make_body(
            confidence_overrides={"C": {"INSUFFICIENT": "reject"}},
        )
        _validate_routing_policy(body, None)  # should not raise

    def test_confidence_overrides_case_normalised_on_input(self):
        """Lowercase confidence keys are normalised to uppercase by the model."""
        from api import RoutingPolicyRequest
        body = RoutingPolicyRequest(
            instant_grades=["A", "B"],
            escrow_grades=["C"],
            reject_grades=["D", "E", "F"],
            confidence_overrides={"C": {"high": "instant_settle"}},
        )
        # The field_validator should have uppercased 'high' → 'HIGH'
        assert "HIGH" in body.confidence_overrides["C"]
        assert "high" not in body.confidence_overrides["C"]


# ── Routing policy endpoint tests ────────────────────────────────────────


class TestRoutingPolicyEndpoints:
    """GET/PUT /ahs/route/policy integration tests."""

    def _api_key_headers(self):
        """Create a test API key and return headers + key_hash."""
        import hashlib
        import db as _db
        raw_key = _db.create_api_key(
            customer_email="test-policy@example.com",
            calls_total=1000,
        )
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        from api import INTERNAL_API_KEY
        return {
            "X-API-Key": raw_key,
            "X-Internal-Key": INTERNAL_API_KEY,
        }, key_hash

    def test_get_returns_defaults(self, client):
        """GET without custom policy → default thresholds."""
        headers, _ = self._api_key_headers()
        resp = client.get("/ahs/route/policy", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["instant_grades"] == ["A", "B"]
        assert data["escrow_grades"] == ["C"]
        assert data["reject_grades"] == ["D", "E", "F"]
        assert data["escrow_disabled"] is False
        assert data["allowlist_count"] == 0

    def test_put_and_get_roundtrip(self, client):
        """PUT custom policy → GET returns it."""
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "B", "C"],
            "escrow_grades": ["D"],
            "reject_grades": ["E", "F"],
            "escrow_disabled": False,
        }
        resp = client.put("/ahs/route/policy", json=put_body, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["instant_grades"] == ["A", "B", "C"]
        assert data["escrow_grades"] == ["D"]
        assert data["reject_grades"] == ["E", "F"]
        assert data["updated_at"] != ""

        # GET should return the same
        resp2 = client.get("/ahs/route/policy", headers=headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["instant_grades"] == ["A", "B", "C"]
        assert data2["escrow_disabled"] is False

    @patch("db.get_latest_ahs_for_address")
    def test_put_with_allowlist(self, mock_db, client):
        """PUT with allowlist → stored and counted."""
        allowlisted_addr = "0x" + "ff" * 20
        mock_db.return_value = _mock_record(grade="B", ahs=85, address=allowlisted_addr)
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
            "allowlist": [allowlisted_addr],
        }
        resp = client.put("/ahs/route/policy", json=put_body, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["allowlist_count"] == 1

    def test_401_without_auth(self, client):
        """No API key or x402 → 401."""
        from api import INTERNAL_API_KEY
        # Internal key bypasses x402 but doesn't provide an API key identity
        headers = {"X-Internal-Key": INTERNAL_API_KEY}
        resp = client.get("/ahs/route/policy", headers=headers)
        assert resp.status_code == 401

    def test_put_invalid_grades_400(self, client):
        """PUT with invalid grade → 400."""
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "X"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
        }
        resp = client.put("/ahs/route/policy", json=put_body, headers=headers)
        assert resp.status_code == 400

    @patch("db.get_latest_ahs_for_address")
    def test_route_respects_custom_policy(self, mock_db, client):
        """Route endpoint uses custom policy when set."""
        headers, key_hash = self._api_key_headers()

        # Set policy: A,B,C → instant_settle
        put_body = {
            "instant_grades": ["A", "B", "C"],
            "escrow_grades": ["D"],
            "reject_grades": ["E", "F"],
            "escrow_disabled": False,
        }
        client.put("/ahs/route/policy", json=put_body, headers=headers)

        # Grade C should now be instant_settle (not escrow)
        mock_db.return_value = _mock_record(grade="C", ahs=65)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "instant_settle"
        assert data["policy_applied"] is True

    @patch("db.get_latest_ahs_for_address")
    def test_route_respects_allowlist_bypass(self, mock_db, client):
        """Allowlisted address → instant_settle regardless of grade."""
        allowlisted_addr = "0x" + "ab" * 20
        headers, key_hash = self._api_key_headers()

        # Grade B record for allowlist validation
        mock_db.return_value = _mock_record(grade="B", ahs=85, address=allowlisted_addr)

        # Set policy with allowlist
        put_body = {
            "instant_grades": ["A"],
            "escrow_grades": ["B"],
            "reject_grades": ["C", "D", "E", "F"],
            "escrow_disabled": False,
            "allowlist": [allowlisted_addr],
        }
        client.put("/ahs/route/policy", json=put_body, headers=headers)

        # Route the allowlisted address — Grade F would normally be rejected
        mock_db.return_value = _mock_record(grade="F", ahs=10, address=allowlisted_addr)
        resp = client.get(f"/ahs/route/{allowlisted_addr}", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "instant_settle"
        assert data["allowlisted"] is True
        assert data["policy_applied"] is True

    @patch("db.get_latest_ahs_for_address")
    def test_route_without_policy_unchanged(self, mock_db, client, auth_headers):
        """Route without any policy → default behavior, policy_applied=False."""
        mock_db.return_value = _mock_record(grade="C", ahs=65)
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "escrow"
        assert data["policy_applied"] is False
        assert data["allowlisted"] is False

    # ── Confidence overrides persistence & integration tests ──────────────

    def test_put_with_confidence_overrides_roundtrip(self, client):
        """PUT with overrides → GET returns them in the response."""
        headers, _ = self._api_key_headers()
        overrides = {"C": {"HIGH": "instant_settle"}, "D": {"HIGH": "escrow"}}
        put_body = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
            "confidence_overrides": overrides,
        }
        resp = client.put("/ahs/route/policy", json=put_body, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_overrides"] == overrides

        # GET should return the same
        resp2 = client.get("/ahs/route/policy", headers=headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["confidence_overrides"] == overrides

    def test_put_without_confidence_overrides_backward_compat(self, client):
        """PUT without the field → GET returns null (existing tests still pass)."""
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
        }
        resp = client.put("/ahs/route/policy", json=put_body, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_overrides"] is None

        resp2 = client.get("/ahs/route/policy", headers=headers)
        assert resp2.status_code == 200
        assert resp2.json()["confidence_overrides"] is None

    @patch("db.get_latest_ahs_for_address")
    def test_route_applies_confidence_override(self, mock_db, client):
        """PUT policy with C+HIGH→instant, then route Grade C address with high confidence."""
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        client.put("/ahs/route/policy", json=put_body, headers=headers)

        mock_db.return_value = _mock_record(grade="C", ahs=65, confidence="HIGH")
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "instant_settle"
        assert data["policy_applied"] is True

    @patch("db.get_latest_ahs_for_address")
    def test_route_no_confidence_override_match(self, mock_db, client):
        """PUT policy with C+HIGH→instant, but address has medium confidence → default escrow."""
        headers, _ = self._api_key_headers()
        put_body = {
            "instant_grades": ["A", "B"],
            "escrow_grades": ["C"],
            "reject_grades": ["D", "E", "F"],
            "escrow_disabled": False,
            "confidence_overrides": {"C": {"HIGH": "instant_settle"}},
        }
        client.put("/ahs/route/policy", json=put_body, headers=headers)

        mock_db.return_value = _mock_record(grade="C", ahs=65, confidence="MEDIUM")
        resp = client.get(f"/ahs/route/{VALID_ADDR}", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_recommendation"] == "escrow"
        assert data["policy_applied"] is True


# ── Routing policy DB CRUD tests ─────────────────────────────────────────


class TestRoutingPolicyDB:
    """Direct tests for routing policy DB functions."""

    @pytest.fixture(autouse=True)
    def _init_db(self):
        import db as _db
        _db.init_db()

    def test_get_policy_none(self):
        """No policy → returns None."""
        import db as _db
        assert _db.get_routing_policy("nonexistent-owner") is None

    def test_upsert_creates_policy(self):
        """First upsert creates a new policy."""
        import db as _db
        owner = "test-owner-create"
        result = _db.upsert_routing_policy(owner, "A,B", "C", "D,E,F", False)
        assert result["owner_id"] == owner
        assert result["instant_grades"] == "A,B"
        assert result["escrow_disabled"] == 0

    def test_upsert_updates_policy(self):
        """Second upsert updates the existing policy."""
        import db as _db
        owner = "test-owner-update"
        _db.upsert_routing_policy(owner, "A,B", "C", "D,E,F", False)
        result = _db.upsert_routing_policy(owner, "A,B,C", "D", "E,F", False)
        assert result["instant_grades"] == "A,B,C"
        assert result["escrow_grades"] == "D"

    def test_allowlist_set_and_get(self):
        """Set and get allowlist round-trip."""
        import db as _db
        owner = "test-owner-allowlist"
        addrs = ["0x" + "aa" * 20, "0x" + "bb" * 20]
        count = _db.set_routing_allowlist(owner, addrs)
        assert count == 2
        result = _db.get_routing_allowlist(owner)
        assert len(result) == 2
        assert ("0x" + "aa" * 20) in result

    def test_allowlist_replace(self):
        """Setting allowlist replaces the previous one."""
        import db as _db
        owner = "test-owner-replace"
        _db.set_routing_allowlist(owner, ["0x" + "aa" * 20, "0x" + "bb" * 20])
        _db.set_routing_allowlist(owner, ["0x" + "cc" * 20])
        result = _db.get_routing_allowlist(owner)
        assert len(result) == 1
        assert result[0] == "0x" + "cc" * 20

    def test_is_address_allowlisted(self):
        """Check individual address allowlist membership."""
        import db as _db
        owner = "test-owner-check"
        _db.set_routing_allowlist(owner, ["0x" + "aa" * 20])
        assert _db.is_address_allowlisted(owner, "0x" + "aa" * 20) is True
        assert _db.is_address_allowlisted(owner, "0x" + "bb" * 20) is False

    def test_owner_id_wallet_format(self):
        """owner_id works with wallet address format."""
        import db as _db
        owner = "0x" + "dd" * 20  # looks like a wallet address
        result = _db.upsert_routing_policy(owner, "A,B,C", "D", "E,F", False)
        assert result["owner_id"] == owner
        fetched = _db.get_routing_policy(owner)
        assert fetched is not None
        assert fetched["instant_grades"] == "A,B,C"
