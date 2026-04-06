"""Tests for the Tiered Trust Routing feature.

Covers:
- _trust_routing() helper mappings
- GET /ahs/route/{address} endpoint: success, stale flag, 404
"""

import os
import sys

os.environ.setdefault("PAYMENT_ADDRESS", "0x" + "a" * 40)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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


def _mock_record(grade="B", ahs=82, confidence="high", hours_ago=2):
    """Build a mock known_wallets row for get_latest_ahs_for_address."""
    scored_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "address": VALID_ADDR,
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
