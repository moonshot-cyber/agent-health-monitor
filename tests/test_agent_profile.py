"""Tests for GET /internal/agent-profile/{address} endpoint."""

import pytest

from api import INTERNAL_API_KEY


def _auth_headers():
    return {"X-Internal-Key": INTERNAL_API_KEY}


class TestAgentProfileAuth:
    """Authentication tests for the internal agent-profile endpoint."""

    def test_missing_key_returns_403(self, client):
        resp = client.get("/internal/agent-profile/0xabc123")
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self, client):
        resp = client.get(
            "/internal/agent-profile/0xabc123",
            headers={"X-Internal-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_valid_key_returns_200(self, client):
        resp = client.get(
            "/internal/agent-profile/0xabc123",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200


class TestAgentProfileUnrated:
    """Tests for addresses with no scan history."""

    def test_unknown_address_returns_unrated(self, client):
        resp = client.get(
            "/internal/agent-profile/0x0000000000000000000000000000000000000000",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unrated"
        assert "address" in data

    def test_unrated_address_is_lowercased(self, client):
        resp = client.get(
            "/internal/agent-profile/0xABCDEF",
            headers=_auth_headers(),
        )
        data = resp.json()
        assert data["address"] == "0xabcdef"


class TestAgentProfileWithData:
    """Tests for addresses that have scan history."""

    def _seed_scan(self, address="0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"):
        """Insert a minimal scan record for testing."""
        import db as scan_db

        scan_db.init_db()
        conn = scan_db.get_connection()
        try:
            conn.execute(
                """INSERT INTO scans
                   (address, endpoint, scan_timestamp, ahs_score, grade, grade_label,
                    d1_score, d2_score, source, confidence, mode)
                   VALUES (?, 'ahs', '2026-04-11T12:00:00Z', 67, 'C', 'Needs Attention',
                           72, 64, 'acp', 'HIGH', '2D')""",
                (address.lower(),),
            )
            scan_id = conn.execute(
                "SELECT id FROM scans WHERE address = ? ORDER BY id DESC LIMIT 1",
                (address.lower(),),
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO scan_patterns (scan_id, pattern_name, severity, description, modifier)
                   VALUES (?, 'consistent_activity', 'info', 'Regular tx pattern', 0)""",
                (scan_id,),
            )
            conn.execute(
                """INSERT INTO scan_patterns (scan_id, pattern_name, severity, description, modifier)
                   VALUES (?, 'low_balance_risk', 'medium', 'Balance below threshold', -5)""",
                (scan_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def test_known_address_returns_profile(self, client):
        addr = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"
        self._seed_scan(addr)

        resp = client.get(
            f"/internal/agent-profile/{addr}",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["address"] == addr
        assert data["ahs_score"] == 67
        assert data["grade"] == "C"
        assert data["d1_score"] == 72
        assert data["d2_score"] == 64
        assert data["source"] == "acp"
        assert data["last_scanned"] == "2026-04-11T12:00:00Z"
        assert "status" not in data

    def test_patterns_included(self, client):
        addr = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"
        self._seed_scan(addr)

        resp = client.get(
            f"/internal/agent-profile/{addr}",
            headers=_auth_headers(),
        )
        data = resp.json()
        assert "patterns" in data
        assert isinstance(data["patterns"], list)
        assert "consistent_activity" in data["patterns"]
        assert "low_balance_risk" in data["patterns"]

    def test_address_case_insensitive(self, client):
        addr_lower = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"
        addr_upper = "0xDE0B295669A9FD93D5F28D9EC85E40F4CB697BAE"
        self._seed_scan(addr_lower)

        resp = client.get(
            f"/internal/agent-profile/{addr_upper}",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ahs_score"] == 67
