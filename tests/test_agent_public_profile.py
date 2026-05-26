"""Tests for GET /api/agent/{address} — public agent profile endpoint.

Covers: allowlist security (no paid fields leak), all free-safe fields
present, unrated handling, gap_to_next_rank, gap_to_next_tier, and
named-but-unranked agents.
"""

import pytest

import db as scan_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_wallet(
    address,
    agent_name=None,
    latest_ahs=None,
    latest_grade=None,
    latest_d1=None,
    latest_d2=None,
    rank=None,
    percentile_rank=None,
    source="api",
    registries="",
):
    """Insert a known_wallets row with the given fields."""
    conn = scan_db.get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO known_wallets
               (address, label, source, first_seen_at, last_scanned_at,
                scan_count, latest_ahs, latest_grade, registries, agent_name,
                latest_d1, latest_d2, rank, percentile_rank)
            VALUES (?, '', ?, datetime('now'), datetime('now'),
                    1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                address.lower(),
                source,
                latest_ahs,
                latest_grade,
                registries,
                agent_name,
                latest_d1,
                latest_d2,
                rank,
                percentile_rank,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Security: paid fields must NOT appear
# ---------------------------------------------------------------------------

class TestAllowlistSecurity:
    """The response model must structurally exclude paid-only fields."""

    def test_d1_d2_absent_from_response(self, client):
        """latest_d1 and latest_d2 must never appear in the JSON response."""
        addr = "0x" + "a1" * 20
        _seed_wallet(
            addr,
            agent_name="SecTest Agent",
            latest_ahs=82,
            latest_grade="B",
            latest_d1=75,
            latest_d2=68,
            rank=5,
            percentile_rank=88.5,
        )
        resp = client.get(f"/api/agent/{addr}")
        assert resp.status_code == 200
        data = resp.json()
        # Key security assertion: dimensional scores must not leak
        assert "latest_d1" not in data
        assert "latest_d2" not in data
        assert "d1_score" not in data
        assert "d2_score" not in data
        assert "shadow_signals_json" not in data
        assert "response_json" not in data
        assert "confidence" not in data
        assert "mode" not in data
        assert "cdp_modifier" not in data


# ---------------------------------------------------------------------------
# All free-safe fields present
# ---------------------------------------------------------------------------

class TestFreeSafeFields:
    """All 13 allowlisted fields must appear in the response."""

    def test_all_fields_present(self, client):
        addr = "0x" + "b2" * 20
        _seed_wallet(
            addr,
            agent_name="Field Check",
            latest_ahs=72,
            latest_grade="C",
            rank=10,
            percentile_rank=65.3,
        )
        resp = client.get(f"/api/agent/{addr}")
        assert resp.status_code == 200
        data = resp.json()
        expected_fields = {
            "address", "agent_name", "registries", "source",
            "latest_ahs", "latest_grade", "rank", "percentile_rank",
            "first_seen_at", "last_scanned_at", "scan_count",
            "gap_to_next_rank", "gap_to_next_tier",
        }
        assert expected_fields == set(data.keys())


# ---------------------------------------------------------------------------
# Unrated handling
# ---------------------------------------------------------------------------

class TestUnrated:
    """Unknown/unscanned addresses return 200 + status=unrated."""

    def test_unknown_address_returns_unrated(self, client):
        resp = client.get(
            "/api/agent/0x0000000000000000000000000000000000099999"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unrated"
        assert "address" in data

    def test_unrated_address_is_lowercased(self, client):
        resp = client.get("/api/agent/0xABCDEF1234567890ABCDEF1234567890ABCDEF12")
        data = resp.json()
        assert data["address"] == "0xabcdef1234567890abcdef1234567890abcdef12"


# ---------------------------------------------------------------------------
# gap_to_next_rank
# ---------------------------------------------------------------------------

class TestGapToNextRank:
    """gap_to_next_rank = AHS of agent at rank-1 minus current AHS."""

    def test_gap_computed_correctly(self, client):
        """Rank 2 (ahs=85) with rank 1 (ahs=95) → gap = 10."""
        # Clear existing ranks so only test-seeded agents have rank values
        conn = scan_db.get_connection()
        try:
            conn.execute("UPDATE known_wallets SET rank = NULL")
            conn.commit()
        finally:
            conn.close()

        addr_r1 = "0x" + "c3" * 20
        addr_r2 = "0x" + "c4" * 20
        _seed_wallet(addr_r1, agent_name="Rank One", latest_ahs=95,
                     latest_grade="A", rank=1, percentile_rank=99.0)
        _seed_wallet(addr_r2, agent_name="Rank Two", latest_ahs=85,
                     latest_grade="B", rank=2, percentile_rank=90.0)

        resp = client.get(f"/api/agent/{addr_r2}")
        data = resp.json()
        assert data["gap_to_next_rank"] == 10

    def test_rank_1_has_null_gap(self, client):
        """Rank 1 → gap_to_next_rank is null."""
        addr = "0x" + "c5" * 20
        _seed_wallet(addr, agent_name="Top Agent", latest_ahs=98,
                     latest_grade="A", rank=1, percentile_rank=99.5)

        resp = client.get(f"/api/agent/{addr}")
        data = resp.json()
        assert data["gap_to_next_rank"] is None

    def test_unranked_agent_has_null_gap(self, client):
        """rank=NULL → gap_to_next_rank is null."""
        addr = "0x" + "c6" * 20
        _seed_wallet(addr, agent_name="Unranked", latest_ahs=50,
                     latest_grade="D", rank=None, percentile_rank=30.0)

        resp = client.get(f"/api/agent/{addr}")
        data = resp.json()
        assert data["gap_to_next_rank"] is None


# ---------------------------------------------------------------------------
# gap_to_next_tier
# ---------------------------------------------------------------------------

class TestGapToNextTier:
    """gap_to_next_tier = next grade's lower bound minus current AHS."""

    def test_grade_b_gap(self, client):
        """Grade B (ahs=82) → next tier A starts at 90 → gap = 8."""
        addr = "0x" + "d7" * 20
        _seed_wallet(addr, agent_name="B Agent", latest_ahs=82,
                     latest_grade="B", rank=3, percentile_rank=85.0)

        resp = client.get(f"/api/agent/{addr}")
        data = resp.json()
        assert data["gap_to_next_tier"] == 8

    def test_grade_a_has_null_gap(self, client):
        """Grade A → already top tier → gap is null."""
        addr = "0x" + "d8" * 20
        _seed_wallet(addr, agent_name="A Agent", latest_ahs=95,
                     latest_grade="A", rank=1, percentile_rank=99.0)

        resp = client.get(f"/api/agent/{addr}")
        data = resp.json()
        assert data["gap_to_next_tier"] is None

    def test_grade_d_gap(self, client):
        """Grade D (ahs=45) → next tier C starts at 60 → gap = 15."""
        addr = "0x" + "d9" * 20
        _seed_wallet(addr, agent_name="D Agent", latest_ahs=45,
                     latest_grade="D", rank=None, percentile_rank=20.0)

        resp = client.get(f"/api/agent/{addr}")
        data = resp.json()
        assert data["gap_to_next_tier"] == 15


# ---------------------------------------------------------------------------
# Named-but-unranked agent
# ---------------------------------------------------------------------------

class TestNamedButUnranked:
    """An agent with agent_name but rank=NULL should return a valid response."""

    def test_valid_response_with_null_rank(self, client):
        addr = "0x" + "e0" * 20
        _seed_wallet(addr, agent_name="Named Unranked", latest_ahs=55,
                     latest_grade="D", rank=None, percentile_rank=25.0)

        resp = client.get(f"/api/agent/{addr}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_name"] == "Named Unranked"
        assert data["rank"] is None
        assert data["gap_to_next_rank"] is None
        # percentile_rank should still be populated
        assert data["percentile_rank"] == 25.0
