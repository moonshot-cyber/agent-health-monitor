"""Tests for the leaderboard section of generate_intelligence_snapshot.py.

Covers: healthiest ranking (sorted, contiguous ranks, named-only, 500 cap),
counts split (named vs unnamed), tie-breaking by last_scanned_at, and
backward compatibility (existing snapshot keys preserved).
"""

import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from scripts.generate_intelligence_snapshot import generate_snapshot, generate_snapshot_from_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_db(db_path: str, wallets: list[dict], scans: list[dict] | None = None):
    """Create a minimal DB with known_wallets and scans tables, then seed rows.

    wallets: list of dicts with keys matching known_wallets columns.
    scans: optional list of dicts for the scans table (needed for ecosystem stats).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS known_wallets (
            address               TEXT PRIMARY KEY,
            label                 TEXT,
            source                TEXT,
            first_seen_at         TEXT NOT NULL,
            last_scanned_at       TEXT,
            scan_count            INTEGER NOT NULL DEFAULT 0,
            latest_ahs            INTEGER,
            latest_grade          TEXT,
            rescan_enabled        INTEGER NOT NULL DEFAULT 1,
            rescan_interval_hours INTEGER NOT NULL DEFAULT 168,
            registries            TEXT DEFAULT '',
            agent_name            TEXT
        );
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            address         TEXT,
            endpoint        TEXT,
            scan_timestamp  TEXT,
            ahs_score       INTEGER,
            grade           TEXT,
            response_data   TEXT
        );
    """)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for w in wallets:
        conn.execute(
            """INSERT INTO known_wallets
               (address, label, source, first_seen_at, last_scanned_at,
                scan_count, latest_ahs, latest_grade, registries, agent_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                w["address"],
                w.get("label", ""),
                w.get("source", "api"),
                now,
                w.get("last_scanned_at", now),
                w.get("scan_count", 1),
                w.get("latest_ahs"),
                w.get("latest_grade"),
                w.get("registries", ""),
                w.get("agent_name"),
            ),
        )

    # Seed at least one scan so ecosystem stats don't return all zeros.
    if scans:
        for s in scans:
            conn.execute(
                "INSERT INTO scans (address, endpoint, scan_timestamp, ahs_score, grade) "
                "VALUES (?, ?, ?, ?, ?)",
                (s["address"], "ahs", s.get("scan_timestamp", now),
                 s.get("ahs_score"), s.get("grade")),
            )
    else:
        # Default: mirror the wallet data into scans so ecosystem section works.
        for w in wallets:
            if w.get("latest_ahs") is not None:
                conn.execute(
                    "INSERT INTO scans (address, endpoint, scan_timestamp, ahs_score, grade) "
                    "VALUES (?, 'ahs', ?, ?, ?)",
                    (w["address"], w.get("last_scanned_at", now),
                     w["latest_ahs"], w.get("latest_grade")),
                )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_WALLETS = [
    # Named + scored — should appear on leaderboard
    {"address": "0x" + "a1" * 20, "agent_name": "AlphaBot", "latest_ahs": 92,
     "latest_grade": "A", "source": "acp_proactive_scan", "registries": "acp",
     "last_scanned_at": "2026-05-20T10:00:00Z"},
    {"address": "0x" + "b2" * 20, "agent_name": "BravoAgent", "latest_ahs": 85,
     "latest_grade": "B", "source": "erc8004_scan", "registries": "erc8004",
     "last_scanned_at": "2026-05-19T10:00:00Z"},
    {"address": "0x" + "c3" * 20, "agent_name": "CharlieBot", "latest_ahs": 72,
     "latest_grade": "C", "source": "celo_agent_wallet", "registries": "celo",
     "last_scanned_at": "2026-05-18T10:00:00Z"},
    {"address": "0x" + "d4" * 20, "agent_name": "DeltaService", "latest_ahs": 55,
     "latest_grade": "D", "source": "arc_agent_wallet", "registries": "arc",
     "last_scanned_at": "2026-05-17T10:00:00Z"},

    # Unnamed + scored — should NOT appear on leaderboard, but count as unnamed_scored
    {"address": "0x" + "e5" * 20, "agent_name": None, "latest_ahs": 88,
     "latest_grade": "B", "source": "olas_service", "registries": "olas"},
    {"address": "0x" + "f6" * 20, "agent_name": "", "latest_ahs": 60,
     "latest_grade": "C", "source": "celo_owner", "registries": "celo"},

    # Named but NOT scored — should NOT appear anywhere in leaderboard
    {"address": "0x" + "07" * 20, "agent_name": "UnscoredBot", "latest_ahs": None,
     "latest_grade": None, "source": "api", "registries": ""},

    # Whitespace-only name — treated as unnamed
    {"address": "0x" + "08" * 20, "agent_name": "   ", "latest_ahs": 70,
     "latest_grade": "C", "source": "api", "registries": ""},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSnapshotLeaderboard:

    def test_healthiest_sorted_descending(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        lb = snapshot["leaderboard"]
        scores = [r["ahs"] for r in lb["healthiest"]]
        assert scores == sorted(scores, reverse=True), (
            f"healthiest not sorted descending: {scores}"
        )

    def test_ranks_contiguous(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        ranks = [r["rank"] for r in snapshot["leaderboard"]["healthiest"]]
        assert ranks == list(range(1, len(ranks) + 1)), (
            f"Ranks not contiguous 1..N: {ranks}"
        )

    def test_only_named_agents(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        for r in snapshot["leaderboard"]["healthiest"]:
            assert r["agent_name"] is not None and r["agent_name"].strip() != "", (
                f"Unnamed agent in leaderboard: {r}"
            )

    def test_unnamed_excluded(self, tmp_path):
        """Agents with NULL, empty, or whitespace-only agent_name are excluded."""
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        addresses = {r["address"] for r in snapshot["leaderboard"]["healthiest"]}
        # e5 has agent_name=None, f6 has agent_name="", 08 has agent_name="   "
        assert "0x" + "e5" * 20 not in addresses
        assert "0x" + "f6" * 20 not in addresses
        assert "0x" + "08" * 20 not in addresses

    def test_unscored_excluded(self, tmp_path):
        """Agent with agent_name but latest_ahs=NULL is excluded."""
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        addresses = {r["address"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "0x" + "07" * 20 not in addresses

    def test_record_fields(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        expected_keys = {"rank", "address", "agent_name", "ahs", "grade", "source", "registries"}
        for r in snapshot["leaderboard"]["healthiest"]:
            assert set(r.keys()) == expected_keys, (
                f"Unexpected fields: {set(r.keys()) - expected_keys}"
            )

    def test_counts_split(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        counts = snapshot["leaderboard"]["counts"]
        # Named + scored: AlphaBot, BravoAgent, CharlieBot, DeltaService = 4
        assert counts["named_scored"] == 4
        # Unnamed + scored: e5 (None), f6 (""), 08 ("   ") = 3
        assert counts["unnamed_scored"] == 3
        assert counts["total_scored"] == 7
        assert counts["named_scored"] + counts["unnamed_scored"] == counts["total_scored"]

    def test_cap_at_500(self, tmp_path):
        """More than 500 named agents → leaderboard capped at 500."""
        db = str(tmp_path / "test_cap.db")
        wallets = [
            {"address": f"0x{i:040x}", "agent_name": f"Agent{i}",
             "latest_ahs": max(1, 100 - (i % 100)), "latest_grade": "C",
             "source": "api", "registries": ""}
            for i in range(600)
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        assert len(snapshot["leaderboard"]["healthiest"]) == 500
        assert snapshot["leaderboard"]["counts"]["named_scored"] == 600

    def test_tiebreak_by_last_scanned_at(self, tmp_path):
        """Equal AHS → most recently scanned first."""
        db = str(tmp_path / "test_tie.db")
        wallets = [
            {"address": "0x" + "01" * 20, "agent_name": "OlderScan", "latest_ahs": 80,
             "latest_grade": "B", "source": "api", "registries": "",
             "last_scanned_at": "2026-05-01T10:00:00Z"},
            {"address": "0x" + "02" * 20, "agent_name": "NewerScan", "latest_ahs": 80,
             "latest_grade": "B", "source": "api", "registries": "",
             "last_scanned_at": "2026-05-20T10:00:00Z"},
            {"address": "0x" + "03" * 20, "agent_name": "NullScan", "latest_ahs": 80,
             "latest_grade": "B", "source": "api", "registries": "",
             "last_scanned_at": None},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        names = [r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]]
        assert names == ["NewerScan", "OlderScan", "NullScan"], (
            f"Tie-breaking order wrong: {names}"
        )

    def test_existing_keys_preserved(self, tmp_path):
        """Adding leaderboard does not remove existing top-level keys."""
        db = str(tmp_path / "test_keys.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        for key in ("generated_at", "ecosystem", "registries", "daily_stats"):
            assert key in snapshot, f"Existing key '{key}' missing from snapshot"

    def test_leaderboard_has_generated_at(self, tmp_path):
        db = str(tmp_path / "test_gen.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        assert "generated_at" in snapshot["leaderboard"]

    def test_no_methodology_internals(self, tmp_path):
        """Leaderboard records must not expose D1/D2/weights/patterns."""
        db = str(tmp_path / "test_fields.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        forbidden = {"d1", "d2", "d1_score", "d2_score", "patterns", "weights",
                     "confidence", "signals", "mode"}
        for r in snapshot["leaderboard"]["healthiest"]:
            leaked = forbidden & set(r.keys())
            assert not leaked, f"Methodology internals leaked: {leaked}"


# ---------------------------------------------------------------------------
# Shared helper parity tests — db.get_leaderboard_data vs generate_snapshot
# ---------------------------------------------------------------------------

class TestSharedHelperParity:
    """The --db path and the endpoint helper must produce structurally
    identical leaderboard output for the same DB state."""

    def test_same_keys_and_order(self, tmp_path):
        """Shared helper and --db snapshot produce the same leaderboard keys,
        ranking order, and count values."""
        import db as db_module

        db_path = str(tmp_path / "parity.db")
        _seed_db(db_path, _WALLETS)

        # --db path via generate_snapshot
        snapshot = generate_snapshot(db_path)
        snapshot_lb = snapshot["leaderboard"]

        # Shared helper with explicit connection (same path the script now uses)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            helper_lb = db_module.get_leaderboard_data(conn=conn)
        finally:
            conn.close()

        # Same top-level keys
        assert set(snapshot_lb.keys()) == set(helper_lb.keys())

        # Same ranking order (addresses in same sequence)
        snap_addrs = [r["address"] for r in snapshot_lb["healthiest"]]
        help_addrs = [r["address"] for r in helper_lb["healthiest"]]
        assert snap_addrs == help_addrs

        # Same counts
        assert snapshot_lb["counts"] == helper_lb["counts"]

        # Same record fields
        for s, h in zip(snapshot_lb["healthiest"], helper_lb["healthiest"]):
            assert set(s.keys()) == set(h.keys())
            assert s["rank"] == h["rank"]
            assert s["ahs"] == h["ahs"]
            assert s["grade"] == h["grade"]


# ---------------------------------------------------------------------------
# /api/leaderboard endpoint tests
# ---------------------------------------------------------------------------

class TestLeaderboardEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/api/leaderboard")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        resp = client.get("/api/leaderboard")
        data = resp.json()
        assert "healthiest" in data
        assert "counts" in data
        assert "generated_at" in data
        assert isinstance(data["healthiest"], list)
        assert isinstance(data["counts"], dict)
        for key in ("named_scored", "unnamed_scored", "total_scored"):
            assert key in data["counts"]

    def test_no_methodology_internals(self, client):
        resp = client.get("/api/leaderboard")
        data = resp.json()
        forbidden = {"d1", "d2", "d1_score", "d2_score", "patterns", "weights",
                     "confidence", "signals", "mode"}
        for r in data["healthiest"]:
            leaked = forbidden & set(r.keys())
            assert not leaked, f"Methodology internals leaked: {leaked}"


# ---------------------------------------------------------------------------
# --api path tests (generate_snapshot_from_api with mocked HTTP)
# ---------------------------------------------------------------------------

_MOCK_ECOSYSTEM = {
    "status": "ok",
    "total_scanned": 100,
    "avg_ahs": 60.0,
    "grade_distribution": {"A": 5, "B": 20, "C": 30, "D": 35, "E": 8, "F": 2},
    "data_sources": {"ACP": 60, "ERC-8004": 40},
    "registry_stats": {
        "ACP": {"avg_score": 55.0, "grade_distribution": {"C": 30, "D": 30}, "top_grade_pct": 0.0},
        "ERC-8004": {"avg_score": 65.0, "grade_distribution": {"B": 20, "C": 20}, "top_grade_pct": 10.0},
    },
    "endpoint_count": 10,
}

_MOCK_QUALITY = {
    "batches": [
        {"batch_date": "2026-05-20T00:00:00Z", "wallets_scanned": 50, "average_ahs": 60.0},
    ]
}

_MOCK_LEADERBOARD = {
    "healthiest": [
        {"rank": 1, "address": "0xaaa", "agent_name": "TestBot", "ahs": 90,
         "grade": "A", "source": "ACP", "registries": "acp"},
    ],
    "counts": {"named_scored": 10, "unnamed_scored": 90, "total_scored": 100},
    "generated_at": "2026-05-23T10:00:00Z",
}


class TestApiPathLeaderboard:

    def _mock_fetch(self, responses: dict):
        """Return a side_effect function that dispatches by URL pattern."""
        def _fetch(url):
            for pattern, data in responses.items():
                if pattern in url:
                    return data
            raise Exception(f"Unmocked URL: {url}")
        return _fetch

    def test_includes_leaderboard_on_success(self):
        responses = {
            "/api/ecosystem-stats": _MOCK_ECOSYSTEM,
            "/scan/quality": _MOCK_QUALITY,
            "/api/leaderboard": _MOCK_LEADERBOARD,
        }
        with patch("scripts.generate_intelligence_snapshot._fetch_json",
                    side_effect=self._mock_fetch(responses)):
            snapshot = generate_snapshot_from_api("https://example.com")

        assert "leaderboard" in snapshot
        assert snapshot["leaderboard"]["healthiest"][0]["agent_name"] == "TestBot"
        assert snapshot["leaderboard"]["counts"]["total_scored"] == 100

    def test_omits_leaderboard_on_failure(self):
        def _fetch_with_failure(url):
            if "/api/leaderboard" in url:
                raise Exception("Connection refused")
            if "/api/ecosystem-stats" in url:
                return _MOCK_ECOSYSTEM
            if "/scan/quality" in url:
                return _MOCK_QUALITY
            raise Exception(f"Unmocked URL: {url}")

        with patch("scripts.generate_intelligence_snapshot._fetch_json",
                    side_effect=_fetch_with_failure):
            snapshot = generate_snapshot_from_api("https://example.com")

        assert "leaderboard" not in snapshot
        # Other keys must still be present
        for key in ("generated_at", "ecosystem", "registries", "daily_stats"):
            assert key in snapshot, f"Key '{key}' missing after leaderboard failure"

    def test_api_and_db_same_top_level_keys(self):
        """When leaderboard fetch succeeds, --api and --db produce the same
        set of top-level keys."""
        responses = {
            "/api/ecosystem-stats": _MOCK_ECOSYSTEM,
            "/scan/quality": _MOCK_QUALITY,
            "/api/leaderboard": _MOCK_LEADERBOARD,
        }
        with patch("scripts.generate_intelligence_snapshot._fetch_json",
                    side_effect=self._mock_fetch(responses)):
            api_snapshot = generate_snapshot_from_api("https://example.com")

        expected_keys = {"generated_at", "ecosystem", "registries", "daily_stats", "leaderboard"}
        assert set(api_snapshot.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Junk-data filter unit tests
# ---------------------------------------------------------------------------

from db import is_mash_name, has_banned_keyword


class TestIsMashName:
    """Unit tests for the keyboard-mash heuristic."""

    # --- MUST be excluded (junk) ---

    def test_mash_digit_heavy(self):
        assert is_mash_name("6w4r4y567y563745678") is True

    def test_mash_no_alpha_run(self):
        assert is_mash_name("6w4rfg5eqr3gfdfg") is True

    def test_mash_with_spaces_digit_heavy(self):
        assert is_mash_name("2344rht4232fs6d 426 7 y21") is True

    def test_mash_long_random(self):
        assert is_mash_name("tyju5763ju76758dfg52567") is True

    # --- MUST be kept (legitimate) ---

    def test_keep_zyfai(self):
        assert is_mash_name("Zyfai Rebalancer Agent for 0x...") is False

    def test_keep_uniagent(self):
        assert is_mash_name("UniAgent ERC8004 Agent #738") is False

    def test_keep_marcus(self):
        assert is_mash_name("Marcus Aurelius") is False

    def test_keep_agentracheel(self):
        assert is_mash_name("agentracheel") is False

    # --- Short names (rule 4) — always kept ---

    def test_keep_short_V(self):
        assert is_mash_name("V") is False

    def test_keep_short_C(self):
        assert is_mash_name("C") is False

    def test_keep_short_Bb(self):
        assert is_mash_name("Bb") is False

    def test_keep_short_kai(self):
        assert is_mash_name("kai") is False

    def test_keep_short_006(self):
        assert is_mash_name("006") is False

    def test_keep_short_1024(self):
        assert is_mash_name("1024") is False

    def test_keep_short_3Jane(self):
        assert is_mash_name("3Jane") is False

    def test_keep_short_aa(self):
        assert is_mash_name("aa") is False


class TestHasBannedKeyword:
    """Unit tests for whole-word banned-keyword matching."""

    # --- Should be excluded (keyword match) ---

    def test_test_client(self):
        assert has_banned_keyword("Test Client") is True

    def test_ag_test_bot(self):
        assert has_banned_keyword("ag-test-bot") is True

    def test_jarvis_ceo_test(self):
        assert has_banned_keyword("JARVIS-CEO-Test") is True

    # --- Should be kept (substring only, not whole word) ---

    def test_contest_tracker(self):
        assert has_banned_keyword("Contest Tracker") is False

    def test_latest_signal(self):
        assert has_banned_keyword("Latest Signal") is False

    def test_attestation_agent(self):
        assert has_banned_keyword("Attestation Agent") is False


# ---------------------------------------------------------------------------
# Integration: get_leaderboard_data with junk filtering
# ---------------------------------------------------------------------------

class TestLeaderboardJunkFilter:
    """Integration tests: junk entries are excluded, legit entries kept,
    counts unchanged, and ranks remain contiguous."""

    _JUNK_WALLETS = [
        # Legitimate — should appear on leaderboard
        {"address": "0x" + "a1" * 20, "agent_name": "AlphaBot", "latest_ahs": 92,
         "latest_grade": "A", "source": "acp", "registries": "acp",
         "last_scanned_at": "2026-05-20T10:00:00Z"},
        {"address": "0x" + "b2" * 20, "agent_name": "BravoAgent", "latest_ahs": 85,
         "latest_grade": "B", "source": "erc8004", "registries": "erc8004",
         "last_scanned_at": "2026-05-19T10:00:00Z"},
        # Short legit name — must be kept
        {"address": "0x" + "c3" * 20, "agent_name": "kai", "latest_ahs": 78,
         "latest_grade": "B", "source": "api", "registries": "",
         "last_scanned_at": "2026-05-18T10:00:00Z"},
        # Burn address — must be excluded
        {"address": "0x000000000000000000000000000000000000dead",
         "agent_name": "DeadAddr", "latest_ahs": 99, "latest_grade": "A",
         "source": "api", "registries": ""},
        # All-zero address — must be excluded
        {"address": "0x0000000000000000000000000000000000000000",
         "agent_name": "NullAddr", "latest_ahs": 95, "latest_grade": "A",
         "source": "api", "registries": ""},
        # Keyboard-mash name — must be excluded
        {"address": "0x" + "d4" * 20, "agent_name": "6w4r4y567y563745678",
         "latest_ahs": 90, "latest_grade": "A", "source": "api", "registries": ""},
        # Banned keyword — must be excluded
        {"address": "0x" + "e5" * 20, "agent_name": "Test Client",
         "latest_ahs": 88, "latest_grade": "B", "source": "api", "registries": ""},
        # Substring "test" in "Contest" — must be KEPT
        {"address": "0x" + "f6" * 20, "agent_name": "Contest Tracker",
         "latest_ahs": 80, "latest_grade": "B", "source": "api", "registries": ""},
        # Unnamed / unscored for count verification
        {"address": "0x" + "07" * 20, "agent_name": None, "latest_ahs": 60,
         "latest_grade": "C", "source": "api", "registries": ""},
    ]

    def test_excludes_burn_addresses(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        addrs = {r["address"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "0x000000000000000000000000000000000000dead" not in addrs
        assert "0x0000000000000000000000000000000000000000" not in addrs

    def test_excludes_mash_names(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "6w4r4y567y563745678" not in names

    def test_excludes_banned_keyword(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "Test Client" not in names

    def test_keeps_substring_keyword(self, tmp_path):
        """'Contest Tracker' contains 'test' as substring only — must be kept."""
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "Contest Tracker" in names

    def test_keeps_short_names(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "kai" in names

    def test_keeps_legit_entries(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "AlphaBot" in names
        assert "BravoAgent" in names

    def test_ranks_contiguous_after_filter(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        ranks = [r["rank"] for r in snapshot["leaderboard"]["healthiest"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_counts_unaffected_by_filter(self, tmp_path):
        """Counts must reflect the TRUE scored population, not the filtered board."""
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        counts = snapshot["leaderboard"]["counts"]
        # All wallets with latest_ahs != None: 8 total (a1,b2,c3,dead,null,d4,e5,f6,07 → 9 wallets, 07 has ahs=60 so 9 total? let's count)
        # a1(92), b2(85), c3(78), dead(99), null-addr(95), d4(90), e5(88), f6(80), 07(60) = 9 scored
        # Named+scored: a1,b2,c3,dead,null-addr,d4,e5,f6 = 8
        # Unnamed+scored: 07 = 1
        assert counts["total_scored"] == 9
        assert counts["named_scored"] == 8
        assert counts["unnamed_scored"] == 1

    def test_filter_fills_500_with_clean(self, tmp_path):
        """When there are >500 clean entries + junk, the board still caps at 500
        with clean entries only."""
        db = str(tmp_path / "fill.db")
        wallets = [
            {"address": f"0x{i:040x}", "agent_name": f"Agent{i}",
             "latest_ahs": max(1, 100 - (i % 100)), "latest_grade": "C",
             "source": "api", "registries": ""}
            for i in range(510)
        ]
        # Add some junk that would have ranked high
        wallets.append({
            "address": "0x000000000000000000000000000000000000dead",
            "agent_name": "BurnBot", "latest_ahs": 100, "latest_grade": "A",
            "source": "api", "registries": "",
        })
        wallets.append({
            "address": "0x" + "ff" * 20, "agent_name": "6w4r4y567y563745678",
            "latest_ahs": 99, "latest_grade": "A", "source": "api", "registries": "",
        })
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)
        lb = snapshot["leaderboard"]
        assert len(lb["healthiest"]) == 500
        names = {r["agent_name"] for r in lb["healthiest"]}
        assert "BurnBot" not in names
        assert "6w4r4y567y563745678" not in names
