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

import scripts.generate_intelligence_snapshot as snapshot_mod
from scripts.generate_intelligence_snapshot import generate_snapshot, generate_snapshot_from_api


# ---------------------------------------------------------------------------
# Lazy db import — the module must not import get_leaderboard_data at load
# ---------------------------------------------------------------------------

class TestLazyDbImport:
    """The script must not have a module-level dependency on db.py so
    it can be fetched standalone and run in --api mode."""

    def test_get_leaderboard_data_not_at_module_scope(self):
        """get_leaderboard_data should NOT be in the module namespace —
        it is imported lazily inside generate_snapshot()."""
        assert not hasattr(snapshot_mod, "get_leaderboard_data"), (
            "get_leaderboard_data is imported at module scope; "
            "it should be a lazy import inside generate_snapshot()"
        )


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
            agent_name            TEXT,
            latest_d1             INTEGER,
            latest_d2             INTEGER
        );
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            address         TEXT,
            endpoint        TEXT,
            scan_timestamp  TEXT,
            ahs_score       INTEGER,
            grade           TEXT,
            response_data   TEXT,
            d1_score        INTEGER,
            d2_score        INTEGER
        );
    """)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for w in wallets:
        conn.execute(
            """INSERT INTO known_wallets
               (address, label, source, first_seen_at, last_scanned_at,
                scan_count, latest_ahs, latest_grade, registries, agent_name,
                latest_d1, latest_d2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                w.get("latest_d1"),
                w.get("latest_d2"),
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
    # Named + scored — should appear on leaderboard.
    # D1/D2 values chosen so the three boards produce different orderings.
    #   AHS order:  Alpha(92) > Bravo(85) > Charlie(72) > Delta(55)
    #   D1 order:   Charlie(95) > Delta(90) > Alpha(70) > Bravo(60)
    #   D2 order:   Bravo(88) > Alpha(80) > Delta(65) > Charlie(50)
    {"address": "0x" + "a1" * 20, "agent_name": "AlphaBot", "latest_ahs": 92,
     "latest_grade": "A", "source": "acp_proactive_scan", "registries": "acp",
     "last_scanned_at": "2026-05-20T10:00:00Z", "latest_d1": 70, "latest_d2": 80},
    {"address": "0x" + "b2" * 20, "agent_name": "BravoAgent", "latest_ahs": 85,
     "latest_grade": "B", "source": "erc8004_scan", "registries": "erc8004",
     "last_scanned_at": "2026-05-19T10:00:00Z", "latest_d1": 60, "latest_d2": 88},
    {"address": "0x" + "c3" * 20, "agent_name": "CharlieBot", "latest_ahs": 72,
     "latest_grade": "C", "source": "celo_agent_wallet", "registries": "celo",
     "last_scanned_at": "2026-05-18T10:00:00Z", "latest_d1": 95, "latest_d2": 50},
    {"address": "0x" + "d4" * 20, "agent_name": "DeltaService", "latest_ahs": 55,
     "latest_grade": "D", "source": "arc_agent_wallet", "registries": "arc",
     "last_scanned_at": "2026-05-17T10:00:00Z", "latest_d1": 90, "latest_d2": 65},

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
            {"address": f"0x{i:040x}", "agent_name": f"Agent {i}",
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
        """Healthiest records must not expose weights/patterns/internals."""
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

from db import is_mash_name, has_banned_keyword, is_handle_spam


class TestIsMashName:
    """Unit tests for the keyboard-mash heuristic."""

    # --- PROBLEM list: every one of these MUST be excluded ---

    def test_mash_6w4r4y567hfddfrh(self):
        assert is_mash_name("6w4r4y567hfddfrh") is True

    def test_mash_6745jrythdfg(self):
        assert is_mash_name("6745jrythdfg") is True

    def test_mash_6745jrythdfg123gfdsg(self):
        assert is_mash_name("6745jrythdfg123gfdsg") is True

    def test_mash_6w4rhtdfgh_er(self):
        assert is_mash_name("6w4rhtdfgh er") is True

    def test_mash_6w4rhtrd_long(self):
        assert is_mash_name("6w4rhtrd dfsh867dfs h345sdf6 7 y") is True

    def test_mash_245ydgsf_fdsg_gfdsg(self):
        assert is_mash_name("245ydgsf fdsg  gfdsg") is True

    def test_mash_245y_gfdsg_hdfg(self):
        assert is_mash_name("245y gfdsg  hdfg hdfg h") is True

    def test_mash_245y_gfdsg(self):
        assert is_mash_name("245y gfdsg") is True

    def test_mash_rew6huj(self):
        assert is_mash_name("rew6huj e56u jhtfg j") is True

    def test_mash_dsaffghjfgg45ty(self):
        assert is_mash_name("dsaffghjfgg45ty 345") is True

    def test_mash_3452gydsfgsdf(self):
        assert is_mash_name("3452gydsfgsdf") is True

    def test_mash_rtg54343125(self):
        assert is_mash_name("rtg54343125") is True

    # --- Original PR #167 MUST-exclude (regression guard) ---

    def test_mash_digit_heavy(self):
        assert is_mash_name("6w4r4y567y563745678") is True

    def test_mash_no_alpha_run(self):
        assert is_mash_name("6w4rfg5eqr3gfdfg") is True

    def test_mash_with_spaces_digit_heavy(self):
        assert is_mash_name("2344rht4232fs6d 426 7 y21") is True

    def test_mash_long_random(self):
        assert is_mash_name("tyju5763ju76758dfg52567") is True

    # --- MUST be kept (legitimate names) ---

    def test_keep_zyfai(self):
        assert is_mash_name(
            "Zyfai Rebalancer Agent for 0x16fA961bE546d7A46d7bD6D3506c57229072452E"
        ) is False

    def test_keep_uniagent(self):
        assert is_mash_name("UniAgent ERC8004 Agent #738") is False

    def test_keep_marcus(self):
        assert is_mash_name("Marcus Aurelius") is False

    def test_keep_agentracheel(self):
        assert is_mash_name("agentracheel") is False

    # --- Short names (≤5 chars) — always kept ---

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

    def test_keep_short_MM4(self):
        assert is_mash_name("MM4") is False

    def test_keep_short_79x(self):
        assert is_mash_name("79x") is False

    # --- Short multi-token abbreviations (≤8 chars with space) ---

    def test_keep_AGI_XBT(self):
        assert is_mash_name("AGI XBT") is False

    def test_keep_DIA_TXT(self):
        assert is_mash_name("DIA TXT") is False

    def test_keep_SX1_AI(self):
        assert is_mash_name("SX1 AI") is False


class TestIsHandleSpam:
    """Unit tests for social-handle spam detection."""

    def test_multi_handle_excluded(self):
        assert is_handle_spam(
            "@sortesfun @calo530G @Peko7g @aom1546  #GOOD #publicgoods #sortes"
        ) is True

    def test_single_hashtag_kept(self):
        assert is_handle_spam("MyAgent #buildinpublic") is False

    def test_single_at_kept(self):
        assert is_handle_spam("Built by @alice") is False

    def test_normal_name_kept(self):
        assert is_handle_spam("AlphaBot") is False


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
        # Handle spam — must be excluded
        {"address": "0x" + "aa" * 20,
         "agent_name": "@sortesfun @calo530G @Peko7g @aom1546  #GOOD #publicgoods #sortes",
         "latest_ahs": 91, "latest_grade": "A", "source": "api", "registries": ""},
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

    def test_excludes_handle_spam(self, tmp_path):
        db = str(tmp_path / "junk.db")
        _seed_db(db, self._JUNK_WALLETS)
        snapshot = generate_snapshot(db)
        names = {r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]}
        assert "@sortesfun @calo530G @Peko7g @aom1546  #GOOD #publicgoods #sortes" not in names

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
        # All wallets with latest_ahs != None:
        # a1(92), b2(85), c3(78), dead(99), null-addr(95), d4(90), e5(88),
        # f6(80), aa-handle-spam(91), 07(60) = 10 scored
        # Named+scored: a1,b2,c3,dead,null-addr,d4,e5,f6,aa = 9
        # Unnamed+scored: 07 = 1
        assert counts["total_scored"] == 10
        assert counts["named_scored"] == 9
        assert counts["unnamed_scored"] == 1

    def test_filter_fills_500_with_clean(self, tmp_path):
        """When there are >500 clean entries + junk, the board still caps at 500
        with clean entries only."""
        db = str(tmp_path / "fill.db")
        wallets = [
            {"address": f"0x{i:040x}", "agent_name": f"Agent {i}",
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


# ---------------------------------------------------------------------------
# D1/D2 dimension boards — schema, write-back, backfill, and boards
# ---------------------------------------------------------------------------

class TestSchemaD1D2:
    """Schema: known_wallets has latest_d1 and latest_d2 after migration."""

    def test_latest_d1_d2_columns_exist(self, tmp_path, monkeypatch):
        import db as db_module
        db_path = str(tmp_path / "schema_test.db")
        monkeypatch.setattr(db_module, "DB_PATH", db_path)
        db_module.init_db()
        conn = sqlite3.connect(db_path)
        try:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(known_wallets)"
            ).fetchall()}
            assert "latest_d1" in cols, "latest_d1 column missing from known_wallets"
            assert "latest_d2" in cols, "latest_d2 column missing from known_wallets"
        finally:
            conn.close()


class TestWriteBackD1D2:
    """Write-back: a logged scan with d1/d2 propagates them to known_wallets."""

    def test_log_scan_propagates_d1_d2(self, tmp_path, monkeypatch):
        import db as db_module
        db_path = str(tmp_path / "writeback_test.db")
        monkeypatch.setattr(db_module, "DB_PATH", db_path)
        db_module.init_db()

        db_module.log_scan(
            address="0x" + "ab" * 20,
            endpoint="ahs",
            scan_timestamp="2026-05-20T10:00:00Z",
            ahs_score=85,
            grade="B",
            d1_score=70,
            d2_score=60,
            agent_name="WriteBackBot",
            source="api",
        )

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT latest_d1, latest_d2 FROM known_wallets WHERE address = ?",
                ("0x" + "ab" * 20,),
            ).fetchone()
            assert row is not None
            assert row["latest_d1"] == 70
            assert row["latest_d2"] == 60
        finally:
            conn.close()


class TestBackfillD1D2:
    """Backfill: pre-existing wallet with scans gets latest_d1/d2 populated
    from its most recent scan during the v12 migration."""

    def test_backfill_populates_from_scans(self, tmp_path, monkeypatch):
        import db as db_module
        db_path = str(tmp_path / "backfill_test.db")
        monkeypatch.setattr(db_module, "DB_PATH", db_path)

        # Create a v11-level DB (no latest_d1/d2 columns)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(db_module._SCHEMA_SQL)
        # Add columns from earlier migrations (v2 + v11)
        for stmt in [
            "ALTER TABLE known_wallets ADD COLUMN registries TEXT DEFAULT ''",
            "ALTER TABLE known_wallets ADD COLUMN agent_name TEXT",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        # Mark as v11
        conn.execute("INSERT INTO schema_version (version) VALUES (11)")

        # Insert a wallet (no latest_d1/d2 columns yet)
        addr = "0x" + "ab" * 20
        conn.execute(
            """INSERT INTO known_wallets
               (address, source, first_seen_at, latest_ahs, latest_grade, agent_name)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (addr, "api", "2026-05-01T00:00:00Z", 80, "B", "BackfillBot"),
        )
        # Insert scans: older and newer, both with d1/d2
        conn.execute(
            """INSERT INTO scans
               (address, endpoint, scan_timestamp, ahs_score, grade,
                d1_score, d2_score, source)
            VALUES (?, 'ahs', '2026-05-10T10:00:00Z', 75, 'B', 60, 55, 'api')""",
            (addr,),
        )
        conn.execute(
            """INSERT INTO scans
               (address, endpoint, scan_timestamp, ahs_score, grade,
                d1_score, d2_score, source)
            VALUES (?, 'ahs', '2026-05-20T10:00:00Z', 80, 'B', 70, 65, 'api')""",
            (addr,),
        )
        conn.commit()
        conn.close()

        # Run init_db → detects v11, applies v12 migration (add cols + backfill)
        db_module.init_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT latest_d1, latest_d2 FROM known_wallets WHERE address = ?",
                (addr,),
            ).fetchone()
            assert row is not None
            assert row["latest_d1"] == 70, f"Expected d1=70 (most recent), got {row['latest_d1']}"
            assert row["latest_d2"] == 65, f"Expected d2=65 (most recent), got {row['latest_d2']}"
        finally:
            conn.close()


class TestLeaderboardD1D2Boards:
    """Leaderboard returns cleanest (D1) and consistent (D2) boards with
    correct ordering, contiguous ranks, dimension fields, junk filters,
    identical counts, and no rising key."""

    def test_cleanest_ordered_by_d1(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        scores = [r["d1"] for r in snapshot["leaderboard"]["cleanest"]]
        assert scores == sorted(scores, reverse=True), (
            f"cleanest not sorted by d1 desc: {scores}"
        )

    def test_consistent_ordered_by_d2(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        scores = [r["d2"] for r in snapshot["leaderboard"]["consistent"]]
        assert scores == sorted(scores, reverse=True), (
            f"consistent not sorted by d2 desc: {scores}"
        )

    def test_cleanest_ranks_contiguous(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        ranks = [r["rank"] for r in snapshot["leaderboard"]["cleanest"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_consistent_ranks_contiguous(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        ranks = [r["rank"] for r in snapshot["leaderboard"]["consistent"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_cleanest_has_d1_field(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        for r in snapshot["leaderboard"]["cleanest"]:
            assert "d1" in r, f"cleanest entry missing 'd1' key: {r}"

    def test_consistent_has_d2_field(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        for r in snapshot["leaderboard"]["consistent"]:
            assert "d2" in r, f"consistent entry missing 'd2' key: {r}"

    def test_junk_filters_apply_to_all_boards(self, tmp_path):
        """Burn addresses, mash names, banned keywords, and handle spam
        are excluded from cleanest and consistent boards."""
        db = str(tmp_path / "junk.db")
        wallets = [
            # Legit
            {"address": "0x" + "a1" * 20, "agent_name": "AlphaBot",
             "latest_ahs": 92, "latest_grade": "A", "source": "acp",
             "registries": "acp", "latest_d1": 70, "latest_d2": 80},
            # Burn address
            {"address": "0x000000000000000000000000000000000000dead",
             "agent_name": "BurnBot", "latest_ahs": 99, "latest_grade": "A",
             "source": "api", "registries": "", "latest_d1": 99, "latest_d2": 99},
            # Mash name
            {"address": "0x" + "d4" * 20, "agent_name": "6w4r4y567y563745678",
             "latest_ahs": 90, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": 95, "latest_d2": 95},
            # Banned keyword
            {"address": "0x" + "e5" * 20, "agent_name": "Test Client",
             "latest_ahs": 88, "latest_grade": "B", "source": "api",
             "registries": "", "latest_d1": 88, "latest_d2": 88},
            # Handle spam
            {"address": "0x" + "aa" * 20,
             "agent_name": "@sortesfun @calo530G @Peko7g",
             "latest_ahs": 91, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": 91, "latest_d2": 91},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)
        for board_name in ("cleanest", "consistent"):
            addrs = {r["address"] for r in snapshot["leaderboard"][board_name]}
            assert "0x000000000000000000000000000000000000dead" not in addrs, (
                f"burn address in {board_name}"
            )
            names = {r["agent_name"] for r in snapshot["leaderboard"][board_name]}
            assert "6w4r4y567y563745678" not in names, (
                f"mash name in {board_name}"
            )
            assert "Test Client" not in names, f"banned keyword in {board_name}"
            assert "@sortesfun @calo530G @Peko7g" not in names, (
                f"handle spam in {board_name}"
            )

    def test_counts_identical_across_boards(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        counts = snapshot["leaderboard"]["counts"]
        # Named + scored: AlphaBot, BravoAgent, CharlieBot, DeltaService = 4
        assert counts["named_scored"] == 4
        # Unnamed + scored: e5 (None), f6 (""), 08 ("   ") = 3
        assert counts["unnamed_scored"] == 3
        assert counts["total_scored"] == 7

    def test_no_rising_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        assert "rising" not in snapshot["leaderboard"]

    def test_null_d1_d2_excluded_from_dimension_boards(self, tmp_path):
        """A wallet with latest_ahs but NULL latest_d1/d2 (older scan
        without dimension scores) appears on healthiest but is excluded
        from cleanest and consistent.  Both dimension boards still have
        contiguous 1..N ranks."""
        db = str(tmp_path / "null_dims.db")
        wallets = [
            # Has AHS but no D1/D2 — older scan without dimension scores
            {"address": "0x" + "01" * 20, "agent_name": "OldSchoolBot",
             "latest_ahs": 90, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": None, "latest_d2": None},
            # Has AHS + D1/D2
            {"address": "0x" + "02" * 20, "agent_name": "ModernBot",
             "latest_ahs": 80, "latest_grade": "B", "source": "api",
             "registries": "", "latest_d1": 75, "latest_d2": 70},
            {"address": "0x" + "03" * 20, "agent_name": "AnotherBot",
             "latest_ahs": 70, "latest_grade": "C", "source": "api",
             "registries": "", "latest_d1": 65, "latest_d2": 85},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)
        lb = snapshot["leaderboard"]

        # OldSchoolBot appears on healthiest
        health_addrs = {r["address"] for r in lb["healthiest"]}
        assert "0x" + "01" * 20 in health_addrs

        # OldSchoolBot excluded from cleanest and consistent
        clean_addrs = {r["address"] for r in lb["cleanest"]}
        consist_addrs = {r["address"] for r in lb["consistent"]}
        assert "0x" + "01" * 20 not in clean_addrs
        assert "0x" + "01" * 20 not in consist_addrs

        # Dimension boards still have contiguous 1..N ranks
        clean_ranks = [r["rank"] for r in lb["cleanest"]]
        assert clean_ranks == list(range(1, len(clean_ranks) + 1))
        consist_ranks = [r["rank"] for r in lb["consistent"]]
        assert consist_ranks == list(range(1, len(consist_ranks) + 1))


class TestSnapshotD1D2:
    """Snapshot includes cleanest and consistent boards, but not rising."""

    def test_snapshot_includes_cleanest_consistent(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        lb = snapshot["leaderboard"]
        assert "cleanest" in lb, "cleanest missing from snapshot leaderboard"
        assert "consistent" in lb, "consistent missing from snapshot leaderboard"
        assert isinstance(lb["cleanest"], list)
        assert isinstance(lb["consistent"], list)

    def test_snapshot_no_rising(self, tmp_path):
        db = str(tmp_path / "test.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)
        assert "rising" not in snapshot["leaderboard"]


# ---------------------------------------------------------------------------
# AHS tiebreak within tied score tiers
# ---------------------------------------------------------------------------

class TestAhsTiebreak:
    """When several agents tie on the ranking score, the secondary sort
    must be latest_ahs DESC so that stronger agents rank higher."""

    def test_cleanest_tiebreak_by_ahs(self, tmp_path):
        """Agents tied on D1 are ordered by AHS descending."""
        db = str(tmp_path / "tie_d1.db")
        wallets = [
            {"address": "0x" + "a1" * 20, "agent_name": "LowAHS",
             "latest_ahs": 40, "latest_grade": "D", "source": "api",
             "registries": "", "latest_d1": 100, "latest_d2": 50,
             "last_scanned_at": "2026-05-20T10:00:00Z"},
            {"address": "0x" + "b2" * 20, "agent_name": "HighAHS",
             "latest_ahs": 95, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": 100, "latest_d2": 80,
             "last_scanned_at": "2026-05-18T10:00:00Z"},
            {"address": "0x" + "c3" * 20, "agent_name": "MidAHS",
             "latest_ahs": 70, "latest_grade": "C", "source": "api",
             "registries": "", "latest_d1": 100, "latest_d2": 60,
             "last_scanned_at": "2026-05-19T10:00:00Z"},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        names = [r["agent_name"] for r in snapshot["leaderboard"]["cleanest"]]
        assert names == ["HighAHS", "MidAHS", "LowAHS"], (
            f"Cleanest tiebreak by AHS wrong: {names}"
        )

    def test_consistent_tiebreak_by_ahs(self, tmp_path):
        """Agents tied on D2 are ordered by AHS descending."""
        db = str(tmp_path / "tie_d2.db")
        wallets = [
            {"address": "0x" + "a1" * 20, "agent_name": "WeakAgent",
             "latest_ahs": 30, "latest_grade": "D", "source": "api",
             "registries": "", "latest_d1": 50, "latest_d2": 85,
             "last_scanned_at": "2026-05-20T10:00:00Z"},
            {"address": "0x" + "b2" * 20, "agent_name": "StrongAgent",
             "latest_ahs": 99, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": 90, "latest_d2": 85,
             "last_scanned_at": "2026-05-18T10:00:00Z"},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        names = [r["agent_name"] for r in snapshot["leaderboard"]["consistent"]]
        assert names == ["StrongAgent", "WeakAgent"], (
            f"Consistent tiebreak by AHS wrong: {names}"
        )

    def test_healthiest_ahs_tiebreak_harmless(self, tmp_path):
        """On healthiest, score_col IS latest_ahs so the secondary AHS key
        only breaks exact ties — same behaviour as before, just explicit."""
        db = str(tmp_path / "tie_ahs.db")
        wallets = [
            {"address": "0x" + "a1" * 20, "agent_name": "TiedOlder",
             "latest_ahs": 80, "latest_grade": "B", "source": "api",
             "registries": "", "latest_d1": 50, "latest_d2": 50,
             "last_scanned_at": "2026-05-01T10:00:00Z"},
            {"address": "0x" + "b2" * 20, "agent_name": "TiedNewer",
             "latest_ahs": 80, "latest_grade": "B", "source": "api",
             "registries": "", "latest_d1": 60, "latest_d2": 60,
             "last_scanned_at": "2026-05-20T10:00:00Z"},
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        names = [r["agent_name"] for r in snapshot["leaderboard"]["healthiest"]]
        # Both tied on AHS=80 so last_scanned_at breaks the tie
        assert names == ["TiedNewer", "TiedOlder"], (
            f"Healthiest AHS tiebreak wrong: {names}"
        )

    def test_all_boards_contiguous_ranks_after_tiebreak(self, tmp_path):
        """All three boards still rank contiguous 1..N after the tiebreak change."""
        db = str(tmp_path / "contiguous.db")
        wallets = [
            {"address": f"0x{i:040x}", "agent_name": f"Agent {i}",
             "latest_ahs": 90 - i, "latest_grade": "A", "source": "api",
             "registries": "", "latest_d1": 100, "latest_d2": 100,
             "last_scanned_at": f"2026-05-{20-i:02d}T10:00:00Z"}
            for i in range(10)
        ]
        _seed_db(db, wallets)
        snapshot = generate_snapshot(db)

        for board_name in ("healthiest", "cleanest", "consistent"):
            board = snapshot["leaderboard"][board_name]
            ranks = [r["rank"] for r in board]
            assert ranks == list(range(1, len(ranks) + 1)), (
                f"{board_name} ranks not contiguous: {ranks}"
            )

    def test_counts_unchanged_after_tiebreak(self, tmp_path):
        """Counts reflect the true scored population regardless of ordering."""
        db = str(tmp_path / "counts.db")
        _seed_db(db, _WALLETS)
        snapshot = generate_snapshot(db)

        counts = snapshot["leaderboard"]["counts"]
        assert counts["named_scored"] == 4
        assert counts["unnamed_scored"] == 3
        assert counts["total_scored"] == 7
