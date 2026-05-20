"""Tests for the leaderboard section of generate_intelligence_snapshot.py.

Covers: healthiest ranking (sorted, contiguous ranks, named-only, 500 cap),
counts split (named vs unnamed), tie-breaking by last_scanned_at, and
backward compatibility (existing snapshot keys preserved).
"""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from scripts.generate_intelligence_snapshot import generate_snapshot


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
