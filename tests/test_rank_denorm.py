"""Tests for rank/percentile denormalisation onto known_wallets (v14 migration).

Covers: schema migration creates columns, refresh_ranks_and_percentiles()
populates rank for named scored agents and percentile_rank for all scored
agents, rank=NULL for unnamed/unranked agents, idempotency.
"""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_db(db_path: str, wallets: list[dict], scans: list[dict] | None = None):
    """Create a minimal DB with known_wallets and scans tables, then seed rows."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

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
            latest_d2             INTEGER,
            rank                  INTEGER,
            percentile_rank       REAL
        );
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            address         TEXT,
            endpoint        TEXT,
            scan_timestamp  TEXT,
            ahs_score       INTEGER,
            grade           TEXT,
            d1_score        INTEGER,
            d2_score        INTEGER
        );
        CREATE TABLE IF NOT EXISTS scan_patterns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id         INTEGER,
            pattern_name    TEXT
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

    if scans:
        for s in scans:
            conn.execute(
                """INSERT INTO scans (address, endpoint, scan_timestamp,
                   ahs_score, grade, d1_score, d2_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    s["address"],
                    s.get("endpoint", "ahs"),
                    s.get("scan_timestamp", now),
                    s.get("ahs_score"),
                    s.get("grade"),
                    s.get("d1_score"),
                    s.get("d2_score"),
                ),
            )

    conn.commit()
    conn.close()


def _get_wallet(db_path: str, address: str) -> dict:
    """Fetch a single known_wallets row as a dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM known_wallets WHERE address = ?", (address,)
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    """v14 migration should add rank (INTEGER) and percentile_rank (REAL)."""

    def test_columns_exist_after_migration(self, tmp_path):
        """init_db creates rank and percentile_rank columns."""
        db_file = str(tmp_path / "test.db")
        with patch.object(db, "DB_PATH", db_file):
            db.init_db()
        conn = sqlite3.connect(db_file)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(known_wallets)")}
        conn.close()
        assert "rank" in cols
        assert "percentile_rank" in cols

    def test_percentile_rank_is_real_type(self, tmp_path):
        """percentile_rank column should be REAL, not INTEGER."""
        db_file = str(tmp_path / "test.db")
        with patch.object(db, "DB_PATH", db_file):
            db.init_db()
        conn = sqlite3.connect(db_file)
        col_info = conn.execute("PRAGMA table_info(known_wallets)").fetchall()
        conn.close()
        pct_col = [c for c in col_info if c[1] == "percentile_rank"]
        assert pct_col, "percentile_rank column not found"
        assert pct_col[0][2] == "REAL"


# ---------------------------------------------------------------------------
# refresh_ranks_and_percentiles
# ---------------------------------------------------------------------------

class TestRefreshRanksAndPercentiles:
    """refresh_ranks_and_percentiles populates rank/percentile correctly."""

    def _setup_db(self, tmp_path, wallets, scans=None):
        """Seed a test DB and return its path."""
        db_file = str(tmp_path / "test.db")
        _seed_db(db_file, wallets, scans)
        return db_file

    def test_named_agents_get_rank(self, tmp_path):
        """Named agents with scores get rank populated."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
            {"address": "0xB", "agent_name": "Beta", "latest_ahs": 85, "latest_grade": "B"},
            {"address": "0xC", "agent_name": "Gamma", "latest_ahs": 75, "latest_grade": "C"},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
            {"address": "0xB", "ahs_score": 85, "grade": "B"},
            {"address": "0xC", "ahs_score": 75, "grade": "C"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        assert _get_wallet(db_file, "0xA")["rank"] == 1
        assert _get_wallet(db_file, "0xB")["rank"] == 2
        assert _get_wallet(db_file, "0xC")["rank"] == 3

    def test_unnamed_agents_no_rank(self, tmp_path):
        """Unnamed agents should have rank=NULL."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
            {"address": "0xU", "agent_name": None, "latest_ahs": 90, "latest_grade": "A"},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
            {"address": "0xU", "ahs_score": 90, "grade": "A"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        assert _get_wallet(db_file, "0xA")["rank"] == 1
        assert _get_wallet(db_file, "0xU")["rank"] is None

    def test_unnamed_agents_get_percentile(self, tmp_path):
        """Unnamed agents with scores still get percentile_rank populated."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
            {"address": "0xU", "agent_name": None, "latest_ahs": 60, "latest_grade": "C"},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
            {"address": "0xU", "ahs_score": 60, "grade": "C"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        pct_a = _get_wallet(db_file, "0xA")["percentile_rank"]
        pct_u = _get_wallet(db_file, "0xU")["percentile_rank"]
        assert pct_a is not None
        assert pct_u is not None
        assert pct_a > pct_u  # higher score = higher percentile

    def test_unscored_agents_no_percentile(self, tmp_path):
        """Agents without latest_ahs should have percentile_rank=NULL."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
            {"address": "0xN", "agent_name": "NoScore", "latest_ahs": None, "latest_grade": None},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        assert _get_wallet(db_file, "0xN")["percentile_rank"] is None

    def test_percentile_rank_is_fractional(self, tmp_path):
        """percentile_rank should preserve fractional values, not truncate to int."""
        # Create enough agents so interpolation produces a fractional result
        wallets = []
        scans = []
        for i in range(20):
            addr = f"0x{i:04x}"
            score = 50 + i * 2  # 50, 52, 54, ..., 88
            wallets.append({
                "address": addr, "agent_name": f"Agent{i}",
                "latest_ahs": score, "latest_grade": "B",
            })
            scans.append({"address": addr, "ahs_score": score, "grade": "B"})
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        # Check that at least one percentile_rank has a fractional part
        all_conn = sqlite3.connect(db_file)
        all_conn.row_factory = sqlite3.Row
        rows = all_conn.execute(
            "SELECT percentile_rank FROM known_wallets WHERE percentile_rank IS NOT NULL"
        ).fetchall()
        all_conn.close()

        has_fractional = any(r["percentile_rank"] != int(r["percentile_rank"]) for r in rows)
        assert has_fractional, "Expected at least one fractional percentile_rank"

    def test_idempotent(self, tmp_path):
        """Calling refresh twice produces the same results."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
            {"address": "0xB", "agent_name": "Beta", "latest_ahs": 85, "latest_grade": "B"},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
            {"address": "0xB", "ahs_score": 85, "grade": "B"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        with patch.object(db, "DB_PATH", db_file):
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            db.refresh_ranks_and_percentiles(conn)
            conn.close()
            first_a = _get_wallet(db_file, "0xA")
            first_b = _get_wallet(db_file, "0xB")

            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            db.refresh_ranks_and_percentiles(conn)
            conn.close()
            second_a = _get_wallet(db_file, "0xA")
            second_b = _get_wallet(db_file, "0xB")

        assert first_a["rank"] == second_a["rank"]
        assert first_b["rank"] == second_b["rank"]
        assert first_a["percentile_rank"] == second_a["percentile_rank"]
        assert first_b["percentile_rank"] == second_b["percentile_rank"]

    def test_top_500_cap(self, tmp_path):
        """Only the top 500 named agents should get a rank; #501 should be NULL."""
        wallets = []
        scans = []
        for i in range(510):
            addr = f"0x{i:04x}"
            score = 100 - (i * 100 // 510)  # scores from ~100 down
            # Names must pass the mash-name filter: include a word token
            # (4+ alpha chars with a vowel).
            wallets.append({
                "address": addr, "agent_name": f"Agent {i}",
                "latest_ahs": max(score, 1), "latest_grade": "B",
            })
            scans.append({
                "address": addr, "ahs_score": max(score, 1), "grade": "B",
            })
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            db.refresh_ranks_and_percentiles(conn)
        conn.close()

        # Count how many have rank set
        check_conn = sqlite3.connect(db_file)
        ranked = check_conn.execute(
            "SELECT COUNT(*) FROM known_wallets WHERE rank IS NOT NULL"
        ).fetchone()[0]
        check_conn.close()
        assert ranked == 500

    def test_return_value(self, tmp_path):
        """refresh_ranks_and_percentiles returns summary dict."""
        wallets = [
            {"address": "0xA", "agent_name": "Alpha", "latest_ahs": 95, "latest_grade": "A"},
        ]
        scans = [
            {"address": "0xA", "ahs_score": 95, "grade": "A"},
        ]
        db_file = self._setup_db(tmp_path, wallets, scans)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        with patch.object(db, "DB_PATH", db_file):
            result = db.refresh_ranks_and_percentiles(conn)
        conn.close()

        assert "ranked" in result
        assert "percentiled" in result
        assert result["ranked"] >= 1
        assert result["percentiled"] >= 1


# ---------------------------------------------------------------------------
# _report_card_percentile (now in db.py)
# ---------------------------------------------------------------------------

class TestReportCardPercentile:
    """Verify _report_card_percentile returns fractional results."""

    def test_empty_percentiles_returns_50(self):
        assert db._report_card_percentile(80, {}) == 50.0

    def test_returns_float(self):
        percentiles = {"p10": 40, "p25": 55, "p50": 65, "p75": 78, "p90": 90}
        result = db._report_card_percentile(60, percentiles)
        assert isinstance(result, float)

    def test_mid_range_interpolation(self):
        percentiles = {"p10": 40, "p25": 55, "p50": 65, "p75": 78, "p90": 90}
        result = db._report_card_percentile(60, percentiles)
        assert 25 < result < 50  # between p25 and p50

    def test_high_score_above_p90(self):
        percentiles = {"p10": 40, "p25": 55, "p50": 65, "p75": 78, "p90": 90}
        result = db._report_card_percentile(95, percentiles)
        assert result > 90
        assert result <= 99

    def test_low_score_below_p10(self):
        percentiles = {"p10": 40, "p25": 55, "p50": 65, "p75": 78, "p90": 90}
        result = db._report_card_percentile(20, percentiles)
        assert result >= 1
        assert result < 10
