"""Tests for the agent_name column on known_wallets (schema v11).

Covers: migration idempotency, backfill correctness for ACP / ERC-8004
label formats, NULL-label safety, and log_scan() persistence.
"""

import sqlite3
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_fresh_db(tmp_path, pre_rows=None):
    """Create a fresh DB, optionally seeding known_wallets rows BEFORE the
    v11 migration runs (to test backfill).

    pre_rows: list of (address, label, source) tuples inserted into a v10
    schema before init_db() upgrades to v11.
    """
    import db as _db

    old_path = _db.DB_PATH
    _db.DB_PATH = str(tmp_path / "test_agent_name.db")
    try:
        if pre_rows:
            # Bootstrap a v10-era database with rows but no agent_name column.
            conn = sqlite3.connect(_db.DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Create just the known_wallets table without agent_name.
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
                    rescan_interval_hours INTEGER NOT NULL DEFAULT 168
                );
                CREATE TABLE IF NOT EXISTS schema_version (
                    version    INTEGER NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                );
            """)
            # Add the registries column (v2) so init_db v2 migration doesn't fail
            try:
                conn.execute("ALTER TABLE known_wallets ADD COLUMN registries TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            # Mark schema as v10 so init_db() only runs >=11 migrations
            conn.execute("INSERT INTO schema_version (version) VALUES (10)")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for addr, label, source in pre_rows:
                conn.execute(
                    "INSERT INTO known_wallets (address, label, source, first_seen_at) "
                    "VALUES (?, ?, ?, ?)",
                    (addr, label, source, now),
                )
            conn.commit()
            conn.close()

        # Now run init_db — this triggers v11 migration (ALTER + backfill)
        _db.init_db()
        return _db
    except Exception:
        _db.DB_PATH = old_path
        raise


def _cleanup_db(_db, old_path):
    _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestSchemaV11Migration:
    """Verify v11 migration adds agent_name column and backfills correctly."""

    def test_column_exists_after_init(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path)
            conn = _db.get_connection()
            try:
                info = conn.execute("PRAGMA table_info(known_wallets)").fetchall()
                col_names = [row["name"] for row in info]
                assert "agent_name" in col_names, (
                    f"agent_name column missing after v11 migration. Got: {col_names}"
                )
                col = next(r for r in info if r["name"] == "agent_name")
                assert col["type"] == "TEXT"
                assert col["notnull"] == 0  # nullable
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_schema_version_bumped_to_11(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path)
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT MAX(version) FROM schema_version"
                ).fetchone()
                assert row[0] >= 11, f"Expected schema_version >= 11, got {row[0]}"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_migration_idempotent(self, tmp_path):
        """Calling init_db() twice must not error."""
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path)
            # Second call — should be a no-op, not raise
            _db.init_db()
        finally:
            _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# Backfill tests
# ---------------------------------------------------------------------------

class TestAgentNameBackfill:
    """Verify the v11 backfill extracts names from the composite label."""

    def test_backfill_acp_label(self, tmp_path):
        """ACP label 'ACP #42 \u2014 MyBot' \u2192 agent_name == 'MyBot' (exact match)."""
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path, pre_rows=[
                ("0x" + "aa" * 20, "ACP #42 \u2014 MyBot", "acp_proactive_scan"),
            ])
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "aa" * 20,),
                ).fetchone()
                assert row["agent_name"] == "MyBot"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_backfill_erc8004_label(self, tmp_path):
        """ERC-8004 label 'ERC-8004 #99 \u2014 FooAgent' \u2192 agent_name == 'FooAgent' (exact match)."""
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path, pre_rows=[
                ("0x" + "bb" * 20, "ERC-8004 #99 \u2014 FooAgent", "erc8004_scan"),
            ])
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "bb" * 20,),
                ).fetchone()
                assert row["agent_name"] == "FooAgent"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_backfill_no_emdash_stays_null(self, tmp_path):
        """Olas-style label without em-dash \u2192 agent_name stays NULL."""
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path, pre_rows=[
                ("0x" + "cc" * 20, "Olas Service #7", "olas_service"),
            ])
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "cc" * 20,),
                ).fetchone()
                assert row["agent_name"] is None
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_backfill_null_label_stays_null(self, tmp_path):
        """Row with label=NULL \u2192 agent_name stays NULL (no crash)."""
        import db as _db
        old_path = _db.DB_PATH
        try:
            _init_fresh_db(tmp_path, pre_rows=[
                ("0x" + "dd" * 20, None, "api"),
            ])
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "dd" * 20,),
                ).fetchone()
                assert row["agent_name"] is None
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# log_scan() persistence tests
# ---------------------------------------------------------------------------

class TestLogScanAgentName:
    """Verify log_scan() writes and preserves agent_name in known_wallets."""

    def test_log_scan_persists_agent_name(self, tmp_path):
        """log_scan(agent_name='TestBot') \u2192 known_wallets.agent_name = 'TestBot'."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_logscan.db")
        try:
            _db.init_db()
            addr = "0x" + "ee" * 20
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            _db.log_scan(
                address=addr,
                endpoint="ahs",
                scan_timestamp=now,
                source="acp_proactive_scan",
                label="ACP #1 \u2014 TestBot",
                ahs_score=75,
                grade="B",
                agent_name="TestBot",
            )
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    (addr,),
                ).fetchone()
                assert row["agent_name"] == "TestBot"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_log_scan_null_agent_name_preserves_existing(self, tmp_path):
        """Existing agent_name is NOT overwritten when log_scan passes None."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_preserve.db")
        try:
            _db.init_db()
            addr = "0x" + "ff" * 20
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # First scan — sets agent_name
            _db.log_scan(
                address=addr,
                endpoint="ahs",
                scan_timestamp=now,
                source="acp_proactive_scan",
                label="ACP #2 \u2014 OriginalName",
                ahs_score=80,
                grade="B",
                agent_name="OriginalName",
            )

            # Second scan — agent_name=None (e.g. from a different scanner)
            _db.log_scan(
                address=addr,
                endpoint="ahs",
                scan_timestamp=now,
                source="acp_proactive_scan",
                label="ACP #2 \u2014 OriginalName",
                ahs_score=82,
                grade="B",
                agent_name=None,
            )

            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    (addr,),
                ).fetchone()
                assert row["agent_name"] == "OriginalName"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
