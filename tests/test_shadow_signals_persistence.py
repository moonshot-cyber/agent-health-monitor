"""Tests for D2 shadow-signal persistence (schema v8).

Verifies that:
- The v8 migration adds a shadow_signals_json column to the scans table.
- log_scan() persists a shadow_signals dict as JSON in that column.
- log_scan() leaves the column NULL when no shadow_signals are passed
  (back-compat for existing callers and non-AHS endpoints).
- The persisted JSON round-trips back to the original dict.
"""

import json


class TestSchemaV8Migration:
    """Verify v8 migration adds shadow_signals_json to scans table."""

    def test_column_exists_after_init(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_v8.db")
        try:
            _db.init_db()
            conn = _db.get_connection()
            try:
                info = conn.execute("PRAGMA table_info(scans)").fetchall()
                col_names = [row["name"] for row in info]
                assert "shadow_signals_json" in col_names, (
                    f"shadow_signals_json column missing after v8 migration. "
                    f"Got columns: {col_names}"
                )
                # Confirm it's TEXT and nullable
                col = next(r for r in info if r["name"] == "shadow_signals_json")
                assert col["type"] == "TEXT"
                assert col["notnull"] == 0
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_schema_version_bumped_to_8(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_v8_version.db")
        try:
            _db.init_db()
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT MAX(version) FROM schema_version"
                ).fetchone()
                assert row[0] >= 8, f"Expected schema_version >= 8, got {row[0]}"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path


class TestShadowSignalsPersistence:
    """Verify log_scan persists shadow signals through to the DB."""

    SAMPLE_SHADOW = {
        "session_continuity_score": 75,
        "abrupt_sessions": 1,
        "budget_exhaustion_count": 0,
        "total_sessions": 4,
        "avg_session_length": 5.5,
        "shadow_patterns": [
            {
                "name": "Budget Exhaustion",
                "detected": False,
                "severity": "warning",
                "description": "test pattern",
                "shadow": True,
            }
        ],
    }

    def test_shadow_signals_round_trip(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shadow.db")
        try:
            _db.init_db()
            _db.log_scan(
                address="0x" + "a" * 40,
                endpoint="ahs",
                scan_timestamp="2026-04-07T12:00:00Z",
                ahs_score=72,
                grade="C",
                d1_score=70,
                d2_score=74,
                shadow_signals=self.SAMPLE_SHADOW,
            )
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT shadow_signals_json FROM scans WHERE address = ?",
                    ("0x" + "a" * 40,),
                ).fetchone()
                assert row is not None, "scan was not inserted"
                assert row["shadow_signals_json"] is not None, (
                    "shadow_signals_json was not persisted"
                )
                decoded = json.loads(row["shadow_signals_json"])
                assert decoded == self.SAMPLE_SHADOW, (
                    f"Round-trip mismatch: {decoded}"
                )
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_shadow_signals_null_when_not_passed(self, tmp_path):
        """Non-AHS endpoints (and back-compat callers) leave the column NULL."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shadow_null.db")
        try:
            _db.init_db()
            _db.log_scan(
                address="0x" + "b" * 40,
                endpoint="health",
                scan_timestamp="2026-04-07T12:00:00Z",
                health_score=88.0,
            )
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT shadow_signals_json FROM scans WHERE address = ?",
                    ("0x" + "b" * 40,),
                ).fetchone()
                assert row is not None
                assert row["shadow_signals_json"] is None
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_empty_dict_stored_as_null(self, tmp_path):
        """An empty dict means 'nothing to store' — should not waste a row."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shadow_empty.db")
        try:
            _db.init_db()
            _db.log_scan(
                address="0x" + "c" * 40,
                endpoint="ahs",
                scan_timestamp="2026-04-07T12:00:00Z",
                ahs_score=50,
                grade="D",
                shadow_signals={},
            )
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT shadow_signals_json FROM scans WHERE address = ?",
                    ("0x" + "c" * 40,),
                ).fetchone()
                assert row["shadow_signals_json"] is None
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_none_session_continuity_score_persists(self, tmp_path):
        """Wallets below the threshold have score=None — but the dict should
        still persist so we can measure coverage (how many wallets are eligible)."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shadow_none_score.db")
        try:
            _db.init_db()
            shadow_with_none = {
                "session_continuity_score": None,
                "abrupt_sessions": 0,
                "budget_exhaustion_count": 0,
                "total_sessions": 0,
                "avg_session_length": 0.0,
                "shadow_patterns": [],
            }
            _db.log_scan(
                address="0x" + "d" * 40,
                endpoint="ahs",
                scan_timestamp="2026-04-07T12:00:00Z",
                ahs_score=55,
                grade="D",
                shadow_signals=shadow_with_none,
            )
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT shadow_signals_json FROM scans WHERE address = ?",
                    ("0x" + "d" * 40,),
                ).fetchone()
                assert row["shadow_signals_json"] is not None
                decoded = json.loads(row["shadow_signals_json"])
                assert decoded["session_continuity_score"] is None
                assert decoded["total_sessions"] == 0
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
