"""Tests for partner key management and usage attribution."""

import hashlib
import os

import pytest


# ---------------------------------------------------------------------------
# DB-level partner key tests
# ---------------------------------------------------------------------------

class TestPartnerFieldsOnApiKeys:
    """Verify partner_id, is_reseller, wholesale_rate columns work."""

    def test_create_key_with_partner_fields(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="partner@example.com",
                tier="pro",
                partner_id="acme-corp",
                is_reseller=True,
                wholesale_rate=0.4,
            )
            assert raw_key.startswith("ahm_live_")

            record = _db.validate_api_key(raw_key)
            assert record is not None
            assert record["partner_id"] == "acme-corp"
            assert record["is_reseller"] == 1
            assert record["wholesale_rate"] == 0.4
        finally:
            _db.DB_PATH = old_path

    def test_default_partner_fields(self, tmp_path):
        """Keys without partner fields should get sensible defaults."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="regular@example.com",
                tier="starter",
            )
            record = _db.validate_api_key(raw_key)
            assert record is not None
            assert record["partner_id"] is None
            assert record["is_reseller"] == 0
            assert record["wholesale_rate"] == 0.5
        finally:
            _db.DB_PATH = old_path


class TestPartnerKeyExpiry:
    """Verify expires_at is persisted and enforced for partner keys."""

    def test_create_key_with_expiry(self, tmp_path):
        """create_api_key(expires_at=...) should store and validate correctly."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_expiry.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="don@nevermined.io",
                tier="enterprise",
                partner_id="nevermined",
                calls_total=None,  # unlimited
                expires_at="2026-07-15T00:00:00Z",
            )
            record = _db.validate_api_key(raw_key)
            assert record is not None
            assert record["expires_at"] == "2026-07-15T00:00:00Z"
            assert record["calls_remaining"] is None  # unlimited
            assert record["partner_id"] == "nevermined"
        finally:
            _db.DB_PATH = old_path

    def test_expired_key_is_rejected(self, tmp_path):
        """A key past its expires_at should fail validation."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_expiry.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="expired@example.com",
                tier="starter",
                expires_at="2020-01-01T00:00:00Z",  # already expired
            )
            record = _db.validate_api_key(raw_key)
            assert record is None, "Expired key should fail validation"
        finally:
            _db.DB_PATH = old_path

    def test_no_expiry_means_no_expiry(self, tmp_path):
        """Keys without expires_at should never expire."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_expiry.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="forever@example.com",
                tier="pro",
            )
            record = _db.validate_api_key(raw_key)
            assert record is not None
            assert record["expires_at"] is None
        finally:
            _db.DB_PATH = old_path


class TestPartnerUsageAttribution:
    """Verify usage logs capture partner_id."""

    def test_log_usage_with_partner_id(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="partner@example.com",
                partner_id="acme-corp",
            )
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            _db.log_api_key_usage(key_hash, "ahs/route", "0x1234", partner_id="acme-corp")
            _db.log_api_key_usage(key_hash, "ahs/route", "0x5678", partner_id="acme-corp")
            _db.log_api_key_usage(key_hash, "risk", "0x9abc")  # no partner

            conn = _db.get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM api_key_usage WHERE partner_id = ?",
                    ("acme-corp",),
                ).fetchall()
                assert len(rows) == 2

                # Non-partner usage should have NULL partner_id
                null_rows = conn.execute(
                    "SELECT * FROM api_key_usage WHERE partner_id IS NULL",
                ).fetchall()
                assert len(null_rows) == 1
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path


class TestGetPartnerUsage:
    """Verify get_partner_usage aggregation."""

    def test_returns_correct_counts(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="partner@example.com",
                partner_id="acme-corp",
                is_reseller=True,
                wholesale_rate=0.4,
            )
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            # Log 5 calls for this partner
            for i in range(5):
                _db.log_api_key_usage(key_hash, "ahs/route", f"0x{i:040x}", partner_id="acme-corp")

            usage = _db.get_partner_usage("acme-corp", days=30)
            assert usage["partner_id"] == "acme-corp"
            assert usage["call_count"] == 5
            assert usage["wholesale_rate"] == 0.4
            # 5 calls * $0.01 retail * 0.4 wholesale = $0.02
            assert usage["wholesale_cost_usd"] == 0.02
            assert "period_start" in usage
            assert "period_end" in usage
        finally:
            _db.DB_PATH = old_path

    def test_empty_partner_returns_zero(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            usage = _db.get_partner_usage("nonexistent", days=30)
            assert usage["call_count"] == 0
            assert usage["wholesale_cost_usd"] == 0.0
        finally:
            _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# API-level partner endpoint tests
# ---------------------------------------------------------------------------

class TestPartnerUsageEndpoint:
    """Verify GET /partners/{partner_id}/usage endpoint."""

    def test_requires_internal_key(self, client):
        resp = client.get("/partners/acme-corp/usage")
        assert resp.status_code == 401

    def test_rejects_wrong_key(self, client):
        resp = client.get(
            "/partners/acme-corp/usage",
            headers={"X-Internal-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_returns_usage_with_valid_key(self, client):
        from api import INTERNAL_API_KEY
        resp = client.get(
            "/partners/acme-corp/usage",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["partner_id"] == "acme-corp"
        assert "call_count" in data
        assert "wholesale_cost_usd" in data
        assert "period_start" in data
        assert "period_end" in data

    def test_custom_days_parameter(self, client):
        from api import INTERNAL_API_KEY
        resp = client.get(
            "/partners/acme-corp/usage?days=7",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["period_days"] == 7


class TestPartnerAttributionViaHeader:
    """Verify X-Partner-Id header flows through consume_api_key to usage logs."""

    def test_consume_api_key_captures_partner_header(self, tmp_path):
        """consume_api_key should pass X-Partner-Id to log_api_key_usage."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="user@example.com",
                tier="pro",
                calls_total=100,
            )
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            record = _db.validate_api_key(raw_key)

            # Simulate a request with X-Partner-Id header
            from unittest.mock import MagicMock
            mock_request = MagicMock()
            mock_request.headers = {"X-Partner-Id": "reseller-xyz"}

            from api import consume_api_key
            consume_api_key(record, "ahs/route", "0x1234", request=mock_request)

            # Verify the usage was logged with partner_id
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT partner_id FROM api_key_usage WHERE key_hash = ?",
                    (key_hash,),
                ).fetchone()
                assert row is not None
                assert row["partner_id"] == "reseller-xyz"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_consume_api_key_falls_back_to_key_partner_id(self, tmp_path):
        """When no X-Partner-Id header, use the key's own partner_id."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_partner.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="partner@example.com",
                partner_id="built-in-partner",
                calls_total=100,
            )
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            record = _db.validate_api_key(raw_key)

            # No X-Partner-Id header
            from unittest.mock import MagicMock
            mock_request = MagicMock()
            mock_request.headers = {}

            from api import consume_api_key
            consume_api_key(record, "ahs/route", "0x1234", request=mock_request)

            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT partner_id FROM api_key_usage WHERE key_hash = ?",
                    (key_hash,),
                ).fetchone()
                assert row is not None
                assert row["partner_id"] == "built-in-partner"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path


class TestSchemaV6Migration:
    """Verify v6 migration adds partner columns to existing databases."""

    def test_migration_adds_columns(self, tmp_path):
        """Create a v5 database, then run init_db to apply v6 migration."""
        import db as _db
        import sqlite3

        old_path = _db.DB_PATH
        db_file = str(tmp_path / "test_migrate.db")
        _db.DB_PATH = db_file
        try:
            # Create v5 schema manually (no partner columns)
            conn = sqlite3.connect(db_file)
            conn.execute("""CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL, customer_email TEXT NOT NULL,
                stripe_customer_id TEXT, stripe_subscription_id TEXT,
                type TEXT NOT NULL, tier TEXT NOT NULL,
                calls_remaining INTEGER, calls_total INTEGER,
                created_at TEXT NOT NULL, expires_at TEXT,
                is_active INTEGER DEFAULT 1
            )""")
            conn.execute("""CREATE TABLE api_key_usage (
                id INTEGER PRIMARY KEY, key_hash TEXT NOT NULL,
                endpoint TEXT NOT NULL, called_at TEXT NOT NULL,
                wallet_queried TEXT
            )""")
            conn.execute("""CREATE TABLE schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )""")
            conn.execute("INSERT INTO schema_version (version) VALUES (5)")
            conn.commit()
            conn.close()

            # Run init_db — should apply v6 migration
            _db.init_db()

            # Verify new columns exist
            conn = _db.get_connection()
            try:
                # Check api_keys has partner columns
                info = conn.execute("PRAGMA table_info(api_keys)").fetchall()
                col_names = [row["name"] for row in info]
                assert "partner_id" in col_names
                assert "is_reseller" in col_names
                assert "wholesale_rate" in col_names

                # Check api_key_usage has partner_id
                info = conn.execute("PRAGMA table_info(api_key_usage)").fetchall()
                col_names = [row["name"] for row in info]
                assert "partner_id" in col_names
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
