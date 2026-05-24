"""End-to-end tests for Stripe API-key delivery.

Asserts the full path: webhook fires → pending key stored → customer
retrieves key exactly once → second retrieval returns 'consumed' →
expired keys are not returned.
"""

import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# DB-level pending_key_delivery tests
# ---------------------------------------------------------------------------

class TestPendingKeyDeliveryDB:
    """Verify store / retrieve / cleanup at the DB layer."""

    def test_store_and_retrieve_returns_key(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
            _db.store_pending_key(session_id, "ahm_live_testkey123")

            result = _db.retrieve_pending_key(session_id)
            assert result["status"] == "ok"
            assert result["key"] == "ahm_live_testkey123"
        finally:
            _db.DB_PATH = old_path

    def test_second_retrieval_returns_consumed(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
            _db.store_pending_key(session_id, "ahm_live_onceonly")

            first = _db.retrieve_pending_key(session_id)
            assert first["status"] == "ok"

            second = _db.retrieve_pending_key(session_id)
            assert second["status"] == "consumed"
            assert "key" not in second
        finally:
            _db.DB_PATH = old_path

    def test_unknown_session_returns_not_found(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            result = _db.retrieve_pending_key("cs_nonexistent")
            assert result["status"] == "not_found"
        finally:
            _db.DB_PATH = old_path

    def test_expired_key_returns_expired(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
            _db.store_pending_key(session_id, "ahm_live_expired")

            # Backdate the created_at to 25 hours ago
            conn = _db.get_connection()
            try:
                old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                conn.execute(
                    "UPDATE pending_key_delivery SET created_at = ? "
                    "WHERE stripe_session_id = ?",
                    (old_ts, session_id),
                )
                conn.commit()
            finally:
                conn.close()

            result = _db.retrieve_pending_key(session_id)
            assert result["status"] == "expired"
            assert "key" not in result
        finally:
            _db.DB_PATH = old_path

    def test_duplicate_store_is_silent(self, tmp_path):
        """INSERT OR IGNORE: duplicate webhook delivery is a no-op."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
            _db.store_pending_key(session_id, "ahm_live_first")
            _db.store_pending_key(session_id, "ahm_live_second")  # no error

            result = _db.retrieve_pending_key(session_id)
            assert result["status"] == "ok"
            assert result["key"] == "ahm_live_first"  # first write wins
        finally:
            _db.DB_PATH = old_path

    def test_cleanup_removes_expired_rows(self, tmp_path):
        """Cleanup deletes rows older than 24h. Fresh consumed rows survive
        so the retrieval endpoint can still report 'consumed'."""
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_delivery.db")
        try:
            _db.init_db()
            conn = _db.get_connection()
            try:
                old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                # Expired row (>24h old) — should be deleted
                conn.execute(
                    "INSERT INTO pending_key_delivery "
                    "(stripe_session_id, plaintext_key, created_at) VALUES (?, ?, ?)",
                    ("cs_old", "key_old", old_ts),
                )
                # Consumed but fresh row — should survive
                conn.execute(
                    "INSERT INTO pending_key_delivery "
                    "(stripe_session_id, plaintext_key, consumed) VALUES (?, ?, 1)",
                    ("cs_done", "key_done"),
                )
                # Fresh, unconsumed row — should survive
                conn.execute(
                    "INSERT INTO pending_key_delivery "
                    "(stripe_session_id, plaintext_key) VALUES (?, ?)",
                    ("cs_fresh", "key_fresh"),
                )
                conn.commit()

                deleted = _db._cleanup_pending_keys(conn)
                assert deleted == 1

                remaining = conn.execute(
                    "SELECT COUNT(*) FROM pending_key_delivery"
                ).fetchone()[0]
                assert remaining == 2
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# API-level end-to-end delivery tests
# ---------------------------------------------------------------------------

class TestKeyDeliveryEndToEnd:
    """Full webhook → store → retrieve → consumed flow via HTTP."""

    def _fire_checkout_webhook(self, client, session_id, email="buyer@example.com"):
        """Simulate a checkout.session.completed webhook event."""
        mock_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "id": session_id,
                "customer": "cus_test_456",
                "customer_email": email,
                "metadata": {"tier": "starter", "type": "payg", "calls": "100"},
            }},
        }
        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch("stripe.Webhook.construct_event", return_value=mock_event):
            return client.post(
                "/stripe/webhook",
                content=b"{}",
                headers={"stripe-signature": "test"},
            )

    def test_webhook_creates_pending_key(self, client):
        """checkout.session.completed stores a retrievable pending key."""
        session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
        resp = self._fire_checkout_webhook(client, session_id)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Key is retrievable via the endpoint
        resp = client.get(f"/stripe/key/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["key"].startswith("ahm_live_")

    def test_key_returned_exactly_once(self, client):
        """Second retrieval returns consumed, not the key."""
        session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
        self._fire_checkout_webhook(client, session_id)

        first = client.get(f"/stripe/key/{session_id}")
        assert first.status_code == 200
        assert first.json()["status"] == "ok"
        assert "key" in first.json()

        second = client.get(f"/stripe/key/{session_id}")
        assert second.status_code == 200
        assert second.json()["status"] == "consumed"
        assert "key" not in second.json()

    def test_unknown_session_returns_404(self, client):
        resp = client.get("/stripe/key/cs_nonexistent_999")
        assert resp.status_code == 404

    def test_expired_key_returns_410(self, client):
        """Keys older than 24h return HTTP 410 Gone."""
        import db as _db
        session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
        self._fire_checkout_webhook(client, session_id)

        # Backdate the row
        conn = _db.get_connection()
        try:
            old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            conn.execute(
                "UPDATE pending_key_delivery SET created_at = ? "
                "WHERE stripe_session_id = ?",
                (old_ts, session_id),
            )
            conn.commit()
        finally:
            conn.close()

        resp = client.get(f"/stripe/key/{session_id}")
        assert resp.status_code == 410

    def test_retrieved_key_is_valid(self, client):
        """The key returned by the delivery endpoint actually validates."""
        import db as _db
        session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
        self._fire_checkout_webhook(client, session_id)

        resp = client.get(f"/stripe/key/{session_id}")
        raw_key = resp.json()["key"]

        record = _db.validate_api_key(raw_key)
        assert record is not None
        assert record["customer_email"] == "buyer@example.com"
        assert record["tier"] == "starter"

    def test_duplicate_webhook_is_idempotent(self, client):
        """Re-delivered webhook doesn't error or overwrite the pending key."""
        session_id = f"cs_test_{uuid.uuid4().hex[:12]}"
        resp1 = self._fire_checkout_webhook(client, session_id)
        assert resp1.status_code == 200

        # Second delivery (Stripe retry) — should not error
        resp2 = self._fire_checkout_webhook(client, session_id)
        assert resp2.status_code == 200

        # Key is still retrievable
        resp = client.get(f"/stripe/key/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_success_page_serves_html(self, client):
        """GET /checkout/success returns the success page HTML."""
        resp = client.get("/checkout/success")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Your API Key" in resp.text


class TestSchemaV13Migration:
    """Verify v13 migration creates pending_key_delivery table."""

    def test_table_exists_after_init(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_v13.db")
        try:
            _db.init_db()
            conn = _db.get_connection()
            try:
                info = conn.execute(
                    "PRAGMA table_info(pending_key_delivery)"
                ).fetchall()
                col_names = [row["name"] for row in info]
                assert "stripe_session_id" in col_names
                assert "plaintext_key" in col_names
                assert "created_at" in col_names
                assert "consumed" in col_names
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
