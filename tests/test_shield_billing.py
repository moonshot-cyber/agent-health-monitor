"""Tests for Shield subscription billing (Stripe integration)."""

import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# DB-level shield subscription tests
# ---------------------------------------------------------------------------

class TestShieldSubscriptionCRUD:
    """Verify shield_subscriptions table CRUD operations."""

    def test_create_and_get_subscription(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(
                customer_email="dev@example.com",
                tier="pro",
            )
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            sub_id = _db.create_shield_subscription(
                api_key_hash=key_hash,
                tier="starter",
                stripe_customer_id="cus_test123",
                stripe_subscription_id="sub_test123",
            )
            assert sub_id > 0

            sub = _db.get_shield_subscription(key_hash)
            assert sub is not None
            assert sub["tier"] == "starter"
            assert sub["status"] == "active"
            assert sub["agent_slots"] == 5
            assert sub["call_quota"] == 10_000
            assert sub["calls_used_this_period"] == 0
        finally:
            _db.DB_PATH = old_path

    def test_get_subscription_returns_none_for_unknown(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            sub = _db.get_shield_subscription("nonexistent_hash")
            assert sub is None
        finally:
            _db.DB_PATH = old_path

    def test_tier_definitions(self):
        import db as _db
        assert _db.SHIELD_TIERS["free"] == (1, 100)
        assert _db.SHIELD_TIERS["starter"] == (5, 10_000)
        assert _db.SHIELD_TIERS["pro"] == (50, 100_000)
        assert _db.SHIELD_TIERS["enterprise"] == (999, 999_999_999)

    def test_enterprise_tier_values(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(customer_email="ent@example.com")
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            _db.create_shield_subscription(api_key_hash=key_hash, tier="enterprise")
            sub = _db.get_shield_subscription(key_hash)
            assert sub["agent_slots"] == 999
            assert sub["call_quota"] == 999_999_999
        finally:
            _db.DB_PATH = old_path


class TestShieldUsageTracking:
    """Verify increment_shield_usage and quota enforcement."""

    def test_increment_usage(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(customer_email="user@example.com")
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            _db.create_shield_subscription(api_key_hash=key_hash, tier="starter")

            # Increment 3 times
            for i in range(3):
                sub = _db.increment_shield_usage(key_hash)
                assert sub is not None
                assert sub["calls_used_this_period"] == i + 1
        finally:
            _db.DB_PATH = old_path

    def test_increment_returns_none_no_subscription(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            result = _db.increment_shield_usage("no_such_hash")
            assert result is None
        finally:
            _db.DB_PATH = old_path


class TestShieldSubscriptionStatus:
    """Verify subscription status updates."""

    def test_update_status(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(customer_email="user@example.com")
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            _db.create_shield_subscription(
                api_key_hash=key_hash, tier="pro",
                stripe_subscription_id="sub_cancel_test",
            )

            updated = _db.update_shield_subscription_status("sub_cancel_test", "cancelled")
            assert updated is True

            # Should no longer appear as active
            sub = _db.get_shield_subscription(key_hash)
            assert sub is None
        finally:
            _db.DB_PATH = old_path

    def test_update_nonexistent_returns_false(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            updated = _db.update_shield_subscription_status("sub_fake", "cancelled")
            assert updated is False
        finally:
            _db.DB_PATH = old_path

    def test_get_by_stripe_id(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_shield.db")
        try:
            _db.init_db()
            raw_key = _db.create_api_key(customer_email="user@example.com")
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            _db.create_shield_subscription(
                api_key_hash=key_hash, tier="pro",
                stripe_subscription_id="sub_lookup_test",
            )

            sub = _db.get_shield_subscription_by_stripe_id("sub_lookup_test")
            assert sub is not None
            assert sub["tier"] == "pro"

            missing = _db.get_shield_subscription_by_stripe_id("sub_nonexistent")
            assert missing is None
        finally:
            _db.DB_PATH = old_path


# ---------------------------------------------------------------------------
# API-level endpoint tests
# ---------------------------------------------------------------------------

class TestShieldSubscribeEndpoint:
    """Verify POST /shield/subscribe endpoint."""

    def test_requires_api_key(self, client):
        with patch("api.STRIPE_SECRET_KEY", "sk_test_fake"):
            resp = client.post("/shield/subscribe", json={"tier": "starter"})
        assert resp.status_code == 401

    def test_rejects_invalid_tier(self, client):
        """Should return 400 for unknown tier, but needs a valid API key first."""
        import db as _db
        _db.init_db()
        raw_key = _db.create_api_key(customer_email="test@example.com", calls_total=100)

        with patch("api.STRIPE_SECRET_KEY", "sk_test_fake"):
            resp = client.post(
                "/shield/subscribe",
                json={"tier": "mega_ultra"},
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 400
        assert "Invalid tier" in resp.json()["detail"]

    def test_returns_503_without_stripe(self, client):
        """Without STRIPE_SECRET_KEY, should return 503."""
        import db as _db
        _db.init_db()
        raw_key = _db.create_api_key(customer_email="test@example.com", calls_total=100)

        with patch("api.STRIPE_SECRET_KEY", ""):
            resp = client.post(
                "/shield/subscribe",
                json={"tier": "starter"},
                headers={"X-API-Key": raw_key},
            )
        assert resp.status_code == 503


class TestShieldWebhookEndpoint:
    """Verify POST /shield/webhook endpoint."""

    def test_rejects_invalid_signature(self, client):
        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"):
            resp = client.post(
                "/shield/webhook",
                content=b"{}",
                headers={"stripe-signature": "bad_sig"},
            )
        assert resp.status_code == 400

    def test_returns_503_without_webhook_secret(self, client):
        with patch("api.STRIPE_WEBHOOK_SECRET", ""):
            resp = client.post("/shield/webhook", content=b"{}")
        assert resp.status_code == 503

    def test_ignores_non_shield_checkout(self, client):
        """checkout.session.completed without product=shield should be ignored."""
        mock_event = {
            "type": "checkout.session.completed",
            "data": {"object": {"metadata": {"product": "core_api"}}},
        }

        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch("stripe.Webhook.construct_event", return_value=mock_event):
            resp = client.post(
                "/shield/webhook",
                content=b"{}",
                headers={"stripe-signature": "test"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_creates_subscription_on_checkout(self, client):
        """checkout.session.completed with product=shield creates subscription."""
        import db as _db
        _db.init_db()
        raw_key = _db.create_api_key(customer_email="buyer@example.com", calls_total=100)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        mock_event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "id": "cs_test_123",
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
                "metadata": {
                    "product": "shield",
                    "tier": "pro",
                    "api_key_hash": key_hash,
                },
            }},
        }

        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch("stripe.Webhook.construct_event", return_value=mock_event):
            resp = client.post(
                "/shield/webhook",
                content=b"{}",
                headers={"stripe-signature": "test"},
            )
        assert resp.status_code == 200
        assert resp.json()["event"] == "subscription_created"

        # Verify subscription was created in DB
        sub = _db.get_shield_subscription(key_hash)
        assert sub is not None
        assert sub["tier"] == "pro"
        assert sub["stripe_subscription_id"] == "sub_test_789"

    def test_cancels_subscription(self, client):
        """customer.subscription.deleted should cancel the subscription."""
        import db as _db
        _db.init_db()
        raw_key = _db.create_api_key(customer_email="cancel@example.com", calls_total=100)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        _db.create_shield_subscription(
            api_key_hash=key_hash, tier="starter",
            stripe_subscription_id="sub_to_cancel",
        )

        mock_event = {
            "type": "customer.subscription.deleted",
            "data": {"object": {"id": "sub_to_cancel"}},
        }

        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch("stripe.Webhook.construct_event", return_value=mock_event):
            resp = client.post(
                "/shield/webhook",
                content=b"{}",
                headers={"stripe-signature": "test"},
            )
        assert resp.status_code == 200
        assert resp.json()["event"] == "subscription_cancelled"

        # Should no longer be active
        sub = _db.get_shield_subscription(key_hash)
        assert sub is None

    def test_updates_subscription_status(self, client):
        """customer.subscription.updated should update status."""
        import db as _db
        _db.init_db()
        raw_key = _db.create_api_key(customer_email="update@example.com", calls_total=100)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        _db.create_shield_subscription(
            api_key_hash=key_hash, tier="pro",
            stripe_subscription_id="sub_to_update",
        )

        mock_event = {
            "type": "customer.subscription.updated",
            "data": {"object": {"id": "sub_to_update", "status": "past_due"}},
        }

        with patch("api.STRIPE_WEBHOOK_SECRET", "whsec_test"), \
             patch("stripe.Webhook.construct_event", return_value=mock_event):
            resp = client.post(
                "/shield/webhook",
                content=b"{}",
                headers={"stripe-signature": "test"},
            )
        assert resp.status_code == 200
        assert resp.json()["event"] == "subscription_updated"


class TestSchemaV7Migration:
    """Verify v7 migration creates shield_subscriptions table."""

    def test_table_exists_after_init(self, tmp_path):
        import db as _db
        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_v7.db")
        try:
            _db.init_db()
            conn = _db.get_connection()
            try:
                info = conn.execute("PRAGMA table_info(shield_subscriptions)").fetchall()
                col_names = [row["name"] for row in info]
                assert "api_key_hash" in col_names
                assert "tier" in col_names
                assert "status" in col_names
                assert "call_quota" in col_names
                assert "calls_used_this_period" in col_names
                assert "period_start" in col_names
                assert "period_end" in col_names
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
