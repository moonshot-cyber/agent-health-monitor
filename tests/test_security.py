"""Security hardening tests for Agent Health Monitor."""

import hmac
import os
import time

import pytest


class TestNoDefaultCoupons:
    """Verify default coupon codes are not hardcoded."""

    def test_no_hardcoded_defaults(self):
        """VALID_COUPONS should be empty when VALID_COUPONS env is not set."""
        from api import VALID_COUPONS
        # If VALID_COUPONS env var is not set, the set should be empty
        # (no hardcoded fallback codes)
        if not os.getenv("VALID_COUPONS"):
            assert len(VALID_COUPONS) == 0, (
                f"Default coupons should not be hardcoded. Found: {VALID_COUPONS}"
            )

    def test_old_defaults_not_valid(self, client):
        """Previously hardcoded coupon codes should not work."""
        old_defaults = ["TEST-ELITE", "EARLY001", "EARLY002", "EARLY003", "REDDIT50", "SUBSTACK1"]
        for code in old_defaults:
            resp = client.get(f"/coupon/validate/{code}")
            if resp.status_code == 200:
                data = resp.json()
                if not os.getenv("VALID_COUPONS"):
                    assert data["valid"] is False, f"Old default coupon {code} should not be valid"


class TestInternalKeyNotLogged:
    """Verify INTERNAL_API_KEY is not exposed in logs."""

    def test_key_not_in_warning_message(self):
        """The auto-generated key should not appear in log messages."""
        import logging
        import io

        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        # Re-import would re-trigger the warning, but the key is already set
        # Just verify the current warning message pattern
        from api import INTERNAL_API_KEY
        stream = handler.stream
        log_output = stream.getvalue()
        # The key value itself should not appear in logs
        # (We can't easily re-trigger module-level code, so check the source)
        import inspect
        import api
        source = inspect.getsource(api)
        # Verify the logging.warning call does NOT include %s for the key value
        assert 'Generated random key: %s' not in source, (
            "INTERNAL_API_KEY should not be logged with its value"
        )

        root_logger.removeHandler(handler)


class TestConstantTimeComparison:
    """Verify internal key comparison uses constant-time algorithm."""

    def test_uses_hmac_compare_digest(self):
        """InternalKeyBypassMiddleware should use hmac.compare_digest."""
        import inspect
        from api import InternalKeyBypassMiddleware

        source = inspect.getsource(InternalKeyBypassMiddleware)
        assert "hmac.compare_digest" in source, (
            "InternalKeyBypassMiddleware should use hmac.compare_digest, not =="
        )
        assert "internal_key ==" not in source.replace("hmac.compare_digest", ""), (
            "Should not use == for key comparison"
        )


class TestSecurityHeaders:
    """Verify security headers are present in responses."""

    def test_x_content_type_options(self, client):
        resp = client.get("/up")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/up")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy(self, client):
        resp = client.get("/up")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


class TestCORSHeaders:
    """Verify CORS is configured."""

    def test_cors_preflight(self, client):
        resp = client.options(
            "/api/info",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should return CORS headers
        assert "access-control-allow-origin" in resp.headers


class TestAddressValidation:
    """Verify address validation rejects malicious input."""

    @pytest.mark.parametrize("address", [
        "not-an-address",
        "0x123",  # too short
        "0x" + "g" * 40,  # invalid hex
        "0x" + "a" * 41,  # too long
        "<script>alert(1)</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "0x" + "a" * 40 + "extra",  # valid hex prefix + extra chars
    ])
    def test_rejects_invalid_addresses(self, client, address):
        """All endpoints should reject invalid address formats.

        x402 middleware may intercept with 402 before the handler validates,
        which is also a valid defense (attacker can't reach the handler).
        """
        # Test free endpoints that don't go through x402
        free_endpoints = [
            f"/retry/preview/{address}",
            f"/agent/protect/preview/{address}",
        ]
        for endpoint in free_endpoints:
            resp = client.get(endpoint)
            assert resp.status_code in (400, 404, 422), (
                f"{endpoint} should reject invalid address, got {resp.status_code}"
            )

        # Paid endpoints may return 402 (x402 blocks first) or 400
        paid_endpoints = [
            f"/health/{address}",
            f"/risk/{address}",
            f"/optimize/{address}",
            f"/retry/{address}",
        ]
        for endpoint in paid_endpoints:
            resp = client.get(endpoint)
            assert resp.status_code in (400, 402, 404, 422), (
                f"{endpoint} should reject invalid address, got {resp.status_code}"
            )


class TestWebhookSSRF:
    """Verify webhook URL validation prevents SSRF."""

    def test_rejects_localhost(self):
        from api import _is_safe_webhook_url
        assert not _is_safe_webhook_url("http://localhost:8080/hook")
        assert not _is_safe_webhook_url("http://127.0.0.1:8080/hook")

    def test_rejects_private_ips(self):
        from api import _is_safe_webhook_url
        assert not _is_safe_webhook_url("http://10.0.0.1/hook")
        assert not _is_safe_webhook_url("http://192.168.1.1/hook")
        assert not _is_safe_webhook_url("http://172.16.0.1/hook")

    def test_rejects_cloud_metadata(self):
        from api import _is_safe_webhook_url
        assert not _is_safe_webhook_url("http://169.254.169.254/latest/meta-data/")
        assert not _is_safe_webhook_url("http://metadata.google.internal/computeMetadata/v1/")

    def test_rejects_internal_domains(self):
        from api import _is_safe_webhook_url
        assert not _is_safe_webhook_url("http://service.internal/hook")
        assert not _is_safe_webhook_url("http://redis.local/hook")

    def test_rejects_non_http(self):
        from api import _is_safe_webhook_url
        assert not _is_safe_webhook_url("file:///etc/passwd")
        assert not _is_safe_webhook_url("ftp://evil.com/payload")

    def test_accepts_valid_urls(self):
        from api import _is_safe_webhook_url
        assert _is_safe_webhook_url("https://hooks.slack.com/services/T0/B0/xxx")
        assert _is_safe_webhook_url("https://discord.com/api/webhooks/123/abc")
        assert _is_safe_webhook_url("https://my-server.com/webhook")


class TestCouponRateLimit:
    """Verify coupon validation is rate-limited."""

    def test_rate_limit_enforced(self, client):
        """Should return 429 after too many attempts."""
        # Reset rate limit state
        from api import _coupon_rate
        _coupon_rate.clear()

        for i in range(5):
            resp = client.get(f"/coupon/validate/TEST{i}")
            assert resp.status_code == 200

        # 6th attempt should be rate-limited
        resp = client.get("/coupon/validate/TEST6")
        assert resp.status_code == 429


class TestChatRateLimit:
    """Verify chat rate limiting works and cleans up."""

    def test_rate_limit_enforced(self, client):
        """Should return 429 after 10 messages."""
        from api import _chat_rate
        _chat_rate.clear()

        for i in range(10):
            resp = client.post("/chat", json={"message": f"test {i}"})
            # May be 503 if ANTHROPIC_API_KEY not set, but should not be 429
            assert resp.status_code != 429

        # 11th should be rate-limited
        resp = client.post("/chat", json={"message": "one too many"})
        assert resp.status_code == 429


class TestRootEndpoint:
    """Verify root endpoint handles missing static dir gracefully."""

    def test_root_no_crash(self, client):
        """Root should return 200 even without static/index.html."""
        resp = client.get("/")
        assert resp.status_code == 200


class TestPrefixOrdering:
    """Verify _ADDRESS_PREFIXES are ordered so longer prefixes match first."""

    def test_prefixes_sorted_by_length_descending(self):
        """Prefixes must be sorted longest-first so /risk/premium/ matches
        before /risk/. If someone adds a new prefix out of order, this fails."""
        from api import AddressValidationMiddleware
        prefixes = AddressValidationMiddleware._ADDRESS_PREFIXES
        assert prefixes == tuple(sorted(prefixes, key=len, reverse=True)), (
            "AddressValidationMiddleware._ADDRESS_PREFIXES must be sorted by "
            "length descending. Shorter prefixes like /risk/ would otherwise "
            "shadow longer ones like /risk/premium/."
        )


ZERO_ADDR = "0x0000000000000000000000000000000000000001"

# Paid endpoints: x402 middleware should return 402.
_PAID_ENDPOINTS = [
    f"/health/{ZERO_ADDR}",
    f"/risk/{ZERO_ADDR}",
    f"/risk/premium/{ZERO_ADDR}",
    f"/counterparties/{ZERO_ADDR}",
    f"/network-map/{ZERO_ADDR}",
    f"/ahs/{ZERO_ADDR}",
    f"/optimize/{ZERO_ADDR}",
    f"/retry/{ZERO_ADDR}",
    f"/agent/protect/{ZERO_ADDR}",
    f"/alerts/subscribe/{ZERO_ADDR}",
]

# Free endpoints: should return 200 (or another valid code), not 400.
_FREE_ENDPOINTS = [
    f"/retry/preview/{ZERO_ADDR}",
    f"/agent/protect/preview/{ZERO_ADDR}",
    f"/alerts/status/{ZERO_ADDR}",
    f"/alerts/unsubscribe/{ZERO_ADDR}",
]


class TestMiddlewareRouting:
    """Verify valid addresses reach the correct handler through the middleware.

    A prefix-ordering bug (e.g. /risk/ before /risk/premium/) causes the
    middleware to extract a wrong address segment, fail the hex regex, and
    return 400 — preventing x402 from returning 402 on paid endpoints.
    """

    @pytest.mark.parametrize("endpoint", _PAID_ENDPOINTS,
                             ids=[e.split("/")[1] + ("/" + e.split("/")[2] if len(e.split("/")) > 3 and not e.split("/")[2].startswith("0x") else "") for e in _PAID_ENDPOINTS])
    def test_paid_endpoint_returns_402(self, client, endpoint):
        """Paid endpoints must return 402 (x402 paywall), never 400."""
        resp = client.get(endpoint)
        assert resp.status_code == 402, (
            f"{endpoint} returned {resp.status_code}; expected 402. "
            "If 400, the address-validation middleware likely matched the "
            "wrong prefix — check _ADDRESS_PREFIXES ordering."
        )

    @pytest.mark.parametrize("endpoint", _FREE_ENDPOINTS,
                             ids=[e.split("/")[1] + "/" + e.split("/")[2] for e in _FREE_ENDPOINTS])
    def test_free_endpoint_not_blocked(self, client, endpoint):
        """Free endpoints must not be blocked by the address-validation
        middleware (status 400) for a valid address."""
        resp = client.get(endpoint)
        assert resp.status_code != 400, (
            f"{endpoint} returned 400 for a valid address. "
            "The address-validation middleware likely matched the wrong "
            "prefix — check _ADDRESS_PREFIXES ordering."
        )


class TestGitignore:
    """Verify .gitignore covers sensitive files."""

    def test_env_covered(self):
        gitignore = open(
            os.path.join(os.path.dirname(__file__), "..", ".gitignore")
        ).read()
        assert ".env" in gitignore
        assert ".env.*" in gitignore
        assert "*.pem" in gitignore
        assert "*.key" in gitignore
