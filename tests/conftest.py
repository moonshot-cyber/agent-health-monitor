"""Test configuration for AHM security tests."""

import os
import sys
from unittest.mock import patch

# Set required env vars before importing api module
os.environ.setdefault("PAYMENT_ADDRESS", "0x" + "a" * 40)

# Ensure api module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True, scope="session")
def _mock_x402_facilitator():
    """Mock x402 facilitator so tests pass regardless of facilitator availability.

    The x402 middleware lazily calls facilitator.get_supported() on the first
    protected request.  If the remote facilitator is unreachable, this raises a
    network error (httpx.ConnectError) that bypasses the middleware's
    FacilitatorResponseError handler, surfacing as HTTP 500 instead of 402.

    Mocking get_supported() with a valid SupportedResponse for the test network
    lets initialize() succeed locally and makes the test suite deterministic.
    """
    from x402.http import HTTPFacilitatorClient
    from x402.schemas.responses import SupportedKind, SupportedResponse

    fake_supported = SupportedResponse(
        kinds=[SupportedKind(x402_version=2, scheme="exact", network="eip155:8453")],
    )
    with patch.object(
        HTTPFacilitatorClient, "get_supported", return_value=fake_supported
    ):
        yield


@pytest.fixture
def client():
    """TestClient with x402 middleware active (real app)."""
    import db as _db
    _db.init_db()  # Ensure DB tables exist (lifespan may not trigger in CI)
    from api import app
    return TestClient(app, raise_server_exceptions=False)
