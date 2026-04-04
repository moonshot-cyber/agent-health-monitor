"""Test configuration for AHM security tests."""

import os
import sys

# Set required env vars before importing api module
os.environ.setdefault("PAYMENT_ADDRESS", "0x" + "a" * 40)

# Ensure api module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient with x402 middleware active (real app)."""
    import db as _db
    _db.init_db()  # Ensure DB tables exist (lifespan may not trigger in CI)
    from api import app
    return TestClient(app, raise_server_exceptions=False)
