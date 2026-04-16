"""Tests for the Olas scanner's status-endpoint surface.

Focus is the /olas-scan/status response shape — in particular the new
registry_total_supply field that lets the dashboard render live
saturation against the on-chain ServiceRegistryL2 count.
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# fetch_registry_total_supply
# ---------------------------------------------------------------------------

class TestFetchRegistryTotalSupply:
    """The read-only helper that powers the dashboard saturation figure."""

    def test_returns_totalsupply_as_int(self):
        """The helper must return a plain Python int — the status endpoint
        serialises directly to JSON so a Web3 return of a BigInt or string
        would leak through as something the dashboard can't format."""
        import olas_scan

        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        with patch.object(olas_scan, "_get_contract",
                          return_value=(mock_w3, mock_contract)):
            with patch.object(olas_scan, "_rpc_call_with_retry",
                              return_value=887):
                result = olas_scan.fetch_registry_total_supply()

        assert result == 887
        assert isinstance(result, int)

    def test_coerces_non_int_result(self):
        """Some web3 builds return totalSupply as a string-formatted int.
        The helper normalises to Python int so downstream formatting
        (saturation %) can't choke on the type."""
        import olas_scan

        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        with patch.object(olas_scan, "_get_contract",
                          return_value=(mock_w3, mock_contract)):
            with patch.object(olas_scan, "_rpc_call_with_retry",
                              return_value="1234"):
                result = olas_scan.fetch_registry_total_supply()

        assert result == 1234
        assert isinstance(result, int)

    def test_calls_totalsupply_not_getservice(self):
        """Regression guard — we want the cheap single-call path, not a
        full registry enumeration. The helper must call contract.functions
        .totalSupply() and nothing else."""
        import olas_scan

        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        captured_fn = {}

        def fake_retry(fn, **_kw):
            captured_fn["fn"] = fn
            return 42

        with patch.object(olas_scan, "_get_contract",
                          return_value=(mock_w3, mock_contract)):
            with patch.object(olas_scan, "_rpc_call_with_retry",
                              side_effect=fake_retry):
                olas_scan.fetch_registry_total_supply()

        # Sanity: the totalSupply() accessor was the one we fetched, and
        # getService was never touched (that would trigger a full scan).
        assert mock_contract.functions.totalSupply.called
        assert not mock_contract.functions.getService.called

    def test_rpc_errors_propagate(self):
        """The helper intentionally does NOT swallow errors — the caller
        (/olas-scan/status) decides how to surface unavailability."""
        import olas_scan

        mock_w3 = MagicMock()
        mock_contract = MagicMock()
        with patch.object(olas_scan, "_get_contract",
                          return_value=(mock_w3, mock_contract)):
            with patch.object(olas_scan, "_rpc_call_with_retry",
                              side_effect=ConnectionError("Base RPC down")):
                import pytest
                with pytest.raises(ConnectionError, match="Base RPC down"):
                    olas_scan.fetch_registry_total_supply()


# ---------------------------------------------------------------------------
# /olas-scan/status endpoint contract
# ---------------------------------------------------------------------------

class TestOlasScanStatusEndpoint:
    """Contract tests for /olas-scan/status — live totalSupply must be
    exposed so the dashboard can render "887 / 900 (98% saturated)"
    and stop looking stalled."""

    def test_status_requires_auth(self, client):
        resp = client.get("/olas-scan/status")
        assert resp.status_code == 401

    def test_status_includes_registry_total_supply(self, client):
        """Successful call must include the new field (int) alongside
        the existing ``running`` and ``next_scheduled_run`` fields."""
        from api import INTERNAL_API_KEY

        with patch("olas_scan.fetch_registry_total_supply", return_value=887):
            resp = client.get(
                "/olas-scan/status",
                headers={"X-Internal-Key": INTERNAL_API_KEY},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "next_scheduled_run" in data
        assert "registry_total_supply" in data
        assert data["registry_total_supply"] == 887

    def test_status_null_on_rpc_failure(self, client):
        """If Base RPC is unreachable the endpoint must still return 200
        with registry_total_supply=null — a stalled RPC should never take
        down the whole admin status view."""
        from api import INTERNAL_API_KEY

        with patch("olas_scan.fetch_registry_total_supply",
                   side_effect=ConnectionError("Base RPC down")):
            resp = client.get(
                "/olas-scan/status",
                headers={"X-Internal-Key": INTERNAL_API_KEY},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["registry_total_supply"] is None
        # Other fields must still be populated so the dashboard can show
        # "supply unavailable, scanner still scheduled" instead of a blank.
        assert "running" in data
        assert "next_scheduled_run" in data

    def test_status_wrong_key_rejected_before_rpc(self, client):
        """Auth must short-circuit before the Base RPC call — otherwise an
        unauthenticated caller could probe the RPC latency as a side
        channel and we'd burn RPC quota on rejected requests."""
        from api import INTERNAL_API_KEY  # noqa: F401 — imported to ensure module loaded

        # If the implementation ordered auth AFTER the RPC, this mock
        # would get called. Assert it never does.
        with patch("olas_scan.fetch_registry_total_supply") as mock_fetch:
            resp = client.get(
                "/olas-scan/status",
                headers={"X-Internal-Key": "wrong-key"},
            )
        assert resp.status_code == 401
        assert not mock_fetch.called
