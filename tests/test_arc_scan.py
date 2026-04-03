"""Tests for Arc Protocol (ERC-8004) agent registry scanner."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# arc_scan module-level tests
# ---------------------------------------------------------------------------

class TestArcScanConfig:
    """Verify arc_scan configuration and constants."""

    def test_default_rpc_url(self):
        import arc_scan
        assert "rpc.testnet.arc.network" in arc_scan.ARC_RPC_URL

    def test_identity_registry_address(self):
        import arc_scan
        assert arc_scan.IDENTITY_REGISTRY_ADDRESS == "0x8004A818BFB912233c491871b3d84c89A494BD9e"

    def test_abi_has_required_entries(self):
        import arc_scan
        abi = arc_scan.IDENTITY_REGISTRY_ABI
        names = {entry.get("name") for entry in abi}
        assert "ownerOf" in names
        assert "tokenURI" in names
        assert "getAgentWallet" in names
        assert "Registered" in names

    def test_abi_registered_event_structure(self):
        import arc_scan
        event = next(e for e in arc_scan.IDENTITY_REGISTRY_ABI if e["name"] == "Registered")
        assert event["type"] == "event"
        input_names = {i["name"] for i in event["inputs"]}
        assert "agentId" in input_names
        assert "agentURI" in input_names
        assert "owner" in input_names
        # agentId and owner should be indexed
        indexed = {i["name"] for i in event["inputs"] if i.get("indexed")}
        assert "agentId" in indexed
        assert "owner" in indexed

    def test_abi_get_agent_wallet_signature(self):
        import arc_scan
        fn = next(e for e in arc_scan.IDENTITY_REGISTRY_ABI if e["name"] == "getAgentWallet")
        assert fn["type"] == "function"
        assert fn["stateMutability"] == "view"
        assert fn["inputs"][0]["type"] == "uint256"
        assert fn["outputs"][0]["type"] == "address"


# ---------------------------------------------------------------------------
# Discovery logic tests (mocked RPC)
# ---------------------------------------------------------------------------

class TestDiscoverArcWallets:
    """Test wallet discovery with mocked Web3 interactions."""

    def _make_mock_event(self, agent_id, owner, agent_uri="https://example.com/meta.json"):
        """Create a mock Registered event."""
        return {
            "args": {
                "agentId": agent_id,
                "owner": owner,
                "agentURI": agent_uri,
            },
            "blockNumber": 100 + agent_id,
        }

    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    @patch("arc_scan._rpc_call_with_retry")
    def test_discover_with_agent_wallets(self, mock_rpc, mock_events, mock_contract):
        """Agents with dedicated wallets should use agent wallet address."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)

        mock_events.return_value = [
            self._make_mock_event(1, "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            self._make_mock_event(2, "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"),
        ]

        # Events are sorted by agentId descending, so agent 2 is processed first
        mock_c.functions.getAgentWallet.return_value.call.side_effect = [
            "0x0000000000000000000000000000000000000000",  # agent 2 (processed first) — no wallet
            "0x1111111111111111111111111111111111111111",  # agent 1 — has wallet
        ]

        # Bypass the retry wrapper — call the contract directly
        with patch.object(arc_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = arc_scan.discover_arc_wallets(max_agents=10)

        assert len(wallets) == 2
        # Agent 2 (first processed) falls back to owner
        assert wallets[0]["source"] == "arc_owner"
        assert wallets[0]["address"] == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        # Agent 1 uses agent_wallet
        assert wallets[1]["source"] == "arc_agent_wallet"
        assert wallets[1]["address"] == "0x1111111111111111111111111111111111111111"

    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_deduplicates_addresses(self, mock_events, mock_contract):
        """Same address from multiple agents should only appear once."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)

        # Two agents owned by the same address, both with no agent wallet
        same_owner = "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        mock_events.return_value = [
            self._make_mock_event(1, same_owner),
            self._make_mock_event(2, same_owner),
        ]

        mock_c.functions.getAgentWallet.return_value.call.return_value = \
            "0x0000000000000000000000000000000000000000"

        with patch.object(arc_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = arc_scan.discover_arc_wallets(max_agents=10)

        assert len(wallets) == 1

    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_empty_registry(self, mock_events, mock_contract):
        """Empty registry should return empty list."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        wallets = arc_scan.discover_arc_wallets()
        assert wallets == []

    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_respects_max_agents(self, mock_events, mock_contract):
        """Should stop after max_agents wallets are found."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)

        # Create 10 agents
        events = [
            self._make_mock_event(i, f"0x{i:040x}")
            for i in range(1, 11)
        ]
        mock_events.return_value = events

        mock_c.functions.getAgentWallet.return_value.call.return_value = \
            "0x0000000000000000000000000000000000000000"

        with patch.object(arc_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = arc_scan.discover_arc_wallets(max_agents=3)

        assert len(wallets) == 3

    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_handles_rpc_errors_gracefully(self, mock_events, mock_contract):
        """RPC errors for individual agents should not crash the scan."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)

        mock_events.return_value = [
            self._make_mock_event(1, "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            self._make_mock_event(2, "0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"),
        ]

        call_count = 0
        def error_then_success(fn, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("RPC timeout")
            return "0x0000000000000000000000000000000000000000"

        with patch.object(arc_scan, "_rpc_call_with_retry", side_effect=error_then_success):
            wallets = arc_scan.discover_arc_wallets(max_agents=10)

        # Agent 2 (processed first, descending) should fail (error),
        # agent 1 should succeed (owner fallback)
        assert len(wallets) == 1
        assert wallets[0]["address"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# RPC retry logic tests
# ---------------------------------------------------------------------------

class TestRpcRetry:
    """Test exponential backoff retry logic."""

    def test_success_on_first_attempt(self):
        import arc_scan

        mock_fn = MagicMock()
        mock_fn.call.return_value = 42

        result = arc_scan._rpc_call_with_retry(mock_fn, label="test")
        assert result == 42
        assert mock_fn.call.call_count == 1

    @patch("arc_scan.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        import arc_scan

        mock_fn = MagicMock()
        mock_fn.call.side_effect = [
            Exception("429 Too Many Requests"),
            42,
        ]

        result = arc_scan._rpc_call_with_retry(mock_fn, label="test")
        assert result == 42
        assert mock_fn.call.call_count == 2
        assert mock_sleep.called

    def test_raises_on_non_retryable_error(self):
        import arc_scan

        mock_fn = MagicMock()
        mock_fn.call.side_effect = ValueError("invalid agent ID")

        with pytest.raises(ValueError, match="invalid agent ID"):
            arc_scan._rpc_call_with_retry(mock_fn, label="test")


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestArcScanEndpoints:
    """Test /arc-scan/* API endpoints."""

    def test_arc_scan_status_requires_auth(self, client):
        """Status endpoint should reject unauthenticated requests."""
        resp = client.get("/arc-scan/status")
        assert resp.status_code == 401

    def test_arc_scan_status_with_valid_key(self, client):
        """Status endpoint should return data with valid key."""
        from api import INTERNAL_API_KEY
        resp = client.get(
            "/arc-scan/status",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "next_scheduled_run" in data

    def test_arc_scan_trigger_requires_auth(self, client):
        """Trigger endpoint should reject unauthenticated requests."""
        resp = client.post("/arc-scan/trigger")
        assert resp.status_code == 401

    def test_arc_scan_trigger_with_valid_key(self, client):
        """Trigger endpoint should accept requests with valid key."""
        from api import INTERNAL_API_KEY
        resp = client.post(
            "/arc-scan/trigger",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        # Should be 200 (triggered) or 409 (already running)
        assert resp.status_code in (200, 409)

    def test_arc_scan_status_wrong_key(self, client):
        """Status endpoint should reject wrong key."""
        resp = client.get(
            "/arc-scan/status",
            headers={"X-Internal-Key": "wrong-key"},
        )
        assert resp.status_code == 401
