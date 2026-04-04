"""Tests for Arc Protocol (ERC-8004) agent registry scanner."""

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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

    def test_event_chunk_size_is_10k(self):
        import arc_scan
        assert arc_scan.EVENT_CHUNK_SIZE == 10_000


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Test block checkpoint load/save."""

    def test_load_missing_checkpoint_returns_empty(self, tmp_path):
        import arc_scan
        with patch.object(arc_scan, "CHECKPOINT_PATH", tmp_path / "missing.json"):
            result = arc_scan.load_checkpoint()
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        import arc_scan
        cp_path = tmp_path / "cp.json"
        with patch.object(arc_scan, "CHECKPOINT_PATH", cp_path):
            arc_scan.save_checkpoint(last_scanned_block=50000, events_found=42)
            result = arc_scan.load_checkpoint()
        assert result["last_scanned_block"] == 50000
        assert result["events_found"] == 42
        assert "saved_at" in result

    def test_load_corrupt_checkpoint_returns_empty(self, tmp_path):
        import arc_scan
        cp_path = tmp_path / "bad.json"
        cp_path.write_text("not json{{{")
        with patch.object(arc_scan, "CHECKPOINT_PATH", cp_path):
            result = arc_scan.load_checkpoint()
        assert result == {}

    def test_save_checkpoint_failure_is_nonfatal(self, tmp_path):
        """save_checkpoint should not raise even if write fails."""
        import arc_scan
        # Point to a directory (not a file) so write_text fails
        with patch.object(arc_scan, "CHECKPOINT_PATH", tmp_path):
            arc_scan.save_checkpoint(last_scanned_block=100, events_found=0)
            # No exception raised


# ---------------------------------------------------------------------------
# Chunked event fetching tests
# ---------------------------------------------------------------------------

class TestFetchRegisteredEvents:
    """Test _fetch_registered_events chunked fetching."""

    def test_fetches_in_10k_chunks(self):
        """Events should be fetched in EVENT_CHUNK_SIZE-block windows."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 25000  # 0→25000 = ~3 chunks of 10k
        mock_contract = MagicMock()

        event1 = {"args": {"agentId": 1, "owner": "0xAAA"}, "blockNumber": 5000}
        event2 = {"args": {"agentId": 2, "owner": "0xBBB"}, "blockNumber": 15000}

        # Return events only on the first and second chunks
        filter_mock = MagicMock()
        call_count = 0

        def create_filter_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            m = MagicMock()
            if call_count == 1:
                m.get_all_entries.return_value = [event1]
            elif call_count == 2:
                m.get_all_entries.return_value = [event2]
            else:
                m.get_all_entries.return_value = []
            return m

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        events = arc_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2
        # Should have made 3 chunk calls: 0-9999, 10000-19999, 20000-25000
        assert call_count == 3

    def test_reduces_chunk_on_413_error(self):
        """413 errors should trigger chunk size reduction and retry."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 15000
        mock_contract = MagicMock()

        call_history = []

        def create_filter_side_effect(**kwargs):
            from_b = kwargs.get("from_block", 0)
            to_b = kwargs.get("to_block", 0)
            call_history.append((from_b, to_b))
            m = MagicMock()

            # First call (0-9999): raise 413
            if len(call_history) == 1:
                raise Exception("413 Client Error: Request Entity Too Large")

            # All subsequent calls succeed with no events
            m.get_all_entries.return_value = []
            return m

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        events = arc_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 0
        # First call should be 0-9999, then retry with halved chunk 0-4999
        assert call_history[0] == (0, 9999)
        assert call_history[1][0] == 0
        assert call_history[1][1] < 9999  # chunk was reduced

    def test_raises_on_non_range_error(self):
        """Non-range errors should propagate immediately."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 5000
        mock_contract = MagicMock()

        mock_contract.events.Registered.create_filter.side_effect = \
            ConnectionError("RPC node unreachable")

        with pytest.raises(ConnectionError, match="RPC node unreachable"):
            arc_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

    def test_handles_too_large_in_error_string(self):
        """Various RPC error formats containing 'too large' should be caught."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 5000
        mock_contract = MagicMock()

        retries = []

        def create_filter_side_effect(**kwargs):
            retries.append(kwargs)
            if len(retries) == 1:
                raise Exception("Response payload too large")
            m = MagicMock()
            m.get_all_entries.return_value = []
            return m

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        # Should not raise — the "too large" error is caught and chunk is reduced
        events = arc_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)
        assert len(retries) > 1  # at least one retry happened

    def test_chunk_reduction_floor_at_1000(self):
        """Chunk size should never go below 1,000 blocks."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 5000
        mock_contract = MagicMock()

        # Keep failing with range errors — eventually chunk hits 1000 floor
        # and then the error re-raises
        call_count = 0

        def create_filter_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("413 Client Error: Request Entity Too Large")

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        with pytest.raises(Exception, match="413"):
            arc_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        # Should have retried several times before giving up at 1000-block floor
        assert call_count >= 3


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

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    @patch("arc_scan._rpc_call_with_retry")
    def test_discover_with_agent_wallets(self, mock_rpc, mock_events, mock_contract,
                                         mock_load_cp, mock_save_cp):
        """Agents with dedicated wallets should use agent wallet address."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
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

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_deduplicates_addresses(self, mock_events, mock_contract,
                                              mock_load_cp, mock_save_cp):
        """Same address from multiple agents should only appear once."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
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

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_empty_registry(self, mock_events, mock_contract,
                                      mock_load_cp, mock_save_cp):
        """Empty registry should return empty list."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        wallets = arc_scan.discover_arc_wallets()
        assert wallets == []

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_respects_max_agents(self, mock_events, mock_contract,
                                           mock_load_cp, mock_save_cp):
        """Should stop after max_agents wallets are found."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
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

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_handles_rpc_errors_gracefully(self, mock_events, mock_contract,
                                                     mock_load_cp, mock_save_cp):
        """RPC errors for individual agents should not crash the scan."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
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

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={"last_scanned_block": 30000})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_resumes_from_checkpoint(self, mock_events, mock_contract,
                                               mock_load_cp, mock_save_cp):
        """Discovery should resume from the checkpointed block."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        arc_scan.discover_arc_wallets()

        # _fetch_registered_events should be called with from_block=30001
        mock_events.assert_called_once_with(mock_w3, mock_c, from_block=30001)

    @patch("arc_scan.save_checkpoint")
    @patch("arc_scan.load_checkpoint", return_value={})
    @patch("arc_scan._get_contract")
    @patch("arc_scan._fetch_registered_events")
    def test_discover_saves_checkpoint_after_scan(self, mock_events, mock_contract,
                                                    mock_load_cp, mock_save_cp):
        """Checkpoint should be saved with the latest block after scanning."""
        import arc_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 5042002
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        arc_scan.discover_arc_wallets()

        mock_save_cp.assert_called_once_with(50000, 0)


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
