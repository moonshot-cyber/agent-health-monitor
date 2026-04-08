"""Tests for Celo ERC-8004 agent registry scanner."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# celo_scan module-level tests
# ---------------------------------------------------------------------------

class TestCeloScanConfig:
    """Verify celo_scan configuration and constants."""

    def test_default_rpc_url(self):
        import celo_scan
        assert "forno.celo.org" in celo_scan.CELO_RPC_URL

    def test_identity_registry_address(self):
        """Address must match the canonical ERC-8004 deployment on Celo mainnet."""
        import celo_scan
        # Verified against the official erc-8004/erc-8004-contracts repo README
        # AND Celoscan ("8004: Identity Registry", ERC1967Proxy delegating to
        # IdentityRegistryUpgradeable). Same address as Base mainnet because
        # the contracts are deployed via deterministic CREATE2.
        assert celo_scan.IDENTITY_REGISTRY_ADDRESS == "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"

    def test_celo_chain_id(self):
        import celo_scan
        assert celo_scan.CELO_CHAIN_ID == 42220

    def test_abi_has_required_entries(self):
        import celo_scan
        abi = celo_scan.IDENTITY_REGISTRY_ABI
        names = {entry.get("name") for entry in abi}
        assert "ownerOf" in names
        assert "tokenURI" in names
        assert "getAgentWallet" in names
        assert "Registered" in names

    def test_abi_registered_event_structure(self):
        import celo_scan
        event = next(e for e in celo_scan.IDENTITY_REGISTRY_ABI if e["name"] == "Registered")
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
        import celo_scan
        fn = next(e for e in celo_scan.IDENTITY_REGISTRY_ABI if e["name"] == "getAgentWallet")
        assert fn["type"] == "function"
        assert fn["stateMutability"] == "view"
        assert fn["inputs"][0]["type"] == "uint256"
        assert fn["outputs"][0]["type"] == "address"

    def test_event_chunk_size_is_conservative(self):
        """Default chunk size must stay well under forno.celo.org's
        eth_getLogs range limit (observed failures at 10,000 blocks)."""
        import celo_scan
        assert celo_scan.EVENT_CHUNK_SIZE <= 1_000
        assert celo_scan.EVENT_CHUNK_SIZE >= celo_scan.MIN_EVENT_CHUNK_SIZE

    def test_chunk_size_floor_is_defined(self):
        """A non-zero floor must exist so adaptive halving can't degenerate."""
        import celo_scan
        assert celo_scan.MIN_EVENT_CHUNK_SIZE > 0
        assert celo_scan.MIN_EVENT_CHUNK_SIZE <= celo_scan.EVENT_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Test block checkpoint load/save."""

    def test_load_missing_checkpoint_returns_empty(self, tmp_path):
        import celo_scan
        with patch.object(celo_scan, "CHECKPOINT_PATH", tmp_path / "missing.json"):
            result = celo_scan.load_checkpoint()
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        import celo_scan
        cp_path = tmp_path / "cp.json"
        with patch.object(celo_scan, "CHECKPOINT_PATH", cp_path):
            celo_scan.save_checkpoint(last_scanned_block=50000, events_found=42)
            result = celo_scan.load_checkpoint()
        assert result["last_scanned_block"] == 50000
        assert result["events_found"] == 42
        assert "saved_at" in result

    def test_load_corrupt_checkpoint_returns_empty(self, tmp_path):
        import celo_scan
        cp_path = tmp_path / "bad.json"
        cp_path.write_text("not json{{{")
        with patch.object(celo_scan, "CHECKPOINT_PATH", cp_path):
            result = celo_scan.load_checkpoint()
        assert result == {}

    def test_save_checkpoint_failure_is_nonfatal(self, tmp_path):
        """save_checkpoint should not raise even if write fails."""
        import celo_scan
        # Point to a directory (not a file) so write_text fails
        with patch.object(celo_scan, "CHECKPOINT_PATH", tmp_path):
            celo_scan.save_checkpoint(last_scanned_block=100, events_found=0)
            # No exception raised


# ---------------------------------------------------------------------------
# Chunked event fetching tests
# ---------------------------------------------------------------------------

class TestFetchRegisteredEvents:
    """Test _fetch_registered_events chunked fetching."""

    def test_fetches_in_fixed_chunks(self):
        """Events should be fetched in EVENT_CHUNK_SIZE-block windows."""
        import celo_scan

        chunk = celo_scan.EVENT_CHUNK_SIZE
        # Pick a range that yields exactly 3 chunks: 0→(3*chunk - 1)
        latest = chunk * 3 - 1
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = latest
        mock_contract = MagicMock()

        event1 = {"args": {"agentId": 1, "owner": "0xAAA"}, "blockNumber": chunk // 2}
        event2 = {"args": {"agentId": 2, "owner": "0xBBB"}, "blockNumber": chunk + chunk // 2}

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

        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2
        # Three non-overlapping chunks: 0→c-1, c→2c-1, 2c→3c-1
        assert call_count == 3

    def test_reduces_chunk_on_413_error(self):
        """413 errors should trigger chunk size reduction and retry."""
        import celo_scan

        chunk = celo_scan.EVENT_CHUNK_SIZE
        mock_w3 = MagicMock()
        # A range that definitely needs more than one chunk at default size
        mock_w3.eth.block_number = chunk * 3
        mock_contract = MagicMock()

        call_history = []

        def create_filter_side_effect(**kwargs):
            from_b = kwargs.get("from_block", 0)
            to_b = kwargs.get("to_block", 0)
            call_history.append((from_b, to_b))
            m = MagicMock()

            # First call at full chunk size: raise 413
            if len(call_history) == 1:
                raise Exception("413 Client Error: Request Entity Too Large")

            # All subsequent calls succeed with no events
            m.get_all_entries.return_value = []
            return m

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 0
        # First call should cover the full default chunk, the retry should
        # start from the same block with a strictly smaller window.
        first_from, first_to = call_history[0]
        assert first_from == 0
        assert first_to == chunk - 1
        second_from, second_to = call_history[1]
        assert second_from == 0
        assert (second_to - second_from + 1) < chunk  # halved window

    def test_raises_after_exhausting_chunk_halving(self):
        """If a persistent error survives all halvings down to the floor,
        the original exception type must propagate."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
        mock_contract = MagicMock()

        mock_contract.events.Registered.create_filter.side_effect = \
            ConnectionError("RPC node unreachable")

        with pytest.raises(ConnectionError, match="RPC node unreachable"):
            celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

    def test_handles_too_large_in_error_string(self):
        """Various RPC error formats containing 'too large' should be caught."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
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
        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)
        assert len(retries) > 1  # at least one retry happened

    def test_chunk_reduction_floor(self):
        """Chunk size should never go below MIN_EVENT_CHUNK_SIZE blocks."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
        mock_contract = MagicMock()

        observed_sizes: list[int] = []

        def create_filter_side_effect(**kwargs):
            from_b = kwargs.get("from_block", 0)
            to_b = kwargs.get("to_block", 0)
            observed_sizes.append(to_b - from_b + 1)
            raise Exception("413 Client Error: Request Entity Too Large")

        mock_contract.events.Registered.create_filter.side_effect = create_filter_side_effect

        with pytest.raises(Exception, match="413"):
            celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        # Should have halved at least once before propagating
        assert len(observed_sizes) >= 2
        # The smallest observed window must be >= the floor (never below)
        assert min(observed_sizes) >= celo_scan.MIN_EVENT_CHUNK_SIZE

    def test_logs_per_chunk_event_count(self, caplog):
        """Each successful chunk should log its event count for diagnosis."""
        import logging

        import celo_scan

        chunk = celo_scan.EVENT_CHUNK_SIZE
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = chunk - 1  # exactly one chunk
        mock_contract = MagicMock()

        sentinel = {"args": {"agentId": 1, "owner": "0xAAA"}, "blockNumber": 1}
        m = MagicMock()
        m.get_all_entries.return_value = [sentinel]
        mock_contract.events.Registered.create_filter.return_value = m

        with caplog.at_level(logging.INFO, logger="ahm.celo_scan"):
            events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 1
        # Per-chunk log must mention the event count so post-mortem is cheap
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "1 events" in joined


# ---------------------------------------------------------------------------
# Discovery logic tests (mocked RPC)
# ---------------------------------------------------------------------------

class TestDiscoverCeloWallets:
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

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    @patch("celo_scan._rpc_call_with_retry")
    def test_discover_with_agent_wallets(self, mock_rpc, mock_events, mock_contract,
                                         mock_load_cp, mock_save_cp):
        """Agents with dedicated wallets should use agent wallet address."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
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
        with patch.object(celo_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = celo_scan.discover_celo_wallets(max_agents=10)

        assert len(wallets) == 2
        # Agent 2 (first processed) falls back to owner
        assert wallets[0]["source"] == "celo_owner"
        assert wallets[0]["address"] == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        # Agent 1 uses agent_wallet
        assert wallets[1]["source"] == "celo_agent_wallet"
        assert wallets[1]["address"] == "0x1111111111111111111111111111111111111111"

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_deduplicates_addresses(self, mock_events, mock_contract,
                                              mock_load_cp, mock_save_cp):
        """Same address from multiple agents should only appear once."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
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

        with patch.object(celo_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = celo_scan.discover_celo_wallets(max_agents=10)

        assert len(wallets) == 1

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_empty_registry(self, mock_events, mock_contract,
                                      mock_load_cp, mock_save_cp):
        """Empty registry should return empty list."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        wallets = celo_scan.discover_celo_wallets()
        assert wallets == []

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_respects_max_agents(self, mock_events, mock_contract,
                                           mock_load_cp, mock_save_cp):
        """Should stop after max_agents wallets are found."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)

        events = [
            self._make_mock_event(i, f"0x{i:040x}")
            for i in range(1, 11)
        ]
        mock_events.return_value = events

        mock_c.functions.getAgentWallet.return_value.call.return_value = \
            "0x0000000000000000000000000000000000000000"

        with patch.object(celo_scan, "_rpc_call_with_retry",
                          side_effect=lambda fn, **kw: fn.call()):
            wallets = celo_scan.discover_celo_wallets(max_agents=3)

        assert len(wallets) == 3

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_handles_rpc_errors_gracefully(self, mock_events, mock_contract,
                                                     mock_load_cp, mock_save_cp):
        """RPC errors for individual agents should not crash the scan."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
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

        with patch.object(celo_scan, "_rpc_call_with_retry", side_effect=error_then_success):
            wallets = celo_scan.discover_celo_wallets(max_agents=10)

        # Agent 2 (processed first, descending) should fail (error),
        # agent 1 should succeed (owner fallback)
        assert len(wallets) == 1
        assert wallets[0]["address"] == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={"last_scanned_block": 30000})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_resumes_from_checkpoint(self, mock_events, mock_contract,
                                               mock_load_cp, mock_save_cp):
        """Discovery should resume from the checkpointed block."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        celo_scan.discover_celo_wallets()

        # _fetch_registered_events should be called with from_block=30001
        mock_events.assert_called_once_with(mock_w3, mock_c, from_block=30001)

    @patch("celo_scan.save_checkpoint")
    @patch("celo_scan.load_checkpoint", return_value={})
    @patch("celo_scan._get_contract")
    @patch("celo_scan._fetch_registered_events")
    def test_discover_saves_checkpoint_after_scan(self, mock_events, mock_contract,
                                                    mock_load_cp, mock_save_cp):
        """Checkpoint should be saved with the latest block after scanning."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.chain_id = 42220
        mock_w3.eth.block_number = 50000
        mock_c = MagicMock()
        mock_contract.return_value = (mock_w3, mock_c)
        mock_events.return_value = []

        celo_scan.discover_celo_wallets()

        mock_save_cp.assert_called_once_with(50000, 0)


# ---------------------------------------------------------------------------
# RPC retry logic tests
# ---------------------------------------------------------------------------

class TestRpcRetry:
    """Test exponential backoff retry logic."""

    def test_success_on_first_attempt(self):
        import celo_scan

        mock_fn = MagicMock()
        mock_fn.call.return_value = 42

        result = celo_scan._rpc_call_with_retry(mock_fn, label="test")
        assert result == 42
        assert mock_fn.call.call_count == 1

    @patch("celo_scan.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        import celo_scan

        mock_fn = MagicMock()
        mock_fn.call.side_effect = [
            Exception("429 Too Many Requests"),
            42,
        ]

        result = celo_scan._rpc_call_with_retry(mock_fn, label="test")
        assert result == 42
        assert mock_fn.call.call_count == 2
        assert mock_sleep.called

    def test_raises_on_non_retryable_error(self):
        import celo_scan

        mock_fn = MagicMock()
        mock_fn.call.side_effect = ValueError("invalid agent ID")

        with pytest.raises(ValueError, match="invalid agent ID"):
            celo_scan._rpc_call_with_retry(mock_fn, label="test")


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------

class TestCeloEcosystemStats:
    """Verify the ecosystem stats endpoint surfaces Celo as a data source."""

    def test_db_data_sources_recognises_celo_agent_wallet(self, tmp_path, monkeypatch):
        """Wallets sourced as 'celo_agent_wallet' should appear under the 'Celo' bucket."""
        import db

        db_path = tmp_path / "test.db"
        monkeypatch.setattr(db, "DB_PATH", str(db_path))
        db.init_db()

        now_iso = "2026-04-07T00:00:00Z"
        # Insert Celo-sourced known_wallets with populated latest_ahs
        conn = db.get_connection()
        try:
            conn.execute(
                """INSERT INTO known_wallets (address, source, first_seen_at, latest_ahs, latest_grade)
                   VALUES (?, ?, ?, ?, ?)""",
                ("0x" + "c" * 40, "celo_agent_wallet", now_iso, 75, "B"),
            )
            conn.execute(
                """INSERT INTO known_wallets (address, source, first_seen_at, latest_ahs, latest_grade)
                   VALUES (?, ?, ?, ?, ?)""",
                ("0x" + "d" * 40, "celo_owner", now_iso, 60, "C"),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.get_ecosystem_dashboard_stats()
        assert "Celo" in stats["data_sources"]
        assert stats["data_sources"]["Celo"] == 2

    def test_db_data_sources_celo_does_not_collide_with_arc(self, tmp_path, monkeypatch):
        """Celo source labels must not bucket into the 'Arc' bucket."""
        import db

        db_path = tmp_path / "test.db"
        monkeypatch.setattr(db, "DB_PATH", str(db_path))
        db.init_db()

        now_iso = "2026-04-07T00:00:00Z"
        conn = db.get_connection()
        try:
            conn.execute(
                """INSERT INTO known_wallets (address, source, first_seen_at, latest_ahs, latest_grade)
                   VALUES (?, ?, ?, ?, ?)""",
                ("0x" + "e" * 40, "celo_agent_wallet", now_iso, 80, "B"),
            )
            conn.commit()
        finally:
            conn.close()

        stats = db.get_ecosystem_dashboard_stats()
        assert stats["data_sources"].get("Celo") == 1
        assert "Arc" not in stats["data_sources"]


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestCeloScanEndpoints:
    """Test /celo-scan/* API endpoints."""

    def test_celo_scan_status_requires_auth(self, client):
        """Status endpoint should reject unauthenticated requests."""
        resp = client.get("/celo-scan/status")
        assert resp.status_code == 401

    def test_celo_scan_status_with_valid_key(self, client):
        """Status endpoint should return data with valid key."""
        from api import INTERNAL_API_KEY
        resp = client.get(
            "/celo-scan/status",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "next_scheduled_run" in data

    def test_celo_scan_trigger_requires_auth(self, client):
        """Trigger endpoint should reject unauthenticated requests."""
        resp = client.post("/celo-scan/trigger")
        assert resp.status_code == 401

    def test_celo_scan_trigger_with_valid_key(self, client):
        """Trigger endpoint should accept requests with valid key."""
        from api import INTERNAL_API_KEY
        resp = client.post(
            "/celo-scan/trigger",
            headers={"X-Internal-Key": INTERNAL_API_KEY},
        )
        # Should be 200 (triggered) or 409 (already running)
        assert resp.status_code in (200, 409)

    def test_celo_scan_status_wrong_key(self, client):
        """Status endpoint should reject wrong key."""
        resp = client.get(
            "/celo-scan/status",
            headers={"X-Internal-Key": "wrong-key"},
        )
        assert resp.status_code == 401
