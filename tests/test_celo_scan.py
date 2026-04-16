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
    """Test _fetch_registered_events chunked fetching.

    These tests exercise the stateless ``w3.eth.get_logs`` path that
    replaced the stateful ``contract.events.Registered.create_filter``
    path (which broke on forno.celo.org with -32000 filter not found).
    """

    @staticmethod
    def _passthrough_process_log(mock_contract):
        """Wire ``contract.events.Registered().process_log(log)`` to return
        the raw log unchanged so tests can feed dict "logs" end-to-end."""
        mock_contract.events.Registered.return_value.process_log.side_effect = \
            lambda log: log

    def test_fetches_in_fixed_chunks(self):
        """Events should be fetched in EVENT_CHUNK_SIZE-block windows via
        stateless eth_getLogs."""
        import celo_scan

        chunk = celo_scan.EVENT_CHUNK_SIZE
        # Pick a range that yields exactly 3 chunks: 0→(3*chunk - 1)
        latest = chunk * 3 - 1
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = latest
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)

        event1 = {"args": {"agentId": 1, "owner": "0xAAA"}, "blockNumber": chunk // 2}
        event2 = {"args": {"agentId": 2, "owner": "0xBBB"}, "blockNumber": chunk + chunk // 2}

        call_count = 0

        def get_logs_side_effect(filter_params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [event1]
            elif call_count == 2:
                return [event2]
            return []

        mock_w3.eth.get_logs.side_effect = get_logs_side_effect

        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2
        # Three non-overlapping chunks: 0→c-1, c→2c-1, 2c→3c-1
        assert call_count == 3

    def test_uses_stateless_get_logs_not_create_filter(self):
        """Must use w3.eth.get_logs (stateless) not contract.events.X.create_filter
        (stateful) because forno.celo.org returns -32000 filter not found on
        the follow-up eth_getFilterLogs call."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE - 1
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)
        mock_w3.eth.get_logs.return_value = []

        celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        # Must have used the stateless endpoint
        assert mock_w3.eth.get_logs.called
        # Must NOT have touched the stateful filter API
        assert not mock_contract.events.Registered.create_filter.called

    def test_get_logs_includes_registered_topic_and_address(self):
        """eth_getLogs filter params must specify the contract address and
        the Registered event topic0 so the RPC can prefilter server-side
        and we don't accidentally pull unrelated logs."""
        from web3 import Web3

        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE - 1
        mock_contract = MagicMock()
        mock_contract.address = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
        self._passthrough_process_log(mock_contract)

        captured: list[dict] = []

        def capture(params):
            captured.append(params)
            return []

        mock_w3.eth.get_logs.side_effect = capture

        celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

        assert len(captured) >= 1
        params = captured[0]
        assert params["address"] == mock_contract.address
        assert params["fromBlock"] == 0
        assert params["toBlock"] == celo_scan.EVENT_CHUNK_SIZE - 1
        expected_topic = Web3.to_hex(
            Web3.keccak(text="Registered(uint256,string,address)")
        )
        assert params["topics"] == [expected_topic]

    def test_reduces_chunk_on_413_error(self):
        """413 errors should trigger chunk size reduction and retry."""
        import celo_scan

        chunk = celo_scan.EVENT_CHUNK_SIZE
        mock_w3 = MagicMock()
        # A range that definitely needs more than one chunk at default size
        mock_w3.eth.block_number = chunk * 3
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)

        call_history: list[tuple[int, int]] = []

        def get_logs_side_effect(filter_params):
            from_b = filter_params["fromBlock"]
            to_b = filter_params["toBlock"]
            call_history.append((from_b, to_b))

            # First call at full chunk size: raise 413
            if len(call_history) == 1:
                raise Exception("413 Client Error: Request Entity Too Large")
            # All subsequent calls succeed with no events
            return []

        mock_w3.eth.get_logs.side_effect = get_logs_side_effect

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
        self._passthrough_process_log(mock_contract)

        mock_w3.eth.get_logs.side_effect = ConnectionError("RPC node unreachable")

        with pytest.raises(ConnectionError, match="RPC node unreachable"):
            celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)

    def test_handles_filter_not_found_error(self):
        """The specific forno error (-32000 filter not found) — and any
        other opaque RPC error — must be caught by the halving fallback."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)

        retries: list[dict] = []

        def get_logs_side_effect(params):
            retries.append(params)
            if len(retries) == 1:
                raise Exception("{'code': -32000, 'message': 'filter not found'}")
            return []

        mock_w3.eth.get_logs.side_effect = get_logs_side_effect

        # Should NOT raise — even the opaque -32000 error triggers halving
        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)
        assert events == []
        assert len(retries) > 1  # at least one halving retry fired

    def test_handles_too_large_in_error_string(self):
        """Various RPC error formats containing 'too large' should be caught."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)

        retries: list[dict] = []

        def get_logs_side_effect(params):
            retries.append(params)
            if len(retries) == 1:
                raise Exception("Response payload too large")
            return []

        mock_w3.eth.get_logs.side_effect = get_logs_side_effect

        # Should not raise — the "too large" error is caught and chunk is reduced
        events = celo_scan._fetch_registered_events(mock_w3, mock_contract, from_block=0)
        assert events == []
        assert len(retries) > 1  # at least one retry happened

    def test_chunk_reduction_floor(self):
        """Chunk size should never go below MIN_EVENT_CHUNK_SIZE blocks."""
        import celo_scan

        mock_w3 = MagicMock()
        mock_w3.eth.block_number = celo_scan.EVENT_CHUNK_SIZE * 2
        mock_contract = MagicMock()
        self._passthrough_process_log(mock_contract)

        observed_sizes: list[int] = []

        def get_logs_side_effect(params):
            from_b = params["fromBlock"]
            to_b = params["toBlock"]
            observed_sizes.append(to_b - from_b + 1)
            raise Exception("413 Client Error: Request Entity Too Large")

        mock_w3.eth.get_logs.side_effect = get_logs_side_effect

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
        self._passthrough_process_log(mock_contract)

        sentinel = {"args": {"agentId": 1, "owner": "0xAAA"}, "blockNumber": 1}
        mock_w3.eth.get_logs.return_value = [sentinel]

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
        """Trigger endpoint should accept requests with valid key.

        The real scan_celo_agents is patched so the background executor
        thread completes instantly — otherwise it would connect to live
        forno.celo.org and scan millions of blocks, blocking pytest's
        executor teardown for its 300s timeout.
        """
        from api import INTERNAL_API_KEY
        with patch("celo_scan.scan_celo_agents", return_value=[]):
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


# ---------------------------------------------------------------------------
# Wallet ingestion + stale-first rotation tests
# ---------------------------------------------------------------------------

class TestIngestDiscoveredWallets:
    """_ingest_discovered_wallets_to_db must persist discoveries to
    known_wallets so they survive the next discovery run's checkpoint
    advance, without ever overwriting an existing scan history."""

    def _discovery(self, address: str, source: str, agent_id: int):
        return {
            "address": address,
            "source": source,
            "metadata": {"agent_id": agent_id},
        }

    def test_inserts_new_rows_with_null_last_scanned(self, tmp_path, monkeypatch):
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        wallets = [
            self._discovery("0x" + "a" * 40, "celo_agent_wallet", 1),
            self._discovery("0x" + "b" * 40, "celo_owner", 2),
        ]
        inserted = celo_scan._ingest_discovered_wallets_to_db(wallets)
        assert inserted == 2

        conn = db.get_connection()
        try:
            rows = conn.execute(
                "SELECT address, source, last_scanned_at, scan_count "
                "FROM known_wallets ORDER BY address"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 2
        # Both rows must carry NULL last_scanned_at so the stale-first query
        # picks them up before any rescan candidates on the next run.
        for r in rows:
            assert r["last_scanned_at"] is None
            assert r["scan_count"] == 0

    def test_existing_rows_are_not_overwritten(self, tmp_path, monkeypatch):
        """INSERT OR IGNORE must preserve an existing row's scan history."""
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        addr = "0x" + "c" * 40
        # Pre-populate with a wallet that already has a scan history.
        now_iso = "2026-04-10T12:00:00Z"
        conn = db.get_connection()
        try:
            conn.execute(
                """INSERT INTO known_wallets
                   (address, label, source, first_seen_at, last_scanned_at, scan_count,
                    latest_ahs, latest_grade)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (addr, "Existing label", "celo_agent_wallet",
                 "2026-01-01T00:00:00Z", now_iso, 5, 72, "B"),
            )
            conn.commit()
        finally:
            conn.close()

        inserted = celo_scan._ingest_discovered_wallets_to_db(
            [self._discovery(addr, "celo_agent_wallet", 42)]
        )
        assert inserted == 0

        conn = db.get_connection()
        try:
            row = conn.execute(
                "SELECT label, last_scanned_at, scan_count, latest_ahs "
                "FROM known_wallets WHERE address = ?", (addr,)
            ).fetchone()
        finally:
            conn.close()
        # Every mutable field must be untouched — no label overwrite,
        # no last_scanned_at reset, no scan_count reset, no AHS loss.
        assert row["label"] == "Existing label"
        assert row["last_scanned_at"] == now_iso
        assert row["scan_count"] == 5
        assert row["latest_ahs"] == 72

    def test_returns_zero_for_empty_list(self, tmp_path, monkeypatch):
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()
        assert celo_scan._ingest_discovered_wallets_to_db([]) == 0

    def test_mixed_new_and_existing_counts_only_new(self, tmp_path, monkeypatch):
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        existing = "0x" + "d" * 40
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO known_wallets (address, source, first_seen_at) "
                "VALUES (?, ?, ?)",
                (existing, "celo_agent_wallet", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
        finally:
            conn.close()

        inserted = celo_scan._ingest_discovered_wallets_to_db([
            self._discovery(existing, "celo_agent_wallet", 1),
            self._discovery("0x" + "e" * 40, "celo_agent_wallet", 2),
            self._discovery("0x" + "f" * 40, "celo_owner", 3),
        ])
        # Only the two genuinely-new addresses count as inserts.
        assert inserted == 2


class TestGetCeloScanCandidates:
    """get_celo_scan_candidates must return wallets stale-first so the
    nightly rotation reaches every agent over time instead of re-scoring
    the same top-agentId batch every night."""

    def _insert_wallet(self, conn, address, source, last_scanned_at):
        conn.execute(
            """INSERT INTO known_wallets
               (address, label, source, first_seen_at, last_scanned_at, scan_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (address, None, source, "2026-01-01T00:00:00Z", last_scanned_at, 0),
        )

    def test_null_last_scanned_at_comes_first(self, tmp_path, monkeypatch):
        """NULLS FIRST — never-scanned wallets outrank every rescan candidate."""
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            # Seed three rows: recent scan, old scan, never-scanned.
            self._insert_wallet(conn, "0x" + "1" * 40, "celo_agent_wallet",
                                "2026-04-15T00:00:00Z")
            self._insert_wallet(conn, "0x" + "2" * 40, "celo_agent_wallet",
                                "2026-01-01T00:00:00Z")
            self._insert_wallet(conn, "0x" + "3" * 40, "celo_owner", None)
            conn.commit()
        finally:
            conn.close()

        candidates = celo_scan.get_celo_scan_candidates(limit=10)
        addrs = [c["address"] for c in candidates]
        assert addrs[0] == "0x" + "3" * 40  # NULL first
        assert addrs[1] == "0x" + "2" * 40  # then oldest
        assert addrs[2] == "0x" + "1" * 40  # then newest

    def test_oldest_scanned_comes_before_newest(self, tmp_path, monkeypatch):
        """Among rescan candidates the oldest scan goes first so rotation is fair."""
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            self._insert_wallet(conn, "0xaaa" + "a" * 37, "celo_agent_wallet",
                                "2026-04-15T00:00:00Z")  # newest
            self._insert_wallet(conn, "0xbbb" + "b" * 37, "celo_agent_wallet",
                                "2026-03-01T00:00:00Z")  # middle
            self._insert_wallet(conn, "0xccc" + "c" * 37, "celo_agent_wallet",
                                "2026-02-01T00:00:00Z")  # oldest
            conn.commit()
        finally:
            conn.close()

        candidates = celo_scan.get_celo_scan_candidates(limit=10)
        addrs = [c["address"] for c in candidates]
        assert addrs == [
            "0xccc" + "c" * 37,
            "0xbbb" + "b" * 37,
            "0xaaa" + "a" * 37,
        ]

    def test_respects_limit(self, tmp_path, monkeypatch):
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            for i in range(10):
                self._insert_wallet(conn, f"0x{i:040x}", "celo_agent_wallet", None)
            conn.commit()
        finally:
            conn.close()

        candidates = celo_scan.get_celo_scan_candidates(limit=3)
        assert len(candidates) == 3

    def test_excludes_non_celo_sources(self, tmp_path, monkeypatch):
        """Arc, Olas and other registries must not bleed into the Celo rotation."""
        import celo_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            self._insert_wallet(conn, "0x" + "a" * 40, "celo_agent_wallet", None)
            self._insert_wallet(conn, "0x" + "b" * 40, "celo_owner",        None)
            self._insert_wallet(conn, "0x" + "c" * 40, "arc_agent_wallet",  None)
            self._insert_wallet(conn, "0x" + "d" * 40, "olas",              None)
            self._insert_wallet(conn, "0x" + "e" * 40, "acp_proactive_scan", None)
            conn.commit()
        finally:
            conn.close()

        candidates = celo_scan.get_celo_scan_candidates(limit=10)
        sources = {c["source"] for c in candidates}
        assert sources == {"celo_agent_wallet", "celo_owner"}
        assert len(candidates) == 2


class TestScanCeloAgentsRotation:
    """End-to-end: scan_celo_agents must feed the stale-first candidate
    list into the AHS scorer so every agent gets its turn over time."""

    def test_scans_in_stale_first_order(self, tmp_path, monkeypatch):
        """With three pre-existing wallets at different ages plus one new
        discovery, scan_celo_agents(max_scans=3) should score the never-
        scanned discovery first, then the two stalest known wallets."""
        import celo_scan
        import db
        import monitor

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        # Pre-populate: three previously-scanned wallets at distinct ages.
        conn = db.get_connection()
        try:
            for addr, ts in [
                ("0x" + "1" * 40, "2026-04-15T00:00:00Z"),  # fresh
                ("0x" + "2" * 40, "2026-02-01T00:00:00Z"),  # older
                ("0x" + "3" * 40, "2026-01-01T00:00:00Z"),  # oldest
            ]:
                conn.execute(
                    """INSERT INTO known_wallets
                       (address, label, source, first_seen_at, last_scanned_at, scan_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (addr, "existing", "celo_agent_wallet",
                     "2026-01-01T00:00:00Z", ts, 1),
                )
            conn.commit()
        finally:
            conn.close()

        # One new discovery (unseen by DB) — must land in the pool first.
        new_addr = "0x" + "9" * 40
        monkeypatch.setattr(
            celo_scan, "discover_celo_wallets",
            lambda max_agents=0: [{
                "address": new_addr,
                "source": "celo_agent_wallet",
                "metadata": {"agent_id": 999},
            }],
        )

        # Stub out the heavy monitor calls so the test doesn't hit the
        # network. Each call returns the same canned AHS result.
        ahs_result = monitor.AHSResult(
            address="0x0",
            agent_health_score=70,
            grade="B",
            grade_label="Good",
            confidence="MEDIUM",
            d1_score=65,
            d2_score=75,
            tx_count=42,
            history_days=30,
        )
        scanned_addresses: list[str] = []

        def fake_calculate_ahs(*, address, **_kwargs):
            scanned_addresses.append(address)
            ahs_result.address = address
            return ahs_result

        monkeypatch.setattr(monitor, "calculate_ahs", fake_calculate_ahs)
        monkeypatch.setattr(monitor, "fetch_transactions", lambda a, **k: [])
        monkeypatch.setattr(monitor, "fetch_tokens_v2", lambda a, **k: [])
        monkeypatch.setattr(monitor, "get_eth_price", lambda **k: 2500.0)
        # Don't actually sleep between scans in tests.
        monkeypatch.setattr(celo_scan.time, "sleep", lambda *_a, **_k: None)

        celo_scan.scan_celo_agents(max_scans=3)

        # Order of calls must be: NULL (new discovery) → oldest → middle.
        # "Fresh" wallet (0x111...) must NOT have been scored this run.
        assert scanned_addresses == [
            new_addr,
            "0x" + "3" * 40,
            "0x" + "2" * 40,
        ]

    def test_rotation_moves_forward_on_next_run(self, tmp_path, monkeypatch):
        """Two back-to-back runs with max_scans=2 on a 4-wallet pool must
        cover all four wallets across the two runs (no agent left behind)."""
        import celo_scan
        import db
        import monitor

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        addrs = [f"0x{i:040x}" for i in (1, 2, 3, 4)]

        # Seed all four with NULL last_scanned_at, distinct first_seen_at
        # so ordering among NULLs is deterministic via the secondary
        # tiebreak that SQLite uses on insertion order.
        conn = db.get_connection()
        try:
            for a in addrs:
                conn.execute(
                    """INSERT INTO known_wallets
                       (address, label, source, first_seen_at, last_scanned_at, scan_count)
                       VALUES (?, ?, ?, ?, NULL, 0)""",
                    (a, None, "celo_agent_wallet", "2026-01-01T00:00:00Z"),
                )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(
            celo_scan, "discover_celo_wallets", lambda max_agents=0: [],
        )

        ahs_result = monitor.AHSResult(
            address="0x0", agent_health_score=60, grade="C",
            grade_label="Needs Attention", confidence="LOW",
            d1_score=55, d2_score=65, tx_count=5, history_days=3,
        )
        scanned: list[str] = []

        def fake_calc(*, address, **_):
            scanned.append(address)
            ahs_result.address = address
            return ahs_result

        monkeypatch.setattr(monitor, "calculate_ahs", fake_calc)
        monkeypatch.setattr(monitor, "fetch_transactions", lambda a, **k: [])
        monkeypatch.setattr(monitor, "fetch_tokens_v2", lambda a, **k: [])
        monkeypatch.setattr(monitor, "get_eth_price", lambda **k: 2500.0)
        monkeypatch.setattr(celo_scan.time, "sleep", lambda *_a, **_k: None)

        # Run 1: picks the two wallets whose last_scanned_at is still NULL.
        celo_scan.scan_celo_agents(max_scans=2)
        first_run = list(scanned)
        scanned.clear()

        # Run 2: those two now have a last_scanned_at; the remaining two
        # are still NULL so they must lead the rotation.
        celo_scan.scan_celo_agents(max_scans=2)
        second_run = list(scanned)

        # Every wallet must have been covered exactly once across two runs.
        assert len(first_run) == 2
        assert len(second_run) == 2
        assert set(first_run + second_run) == set(addrs)
        # Second run must NOT pick the same wallets as the first.
        assert set(first_run).isdisjoint(set(second_run))
