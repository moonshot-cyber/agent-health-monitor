"""Tests for shared URI resolution and agent name extraction.

Covers: resolve_uri success + failure modes, extract_name_from_json key
priority, resolve_agent_names batch helper (Celo/Arc wallet-dict format),
and Celo ingestion persistence of agent_name.
"""

import base64
import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# resolve_uri — low-level resolver
# ---------------------------------------------------------------------------

class TestResolveUri:
    """Unit tests for uri_utils.resolve_uri."""

    def test_http_success(self):
        from uri_utils import resolve_uri

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "TestAgent"}
        mock_resp.raise_for_status = MagicMock()
        with patch("uri_utils.requests.get", return_value=mock_resp):
            data, err = resolve_uri("https://example.com/agent.json")
        assert err == ""
        assert data == {"name": "TestAgent"}

    def test_empty_uri(self):
        from uri_utils import resolve_uri

        data, err = resolve_uri("")
        assert data is None
        assert err == "empty"

    def test_none_uri(self):
        from uri_utils import resolve_uri

        data, err = resolve_uri(None)
        assert data is None
        assert err == "empty"

    def test_timeout(self):
        import requests as req
        from uri_utils import resolve_uri

        with patch("uri_utils.requests.get", side_effect=req.Timeout()):
            data, err = resolve_uri("https://example.com/slow")
        assert data is None
        assert err == "timeout"

    def test_invalid_json(self):
        from uri_utils import resolve_uri

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("", "", 0)
        with patch("uri_utils.requests.get", return_value=mock_resp):
            data, err = resolve_uri("https://example.com/not-json")
        assert data is None
        assert err == "invalid_json"

    def test_unsupported_scheme(self):
        from uri_utils import resolve_uri

        data, err = resolve_uri("ftp://example.com/agent")
        assert data is None
        assert "unsupported_scheme" in err

    def test_data_uri(self):
        from uri_utils import resolve_uri

        payload = json.dumps({"name": "DataAgent"})
        encoded = base64.b64encode(payload.encode()).decode()
        data, err = resolve_uri(f"data:application/json;base64,{encoded}")
        assert err == ""
        assert data["name"] == "DataAgent"

    def test_ipfs_uri(self):
        from uri_utils import resolve_uri

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "IPFSAgent"}
        mock_resp.raise_for_status = MagicMock()
        with patch("uri_utils.requests.get", return_value=mock_resp) as mock_get:
            data, err = resolve_uri("ipfs://QmTest123")
        assert err == ""
        assert data["name"] == "IPFSAgent"
        call_url = mock_get.call_args[0][0]
        assert "ipfs.io/ipfs/QmTest123" in call_url

    def test_connection_error(self):
        import requests as req
        from uri_utils import resolve_uri

        with patch("uri_utils.requests.get", side_effect=req.ConnectionError()):
            data, err = resolve_uri("https://unreachable.example.com")
        assert data is None
        assert err == "connection_error"

    def test_http_error(self):
        import requests as req
        from uri_utils import resolve_uri

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_err = req.HTTPError(response=mock_resp)
        with patch("uri_utils.requests.get", side_effect=http_err):
            data, err = resolve_uri("https://example.com/missing")
        assert data is None
        assert err == "http_404"


# ---------------------------------------------------------------------------
# extract_name_from_json — name key priority
# ---------------------------------------------------------------------------

class TestExtractNameFromJson:
    """Unit tests for uri_utils.extract_name_from_json."""

    def test_name_key(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"name": "AgentFoo"}) == "AgentFoo"

    def test_title_key(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"title": "AgentBar"}) == "AgentBar"

    def test_agent_name_key(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"agent_name": "AgentBaz"}) == "AgentBaz"

    def test_agentName_key(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"agentName": "AgentQux"}) == "AgentQux"

    def test_priority_order(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"name": "First", "title": "Second"}) == "First"

    def test_empty_name_falls_through(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"name": "", "title": "Fallback"}) == "Fallback"

    def test_no_name_returns_empty(self):
        from uri_utils import extract_name_from_json
        assert extract_name_from_json({"foo": "bar"}) == ""

    def test_truncates_long_names(self):
        from uri_utils import extract_name_from_json
        long_name = "A" * 200
        result = extract_name_from_json({"name": long_name})
        assert len(result) == 80


# ---------------------------------------------------------------------------
# resolve_agent_names — batch helper (Celo / Arc wallet-dict format)
# ---------------------------------------------------------------------------

class TestResolveAgentNames:
    """Tests for the batch resolve_agent_names helper."""

    def test_successful_resolution(self):
        from uri_utils import resolve_agent_names

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "ResolvedAgent"}
        mock_resp.raise_for_status = MagicMock()
        wallets = [
            {"address": "0x" + "aa" * 20, "source": "celo_agent_wallet",
             "metadata": {"agent_id": 1, "agent_uri": "https://example.com/agent.json"}},
        ]
        with patch("uri_utils.requests.get", return_value=mock_resp), \
             patch("uri_utils.time.sleep"):
            errors = resolve_agent_names(wallets, delay=0)
        assert wallets[0]["metadata"]["agent_name"] == "ResolvedAgent"
        assert not errors

    def test_no_uri_tracked(self):
        from uri_utils import resolve_agent_names

        wallets = [
            {"address": "0x" + "bb" * 20, "source": "celo_owner",
             "metadata": {"agent_id": 2, "agent_uri": ""}},
        ]
        with patch("uri_utils.time.sleep"):
            errors = resolve_agent_names(wallets, delay=0)
        assert errors.get("no_uri") == 1
        assert "agent_name" not in wallets[0].get("metadata", {})

    def test_timeout_tracked(self):
        import requests as req
        from uri_utils import resolve_agent_names

        wallets = [
            {"address": "0x" + "cc" * 20, "source": "arc_agent_wallet",
             "metadata": {"agent_id": 3, "agent_uri": "https://slow.example.com/agent"}},
        ]
        with patch("uri_utils.requests.get", side_effect=req.Timeout()), \
             patch("uri_utils.time.sleep"):
            errors = resolve_agent_names(wallets, delay=0)
        assert errors.get("timeout") == 1
        assert "agent_name" not in wallets[0].get("metadata", {})

    def test_invalid_json_tracked(self):
        from uri_utils import resolve_agent_names

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("", "", 0)
        wallets = [
            {"address": "0x" + "dd" * 20, "source": "arc_owner",
             "metadata": {"agent_id": 4, "agent_uri": "https://example.com/bad"}},
        ]
        with patch("uri_utils.requests.get", return_value=mock_resp), \
             patch("uri_utils.time.sleep"):
            errors = resolve_agent_names(wallets, delay=0)
        assert errors.get("invalid_json") == 1

    def test_scan_completes_despite_errors(self):
        """All wallets are processed even when some fail."""
        import requests as req
        from uri_utils import resolve_agent_names

        mock_resp_ok = MagicMock()
        mock_resp_ok.json.return_value = {"name": "GoodAgent"}
        mock_resp_ok.raise_for_status = MagicMock()

        wallets = [
            {"address": "0x01", "source": "celo_owner",
             "metadata": {"agent_id": 1, "agent_uri": "https://bad.example.com"}},
            {"address": "0x02", "source": "celo_owner",
             "metadata": {"agent_id": 2, "agent_uri": "https://good.example.com"}},
        ]

        def side_effect(url, **kwargs):
            if "bad" in url:
                raise req.Timeout()
            return mock_resp_ok

        with patch("uri_utils.requests.get", side_effect=side_effect), \
             patch("uri_utils.time.sleep"):
            errors = resolve_agent_names(wallets, delay=0)

        assert errors.get("timeout") == 1
        assert "agent_name" not in wallets[0].get("metadata", {})
        assert wallets[1]["metadata"]["agent_name"] == "GoodAgent"

    def test_uri_cache_avoids_duplicate_fetches(self):
        """Same URI resolved only once even when shared by multiple wallets."""
        from uri_utils import resolve_agent_names

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"name": "SharedAgent"}
        mock_resp.raise_for_status = MagicMock()
        shared_uri = "https://example.com/shared.json"
        wallets = [
            {"address": "0x01", "source": "celo_owner",
             "metadata": {"agent_id": 1, "agent_uri": shared_uri}},
            {"address": "0x02", "source": "celo_owner",
             "metadata": {"agent_id": 2, "agent_uri": shared_uri}},
        ]
        with patch("uri_utils.requests.get", return_value=mock_resp) as mock_get, \
             patch("uri_utils.time.sleep"):
            resolve_agent_names(wallets, delay=0)
        assert mock_get.call_count == 1
        assert wallets[0]["metadata"]["agent_name"] == "SharedAgent"
        assert wallets[1]["metadata"]["agent_name"] == "SharedAgent"


# ---------------------------------------------------------------------------
# Celo ingestion — agent_name persisted to known_wallets
# ---------------------------------------------------------------------------

class TestCeloIngestionAgentName:
    """Test that Celo's _ingest_discovered_wallets_to_db writes agent_name."""

    def test_celo_ingestion_persists_agent_name(self, tmp_path):
        import db as _db

        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_celo_ingest.db")
        try:
            _db.init_db()
            from celo_scan import _ingest_discovered_wallets_to_db

            wallets = [{
                "address": "0x" + "aa" * 20,
                "source": "celo_agent_wallet",
                "metadata": {"agent_id": 42, "type": "agent_wallet", "agent_name": "CeloBot"},
            }]
            inserted = _ingest_discovered_wallets_to_db(wallets)
            assert inserted == 1

            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "aa" * 20,),
                ).fetchone()
                assert row["agent_name"] == "CeloBot"
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path

    def test_celo_ingestion_no_name_leaves_null(self, tmp_path):
        import db as _db

        old_path = _db.DB_PATH
        _db.DB_PATH = str(tmp_path / "test_celo_null.db")
        try:
            _db.init_db()
            from celo_scan import _ingest_discovered_wallets_to_db

            wallets = [{
                "address": "0x" + "bb" * 20,
                "source": "celo_owner",
                "metadata": {"agent_id": 99, "type": "owner"},
            }]
            inserted = _ingest_discovered_wallets_to_db(wallets)
            assert inserted == 1

            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT agent_name FROM known_wallets WHERE address = ?",
                    ("0x" + "bb" * 20,),
                ).fetchone()
                assert row["agent_name"] is None
            finally:
                conn.close()
        finally:
            _db.DB_PATH = old_path
