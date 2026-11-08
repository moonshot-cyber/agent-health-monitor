"""An upstream failure must never be scored as a quiet wallet.

Between 7 and 27 October 2026 the Base chain data API went behind a bot
challenge. Every layer degraded silently: basescan_get swallowed the error and
returned {}, fetch_transactions read that as "no transactions" and returned [],
and calculate_d2_score read THAT as "insufficient history" and returned its 50
baseline. The composite landed on a confident 58 / Grade D for every wallet,
and ~19,000 of those were written to the database as fact — overwriting 141
real Grade A scores in the process.

These tests pin the distinction that was missing: failure raises, emptiness
does not.
"""

import pytest
import requests

import monitor


class _Resp:
    def __init__(self, payload=None, *, text=None, status_code=200, raise_http=False):
        self._payload, self._text = payload, text
        self.status_code, self._raise_http = status_code, raise_http

    @property
    def text(self):
        return self._text if self._text is not None else str(self._payload)

    def raise_for_status(self):
        if self._raise_http:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON object could be decoded")
        return self._payload


def _patch(monkeypatch, resp=None, exc=None):
    def fake_get(*_a, **_kw):
        if exc:
            raise exc
        return resp
    monkeypatch.setattr(monitor.requests, "get", fake_get)


ADDR = "0x" + "ab" * 20


# -- failures must raise -------------------------------------------------

def test_transport_error_raises(monkeypatch):
    _patch(monkeypatch, exc=requests.ConnectionError("dns failure"))
    with pytest.raises(monitor.ChainDataUnavailable):
        monitor.fetch_transactions(ADDR)


def test_http_error_raises(monkeypatch):
    _patch(monkeypatch, resp=_Resp({}, status_code=403, raise_http=True))
    with pytest.raises(monitor.ChainDataUnavailable):
        monitor.fetch_transactions(ADDR)


def test_bot_challenge_html_raises(monkeypatch):
    """HTTP 200 carrying an HTML interstitial — exactly how October presented.

    raise_for_status() passes because the status is 200; only the JSON decode
    reveals it. This is the case the original code could not see.
    """
    _patch(monkeypatch, resp=_Resp(None, text="<!DOCTYPE html><title>Just a moment...</title>"))
    with pytest.raises(monitor.ChainDataUnavailable) as e:
        monitor.fetch_transactions(ADDR)
    assert "non-JSON" in str(e.value)


def test_api_error_status_raises(monkeypatch):
    """status "0" with an error message is a failure, not an empty wallet."""
    _patch(monkeypatch, resp=_Resp({"status": "0", "message": "NOTOK",
                                    "result": "Max rate limit reached"}))
    with pytest.raises(monitor.ChainDataUnavailable):
        monitor.fetch_transactions(ADDR)


# -- genuine emptiness must NOT raise ------------------------------------

def test_genuinely_empty_wallet_returns_empty(monkeypatch):
    """The whole point: a quiet wallet is still scoreable."""
    _patch(monkeypatch, resp=_Resp({"items": [], "next_page_params": None}))
    assert monitor.fetch_transactions(ADDR) == []


def test_populated_wallet_returns_rows(monkeypatch):
    _patch(monkeypatch, resp=_Resp({"items": [_v2_tx(), _v2_tx(hash="0x" + "cd" * 32)],
                                    "next_page_params": None}))
    rows = monitor.fetch_transactions(ADDR)
    assert len(rows) == 2
    assert rows[0]["from"] == "0xsender"   # mapped, not raw v2


# -- the regression itself ------------------------------------------------

def test_outage_cannot_produce_a_grade_d_score(monkeypatch):
    """The October bug, pinned.

    Previously this produced D1=75 / D2=50 -> composite 58 -> Grade D, which
    was written to the database. It must now be impossible to get a score at
    all when the upstream is unavailable.
    """
    _patch(monkeypatch, resp=_Resp(None, text="<html>Just a moment...</html>"))
    with pytest.raises(monitor.ChainDataUnavailable):
        txs = monitor.fetch_transactions(ADDR)
        monitor.calculate_ahs(address=ADDR, tokens=[], transactions=txs, eth_price=2000.0)


def test_eth_price_still_degrades_gracefully(monkeypatch):
    """Price is not load-bearing for a score, so it keeps its fallbacks."""
    _patch(monkeypatch, resp=_Resp(None, text="<html>Just a moment...</html>"))
    assert monitor.get_eth_price() == monitor.ETH_PRICE_FALLBACK


# -- Blockscout v2 adapter -------------------------------------------------
#
# The per-instance etherscan-shaped API was deprecated on 7 October 2026. The
# replacement is a REST route with a different response shape, so the field
# mapping is now load-bearing: get it wrong and the scorers silently read
# defaults, which is the same failure this module exists to prevent.


def _v2_tx(**over):
    tx = {
        "hash": "0x" + "ab" * 32,
        "from": {"hash": "0xsender", "is_contract": False},
        "to": {"hash": "0xdest", "is_contract": True},
        "timestamp": "2026-10-15T10:26:51.474448Z",
        "gas_limit": "21000", "gas_used": "19500", "gas_price": "1000000",
        "nonce": 42, "status": "ok", "result": "success",
        "raw_input": "0xa9059cbb", "value": "1000", "block_number": 22566361,
    }
    tx.update(over)
    return tx


def test_v2_nested_addresses_are_flattened():
    out = monitor._v2_to_etherscan(_v2_tx())
    assert out["from"] == "0xsender"
    assert out["to"] == "0xdest"


def test_v2_iso_timestamp_becomes_unix_seconds():
    out = monitor._v2_to_etherscan(_v2_tx())
    assert out["timeStamp"].isdigit()
    assert int(out["timeStamp"]) > 1_700_000_000


def test_v2_status_maps_to_both_failure_flags():
    ok = monitor._v2_to_etherscan(_v2_tx())
    assert (ok["isError"], ok["txreceipt_status"]) == ("0", "1")
    bad = monitor._v2_to_etherscan(_v2_tx(status="error", result="out of gas"))
    assert (bad["isError"], bad["txreceipt_status"]) == ("1", "0")


def test_v2_contract_creation_has_no_to():
    out = monitor._v2_to_etherscan(
        _v2_tx(to=None, created_contract={"hash": "0xnew"}))
    assert out["to"] == ""
    assert out["contractAddress"] == "0xnew"


def test_v2_supplies_every_field_the_scorers_read():
    out = monitor._v2_to_etherscan(_v2_tx())
    for key in ("hash", "from", "to", "timeStamp", "gas", "gasUsed", "gasPrice",
                "nonce", "isError", "txreceipt_status", "input", "value"):
        assert key in out, key


def test_v2_data_produces_a_real_score_not_the_outage_default():
    """The regression that matters: v2 data must score like real data."""
    txs = []
    for i in range(40):
        txs.append(monitor._v2_to_etherscan(_v2_tx(
            hash=f"0x{i:064x}", nonce=i,
            # must match the scored address, or none of these count as outgoing
            **{"from": {"hash": "0xagent", "is_contract": False}},
            to={"hash": f"0xpeer{i % 5}", "is_contract": False},
            timestamp=f"2026-10-{(i % 27) + 1:02d}T0{i % 10}:00:00.000000Z",
            gas_limit=str(21000 + i * 100), gas_used=str(19000 + i * 90),
            status="error" if i % 11 == 0 else "ok")))
    r = monitor.calculate_ahs(address="0xagent", tokens=[], transactions=txs,
                              eth_price=2000.0)
    assert (r.agent_health_score, r.d1_score, r.d2_score) != (58, 75, 50)
    assert r.tx_count == 40


def test_402_without_a_key_raises_rather_than_returning_empty(monkeypatch):
    """A missing or rejected key must never read as 'no transactions'."""
    _patch(monkeypatch, resp=_Resp({"error": "Proceed with API key"},
                                   status_code=402))
    with pytest.raises(monitor.ChainDataUnavailable) as e:
        monitor.fetch_transactions(ADDR)
    assert "BLOCKSCOUT_API_KEY" in str(e.value)


def test_missing_items_array_raises(monkeypatch):
    _patch(monkeypatch, resp=_Resp({"unexpected": "shape"}))
    with pytest.raises(monitor.ChainDataUnavailable):
        monitor.fetch_transactions(ADDR)


def test_empty_items_is_a_real_answer(monkeypatch):
    _patch(monkeypatch, resp=_Resp({"items": [], "next_page_params": None}))
    assert monitor.fetch_transactions(ADDR) == []
