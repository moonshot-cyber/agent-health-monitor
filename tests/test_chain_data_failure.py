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
    _patch(monkeypatch, resp=_Resp({"status": "0", "message": "No transactions found",
                                    "result": []}))
    assert monitor.fetch_transactions(ADDR) == []


def test_populated_wallet_returns_rows(monkeypatch):
    rows = [{"hash": "0x1"}, {"hash": "0x2"}]
    _patch(monkeypatch, resp=_Resp({"status": "1", "message": "OK", "result": rows}))
    assert monitor.fetch_transactions(ADDR) == rows


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
