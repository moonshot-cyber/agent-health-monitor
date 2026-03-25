"""Tests for CDP pattern detection with both txlist and tokentx data sources."""

import time
from unittest.mock import patch

import pytest

from monitor import detect_cdp_patterns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_txlist_signals(**overrides):
    """Full 8-signal dict from normal txlist path."""
    signals = {
        "failed_pct_24h": 0,
        "d1_gas_eff_score": 50,
        "tx_diversity_ratio": 0.1,
        "unique_contracts": 5,
        "max_consecutive_failures": 0,
        "gas_adaptation_cv": 0.2,
        "persistent_nonce_gaps": 0,
        "gap_ratio": 1.0,
        "storm_events": 0,
        "gas_adaptation_score": 70,
        "d2_data_source": "txlist",
    }
    signals.update(overrides)
    return signals


def _base_tokentx_signals(**overrides):
    """4-signal dict from token transfer fallback, matching
    calculate_d2_score_from_transfers output."""
    signals = {
        # Available (real values from transfer analysis)
        "timing_score": 70,
        "timing_cv": 0.3,
        "gap_ratio": 2.0,
        "burst_count": 1,
        "tx_diversity_score": 60,
        "tx_diversity_ratio": 0.08,
        "unique_pairs": 5,
        "contract_breadth_score": 65,
        "breadth_ratio": 0.1,
        "unique_contracts": 4,
        "activity_gap_score": 75,
        "activity_gap_ratio": 1.5,
        # Unavailable (placeholders from fallback)
        "repeated_failure_score": None,
        "max_consecutive_failures": 0,
        "has_recovery": False,
        "gas_adaptation_score": None,
        "gas_adaptation_cv": 0.0,
        "nonce_management_score": None,
        "persistent_nonce_gaps": 0,
        "retry_storm_score": None,
        "storm_events": 0,
        # Source marker
        "d2_data_source": "tokentx",
        "d2_signals_used": 4,
    }
    signals.update(overrides)
    return signals


def _make_txs(count, days_back=3, fail_rate=0.0):
    """Generate mock transactions spread over `days_back` days."""
    now = int(time.time())
    interval = (days_back * 86400) // max(count, 1)
    txs = []
    for i in range(count):
        is_fail = (i / max(count, 1)) < fail_rate
        txs.append({
            "timeStamp": str(now - (count - i) * interval),
            "from": "0xagent",
            "to": f"0xcontract{i % 5}",
            "isError": "1" if is_fail else "0",
            "txreceipt_status": "0" if is_fail else "1",
            "nonce": str(i),
        })
    return txs


# ---------------------------------------------------------------------------
# tokentx path: patterns SHOULD fire
# ---------------------------------------------------------------------------

class TestTokentxZombieAgent:
    """Zombie Agent should fire on tokentx when diversity is near-zero."""

    def test_fires_single_counterparty(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.01,
            unique_contracts=1,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Zombie Agent" in names
        assert modifier == -15

    def test_does_not_fire_with_diversity(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.1,
            unique_contracts=5,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Zombie Agent" not in names


class TestTokentxHealthyOperator:
    """Healthy Operator should fire on tokentx with good D1 + diverse transfers."""

    def test_fires_healthy_wallet(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.1,
            unique_contracts=5,
        )
        modifier, patterns = detect_cdp_patterns(
            d1_score=85, d2_score=75, d3_score=None,
            signals=signals, transactions=[],
        )
        names = [p["name"] for p in patterns]
        assert "Healthy Operator" in names
        assert modifier == 5

    def test_does_not_fire_low_d1(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.1,
            unique_contracts=5,
        )
        modifier, patterns = detect_cdp_patterns(
            d1_score=60, d2_score=75, d3_score=None,
            signals=signals, transactions=[],
        )
        names = [p["name"] for p in patterns]
        assert "Healthy Operator" not in names

    def test_does_not_fire_low_d2(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.1,
            unique_contracts=5,
        )
        modifier, patterns = detect_cdp_patterns(
            d1_score=85, d2_score=50, d3_score=None,
            signals=signals, transactions=[],
        )
        names = [p["name"] for p in patterns]
        assert "Healthy Operator" not in names

    def test_does_not_fire_few_counterparties(self):
        signals = _base_tokentx_signals(
            tx_diversity_ratio=0.1,
            unique_contracts=2,
        )
        modifier, patterns = detect_cdp_patterns(
            d1_score=85, d2_score=75, d3_score=None,
            signals=signals, transactions=[],
        )
        names = [p["name"] for p in patterns]
        assert "Healthy Operator" not in names


# ---------------------------------------------------------------------------
# tokentx path: gas/nonce patterns must NOT fire
# ---------------------------------------------------------------------------

class TestTokentxSkippedPatterns:
    """Patterns requiring gas/nonce/failure data must not fire on tokentx,
    even if the placeholder values would accidentally match."""

    def test_cascading_infra_skipped(self):
        signals = _base_tokentx_signals(persistent_nonce_gaps=5, gap_ratio=20)
        txs = _make_txs(20, days_back=5, fail_rate=0.8)
        modifier, patterns = detect_cdp_patterns(30, 30, None, signals, txs)
        names = [p["name"] for p in patterns]
        assert "Cascading Infrastructure Failure" not in names

    def test_stale_strategy_skipped(self):
        signals = _base_tokentx_signals(
            max_consecutive_failures=10,
            tx_diversity_ratio=0.01,
            gas_adaptation_cv=0.01,
        )
        modifier, patterns = detect_cdp_patterns(30, 30, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Stale Strategy" not in names

    def test_gas_war_casualty_skipped(self):
        signals = _base_tokentx_signals(
            gas_adaptation_cv=0.5,
            storm_events=5,
            gas_adaptation_score=30,
        )
        signals["failed_pct_24h"] = 20
        modifier, patterns = detect_cdp_patterns(30, 30, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Gas War Casualty" not in names

    def test_recovery_skipped(self):
        signals = _base_tokentx_signals()
        txs = _make_txs(30, days_back=5, fail_rate=0.5)
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, txs)
        names = [p["name"] for p in patterns]
        assert "Recovery in Progress" not in names


# ---------------------------------------------------------------------------
# txlist path: patterns still work as before
# ---------------------------------------------------------------------------

class TestTxlistZombieAgent:
    def test_fires_full_criteria(self):
        signals = _base_txlist_signals(
            d1_gas_eff_score=95,
            failed_pct_24h=0,
            tx_diversity_ratio=0.01,
            unique_contracts=1,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Zombie Agent" in names

    def test_does_not_fire_without_gas_eff(self):
        signals = _base_txlist_signals(
            d1_gas_eff_score=50,
            failed_pct_24h=0,
            tx_diversity_ratio=0.01,
            unique_contracts=1,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Zombie Agent" not in names


class TestTxlistHealthyOperator:
    def test_fires_full_criteria(self):
        signals = _base_txlist_signals(
            gas_adaptation_cv=0.2,
            tx_diversity_ratio=0.1,
            storm_events=0,
        )
        modifier, patterns = detect_cdp_patterns(85, 75, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Healthy Operator" in names


class TestTxlistStaleStrategy:
    def test_fires(self):
        signals = _base_txlist_signals(
            max_consecutive_failures=8,
            tx_diversity_ratio=0.02,
            gas_adaptation_cv=0.03,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Stale Strategy" in names


class TestTxlistGasWarCasualty:
    def test_fires(self):
        signals = _base_txlist_signals(
            gas_adaptation_cv=0.5,
            failed_pct_24h=20,
            storm_events=3,
            gas_adaptation_score=40,
        )
        modifier, patterns = detect_cdp_patterns(30, 30, None, signals, [])
        names = [p["name"] for p in patterns]
        assert "Gas War Casualty" in names


# ---------------------------------------------------------------------------
# Modifier clamping
# ---------------------------------------------------------------------------

class TestModifierClamping:
    def test_clamped_to_negative_15(self):
        """Even if multiple negative patterns fire, modifier clamps to -15."""
        signals = _base_txlist_signals(
            d1_gas_eff_score=95,
            failed_pct_24h=0,
            tx_diversity_ratio=0.01,
            unique_contracts=1,
            max_consecutive_failures=8,
            gas_adaptation_cv=0.03,
        )
        modifier, patterns = detect_cdp_patterns(50, 50, None, signals, [])
        assert modifier >= -15

    def test_clamped_to_positive_5(self):
        signals = _base_txlist_signals(
            gas_adaptation_cv=0.2,
            tx_diversity_ratio=0.1,
            storm_events=0,
        )
        modifier, patterns = detect_cdp_patterns(85, 75, None, signals, [])
        assert modifier <= 5
