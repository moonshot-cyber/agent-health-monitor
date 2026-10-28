"""The recovery tool, proven against a database that reproduces the outage.

Production is a SQLite file on a Railway volume and cannot be reached from a
dev machine, so the tool is verified here against a synthetic database built to
match the real damage pattern: good scans before 7 October, flat 58 / tx_count 0
scans after, and a known_wallets cache overwritten by the latter.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "recover_outage_scans.py"

SCHEMA = """
CREATE TABLE scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL, endpoint TEXT NOT NULL, scan_timestamp TEXT NOT NULL,
    health_score REAL, risk_score INTEGER, ahs_score INTEGER,
    cleanliness_score INTEGER, grade TEXT, grade_label TEXT, confidence TEXT,
    mode TEXT, d1_score INTEGER, d2_score INTEGER, d3_score INTEGER,
    cdp_modifier INTEGER, response_json TEXT,
    source TEXT NOT NULL DEFAULT 'api', tx_count INTEGER, history_days INTEGER,
    shadow_signals_json TEXT, created_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE known_wallets (
    address TEXT PRIMARY KEY, label TEXT, source TEXT, first_seen_at TEXT NOT NULL,
    last_scanned_at TEXT, scan_count INTEGER NOT NULL DEFAULT 0,
    latest_ahs INTEGER, latest_grade TEXT, rescan_enabled INTEGER NOT NULL DEFAULT 1,
    rescan_interval_hours INTEGER NOT NULL DEFAULT 168,
    registries TEXT DEFAULT '', agent_name TEXT, latest_d1 INTEGER, latest_d2 INTEGER
);
"""

GOOD = [  # (address, ahs, grade, d1, d2, tx_count) scanned before the outage
    ("0xaaa1", 93, "A", 88, 95, 412),
    ("0xaaa2", 81, "B", 79, 82, 150),
    ("0xaaa3", 67, "C", 71, 65, 44),
    ("0xaaa4", 45, "D", 60, 39, 18),   # genuinely D, must stay D
]
ORPHAN = "0xbbb1"   # only ever scanned during the outage


def _poison(cur, addr, ts, source="celo"):
    """A scan produced from no data: the exact shape the outage wrote."""
    cur.execute(
        "INSERT INTO scans (address, endpoint, scan_timestamp, ahs_score, grade, "
        "grade_label, confidence, mode, d1_score, d2_score, source, tx_count) "
        "VALUES (?,'ahs',?,58,'D','Degraded','LOW','2D',75,50,?,0)",
        (addr, ts, source),
    )


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "ahm_history.db"
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    for addr, ahs, grade, d1, d2, txc in GOOD:
        cur.execute(
            "INSERT INTO scans (address, endpoint, scan_timestamp, ahs_score, grade, "
            "grade_label, confidence, mode, d1_score, d2_score, source, tx_count) "
            "VALUES (?,'ahs','2026-09-20T02:00:00Z',?,?,'x','HIGH','2D',?,?,'celo',?)",
            (addr, ahs, grade, d1, d2, txc),
        )
        # Then clobbered during the outage, as the rescan loop did.
        _poison(cur, addr, "2026-10-15T02:45:00Z")
        cur.execute(
            "INSERT INTO known_wallets (address, first_seen_at, last_scanned_at, "
            "scan_count, latest_ahs, latest_grade, latest_d1, latest_d2) "
            "VALUES (?,'2026-06-01T00:00:00Z','2026-10-15T02:45:00Z',2,58,'D',75,50)",
            (addr,),
        )

    _poison(cur, ORPHAN, "2026-10-16T02:45:00Z")
    cur.execute(
        "INSERT INTO known_wallets (address, first_seen_at, last_scanned_at, "
        "scan_count, latest_ahs, latest_grade, latest_d1, latest_d2) "
        "VALUES (?,'2026-10-16T02:45:00Z','2026-10-16T02:45:00Z',1,58,'D',75,50)",
        (ORPHAN,),
    )
    conn.commit()
    conn.close()
    return p


def run(db, *args):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db), *args],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def latest(db, addr):
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT ahs_score, grade FROM scans WHERE address=? AND endpoint='ahs' "
        "ORDER BY scan_timestamp DESC LIMIT 1", (addr,)).fetchone()
    cached = conn.execute(
        "SELECT latest_ahs, latest_grade FROM known_wallets WHERE address=?",
        (addr,)).fetchone()
    conn.close()
    return row, cached


def test_assess_is_read_only(db):
    before = db.read_bytes()
    out = run(db)
    assert "RECOVERABLE" in out
    assert db.read_bytes() == before, "assess must not modify the database"


def test_assess_counts_recoverable_and_orphans(db):
    out = run(db)
    assert "RECOVERABLE         : 4 wallets" in out
    assert "no prior good scan  : 1 wallets" in out


def test_apply_restores_the_real_scores(db):
    run(db, "--apply", "--no-backup")
    for addr, ahs, grade, *_ in GOOD:
        row, cached = latest(db, addr)
        assert row == (ahs, grade), f"{addr} scan not restored: {row}"
        assert cached == (ahs, grade), f"{addr} cache not rebuilt: {cached}"


def test_genuine_d_wallet_stays_d(db):
    """A wallet that really was Grade D must not be mistaken for poisoned."""
    run(db, "--apply", "--no-backup")
    row, _ = latest(db, "0xaaa4")
    assert row == (45, "D")


def test_orphan_has_no_surviving_scan(db):
    """Only ever scored during the outage, so there is nothing to restore."""
    run(db, "--apply", "--no-backup")
    row, _ = latest(db, ORPHAN)
    assert row is None


def test_apply_is_reversible(db):
    run(db, "--apply", "--no-backup")
    assert latest(db, "0xaaa1")[0] == (93, "A")
    run(db, "--rollback")
    # The poisoned scan is back and is latest again.
    assert latest(db, "0xaaa1")[0] == (58, "D")


def test_poisoned_rows_are_quarantined_not_deleted(db):
    run(db, "--apply", "--no-backup")
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM scans_quarantine").fetchone()[0]
    reason = conn.execute(
        "SELECT DISTINCT quarantine_reason FROM scans_quarantine").fetchone()[0]
    conn.close()
    assert n == 5, "4 clobbered + 1 orphan"
    assert "outage" in reason
