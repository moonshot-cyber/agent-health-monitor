"""Scan history persistence layer for Agent Health Monitor.

SQLite-backed append-only scan log. All database logic lives here
so a future PostgreSQL migration only touches this file.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

logger = logging.getLogger("ahm.db")

DB_PATH = os.getenv("DB_PATH", "./ahm_history.db")

_SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    address           TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    scan_timestamp    TEXT NOT NULL,
    health_score      REAL,
    risk_score        INTEGER,
    ahs_score         INTEGER,
    cleanliness_score INTEGER,
    grade             TEXT,
    grade_label       TEXT,
    confidence        TEXT,
    mode              TEXT,
    d1_score          INTEGER,
    d2_score          INTEGER,
    d3_score          INTEGER,
    cdp_modifier      INTEGER,
    response_json     TEXT,
    source            TEXT NOT NULL DEFAULT 'api',
    tx_count          INTEGER,
    history_days      INTEGER,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS scan_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id         INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    pattern_name    TEXT NOT NULL,
    severity        TEXT NOT NULL,
    description     TEXT,
    modifier        INTEGER
);

CREATE TABLE IF NOT EXISTS known_wallets (
    address               TEXT PRIMARY KEY,
    label                 TEXT,
    source                TEXT,
    first_seen_at         TEXT NOT NULL,
    last_scanned_at       TEXT,
    scan_count            INTEGER NOT NULL DEFAULT 0,
    latest_ahs            INTEGER,
    latest_grade          TEXT,
    rescan_enabled        INTEGER NOT NULL DEFAULT 1,
    rescan_interval_hours INTEGER NOT NULL DEFAULT 168
);

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_scans_address_endpoint ON scans(address, endpoint, scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scans_address_time     ON scans(address, scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scans_grade            ON scans(grade);
CREATE INDEX IF NOT EXISTS idx_patterns_scan_id       ON scan_patterns(scan_id);
CREATE INDEX IF NOT EXISTS idx_patterns_name          ON scan_patterns(pattern_name);
CREATE INDEX IF NOT EXISTS idx_wallets_rescan         ON known_wallets(rescan_enabled, last_scanned_at);
"""


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and indexes if they don't exist. Runs on app startup."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        # Track schema version
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] if row[0] is not None else 0

        # v2: Add registries column for cross-registry tracking
        if current < 2:
            try:
                conn.execute("ALTER TABLE known_wallets ADD COLUMN registries TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.execute(
                "UPDATE known_wallets SET registries = source "
                "WHERE (registries IS NULL OR registries = '') AND source IS NOT NULL"
            )

        if current < _SCHEMA_VERSION:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
            conn.commit()
        logger.info("Database initialized at %s (schema v%d)", DB_PATH, _SCHEMA_VERSION)
    finally:
        conn.close()


def log_scan(
    address: str,
    endpoint: str,
    scan_timestamp: str,
    response_data: dict | None = None,
    source: str = "api",
    health_score: float | None = None,
    risk_score: int | None = None,
    ahs_score: int | None = None,
    cleanliness_score: int | None = None,
    grade: str | None = None,
    grade_label: str | None = None,
    confidence: str | None = None,
    mode: str | None = None,
    d1_score: int | None = None,
    d2_score: int | None = None,
    d3_score: int | None = None,
    cdp_modifier: int | None = None,
    patterns: list[dict] | None = None,
    tx_count: int | None = None,
    history_days: int | None = None,
    label: str | None = None,
):
    """Insert a scan record and any detected patterns. Thread-safe."""
    addr = address.lower()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Strip Nansen-specific data from response before storing
    response_json = None
    if response_data:
        safe = {k: v for k, v in response_data.items()
                if k not in ("nansen_labels", "nansen_counterparties", "nansen_pnl")}
        response_json = json.dumps(safe, default=str)

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO scans (
                address, endpoint, scan_timestamp, health_score, risk_score,
                ahs_score, cleanliness_score, grade, grade_label, confidence,
                mode, d1_score, d2_score, d3_score, cdp_modifier,
                response_json, source, tx_count, history_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (addr, endpoint, scan_timestamp, health_score, risk_score,
             ahs_score, cleanliness_score, grade, grade_label, confidence,
             mode, d1_score, d2_score, d3_score, cdp_modifier,
             response_json, source, tx_count, history_days),
        )
        scan_id = cur.lastrowid

        if patterns:
            for p in patterns:
                conn.execute(
                    """INSERT INTO scan_patterns (scan_id, pattern_name, severity, description, modifier)
                    VALUES (?, ?, ?, ?, ?)""",
                    (scan_id, p.get("name", ""), p.get("severity", ""),
                     p.get("description", ""), p.get("modifier")),
                )

        # Upsert known_wallets
        rescan_interval = 168  # default weekly
        if grade in ("A", "B"):
            rescan_interval = 720  # monthly
        elif grade == "C":
            rescan_interval = 336  # biweekly

        # Map source to registry name for cross-registry tracking
        registry = source
        if registry == "api":
            registry = "ahm_api"

        conn.execute(
            """INSERT INTO known_wallets (address, label, source, first_seen_at, last_scanned_at,
                scan_count, latest_ahs, latest_grade, rescan_interval_hours, registries)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                last_scanned_at = excluded.last_scanned_at,
                scan_count = known_wallets.scan_count + 1,
                latest_ahs = COALESCE(excluded.latest_ahs, known_wallets.latest_ahs),
                latest_grade = COALESCE(excluded.latest_grade, known_wallets.latest_grade),
                label = COALESCE(excluded.label, known_wallets.label),
                rescan_interval_hours = excluded.rescan_interval_hours,
                registries = CASE
                    WHEN known_wallets.registries IS NULL OR known_wallets.registries = ''
                        THEN excluded.registries
                    WHEN instr(known_wallets.registries, excluded.registries) > 0
                        THEN known_wallets.registries
                    ELSE known_wallets.registries || ',' || excluded.registries
                END""",
            (addr, label, source, now_iso, now_iso, ahs_score, grade, rescan_interval, registry),
        )

        conn.commit()
    except Exception:
        logger.exception("Failed to log scan for %s/%s", endpoint, addr)
    finally:
        conn.close()


def get_wallets_due_for_rescan(limit: int = 50) -> list[dict]:
    """Return wallets whose rescan interval has elapsed."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT address, label, rescan_interval_hours, last_scanned_at
            FROM known_wallets
            WHERE rescan_enabled = 1
            AND (
                last_scanned_at IS NULL
                OR datetime(last_scanned_at, '+' || rescan_interval_hours || ' hours') < datetime('now')
            )
            ORDER BY last_scanned_at ASC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trust_registry_stats() -> dict:
    """Aggregate stats for the trust registry endpoint."""
    conn = get_connection()
    try:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Summary
        row = conn.execute("SELECT COUNT(DISTINCT address), COUNT(*) FROM scans").fetchone()
        total_addresses = row[0]
        total_scans = row[1]

        row = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE endpoint = 'ahs'"
        ).fetchone()
        total_ahs_scans = row[0]

        row = conn.execute(
            "SELECT MIN(scan_timestamp), MAX(scan_timestamp) FROM scans"
        ).fetchone()
        first_scan = row[0]
        last_scan = row[1]

        # Grade distribution (latest AHS per address)
        grade_rows = conn.execute(
            """SELECT grade, COUNT(*) as cnt FROM (
                SELECT address, grade,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND grade IS NOT NULL
            ) WHERE rn = 1 GROUP BY grade ORDER BY grade"""
        ).fetchall()
        grade_distribution = {r["grade"]: r["cnt"] for r in grade_rows}

        # Average AHS (latest per address)
        avg_row = conn.execute(
            """SELECT AVG(ahs_score) FROM (
                SELECT address, ahs_score,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            ) WHERE rn = 1"""
        ).fetchone()
        average_ahs = round(avg_row[0], 1) if avg_row[0] is not None else None

        # Pattern frequency
        pattern_rows = conn.execute(
            "SELECT pattern_name, COUNT(*) as cnt FROM scan_patterns GROUP BY pattern_name ORDER BY cnt DESC"
        ).fetchall()
        pattern_frequency = {r["pattern_name"]: r["cnt"] for r in pattern_rows}

        # Top scanned wallets
        top_rows = conn.execute(
            """SELECT address, label, scan_count, latest_ahs, latest_grade, last_scanned_at
            FROM known_wallets ORDER BY scan_count DESC LIMIT 20"""
        ).fetchall()
        top_wallets = [dict(r) for r in top_rows]

        # 24h scan volume by endpoint
        volume_rows = conn.execute(
            """SELECT endpoint, COUNT(*) as cnt FROM scans
            WHERE scan_timestamp > datetime('now', '-1 day')
            GROUP BY endpoint ORDER BY cnt DESC"""
        ).fetchall()
        scan_volume_24h = {r["endpoint"]: r["cnt"] for r in volume_rows}

        # Baseline calibration — score percentiles
        scores = conn.execute(
            """SELECT ahs_score FROM (
                SELECT address, ahs_score,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            ) WHERE rn = 1 ORDER BY ahs_score"""
        ).fetchall()
        score_list = [r[0] for r in scores]
        percentiles = {}
        if score_list:
            n = len(score_list)
            for p in (10, 25, 50, 75, 90):
                idx = min(int(n * p / 100), n - 1)
                percentiles[f"p{p}"] = score_list[idx]

        # Grade boundary stats
        grade_boundaries = {}
        boundaries = [("A", 90, 100), ("B", 75, 89), ("C", 60, 74),
                      ("D", 40, 59), ("E", 20, 39), ("F", 0, 19)]
        total_graded = sum(grade_distribution.values()) if grade_distribution else 0
        for g, lo, hi in boundaries:
            cnt = grade_distribution.get(g, 0)
            grade_boundaries[g] = {
                "min": lo, "max": hi, "count": cnt,
                "pct": round(cnt / total_graded * 100, 1) if total_graded > 0 else 0,
            }

        return {
            "status": "ok",
            "generated_at": now_iso,
            "summary": {
                "total_unique_addresses": total_addresses,
                "total_scans": total_scans,
                "total_ahs_scans": total_ahs_scans,
                "date_range": {"first_scan": first_scan, "last_scan": last_scan},
            },
            "grade_distribution": grade_distribution,
            "average_ahs": average_ahs,
            "pattern_frequency": pattern_frequency,
            "top_scanned_wallets": top_wallets,
            "scan_volume_24h": scan_volume_24h,
            "baseline_calibration": {
                "grade_boundaries": grade_boundaries,
                "score_percentiles": percentiles,
            },
        }
    finally:
        conn.close()


def forget_address(address: str) -> int:
    """Delete all data for an address. Returns number of scans deleted."""
    addr = address.lower()
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM scan_patterns WHERE scan_id IN (SELECT id FROM scans WHERE address = ?)",
            (addr,),
        )
        cur = conn.execute("DELETE FROM scans WHERE address = ?", (addr,))
        deleted = cur.rowcount
        conn.execute("DELETE FROM known_wallets WHERE address = ?", (addr,))
        conn.commit()
        return deleted
    finally:
        conn.close()
