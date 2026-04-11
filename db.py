"""Scan history persistence layer for Agent Health Monitor.

SQLite-backed append-only scan log. All database logic lives here
so a future PostgreSQL migration only touches this file.
"""

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("ahm.db")

DB_PATH = os.getenv("DB_PATH", "./ahm_history.db")

_SCHEMA_VERSION = 8

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
    shadow_signals_json TEXT,
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

CREATE TABLE IF NOT EXISTS scan_batch_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_date         TEXT NOT NULL,
    source             TEXT NOT NULL,
    wallets_scanned    INTEGER NOT NULL,
    average_ahs        REAL,
    min_ahs            INTEGER,
    max_ahs            INTEGER,
    grade_distribution TEXT,
    avg_d1             REAL,
    avg_d2             REAL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS api_keys (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash                TEXT UNIQUE NOT NULL,
    key_prefix              TEXT NOT NULL,
    customer_email          TEXT NOT NULL,
    stripe_customer_id      TEXT,
    stripe_subscription_id  TEXT,
    type                    TEXT NOT NULL,
    tier                    TEXT NOT NULL,
    calls_remaining         INTEGER,
    calls_total             INTEGER,
    created_at              TEXT NOT NULL,
    expires_at              TEXT,
    is_active               INTEGER DEFAULT 1,
    partner_id              TEXT,
    is_reseller             INTEGER DEFAULT 0,
    wholesale_rate          REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS api_key_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash        TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    called_at       TEXT NOT NULL,
    wallet_queried  TEXT,
    partner_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_key_usage_hash ON api_key_usage(key_hash, called_at DESC);

CREATE TABLE IF NOT EXISTS security_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    ip_address  TEXT NOT NULL,
    details     TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_security_events_ip   ON security_events(ip_address, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS shield_subscriptions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_hash            TEXT NOT NULL,
    stripe_customer_id      TEXT,
    stripe_subscription_id  TEXT UNIQUE,
    tier                    TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'active',
    agent_slots             INTEGER NOT NULL,
    call_quota              INTEGER NOT NULL,
    calls_used_this_period  INTEGER NOT NULL DEFAULT 0,
    period_start            TEXT NOT NULL,
    period_end              TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_shield_subs_key ON shield_subscriptions(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_shield_subs_stripe ON shield_subscriptions(stripe_subscription_id);
CREATE INDEX IF NOT EXISTS idx_shield_subs_status ON shield_subscriptions(status);
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

        # v3: Add scan_batch_log table (created via _SCHEMA_SQL above)
        # No ALTER needed — table is created by CREATE TABLE IF NOT EXISTS.

        # v4: Add api_keys and api_key_usage tables (created via _SCHEMA_SQL above)
        # No ALTER needed — tables are created by CREATE TABLE IF NOT EXISTS.

        # v5: Add security_events table (created via _SCHEMA_SQL above)
        # No ALTER needed — table is created by CREATE TABLE IF NOT EXISTS.

        # v6: Add partner fields to api_keys and api_key_usage
        if current < 6:
            for col, default in [
                ("partner_id", None),
                ("is_reseller", 0),
                ("wholesale_rate", 0.5),
            ]:
                try:
                    if default is None:
                        conn.execute(f"ALTER TABLE api_keys ADD COLUMN {col} TEXT")
                    elif isinstance(default, float):
                        conn.execute(f"ALTER TABLE api_keys ADD COLUMN {col} REAL DEFAULT {default}")
                    else:
                        conn.execute(f"ALTER TABLE api_keys ADD COLUMN {col} INTEGER DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            try:
                conn.execute("ALTER TABLE api_key_usage ADD COLUMN partner_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Create indexes on new columns (safe after columns exist)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_partner ON api_keys(partner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_key_usage_partner ON api_key_usage(partner_id, called_at DESC)")

        # v7: Add shield_subscriptions table (created via _SCHEMA_SQL above)
        # No ALTER needed — table is created by CREATE TABLE IF NOT EXISTS.

        # v8: Add shadow_signals_json column to scans for D2 shadow-mode persistence.
        # See ahm_backlog.md "Session Continuity Shadow Mode Review" — without this
        # column, shadow signals computed in monitor.py are returned in the API
        # response but never written to the DB, so distribution analysis is impossible.
        if current < 8:
            try:
                conn.execute("ALTER TABLE scans ADD COLUMN shadow_signals_json TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

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
    shadow_signals: dict | None = None,
):
    """Insert a scan record and any detected patterns. Thread-safe.

    The optional shadow_signals dict is JSON-encoded and stored verbatim in
    the scans.shadow_signals_json column. Used for D2 session-continuity
    shadow-mode burn-in (see ahm_backlog.md).
    """
    addr = address.lower()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Strip Nansen-specific data from response before storing
    response_json = None
    if response_data:
        safe = {k: v for k, v in response_data.items()
                if k not in ("nansen_labels", "nansen_counterparties", "nansen_pnl")}
        response_json = json.dumps(safe, default=str)

    shadow_json = json.dumps(shadow_signals, default=str) if shadow_signals else None

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO scans (
                address, endpoint, scan_timestamp, health_score, risk_score,
                ahs_score, cleanliness_score, grade, grade_label, confidence,
                mode, d1_score, d2_score, d3_score, cdp_modifier,
                response_json, source, tx_count, history_days, shadow_signals_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (addr, endpoint, scan_timestamp, health_score, risk_score,
             ahs_score, cleanliness_score, grade, grade_label, confidence,
             mode, d1_score, d2_score, d3_score, cdp_modifier,
             response_json, source, tx_count, history_days, shadow_json),
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


def get_latest_ahs_for_address(address: str) -> dict | None:
    """Return the latest AHS score summary for a single address from known_wallets.

    Returns dict with keys: address, latest_ahs, latest_grade, confidence,
    last_scanned_at — or None if the address has never been scored.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT w.address, w.latest_ahs, w.latest_grade, w.last_scanned_at,
                      s.confidence
               FROM known_wallets w
               LEFT JOIN (
                   SELECT address, confidence,
                       ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                   FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
               ) s ON s.address = w.address AND s.rn = 1
               WHERE w.address = ? AND w.latest_ahs IS NOT NULL""",
            (address.lower(),),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
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


def get_ecosystem_dashboard_stats() -> dict:
    """Lightweight aggregate stats for the public dashboard page."""
    conn = get_connection()
    try:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Total unique agents scanned (AHS endpoint only — real scores)
        row = conn.execute(
            "SELECT COUNT(DISTINCT address) FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL"
        ).fetchone()
        total_scanned = row[0]

        # Average AHS and dimension scores (latest per address)
        avg_row = conn.execute(
            """SELECT AVG(ahs_score), AVG(d1_score), AVG(d2_score) FROM (
                SELECT address, ahs_score, d1_score, d2_score,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            ) WHERE rn = 1"""
        ).fetchone()
        avg_ahs = round(avg_row[0], 1) if avg_row[0] is not None else 0
        avg_d1 = round(avg_row[1], 1) if avg_row[1] is not None else 0
        avg_d2 = round(avg_row[2], 1) if avg_row[2] is not None else 0

        # Last scan timestamp
        last_row = conn.execute(
            "SELECT MAX(scan_timestamp) FROM scans WHERE endpoint = 'ahs'"
        ).fetchone()
        last_updated = last_row[0] if last_row[0] else now_iso

        # Grade distribution (latest per address)
        grade_rows = conn.execute(
            """SELECT grade, COUNT(*) as cnt FROM (
                SELECT address, grade,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND grade IS NOT NULL
            ) WHERE rn = 1 GROUP BY grade ORDER BY grade"""
        ).fetchall()
        grade_distribution = {r["grade"]: r["cnt"] for r in grade_rows}

        # Pattern distribution (percentage-based from latest scan per address)
        pattern_rows = conn.execute(
            """SELECT sp.pattern_name, COUNT(DISTINCT latest.address) as cnt FROM (
                SELECT id, address,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND ahs_score IS NOT NULL
            ) latest
            JOIN scan_patterns sp ON sp.scan_id = latest.id
            WHERE latest.rn = 1
            GROUP BY sp.pattern_name
            ORDER BY cnt DESC"""
        ).fetchall()
        pattern_distribution = {r["pattern_name"]: r["cnt"] for r in pattern_rows}

        # Data sources breakdown from known_wallets.source
        source_rows = conn.execute(
            """SELECT source, COUNT(*) as cnt FROM known_wallets
            WHERE latest_ahs IS NOT NULL
            GROUP BY source ORDER BY cnt DESC"""
        ).fetchall()
        data_sources = {}
        for r in source_rows:
            key = r["source"]
            if key and "acp" in key.lower():
                data_sources["ACP"] = data_sources.get("ACP", 0) + r["cnt"]
            elif key and "olas" in key.lower():
                data_sources["Olas"] = data_sources.get("Olas", 0) + r["cnt"]
            elif key and "celo" in key.lower():
                data_sources["Celo"] = data_sources.get("Celo", 0) + r["cnt"]
            elif key and "erc8004" in key.lower():
                data_sources["ERC-8004"] = data_sources.get("ERC-8004", 0) + r["cnt"]
            elif key and "arc" in key.lower():
                data_sources["Arc"] = data_sources.get("Arc", 0) + r["cnt"]
            else:
                data_sources["API"] = data_sources.get("API", 0) + r["cnt"]

        return {
            "total_scanned": total_scanned,
            "avg_ahs": avg_ahs,
            "avg_d1": avg_d1,
            "avg_d2": avg_d2,
            "last_updated": last_updated,
            "grade_distribution": grade_distribution,
            "pattern_distribution": pattern_distribution,
            "data_sources": data_sources,
        }
    finally:
        conn.close()


def backfill_zombie_patterns() -> int:
    """One-time backfill: add Zombie Agent patterns for scans with low D2 scores.

    Targets the latest AHS scan per address where D2 <= 40 (indicating
    zombie-like behaviour on the tokentx path) and no patterns already exist
    in scan_patterns. Returns number of patterns inserted.
    """
    conn = get_connection()
    try:
        # Find latest AHS scan per address with low D2 and no existing patterns
        rows = conn.execute(
            """SELECT latest.id, latest.address, latest.d2_score FROM (
                SELECT id, address, d2_score,
                    ROW_NUMBER() OVER (PARTITION BY address ORDER BY scan_timestamp DESC) as rn
                FROM scans WHERE endpoint = 'ahs' AND d2_score IS NOT NULL
            ) latest
            WHERE latest.rn = 1 AND latest.d2_score <= 40
            AND NOT EXISTS (
                SELECT 1 FROM scan_patterns sp WHERE sp.scan_id = latest.id
            )"""
        ).fetchall()

        inserted = 0
        for scan_id, address, d2_score in rows:
            conn.execute(
                """INSERT INTO scan_patterns (scan_id, pattern_name, severity, description, modifier)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    scan_id,
                    "Zombie Agent",
                    "critical",
                    "Agent token transfers show very low behavioural diversity — "
                    "repetitive patterns with few counterparties. Possible crashed "
                    "strategy module or abandoned agent still receiving payments.",
                    -15,
                ),
            )
            inserted += 1

        conn.commit()
        logger.info("Backfilled %d Zombie Agent patterns (D2 <= 40)", inserted)
        return inserted
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


# ---------------------------------------------------------------------------
# Batch quality tracking
# ---------------------------------------------------------------------------

def log_batch_quality(
    source: str,
    wallets_scanned: int,
    average_ahs: float | None = None,
    min_ahs: int | None = None,
    max_ahs: int | None = None,
    grade_distribution: dict | None = None,
    avg_d1: float | None = None,
    avg_d2: float | None = None,
) -> None:
    """Log aggregate quality stats for a completed scan batch."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO scan_batch_log
               (batch_date, source, wallets_scanned, average_ahs,
                min_ahs, max_ahs, grade_distribution, avg_d1, avg_d2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                wallets_scanned,
                average_ahs,
                min_ahs,
                max_ahs,
                json.dumps(grade_distribution) if grade_distribution else None,
                avg_d1,
                avg_d2,
            ),
        )
        conn.commit()
        logger.info(
            "Batch quality logged: source=%s wallets=%d avg_ahs=%.1f",
            source, wallets_scanned, average_ahs or 0,
        )
    finally:
        conn.close()


def get_batch_quality_history(days: int = 30) -> list[dict]:
    """Return batch quality records from the last N days."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT batch_date, source, wallets_scanned, average_ahs,
                      min_ahs, max_ahs, grade_distribution, avg_d1, avg_d2
               FROM scan_batch_log
               WHERE batch_date >= datetime('now', ?)
               ORDER BY batch_date DESC""",
            (f"-{days} days",),
        ).fetchall()

        result = []
        for r in rows:
            entry = dict(r)
            gd = entry.get("grade_distribution")
            if gd:
                try:
                    entry["grade_distribution"] = json.loads(gd)
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(entry)
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API Key management (Stripe fiat access path)
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key. Returns (raw_key, key_hash, key_prefix).

    Raw key format: "ahm_live_" + 32 random hex bytes (64 hex chars).
    Only the SHA-256 hash is stored — the raw key is returned once and never persisted.
    """
    raw_hex = secrets.token_hex(32)
    raw_key = f"ahm_live_{raw_hex}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = f"ahm_live_{raw_hex[:8]}..."
    return raw_key, key_hash, key_prefix


def create_api_key(
    customer_email: str,
    stripe_customer_id: str | None = None,
    key_type: str = "payg",
    tier: str = "starter",
    calls_total: int | None = 100,
    partner_id: str | None = None,
    is_reseller: bool = False,
    wholesale_rate: float = 0.5,
) -> str:
    """Create an API key record. Returns the raw key (only time it's available)."""
    raw_key, key_hash, key_prefix = generate_api_key()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO api_keys
               (key_hash, key_prefix, customer_email, stripe_customer_id,
                type, tier, calls_remaining, calls_total, created_at,
                partner_id, is_reseller, wholesale_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key_hash, key_prefix, customer_email, stripe_customer_id,
             key_type, tier, calls_total, calls_total, now_iso,
             partner_id, int(is_reseller), wholesale_rate),
        )
        conn.commit()
        logger.info("API key created: prefix=%s email=%s tier=%s calls=%s partner=%s",
                     key_prefix, customer_email, tier, calls_total, partner_id)
        return raw_key
    finally:
        conn.close()


def validate_api_key(raw_key: str) -> dict | None:
    """Validate a raw API key. Returns key record dict or None if invalid.

    Checks: key exists, is_active=1, calls_remaining > 0 (or NULL for unlimited).
    """
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT id, key_hash, key_prefix, customer_email,
                      stripe_customer_id, stripe_subscription_id,
                      type, tier, calls_remaining, calls_total,
                      created_at, expires_at, is_active,
                      partner_id, is_reseller, wholesale_rate
               FROM api_keys
               WHERE key_hash = ? AND is_active = 1""",
            (key_hash,),
        ).fetchone()
        if not row:
            return None

        record = dict(row)

        # Check expiry
        if record["expires_at"]:
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if now_iso > record["expires_at"]:
                return None

        # Check calls remaining (NULL means unlimited)
        if record["calls_remaining"] is not None and record["calls_remaining"] <= 0:
            return None

        return record
    finally:
        conn.close()


def decrement_api_key(key_hash: str) -> None:
    """Decrement calls_remaining by 1 if not NULL (unlimited keys are unaffected)."""
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE api_keys
               SET calls_remaining = calls_remaining - 1
               WHERE key_hash = ? AND calls_remaining IS NOT NULL""",
            (key_hash,),
        )
        conn.commit()
    finally:
        conn.close()


def log_api_key_usage(
    key_hash: str,
    endpoint: str,
    wallet_queried: str | None = None,
    partner_id: str | None = None,
) -> None:
    """Log an API key usage event, optionally attributed to a partner."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO api_key_usage (key_hash, endpoint, called_at, wallet_queried, partner_id)
               VALUES (?, ?, ?, ?, ?)""",
            (key_hash, endpoint, now_iso, wallet_queried, partner_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Partner usage reporting
# ---------------------------------------------------------------------------

def get_partner_usage(partner_id: str, days: int = 30) -> dict:
    """Aggregate usage stats for a partner over the last N days.

    Returns call count, cost at wholesale rate, and period boundaries.
    Looks up the partner's wholesale_rate from their api_keys record,
    and counts all usage rows attributed to the partner_id.
    """
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc)
        period_end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        period_start_dt = now - timedelta(days=days)
        period_start = period_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get wholesale rate from partner's own key (first match)
        rate_row = conn.execute(
            "SELECT wholesale_rate FROM api_keys WHERE partner_id = ? AND is_active = 1 LIMIT 1",
            (partner_id,),
        ).fetchone()
        wholesale_rate = rate_row["wholesale_rate"] if rate_row else 0.5

        # Count calls attributed to this partner in the period
        row = conn.execute(
            """SELECT COUNT(*) as call_count
               FROM api_key_usage
               WHERE partner_id = ? AND called_at >= ?""",
            (partner_id, period_start),
        ).fetchone()
        call_count = row["call_count"]

        # Default per-call retail price is the /ahs/route price ($0.01)
        retail_per_call = 0.01
        wholesale_cost = round(call_count * retail_per_call * wholesale_rate, 4)

        return {
            "partner_id": partner_id,
            "call_count": call_count,
            "wholesale_rate": wholesale_rate,
            "retail_per_call_usd": retail_per_call,
            "wholesale_cost_usd": wholesale_cost,
            "period_start": period_start,
            "period_end": period_end,
            "period_days": days,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Shield subscriptions
# ---------------------------------------------------------------------------

# Tier definitions: tier -> (agent_slots, call_quota)
SHIELD_TIERS = {
    "free":       (1,      100),
    "starter":    (5,    10_000),
    "pro":        (50,  100_000),
    "enterprise": (999, 999_999_999),  # effectively unlimited
}


def create_shield_subscription(
    api_key_hash: str,
    tier: str,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    period_days: int = 30,
) -> int:
    """Create a Shield subscription. Returns the new subscription ID."""
    now = datetime.now(timezone.utc)
    period_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    period_end = (now + timedelta(days=period_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    agent_slots, call_quota = SHIELD_TIERS.get(tier, SHIELD_TIERS["starter"])

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO shield_subscriptions
               (api_key_hash, stripe_customer_id, stripe_subscription_id,
                tier, status, agent_slots, call_quota,
                calls_used_this_period, period_start, period_end)
               VALUES (?, ?, ?, ?, 'active', ?, ?, 0, ?, ?)""",
            (api_key_hash, stripe_customer_id, stripe_subscription_id,
             tier, agent_slots, call_quota, period_start, period_end),
        )
        conn.commit()
        sub_id = cur.lastrowid
        logger.info("Shield subscription created: id=%d tier=%s key=%s…",
                     sub_id, tier, api_key_hash[:12])
        return sub_id
    finally:
        conn.close()


def get_shield_subscription(api_key_hash: str) -> dict | None:
    """Return the active Shield subscription for a given API key hash, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT * FROM shield_subscriptions
               WHERE api_key_hash = ? AND status = 'active'
               ORDER BY created_at DESC LIMIT 1""",
            (api_key_hash,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_shield_subscription_by_stripe_id(stripe_subscription_id: str) -> dict | None:
    """Look up a Shield subscription by its Stripe subscription ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM shield_subscriptions WHERE stripe_subscription_id = ?",
            (stripe_subscription_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def increment_shield_usage(api_key_hash: str) -> dict | None:
    """Increment calls_used_this_period for the active subscription.

    Returns the updated subscription dict, or None if no active subscription.
    Resets the counter and advances the period if period_end has passed.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT * FROM shield_subscriptions
               WHERE api_key_hash = ? AND status = 'active'
               ORDER BY created_at DESC LIMIT 1""",
            (api_key_hash,),
        ).fetchone()
        if not row:
            return None

        sub = dict(row)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Auto-reset if period has elapsed
        if now_iso > sub["period_end"]:
            new_start = now_iso
            new_end = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                """UPDATE shield_subscriptions
                   SET calls_used_this_period = 1, period_start = ?, period_end = ?, updated_at = ?
                   WHERE id = ?""",
                (new_start, new_end, now_iso, sub["id"]),
            )
            conn.commit()
            sub["calls_used_this_period"] = 1
            sub["period_start"] = new_start
            sub["period_end"] = new_end
        else:
            conn.execute(
                """UPDATE shield_subscriptions
                   SET calls_used_this_period = calls_used_this_period + 1, updated_at = ?
                   WHERE id = ?""",
                (now_iso, sub["id"]),
            )
            conn.commit()
            sub["calls_used_this_period"] += 1

        return sub
    finally:
        conn.close()


def update_shield_subscription_status(stripe_subscription_id: str, status: str) -> bool:
    """Update the status of a Shield subscription by Stripe ID. Returns True if updated."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = get_connection()
    try:
        cur = conn.execute(
            """UPDATE shield_subscriptions
               SET status = ?, updated_at = ?
               WHERE stripe_subscription_id = ?""",
            (status, now_iso, stripe_subscription_id),
        )
        conn.commit()
        updated = cur.rowcount > 0
        if updated:
            logger.info("Shield subscription %s -> %s", stripe_subscription_id, status)
        return updated
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Security event logging
# ---------------------------------------------------------------------------

def log_security_event(event_type: str, ip_address: str, details: str | None = None) -> None:
    """Log a security event (rate limit hit, suspicious pattern, etc.)."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO security_events (event_type, ip_address, details)
               VALUES (?, ?, ?)""",
            (event_type, ip_address, details),
        )
        conn.commit()
    except Exception:
        logger.exception("Failed to log security event: %s from %s", event_type, ip_address)
    finally:
        conn.close()


def get_agent_profile(address: str) -> dict | None:
    """Return the latest AHS profile for an agent address.

    Used by /internal/agent-profile/{address} for ahm-verify integration.
    Returns dict with ahs_score, grade, d1_score, d2_score, patterns,
    last_scanned, source — or None if address has never been scanned.
    """
    addr = address.lower()
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT s.address, s.ahs_score, s.grade, s.d1_score, s.d2_score,
                      s.scan_timestamp, s.source
               FROM scans s
               WHERE s.address = ? AND s.endpoint = 'ahs' AND s.ahs_score IS NOT NULL
               ORDER BY s.scan_timestamp DESC
               LIMIT 1""",
            (addr,),
        ).fetchone()
        if row is None:
            return None

        scan_id = conn.execute(
            "SELECT id FROM scans WHERE address = ? AND endpoint = 'ahs' ORDER BY scan_timestamp DESC LIMIT 1",
            (addr,),
        ).fetchone()

        patterns = []
        if scan_id:
            pattern_rows = conn.execute(
                "SELECT pattern_name FROM scan_patterns WHERE scan_id = ?",
                (scan_id["id"],),
            ).fetchall()
            patterns = [r["pattern_name"] for r in pattern_rows]

        return {
            "address": row["address"],
            "ahs_score": row["ahs_score"],
            "grade": row["grade"],
            "d1_score": row["d1_score"],
            "d2_score": row["d2_score"],
            "patterns": patterns,
            "last_scanned": row["scan_timestamp"],
            "source": row["source"],
        }
    finally:
        conn.close()


def get_security_events(
    limit: int = 100,
    event_type: str | None = None,
) -> list[dict]:
    """Retrieve recent security events, optionally filtered by type."""
    conn = get_connection()
    try:
        if event_type:
            rows = conn.execute(
                """SELECT id, event_type, ip_address, details, created_at
                   FROM security_events
                   WHERE event_type = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, event_type, ip_address, details, created_at
                   FROM security_events
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
