"""Crawler Observatory persistence layer.

Separate SQLite database (crawler_hits.db) for request-level crawler
logging.  Kept out of the main ahm_history.db to avoid write contention
with the core scan/scoring workload.

Write path:
  - Callers push rows into a bounded asyncio.Queue via ``enqueue()``.
  - A single background consumer (``run_consumer()``) batch-inserts in a
    dedicated transaction on its own connection.
  - On shutdown, ``flush()`` drains the queue so a redeploy doesn't lose rows.
"""

import asyncio
import logging
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger("ahm.crawler_db")

DB_PATH = os.getenv("CRAWLER_DB_PATH", "./crawler_hits.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crawler_hits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    host            TEXT,
    path            TEXT NOT NULL,
    method          TEXT NOT NULL,
    status          INTEGER NOT NULL,
    user_agent      TEXT,
    client_ip       TEXT,
    country         TEXT,
    ua_class        TEXT NOT NULL,
    indexer_name    TEXT,
    is_discovery_path INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_crawler_hits_ts
    ON crawler_hits(ts);
CREATE INDEX IF NOT EXISTS idx_crawler_hits_indexer
    ON crawler_hits(indexer_name);
CREATE INDEX IF NOT EXISTS idx_crawler_hits_discovery
    ON crawler_hits(is_discovery_path);
"""

# Queue capacity — under sustained bot bursts, excess rows are dropped
# rather than blocking the request path.
_QUEUE_SIZE = 4096

# Batch size for consumer inserts.
_BATCH_SIZE = 128

# Max seconds to wait before flushing a partial batch.
_FLUSH_INTERVAL = 5.0

_INSERT_SQL = """
INSERT INTO crawler_hits
    (ts, host, path, method, status, user_agent, client_ip, country,
     ua_class, indexer_name, is_discovery_path)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Module-level queue; initialised lazily on first enqueue.
_queue: Optional[asyncio.Queue] = None
_dropped: int = 0


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    """Create the crawler_hits table if it doesn't exist."""
    conn = _get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        logger.info("Crawler observatory DB initialised at %s", DB_PATH)
    finally:
        conn.close()


def _ensure_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
    return _queue


def enqueue(row: tuple) -> None:
    """Non-blocking enqueue.  Drops the row on overflow."""
    global _dropped
    q = _ensure_queue()
    try:
        q.put_nowait(row)
    except asyncio.QueueFull:
        _dropped += 1
        if _dropped % 500 == 1:
            logger.warning(
                "Crawler observatory queue full — %d rows dropped so far",
                _dropped,
            )


def _flush_batch(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    """Insert a batch of rows in a single transaction."""
    if not batch:
        return
    try:
        conn.executemany(_INSERT_SQL, batch)
        conn.commit()
    except Exception:
        logger.exception("Failed to insert %d crawler_hits rows", len(batch))
        try:
            conn.rollback()
        except Exception:
            pass


async def run_consumer() -> None:
    """Background consumer: drains the queue and batch-inserts rows.

    Runs until cancelled.  Uses its own dedicated SQLite connection so
    it never contends with the main DB or other consumers.
    """
    q = _ensure_queue()
    conn = _get_connection()
    batch: list[tuple] = []
    last_flush = time.monotonic()

    try:
        while True:
            # Wait for at least one item (with timeout for partial-batch flush).
            try:
                row = await asyncio.wait_for(q.get(), timeout=_FLUSH_INTERVAL)
                batch.append(row)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

            # Drain remaining items up to batch size.
            while len(batch) < _BATCH_SIZE:
                try:
                    batch.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            now = time.monotonic()
            if batch and (len(batch) >= _BATCH_SIZE or now - last_flush >= _FLUSH_INTERVAL):
                _flush_batch(conn, batch)
                batch.clear()
                last_flush = now

    except asyncio.CancelledError:
        # Drain whatever is left before exiting.
        while not q.empty():
            try:
                batch.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        _flush_batch(conn, batch)
        logger.info("Crawler consumer shutdown — flushed final %d rows", len(batch))
    finally:
        conn.close()


async def flush() -> None:
    """Drain the queue into the DB.  Called during app shutdown."""
    q = _ensure_queue()
    conn = _get_connection()
    batch: list[tuple] = []
    while not q.empty():
        try:
            batch.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    if batch:
        _flush_batch(conn, batch)
        logger.info("Crawler flush on shutdown — wrote %d rows", len(batch))
    conn.close()
