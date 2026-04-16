"""Tests for ACP proactive scanner's stale-first rotation.

Covers the same rotation-fix pattern applied to Celo in PR #102:
  - _ingest_discovered_wallets_to_db persists via INSERT OR IGNORE
  - get_acp_scan_candidates returns stale-first ordering
  - scan_wallets rotates coverage across the full registry
"""

import os
import sys

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Wallet ingestion tests
# ---------------------------------------------------------------------------

class TestIngestDiscoveredWallets:
    """_ingest_discovered_wallets_to_db must persist discoveries to
    known_wallets so they survive across nightly runs, without ever
    overwriting an existing scan history."""

    def _agent(self, acp_id: int, address: str, name: str = ""):
        from acp_proactive_scan import ACPAgent
        return ACPAgent(acp_id=acp_id, wallet_address=address, name=name)

    def test_inserts_new_rows_with_null_last_scanned(self, tmp_path, monkeypatch):
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        agents = [
            self._agent(1, "0x" + "a" * 40, "Agent Alpha"),
            self._agent(2, "0x" + "b" * 40, "Agent Beta"),
        ]
        inserted = acp_proactive_scan._ingest_discovered_wallets_to_db(agents)
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
        for r in rows:
            assert r["last_scanned_at"] is None
            assert r["scan_count"] == 0

    def test_existing_rows_are_not_overwritten(self, tmp_path, monkeypatch):
        """INSERT OR IGNORE must preserve an existing row's scan history."""
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        addr = "0x" + "c" * 40
        now_iso = "2026-04-10T12:00:00Z"
        conn = db.get_connection()
        try:
            conn.execute(
                """INSERT INTO known_wallets
                   (address, label, source, first_seen_at, last_scanned_at, scan_count,
                    latest_ahs, latest_grade)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (addr, "Existing label", "acp_proactive_scan",
                 "2026-01-01T00:00:00Z", now_iso, 5, 72, "B"),
            )
            conn.commit()
        finally:
            conn.close()

        inserted = acp_proactive_scan._ingest_discovered_wallets_to_db(
            [self._agent(42, addr, "New Name")]
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
        assert row["label"] == "Existing label"
        assert row["last_scanned_at"] == now_iso
        assert row["scan_count"] == 5
        assert row["latest_ahs"] == 72

    def test_returns_zero_for_empty_list(self, tmp_path, monkeypatch):
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()
        assert acp_proactive_scan._ingest_discovered_wallets_to_db([]) == 0

    def test_mixed_new_and_existing_counts_only_new(self, tmp_path, monkeypatch):
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        existing = "0x" + "d" * 40
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO known_wallets (address, source, first_seen_at) "
                "VALUES (?, ?, ?)",
                (existing, "acp_proactive_scan", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
        finally:
            conn.close()

        inserted = acp_proactive_scan._ingest_discovered_wallets_to_db([
            self._agent(1, existing, "Old"),
            self._agent(2, "0x" + "e" * 40, "New One"),
            self._agent(3, "0x" + "f" * 40, "New Two"),
        ])
        assert inserted == 2

    def test_deduplicates_shared_wallet_agents(self, tmp_path, monkeypatch):
        """Multiple ACP agents sharing the same wallet should produce only
        one known_wallets row."""
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        shared = "0x" + "a" * 40
        inserted = acp_proactive_scan._ingest_discovered_wallets_to_db([
            self._agent(1, shared, "Agent A"),
            self._agent(2, shared, "Agent B"),
        ])
        assert inserted == 1

        conn = db.get_connection()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM known_wallets WHERE address = ?",
                (shared,),
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# Stale-first candidate selection tests
# ---------------------------------------------------------------------------

class TestGetAcpScanCandidates:
    """get_acp_scan_candidates must return wallets stale-first so the
    nightly rotation reaches every agent over time instead of re-scoring
    the same batch every night."""

    def _insert_wallet(self, conn, address, source, last_scanned_at):
        conn.execute(
            """INSERT INTO known_wallets
               (address, label, source, first_seen_at, last_scanned_at, scan_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (address, None, source, "2026-01-01T00:00:00Z", last_scanned_at, 0),
        )

    def test_null_last_scanned_at_comes_first(self, tmp_path, monkeypatch):
        """NULLS FIRST — never-scanned wallets outrank every rescan candidate."""
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            self._insert_wallet(conn, "0x" + "1" * 40, "acp_proactive_scan",
                                "2026-04-15T00:00:00Z")
            self._insert_wallet(conn, "0x" + "2" * 40, "acp_proactive_scan",
                                "2026-01-01T00:00:00Z")
            self._insert_wallet(conn, "0x" + "3" * 40, "acp_proactive_scan", None)
            conn.commit()
        finally:
            conn.close()

        candidates = acp_proactive_scan.get_acp_scan_candidates(limit=10)
        addrs = [c["address"] for c in candidates]
        assert addrs[0] == "0x" + "3" * 40  # NULL first
        assert addrs[1] == "0x" + "2" * 40  # then oldest
        assert addrs[2] == "0x" + "1" * 40  # then newest

    def test_oldest_scanned_comes_before_newest(self, tmp_path, monkeypatch):
        """Among rescan candidates the oldest scan goes first."""
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            self._insert_wallet(conn, "0xaaa" + "a" * 37, "acp_proactive_scan",
                                "2026-04-15T00:00:00Z")
            self._insert_wallet(conn, "0xbbb" + "b" * 37, "acp_proactive_scan",
                                "2026-03-01T00:00:00Z")
            self._insert_wallet(conn, "0xccc" + "c" * 37, "acp_proactive_scan",
                                "2026-02-01T00:00:00Z")
            conn.commit()
        finally:
            conn.close()

        candidates = acp_proactive_scan.get_acp_scan_candidates(limit=10)
        addrs = [c["address"] for c in candidates]
        assert addrs == [
            "0xccc" + "c" * 37,
            "0xbbb" + "b" * 37,
            "0xaaa" + "a" * 37,
        ]

    def test_respects_limit(self, tmp_path, monkeypatch):
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            for i in range(10):
                self._insert_wallet(conn, f"0x{i:040x}", "acp_proactive_scan", None)
            conn.commit()
        finally:
            conn.close()

        candidates = acp_proactive_scan.get_acp_scan_candidates(limit=3)
        assert len(candidates) == 3

    def test_excludes_non_acp_sources(self, tmp_path, monkeypatch):
        """Celo, Arc, Olas and other registries must not bleed into the ACP rotation."""
        import acp_proactive_scan
        import db

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        conn = db.get_connection()
        try:
            self._insert_wallet(conn, "0x" + "a" * 40, "acp_proactive_scan", None)
            self._insert_wallet(conn, "0x" + "b" * 40, "celo_agent_wallet", None)
            self._insert_wallet(conn, "0x" + "c" * 40, "arc_agent_wallet", None)
            self._insert_wallet(conn, "0x" + "d" * 40, "olas", None)
            conn.commit()
        finally:
            conn.close()

        candidates = acp_proactive_scan.get_acp_scan_candidates(limit=10)
        sources = {c["source"] for c in candidates}
        assert sources == {"acp_proactive_scan"}
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# End-to-end rotation tests
# ---------------------------------------------------------------------------

class TestScanWalletsRotation:
    """End-to-end: scan_wallets must feed the stale-first candidate
    list into the AHS scorer so every agent gets its turn over time."""

    def _agent(self, acp_id: int, address: str, name: str = ""):
        from acp_proactive_scan import ACPAgent
        return ACPAgent(acp_id=acp_id, wallet_address=address, name=name)

    def test_scans_in_stale_first_order(self, tmp_path, monkeypatch):
        """With three pre-existing wallets at different ages plus one new
        discovery, scan_wallets(max_scans=3) should score the never-
        scanned discovery first, then the two stalest known wallets."""
        import acp_proactive_scan
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
                    (addr, "existing", "acp_proactive_scan",
                     "2026-01-01T00:00:00Z", ts, 1),
                )
            conn.commit()
        finally:
            conn.close()

        # One new discovery (unseen by DB) — must land in the pool first.
        new_addr = "0x" + "9" * 40
        new_agents = [self._agent(999, new_addr, "New Agent")]

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
        monkeypatch.setattr(acp_proactive_scan.time, "sleep", lambda *_a, **_k: None)
        # Suppress checkpoint file creation in tmp_path
        monkeypatch.setattr(acp_proactive_scan, "CHECKPOINT_PATH",
                            str(tmp_path / "checkpoint.json"))

        acp_proactive_scan.scan_wallets(new_agents, max_scans=3)

        # Order: NULL (new discovery) -> oldest -> middle.
        # "Fresh" wallet (0x111...) must NOT have been scored.
        assert scanned_addresses == [
            new_addr,
            "0x" + "3" * 40,
            "0x" + "2" * 40,
        ]

    def test_rotation_moves_forward_on_next_run(self, tmp_path, monkeypatch):
        """Two back-to-back runs with max_scans=2 on a 4-wallet pool must
        cover all four wallets across the two runs (no agent left behind)."""
        import acp_proactive_scan
        import db
        import monitor

        monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
        db.init_db()

        addrs = [f"0x{i:040x}" for i in (1, 2, 3, 4)]

        conn = db.get_connection()
        try:
            for a in addrs:
                conn.execute(
                    """INSERT INTO known_wallets
                       (address, label, source, first_seen_at, last_scanned_at, scan_count)
                       VALUES (?, ?, ?, ?, NULL, 0)""",
                    (a, None, "acp_proactive_scan", "2026-01-01T00:00:00Z"),
                )
            conn.commit()
        finally:
            conn.close()

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
        monkeypatch.setattr(acp_proactive_scan.time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setattr(acp_proactive_scan, "CHECKPOINT_PATH",
                            str(tmp_path / "checkpoint.json"))

        # Run 1: scores 2 of the 4 wallets.
        acp_proactive_scan.scan_wallets([], max_scans=2)
        first_run = list(scanned)
        scanned.clear()

        # Clean checkpoint between runs like production does.
        cp = tmp_path / "checkpoint.json"
        if cp.exists():
            cp.unlink()

        # Run 2: the remaining 2 wallets (still NULL or stalest).
        acp_proactive_scan.scan_wallets([], max_scans=2)
        second_run = list(scanned)

        assert len(first_run) == 2
        assert len(second_run) == 2
        assert set(first_run + second_run) == set(addrs)
        assert set(first_run).isdisjoint(set(second_run))
