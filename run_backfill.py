#!/usr/bin/env python3
"""One-time script to run zombie pattern backfill."""
import db
db.init_db()
filled = db.backfill_zombie_patterns()
print(f"Backfilled: {filled}")

import sqlite3
conn = sqlite3.connect(db.DB_PATH)
rows = conn.execute("SELECT pattern_name, COUNT(*) FROM scan_patterns GROUP BY pattern_name ORDER BY COUNT(*) DESC").fetchall()
for name, cnt in rows:
    print(f"  {name}: {cnt}")
total = conn.execute("SELECT COUNT(DISTINCT address) FROM scans WHERE endpoint='ahs' AND ahs_score IS NOT NULL").fetchone()[0]
zombie = sum(cnt for name, cnt in rows if name == "Zombie Agent")
print(f"Zombie rate: {zombie}/{total} = {zombie/total*100:.1f}%")
conn.close()
