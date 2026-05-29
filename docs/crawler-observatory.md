# Crawler Observatory

Origin-side request logger that records which indexers, AI crawlers, and bots
probe the AHM API service — what they request, how they identify themselves,
and whether they target agent-discovery endpoints.

## Architecture

- **Middleware** (`CrawlerObservatoryMiddleware` in `api.py`) — outermost ASGI
  middleware, captures every request after the response status is known.
- **Queue** — non-blocking `asyncio.Queue` (bounded, 4096 slots). Excess rows
  are dropped under burst; the request path is never blocked.
- **Consumer** — single background task batch-inserts from the queue into
  `crawler_hits.db` on a dedicated SQLite connection.
- **Classifier** (`crawler_classifier.py`) — data-driven pattern table mapping
  user-agents to `(ua_class, indexer_name)`.  Edit the `_PATTERN_DEFS` list to
  add new entries.
- **Storage** — separate SQLite file (`crawler_hits.db`), not the main
  `ahm_history.db`, to avoid write contention.

## UA classes

| Class | Meaning |
|---|---|
| `known_indexer` | Agent-economy indexer (ari-indexer, A2A-Registry-TaskProbe, etc.) |
| `ai_crawler` | AI training / retrieval crawler (GPTBot, ClaudeBot, etc.) |
| `scanner` | Security scanner (l9scan, Nuclei, zgrab) |
| `generic_bot` | Bot-like UA without a specific identity |
| `browser` | Standard browser engine string |
| `unknown` | Unclassified |

## Example queries

**Top indexers in the last 7 days:**
```sql
SELECT indexer_name, ua_class, COUNT(*) AS hits
FROM   crawler_hits
WHERE  ts >= datetime('now', '-7 days')
  AND  indexer_name IS NOT NULL
GROUP  BY indexer_name, ua_class
ORDER  BY hits DESC;
```

**Discovery-path hits grouped by indexer:**
```sql
SELECT indexer_name, path, COUNT(*) AS hits
FROM   crawler_hits
WHERE  is_discovery_path = 1
GROUP  BY indexer_name, path
ORDER  BY hits DESC;
```

**Novel user-agents by first seen:**
```sql
SELECT user_agent, MIN(ts) AS first_seen, COUNT(*) AS total_hits
FROM   crawler_hits
WHERE  ua_class = 'unknown'
GROUP  BY user_agent
ORDER  BY first_seen DESC;
```

**Unknown-UA volume over time (daily):**
```sql
SELECT DATE(ts) AS day, COUNT(*) AS unknown_hits
FROM   crawler_hits
WHERE  ua_class = 'unknown'
GROUP  BY day
ORDER  BY day DESC
LIMIT  30;
```

## Data retention (GDPR)

`client_ip` is personal data under UK GDPR.  A rolling purge should delete rows
older than N days (value TBD by the data controller):

```sql
DELETE FROM crawler_hits WHERE ts < datetime('now', '-N days');
```

This can be run as a cron job or added to the nightly scheduler.

## Out of scope (follow-up)

- **ASN / org lookup and reverse DNS** — belongs in a separate periodic
  enrichment job, not inline in the request path (latency).  The raw IP is
  logged now; enrich later.
