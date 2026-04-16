# Nightly Agent-Registry Scans — Automated Scheduling

Automated nightly scanning of every agent-registry source AHM tracks.
All four scans run inside the FastAPI web process via APScheduler — no
separate Railway cron service is needed.

| Scan | Schedule (UTC) | Source |
|------|----------------|--------|
| ACP      | 02:00 | agdp.io REST API |
| Olas     | 02:30 | Olas ServiceRegistryL2 (Base) |
| Celo     | 02:45 | ERC-8004 IdentityRegistry (Celo mainnet) |
| Arc      | 03:00 | ERC-8004 IdentityRegistry (Arc testnet) |
| ERC-8004 | 03:15 | ERC-8004 IdentityRegistry (Base mainnet) |

---

## Production: APScheduler (in-process)

The scheduler is configured in `api.py` lifespan. Each scan adds an
`add_job` call with a unique id and a `CronTrigger` at the time slot above:

```python
scheduler.add_job(
    run_acp_scan,
    trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
    id="acp_nightly_scan",
    coalesce=True,
    misfire_grace_time=3600,
    max_instances=1,
)
```

### How it works

- `run_acp_scan()` imports from `acp_proactive_scan.py` and offloads sync I/O to `run_in_executor`
- An `asyncio.Lock` prevents concurrent runs
- Already-scanned wallets are automatically skipped
- Configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACP_MAX_AGENTS` | `500` | Max agents to fetch from ACP API |
| `ACP_MAX_SCANS` | `100` | Max AHS scans per run |
| `ACP_SORT` | `successfulJobCount:desc` | API sort order |
| `ACP_MAX_RUNTIME` | `3600` | Safety timeout in seconds |
| `OLAS_MAX_SCANS` | `200` | Max wallets to AHS-score per Olas run |
| `CELO_MAX_SCANS` | `200` | Max wallets to AHS-score per Celo run. Celo registry currently holds ~7,188 agents (verified 2026-04-16 via `ownerOf()` binary search), so at the default 200/night full rotation takes ~36 days. Raise to 400–500 if Alchemy latency permits to halve rotation time. |
| `CELO_RPC_URL` | `https://forno.celo.org` | Celo mainnet RPC endpoint |
| `CELO_CHECKPOINT_PATH` | `celo_scan_checkpoint.json` | Block checkpoint persistence |
| `ARC_MAX_SCANS` | `200` | Max wallets to AHS-score per Arc run |
| `ERC8004_MAX_SCANS` | `100` | Max wallets to AHS-score per ERC-8004 Base run |
| `ERC8004_START_ID` | `1` | Agent ID to start enumeration from (increment to rotate coverage across the 32k+ Base registry) |

### Manual trigger

```bash
# Trigger scan immediately (requires X-Internal-Key header)
curl -X POST https://agenthealthmonitor.xyz/acp-scan/trigger \
  -H "X-Internal-Key: $INTERNAL_API_KEY"

# Check scan status
curl https://agenthealthmonitor.xyz/acp-scan/status \
  -H "X-Internal-Key: $INTERNAL_API_KEY"
```

### Why there's no Procfile

This repo intentionally has **no Procfile**. Nixpacks treats the Procfile `web:` command as the highest-priority start command, overriding `startCommand` in `railway.json`. The start command is defined exclusively in `railway.json`.

---

## Local Fallback: Windows Task Scheduler

Use this if you want to run scans from your local machine.

### Automatic Setup (one command)

Open **PowerShell as Administrator** and run:

```powershell
schtasks /create /tn "ACP Nightly Scan" /xml "C:\Users\Pablo\agent-health-monitor\scripts\acp_scan_task.xml" /f
```

This imports the pre-built task definition that runs daily at 02:00 UTC.

### Manual Setup

1. Open **Task Scheduler** (`taskschd.msc`)
2. Click **"Create Task"** (not "Create Basic Task")
3. **General tab:**
   - Name: `ACP Nightly Scan`
   - Run whether user is logged on or not: **No** (use "Run only when user is logged on" to see console output)
   - Do not store password: **checked**
4. **Triggers tab → New:**
   - Begin the task: On a schedule
   - Daily, start at **02:00** (adjust for your UTC offset — UK is UTC+0 in winter, UTC+1 in BST)
   - Enabled: **checked**
5. **Actions tab → New:**
   - Action: Start a program
   - Program: `C:\Users\Pablo\agent-health-monitor\scripts\run_acp_scan.bat`
   - Start in: `C:\Users\Pablo\agent-health-monitor`
6. **Conditions tab:**
   - Start only if network connection is available: **checked**
   - Stop if goes on battery: **unchecked**
7. **Settings tab:**
   - If task is already running: Do not start a new instance
   - Stop task if it runs longer than: **2 hours**
   - If task fails, restart every: **5 minutes**, up to **2 times**

### Task Management Commands

```powershell
# Check task status
schtasks /query /tn "ACP Nightly Scan" /fo LIST /v

# Run immediately (test)
schtasks /run /tn "ACP Nightly Scan"

# Disable temporarily
schtasks /change /tn "ACP Nightly Scan" /disable

# Re-enable
schtasks /change /tn "ACP Nightly Scan" /enable

# Delete the task
schtasks /delete /tn "ACP Nightly Scan" /f
```

### Log Location

Logs are written to `C:\Users\Pablo\agent-health-monitor\logs\acp_scan_YYYYMMDD_HHMM.log`.
Logs older than 30 days are automatically cleaned up.

---

## Monitoring

### Production (APScheduler)

- **Logs:** Railway dashboard → web service → **Deployments** tab → search for `ACP nightly scan`
- **Status endpoint:** `GET /acp-scan/status` (protected by `X-Internal-Key`) returns `running` flag and `next_scheduled_run`
- **Manual trigger:** `POST /acp-scan/trigger` (protected by `X-Internal-Key`)

### Windows Task Scheduler

- **Last run result:** `schtasks /query /tn "ACP Nightly Scan" /fo LIST /v` → check "Last Result"
  - `0` = success
  - `1` = error (check log file)
- **Log files:** `C:\Users\Pablo\agent-health-monitor\logs\acp_scan_*.log`
- **Event Viewer:** `eventvwr.msc` → Windows Logs → Application → filter by Source: "Task Scheduler"

### Quick Health Checks

```bash
# How many wallets scanned total?
python -c "import db; db.init_db(); c=db.get_connection(); print(c.execute(\"SELECT COUNT(DISTINCT address) FROM scans WHERE source='acp_proactive_scan'\").fetchone()[0]); c.close()"

# Last scan timestamp?
python -c "import db; db.init_db(); c=db.get_connection(); print(c.execute(\"SELECT MAX(scan_timestamp) FROM scans WHERE source='acp_proactive_scan'\").fetchone()[0]); c.close()"

# Grade distribution from ACP scans?
python -c "import db; db.init_db(); c=db.get_connection(); rows=c.execute(\"SELECT grade, COUNT(*) FROM scans WHERE source='acp_proactive_scan' AND grade IS NOT NULL GROUP BY grade ORDER BY grade\").fetchall(); [print(f'{g}: {n}') for g,n in rows]; c.close()"
```

---

## Adjusting Batch Size

The scan processes agents in two stages — **discovery** (fetching from ACP API) and **scanning** (running AHS on each wallet). Both are independently configurable.

### Via Environment Variables (Railway)

Set these in the Railway web service's Variables tab:

| Variable | Default | Effect |
|----------|---------|--------|
| `ACP_MAX_AGENTS` | `500` | Agents fetched from ACP API. More = broader coverage but slower discovery. |
| `ACP_MAX_SCANS` | `100` | AHS scans per run. Each scan takes ~4s (2s Blockscout rate limit × 2 calls). 100 scans ≈ 7 min. |
| `ACP_MAX_RUNTIME` | `3600` | Safety timeout. If discovery phase exceeds 80% of this, scanning is skipped. |

### Via CLI Arguments (local)

```bash
# Conservative (quick run, ~3 min)
python cron_acp_scan.py --max-agents 200 --max-scans 50

# Default (balanced, ~10 min)
python cron_acp_scan.py --max-agents 500 --max-scans 100

# Aggressive (full coverage, ~30 min)
python cron_acp_scan.py --max-agents 2000 --max-scans 500

# Discovery only (no scanning, ~30s)
python acp_proactive_scan.py --max-agents 500 --skip-scan
```

### Batch Size Estimation

| max-scans | Approx. Runtime | Wallets/Night | Days to Scan 500 Wallets |
|-----------|-----------------|---------------|--------------------------|
| 50 | ~4 min | 50 new | 10 days |
| 100 | ~8 min | 100 new | 5 days |
| 200 | ~15 min | 200 new | 3 days |
| 500 | ~35 min | 500 new | 1 day |

Runtime assumes ~4s per wallet (2× Blockscout calls with 2s delay each). Already-scanned wallets are automatically skipped, so subsequent runs only scan new agents that appeared since the last run.

### When to Increase Batch Size

- **ACP registry is growing fast** — increase `ACP_MAX_AGENTS` to discover new agents
- **Many new wallets per day** — increase `ACP_MAX_SCANS` to keep up
- **Initial backfill** — temporarily set `ACP_MAX_SCANS=500` to scan the full ecosystem, then reduce

### When to Decrease Batch Size

- **Blockscout rate limiting (429 errors)** — reduce `ACP_MAX_SCANS` or add `BASESCAN_API_KEY`
- **Scan taking too long** — reduce both values
- **API costs matter** — lower `ACP_MAX_AGENTS` to reduce Blockscout calls

---

## Files Reference

| File | Purpose |
|------|---------|
| `api.py` (`run_acp_scan`)  | Production entry point — APScheduler job at 02:00 UTC |
| `api.py` (`run_olas_scan`) | APScheduler job at 02:30 UTC |
| `api.py` (`run_celo_scan`) | APScheduler job at 02:45 UTC |
| `api.py` (`run_arc_scan`)  | APScheduler job at 03:00 UTC |
| `api.py` (`run_erc8004_scan`) | APScheduler job at 03:15 UTC |
| `acp_proactive_scan.py` | Core scanner — discovery, dedup, AHS scanning, reporting |
| `olas_scan.py` | Olas ServiceRegistryL2 (Base) discovery + AHS scoring |
| `celo_scan.py` | Celo ERC-8004 IdentityRegistry discovery + AHS scoring |
| `arc_scan.py`  | Arc ERC-8004 IdentityRegistry discovery + AHS scoring |
| `erc8004_scan.py` | ERC-8004 IdentityRegistry on Base mainnet discovery + AHS scoring |
| `cron_acp_scan.py` | Standalone CLI wrapper for manual runs |
| `scripts/run_acp_scan.bat` | Windows batch runner with log rotation |
| `scripts/acp_scan_task.xml` | Windows Task Scheduler task definition |
| `CRON_SETUP.md` | This file |
