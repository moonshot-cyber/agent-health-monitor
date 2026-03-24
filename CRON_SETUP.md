# ACP Nightly Scan — Automated Scheduling

Automated nightly scanning of the ACP (agdp.io) agent ecosystem.
Runs `acp_proactive_scan.py` daily at **02:00 UTC**, scanning up to 500 agents and 100 wallets per run.

---

## Option A: Railway Cron Service (Recommended)

Railway supports cron jobs on all plans. The scan runs as a **separate service** in the same Railway project — it won't interfere with the web API.

### Setup Steps

1. **Open your Railway project** (the one running `agent-health-monitor`)

2. **Create a new service:**
   - Click **"+ New"** → **"GitHub Repo"** → select the same `agent-health-monitor` repo
   - Or click **"+ New"** → **"Empty Service"** and connect the repo

3. **Set the config file** (Settings tab → "Config as Code"):
   - Set **Config File Path** to `/railway.cron.json`
   - This is the critical step — without it, Railway reads `railway.json` and runs `uvicorn api:app` instead of the scan script

4. **Configure the cron schedule** (Settings tab):

   | Setting | Value |
   |---------|-------|
   | **Cron Schedule** | `0 2 * * *` |

   The start command (`python cron_acp_scan.py`) and restart policy (`Never`) are already set in `railway.cron.json`. You do **not** need to set them manually in the dashboard.

5. **Add environment variables** (Variables tab):
   - The cron service needs the same `DB_PATH` and `BASESCAN_API_KEY` as the web service
   - Copy shared variables from the web service, or use Railway's **shared variables** feature
   - Optional overrides (defaults shown):

   | Variable | Default | Description |
   |----------|---------|-------------|
   | `ACP_MAX_AGENTS` | `500` | Max agents to fetch from ACP API |
   | `ACP_MAX_SCANS` | `100` | Max AHS scans per run |
   | `ACP_SORT` | `successfulJobCount:desc` | API sort order |
   | `ACP_MAX_RUNTIME` | `3600` | Safety timeout in seconds |

6. **Deploy** — Railway will build the service and start running it on schedule

### Cron Schedule Reference

| Schedule | Cron Expression |
|----------|-----------------|
| Daily at 02:00 UTC | `0 2 * * *` |
| Every 12 hours | `0 */12 * * *` |
| Weekdays only at 02:00 | `0 2 * * 1-5` |
| Every 6 hours | `0 */6 * * *` |

### Railway Cron Constraints

- Minimum interval: **5 minutes** between runs
- Schedule is in **UTC** (not local time)
- The service must **exit** when done — `cron_acp_scan.py` handles this
- Execution time may vary by a few minutes from the scheduled time

---

## Option B: Windows Task Scheduler (Local Fallback)

Use this if Railway cron doesn't work for your plan, or if you want to run scans from your local machine.

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

### Railway

- **Logs:** Railway dashboard → select the cron service → **Deployments** tab → click a run to see stdout/stderr
- **Status:** Green checkmark = success, red X = failed (non-zero exit)
- **Alerts:** Set up Railway notifications in Project Settings → Notifications to get notified on failures

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

Set these in the Railway cron service's Variables tab:

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
- **Railway cron timing out** — reduce both values
- **API costs matter** — lower `ACP_MAX_AGENTS` to reduce Blockscout calls

---

## Files Reference

| File | Purpose |
|------|---------|
| `cron_acp_scan.py` | Cron wrapper — handles logging, exit codes, runtime guard |
| `acp_proactive_scan.py` | Core scanner — discovery, dedup, AHS scanning, reporting |
| `railway.cron.json` | Railway service config for the cron service |
| `scripts/run_acp_scan.bat` | Windows batch runner with log rotation |
| `scripts/acp_scan_task.xml` | Windows Task Scheduler task definition |
| `CRON_SETUP.md` | This file |
