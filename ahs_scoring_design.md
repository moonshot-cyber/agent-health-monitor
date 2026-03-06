# Agent Health Score (AHS) — Proprietary Scoring Model Design

**Version:** AHS v1.0 Design Spike
**Date:** 2026-03-06
**Status:** Design — no implementation

---

## 1. Scoring Architecture

### Composite Formula

```
AHS = w1 * D1 + w2 * D2 + w3 * D3 + CDP_modifier
```

| Symbol | Name | Weight (3D mode) | Weight (2D fallback) |
|--------|------|-------------------|----------------------|
| D1 | Wallet Hygiene | 0.25 | 0.30 |
| D2 | Behavioural Patterns | 0.45 | 0.70 |
| D3 | Infrastructure Health | 0.30 | — (not available) |
| CDP | Cross-Dimensional Patterns | -15 to +5 | -10 to +5 |

**Rationale for weights:**
- D2 is heaviest because behavioural signals are the hardest to replicate and the most diagnostic of agent health. Any wallet scanner can count dust tokens (D1). Only AHM analyses HOW the agent transacts.
- D3 is significant when available because it catches infrastructure failures that on-chain data shows with a delay.
- D1 is lowest because it's largely static between scans — dust and spam accumulate slowly.
- CDP is an additive modifier (not a multiplied dimension) so that cross-dimensional patterns can push scores down hard when multiple systems are failing simultaneously.

### Score Range and Grades

| AHS Range | Grade | Label |
|-----------|-------|-------|
| 90–100 | A | Excellent |
| 75–89 | B | Good |
| 60–74 | C | Needs Attention |
| 40–59 | D | Degraded |
| 20–39 | E | Critical |
| 0–19 | F | Failing |

### Confidence Level

Each scan produces a confidence level based on data availability:

| Confidence | Conditions |
|------------|------------|
| **High** | ≥100 txs, ≥7 days history, all 3 dimensions scored |
| **Medium** | ≥50 txs, ≥3 days history, D1+D2 scored |
| **Low** | ≥10 txs, D1+D2 scored |
| **Insufficient** | <10 txs — return score with explicit warning |

The confidence level is exposed in the API response. It tells the caller how much to trust the score. A "Low confidence 85" is less meaningful than a "High confidence 72."

### Missing Dimension Handling

When `agent_url` is not provided (D3 unavailable):
- Weights redistribute: D1=0.30, D2=0.70
- Confidence capped at "Medium" (can't be High without D3)
- CDP patterns that require D3 are skipped
- Response includes `dimensions_scored: ["hygiene", "behaviour"]` so the caller knows

When transaction history is very short (<10 txs):
- D2 scores default to 50 (neutral) with high uncertainty
- Confidence is "Insufficient"
- Only D1 signals are reliable

---

## 2. Signal Catalogue

### Dimension 1: Wallet Hygiene (D1)

All signals derived from existing `/wash` analysis. D1 reuses the cleanliness score infrastructure but reweights for the AHS context.

| Signal | Measurement | Range | Good | Bad | Weight in D1 | Data Source |
|--------|-------------|-------|------|-----|--------------|-------------|
| **Dust token count** | Count of tokens with value < $0.01 USD | 0–∞ | 0–5 | >50 | 0.15 | V2 token API + exchange_rate |
| **Dust total value** | Sum of all dust token USD values | $0–∞ | <$0.10 | >$1.00 | 0.05 | V2 token API |
| **Spam token count** | Count of tokens matching spam heuristics | 0–∞ | 0–3 | >20 | 0.20 | V2 token API (name, holders, volume, market_cap) |
| **Gas efficiency (avg)** | Mean(gasUsed / gasLimit) for successful txs | 0–100% | 40–85% | <30% or >95% | 0.25 | txlist: gas, gasUsed |
| **Failed tx rate (24h)** | failed_count / total_count in 24h window | 0–100% | <5% | >20% | 0.20 | txlist: isError, timeStamp |
| **Nonce gap count** | Missing nonce values in sequence | 0–∞ | 0 | >3 | 0.15 | txlist: nonce |

**D1 Score Calculation:**

```
d1_dust      = max(0, 100 - dust_count * 1.5)          # capped at 0
d1_spam      = max(0, 100 - spam_count * 2)             # capped at 0
d1_gas_eff   = gas_efficiency_pct if in [40,85] else penalty curve (see below)
d1_fail_rate = max(0, 100 - failed_pct * 3)
d1_nonce     = max(0, 100 - nonce_gaps * 15)
d1_dust_val  = 100 if dust_total < 0.10 else max(0, 100 - (dust_total - 0.10) * 50)

D1 = (d1_dust * 0.15) + (d1_dust_val * 0.05) + (d1_spam * 0.20)
   + (d1_gas_eff * 0.25) + (d1_fail_rate * 0.20) + (d1_nonce * 0.15)
```

Gas efficiency penalty curve (same as existing health score):
```
if 0.40 <= avg_eff <= 0.85:
    d1_gas_eff = 100
elif avg_eff < 0.40:
    d1_gas_eff = (avg_eff / 0.40) * 100     # linear ramp up to sweet spot
else:  # > 0.85 — running too close to limit
    d1_gas_eff = max(0, 100 - (avg_eff - 0.85) * 500)
```

---

### Dimension 2: Behavioural Patterns (D2)

All signals derived from txlist data. This is the proprietary core.

| Signal | Measurement | Range | Good | Bad | Weight in D2 | Data Source |
|--------|-------------|-------|------|-----|--------------|-------------|
| **Repeated failure patterns** | Max consecutive failures to same contract:method | 0–∞ | 0–2 | >5 | 0.20 | txlist: to, input[:10], isError |
| **Gas adaptation index** | StdDev of gasPrice over 24h / mean gasPrice | 0–∞ | >0.15 | <0.05 | 0.15 | txlist: gasPrice, timeStamp |
| **Nonce management quality** | Nonce gaps persisting >48h | 0–∞ | 0 | >2 | 0.10 | txlist: nonce, timeStamp |
| **Timing regularity** | Coefficient of variation of inter-tx intervals | 0–∞ | 0.3–2.0 | <0.05 or >5.0 | 0.15 | txlist: timeStamp |
| **Transaction diversity** | Unique (contract, method_id) pairs / total txs | 0–1.0 | >0.05 | <0.01 | 0.10 | txlist: to, input[:10] |
| **Retry storm frequency** | Count of 3+ identical txs within 5-min windows | 0–∞ | 0 | >3 events | 0.15 | txlist: to, input, timeStamp |
| **Contract interaction breadth** | Unique `to` addresses / total txs (last 7d) | 0–1.0 | >0.10 | <0.02 | 0.10 | txlist: to |
| **Activity gap detection** | Max gap between txs / median gap (last 7d) | 1.0–∞ | <5.0 | >20.0 | 0.05 | txlist: timeStamp |

#### Signal Detail: Repeated Failure Patterns (weight: 0.20)

**What it detects:** Agent stuck retrying a broken interaction — stale cache, removed liquidity pool, revoked approval, contract upgraded.

**Measurement:**
```python
# Group txs by (to_address, method_id) where method_id = input[:10]
# For each group, find max consecutive failures in chronological order
# Also track: does the agent EVER succeed between failures?

for group in tx_groups:
    consecutive = max_consecutive_failures(group)
    ever_recovers = any_success_after_failure(group)

max_consec = max(consecutive for all groups)
has_recovery = any(ever_recovers for all groups with failures)
```

**Scoring:**
```python
if max_consec <= 2:
    score = 100
elif max_consec <= 5:
    score = 80 - (max_consec - 2) * 10   # 70, 60, 50
elif max_consec <= 10:
    score = 50 - (max_consec - 5) * 8    # 42, 34, 26, 18, 10
else:
    score = 0

# Recovery bonus: if agent eventually succeeds, it's adapting
if has_recovery and score < 80:
    score = min(80, score + 15)
```

#### Signal Detail: Gas Adaptation Index (weight: 0.15)

**What it detects:** Agent using hardcoded gas price vs dynamically adapting to network conditions.

**Measurement:**
```python
# Collect gasPrice values from last 24h of txs
# Compute coefficient of variation: std / mean
# A healthy agent adjusts gas price per tx based on network congestion
# A broken agent uses the same gas price every time

gas_prices_24h = [int(tx["gasPrice"]) for tx in recent_txs]
if len(gas_prices_24h) < 3:
    return None  # insufficient data

mean_gp = mean(gas_prices_24h)
std_gp = stdev(gas_prices_24h)
gas_adaptation_index = std_gp / mean_gp if mean_gp > 0 else 0
```

**Scoring:**
```python
if gas_adaptation_index >= 0.15:
    score = 100                              # adapting well
elif gas_adaptation_index >= 0.05:
    score = 60 + (gas_adaptation_index - 0.05) * 400  # linear 60→100
elif gas_adaptation_index >= 0.01:
    score = 30 + (gas_adaptation_index - 0.01) * 750  # linear 30→60
else:
    score = max(0, gas_adaptation_index * 3000)        # 0→30
```

**Edge case:** If there are <5 txs in 24h, gas adaptation is scored at 50 (neutral) — not enough data to judge.

#### Signal Detail: Nonce Management Quality (weight: 0.10)

**What it detects:** Persistent infrastructure problems (stuck tx manager, crashed signer, wallet lock contention) vs one-off nonce collisions.

**Measurement:**
```python
# Use existing detect_nonce_issues() to get gaps
# Then check: do gaps persist across >48h of tx history?
# A one-off gap that resolves quickly = minor issue
# A gap persisting for days = infrastructure failure

nonce_gaps, retries = detect_nonce_issues(transactions)

# Check gap persistence: for each gap, find first tx before and after
# If gap age > 48h, flag as persistent
persistent_gaps = count_gaps_older_than(transactions, hours=48)
```

**Scoring:**
```python
if persistent_gaps == 0 and nonce_gaps <= 1:
    score = 100
elif persistent_gaps == 0:
    score = max(60, 100 - nonce_gaps * 10)   # transient gaps, mild penalty
elif persistent_gaps <= 2:
    score = max(20, 60 - persistent_gaps * 20)
else:
    score = 0
```

#### Signal Detail: Timing Regularity (weight: 0.15)

**What it detects:** Agent crashes (long unexpected gaps), restarts (bursts after gaps), and healthy cron behaviour (regular intervals).

**Measurement:**
```python
timestamps = sorted([int(tx["timeStamp"]) for tx in transactions])
intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

if len(intervals) < 5:
    return None  # insufficient data

median_interval = median(intervals)
max_interval = max(intervals)

# Coefficient of variation
cv = stdev(intervals) / mean(intervals) if mean(intervals) > 0 else 0

# Gap ratio: largest gap vs median
gap_ratio = max_interval / median_interval if median_interval > 0 else float('inf')

# Burst detection: count intervals < median/10 (rapid-fire after silence)
burst_count = sum(1 for i in intervals if i < median_interval / 10)
```

**Scoring:**
```python
# Sweet spot: CV between 0.3 and 2.0 (some variation but not erratic)
if 0.3 <= cv <= 2.0:
    timing_base = 100
elif cv < 0.3:
    # Too regular — could be healthy cron OR stuck in a loop
    # Score moderately (loop detection handled by tx_diversity)
    timing_base = 75
else:  # cv > 2.0
    timing_base = max(0, 100 - (cv - 2.0) * 15)

# Gap penalty
if gap_ratio > 20:
    timing_base -= 30       # major crash suspected
elif gap_ratio > 10:
    timing_base -= 15       # moderate outage

# Burst penalty (rapid-fire after gap = restart storm)
if burst_count > 5:
    timing_base -= 20

score = max(0, min(100, timing_base))
```

#### Signal Detail: Transaction Diversity (weight: 0.10)

**What it detects:** Agent stuck in a loop (doing the same thing repeatedly) vs adaptive behaviour.

**Measurement:**
```python
# Unique (to, method_id) pairs in last 7 days
recent_txs = [tx for tx in transactions if age(tx) <= 7 * 86400]
pairs = set((tx["to"], tx["input"][:10]) for tx in recent_txs if tx.get("input", "0x") != "0x")

# Also count plain ETH transfers (input == "0x")
eth_transfers = sum(1 for tx in recent_txs if tx.get("input", "0x") == "0x")

diversity_ratio = len(pairs) / len(recent_txs) if recent_txs else 0

# Absolute diversity: unique pairs count
unique_pair_count = len(pairs)
```

**Scoring:**
```python
if unique_pair_count >= 10 or diversity_ratio >= 0.10:
    score = 100
elif unique_pair_count >= 5 or diversity_ratio >= 0.05:
    score = 75
elif unique_pair_count >= 2 or diversity_ratio >= 0.02:
    score = 50
elif unique_pair_count == 1:
    score = 25      # doing exactly one thing on repeat
else:
    score = 0       # no contract interactions at all
```

#### Signal Detail: Retry Storm Frequency (weight: 0.15)

**What it detects:** Agent blindly retrying failed transactions without backoff — wasting gas and clogging nonces.

**Measurement:**
```python
# Group txs by (to, input) — identical calldata
# Within each group, find 5-minute windows with 3+ txs
# Each such window = one retry storm event

storm_events = 0
for group in identical_tx_groups:
    sorted_txs = sort_by_timestamp(group)
    window_start = sorted_txs[0]["timeStamp"]
    window_count = 1
    for tx in sorted_txs[1:]:
        if tx["timeStamp"] - window_start <= 300:  # 5 min
            window_count += 1
        else:
            if window_count >= 3:
                storm_events += 1
            window_start = tx["timeStamp"]
            window_count = 1
    if window_count >= 3:
        storm_events += 1
```

**Scoring:**
```python
if storm_events == 0:
    score = 100
elif storm_events <= 2:
    score = 70 - storm_events * 10
elif storm_events <= 5:
    score = 50 - (storm_events - 2) * 10
else:
    score = max(0, 20 - (storm_events - 5) * 5)
```

#### Signal Detail: Contract Interaction Breadth (weight: 0.10)

**Measurement:**
```python
recent_txs = last_7_days(transactions)
unique_contracts = set(tx["to"] for tx in recent_txs if tx.get("to"))
breadth_ratio = len(unique_contracts) / len(recent_txs) if recent_txs else 0
```

**Scoring:**
```python
if breadth_ratio >= 0.15 or len(unique_contracts) >= 8:
    score = 100
elif breadth_ratio >= 0.05 or len(unique_contracts) >= 4:
    score = 70
elif len(unique_contracts) >= 2:
    score = 40
else:
    score = 10    # single contract — stuck or very specialized
```

#### Signal Detail: Activity Gap Detection (weight: 0.05)

**Measurement:**
```python
recent = last_7_days(transactions)
intervals = compute_intervals(recent)
max_gap = max(intervals)
median_gap = median(intervals)
gap_ratio = max_gap / median_gap if median_gap > 0 else 1.0
```

**Scoring:**
```python
if gap_ratio < 5:
    score = 100
elif gap_ratio < 10:
    score = 70
elif gap_ratio < 20:
    score = 40
else:
    score = 10    # major outage detected
```

---

### Dimension 3: Infrastructure Health (D3)

Only scored when `agent_url` parameter is provided. All signals derived from HTTP probes to the agent's service.

| Signal | Measurement | Range | Good | Bad | Weight in D3 | Probe Method |
|--------|-------------|-------|------|-----|--------------|--------------|
| **Endpoint availability** | HTTP GET to agent_url — 2xx response? | 0/1 | 1 | 0 | 0.30 | HTTP GET with 10s timeout |
| **Response latency** | Time to first byte (ms) | 0–10000 | <500ms | >3000ms | 0.20 | HTTP GET timing |
| **x402 header correctness** | Valid PAYMENT-REQUIRED on 402 response | 0–100 | all fields valid | missing/malformed | 0.15 | HTTP GET, parse 402 headers |
| **API metadata quality** | /.well-known/x402 or /api/info exists and valid | 0–100 | complete metadata | missing/404 | 0.15 | HTTP GET to discovery endpoints |
| **Data freshness** | Block numbers/timestamps in response are current | 0–100 | within 5 min | >1 hour stale | 0.20 | Parse response body timestamps |

#### Probe Execution Design

All probes run concurrently with a 10-second global timeout. If the agent is completely unreachable, D3 = 0.

```python
async def probe_infrastructure(agent_url: str) -> D3Result:
    probes = await asyncio.gather(
        probe_availability(agent_url),      # GET /
        probe_latency(agent_url),           # timed GET /
        probe_x402_headers(agent_url),      # GET any paid endpoint, check 402
        probe_api_metadata(agent_url),      # GET /.well-known/x402, GET /api/info
        probe_data_freshness(agent_url),    # GET a cheap endpoint, check timestamps
        return_exceptions=True
    )
    return aggregate(probes)
```

#### Signal Detail: Endpoint Availability (weight: 0.30)

```python
async def probe_availability(url: str) -> int:
    try:
        resp = await httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code < 500:
            return 100          # any non-5xx = agent is up (402 counts as up)
        return 20               # 5xx = partially up but erroring
    except (ConnectError, TimeoutError):
        return 0                # completely unreachable
```

#### Signal Detail: Response Latency (weight: 0.20)

```python
async def probe_latency(url: str) -> int:
    start = time.monotonic()
    try:
        await httpx.get(url, timeout=10)
        ms = (time.monotonic() - start) * 1000
    except:
        return 0

    if ms < 200:   return 100
    if ms < 500:   return 85
    if ms < 1000:  return 65
    if ms < 3000:  return 35
    return 10                   # >3s, severely overloaded
```

#### Signal Detail: x402 Header Correctness (weight: 0.15)

```python
async def probe_x402_headers(url: str) -> int:
    # Hit a known paid endpoint to get a 402
    resp = await httpx.get(url, timeout=10)
    if resp.status_code != 402:
        return 50               # can't assess, neutral score

    score = 0
    # Check for required x402 fields
    body = resp.json()
    if "payTo" in body:         score += 25
    if "maxAmountRequired" in body: score += 25
    if "network" in body:       score += 25
    if "resource" in body:      score += 25
    return score
```

#### Signal Detail: API Metadata Quality (weight: 0.15)

```python
async def probe_api_metadata(url: str) -> int:
    score = 0

    # Check /.well-known/x402
    resp1 = await httpx.get(f"{url}/.well-known/x402", timeout=5)
    if resp1.status_code == 200:
        data = resp1.json()
        if "endpoints" in data or "routes" in data:
            score += 50

    # Check /api/info
    resp2 = await httpx.get(f"{url}/api/info", timeout=5)
    if resp2.status_code == 200:
        data = resp2.json()
        if "version" in data:
            score += 25
        if "endpoints" in data:
            score += 25

    return score
```

#### Signal Detail: Data Freshness (weight: 0.20)

```python
async def probe_data_freshness(url: str) -> int:
    # Try to get a response with timestamps
    # Look for analyzed_at, scan_timestamp, block_number fields
    resp = await httpx.get(f"{url}/api/info", timeout=5)
    if resp.status_code != 200:
        return 50               # can't assess

    data = resp.json()
    # Check if any timestamp fields exist and are recent
    for field in ["last_updated", "analyzed_at", "scan_timestamp"]:
        if field in data:
            age_seconds = now() - parse_timestamp(data[field])
            if age_seconds < 300:       return 100   # <5 min
            if age_seconds < 3600:      return 70    # <1 hour
            if age_seconds < 86400:     return 40    # <1 day
            return 10                                 # stale

    return 50    # no timestamp fields found — neutral
```

---

## 3. Cross-Dimensional Patterns (CDP)

These are the real IP. Each pattern detects a specific failure mode that no single signal would catch. The patterns apply additive modifiers to the composite score.

### Pattern 1: "Zombie Agent" (modifier: -15)

**Detection:** High gas efficiency + zero failed txs + very low tx diversity + low contract breadth

**Logic:**
```python
if (d1_gas_eff > 90 and
    failed_pct_24h == 0 and
    tx_diversity_ratio < 0.02 and
    unique_contracts <= 1):
    cdp_modifier -= 15
```

**What it means:** The agent is technically functional — it's not failing. But it's doing the same single thing on repeat. It's alive but not useful. This is common when an agent's strategy module crashes but the transaction sender keeps running on the last cached instruction.

**Why single signals miss it:** Each individual signal looks fine. Low failure rate? Good. High gas efficiency? Good. Only the *combination* reveals the agent is brain-dead.

---

### Pattern 2: "Cascading Infrastructure Failure" (modifier: -15)

**Detection:** Rising failure rate + widening nonce gaps + timing anomalies (long gap followed by burst)

**Logic:**
```python
# Compare failure rate in first half vs second half of recent txs
recent = last_7_days(transactions)
mid = len(recent) // 2
first_half_fail = failure_rate(recent[:mid])
second_half_fail = failure_rate(recent[mid:])
failure_rising = second_half_fail > first_half_fail * 1.5

if (failure_rising and
    persistent_nonce_gaps > 0 and
    gap_ratio > 10):
    cdp_modifier -= 15
```

**What it means:** The agent is actively degrading. Failures are compounding — each failure creates nonce gaps, which cause more failures, which cause the agent to crash and restart (timing burst). This is an infrastructure death spiral.

---

### Pattern 3: "Stale Strategy" (modifier: -10)

**Detection:** Repeated failures to same contract + declining tx diversity over time + no gas adaptation

**Logic:**
```python
if (max_consecutive_failures > 5 and
    tx_diversity_ratio < 0.03 and
    gas_adaptation_index < 0.05):
    cdp_modifier -= 10
```

**What it means:** The agent is trying to execute a strategy that no longer works (liquidity removed, contract upgraded, approval revoked) and isn't adapting. The gas price is hardcoded (no adaptation), it's not trying new contracts (no diversity), and it keeps failing on the same one.

---

### Pattern 4: "Healthy Operator" (modifier: +5)

**Detection:** Clean wallet + good gas adaptation + diverse interactions + no retry storms + (if D3: fast response + good metadata)

**Logic:**
```python
if (D1 >= 80 and
    gas_adaptation_index > 0.15 and
    tx_diversity_ratio > 0.05 and
    storm_events == 0 and
    (D3 is None or D3 >= 80)):
    cdp_modifier += 5
```

**What it means:** Everything is working well across all dimensions. The agent is actively adapting to network conditions, maintaining a clean wallet, and engaging with multiple contracts. This bonus rewards truly healthy agents.

---

### Pattern 5: "Gas War Casualty" (modifier: -10)

**Detection:** High gas price variance + high failure rate + many retry storms + gas efficiency dropping

**Logic:**
```python
if (gas_adaptation_index > 0.40 and    # gas prices swinging wildly
    failed_pct_24h > 15 and
    storm_events > 2 and
    d1_gas_eff < 60):
    cdp_modifier -= 10
```

**What it means:** The agent IS adapting its gas price (high variance), but it's losing. It's caught in gas wars — bidding up, getting outbid, retrying, failing. The gas efficiency is dropping because it's setting limits too high in desperation. This agent needs to implement better MEV protection or gas bidding strategy.

---

### Pattern 6: "Phantom Activity" (modifier: -8)

**Detection:** Agent URL responds (D3 up) + zero or near-zero on-chain activity in 7 days + infrastructure looks healthy

**Logic:**
```python
recent_7d = last_7_days(transactions)
if (D3 is not None and D3 >= 70 and
    len(recent_7d) < 3 and
    len(transactions) > 20):    # was active historically
    cdp_modifier -= 8
```

**What it means:** The agent's server is running and responding to health checks, but it's not actually doing anything on-chain. The service is up but the agent is idle. This catches "dashboard agents" that look alive but aren't operating. Only detectable when D3 is available.

---

### Pattern 7: "Recovery in Progress" (modifier: +3)

**Detection:** Recent failure rate lower than historical + nonce gaps closing + tx diversity increasing

**Logic:**
```python
recent = last_24_hours(transactions)
older = last_7_days_minus_24h(transactions)
if (len(recent) > 5 and len(older) > 10 and
    failure_rate(recent) < failure_rate(older) * 0.5 and
    nonce_gaps_recent < nonce_gaps_historical):
    cdp_modifier += 3
```

**What it means:** The agent was sick but is getting better. Failure rate is dropping, nonce management is improving. This prevents the score from staying low after an operator fixes the underlying issue — it rewards recovery.

---

## 4. Temporal Model

### The Cold Start Problem

First scan has no history. The AHS must still be useful.

| Scan | Temporal Component | Approach |
|------|-------------------|----------|
| 1st scan | None | Score based purely on current snapshot. Confidence: Low/Medium |
| 2nd scan | Delta from scan 1 | Trend indicator enabled. Score still snapshot-dominant |
| 3rd+ scan | Exponential moving average | Full temporal model active |

### Score Evolution Formula

```python
if scan_count == 1:
    AHS_final = AHS_current
elif scan_count == 2:
    AHS_final = AHS_current * 0.8 + AHS_previous * 0.2
else:
    # Exponential moving average with recency bias
    alpha = 0.6    # weight on current scan (higher = more responsive)
    AHS_final = AHS_current * alpha + AHS_ema_previous * (1 - alpha)
```

**Why alpha = 0.6:** We want the score to respond quickly to degradation (agent is sick NOW) but not overreact to a single bad scan. At alpha=0.6:
- A sudden drop from 90 to 30 would show: 90 → 54 → 39.6 (converges in ~3 scans)
- A recovery from 30 to 90 would show: 30 → 66 → 80.4 (converges in ~3 scans)

### Trend Indicator

Exposed in the API response. Requires 2+ scans.

```python
if scan_count < 2:
    trend = "new"          # insufficient data
elif AHS_current > AHS_previous + 5:
    trend = "improving"
elif AHS_current < AHS_previous - 5:
    trend = "declining"
else:
    trend = "stable"
```

The ±5 deadband prevents noise from causing "improving/declining" flicker.

### Degradation Speed

How quickly should the score reflect a problem?

**Design principle:** Fast down, slow up.

- **Degradation:** The alpha=0.6 means a sudden crisis shows 60% of its impact on the very next scan. This is intentional — if an agent goes from 90 to 20, the operator needs to know NOW, not in 3 scans.
- **Recovery:** Same alpha, but the score recovers at the same rate. This is acceptable because recovery IS fast once the problem is fixed. If we wanted slower recovery, we could use asymmetric alpha (0.6 down, 0.4 up), but this adds complexity without clear benefit in v1.

### Nth Scan Value

Each successive scan becomes more valuable because:

1. **Baseline comparison** — "Score dropped 15 points since last week" is more actionable than "Score is 65"
2. **Trend confidence** — 5 scans showing decline is much more alarming than 1 low score
3. **Anomaly detection** — with enough history, we can flag when a score deviates >2 sigma from the running mean
4. **Seasonal patterns** — weekly/monthly cycles become visible (e.g., agent runs hot on Mondays)

This is the subscription hook: first wash shows current state, ongoing washes show trajectory.

---

## 5. API Response Design

### Endpoint

```
POST /ahs/{address}
POST /ahs/{address}?agent_url=https://agent.example.com
```

Price: TBD (likely $1.00 — premium over /wash and /health)

### Response: What's Exposed

```json
{
  "status": "ok",
  "ahs": {
    "address": "0xd8dA...96045",
    "score": 72,
    "grade": "C",
    "grade_label": "Needs Attention",
    "confidence": "high",
    "trend": "declining",
    "dimensions": {
      "hygiene": {
        "score": 85,
        "grade": "B",
        "top_issue": "131 dust tokens cluttering wallet"
      },
      "behaviour": {
        "score": 58,
        "grade": "D",
        "top_issue": "Repeated failures to same contract (8 consecutive)"
      },
      "infrastructure": {
        "score": 91,
        "grade": "A",
        "top_issue": null
      }
    },
    "dimensions_scored": ["hygiene", "behaviour", "infrastructure"],
    "contributing_factors": [
      {
        "signal": "repeated_failure_patterns",
        "impact": "high",
        "description": "8 consecutive failures to 0x6bded...7891:0xb858183f"
      },
      {
        "signal": "gas_adaptation",
        "impact": "medium",
        "description": "Gas price variance is low (0.03) — agent may not be adapting to network conditions"
      },
      {
        "signal": "retry_storms",
        "impact": "medium",
        "description": "2 retry storm events detected in last 24h"
      }
    ],
    "patterns_detected": [
      {
        "name": "Stale Strategy",
        "severity": "high",
        "description": "Agent is repeatedly failing on the same contract interaction without adapting. Possible causes: revoked approval, removed liquidity, contract upgrade."
      }
    ],
    "recommendations": [
      "Investigate repeated failures to 0x6bded...7891 — 8 consecutive reverts suggest stale strategy",
      "Enable dynamic gas pricing — current gas price variance (0.03) indicates hardcoded values",
      "Clear 131 dust tokens to improve wallet hygiene"
    ],
    "scan_metadata": {
      "scan_id": "ahs_v1_abc123",
      "model_version": "AHS-v1.0",
      "transactions_analyzed": 847,
      "history_days": 42,
      "scan_timestamp": "2026-03-06T14:30:00Z",
      "previous_scan": "2026-02-28T10:15:00Z",
      "score_delta": -8
    },
    "next_scan_recommended": "3 days"
  }
}
```

### What's Hidden (Server-Side Only)

These are NEVER exposed in the API response:

| Hidden Element | Reason |
|----------------|--------|
| Exact dimension weights (0.25/0.45/0.30) | Core IP — prevents competitors from replicating the composite |
| Individual signal weights within dimensions | Prevents gaming — if you know gas_adaptation is 15%, you can fake it |
| CDP pattern thresholds (exact numbers) | Prevents gaming specific patterns |
| Alpha value for temporal EMA | Prevents predicting future scores |
| Raw signal scores (only dimension-level exposed) | Keeps the scoring internals opaque |
| CDP modifier amount (only pattern name shown) | Hides how much each pattern affects score |

**Design principle:** Show WHAT is wrong (contributing factors, patterns). Hide HOW MUCH each thing affects the score. Users get actionable information. Competitors can't reverse-engineer the model.

### Model Versioning

```
X-AHS-Model-Version: AHS-v1.0
```

Returned as both a response header and in `scan_metadata.model_version`.

When we update the model (AHS-v1.1, v2.0):
- Scores from different model versions are NOT directly comparable
- The `score_delta` field only computes delta against scans from the same major version
- The API response includes `model_version` so callers can track version changes
- Major version bumps (v1→v2) reset the temporal EMA — new baseline needed

---

## 6. Data Storage Considerations

### What We Need to Persist Per Scan

For the temporal model to work, we need to store a minimal scan record:

```json
{
  "address": "0x...",
  "scan_id": "ahs_v1_abc123",
  "model_version": "AHS-v1.0",
  "timestamp": "2026-03-06T14:30:00Z",
  "ahs_score": 72,
  "d1_score": 85,
  "d2_score": 58,
  "d3_score": 91,
  "cdp_modifier": -10,
  "confidence": "high",
  "tx_count_at_scan": 847,
  "ema_score": 74.2
}
```

**Size:** ~300 bytes per scan. At 1000 scans/day = 300KB/day = ~10MB/month. Trivial.

### Storage Options (No Database Initially)

#### Option A: Signed Score Token (Stateless — Recommended for v1)

The caller passes back a `previous_score_token` from their last scan. The token is a signed JWT containing the minimal scan record.

```python
import jwt

def create_score_token(scan_record: dict) -> str:
    return jwt.encode(scan_record, SECRET_KEY, algorithm="HS256")

def verify_score_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return None
```

**API usage:**
```
POST /ahs/0x1234...
Body: { "previous_score_token": "eyJ..." }
```

**Pros:** Zero storage, no database, no privacy concerns (data lives with the caller).
**Cons:** Caller must store and pass the token. Lost token = cold start. Can't query historical data server-side.

#### Option B: Redis with TTL (Lightweight Persistence)

Store scan records in Redis with a 90-day TTL. Key = `ahs:{address}:latest`.

**Pros:** Automatic history without caller effort. Enables server-side trend analysis.
**Cons:** Requires Redis (Railway has a Redis addon). Stores wallet health data (privacy).

#### Option C: SQLite (Embedded, No Infrastructure)

Single SQLite file for scan history. Works on Railway with persistent volume.

**Pros:** SQL queries for analytics, no external dependency.
**Cons:** Persistent volume needed, single-writer limitation, no horizontal scaling.

### Recommendation for v1

**Use Option A (signed tokens) for launch.** It requires zero infrastructure changes and respects user privacy. The response includes a `score_token` field that the caller can store and pass back.

**Migrate to Option B (Redis) when:**
- We want server-side trend analytics
- We want to show "average AHS for agents in this category"
- We want to build a leaderboard or benchmarks

### Privacy Implications

| Storage Model | Privacy Risk | Mitigation |
|---------------|-------------|------------|
| Signed tokens | None — data stays with caller | N/A |
| Redis/SQLite | We store wallet health profiles | Hash addresses, TTL expiry, no PII beyond public address |

Public blockchain addresses are not PII under most frameworks (they're public data). However, correlating health scores over time could reveal operational patterns. If we move to server-side storage:
- Store address as keccak256 hash (can still lookup by address, can't enumerate)
- 90-day TTL on all records
- No cross-address correlation stored
- Clear data retention policy in docs

---

## 7. Implementation Roadmap

### Phase 1: D1 + D2 Core (Can Build Now)

Everything needed is already fetched by the existing endpoints.

| Component | Effort | Dependency |
|-----------|--------|------------|
| D1 score calculator | Low | Reuses `analyze_wash()` output |
| D2 signal extraction from txlist | Medium | New analysis on existing `fetch_transactions()` data |
| D2 scoring functions (8 signals) | Medium | Pure computation, no new data sources |
| CDP pattern detector (patterns 1-5) | Medium | Combines D1+D2 signals |
| Composite score + grade | Low | Weighted sum |
| Signed score token (JWT) | Low | `PyJWT` package |
| Pydantic models + endpoint | Low | Standard FastAPI pattern |
| x402 route + coupon route | Low | Copy from existing endpoints |

**Estimated total:** ~400 lines of new code in monitor.py + ~100 in api.py

### Phase 2: D3 Infrastructure Probing

| Component | Effort | Dependency |
|-----------|--------|------------|
| HTTP probe framework (async) | Medium | `httpx` (already in requirements) |
| 5 probe functions | Medium | Network access to agent URLs |
| D3 score aggregation | Low | Weighted sum of probe results |
| CDP patterns 6-7 (require D3) | Low | D3 signals available |
| SSRF protection for agent_url | Medium | Must validate URL is not internal/private |

**Key risk:** SSRF — we must validate that `agent_url` is a public endpoint, not `localhost`, `169.254.x.x`, or internal RFC 1918 ranges.

### Phase 3: Temporal Model

| Component | Effort | Dependency |
|-----------|--------|------------|
| JWT score token generation | Low | Phase 1 complete |
| Token verification + EMA calculation | Low | Phase 1 complete |
| Trend indicator | Low | 2+ scans for same address |
| Delta reporting | Low | Previous scan data available |

### Phase 4: Server-Side History (Future)

| Component | Effort | Dependency |
|-----------|--------|------------|
| Redis integration | Medium | Railway Redis addon |
| Scan record persistence | Low | Redis client |
| Historical trend API | Medium | New endpoint |
| Benchmark/percentile data | High | Enough scan volume to be meaningful |

---

## 8. Competitive Moat Analysis

### Why This Is Hard to Replicate

#### 1. Composite Weighting Is Opaque
The exact weights (D1=0.25, D2=0.45, D3=0.30) and signal weights within dimensions are server-side only. A competitor would need to reverse-engineer the model from API responses, which only expose dimension-level scores and contributing factors — not the math behind them.

#### 2. Cross-Dimensional Patterns Are Novel
No existing wallet analytics product detects "Zombie Agent" or "Cascading Infrastructure Failure" patterns. These require combining hygiene signals with behavioural analysis with infrastructure probing — three domains that are currently served by separate, unrelated tools.

#### 3. Temporal Data Compounds
Every scan makes the next scan more valuable. After 10 scans, AHM knows:
- The agent's normal operating profile (baseline)
- Whether current state is anomalous (deviation from baseline)
- The trajectory (improving/declining trend over weeks)
- Seasonal patterns (high-load periods, maintenance windows)

A competitor starting from scratch has zero history. They can replicate the formula but not the data. This is the same moat FICO has — the algorithm is known, but the credit history database is irreplaceable.

#### 4. Behavioural Signals Require Domain Expertise
Knowing that gas price variance < 0.05 indicates a hardcoded gas strategy requires understanding agent architecture. Knowing that 3+ identical txs in a 5-minute window is a retry storm requires understanding agent failure modes. These heuristics come from deep understanding of autonomous agent operations, not just blockchain data analysis.

#### 5. Infrastructure Probing Creates a Network Effect
As more agents register their `agent_url` for D3 scoring, AHM builds a directory of agent service endpoints. This enables future features:
- Agent-to-agent trust scores ("I'm about to transact with agent X — is it healthy?")
- Ecosystem health dashboards
- Downtime alerting across the agent network

#### 6. The Model Improves Over Time
As we collect more scans, we can:
- Tune weights based on which signals are most predictive of actual agent failures
- Add new patterns discovered in production data
- Build ML models trained on real agent health trajectories
- Publish percentile benchmarks ("Your agent is in the 72nd percentile")

The first-mover with the most data wins. This is why shipping v1 fast matters more than shipping v1 perfectly.

### What Competitors CAN Replicate

| Replicable | Not Replicable |
|-----------|----------------|
| Dust/spam token detection | Exact composite weights |
| Basic gas efficiency metrics | Cross-dimensional pattern library |
| Failed transaction counting | Temporal baseline per address |
| Nonce gap detection | Behavioural signal heuristics (thresholds) |
| HTTP availability probes | Historical scan data (requires adoption) |

### The Moat Summary

**Short-term moat (0-6 months):** First-to-market with agent-specific health scoring. No one else is doing D2 behavioural analysis.

**Medium-term moat (6-18 months):** Historical scan data + tuned model weights based on production data. Competitors can copy the approach but start with zero calibration.

**Long-term moat (18+ months):** Network effects from agent directory (D3), benchmark database from thousands of scans, and potential ML models trained on real failure/recovery data.

---

## Appendix A: Signal Interaction Matrix

Shows which signals reinforce or contradict each other, guiding CDP design.

```
                    GasAdapt  RetryStorm  Diversity  Breadth  Timing  Failures  NonceMgmt
GasAdapt               —        CONTRA     REINF     REINF    WEAK     CONTRA     WEAK
RetryStorm          CONTRA         —       CONTRA    CONTRA   REINF    REINF      REINF
Diversity            REINF      CONTRA        —       REINF    WEAK     WEAK      WEAK
Breadth              REINF      CONTRA      REINF       —      WEAK     WEAK      WEAK
Timing               WEAK       REINF       WEAK      WEAK      —      REINF     REINF
Failures             CONTRA     REINF       WEAK      WEAK    REINF      —        REINF
NonceMgmt            WEAK       REINF       WEAK      WEAK    REINF    REINF        —
```

**REINF** = both being bad reinforces the problem (multiplicative risk).
**CONTRA** = both being bad in opposite directions suggests different root causes.
**WEAK** = low correlation.

---

## Appendix B: Example Score Walkthrough

**Wallet:** Agent trading bot, 500 txs over 30 days.

| Signal | Raw Value | Signal Score |
|--------|-----------|-------------|
| **D1: Hygiene** | | |
| Dust tokens | 8 | 88 |
| Spam tokens | 3 | 94 |
| Gas efficiency | 72% | 100 |
| Failed rate (24h) | 12% | 64 |
| Nonce gaps | 1 | 85 |
| **D1 weighted** | | **87.3** |
| **D2: Behaviour** | | |
| Repeated failures | 6 consecutive | 42 |
| Gas adaptation | 0.08 | 54 |
| Nonce mgmt quality | 0 persistent | 100 |
| Timing regularity | CV=1.2 | 100 |
| Tx diversity | 0.04 | 50 |
| Retry storms | 1 event | 60 |
| Contract breadth | 5 contracts | 70 |
| Activity gaps | ratio 8 | 70 |
| **D2 weighted** | | **63.1** |
| **D3: Infrastructure** | (no agent_url) | **N/A** |

**Composite (2D mode):**
```
AHS = 0.30 * 87.3 + 0.70 * 63.1 = 26.19 + 44.17 = 70.36
```

**CDP check:**
- Repeated failures (6) + low diversity (0.04) + low gas adaptation (0.08) → "Stale Strategy" detected → -10

**Final AHS:** 70.36 - 10 = **60** (Grade: C — Needs Attention)

**Trend:** First scan → trend = "new"

**Confidence:** 500 txs, 30 days history, 2D only → "Medium"
