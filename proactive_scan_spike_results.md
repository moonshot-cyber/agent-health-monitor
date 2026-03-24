# Proactive Ecosystem Scan Spike — ACP (agdp.io)

> Scanned: 2026-03-24 12:08 UTC
> Source: ACP API (`acpx.virtuals.io/api/agents`)
> Scan mode: 2D (D1+D2 only, free APIs, no Nansen)

## Spike Objective

Prove the end-to-end proactive scanning flow: discover agent wallets from an external registry, run AHS scans, store results in the database. This is the first step toward Priority 4 (Proactive Ecosystem Scanning) in the AHM backlog.

## Source Assessment

| Source | Status | Notes |
|--------|--------|-------|
| Virtuals ACP (agdp.io) | **Selected** | Free API, no auth, 40K+ agents with `walletAddress` field |
| x402scan | Blocked | All endpoints paywalled ($0.01/call via x402) |
| 402index.io | Viable (backup) | Free API but no payTo addresses — requires 2-step probe |
| Virtuals Protocol API | Blocked | Agents share TBA shards — per-agent AHS not meaningful |

## Discovery Summary

- **Total agents in ACP registry:** 40,505
- **Agents fetched (sorted by successfulJobCount:desc):** 20
- **Unique wallet addresses:** 20
- **Shared wallets (>1 agent per wallet):** 0
- **Max agents sharing one wallet:** 1
- **Owner/agent wallet overlap:** 0

**Wallet sharing assessment:** No wallet sharing detected. ACP agent wallets are independent — ideal for per-agent health scoring.

## AHS Scan Results

- **Wallets scanned:** 5
- **Average AHS:** 53.4
- **AHS range:** 29-70
- **Grade distribution:** C=3, D=1, E=1

### Grade Distribution

| Grade | Count | % | Score Range |
|-------|-------|---|-------------|
| A | 0 | 0.0% | 90-100 |
| B | 0 | 0.0% | 75-89 |
| C | 3 | 60.0% | 60-74 |
| D | 1 | 20.0% | 40-59 |
| E | 1 | 20.0% | 20-39 |
| F | 0 | 0.0% | 0-19 |

### Scanned Agents (by AHS)

| ACP ID | Name | Wallet | Jobs | Revenue | AHS | Grade | D1 | D2 | Patterns |
|--------|------|--------|------|---------|-----|-------|----|----|----------|
| 3673 | ASCII Artist | `0x1e1633...14d0` | 13,734 | $91,598 | 70 | C Needs Attention | 75 | 68 |  |
| 1048 | Wasabot | `0x5dfc18...6f5d` | 15,116 | $5,924 | 61 | C Needs Attention | 57 | 63 |  |
| 110 | Cybercentry | `0x228f70...bb63` | 13,462 | $3,153 | 61 | C Needs Attention | 58 | 62 |  |
| 292 | BasisOS | `0xa908b2...c7f9` | 12,289 | $0 | 46 | D Degraded | 66 | 38 |  |
| 1638 | Daredevil | `0x718281...a5f8` | 12,059 | $11,397 | 29 | E Critical | 74 | 10 |  |

## Database Storage

- All 5 scan results persisted to `ahm_history.db` via `db.log_scan()`
- Source: `acp_proactive_scan`
- Registry tracking: `registries` column updated with `acp_proactive_scan`
- Labels format: `ACP #<id> — <name>`

## Spike Conclusions

### What worked

1. **ACP API is the best discovery source** — free, no auth, returns wallet addresses directly
2. **End-to-end flow proven** — discovery → dedup → scan → store → report
3. **Existing AHS engine works unchanged** — `calculate_ahs()` handles ACP wallets identically to ERC-8004
4. **`db.log_scan()` cross-registry tracking** — ACP scans integrate cleanly with existing schema

### Next steps to full pipeline

1. **Scheduled scanning** — cron/scheduler to run ACP discovery + AHS scans periodically
2. **402index.io integration** — second discovery source via payTo address extraction
3. **Dedup across registries** — merge ACP + ERC-8004 + 402index wallet sets
4. **Public Health Dashboard** — aggregate stats from all scanned wallets (backlog P4 deliverable)
