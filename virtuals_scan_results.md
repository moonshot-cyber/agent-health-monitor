# Virtuals ACP Agent Scan Results

> Scanned: 2026-03-17 14:50 UTC
> API: `https://api.virtuals.io/api/virtuals`

## Summary

- **Total Virtuals agents:** 22,195
- **Agents enumerated:** 1000
- **With ACP ID:** 100
- **Cross-registry overlap (already in DB):** 0
- **Unique wallets scanned (AHS):** 12
- **Average AHS:** 56.8
- **AHS range:** 50-78
- **Grade distribution:** B=1, D=11

## Grade Distribution

| Grade | Count | % | Score Range |
|-------|-------|---|-------------|
| A | 0 | 0.0% | 90-100 |
| B | 1 | 8.3% | 75-89 |
| C | 0 | 0.0% | 60-74 |
| D | 11 | 91.7% | 40-59 |
| E | 0 | 0.0% | 20-39 |
| F | 0 | 0.0% | 0-19 |

## Healthy Wallets (B+ Grade) — Case Studies

| Virtuals ID | Name | Symbol | Wallet | AHS | Grade | D1 | D2 | Holders | Agents Sharing |
|-------------|------|--------|--------|-----|-------|----|----|---------|----------------|
| 69 | Iona | IONA | `0x74d9cea5...` | 78 | B Good | 78 | 78 | 99,867 | 100 |
| 70 | Olyn | OLYN | `0xb01e1de4...` | 78 | B Good | 78 | 78 | 68,505 | 100 |

## All Scanned Wallets (by AHS, deduplicated)

| Virtuals ID | Name | Symbol | Wallet | AHS | Grade | D1 | D2 | Patterns | Agents Sharing |
|-------------|------|--------|--------|-----|-------|----|----|----------|----------------|
| 69 | Iona | IONA | `0x74d9cea5...` | 78 | B Good | 78 | 78 |  | 100 |
| 70 | Olyn | OLYN | `0xb01e1de4...` | 78 | B Good | 78 | 78 |  | 100 |
| 234 | Asuka Miyu | ASUKA | `0x6a8a8ec8...` | 58 | D Degraded | 75 | 50 |  | 100 |
| 238 | Carlos Ju | CARLOS | `0x4c11085a...` | 58 | D Degraded | 75 | 50 |  | 100 |
| 239 | Conrad Evans | CONRAD | `0x272ca619...` | 58 | D Degraded | 75 | 50 |  | 100 |
| 237 | Adrian Steele | ADRIAN | `0xab0c4a0b...` | 57 | D Degraded | 72 | 50 |  | 100 |
| 236 | Isabella | ISABEL | `0x4e25e8ce...` | 56 | D Degraded | 69 | 50 |  | 100 |
| 206 | Lady Tsunade | TSUNAD | `0xcccbd75d...` | 53 | D Degraded | 61 | 50 |  | 100 |
| 235 | Reiko-chan | REIKO | `0xa4bc6c07...` | 51 | D Degraded | 53 | 50 |  | 100 |
| 68 | Luna | LUNA | `0x979a8c37...` | 50 | D Degraded | 51 | 50 |  | 100 |

## Wallet Sharing Analysis

- **Unique wallets:** 10
- **Shared by 2+ agents:** 10
- **Max agents per wallet:** 100
- **Total agents enumerated:** 1000
- **Avg agents per wallet:** 100.0

## Key Finding: Shared Infrastructure Architecture

**Virtuals agents do NOT have independent wallets.** Unlike ERC-8004 agents (which each have their own deployer/operator wallet), Virtuals agents are token-based entities that share platform-level infrastructure:

- **10 TBA (Token Bound Account) shards** — each shared by exactly 100 agents
- **1 creator wallet** (`0x14059190...`) — the Virtuals protocol deployer, shared by all 300 agents that have a walletAddress
- **1 sentient wallet** (`0x0D177181...`) — shared by all 100 agents with sentient capabilities
- **0 cross-registry overlap** — no Virtuals wallets found in ERC-8004 or other registries

**Implications for AHM:**

1. **AHS scores reflect Virtuals protocol health, not individual agent health** — all 100 agents sharing a TBA get the same score
2. **Wallet-level scanning is not meaningful** for differentiating Virtuals agents — need token-level metrics (holders, volume, TVL) instead
3. **Cross-registry value is low** for Virtuals — their agents operate on shared infrastructure with no overlap to ERC-8004 ecosystem
4. **AHM's ERC-8004 scanning is more valuable** — those agents have independent wallets with real per-agent health signals
5. **Potential AHM feature**: A "Virtuals protocol health" endpoint that monitors the 10 TBA shards as infrastructure, not as individual agents
