# Off-Chain Agent Scanning Feasibility Report

**Date**: 2026-05-05
**Spike**: Off-chain agent identification via x402 facilitator-side observation
**Status**: Feasibility assessment

---

## Executive Summary

This spike tested whether software agents paying via x402 but not registered on
AHM's four on-chain registries (ACP/Virtuals, Olas, Celo, ERC-8004) can be
reliably identified and characterised from facilitator-side observation alone.

**Key findings:**
- **27** unique buyer wallets observed in x402 payment data
- **2** (7.4%) matched existing registry coverage
- **25** (92.6%) are unregistered — potential off-chain agents
- Signal quality: Moderate — see details below

---

## Data Sources Accessed

| Source | Accessible | Notes |
|--------|-----------|-------|
| Base Blockscout API | Yes (free, no auth) | Transaction and token transfer data |
| AHM known_wallets DB | Yes (local) | 2 wallets from 4 registries |
| ERC-8004 IdentityRegistry | Yes (on-chain, free RPC) | balanceOf check per wallet |
| Olas ServiceRegistryL2 | Yes (on-chain, free RPC) | balanceOf check per wallet |
| Bitquery GraphQL | No (requires API key registration) | Would provide richer analytics |
| x402scan SQL API | No (undocumented, frontend-only) | Would be ideal primary source |
| Dune Analytics | Partial (dashboard exists, API requires key) | x402-analytics dashboard by hashed_official |
| agentic.market | Yes (service discovery only) | No buyer/transaction data exposed |
| 402index.io | Yes (service directory only) | 54K endpoints indexed, no buyer data |

- Fetched 1669 transactions from Blockscout (x402 proxy contract)

---

## Methodology

1. **Data collection**: Queried Base Blockscout for token transfers involving
   the x402ExactPermit2Proxy contract (`0x402085c248EeA27D92E8b30b2C58ed07f9E20001`). This captures
   USDC settlements where the proxy mediates permit-based transfers from
   buyer to seller.

2. **Registry cross-reference**: Each buyer wallet was checked against:
   - AHM's `known_wallets` table (968 wallets from ACP, Olas, Celo, ERC-8004 scans)
   - ERC-8004 IdentityRegistry on Base (`balanceOf > 0` check)
   - Olas ServiceRegistryL2 on Base (`balanceOf > 0` check)

3. **Candidate selection**: 10 candidates selected for diversity in transaction
   count, service consumption breadth, and spend volume.

4. **Characterisation**: Each candidate profiled for spend, cadence, service
   mix, and autonomy confidence.

---

## Ten Candidate Off-Chain Agents

### Candidate 1: `0x01440cd9c0ea48fce2f8781ffef0bf853efe9fb9`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $920000.00 |
| Transaction count | 2 |
| Time window | 2026-04-03 to 2026-04-03 |
| Distinct services consumed | 1 |
| Cadence pattern | sporadic |
| Inferred purpose | light consumer (insufficient data) |
| Confidence | **ambiguous** |

**Services consumed:**
- `0x8808029ec8b32d38c061ac1876a080c03f112db8`

### Candidate 2: `0x9e753c5c0051277c2a9600fcdf14e28eafd7a7db`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $12718.23 |
| Transaction count | 84 |
| Time window | 2026-03-18 to 2026-03-19 |
| Distinct services consumed | 1 |
| Cadence pattern | bursty |
| Inferred purpose | single-service consumer (dedicated integration) |
| Confidence | **likely_autonomous** |

**Services consumed:**
- `0x402c1246842f2cdbc8e0b98a67d7a59aae22b394`

### Candidate 3: `0x9dc7a139db0176c9ac6fe64dcdf78398149e2f58`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $7190.20 |
| Transaction count | 54 |
| Time window | 2026-04-11 to 2026-04-18 |
| Distinct services consumed | 2 |
| Cadence pattern | bursty |
| Inferred purpose | dual-service consumer |
| Confidence | **clearly_autonomous** |

**Services consumed:**
- `0x4ec4a3e5b6a442c86c7bc6313ebeac2393abf933`
- `0x166bf3f45b5b40738e243b68d20862dc15925be0`

### Candidate 4: `0xc2cc8d45e0e5690e103b098cad72f4475938d212`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $3000.00 |
| Transaction count | 3 |
| Time window | 2026-04-15 to 2026-04-15 |
| Distinct services consumed | 1 |
| Cadence pattern | sporadic |
| Inferred purpose | light consumer (insufficient data) |
| Confidence | **ambiguous** |

**Services consumed:**
- `0x4ec4a3e5b6a442c86c7bc6313ebeac2393abf933`

### Candidate 5: `0x89856fa8b52c63f5db6ef6e7cfa8994c8575d18b`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $1018.10 |
| Transaction count | 96 |
| Time window | 2026-03-29 to 2026-04-07 |
| Distinct services consumed | 1 |
| Cadence pattern | bursty |
| Inferred purpose | single-service consumer (dedicated integration) |
| Confidence | **likely_autonomous** |

**Services consumed:**
- `0x166bf3f45b5b40738e243b68d20862dc15925be0`

### Candidate 6: `0xc87763f4e1fe8a8fae04963d1122d2e1fac1d669`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $59.40 |
| Transaction count | 11 |
| Time window | 2026-03-27 to 2026-03-27 |
| Distinct services consumed | 1 |
| Cadence pattern | bursty |
| Inferred purpose | single-service consumer (dedicated integration) |
| Confidence | **likely_autonomous** |

**Services consumed:**
- `0x8b29dabd6fbb5a09dacbc7978eaed66a8540721d`

### Candidate 7: `0x9b837573e9ceb4a4e6309ffa2040b2c3d2978b35`

| Dimension | Value |
|-----------|-------|
| Wallet type | Smart contract |
| On-chain name | — |
| Total x402 spend | $8.57 |
| Transaction count | 857 |
| Time window | 2026-04-20 to 2026-04-23 |
| Distinct services consumed | 1 |
| Cadence pattern | bursty |
| Inferred purpose | single-service consumer (dedicated integration) |
| Confidence | **likely_autonomous** |

**Services consumed:**
- `0x158e90dd58fbe897ed8c244f472febee37283d00`

### Candidate 8: `0xf9d9d8a1ac99f9498878f1126b1d2ec3add9da69`

| Dimension | Value |
|-----------|-------|
| Wallet type | Smart contract |
| On-chain name | — |
| Total x402 spend | $3.00 |
| Transaction count | 1 |
| Time window | 2026-05-05 to 2026-05-05 |
| Distinct services consumed | 1 |
| Cadence pattern | single_tx |
| Inferred purpose | light consumer (insufficient data) |
| Confidence | **ambiguous** |

**Services consumed:**
- `0xd2abb4834f33fffd5586ca50dab478faddfd2965`

### Candidate 9: `0x28fb95d0cd49fe387ae9791758cc6aff208841d7`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $3.00 |
| Transaction count | 3 |
| Time window | 2026-03-08 to 2026-03-08 |
| Distinct services consumed | 1 |
| Cadence pattern | sporadic |
| Inferred purpose | light consumer (insufficient data) |
| Confidence | **ambiguous** |

**Services consumed:**
- `0xc7e38c5dec3deebe6b643d332db89a27c6df204d`

### Candidate 10: `0x68ed8b810427521a13bef83d455c2a5c2d8c63bc`

| Dimension | Value |
|-----------|-------|
| Wallet type | EOA |
| On-chain name | — |
| Total x402 spend | $2.10 |
| Transaction count | 6 |
| Time window | 2026-03-20 to 2026-03-20 |
| Distinct services consumed | 2 |
| Cadence pattern | sporadic |
| Inferred purpose | dual-service consumer |
| Confidence | **likely_autonomous** |

**Services consumed:**
- `0x2cd2a06b47bcd9bafdd0ec8c0f312d4bdf2df7dd`
- `0x3d54426879099855d2ae4bf5f81297454244ff98`

---

## Scope Limitations

This spike only captured settlements via the **Permit2 proxy path**
(`x402ExactPermit2Proxy`) on **Base mainnet**. The x402 protocol has three
settlement methods on EVM:

1. **EIP-3009 (transferWithAuthorization)** -- direct to the token contract, no
   proxy involved. These transactions are indistinguishable from regular USDC
   transfers without parsing transaction input data.
2. **Permit2 (via x402ExactPermit2Proxy)** -- routed through the proxy contract.
   **This is what we captured.**
3. **ERC-7710 (delegation)** -- via delegation manager contracts. Not yet widely
   deployed.

The x402 protocol has processed **165M+ transactions** across all chains and
methods. Our 1,669 transactions from 27 unique buyers represent only the Permit2
slice on Base. The actual unregistered off-chain agent population is likely much
larger.

Additionally, x402 operates on **Polygon, Arbitrum, World, and Solana** -- none
of which were checked in this spike.

### Non-USDC token amounts

Some transactions use ERC-20 tokens with 18 decimals (not USDC's 6). Dollar
amounts for these transactions are **token-denominated, not USD-equivalent**.
Candidate 1's $920K figure represents 920,000 tokens of an unknown ERC-20 --
the actual USD value is unknown without a price feed.

### Seller identification

Cross-referencing seller addresses with Blockscout reveals partial identity:

| Seller | Type | Identity |
|--------|------|----------|
| `0x158e90dd...` | SafeProxy contract | Production service behind multi-sig |
| `0x3d544268...` | Contract | ENS: `givemd.eth` |
| `0x8808029e...` | Contract | Unnamed |
| `0xd2abb483...` | Contract | Unnamed |
| `0xb26e0fF3...` | EOA | Unnamed (most common seller, 504+ txns) |

The most active seller (`0xb26e0fF3...`) received 504+ payments and is an EOA --
possibly a developer running an x402-enabled API server.

---

## Feasibility Verdict

### Is the workflow repeatable?

Yes -- the Blockscout API provides free, unauthenticated access to
x402 transaction data on Base. The pipeline (fetch, cross-reference, filter,
characterise) runs end-to-end in a single script. Rate limits are the main
constraint (~2 req/sec sustained for Blockscout, ~1 req/sec for free Base RPC).

### What fraction of x402-paying wallets are unregistered?

**~92.6%** (25/27) of observed buyer wallets had no presence in any of AHM's
four registries. However, **this sample is narrow**: only Permit2-path
settlements on Base, representing ~1,669 out of 165M+ total x402 transactions.
The percentage is directionally useful but the absolute count is too small to
generalise. The important signal is that the vast majority of x402 payers are
not in any agent registry -- they are off-chain agents or human developers
interacting with x402 APIs directly.

### Signal quality

Moderate: Unregistered wallets show diverse patterns — from single-transaction curiosity testers to steady multi-service consumers.
The noise floor is manageable — most unregistered wallets with 5+ transactions show clearly autonomous behaviour patterns.

Key signal/noise observations:
- **EOA vs smart contract**: Smart contract wallets (Safe multisigs, smart accounts)
  are more likely to be genuine autonomous agents
- **Transaction count**: Wallets with 10+ x402 transactions are almost certainly
  automated — human-driven x402 usage is rare
- **Service diversity**: Consuming 3+ distinct services strongly suggests an
  orchestrator agent rather than a human testing a single API
- **Cadence**: Steady or bursty patterns with sub-hour gaps indicate automation

### What's characterisable from facilitator data alone?

**Observable:**
- Wallet address and type (EOA vs contract)
- Total spend and transaction count
- Time window and cadence pattern
- Number and addresses of services consumed
- Whether the wallet is a smart account (Safe, ERC-4337)

**NOT observable (from facilitator data alone):**
- Agent name, description, or capabilities
- What the agent actually does with the API responses
- Whether the agent is autonomous or human-supervised
- The hosting infrastructure (Vercel, AWS, self-hosted)
- The agent's framework (LangChain, CrewAI, custom)
- Business context or operator identity

### What additional data sources would help?

1. **x402scan SQL API** (if access granted): Richer transaction analytics,
   server-side metadata, resource categorisation
2. **Bitquery GraphQL**: Cross-chain x402 data, historical depth, analytics
3. **Dune Analytics API**: The hashed_official/x402-analytics dashboard has
   curated queries; API access would enable programmatic use
4. **ENS / Basenames reverse resolution**: Map wallets to human-readable names
5. **Safe Transaction Service API**: For smart contract wallets, reveals
   signers and module configuration
6. **agentic.market seller metadata**: Cross-reference seller addresses with
   service categories to infer what buyers are consuming

---

## Failure Modes Encountered

1. **x402scan frontend-only**: The SQL API mentioned in x402scan's marketing is
   not publicly documented. The site renders via Next.js with client-side data
   fetching — no accessible REST API was found.

2. **Bitquery requires registration**: Free tier exists but requires account
   creation and API key. Not blocked, just a setup step.

3. **Blockscout pagination**: Token transfer data is paginated at 50 items/page
   with rate limiting. Collecting a full dataset requires patience.

4. **Registry cross-reference is one-directional**: ERC-8004 and Olas registries
   don't have reverse lookups (wallet → agent). We check `balanceOf` as a proxy,
   which catches NFT owners but may miss agents whose wallets differ from their
   registry owner addresses.

5. **Virtuals ACP check skipped**: The Virtuals API doesn't support wallet-based
   lookup. AHM's existing known_wallets DB covers ACP agents already scanned.

6. **EIP-3009 path invisible**: x402 payments settled via `transferWithAuthorization`
   go directly to the USDC contract, not through the proxy. These are
   indistinguishable from regular USDC transfers without decoding tx input.
   This path likely carries significant volume.

7. **Multi-chain gap**: Only Base was checked. Polygon, Arbitrum, World, and Solana
   x402 activity was not sampled.

---

## Recommended Next Steps

1. **Expand settlement coverage**: Add EIP-3009 (`transferWithAuthorization`)
   transaction decoding to capture the other major x402 settlement path. This
   likely represents the majority of x402 volume.
2. **Register for Bitquery free tier** -- enables GraphQL queries for deeper
   x402 transaction analytics across all chains, bypassing Blockscout rate limits.
3. **Contact x402scan team** about SQL API access -- they index all settlement
   methods and could be the single best data source.
4. **Multi-chain expansion**: Replicate the Permit2 proxy query on Polygon
   (`0x402085c2...` same address via CREATE2), Arbitrum, and other chains.
5. **Build a persistent wallet index** -- accumulate unregistered buyer wallets
   over time, re-score weekly for cadence analysis.
6. **Add ENS/Basename resolution** -- many agent operators register names.
7. **Cross-reference with Safe Transaction Service** -- identify multi-sig
   agent wallets and their governance structure.
8. **Explore ERC-7710 delegation traces** -- x402 V2 supports delegation-based
   payments, which are a strong autonomy signal.

---

## Scripts and Data

- Scanning script: `prompts/spikes/scripts/offchain_agent_scan.py`
- Candidate data: `prompts/spikes/scripts/offchain_candidates.json`
- This report: `prompts/spikes/off-chain-agent-feasibility.md`
