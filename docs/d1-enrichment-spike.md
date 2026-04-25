# D1 Enrichment Spike — Wallet Risk Intelligence for Solvency Scoring

**Version:** Spike v0.1
**Date:** 2026-04-13
**Status:** Research / Design — no implementation
**Classification:** Proprietary — do not commit to public repo

---

## 1. Objective

Enrich the D1 (Wallet Hygiene) dimension of the AHS scoring model with signals
that detect high-risk wallet behaviour: mixer interactions, sanctioned address
associations, dark web links, and known illicit activity.

The current D1 scores wallet cleanliness (dust, spam, gas efficiency, failed
txs, nonce gaps) but has **zero awareness of whether a wallet interacts with
sanctioned, criminal, or mixer-associated addresses**. A wallet can score D1=95
while regularly transacting with OFAC-sanctioned entities or Tornado Cash
intermediaries.

### Design Constraints

- No code changes in this spike — design only
- All 232 existing tests must remain passing
- Target free or very-low-cost sources appropriate for current stage
- Must work on Base chain (primary) with Ethereum fallback

---

## 2. Source Analysis

### 2.1 Chainalysis Free Sanctions Screening API

**What it provides:** REST API that checks whether a cryptocurrency address
appears on any sanctions list (primarily OFAC SDN). Returns entity name,
description, and OFAC listing URL when a match is found.

**Endpoint:**
```
GET https://public.chainalysis.com/api/v1/address/{address}
Header: X-API-Key: <free-api-key>
```

**Response (sanctioned):**
```json
{
  "identifications": [
    {
      "category": "sanctions",
      "name": "LAZARUS GROUP",
      "description": "North Korean state-sponsored...",
      "url": "https://home.treasury.gov/..."
    }
  ]
}
```

**Response (clean):** `{ "identifications": [] }`

| Attribute | Detail |
|-----------|--------|
| Cost | Free — register at go.chainalysis.com |
| Rate limits | Not publicly documented; intended for production use |
| Coverage | All chain types including Ethereum and Base |
| Data freshness | Maintained by Chainalysis; reflects current OFAC SDN list |
| Auth | API key via free registration form |
| Latency | Single REST call, typically <500ms |

**Strengths:**
- Zero cost, production-grade sanctions screening
- Single HTTP call per address — trivial to integrate
- Returns structured entity metadata (not just boolean)
- Case-sensitive address matching (no normalization bugs)

**Weaknesses:**
- Sanctions only — no coverage of non-sanctioned scams, phishing, or mixers
- No SLA or uptime guarantee on free tier
- Rate limits undocumented — could be throttled without warning

---

### 2.2 Chainalysis Sanctions Oracle (On-Chain Smart Contract)

**What it provides:** A `view` function on a deployed smart contract that
returns `true/false` for whether an address is sanctioned.

**Contract:** `0x40C57923924B5c5c5455c48D93317139ADDaC8fb` (same address on
Ethereum, Polygon, BSC, Avalanche, Optimism, Arbitrum, Fantom, Celo, Blast)

**Base chain:** `0x3A91A31cB3dC49b4db9Ce721F50a9D076c8D739B`

**Interface:**
```solidity
function isSanctioned(address addr) external view returns (bool)
```

| Attribute | Detail |
|-----------|--------|
| Cost | Free — `eth_call` (view function, no gas) |
| Rate limits | Bounded only by RPC provider rate limits |
| Coverage | OFAC/EU/UN sanctions; Ethereum + 9 other chains including Base |
| Data freshness | Updated by Chainalysis; no published SLA |
| Auth | None — permissionless smart contract call |
| Latency | Single RPC call, typically <200ms |

**Strengths:**
- Completely permissionless — no API key, no registration
- Works on Base (our primary chain) via dedicated contract address
- Boolean response is unambiguous — no parsing needed
- Can be called alongside existing RPC calls with minimal overhead

**Weaknesses:**
- Boolean only — no entity metadata (who is sanctioned, why)
- Sanctions only — same coverage limitation as the REST API
- Depends on Chainalysis maintaining the contract (no contractual obligation)

---

### 2.3 OFAC SDN List (Self-Hosted)

**What it provides:** The authoritative US Treasury sanctions list, downloadable
in XML/CSV format, containing digital currency addresses for sanctioned
entities.

**Download URL:** `https://www.treasury.gov/ofac/downloads/sanctions/1.0/sdn_advanced.xml`

**Current coverage:** ~1,245 total crypto addresses across all chains; ~91
Ethereum addresses across 37 sanctioned entities (Lazarus Group, Tornado Cash
deployers, etc.).

| Attribute | Detail |
|-----------|--------|
| Cost | Free — US government public data |
| Rate limits | None (download a file) |
| Coverage | OFAC SDN only; ~91 ETH addresses |
| Data freshness | Updated irregularly (days to weeks); no fixed schedule |
| Auth | None |
| Format | XML (`sdn_advanced.xml`); field type = `"Digital Currency Address - ETH"` |

**Parsing tools:**
- [0xB10C/ofac-sanctioned-digital-currency-addresses](https://github.com/0xB10C/ofac-sanctioned-digital-currency-addresses) — Python script that outputs per-chain TXT/JSON files
- [ultrasoundmoney/ofac-ethereum-addresses](https://github.com/ultrasoundmoney/ofac-ethereum-addresses) — produces `data.csv` with entity names

**Strengths:**
- Authoritative source — this IS the sanctions list, not a derivative
- Zero external dependency at query time (local HashSet lookup)
- No rate limits, no API keys, no network calls during scoring
- Enables offline/air-gapped compliance checking

**Weaknesses:**
- Only ~91 ETH addresses — very small dataset
- Requires periodic re-download and re-parse (cron job or startup hook)
- XML parsing adds build complexity
- Does not cover non-OFAC risks (scams, mixers, exploits)

---

### 2.4 Forta Network

**What it provides:** Decentralized threat detection network with community-built
detection bots. Alerts cover mixer interactions, exploit detection, scam
activity, sanctions violations, and anomaly detection. Data is accessible via a
GraphQL API.

**Endpoint:** `https://api.forta.network/graphql`

**Key queries:**
- `alerts` — fetch security alerts filtered by address, chain, severity, bot ID
- `labels` — query threat intelligence labels on addresses (e.g., "scammer",
  "phishing", "sanctioned") with confidence scores (0.0–1.0)

**Relevant detection bots:**

| Bot | ID (truncated) | Coverage |
|-----|-----------------|----------|
| Chainalysis Sanctioned Addresses | `0x9a8134...` | OFAC sanctions via on-chain oracle |
| Tornado Cash Funded Account | `0x617c35...` | Mixer-funded account interactions |
| Exploiter Addresses | `0x0e8298...` | Known exploiter address detection |
| Attack Detector Feed | `0x80ed80...` | Full attack lifecycle (funding → exploit → laundering) |
| Scam Detector | `0x1d646c...` | Phishing, address poisoning, rug pulls |

| Attribute | Detail |
|-----------|--------|
| Cost | General Plan: 250 FORT/month (~$3–5/month at current prices) |
| Free trial | 1 month, by application |
| Rate limits | "Unlimited API calls" on both plans (not independently verified) |
| Coverage | Ethereum, Base (not listed), Polygon, BSC, Arbitrum, Optimism + others |
| Data freshness | Near-real-time — alerts emitted seconds after block confirmation |
| Auth | API key required (wallet-authenticated via Forta App) |

**Strengths:**
- Broadest signal coverage of all sources: sanctions + mixers + exploits + scams
- Labels query returns confidence scores — enables nuanced scoring
- Near-real-time data freshness
- Very low cost (~$3–5/month)
- Community-maintained bots evolve with new threat vectors

**Weaknesses:**
- **Base chain support is unclear** — Forta lists Ethereum, Polygon, BSC, etc. but Base is not in the documented chain list. This is a critical gap to validate before committing.
- Requires FORT token for payment (adds operational complexity)
- GraphQL API is more complex to integrate than REST
- Premium feeds (Attack Detector, Scam Detector) cost extra
- Bot quality varies — community-maintained, not Chainalysis-grade
- Alert volume per address could be noisy; requires filtering logic

---

### 2.5 Etherscan Labels API

**What it provides:** Curated address labels (exchange, mixer, phish-hack,
OFAC-sanctioned, scam) maintained by Etherscan's team.

| Attribute | Detail |
|-----------|--------|
| Cost | **$899/month minimum** (Pro Plus tier for single lookups) |
| Rate limits | 2 calls/second hard cap on label endpoints |
| Coverage | ~45,000+ labeled addresses across EVM chains |
| Auth | API key (paid tier) |

**Verdict: Not viable for current stage.** The $899/month minimum for label
API access is prohibitive. The free Etherscan API tier provides zero access to
label/tag data. Noted for future reference only.

**Partial workaround:** The community-scraped dataset at
[brianleect/etherscan-labels](https://github.com/brianleect/etherscan-labels)
provides ~45,000 labeled addresses in JSON format, but the data is from mid-2023
and the scraper is currently broken. Could be used as a supplementary static
dataset but should not be relied upon as a primary source.

---

### 2.6 Supplementary Open-Source Blacklists

These are free, community-maintained datasets that can supplement the primary
sources above:

| Dataset | Maintained By | Coverage | Format | Update Frequency |
|---------|---------------|----------|--------|------------------|
| [ScamSniffer scam-database](https://github.com/scamsniffer/scam-database) | ScamSniffer | Phishing addresses + domains | JSON | Daily (7-day public delay) |
| [MyEtherWallet ethereum-lists](https://github.com/MyEtherWallet/ethereum-lists) | MEW | `addresses-darklist.json` — known phisher/scammer addresses | JSON | Community PRs |
| [CryptoScamDB blacklist](https://github.com/CryptoScamDB/blacklist) | CryptoScamDB | 6,000+ malicious URLs + profiles | JSON | Community PRs |

These provide scam/phishing coverage that OFAC and Chainalysis do not cover.
They can be bundled as static lookup sets, refreshed on deploy.

---

## 3. Integration Design

### 3.1 New D1 Signal: `wallet_risk_flags`

Add a new composite signal to D1 that aggregates risk intelligence from
multiple sources into a single score component.

**Proposed D1 weight redistribution:**

| Signal | Current Weight | Proposed Weight | Delta |
|--------|---------------|-----------------|-------|
| Dust token count | 0.15 | 0.12 | -0.03 |
| Dust total value | 0.05 | 0.04 | -0.01 |
| Spam token count | 0.20 | 0.16 | -0.04 |
| Gas efficiency | 0.25 | 0.22 | -0.03 |
| Failed tx rate | 0.20 | 0.17 | -0.03 |
| Nonce gap count | 0.15 | 0.12 | -0.03 |
| **Wallet risk flags** | **—** | **0.17** | **+0.17** |
| **Total** | **1.00** | **1.00** | **0.00** |

**Rationale for 0.17 weight:** Risk flags are a hard disqualifier — a
sanctioned address should devastate the D1 score regardless of how clean the
wallet is otherwise. At 0.17 weight, a sanctioned address (risk_flags score = 0)
reduces D1 by 17 points, which combined with the CDP modifier (see 3.3) creates
the intended severity.

### 3.2 Risk Flag Scoring Logic

```python
def calculate_risk_flags_score(address: str, risk_data: RiskData) -> int:
    """
    Returns 0-100 score. Lower = more risk flags detected.

    Severity tiers (not cumulative — worst match wins):
      - OFAC sanctioned:           score = 0   (hard zero)
      - Chainalysis sanctioned:    score = 0   (hard zero)
      - Known exploiter:           score = 10
      - Mixer-funded (high conf):  score = 20
      - Mixer-funded (low conf):   score = 40
      - Scam/phishing flagged:     score = 30
      - No flags:                  score = 100
    """

    if risk_data.ofac_sanctioned:
        return 0

    if risk_data.chainalysis_sanctioned:
        return 0

    if risk_data.is_known_exploiter:
        return 10

    if risk_data.scam_flagged:
        return 30

    if risk_data.mixer_funded:
        confidence = risk_data.mixer_confidence or 0.5
        if confidence >= 0.7:
            return 20
        else:
            return 40

    return 100
```

**Design decision — worst-match-wins vs. cumulative penalties:**
We use worst-match-wins because risk flags are categorical, not additive. An
address that is both sanctioned AND mixer-funded is not "twice as bad" — it's
sanctioned, full stop. The scoring reflects severity tiers, not accumulation.

### 3.3 New CDP Pattern: "Sanctioned Counterparty" (modifier: -20)

A new cross-dimensional pattern that fires when a wallet has transacted with
sanctioned or high-risk addresses.

```python
# CDP Pattern 8: "Sanctioned Counterparty"
# Fires when wallet has interacted with sanctioned/high-risk addresses
# AND continues to do so (not a one-time historical accident)

if (risk_flags_score <= 20 and
    recent_sanctioned_interactions > 0):   # interactions in last 7 days
    cdp_modifier -= 20
```

**Why -20 (the most severe CDP):** Interacting with sanctioned addresses is
not a software bug — it's a compliance/legal red flag. The -20 modifier can
push a wallet from Grade C to Grade E, which is the intended severity for
active sanctions violations.

### 3.4 Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    AHS Scan Request                      │
│                  POST /ahs/{address}                     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   Existing D1 Flow    │
         │  (dust, spam, gas,    │
         │   fails, nonce)       │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────────┐
    │                │                    │
    ▼                ▼                    ▼
┌────────┐   ┌──────────────┐   ┌────────────────┐
│ OFAC   │   │ Chainalysis  │   │ Static         │
│ Local  │   │ Sanctions    │   │ Blacklists     │
│ Set    │   │ API          │   │ (scam/phish)   │
│        │   │              │   │                │
│ O(1)   │   │ 1 HTTP call  │   │ O(1) HashSet   │
│ lookup │   │ ~200-500ms   │   │ lookup         │
└───┬────┘   └──────┬───────┘   └───────┬────────┘
    │               │                   │
    └───────────────┼───────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  RiskData struct │
          │  (aggregate)     │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ risk_flags_score │
          │  (0-100)         │
          └────────┬────────┘
                   │
         ┌─────────┴──────────┐
         │   D1 Score Calc     │
         │  (existing signals  │
         │   + risk_flags)     │
         └─────────┬──────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   CDP Checker    │
         │  (pattern 8:     │
         │   Sanctioned     │
         │   Counterparty)  │
         └─────────────────┘
```

### 3.5 Integration with Existing Code

**File: `monitor.py`**

New additions:
- `RiskData` dataclass — aggregated risk flags from all sources
- `fetch_risk_data(address)` — orchestrates lookups across sources
- `calculate_risk_flags_score(address, risk_data)` — converts to 0-100
- Modify `calculate_d1_score()` to include `risk_flags` as 7th signal
- Add CDP pattern 8 to `detect_cdp_patterns()`

**File: `api.py`**

Modifications:
- Add `risk_flags` to `WashResult` and `AHSResult` response models
- Add `risk_data` to contributing factors in AHS response
- Add `sanctioned_counterparty` to patterns_detected when applicable

**New file: `risk_sources.py`** (or inline in `monitor.py`)

Contains:
- `OFACLocalSet` — loads and caches parsed OFAC addresses
- `chainalysis_check(address)` — async HTTP call to free API
- `static_blacklist_check(address)` — HashSet lookup against bundled lists
- Source-level caching (see section 4)

**No changes to existing signal calculations.** The risk_flags signal is
purely additive — it does not modify dust, spam, gas, failure, or nonce
scoring logic.

---

## 4. Data Freshness & Caching Strategy

### 4.1 Source-Level Caching

| Source | Cache Strategy | TTL | Rationale |
|--------|---------------|-----|-----------|
| OFAC local set | Load on startup, refresh daily | 24h | List updates irregularly (days/weeks); daily refresh is sufficient |
| Chainalysis API | Per-address LRU cache | 1h | Sanctions status changes rarely; 1h prevents hammering the free API |
| Static blacklists | Load on startup, refresh on deploy | Per-deploy | Updated via git; no runtime refresh needed |
| Forta labels (future) | Per-address LRU cache | 30min | Alerts are near-real-time; shorter TTL for fresher threat intel |

### 4.2 OFAC Refresh Implementation

```python
class OFACLocalSet:
    """
    Downloads and parses sdn_advanced.xml on startup.
    Refreshes every 24h via background task.
    Falls back to last-known-good set on download failure.
    """

    def __init__(self):
        self._addresses: set[str] = set()
        self._last_refresh: datetime | None = None
        self._xml_url = "https://www.treasury.gov/ofac/downloads/sanctions/1.0/sdn_advanced.xml"

    async def refresh(self):
        # Download XML
        # Parse for "Digital Currency Address - ETH" features
        # Normalize addresses to lowercase
        # Store in HashSet
        # Log count: "OFAC refresh: {n} ETH addresses loaded"
        pass

    def is_sanctioned(self, address: str) -> bool:
        return address.lower() in self._addresses
```

**Startup behaviour:**
1. Attempt to download and parse `sdn_advanced.xml`
2. If download fails, check for cached copy on disk (`data/ofac_eth.json`)
3. If no cached copy, log warning and continue with empty set (graceful degradation)
4. Schedule `asyncio` background task to refresh every 24h

### 4.3 Chainalysis API Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

# In-memory cache with 1h expiry, max 10,000 entries
_chainalysis_cache: dict[str, tuple[bool, datetime]] = {}

async def chainalysis_check(address: str) -> bool:
    cached = _chainalysis_cache.get(address.lower())
    if cached and (datetime.utcnow() - cached[1]) < timedelta(hours=1):
        return cached[0]

    # HTTP GET to public.chainalysis.com/api/v1/address/{address}
    # Parse response
    result = ...
    _chainalysis_cache[address.lower()] = (result, datetime.utcnow())
    return result
```

### 4.4 Scan Latency Impact

| Source | Lookup Time | When Called |
|--------|------------|-------------|
| OFAC local set | <1ms | Always (HashSet lookup) |
| Chainalysis API | ~200-500ms (uncached), <1ms (cached) | Always |
| Static blacklists | <1ms | Always (HashSet lookup) |
| **Total added latency** | **~200-500ms first call; <1ms cached** | |

This is acceptable. The existing `analyze_wash()` already makes multiple
Etherscan/Basescan API calls totaling 1-3 seconds. Adding 200-500ms for a
single Chainalysis call is marginal.

---

## 5. Rate Limit & Cost Considerations

### 5.1 Cost Summary

| Source | Monthly Cost | Notes |
|--------|-------------|-------|
| OFAC SDN list | $0 | Free US government data |
| Chainalysis Sanctions API | $0 | Free tier, registration required |
| Chainalysis Sanctions Oracle | $0 | Free view call (no gas) |
| Static blacklists (ScamSniffer, MEW, CryptoScamDB) | $0 | Open-source GitHub repos |
| Forta General Plan | ~$3-5/month in FORT | If/when we add Forta integration |
| **Total (Phase 1)** | **$0** | |
| **Total (Phase 2 with Forta)** | **~$3-5/month** | |

### 5.2 Rate Limit Risk Assessment

| Source | Known Rate Limit | Our Expected Volume | Risk |
|--------|-----------------|---------------------|------|
| Chainalysis API | Undocumented | ~100-500 unique addresses/day (with 1h cache) | **Low** — free tier is intended for production; 1h cache reduces calls dramatically |
| OFAC download | None | 1 download/day | **None** |
| Static blacklists | None | Loaded at startup | **None** |
| Forta API | "Unlimited" (General Plan) | ~100-500 queries/day | **Low** — but "unlimited" is unverified |

### 5.3 Degradation Strategy

If any external source is unavailable:

| Source Down | Behaviour | Impact on Score |
|-------------|-----------|-----------------|
| Chainalysis API unreachable | Use OFAC local set only; log warning | Slightly reduced coverage; sanctions still detected via local set |
| OFAC download fails | Use last-known-good cached set | Stale by at most 24h; acceptable given irregular update cadence |
| Forta API unreachable (future) | Skip Forta signals; score from remaining sources | Loses mixer/scam detection; sanctions still covered |
| All sources down | `risk_flags_score = 50` (neutral) + `confidence` reduced | Score is less meaningful but not misleading |

**Design principle:** Risk intelligence is additive. Failure of any single
source degrades coverage but does not break the scoring pipeline. The scan
still completes with remaining signals.

---

## 6. Recommendation: Implementation Order

### Phase 1 — Ship First (Estimated: ~100 lines of new code)

**Sources:** OFAC local set + Chainalysis Sanctions API + Static blacklists

**Rationale:**
1. **All free** — zero incremental cost
2. **Covers the highest-severity risk** — sanctioned addresses are the most
   legally dangerous interaction; detecting them is table-stakes compliance
3. **Minimal integration complexity** — one REST call + two HashSet lookups
4. **Fast** — adds <500ms to scan latency (uncached), <1ms cached
5. **OFAC + Chainalysis provide belt-and-suspenders** — if one source lags,
   the other catches it

This phase adds the `wallet_risk_flags` signal to D1 and the "Sanctioned
Counterparty" CDP pattern. It does NOT add mixer detection or Forta integration.

### Phase 2 — Mixer & Scam Detection (Estimated: ~150 lines)

**Source:** Forta Network (General Plan, ~$3-5/month)

**Rationale:**
1. Forta is the only free/cheap source for **mixer interaction detection** and
   **scam/phishing address flagging** — these are the gaps Phase 1 doesn't cover
2. The `labels` GraphQL query provides confidence scores that map naturally to
   our tiered risk scoring (high confidence → score 20, low confidence → score 40)
3. **Blocker to validate first:** Confirm Base chain support in Forta. If Base
   is not supported, this phase may need to be deferred or work only for
   Ethereum-hosted agents

**Pre-requisites:**
- Validate Forta Base chain support (critical — our primary chain)
- Register for 1-month free trial to test API before committing to FORT payment
- Determine which bot IDs to subscribe to (recommend: Chainalysis Sanctioned,
  Tornado Cash Funded, Exploiter Addresses as minimum set)

### Phase 3 — Counterparty Graph Analysis (Future)

**Not designed in this spike.** This would involve:
- For each address in the wallet's recent transaction history (`to` addresses
  from txlist), run risk checks against Phase 1+2 sources
- Score based on degree of separation from flagged addresses
- Significantly increases API call volume (N counterparties × M sources)
- Requires careful rate limiting and caching strategy

This is where enterprise-grade tools (see Section 7) become necessary.

---

## 7. Enterprise Upgrade Path (Future Reference)

The following are **not designed for in this spike** but are noted as the
named upgrade path when AHM reaches a stage where enterprise compliance
tooling is justified.

### TRM Labs

- **Product:** TRM Wallet Screening API
- **Coverage:** 1M+ flagged entities; sanctions, terrorism financing, ransomware,
  darknet markets, child exploitation, fraud, stolen funds, mixer interactions
- **Differentiator:** Proprietary entity clustering + cross-chain tracing
- **Pricing:** Enterprise contracts; estimated $15,000-50,000+/year
- **Integration:** REST API with risk scores, entity labels, and investigation data
- **When to evaluate:** When AHM processes >10,000 scans/month or has enterprise
  clients with compliance requirements

### Scorechain

- **Product:** Scorechain Blockchain Analytics API
- **Coverage:** AML risk scoring, sanctions screening, transaction monitoring
- **Differentiator:** European-focused compliance (MiCA, AMLD); multi-chain
  coverage including Bitcoin, Ethereum, and stablecoins
- **Pricing:** Enterprise contracts; pricing not public
- **Integration:** REST API with risk scores and compliance reports
- **When to evaluate:** When AHM needs European regulatory compliance (MiCA)
  or has EU-based enterprise clients

### Chainalysis KYT (Know Your Transaction)

- **Product:** Real-time transaction monitoring API
- **Coverage:** Full transaction-level AML risk scoring, not just address screening
- **Differentiator:** Industry standard for exchanges and custodians; deepest
  entity database; real-time alerts
- **Pricing:** Enterprise contracts; estimated $10,000+/year per seat
- **When to evaluate:** When AHM moves from address-level to transaction-level
  risk assessment

### Upgrade Decision Framework

| Signal | Action |
|--------|--------|
| >10,000 AHS scans/month | Evaluate TRM Labs or Chainalysis KYT |
| Enterprise client requires compliance cert | Evaluate TRM Labs (SOC 2 compliant) |
| EU regulatory requirement (MiCA) | Evaluate Scorechain |
| Need transaction-level (not address-level) risk | Evaluate Chainalysis KYT |
| Current stage (<1,000 scans/month) | Free sources (Phase 1+2) are sufficient |

---

## 8. Open Questions

1. **Base chain coverage in Forta:** Forta's documented chain list does not
   include Base. Need to verify whether Base bots exist or if we'd need to
   rely on Ethereum-only alerts for mixer/scam detection. This is a Phase 2
   blocker.

2. **Chainalysis free API durability:** The free sanctions API has no published
   SLA. If Chainalysis deprecates it, we fall back to OFAC local set only
   (sanctions) and Forta (mixer/scam). The OFAC local set is a government
   resource and will not disappear.

3. **Counterparty depth:** Phase 1 only checks the wallet address itself. It
   does NOT check the wallet's counterparties (addresses it transacts with).
   A wallet that isn't sanctioned but regularly sends funds to Tornado Cash
   intermediaries would not be flagged until Phase 3. This is an accepted
   limitation for v1.

4. **False positive handling:** What happens when a legitimate agent wallet
   receives an unsolicited transfer from a sanctioned address? The current
   design would penalize the recipient. Consider adding a "direction" check:
   only penalize outgoing transactions to flagged addresses, not incoming.
   This requires txlist analysis and increases complexity.

5. **Weight calibration:** The proposed 0.17 weight for `risk_flags` in D1
   is a design estimate. After implementation, we should score a sample of
   known-good and known-bad wallets to validate that the weight produces
   expected grade distributions.

---

## 9. Summary

| Phase | Sources | Cost | Coverage | Complexity |
|-------|---------|------|----------|------------|
| **1 (Ship First)** | OFAC local + Chainalysis API + Static blacklists | $0/month | Sanctions + scam/phishing addresses | Low (~100 LOC) |
| **2 (Mixer/Scam)** | + Forta Network | ~$3-5/month | + Mixer interactions, exploit addresses, ML anomalies | Medium (~150 LOC) |
| **3 (Counterparty)** | Phase 1+2 applied to txlist counterparties | Same sources | + 1-hop counterparty risk | High (rate limit mgmt) |
| **Enterprise** | TRM Labs / Scorechain / Chainalysis KYT | $10K-50K+/year | Full AML/CTF compliance | Vendor integration |

**Recommendation:** Implement Phase 1 immediately. It's free, adds <500ms
latency, requires ~100 lines of new code, and covers the most legally critical
risk signal (sanctions). Phase 2 should follow after validating Forta's Base
chain support.
