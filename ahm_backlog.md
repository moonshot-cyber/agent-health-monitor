# AHM Backlog

> Single source of truth for Agent Health Monitor. Update at end of every session.

---

## Current State (as of Mar 17 2026)

- **13 endpoints** live on Base mainnet at agenthealthmonitor.xyz
- **ERC-8004 registered** — agentId #32328 on Base mainnet
- **Nansen integration** — 4 direct API connections (labels, counterparties, PnL, related wallets)
- **Listed on:** Virtuals ACP (11 offerings), x402scan, Bankr Skills, agdp.io, 8004scan.io
- **Stack:** FastAPI, x402 SDK v2, Nansen API, Blockscout API, Base Mainnet, Railway
- **Repo:** github.com/moonshot-cyber/agent-health-monitor
- **Calibration dataset:** 245 wallets scanned (24 seed + 26 ERC-8004 IDs 1-352 + 183 ERC-8004 IDs 30000+ + 12 Virtuals TBA shards)
- **Cross-registry tracking:** `registries` column in `known_wallets` (schema v2), tracks erc8004/virtuals/etc per wallet

---

## P1 — Active / In-Flight

### Distribution & Partnerships

- [ ] **KAMIYO (@kamiyoai)** — Singularity platform launching this week, has "hard risk checks" pre-agent-funding. AHM `/risk` and `/ahs` are natural pre-funding health verification layer. Reach out
- [ ] **@BaseHubHB** — Weekly Base launch curator, 2.5K+ views per post. Engage every week to get included in future roundups
- [ ] **FairScale (@fairscalexyz)** — Solana credibility scoring, complementary to AHM on Base. Monitor and engage

### Pending Response / Follow-up (Mar 16)

- [ ] **Daydreams** — DM sent Mar 16, flagged 38/E Critical "Stale Strategy" finding. Chase if no response by **Thu Mar 19**
- [ ] **PayAI** — DM sent Mar 16, shared signer wallet scores (51/D and 61/C). Chase if no response by **Thu Mar 19**
- [ ] **Bankr Skills PR #195** — Chased via X post on @bankrbot Mar 16 in addition to GitHub bumps. Chase again if no response by **Fri Mar 20**
- [ ] **OpenServ hackathon** — Submitted, awaiting selection decision. If selected, ship on OpenServ within 2 weeks. Prep needed: demo recording (Loom), add `issa-me-sush` as GitHub collaborator

### ERC-8004 Scan Outreach (added Mar 17)

Based on ERC-8004 registry scan of IDs 30000-30355 (183 wallets, avg AHS 59.5, range 38-86).

**Critical outreach targets (E/F grade):**
- [ ] **Zombie Agent wallet** `0xf7a199...1944` — AHS 38/E Critical, Zombie Agent pattern detected. 915 TXs in 70 days then stalled. Strong outreach candidate: "your wallet is healthy but your agent appears stalled"

**Healthy agent case studies (B+ grade) — potential co-marketing/testimonials:**
- [ ] **STN** (#30248) — AHS 86/B, top scorer. Owner `0x812928...df80`
- [ ] **Agent Seven** (#30267) — AHS 85/B, 706 days active (nearly 2 years)
- [ ] **Morning Person** (#30296) — AHS 82/B, 843 days active (2.3 years, longest)
- [ ] **Makanan** (#30304) — AHS 78/B, 171 TXs over 635 days, balanced D1/D2

**Recognizable projects discovered (potential partnerships):**
- [ ] **Questflow** — Mia's Alpha Hunter (#30354), AHS 74/C. Agent platform via api.questflow.ai
- [ ] **Zyfai** — 5 DeFi rebalancer agents, all 55-58/D. Bulk registration pattern
- [ ] **Con+ Audits** (#30237, #30263) — Auditing platform, AHS 65/C
- [ ] **Bob Suite** — 5 agents (BobDeFi, BobPrompt, BobCompliance, BobGrowth, BobThumbnail), same owner `0x955ab7...`

### Update all references to endpoint count and add /ahs/batch to documentation

Now that /ahs/batch is live, the following need updating:
- [ ] static/index.html — endpoint count shows "12 endpoints", update to 13
- [ ] static/ahm-roadmap.html — add /ahs/batch to live endpoints list
- [ ] Any endpoint showcase cards or marketing copy referencing endpoint count
- [ ] openapi.json / API docs — confirm /ahs/batch appears correctly
- [ ] 402index.io — manually register /ahs/batch endpoint (manual step, $10.00 for up to 10 wallets)
- [ ] PARTNERSHIPS.md or any outreach docs referencing endpoint count
- [ ] README if it lists endpoints

### ERC-8004 (added Mar 13-16)

ERC-8004 deployed on Ethereum mainnet Jan 29 2026. Identity + reputation standard for agents, complements x402. AHM health scoring sits naturally on top of this stack.

- [ ] Monitor ecosystem: **@quantu_ai** (indexing attestations), **@cascade_fyi** (SATI attestations), **Warden** ($4M funded, Messari-backed)
- [ ] Consider ERC-8004 integration angle for AHS — health scores as attestation data

---

## P2 — Product Backlog

### New Endpoints / Features

- [ ] **Messari x402 integration** — Explore consuming Messari's x402 endpoints (market data, token unlocks, X mindshare) to enrich AHS D3 scoring. Also a co-marketing story
- [ ] **Messari signal integration** — X mindshare, token unlocks, fundraising data as enrichment layer for AHS D3 or new premium endpoint
- [ ] **B2B customer angle** — Agent credit providers, lending protocols, agent creation platforms are natural AHM customers (not just agents themselves). Develop pitch for this segment
- [ ] **Wash Phase 2** — Token approvals scan + dead contract detection (deferred from wash MVP, see wash_spike_results.md)
- [ ] **ERC-8183 integration** — Pre-flight health check before a client funds an ERC-8183 job escrow. Add to product backlog as agentic commerce verification layer
- [ ] **ERC-8183 provider integration** — 3-4 day build, additive via new `acp_worker.py` module. Event listener watches ACP contract for jobs where provider=AHM, routes to existing `calculate_ahs()`, submits result hash on-chain. Reuses ERC-8004 identity (#32328). Build when ERC-8183 deploys to Base mainnet and first real jobs appear. Technical assessment complete — see session notes
- [x] **/ahs/batch endpoint** (completed Mar 31) — POST `/ahs/batch` scores multiple agent wallets in a single API call. Up to 10 wallets per x402 call ($10.00 flat), up to 25 via API key (1 credit/wallet, partial results supported). Concurrent scoring with semaphore-limited RPC. PRs #42, #43, #44. Design partner validated (Alfred Zhang, httppay.xyz)

### Long-term Product Visions

- [ ] **AHM Shield** — Always-on agent protection middleware. SDK → Proxy → Enterprise Fleet. Business model evolution: pay-per-call → per-transaction → subscription → enterprise per-agent pricing. "Norton/CrowdStrike for the agent economy"
- [ ] **Agent Title Registry** — Ownership transparency modelled on UK Land Registry (proprietorship, property, charges & restrictions)
- [ ] **Agent Certification** — On-chain attestation badges (Gold/Silver/Bronze), 90-day renewal
- [ ] **Agent Power Index** — Comprehensive measure of agent digital footprint and influence
- [ ] **Micro-Utility Portfolio** — 20-50 tiny single-purpose x402 endpoints, separate strategy
- [ ] **ERC-8183 pre-flight check** — AHM as the health verification layer before a client funds an ERC-8183 job escrow. Natural fit: x402 (payments) + ERC-8004 (identity) + ERC-8183 (commerce) + AHM (health) = complete agent stack

### AHS Enhancements

- [ ] AHS D3 infrastructure probing — expand probe coverage (uptime, latency, error rates)
- [ ] Cross-dimensional pattern library — expand beyond Zombie Agent, Cascading Infrastructure Failure, Spam Drain, Gas Hemorrhage
- [ ] Trend tracking improvements — JWT-based score history
- [x] **Smart contract wallet scoring** (completed Mar 24) — When `txlist` returns < 10 outgoing txs, `calculate_ahs()` now falls back to Blockscout V2 `token-transfers` API. D2 uses 4/8 signals (timing regularity, transfer diversity, counterparty breadth, activity gaps) with redistributed weights. Verified: ACP agents now score D2 10-68 (was baseline 50). EOA wallets unaffected (64/64 tests pass). New: `fetch_token_transfers()`, `calculate_d2_score_from_transfers()`, `_calc_token_transfer_diversity_score()`, `AHSResult.d2_data_source` field

---

## P3 — Tech Debt / Frontend Fixes

- [ ] **Proper OG banner (1200x630)** — `generate_og_banner.py` created, interim logo fix live. Complete this week
- [ ] Wash scan composite scoring refinement (see wash_spike_results.md for formula)

---

## P4 — Ecosystem Scanning / Monitoring

- [ ] **ERC-8004 ecosystem** — @quantu_ai, @cascade_fyi, Warden ($4M funded, Messari-backed)
- [ ] **FairScale (@fairscalexyz)** — Solana credibility scoring, track for potential cross-chain angle
- [ ] **PayAI Network** — monitor new analytics dashboard when it ships
- [ ] **Virtuals ACP** — maintain 11 offerings. **Note:** Virtuals agents share 10 TBA shards (no independent wallets), so per-agent AHS is not meaningful. Consider a "Virtuals protocol health" monitoring endpoint instead of per-agent scans
- [ ] **ERC-8004 scan coverage** — 32,738 agents registered, only 400 scanned so far (IDs 1-352 + 30000-30355). Consider periodic scans of new registrations and mid-range IDs (1000-29999) for broader calibration
- [ ] **Major registrar `0x6ffa1e...99bc`** — owns ~40% of agents in 30000+ range, likely a registration platform (Synthesis?). AHS 48/D. Monitor for platform-level outreach opportunity
- [x] **ACP proactive scan spike** (completed Mar 24) — Proved end-to-end flow: ACP API discovery (40,497 agents) → wallet dedup (unique per agent, no sharing) → AHS scan → DB storage. **Best source: `acpx.virtuals.io/api/agents`** (free, no auth). **Blocker found:** ACP wallets are smart contract wallets — AHS D2 returns baseline because `fetch_transactions()` misses token transfer activity. See `proactive_scan_spike_results.md` and `acp_proactive_scan.py`
- [ ] **ACP full pipeline** — Depends on smart contract wallet scoring enhancement (P2 AHS Enhancements). Once D2 can score token transfers, run full ACP scan (40K+ agents) and schedule periodic rescans
- [ ] **402index.io proactive scan** — Second discovery source. Free API returns service URLs, then probe each for HTTP 402 → extract `payTo` from `accepts[0].payTo`. Higher signal (every payTo is a confirmed x402 merchant wallet) but more complex than ACP

### Public Agent Economy Health Dashboard (depends on P4)

A live public dashboard showing aggregate health stats across all agent wallets scanned by AHM. Inspired by Ryan Gentry's 402index ecosystem overview dashboard (seen March 22 2026, endorsed by Brian Armstrong).

**Positioning:** "The pulse of the agent economy" — AHM owns the agent health data layer the same way 402index owns the endpoint directory layer. Complementary, not competing.

**Dashboard metrics to show:**

- [ ] Total agents scanned
- [ ] AHS score distribution (healthy / degraded / at-risk) with breakdown
- [ ] Most common anomaly patterns detected (Zombie Agent, Phantom Activity, Cascading Failure etc.)
- [ ] Average cleanliness score across the ecosystem
- [ ] Trend lines — is the agent economy getting healthier or degrading over time?
- [ ] Protocol breakdown (x402 / L402 / MPP agents)

**Content/marketing angle:**

- [ ] Weekly "Agent Economy Health Report" tweet with dashboard screenshot — highly shareable in x402/agentic payments circles
- [ ] Ryan Gentry likely to engage/retweet given complementary positioning to 402index

> **Do not build yet** — depends on Priority 4 (Proactive Ecosystem Scanning) being live first. Build order: scanning → data → dashboard → public reports.

---

## Market Research — 402index.io Analysis (March 2026)

Full ecosystem scan of 402index.io service directory. 15,658 indexed services, but ~3,000 real unique services after removing spam (single provider "lowpaymentfee" accounts for ~10,000+ duplicate "Premium API Access" entries at $0.02/call).

### Protocol Landscape

| Protocol | Real Services | Payment | Character |
|----------|--------------|---------|-----------|
| x402 | ~4,600 | USDC on Base | Crypto-native, agent-first. Many on Base Sepolia (testnet/experimental) |
| L402 | 553 | BTC (sats) on Lightning | Most organically diverse — podcasts, academic data, gov records, price oracles, Nostr social graph |
| MPP | 490 | USDC on Tempo | Most enterprise-grade — wraps Google, OpenAI, Anthropic, Stability AI, Firecrawl, Alchemy |

### Key Findings

- **Price sweet spot is $0.01–$0.05 per call** (~35% of services). AHM's pricing sits above this, justified by diagnostic complexity (multi-dimensional scoring, cross-chain lookups, Nansen enrichment)
- **Reliability is a differentiator** — many services show degraded/down health status, low x402 payment validation rates (~40-60%). AHM's health monitoring positioning is validated by the ecosystem's own quality problems
- **Saturated categories:** crypto/DeFi, AI/LLM inference, web scraping, image generation, social media data, flight/travel
- **Empty categories:** healthcare, legal, HR/recruiting, logistics, insurance, education, non-US government data, identity/KYC
- **Top providers:** Merit Systems (MPP, ~200+ services), Mycelia Signal (L402, ~70+ oracles), Google (MPP, 48), Lightning Enable (L402, ~40 data proxies)

### Opportunities — Build Now

- [ ] **Trust Registry + Agent Certification** — Ecosystem is actively asking for a reputation/quality layer. Ryan Gentry and Austin Danson both identified this gap publicly. Directly extends backlog Spike 13 (Agent Certification). WoT Scoring API exists for Nostr-only; DJD Agent Score exists but is alone. AHM is positioned to own this for on-chain agents
- [ ] **Proactive Ecosystem Scanning** — Already P4 in backlog. Confirmed as the right next build based on market analysis. The index itself demonstrates demand for health/quality monitoring (health_status, reliability_score, uptime_30d fields are core to the directory)

### Opportunities — Build Next (1–3 months)

- [ ] **AHM Shield** — Already in backlog (Long-term Product Visions). Market analysis confirms recurring subscription model would differentiate from pay-per-call competitors. No other provider offers always-on monitoring. The ecosystem's reliability problems (services going down, payment validation failures) create demand for continuous health assurance

### Opportunities — Park for Later

- [ ] **UK Government Data API** — Zero competition on 402index for UK public data (Companies House, Land Registry, HMRC, NHS, DWP APIs). Daniel's professional background (HMLR, NHS England, DWP, HMRC, DfE) is a unique advantage. Revisit when AHM has more momentum. **NOTE:** This is a completely separate product from AHM — do not conflate with Agent Title Registry (Spike 12) which is a conceptual governance layer inspired by land registry model, not a literal HMLR data integration

### Parked — No Current Fit

- **Healthcare/legal/HR data APIs** — Big white space but high regulatory risk, no fit with AHM's agent health positioning
- **E-commerce product data resale** — Check Rainforest API ToS before considering

---

## Partnership & Outreach Targets (from 402index.io market analysis, March 2026)

Approach: don't cold pitch — show up in their threads with genuine insight first, then find a specific integration angle.

### Tier 1 — Direct Integration Partners (highest priority)

- [ ] **Questflow** — Multi-step agent workflow orchestration. Agents running workflows need health checks before executing. Natural pre-flight AHM integration point. Find founder, engage on X. **Note:** Already identified in ERC-8004 scan — Mia's Alpha Hunter (#30354), AHS 74/C
- [ ] **DJD Agent Score** — Only other agent scoring system in the directory. Could be competitor or complementary — reach out to understand their angle. Differentiation or collab story TBD
- [ ] **WoT Scoring API** — Agent identity/trust for Nostr. Different ecosystem (Nostr vs on-chain) but same problem space (agent reputation). Worth a conversation
- [ ] **CertVera** — Compliance timestamps on Bitcoin. Natural fit alongside AHM's future Agent Certification concept (backlog Spike 13)

### Tier 2 — Awareness / Co-marketing (medium priority)

- [ ] **Merit Systems** — Biggest MPP provider, 200+ endpoints. Their agents need health monitoring. Cold outreach when AHM has more traction
- [ ] **Mycelia Signal** — 70+ price oracle endpoints on L402. High-frequency trading bots are exactly the agents that need AHS scoring
- [ ] **Alchemy** — Already in x402 ecosystem. Enterprise credibility, large developer audience
- [ ] **Firecrawl / Browserbase** — Web scraping agents are active x402 users. Their customers are building autonomous agents that would benefit from AHM

### Tier 3 — Monitor and Engage Opportunistically

- [ ] **Sats4AI** — Full AI suite on L402. Different payment rail but same developer community
- [ ] **Lightning Enable** — US government data proxy. Complementary positioning, no overlap
- [ ] **proxy402 / The Ark** — LLM inference proxies. Agents using these are AHM's target users

---

## Content & Marketing (added Mar 13-16)

- [ ] **Demo recording** — Short Loom walkthrough of AHM for hackathon submissions, partnership pitches, cold outreach. Record this week
- [ ] **@BaseHubHB engagement** — weekly Base launch curator, 2.5K+ views per post. Engage every week for inclusion in roundups
- [ ] **Pin a new X post** — Draft and pin a fresh @AHMprotocol post once a meaningful milestone is hit (e.g. first organic payment, OpenServ selection, Coinbase PR merge). Replace the unpinned post with something current

---

## Scheduled Reviews

### Session Continuity Shadow Mode Review — 7 April 2026

Review shadow_signals data across the trust registry and decide whether to promote
session continuity scoring from shadow mode to live D2 weighting.

Checklist:
- [ ] Query scan results for distribution of session_continuity_score across all wallets
- [ ] Confirm score distribution is stable (not causing unexpected AHS shifts)
- [ ] Check how many wallets are triggering Budget Exhaustion shadow pattern
- [ ] Review whether session_continuity_score correlates with existing D2 scores
- [ ] If distribution looks stable, implement live D2 weighting (weight ~0.10,
      redistribute from timing_regularity and retry_storm)
- [ ] Update AHS model version to AHS-v2 when promoted to live

---

## Completed

- [x] 11 endpoints live (risk, premium, counterparties, network-map, health, wash, ahs, alerts, optimize, retry, agent/protect)
- [x] Nansen integration — 4 direct API connections
- [x] Virtuals ACP listing (11 offerings)
- [x] x402scan listing
- [x] Bankr Skills listing
- [x] agdp.io listing
- [x] PayAI Network merchant registration
- [x] Wash MVP (Phase 1) — dust, spam, gas efficiency, failed tx patterns
- [x] AHS 3D mode with infrastructure probing
- [x] Alert monitoring with Slack/Discord/generic webhooks
- [x] RetryBot with non-custodial ready-to-sign transactions
- [x] Protection Agent autonomous triage
- [x] Wash API spike completed (see wash_spike_results.md)

### Completed Mar 17 2026

- [x] ERC-8004 registry scan (IDs 30000+) — 200 agents enumerated, 183 wallets scanned, avg AHS 59.5, range 38-86, grades B=7/C=17/D=158/E=1. Results in `erc8004_scan_results.csv` + `erc8004_scan_results.md`. All scans persisted to DB
- [x] Bug fix: UTF-8 encoding for CSV/MD report generation (`erc8004_scan.py` — Windows cp1252 crash on Unicode agent names)
- [x] Identified 7 healthy agents (B-grade) as case studies and 1 Critical (E-grade Zombie Agent) as outreach target
- [x] Identified recognizable projects: Questflow, Zyfai, Con+ Audits, Bob Suite, OpenClaw, LEDGR, Sentinel
- [x] Virtuals ACP wallet scan spike — 1000 agents enumerated via `api.virtuals.io`, only 12 unique wallets found (10 TBA shards + 1 creator + 1 sentient). Avg AHS 56.8, range 50-78, grades B=1/D=11. **Key finding: Virtuals agents share platform-level infrastructure wallets, not independent wallets like ERC-8004 agents.** Results in `virtuals_scan_results.csv` + `virtuals_scan_results.md`
- [x] Cross-registry tracking — `db.py` schema v2: added `registries` column to `known_wallets`, updated `log_scan()` upsert to append registry names. Backfill migration for existing records
- [x] `virtuals_scan.py` created — 5-phase scanner (discovery, enumeration, dedup, AHS scan, report). CLI: `--max-pages`, `--max-scans`, `--skip-scan`

### Completed Mar 16 2026

- [x] ERC-8004 registration — agentId #32328, owner `0xB109A7...13aD`, tx [`0x048b55ea...`](https://basescan.org/tx/0x048b55ea60e24896eb932f250711c346d16a97382f88391c646c75f418fc7fe9), Base mainnet
- [x] 8004scan.io listing — x402 enabled, reputation trust, active, verified owner
- [x] Ecosystem scan — 18 agents scanned, results in ecosystem_scan_results.md, average AHS 60/D
- [x] Security audit & hardening — 49/49 tests passing, 3 findings fixed (HIGH: address validation middleware, MEDIUM: global exception handler, MEDIUM: coupon rate limiting)
- [x] GitHub Actions CI — security tests run on every push/PR to master, branch protection enabled
- [x] Outreach scan — 7 named services scanned, results in ecosystem_scan_outreach.md, Daydreams 38/E Critical flagged
- [x] Daydreams DM sent — flagged 38/E Critical "Stale Strategy" finding on facilitator wallet `0x279e08...4653`
- [x] PayAI DM sent — shared signer wallet scores (51/D and 61/C)
- [x] Bankr Skills PR #195 chased — via X post on @bankrbot
- [x] synthesis.md registration — completed
- [x] OG image interim fix — og:image updated to self-hosted ahm-logo.png, descriptions updated, deployed
- [x] Coinbase PR #1207 — Merged by @Must-be-Ash on Fri Mar 14 2026. AHM now listed in official Coinbase x402 ecosystem
