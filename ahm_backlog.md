# AHM Backlog

> Single source of truth for Agent Health Monitor. Update at end of every session.

---

## Current State (as of Mar 17 2026)

- **14 endpoints** live on Base mainnet at agenthealthmonitor.xyz
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
- [x] static/index.html — endpoint count updated to 14
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

### ERC-8210 (added Apr 6 2026)

ERC-8210 verification schema names AHM as a reference implementation. AHS D1/D2/D3 will be cited as a concrete composition example in the spec's multi-hop reference scenarios.

- [ ] **cmayorga (Carlos Mayorga, ETH Magicians)** — CTO at DeFiRe.finance, Madrid. ERC-8210 contributor, active in ERC-8183 thread. Has named AHM as a reference implementation in ERC-8210 verification schema (assessor type: `ahs-d1-d2-d3`). AHM will be included as a concrete composition example in ERC-8210 multi-hop reference scenarios. **Note:** defire.finance flagged as phishing site by NordVPN — verify before any direct coordination. Keep engagement on public forum only for now

---

## P2 — Product Backlog

### New Endpoints / Features

- [ ] **Messari x402 integration** — Explore consuming Messari's x402 endpoints (market data, token unlocks, X mindshare) to enrich AHS D3 scoring. Also a co-marketing story
- [ ] **Messari signal integration** — X mindshare, token unlocks, fundraising data as enrichment layer for AHS D3 or new premium endpoint
- [ ] **B2B customer angle** — Agent credit providers, lending protocols, agent creation platforms are natural AHM customers (not just agents themselves). Develop pitch for this segment
- [ ] **Wash Phase 2** — Token approvals scan + dead contract detection (deferred from wash MVP, see wash_spike_results.md)
- [ ] **ERC-8183 integration** — Pre-flight health check before a client funds an ERC-8183 job escrow. Add to product backlog as agentic commerce verification layer
- [ ] **ERC-8183 provider integration** — 3-4 day build, additive via new `acp_worker.py` module. Event listener watches ACP contract for jobs where provider=AHM, routes to existing `calculate_ahs()`, submits result hash on-chain. Reuses ERC-8004 identity (#32328). Build when ERC-8183 deploys to Base mainnet and first real jobs appear. Technical assessment complete — see session notes
- [ ] **ERC-8183 assessIndependence interface** — agentltsoh (AAP) proposed a standardised signal interface `assessIndependence(addr, bytes)` returning `{independent: bool, signals, confidence: bytes}` that ERC-8183 hooks could call optionally. AHM's scoring output already maps to this shape. Monitor whether this gets formalised into the ERC-8183 spec. If it does, implement the interface so AHM becomes a drop-in pluggable assessor. No action until spec movement confirmed. Source: ETH Magicians ERC-8183 thread, Apr 7 2026
- [ ] **ERC-8183 evaluator monetisation (Phase 2)** — Currently the AHM evaluator daemon calls `/ahs/route/` at $0.01 per job (self-funded x402 revenue). Direct protocol payment for evaluation services is not part of ERC-8183 today. Phase 2 monetisation options to explore: (1) negotiate a per-evaluation fee with protocol deployers once volume is proven on mainnet; (2) offer a pre-scoring / agent whitelisting service where providers pay AHM to get a health certificate before submitting to high-value job marketplaces. No action until mainnet evaluator role is established and job volume is measurable. Source: ERC-8183 evaluator daemon build, Apr 7 2026
- [ ] **ERC-8210 assessor schema alignment** — Review ERC-8210 draft spec when available. Ensure AHM's D1/D2/D3 output format matches the assessor schema (`type: rule`, `id: ahs-d1-d2-d3`, `verdict: APPROVE/REJECT`, `confidence: 0-1`). Consider publishing AHS verdicts as IPFS-pinned outputs to serve as verifiable assessor outputs (links to EAS integration backlog item). No action until ERC-8210 draft is shared
- [ ] **Agent Alerts System** — Build an alerting layer allowing integrators and agents to set custom alerts on wallet/agent activity. Use cases to think through before building: (1) **Integrator alerts** — notify when an agent they are routing drops below a configured AHS threshold; (2) **Abuse detection alerts** — flag agents attempting to manipulate routing policies or allowlists. Requirements and scope TBD — needs design spike before building
- [ ] **Evidence object in API responses** — Add a structured `evidence` object to `/ahs` and `/ahs/route/{address}` endpoint responses. Should include: per-dimension signal breakdown, weighted scores, data sources, and reasoning — not just the final grade. Rationale: strengthens "no black boxes" positioning, makes AHM more composable for `IRiskHook` implementations per ERC-8210. Inspired by RNWY's public disclosure of their evidence object pattern (Apr 8 2026). **Do not build yet.** Review after D3 Operational Stability is live
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
- [ ] **D4 Output Quality Score (concept, Phase 2)** — Potential future AHS dimension derived from aggregated post-transaction output quality verdicts across an agent's job history. If an agent consistently produces low-quality deliverables that get rejected by content verification evaluators (e.g. ThoughtProof-style DISSENT/HOLD verdicts), that rejection pattern should feed into their AHS score as a fourth dimension alongside D1 Solvency, D2 Behavioural Consistency, D3 Operational Stability. Would require either: (a) a data-sharing agreement with content verification providers, or (b) reading rejection patterns from on-chain ERC-8183 job history. No action until D3 is live and ERC-8183 mainnet job volume is measurable. Source: ThoughtProof research, Apr 7 2026
- [ ] **D5 Security Posture (spike, concept)** — Explore whether on-chain security signals could form a new AHM scoring dimension. Signal sources to investigate: flagged contract interactions, known exploit wallets, phishing-adjacent addresses, sanctioned addresses (Chainalysis/OFAC lists). Context: Project Glasswing (Anthropic + Linux Foundation + Microsoft/Google/NVIDIA coalition) launched Apr 8 2026 — focused on code/software vulnerability detection via Claude Mythos Preview. AHM angle: Glasswing secures agent code, AHM monitors live runtime behaviour — complementary positioning opportunity. Linux Foundation is also an x402 founding partner — potential ecosystem overlap worth monitoring. **Do not build yet.** Review after D3 Operational Stability is live
- [ ] **Reputation decay scoring (concept)** — Agents that were once healthy but have gone dormant should score progressively lower over time. Influence/score should decay without fresh verified on-chain activity — health must be continuously re-earned. Aligns with AHM's existing Zombie Agent pattern detection — decay is the scoring complement to zombie flagging. Rationale: validated by RNWY's public reputation decay concept (Apr 8 2026). **Do not build yet.** Review alongside D3 Operational Stability
- [x] **Smart contract wallet scoring** (completed Mar 24) — When `txlist` returns < 10 outgoing txs, `calculate_ahs()` now falls back to Blockscout V2 `token-transfers` API. D2 uses 4/8 signals (timing regularity, transfer diversity, counterparty breadth, activity gaps) with redistributed weights. Verified: ACP agents now score D2 10-68 (was baseline 50). EOA wallets unaffected (64/64 tests pass). New: `fetch_token_transfers()`, `calculate_d2_score_from_transfers()`, `_calc_token_transfer_diversity_score()`, `AHSResult.d2_data_source` field

### Bootstrapping Problem — Zero-History Wallet Treatment
- Currently a wallet with no on-chain history scores similarly to one with demonstrably degraded patterns due to D2 behavioural consistency weighting (70%)
- Raised by Bakugo32 (Arc/ERC-8183 protocol team) Apr 9 2026 after Job #7 evaluation — D grade on zero-history wallet triggered reject when HOLD was more appropriate
- Fix: introduce a separate scoring path for zero-history wallets — score as "Unrated" rather than mapping to D/E grade
- Unrated wallets should route to escrow (not reject) by default — client assumes counterparty risk with funds protected
- Review alongside D2 session continuity gate (April 21 2026)

---

## Future Dimensions

### Spike: D5 Security Posture
- Explore whether on-chain security signals could form a new AHM scoring dimension
- Signal sources to investigate: flagged contract interactions, known exploit wallets, phishing-adjacent addresses, sanctioned addresses (Chainalysis/OFAC lists)
- Context: Project Glasswing (Anthropic + Linux Foundation + Microsoft/Google/NVIDIA coalition) launched Apr 8 2026 — focused on code/software vulnerability detection via Claude Mythos Preview
- AHM angle: Glasswing secures agent code, AHM monitors live runtime behaviour — complementary positioning opportunity
- Linux Foundation is also an x402 founding partner — potential ecosystem overlap worth monitoring
- Do not build yet. Review after D3 Operational Stability is live.

---

## P3 — Tech Debt / Frontend Fixes

### Frontend / UI

- [ ] **AHM Logo — Homepage Link** — The AHM logo on the app page (agenthealthmonitor.xyz/app) has no link. It should link back to the homepage (agenthealthmonitor.xyz). Quick fix, low priority
- [ ] **Global Navigation on App Page** — No navigation bar present on the app page. Add a minimal nav bar with links to Homepage, Docs (docs.agenthealthmonitor.xyz), and App for consistency across the site. Low priority but worth tidying before design partner demos

### Other

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

## Ecosystem & Protocol

### EvaluatorRegistry Metadata Field — Methodology Declaration
- Bakugo32 (Arc protocol team) proposed adding a metadata field to EvaluatorRegistry so evaluators can declare their methodology on-chain
- AHM should declare its methodology when this field ships: counterparty trust scoring (D1 solvency, D2 behavioural consistency, D3 operational stability) — distinct from output quality evaluation
- This enables clients to select evaluators by methodology rather than address alone
- Monitor ERC-8183 thread for formal proposal — support it when raised
- First noted: Apr 9 2026

### x402 "Upto" Scheme — Variable Cost Endpoint Support
- Coinbase shipped the "upto" payment scheme for x402 on Apr 9 2026
- "Upto" authorises up to a maximum amount and charges only what is consumed — designed for variable-cost services like LLM inference
- Current AHM core endpoints are all "exact" scheme (fixed price) — no change needed
- Relevant for AHM Verify: LLM panel cost per verdict varies by spec/deliverable length, domain, and whether deep mode is triggered — "upto" is the natural payment scheme for /verify endpoints
- Relevant for AHM Shield Phase 2: if Shield evolves toward risk-premium pricing, variable per-agent costs map better to "upto" than "exact"
- Action: adopt "upto" scheme for AHM Verify payment endpoints when building the service. Review x402 SDK upto implementation before AHM Verify v0.1 build starts.
- First noted: Apr 9 2026

### Arc Mainnet Migration — Evaluator Readiness
- Arc announced open sourcing of testnet code and approach to mainnet on Apr 9 2026
- Bug bounty live on HackerOne (hackerone.com/circle-bbp)
- AHM is live as ERC-8183 evaluator on Arc testnet (evaluator wallet: 0x35eeDdcbE5E1AE01396Cb93Fc8606cE4C713d7BC)
- Bakugo32 confirmed contract redeployment this week with new addresses — AHM must restake on new contracts
- Mainnet migration = real USDC evaluator fees, real economic value for AHM evaluator role
- Actions required:
  1. Watch ERC-8183 ETH Magicians thread for new contract addresses (email notification active)
  2. Restake on new contracts immediately when posted — Bakugo32 confirmed they will fund AHM wallet with VRT and ETH
  3. Remove ON_CHAIN_START_BLOCK env var from Railway after Job #7 resolved (DONE Apr 9 2026)
  4. Update ARC_CONTRACT_ADDRESS in Railway env vars when new contracts are live
  5. Monitor Arc mainnet timeline — when mainnet launches, evaluator daemon needs pointing at mainnet RPC and contract addresses
- Arc testnet code now open source — review for wallet behaviour patterns that could feed AHM D1/D2 scoring signals
- First noted: Apr 9 2026

---

## Phase 3 — Future Products

### AHM Verify — Post-Transaction Output Quality Scoring

- Standalone service (separate repo: `ahm-verify`, separate Railway project)
- Multi-LLM adjudication panel (3-model: Claude, GPT, Gemini) scores delivered output against pre-registered job spec
- Core moat: AHM trust registry cross-reference makes verdicts stateful and identity-anchored — ThoughtProof cannot replicate without building a competing registry
- Architecture: Option C — standalone service with read-only access to AHM core via `/internal/agent-profile/{address}`
- Monetisation: $0.25/verdict (single), $0.35 combined verdict+AHS report, subscription TBD
- Submission flow MVP: client-submitted only. Phase 2: agent self-submission. Phase 3: ERC-8183 evaluator role
- D4 composite feedback into AHM core AHS: deferred until >500 verdicts/week and D3 is live
- Full feasibility spike saved at `docs/ahm_verify_spike.md`
- Positioning: Monitor (before) + Shield (during) + Verify (after) = complete agent health lifecycle
- **Do not build yet.** Next step: prompt design doc + 20-30 hand-labelled test set to validate panel agreement rate before any code
- Spike completed: Apr 8 2026

---

## Future Product Concepts

### AHM Challenge Protocol — Decentralised Task Market with Built-in Verification

**Inspiration:** Discovery of ProblemManager contract (0x7d563ae2881d2fc72f5f4c66334c079b4cc051c6)
during taxonomy POC — a heavily-used (1,336+ interactions) decentralised problem-solving
marketplace where agents submit answers and are rewarded for verified correctness.

**Concept:**
A decentralised task market where AHM scoring and AHM Verify are embedded as the
verification layer:

1. Client posts a task on-chain with a bounty
2. Multiple agents compete to complete it
3. AHM Verify runs adversarial evaluation on each submission
4. Highest-scoring agent (by AHS + Verify verdict combined) wins the bounty
5. Participating agents' AHS scores update based on task outcomes — creating
   longitudinal behavioural data for D2 scoring

**Why this matters:**
- Creates protocol-level demand for AHM scoring and Verify
- Gives agents a direct financial incentive to maintain high AHS scores
- Generates continuous AHM Verify revenue stream
- Creates a flywheel: better agents score higher → win more tasks → build
  richer history → score even higher
- Validates Research/Verification agent categories in the taxonomy

**Market validation:**
ProblemManager's 1,336+ interactions from top-200 agents by tx count suggests
strong demand for structured task markets with verifiable outcomes. Agents are
actively seeking verification of outputs and competing for quality rewards.

**Connection to existing AHM products:**
- AHM Verify (D4) is the natural evaluation engine
- AHS trust routing determines which agents are eligible to compete
- ERC-8183 job lifecycle (client → provider → evaluator) maps directly to
  this model
- x402 micropayments for Verify calls fit naturally into the bounty flow

**Priority:** Backlog — concept only, requires further research and design
**Status:** Not started

---

## AHM Intelligence — Public KPI Dashboard & Agent Taxonomy

### Vision
Build on AHM's data moat (10,000+ agents scanned) to create a public-facing intelligence platform at intelligence.agenthealthmonitor.xyz. This serves multiple strategic purposes: establishes AHM as the authoritative source of agent economy insights, drives SEO and inbound traffic, creates shareable/viral visual content for social media, and builds ecosystem credibility ahead of commercial opportunities.

Long-term this could become a commercial product in its own right. Free for the foreseeable future.

---

### Layer 1 — v1 KPI Dashboard (build now)
**Subdomain:** intelligence.agenthealthmonitor.xyz
**Deployment:** Separate Railway service, nightly snapshot updates
**Stack:** Lightweight static site, same dark theme as main AHM site

**Three sections for v1:**

1. Ecosystem Health — headline numbers
   - Total agents scanned, avg AHS, zombie rate, grade distribution
   - Single "Ecosystem Health Score" — weighted composite of all agents (like a stock market index for agent health) — primary shareable number
   - Last updated timestamp

2. Registry Leaderboard
   - Side-by-side comparison: ACP, Olas, Celo, ERC-8004, Arc
   - Avg AHS per registry, grade distribution, zombie rate per registry
   - Inherently shareable with each protocol's community

3. Trend Line
   - Weekly ecosystem health snapshots going back as far as data allows
   - Shows whether the agent economy is getting healthier or sicker over time
   - Most shareable single visual

**Deferred to v2:** Individual agent leaderboards, registry drill-down, downloadable reports, API access to intelligence data.

---

### Layer 2 — Agent Taxonomy (define soon, publish alongside dashboard)
**Output:** A structured classification of the agentic economy — the first authoritative taxonomy of its kind. Starts simple, grows as the ecosystem matures. Published as a document on intelligence.agenthealthmonitor.xyz.

**Why this matters:**
- No authoritative agent taxonomy exists yet — genuine first-mover opportunity
- Industries will need this reference framework in coming years
- Drives significant SEO and inbound citations
- Positions AHM as thought leader, not just a tool
- Creates natural content calendar as new agent categories emerge
- Provides logical backbone for Layer 3 enriched reports

**Proposed v1 structure (to be refined):**
Top-level categories:
- Utility Agents — task execution, automation, workflow
- Financial Agents — trading, payments, portfolio management, DeFi
- Physical World Agents — robotics, drones, IoT, embedded systems

Industry sectors (cross-cutting):
- Finance & Banking
- Healthcare
- Defence & Security
- Space & Aerospace
- Supply Chain & Logistics
- Legal & Compliance
- Research & Science

**Action:** Draft v1 taxonomy as a document. Publish on intelligence subdomain. Invite community input and iteration. Keep it simple to start.

---

### Layer 3 — Enriched Intelligence Reports (future / v3+)
Fine-grained KPI reports organised by taxonomy category. Requires both the taxonomy (Layer 2) and data enrichment work to be completed first.

Examples of future reports:
- What are trading agents actually doing vs utility agents?
- How does agent health differ between finance and defence sectors?
- What tasks are agents performing, and how do those differ by sector?
- Which registries host which categories of agent?

---

### Data Enrichment Spike (required before Layer 3)
Current AHM data captures wallet behaviour, financial health, and operational stability — but not what an agent does or what sector it operates in. A research spike is needed to identify enrichment sources and plan how to fill gaps.

**Spike scope:**
- On-chain metadata and contract interactions — can agent purpose be inferred?
- Registry metadata — do ACP, Olas, ERC-8004 tag agents by category?
- Off-chain sources — agent project websites, GitHub repos, social media, blog posts, news articles
- Emerging standards — does any ERC or protocol define agent categories?
- Gap analysis — what data exists vs what we need for Layer 3 reports

**Output:** A spike document identifying available enrichment sources, recommended data collection approach, and a prioritised gap list.

---

### Suggested Sequencing
1. Build v1 KPI dashboard with existing data (this week)
2. Draft and publish v1 Agent Taxonomy alongside dashboard (this week)
3. Run data enrichment spike (next month)
4. Build enriched taxonomy-organised views as data improves (v2/v3)
5. Explore commercial model for intelligence product (longer term)

**Priority:** High — 10k agent milestone is the right moment to launch this.
**Status:** Backlog — ready to begin Layer 1 build and Layer 2 taxonomy draft.

---

### Taxonomy Category in AHS API Response

Once taxonomy classification is productionised (Phase 5), add `taxonomy_category`
as an optional field to the AHS API response:

- Add `taxonomy_category: str | None` to `AHSReport` Pydantic model in `api.py`
- Defaults to `None` until taxonomy classification is available for that address
- Populated from the taxonomy classification DB table once Phase 5 is complete
- Surface in the frontend AHS result card as a category badge
- Include in batch endpoint response

**Example response addition:**
```json
{
  "ahs_score": 74,
  "grade": "C",
  "taxonomy_category": "Financial Agents",
  ...
}
```

**Priority:** Low — depends on Phase 5 taxonomy DB integration
**Status:** Backlog

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

## Competitive Intelligence

### Verdict Protocol (@verdictprotocol, verdict-protocol.xyz) — ELEVATED THREAT
- Joined: March 2026. 123 followers. Whitepaper published. $VRDCT token live on Virtuals.io.
- Positioning: "The trust layer for agent commerce" — six-layer protocol above ERC-8183 covering Evaluator Network, Hook Registry, Trust & Reputation Index, Facilitator Layer, Policy & Underwriting, $VRDCT coordination
- Direct overlap: Trust & Reputation Index (multidimensional scoring across providers, evaluators, hooks, clients) and Evaluator Network (routing and coordinating ERC-8183 evaluators)
- No overlap: pre-transaction wallet health scoring (D1/D2/D3), zombie detection, AHM Shield runtime middleware
- Key differentiator vs AHM: token-coordinated protocol play requiring $VRDCT staking/governance adoption — much heavier lift than AHM's direct API/SDK model
- Strategic opportunity: Verdict lists "Reputation API" as Phase 5 (4 phases away). AHM has live reputation data today. Potential B2B data partnership — AHM as data provider for Verdict's Trust & Reputation Index rather than head-to-head competitor
- Threat level: Medium-high long term if they execute. 6-12 month window before they can meaningfully compete on reputation data — use it to establish AHM as default scoring layer
- Action: Monitor only. Do not engage publicly. Review positioning if they reach Phase 3+ or follower count exceeds 1K.
- First noted: Apr 8 2026

### AgentProof (@agentproof, agentproof.sh) — ELEVATED THREAT, NEEDS INVESTIGATION
- Positioning: "Know Your Agent" — on-chain reputation oracle for AI agents
- Scale: 158.2K agents indexed, 222.7K evaluations, 375.7K+ screenings, 21 chains, oracle live
- Notable features: agent leaderboard with trust scores, Deployer Storm visualisation (shared deployer rings, coordinated deployment detection, wallet age colour coding), live threat intelligence, flagged agents, zero-history deployer detection, Agent Directory
- Direct overlap: on-chain reputation scoring, agent leaderboard, sybil/collusion detection, multi-chain coverage (21 chains vs AHM's 5)
- Scale gap: 158K indexed agents vs AHM's 5K — significant
- Notable: tweet from @BuilderBenv1 "You basically outlined AgentProof..." — suggests AHM concept maps closely to their existing product
- Not previously captured in competitive intelligence — discovered Apr 8 2026
- Threat level: ELEVATED. Larger scale, more chains, live oracle, sophisticated visualisations.
- Action: Full competitive analysis needed. Investigate funding, team, API, pricing, and differentiation from AHM. Schedule for next session.

### t54.ai (@t54ai, t54.ai) — ELEVATED THREAT
- Founded: September 2024. San Francisco. 10.6K followers.
- Funding: $5M seed — Anagram, PL Capital, Franklin Templeton (strategic), Ripple (strategic). Hiring DevRel/BD and AI Researcher.
- Positioning: "Empowering Trusted Agentic Economy" — full-stack trust infrastructure
- Product suite: KYA verification + programmable guardrail, Trustline (real-time risk controls: identity, code audit, mandates, behavioural patterns, device context), x402 Merchant (verified identity layer), x402 Secure (security layer for A2A payments, real-time ranking/scores, anomaly detection), official x402 server leaderboard supported by Coinbase x402 Bazaar
- Direct overlap: Trustline (behavioural patterns, anomaly detection, real-time scoring = AHM D1/D2/D3), x402 Secure dashboard (agent ranking + security scores = AHM public dashboard)
- Critical differentiator vs AHM: official Coinbase x402 Bazaar relationship — direct distribution through x402 ecosystem
- AHM differentiation: ERC-8183 evaluator live (t54 has no visible ERC-8183 presence), protocol registry scanning (ACP, Olas, Arc, Celo, ERC-8004), AHS publicly callable API, 5K+ scanned agents live today
- Also published ARS paper (arxiv 2604.03976) with Microsoft Research, Google DeepMind, Columbia
- Threat level: HIGH. Best-funded competitor seen to date. Coinbase relationship, Franklin Templeton signals regulated finance ambitions.
- Action: Monitor closely. Accelerate Invoica design partner lock-in. Double down on ERC-8183 as unclaimed territory. Explore B2B data partnership angle.
- First noted: Apr 8 2026

---

## Strategic Positioning

### ARS (Agentic Risk Standard) — Positioning Opportunity
- Paper: "Quantifying Trust: Financial Risk Management for Trustworthy AI Agents" (arxiv 2604.03976)
- Authors: Microsoft Research, Google DeepMind, Columbia, Virtuals ACP (t54.ai)
- Published/promoted Apr 8 2026 — 509K impressions on Virtuals Protocol tweet within hours
- Relationship to AHM: complementary. ARS defines settlement/compensation when things go wrong. AHM defines whether to trust the counterparty before things go wrong.
- Key paper quote: "This shifts the trustworthy-agent problem from model-internal safety to a measurement problem: estimating failure probabilities and loss magnitudes." AHM is that measurement layer.
- ARS explicitly states the bottleneck is risk modeling not settlement mechanics — AHM's AHS scoring is the risk model ARS underwriters need to price correctly.

#### Three backlog items arising from ARS:

##### Fee-only vs Fund-involving Task Distinction in Tiered Trust Routing
- ARS distinguishes fee-only jobs (only service fee at risk) from fund-involving jobs (agent controls user capital pre-execution)
- AHM's current A/B/C/D/F routing treats all jobs the same — should apply higher AHS grade thresholds for fund-involving jobs
- Example: fund-involving job requires grade B minimum; fee-only job accepts grade C
- Review alongside D3 Operational Stability and tiered routing design

##### AHM Shield as Underwriting Intelligence Layer (Phase 2)
- ARS defines underwriter role: prices outcome risk, requires collateral, commits to reimbursement on failure
- AHM Shield currently routes on AHS grade — Phase 2 could evolve to quote a risk premium per job
- This is the "underwriting intelligence layer" ARS says does not yet exist — AHM Shield is positioned to build it
- Do not build yet. Review after Invoica design partner locked in and Shield v1 adoption measured.

##### AHM Verify as ARS EvaluateOutcome Implementation
- ARS requires evaluator to emit auditable outcome record with evidence reference (EvaluateOutcome action)
- AHM Verify's verdict + evidence object is exactly that evaluator implementation
- Strengthens AHM Verify positioning: not just a quality scorer but a standards-compliant ARS evaluator
- Reference when drafting AHM Verify launch messaging

---

## Partnership & Outreach Targets (from 402index.io market analysis, March 2026)

Approach: don't cold pitch — show up in their threads with genuine insight first, then find a specific integration angle.

### Tier 1 — Direct Integration Partners (highest priority)

- [ ] **Questflow** — Multi-step agent workflow orchestration. Agents running workflows need health checks before executing. Natural pre-flight AHM integration point. Find founder, engage on X. **Note:** Already identified in ERC-8004 scan — Mia's Alpha Hunter (#30354), AHS 74/C
- [ ] **DJD Agent Score** — Only other agent scoring system in the directory. Could be competitor or complementary — reach out to understand their angle. Differentiation or collab story TBD
- [ ] **WoT Scoring API** — Agent identity/trust for Nostr. Different ecosystem (Nostr vs on-chain) but same problem space (agent reputation). Worth a conversation
- [ ] **CertVera** — Compliance timestamps on Bitcoin. Natural fit alongside AHM's future Agent Certification concept (backlog Spike 13)
- [ ] **ThoughtProof (thoughtproof.ai)** — Reasoning verification service. Core product: adversarial multi-model critique API returning ALLOW/HOLD/UNCERTAIN/DISSENT verdict + confidence score + objections. x402-gated on Base USDC. MCP server available (`@thoughtproof/mcp-server`). Also active ERC-8183 evaluator on Base Sepolia (first external evaluator, staked Apr 7 2026). Their evaluator role is **complementary not competing**: ThoughtProof evaluates deliverable quality post-submission, AHM evaluates counterparty trustworthiness pre-payment. Potential partnership pattern: AHM gates entry (should we trust this agent?), ThoughtProof gates completion (was the output correct?). Explore as a co-marketing or technical integration partner once AHM mainnet evaluator role is established. **Do not approach until then.** Long-term strategic goal: position AHM as the single assigned ERC-8183 evaluator covering the full job lifecycle — D1/D2/D3 counterparty trust pre-funding (via hook) + D4 output quality post-submission (via evaluator role) — removing the need for job creators to choose between trust-based and quality-based evaluators. ThoughtProof partnership or white-label arrangement could accelerate D4 capability rather than build from scratch.

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

### Session Continuity Shadow Mode Review — 7 April 2026 — NOT PROMOTED

**Outcome (7 April 2026):** Shadow signal **not promoted** to live D2 weighting.

**Reason:** Shadow signals were computed in `monitor.py:_calc_session_continuity_score()`
and returned in the AHS API response, but never persisted to the `scans` table.
The review checklist's first three items (distribution, stability, correlation)
were all unrunnable because there was no historical shadow data on disk to
analyse. Promoting blind would have made the AHS-v2 weighting decision without
any evidence that the signal behaves as designed.

**Action taken:** Persistence patch landed in PR `feat/d2-shadow-persistence`
(schema v8, adds `scans.shadow_signals_json` column, wires `shadow_signals`
through `db.log_scan()` and the AHS endpoint in `api.py`). From this point
onward every AHS scan persists the full shadow-signals payload (including
`session_continuity_score`, `abrupt_sessions`, `budget_exhaustion_count`,
`total_sessions`, `avg_session_length`, and `shadow_patterns`).

### Session Continuity Shadow Mode Review — 21 April 2026 (next gate)

Re-run the review after a 2-week burn-in window with the persistence patch live.

**Prerequisites before running this gate:**
- [x] Persistence patch (`feat/d2-shadow-persistence`) merged and deployed
- [ ] At least 2 weeks of `shadow_signals_json` data populated in production
- [ ] **Smart-contract wallet coverage decision** — `calculate_d2_score_from_transfers()`
      hardcodes `session_continuity_score: None` because token-transfer rows lack
      `isError`/`txreceipt_status`. Before promoting the signal into live D2
      weighting we need to either (a) extend the signal to the token-transfers
      path, or (b) explicitly document that session-continuity only applies to
      EOA-with-history wallets and gate the new D2 weight on
      `d2_data_source == "txlist"`. The current "silently None" behaviour will
      become a bug the moment the signal is load-bearing.

**Checklist (re-run when prerequisites met):**
- [ ] Query `scans.shadow_signals_json` for distribution of `session_continuity_score`
      across all wallets, broken out by `d2_data_source` (txlist vs tokentx)
- [ ] Confirm coverage rate — what % of AHS scans produced a non-None score?
      What % were excluded by the ≥20-tx / ≥3-session gating in
      `_calc_session_continuity_score`?
- [ ] Confirm score distribution is stable (not causing unexpected AHS shifts)
- [ ] Check how many wallets are triggering Budget Exhaustion shadow pattern
- [ ] Compute correlation between `session_continuity_score` and live `d2_score`
      on the same wallet — is it adding signal or duplicating existing D2 components?
- [ ] If distribution looks stable AND correlation is low enough to add signal,
      implement live D2 weighting (weight ~0.10, redistribute from
      `timing_regularity` and `retry_storm`)
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
