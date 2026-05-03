# AHM Backlog

> Single source of truth for Agent Health Monitor. Update at end of every session.

---

## Current State (as of May 2 2026)

- **14 endpoints** live on Base mainnet at agenthealthmonitor.xyz, including /ahs/batch and configurable routing via PUT /ahs/route/policy (PR #112)
- **ERC-8004 registered** — agentId #32328; live ERC-8183 evaluator on Base Sepolia (Jobs #1, #2, #3 completed Apr 7-27 2026)
- **Taxonomy POC complete:** 754 agents classified, 6 anchored categories, live at intelligence.agenthealthmonitor.xyz/taxonomy with methodology disclosure
- **Nansen integration** — 4 direct API connections (labels, counterparties, PnL, related wallets), Blockscout + token transfer fallback for smart contract wallets
- **Listed on:** Virtuals ACP, x402scan, Bankr Skills, agdp.io, 8004scan.io, 402index.io, Coinbase x402 ecosystem
- **Stack unchanged:** FastAPI, x402 SDK v2, Nansen, Blockscout, Base mainnet, Railway
- **Cross-registry tracking** across ACP / Olas / ERC-8004 / Arc / Celo
- **Agent scan count:** 13K agents (as of Apr 28 2026, supersedes earlier "5K" / "245 wallets" calibration figures from earlier scanning phases)
- **Nevermined walkthrough deck shipped** May 1 2026 — corrected D4 framing; May 6 follow-up call pending confirmation
- **@AHMprotocol restored** Apr 28 2026 — verified, branded, 57 followers, pinned post intact

---

## P1 — Active / In-Flight

### Homepage D4 Framing Fix (pre-Don visibility priority)

- [ ] **static/index.html D4 formula correction** — Homepage currently shows D4 as part of the AHS composite formula but production scoring code (`monitor.py:2972-2976`) only uses D1/D2/D3. Need to fix homepage to match production reality. AHM Verify stays as a separate product surface (not a fourth AHS dimension). Priority: fix before May 6 Nevermined call — Don should not see mismatched framing if he visits the site. First noted: May 2 2026

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

### Pending Outreach / Re-engagement

- [ ] **Alfred Zhang (@Alfredz0x) / OpenPasskey integration angle — relationship status to clarify** — April 3 2026 X exchange (visible in current notifications on restored @AHMprotocol account): Alfred replied substantively to AHM, identifying integration thesis — OpenPasskey HTTP tap logs + AHM on-chain signals as composite intelligence layer for physical-to-onchain payment verification (RIP-7212 / P-256 on Base L2). Alfred published a relevant deep-dive on dev.to (RIP-7212 / OpenPasskey card verification). His tweets quoted: "got it! checking email now. excited to explore the data share angle... could surface patterns neither of us sees alone" and "the data layer between physical taps and on-chain settlement is exactly where something like AHM would fit". Status of follow-through unclear in current records. Shadow analysis work referenced in earlier sessions but state unknown. Action: Locate prior email thread with Alfred (search Proton inbox for "Alfred Zhang" / "Alfredz0x" / "OpenPasskey"), locate any AHM repo or local notes on OpenPasskey, RIP-7212, P-256, card taps, read his dev.to article if not already, decide based on findings: re-engage with substantive update, close cleanly with appreciation, or treat as cold and let lie. Do not re-engage on X without first reconstructing the state. Premature reply risks looking either negligent (if commitments were made and missed) or naive (if forgotten context exists). @AHMprotocol account restoration (Apr 28) makes any future re-engagement easier — original handle Alfred tagged is now functional again. First noted: Apr 29 2026

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
- [ ] **Confidence-based routing policy — next refinement of PR #112** — Configurable routing currently supports grade mappings, escrow_disabled flag, and address allowlists with Grade C+ floor; does not support confidence-based rules. Schema change: add optional confidence_overrides: dict[str, str] to routing policy mapping confidence levels (INSUFFICIENT, LOW, MEDIUM, HIGH) to routing decisions (instant_settle, escrow, reject). Driver: Job #3 (Apr 27) surfaced this as a real methodology gap — INSUFFICIENT verdicts should not auto-route to reject regardless of grade. Cross-references: Bootstrapping Problem (zero-history wallet treatment); abstain() architectural argument made publicly on ERC-8183 thread, where confidence-based middleware was positioned as the right layer for this fix vs protocol-level abstain(). Public commitment made — this is the next build pencilled into the next session's prioritisation. First noted: Apr 27 2026
- [ ] **Trishir / vaultum-agent-bonds — Dynamic Bond Pricing Integration** — AgentBond requires agents to post performance bonds before accepting work. Trishir proposed that AHS behavioural scores could inform dynamic bond sizing — strong AHS score (consistent D2 patterns, clean D1) qualifies for lower bond percentage; weak score requires higher collateral. Credit score → loan terms model applied to agent work escrow; maps directly to AHM's existing D1/D2 scoring outputs without requiring new AHM infrastructure. Trust level on ETH Magicians now allows DM follow-up; relationship to be developed alongside ERC-8210 v2 engagement (Trishir is active in same thread). First noted: Apr 27 2026
- [ ] **ERC-8210 assessor schema alignment** — Review ERC-8210 draft spec when available. Ensure AHM's D1/D2/D3 output format matches the assessor schema (`type: rule`, `id: ahs-d1-d2-d3`, `verdict: APPROVE/REJECT`, `confidence: 0-1`). Consider publishing AHS verdicts as IPFS-pinned outputs to serve as verifiable assessor outputs (links to EAS integration backlog item). No action until ERC-8210 draft is shared
- [ ] **Agent Alerts System** — Build an alerting layer allowing integrators and agents to set custom alerts on wallet/agent activity. Use cases to think through before building: (1) **Integrator alerts** — notify when an agent they are routing drops below a configured AHS threshold; (2) **Abuse detection alerts** — flag agents attempting to manipulate routing policies or allowlists. Requirements and scope TBD — needs design spike before building
- [ ] **Evidence object in API responses** — Add a structured `evidence` object to `/ahs` and `/ahs/route/{address}` endpoint responses. Should include: per-dimension signal breakdown, weighted scores, data sources, and reasoning — not just the final grade. Rationale: strengthens "no black boxes" positioning, makes AHM more composable for `IRiskHook` implementations per ERC-8210. Inspired by RNWY's public disclosure of their evidence object pattern (Apr 8 2026). **Do not build yet.** Review after D3 Operational Stability is live
- [x] **/ahs/batch endpoint** (completed Mar 31) — POST `/ahs/batch` scores multiple agent wallets in a single API call. Up to 10 wallets per x402 call ($10.00 flat), up to 25 via API key (1 credit/wallet, partial results supported). Concurrent scoring with semaphore-limited RPC. PRs #42, #43, #44. Design partner validated (Alfred Zhang, httppay.xyz)

### Long-term Product Visions

- [ ] **AHM Shield** — Always-on agent protection middleware. SDK → Proxy → Enterprise Fleet. Business model evolution: pay-per-call → per-transaction → subscription → enterprise per-agent pricing. "Norton/CrowdStrike for the agent economy". Separate repo (moonshot-cyber/ahm-shield), separate PyPI package, partner API keys at wholesale rate (~50% retail), white-label wrappers.
- [ ] **Agent Title Registry** — Ownership transparency modelled on UK Land Registry (proprietorship, property, charges & restrictions)
- [ ] **Agent Certification** — On-chain attestation badges (Gold/Silver/Bronze), 90-day renewal. Multi-dimensional assessment covering ISO 25010, OWASP API Security Top 10, x402 v2 spec, OpenAPI, MCP, RFC 9457, ERC-8004, ERC-7730. Tiered certification on-chain via EAS.
- [ ] **Agent Power Index** — Comprehensive measure of agent digital footprint and influence
- [ ] **Micro-Utility Portfolio** — 20-50 tiny single-purpose x402 endpoints, separate strategy
- [ ] **ERC-8183 pre-flight check** — AHM as the health verification layer before a client funds an ERC-8183 job escrow. Natural fit: x402 (payments) + ERC-8004 (identity) + ERC-8183 (commerce) + AHM (health) = complete agent stack
- [ ] **AHM Forecast / Agent Risk Prediction Layer (Phase 3+)** — Predictive probability estimates derived from D1/D2/D3 + D4 verdict history; ARS-style underwriting basis. Answers "what is the probability this agent fails its next job?" rather than "is it healthy right now?" Source: ARS paper positioning, Apr 8 2026
- [ ] **Prediction Market Angle (Phase 4)** — AHM-powered binary outcomes pricing (e.g. "Will agent X complete next 5 jobs with ALLOW verdicts?"). Pricing engine = AHM scoring model. Requires AHM Forecast layer to be built first. Far future concept.
- [ ] **AHM Compromise Detection (concept)** — Behavioural baseline + Nansen labels + supply chain dependency monitoring. Detect when an agent has been compromised (wallet key stolen, agent logic replaced). Distinct from AHS scoring — binary detection rather than composite scoring.
- [ ] **EAS Integration (Phase 2)** — Publish AHM scores as on-chain attestations on Base via Ethereum Attestation Service. Makes AHS composable with any protocol that reads EAS. Phase 2 after design partner locked in. Source: ERC-8210 assessor schema alignment discussion.
- [ ] **ENS Agent Identity (ENSIP-25/26)** — ERC-8004 + ENS as emerging agent identity stack. Monitor ENSIP-25/26 progress; consider ENS name resolution in AHS API. No action until ENS agent naming conventions stabilise.
- [ ] **Multi-chain support (Solana priority, deferred)** — Declined PayAI Solana Alphathon Feb 2026 ($1K prize for week of multi-chain port). Decision: multi-chain deferred until Nansen work completed and feature set deeper. Solana entry conditional on broader multi-chain rationale. Revisit when product depth justifies.
- [ ] **Soulbound Tokens (Phase 2)** — Non-transferable AHS score attestation tokens. Phase 2 after Shield + design partner. Aligns with Agent Certification concept.

#### Research / Open Concepts (not on build path, periodic revisit)

- [ ] **Agent Intent Detection** — Measure drift from declared agent purpose to unrelated contract interactions. Build a 30-day baseline behaviour profile per agent, flag deviations. No commitment to build; track for periodic revisit. First noted: May 2 2026 (chat history extract)
- [ ] **Decision Quality / Learning Rate Measurement** — Measure whether agents adapt after failures (post-failure gas adjustment = healthy learning) or repeat identical failures (broken decision loop). Distinct from D2 retry storm detection. No commitment to build. First noted: May 2 2026 (chat history extract)
- [ ] **Hidden Backdoor Detection** — Adjacency to Project Glasswing. On-chain code analysis of agent contracts for suspicious patterns. No commitment to build; track as ecosystem awareness item. First noted: May 2 2026 (chat history extract)

### AHS Enhancements

- [ ] AHS D3 infrastructure probing — expand probe coverage (uptime, latency, error rates)
- [ ] Cross-dimensional pattern library — expand beyond Zombie Agent, Cascading Infrastructure Failure, Spam Drain, Gas Hemorrhage
- [ ] Trend tracking improvements — JWT-based score history
- [ ] **Per-dimension methodology versioning** — Currently `model_version: "AHS-v1"` is report-level. Patrick / ERC-8240 coordination may require finer per-dimension version tags (e.g. `d1_version: "D1-v2"`, `d2_version: "D2-v1"`) to signal when individual dimension methodologies change without a full composite version bump. Review when ERC-8240 alignment work progresses. First noted: May 2 2026
- [ ] **D4 Output Quality Score (concept, Phase 2)** — Potential future AHS dimension derived from aggregated post-transaction output quality verdicts across an agent's job history. If an agent consistently produces low-quality deliverables that get rejected by content verification evaluators (e.g. ThoughtProof-style DISSENT/HOLD verdicts), that rejection pattern should feed into their AHS score as a fourth dimension alongside D1 Solvency, D2 Behavioural Consistency, D3 Operational Stability. Would require either: (a) a data-sharing agreement with content verification providers, or (b) reading rejection patterns from on-chain ERC-8183 job history. No action until D3 is live and ERC-8183 mainnet job volume is measurable. Source: ThoughtProof research, Apr 7 2026
- [ ] **D4 fold-in gates report (decision input, not feature)** — Before deciding whether to accelerate D4 fold-in into AHS composite, produce a structured report covering: (1) current AHM Verify weekly verdict volume + trend; (2) D3 operational status; (3) engineering scope of fold-in (file changes, tests, JWT/cached score migration); (4) calibration methodology question (documented or needs deriving?); (5) customer impact analysis (how many AHS customers' scores would meaningfully shift). Backlog item gates remain: >500 verdicts/week + D3 operational stability. Worth re-examining whether gates still bind given commercial momentum. This is a decision input deliverable, not the fold-in itself. First noted: May 1 2026
- [ ] **ThoughtProof PLV (Plan-Level Verification) composition** — ThoughtProof proposed PLV as middleware-level signal that composes with AHM's behavioural score. Reasoning-trace verification: per-step verdicts, evidence support, provenance gaps, criticality flags. Explicitly does NOT modify ERC-8183 binary primitive. AHM positions PLV as occupying a potential future dimension within AHM's framework (potential 5th dimension or external composable signal), not as peer over-architecture. Long-term: if PLV matures, could be folded into AHS composite at the routing layer rather than as an internal dimension. **Update May 2 2026:** Job #3 verdict artefact (`verdict-job3.json`) provided to ThoughtProof publicly on EthMag ERC-8183 thread with hash verification (`keccak256(verdict-job3.json) == 0xbe9c3ba2eca135824a330c89b78889dbe0588a365d217d966a929ed59bf50915`). PLV pass invited as worked example of composition pattern. Composition pattern publicly affirmed: AHM = signal + boundary, PLV = process audit, ERC-8183 = binary primitive at protocol layer. First noted: May 2 2026
- [ ] **D5 Security Posture (spike, concept)** — Explore whether on-chain security signals could form a new AHM scoring dimension. Signal sources to investigate: flagged contract interactions, known exploit wallets, phishing-adjacent addresses, sanctioned addresses (Chainalysis/OFAC lists). Context: Project Glasswing (Anthropic + Linux Foundation + Microsoft/Google/NVIDIA coalition) launched Apr 8 2026 — focused on code/software vulnerability detection via Claude Mythos Preview. AHM angle: Glasswing secures agent code, AHM monitors live runtime behaviour — complementary positioning opportunity. Linux Foundation is also an x402 founding partner — potential ecosystem overlap worth monitoring. **Do not build yet.** Review after D3 Operational Stability is live
- [ ] **Reputation decay scoring (concept)** — Agents that were once healthy but have gone dormant should score progressively lower over time. Influence/score should decay without fresh verified on-chain activity — health must be continuously re-earned. Aligns with AHM's existing Zombie Agent pattern detection — decay is the scoring complement to zombie flagging. Rationale: validated by RNWY's public reputation decay concept (Apr 8 2026). **Do not build yet.** Review alongside D3 Operational Stability
- [x] **Smart contract wallet scoring** (completed Mar 24) — When `txlist` returns < 10 outgoing txs, `calculate_ahs()` now falls back to Blockscout V2 `token-transfers` API. D2 uses 4/8 signals (timing regularity, transfer diversity, counterparty breadth, activity gaps) with redistributed weights. Verified: ACP agents now score D2 10-68 (was baseline 50). EOA wallets unaffected (64/64 tests pass). New: `fetch_token_transfers()`, `calculate_d2_score_from_transfers()`, `_calc_token_transfer_diversity_score()`, `AHSResult.d2_data_source` field

### Bootstrapping Problem — Zero-History Wallet Treatment
- Currently a wallet with no on-chain history scores similarly to one with demonstrably degraded patterns due to D2 behavioural consistency weighting (70%)
- Raised by Bakugo32 (Arc/ERC-8183 protocol team) Apr 9 2026 after Job #7 evaluation — D grade on zero-history wallet triggered reject when HOLD was more appropriate
- Fix: introduce a separate scoring path for zero-history wallets — score as "Unrated" rather than mapping to D/E grade
- Unrated wallets should route to escrow (not reject) by default — client assumes counterparty risk with funds protected
- Review alongside D2 session continuity gate (April 21 2026)
- Job #3 (Apr 27 2026) was the first live test case of this gap. ThoughtProof's wallet scored 58/D with INSUFFICIENT confidence (zero transaction history). AHM called complete() rather than reject(), with reasoning hashed on-chain (tx 0x2a33b40e...) and explained publicly on the ETH Magicians ERC-8183 thread. Bakugo32 cited this case as design input for Treasury.sol fee structure
- Confidence-based routing is the committed next refinement (see new entry under P2 — Product Backlog → New Endpoints / Features). Public commitment made in the abstain() architectural reply on the ERC-8183 thread

---

## P3 — Tech Debt / Frontend Fixes

### Frontend / UI

- [ ] **AHM Logo — Homepage Link** — The AHM logo on the app page (agenthealthmonitor.xyz/app) has no link. It should link back to the homepage (agenthealthmonitor.xyz). Quick fix, low priority
- [ ] **Global Navigation on App Page** — No navigation bar present on the app page. Add a minimal nav bar with links to Homepage, Docs (docs.agenthealthmonitor.xyz), and App for consistency across the site. Low priority but worth tidying before design partner demos
- [ ] **Docs site: dedicated Downloads/Resources section** — Currently the AHM Getting Started PDF is accessible via a link on the Getting Started page (docs.agenthealthmonitor.xyz/getting-started) but is not discoverable from the side nav. Anyone landing on the docs site looking for downloadable assets (PDF guides, future case studies, integration patterns) has no clear path. Add a top-level "Resources" or "Downloads" nav item that surfaces all downloadable assets in one place. Pre-build the section structure so future assets (Nevermined integration walkthrough, suite framing one-pager, design partner case studies) land in a predictable home rather than getting buried as inline links across content pages. First noted: Apr 29 2026

### Other

- [ ] **Site analytics + source attribution** — No web analytics implemented on agenthealthmonitor.xyz (no GA, no Plausible, no Cloudflare Web Analytics surfaced). Cloudflare aggregate shows 4.11K unique visitors / 73K requests / 13.67% cache rate over 30 days, with April 25-28 spike correlating to ETH Magicians activity (Job #3, abstain() argument, Treasury observations) — suggests standards-engagement drives traffic but cannot prove it without referrer attribution data. Action: Implement Plausible (privacy-friendly, ~£8/month, GDPR-clean — fits AHM's trust positioning) or Cloudflare Web Analytics (free) on agenthealthmonitor.xyz to capture page-level views, referrer data, traffic source attribution. Goal: prove the standards-thread → site visit pipeline quantitatively, identify which content pages convert best, surface unexpected traffic sources (e.g., did Don visit the site after the bump email lands?). First noted: Apr 28 2026
- [ ] **Contact form audit** — Verify whether agenthealthmonitor.xyz currently has a working contact form for inbound enquiries. Documented contact channels in customer-facing materials are pablo@agenthealthmonitor.xyz and @agenttrust on X — both are friction-heavy compared to a one-click form. If form is missing or broken, this represents a direct conversion leak: every potential design partner equivalent to Don who lands on the site has to email cold or DM on Twitter, both of which are higher-friction than a one-click contact form. Action: open agenthealthmonitor.xyz in incognito, confirm form presence and function. If absent, build/restore as a high-priority fix. First noted: Apr 28 2026
- [ ] **Conversion analytics consolidation** — No single view of how site traffic converts to revenue. Stripe shows human API key purchases (currently zero). x402 revenue wallet shows agent micropayments. Application logs show endpoint hits. All separate, no consolidated funnel view. Action: maintain a fortnightly conversion summary mapping: visitors → API key signups → x402 paying calls → genuine inbound enquiries. Foundation for understanding which marketing channels (standards threads, Twitter, search) actually produce revenue. Pairs naturally with site analytics implementation (3d). First noted: Apr 28 2026
- [ ] **Public roadmap refresh (static/ahm-roadmap.html) — deferred** — Public roadmap is dated March 2026 and reads as stale. Refresh once strategic prioritisation conversation produces stable positioning. Likely changes: add suite framing (Health/Shield/Verify/Intelligence) if explicit positioning is the agreed strategic direction; refresh "LIVE" phase to reflect ERC-8183 evaluator role and configurable routing (PR #112); refresh "NEXT" phase based on agreed 60-day priorities from strategic conversation. Do not update piecemeal. Wait for stable positioning to avoid churning the public-facing version. First noted: Apr 29 2026
- [ ] **Evaluator daemon receipt-based fallback** — Daemon currently relies solely on eth_getLogs polling to detect EvaluatorAssigned events. Base Sepolia RPC indexing lag has been observed to exceed 24 hours, causing the daemon to miss assignments confirmed in transaction receipts (Job #3, Apr 26). Bakugo32's EVALUATOR.md guide specifies eth_getTransactionReceipt with topics[2] filtering as the resilient fallback. Implementation: extend evaluator-daemon.ts OnChainWatcher to maintain recent transaction hashes (received via webhook or polling JobFunded events on AgentJobManager) and as fallback fetch their receipts to extract any EvaluatorAssigned events with the daemon's address in topics[2]. Future-proofs daemon for similar indexer outages; Job #3 was first known incident requiring manual workaround. First noted: Apr 27 2026
- [ ] **Issue #140 — Wallet-primitive bucket in classify_agents_taxonomy.py** — Currently agents holding wallet primitives (Gnosis Safe, Multicall3) are classified as unclassifiable. Conflates two distinct cases: agents we have not yet built an anchor for vs agents whose on-chain footprint is fundamentally wallet operations rather than functional agent activity. Proposed three-bucket model: classified | wallet-only | unclassifiable. Improves taxonomy coverage interpretation and future categorisation methodology. Filed as GitHub issue Apr 26 2026 in PR #139 / 140 cycle
- [ ] **Configurable routing docs update** — PR #112 shipped configurable routing (PUT /ahs/route/policy) but public docs at docs.agenthealthmonitor.xyz have not been updated. Public commitment made in the abstain() architectural reply on ERC-8183 thread referenced PR #112 as the policy infrastructure foundation; integrators reading that post and looking for documentation will hit a gap. Action: update docs site with /ahs/route/policy endpoint reference, schema description, and example configurations including grade mappings, escrow_disabled, and address allowlists. Pre-document confidence_overrides slot for the upcoming confidence-based routing build so integrators see the trajectory. First noted: Apr 27 2026
- [ ] **Proper OG banner (1200x630)** — `generate_og_banner.py` created, interim logo fix live. Complete this week
- [ ] Wash scan composite scoring refinement (see wash_spike_results.md for formula)
- [ ] **USDC sweep automation** — Auto-sweep revenue wallet to cold storage/burner on a schedule. Nice-to-have operational hygiene. No priority until x402 revenue volume justifies the build.
- [x] **Evaluator daemon evidence persistence** — COMPLETED May 2 2026 (ahm-staking PR #6). See cross-reference under Arc Mainnet Migration — Evaluator Readiness for full details. Content-addressable JSON hashing now replaces template-string hashing; verdict files persist to /data/verdicts/ on Railway volume; /verdicts/:jobId endpoint serves JSON for external verification. First noted: May 2 2026
- [ ] **AHM Unlimited pricing decision** — Decide whether the $99/mo Unlimited tier should include D4 (AHM Verify) verdicts or keep them separately priced. Currently zero Stripe conversions across all tiers, so decision is not urgent but should be made before any pricing refresh. Source: chat history

---

## P4 — Ecosystem Scanning / Monitoring

- [ ] **ERC-8004 ecosystem** — @quantu_ai, @cascade_fyi, Warden ($4M funded, Messari-backed)
- [ ] **FairScale (@fairscalexyz)** — Solana credibility scoring, track for potential cross-chain angle
- [ ] **PayAI Network** — confirmed infrastructure dependency, NOT competitor (May 2 2026). AHM has used `https://facilitator.payai.network` as x402 facilitator since February 2026. PayAI's "Under the Hood: how trust is handled between agents" thread (May 2 2026) is positioning by an existing infrastructure partner, not a new entrant. Monitor new analytics dashboard when it ships. Complementary infrastructure positioning confirmed
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

### ERC-1705 — Trust Infrastructure for Agents and Assets
- Five-interface ERC drafted by Patrick Nicolas Badoui (avwatari.io); Nicopat is primary advocate
- Interfaces: IAttestation (MUST), IDecisionTrail (SHOULD), IAccountability (SHOULD), IRiskSignal (MAY), IRWAPassport (MAY)
- Reference implementation deployed: 19 contracts on Base mainnet, 7 on Gnosis Chain, 439 tests passing, 120+ tokenized assets scored continuously
- AHM is a textbook IAttestation implementer — AHS scoring output (factual, signed, multi-dimensional measurements with INSUFFICIENT confidence flags) maps directly to the IAttestation interface signature
- Patrick's adversarial constraints (attester collusion, score manipulation) explicitly require multi-evaluator implementations; AHM's multi-registry coverage (ACP, Olas, ERC-8004, Arc, Celo) is exactly that
- Patrick has 5 explicit feedback questions on the thread; AHM has lived experience of all 5 (interface granularity, score range, regime states, deployment topology, ERC-8192 composability)
- Action: read full spec, draft public response on ETH Magicians thread, decide whether to build a reference IAttestation implementation against AHM's AHS scoring as canonical AI-agent example alongside Patrick's RWA-focused reference deployment
- First noted: Apr 26 2026

### ERC-8239 / ERC-8240 — Agent Skill Registry + Quality Scoring
- ERC-8239 (Agent Skill Registry, bransdotcom): capability registry — what an agent can do; jobContract + jobId + manifestHash + taxonomy classification fields
- ERC-8240: quality scoring layer composable with ERC-8239; getQualityForAgent returns score (0-100) + trend + volatility + timestamp; AAA-CCC tier filtering for marketplace consumers
- ERC-8240 is functionally a standardisation of what AHM already does — score range, tier system, trend tracking all map directly to AHS
- Strategic choice: become reference implementation for ERC-8240 (the way ERC-8210 names AHM), or risk being left out as the spec firms up
- AHM has already engaged on ERC-8239 (capability composability point about taxonomy classifications as structured enums vs free-form strings); Nicopat replied Apr 27 asking how AHM sees the skill-to-quality link
- Action: respond substantively to Nicopat once strategic prioritisation allows; engage with ERC-8240 spec drafting; consider proposing AHS as the reference quality scoring methodology
- First noted: Apr 26-27 2026

#### subjectId Translation Layer Commitment (May 1 2026)
- AHM willing to converge on `keccak256(abi.encode(uint256 chainId, address registry, address agent))` as translation layer at ERC-8240 emission boundary if it becomes the standard
- Internal storage stays as raw lowercased addresses (db.py:302) — no migration needed
- Triggered by ERC-8239 coordination with bransdotcom; natural convergence point between AHM's existing storage scheme and the registry's addressable identity format
- First noted: May 1 2026

#### ERC-8239 Public Reply — Drafted, Held Pending Verification
- Three-question response drafted covering: (1) confidence format — string enum HIGH/MED/LOW/INSUFFICIENT plus 0-100 numeric score; (2) aggregation — weighted multi-dimensional + EMA-based temporal scoring, single-evaluator architecture; (3) subjectId — raw lowercased addresses, chain ambient per deployment, willing to converge on keccak256 translation layer
- Closing position: async preferred, subjectId as smallest-decision-first convergence point
- Held pending Patrick / ALIA verification (see below)
- First noted: May 1 2026

#### Patrick / Nicopat / ALIA Collaboration Proposal — HOLD Pending Verification
- Patrick (Nicopat on EthMag) sent private DM May 1 with five-axis collaboration proposal:
  1. Cross-evaluator calibration framework co-spec
  2. Confidence propagation in multi-evaluator systems co-research
  3. Bidirectional data exchange under MoU (AHM scan-level signals exchanged for ALIA on-chain detection signals)
  4. 8004×8240 quality bridge co-design
  5. ZK1 privacy-preserving evaluator (longer term)
- Also offering AHM reference in ALIA-published end-of-May/early-June industry report ("mutual credibility" framing)
- **Verification status (May 2 2026):** Verification audit completed. Results: 2 verified / 7 partially verified / 2 unverified / 1 contradicted (12 claims total).
  - Significant positive findings: Julien Piguet fully verified as professional GT4 racing driver and Avvatar co-founder; Racingboyz / AVR Racing flagship is real, multi-season competing team.
  - Significant concerning findings: oracle.alia.network (claimed live ALIA dashboard) redirects to expired domain parking page — directly contradicts Patrick's infrastructure claim; Emmanuel Hubert / SCOR board membership claim cannot be substantiated through public sources; Patrick Nicolas Badoui LinkedIn returns 404; no corporate registry records accessible for Avvatar Labs.
  - Net assessment: consistent with "early-stage operating entity with one strong real asset (racing team) and significant gap between marketing claims and verifiable infrastructure." Not consistent with elaborate fabrication; not consistent with established institutional player.
- **Engagement decisions per axis:**
  - Axis 1 (calibration framework on EthMag): PROCEED when Patrick posts. Public-thread, low-risk.
  - Axis 3 (bidirectional data exchange under MoU): RETRACTED by Patrick himself May 2 — verification posture worked as designed.
  - Axis 4 (end-of-May/early-June industry report co-publication): PROVISIONALLY ACCEPT subject to seeing structural draft before AHM's section is written. Reply if/when Patrick raises it: "Happy to consider. Before drafting AHM's section I'd find it useful to see the report's structural framing — how the three implementations are introduced, what the report says about ALIA's current production status, and the intended distribution. Editorial control on AHM's section is already understood."
  - Axis 5 (ZK1 privacy-preserving evaluator): Long-term, no immediate action.
- Audit saved at prompts/alia-patrick-verification-audit.md (gitignored, local only).
- Standing rule validation: Patrick's response to AHM's verification-before-engagement holding posture was to retract the most contentious axis, accept the call decline, and provide concrete entity detail. That's the desired counterparty behaviour. Treat as evidence the verification rule produces correct outcomes.
- **Hard constraints:**
  - Any signal-level data exchange involving Nevermined-derived signals requires explicit Nevermined consent first
  - No MoU language without prior verification
  - No private-call commitment until verification complete and technical advisor in loop
  - Async-first posture preferred
- First noted: May 1 2026

### ERC-8210 v2 active drafting
- JackyWang (ERC-8210 author) is in active v2 changelog drafting mode; v2 imminent
- Open questions on RoleCollusion, behavioural similarity, IIndependenceSignal vs IRiskHook hierarchy — all directly affect how AHM's D1/D2/D3 outputs compose in multi-evaluator scenarios
- Bakugo32 cross-referencing ERC-8183 settlement layer (EvaluatorSlashed event with jobId field) into ERC-8210 fileClaim, anchoring outcomes to slash records
- AHM is named as reference implementation in v1 (assessor type: ahs-d1-d2-d3); v2 changes risk weakening this status if AHM doesn't engage
- Pablo Castejon Espejo (separate person, ETH Magicians username "Pablo / AHM") is shaping v2 directly with substantive technical input
- Action: read full thread, decide whether to engage on v2 with AHM perspective on the open questions before changelog draft closes
- First noted: Apr 27 2026

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
  6. Generate a fresh evaluator wallet address before mainnet migration. The current Base Sepolia evaluator key was exposed in conversational context during testnet operations on April 26 2026. Acceptable risk on testnet (negligible funds, no economic value, no real reputation stake), but mainnet must start with a clean, never-exposed key. Steps: generate new wallet, fund with ETH and VRT from a fresh source, stake on mainnet EvaluatorRegistry, update Railway env vars in the ahm-staking project, notify Bakugo32 of the new evaluator address so future job assignments route correctly. Time this with the mainnet contract addresses being published — bundle the rotation into the natural mainnet transition rather than disrupting testnet operations.
- Arc testnet code now open source — review for wallet behaviour patterns that could feed AHM D1/D2 scoring signals
- **May 2 2026 — Daemon evidence persistence fix shipped (ahm-staking PR #6 merged).** Pre-fix problem: ERC-8183 evaluator daemon submitted on-chain reason hashes from a template string (keccak256("ahm-job-{id}-{grade}-{action}")), not from actual verdict content. Job #3 (manually completed) had content-addressable evidence; future daemon-processed jobs would not. Fix replaces template-string hashing with content-addressable JSON hashing — daemon now generates a 6-field verdict JSON matching Job #3's reference structure, persists to /data/verdicts/verdict-job-{id}.json on Railway volume, hashes the file bytes, submits that hash on-chain. New /verdicts/:jobId endpoint serves the JSON for external verification. 20 tests passing including Job #3 reference hash assertion. Strict failsafe: if write fails, no on-chain tx. Daemon currently PAUSED on Railway pending Step 7 (manual test job verification before resume). Repo: moonshot-cyber/ahm-staking, master at commit b168cbc.
- First noted: Apr 9 2026

---

## Phase 3 — Future Products

### AHM Verify — production status to verify

- Customer-facing materials (AHM Overview deck sent to Don Gossen Apr 15 2026; Getting Started guide drafted Apr 28 2026) describe AHM Verify as live at verify.agenthealthmonitor.xyz with POST /v1/outputs endpoint, $0.50 standard / $1.50 deep verdict pricing, six-model LLM adjudication panel (Claude, GPT-4o, Gemini, DeepSeek)
- Earlier backlog state (PR #143 and prior) said "do not build yet"
- Action required: Verify actual production state of AHM Verify before May 6 Nevermined call. Three possibilities:
  - Verify is genuinely live and the backlog was stale on this point
  - Verify is partially live (endpoint returns plausible output but not production-grade)
  - Customer materials over-positioned and Verify needs shipping work before May 6

- First-hand reading suggests Verify shipped further than backlog reflected, but explicit CC verification of the codebase, deployment status, and endpoint behaviour is required before treating Verify as a confident demo asset on the May 6 call
- Verify spike doc (AHM_VERIFY_DESIGN_SPIKE.md if exists) and prompt design notes should be located and reviewed
- First noted (verification gap): Apr 28 2026

**Architectural reasoning (subject to verification):**

- Standalone service (separate repo: `ahm-verify`, separate Railway project)
- Multi-LLM adjudication panel — original design: 3-model (Claude, GPT, Gemini); customer materials cite 6-model (Claude, GPT-4o, Gemini, DeepSeek) — reconcile count discrepancy during verification
- Core moat: AHM trust registry cross-reference makes verdicts stateful and identity-anchored — ThoughtProof cannot replicate without building a competing registry
- Architecture: Option C — standalone service with read-only access to AHM core via `/internal/agent-profile/{address}`
- Monetisation: **CONFLICT FLAGGED** — original spike pricing was $0.25/verdict (single), $0.35 combined verdict+AHS report, subscription TBD; customer materials cite $0.50 standard / $1.50 deep verdict. Verify actual pricing tier during codebase review.
- Submission flow MVP: client-submitted only. Phase 2: agent self-submission. Phase 3: ERC-8183 evaluator role
- D4 composite feedback into AHM core AHS: deferred until >500 verdicts/week and D3 is live
- Full feasibility spike saved at `docs/ahm_verify_spike.md`
- Positioning: Monitor (before) + Shield (during) + Verify (after) = complete agent health lifecycle
- Original guidance: **Do not build yet.** Next step: prompt design doc + 20-30 hand-labelled test set to validate panel agreement rate before any code
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

### ACP/Virtuals Integration — New Registry Type (Architectural)

ACP agents operate via ERC-4337 smart wallets routing through a single protocol
contract rather than individual wallet addresses. Adding Virtuals/ACP as a registry
source would require a fundamentally different data model — service-based scoring
rather than wallet-based scoring.

**What this would require:**
- New scanner: `acp_service_scanner.py` — monitors ACP protocol contract
  (`0xa6C9BA866992cfD7fd6460ba912bfa405adA9df0`) for job/service events
- New scoring model: score agent services/jobs rather than wallet addresses
- New DB schema: `service_scans` table alongside existing wallet-based scans
- Taxonomy enrichment: ACP API text fields (`description`, `jobs[]`, `offerings[]`)
  are the best classification signal — NLP classifier on free-text descriptions
- Scale: 41,946 ACP agents available via `acpx.virtuals.io/api/agents`

**Priority:** Low — significant architectural change, no immediate commercial driver
**Status:** Backlog — awaiting commercial justification

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

#### Per-category landing pages (SEO + content strategy)
- Each of the 10 v1 taxonomy categories gets a dedicated landing page on intelligence.agenthealthmonitor.xyz/taxonomy/{category}
- Goals: SEO discovery (long-tail searches like "trading agent", "verification agent"), first-mover taxonomy authority before competitors capitalise, citation bait for ERC discussions (1705, 8210, 8183, 8239/8240), conversion path to paid AHM endpoints
- Common page structure: hero with stat snapshot, definition and scope, sub-category breakdown, 3-5 real-world examples with project links and on-chain references, use cases, trust considerations specific to category, AHM endpoint CTAs, related categories, citations and further reading, methodology footer linking to taxonomy POC summary
- Schema.org JSON-LD Article structured data on every page for rich snippets; "Cite this page" snippet (BibTeX-style) for researcher citation; cross-page consistency in voice and depth
- Phased build: 3 anchor pages first (Financial, Intelligence & Analytics, Creative — the HIGH-coverage categories), then 3 MEDIUM, then 4 EMERGING. Estimated 2-3 hours per page including research; 20-30 hours total
- Long-term content strategy foundation; future Layer 3 visualisations plug into these pages rather than replacing them
- First noted: Apr 27 2026

#### Two-axis taxonomy v2 (function × sector)
- V1 taxonomy classifies agents by function (what they do): 10 categories shipped April 2026
- Real-world deployment patterns indicate a parallel sector axis (where they operate / who governs them) is becoming equally significant — driven by emerging sovereign-scale procurement (UAE 50% government services to agentic AI, Apr 2026), regulated finance positioning (Franklin Templeton via t54.ai), and sector-specific trust requirements that don't reduce to function categories alone
- V2 design: each agent gets both a function tag (existing 10 categories) and a sector tag (proposed: Public Sector, Financial Services, Healthcare, Sales & Marketing, Defence, Supply Chain, Research & Science, Consumer / Retail, Industrial, Cross-sector). Examples: UAE government trading bot = Financial Agent + Public Sector; pharma research automation = Research Agent + Healthcare
- Strategic value: enables sector-specific commercial intelligence reports (high-leverage content); doubles SEO surface area (function landing pages + sector landing pages); positions AHM to reason about sector-specific trust requirements; creates natural extension path as new domains emerge
- Do not build until v1 function pages are stable and at least 3 anchor pages have shipped
- First noted: Apr 27 2026 (UAE 50% government services to agentic AI announcement as validating signal)

---

### Layer 3 — Enriched Intelligence Reports (future / v3+)
Fine-grained KPI reports organised by taxonomy category. Requires both the taxonomy (Layer 2) and data enrichment work to be completed first.

Examples of future reports:
- What are trading agents actually doing vs utility agents?
- How does agent health differ between finance and defence sectors?
- What tasks are agents performing, and how do those differ by sector?
- Which registries host which categories of agent?

#### Commercial behavioural intelligence (Layer 3 anchor — Financial agents slice)
- Cross-reference AHS scoring data with transaction-level behavioural data to surface commercial differentiators between healthy and unhealthy agents
- Initial slice: Financial Agents category (413 agents post-Phase 2 classification)
- Metrics to explore: average transaction value, spending velocity, counterparty diversity, failed-transaction ratio, gas spend efficiency, dwell time between transactions, ratio of value-creating vs value-destroying activity
- Hypothesis: B-grade Financial agents will show fundamentally different commercial signatures from D-grade ones, and those signatures are themselves saleable insight
- Strategic value: bridges AHM's two existing data assets (scoring + behavioural) into shareable narrative content; high SEO and social shareability; validates AHM as authoritative source of commercial intelligence on the agent economy not just a scoring API; citation bait for media and ERC discussions; reuses Phase 2 taxonomy classifications directly with no new data infrastructure required
- Sequencing: build after taxonomy landing pages so Financial Agents page can launch with this analysis as anchor content; other category pages follow same template; Layer 3's recurring content engine
- First noted: Apr 27 2026

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

- [ ] **Proactive Ecosystem Scanning** — Already P4 in backlog. Confirmed as the right next build based on market analysis. The index itself demonstrates demand for health/quality monitoring (health_status, reliability_score, uptime_30d fields are core to the directory)

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

### AgentProof (@agentproof, agentproof.sh) — Complementary ecosystem partner

- On-chain reputation oracle, 158K+ agents indexed, 21 chains, live oracle
- Reclassified from "ELEVATED THREAT" (April 8 2026) to complementary positioning following deeper review: AgentProof addresses on-chain reputation aggregation; AHM addresses behavioural trust scoring with INSUFFICIENT confidence handling and ERC-8183 evaluator role. Different layers of the trust stack, not competing offers.
- The ERC-8183 evaluator collaboration with ThoughtProof (Job #2 and Job #3 cycle) demonstrates the cooperative dynamic in the trust-infrastructure space — multiple specialised evaluators and oracles co-exist by addressing different trust questions.
- Strategic monitoring rather than competitive response: track AgentProof's API surface and pricing for ecosystem mapping, but no defensive positioning required.
- First noted: Apr 8 2026 (as threat). Reclassified: Apr 28 2026.

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

### Parallel Web Systems (@p0, parallel.ai) — TIER 3 ECOSYSTEM WATCH
- Founder Parag Agrawal (former Twitter CEO); $130M total funding ($100M Series A Nov 2025 co-led by Kleiner Perkins and Index Ventures, $740M valuation, board seat for Mamoon Hamid at KP)
- Product: web infrastructure APIs purpose-built for AI agents — Search, Extract, Deep Research; x402 payment support announced Apr 27 2026 starting at $0.01/call, tagging @CoinbaseDev and @linuxfoundation
- Not a direct competitor to AHM — horizontal web research API, not agent trust infrastructure
- Long-term ecosystem angle: Parallel's enterprise customers (M&A research, insurance underwriting, sales analysis) are exactly the high-value agent buyers AHM is positioned to score; their planned "open market mechanism" for content publishers implicitly requires identifying and trusting agent buyers — AHM scoring is a natural fit. Don't pitch yet (AHM scale-mismatch); revisit when AHM has measurable mainnet evaluator volume
- Their success expands the addressable market AHM serves — tailwind, not headwind
- First noted: Apr 27 2026

### termixai / TermiX-official AACP — INVESTIGATION REQUIRED
- Posted on ERC-8183 thread Apr 27 2026: "We propose our new business agreement based on 8183. GitHub - TermiX-official/aacp-whitepaper: Agent Autonomous Commerce Protocol"
- Same general space as Verdict Protocol — a layer/protocol proposal on top of ERC-8183
- No analysis done yet; classify as competitor / partner / irrelevant after reading the whitepaper
- Action: read whitepaper and TermiX repo, classify, then either add a substantive Competitive Intelligence entry or note as "reviewed and irrelevant" in a deferred update
- First noted: Apr 27 2026

---

## Active Design Partner Relationships

### Nevermined (Don Gossen) — primary B2B design partner

- Status: active design partner with live partner API key
- Relationship sequence:
  - Apr 8 2026: Cold contact via Nevermined website form (positioned AHM as "trust gate within Nevermined payment flows")
  - Apr 9 2026: Don replied within 17 hours requesting a Calendly link
  - Apr 15 2026: Initial 30-min call held
  - Apr 15 2026 (post-call): 3-month complimentary access offered (coupon TEST-ELITE), enterprise partner API key issued (ahm_live_8bc21d3b..., partner_id "nevermined", expires Jul 15 2026)
  - Apr 28 2026: Bump email drafted (sending Apr 29) including AHM Getting Started guide as additional context
  - May 1 2026: `AHM-Nevermined-Integration-Walkthrough.pdf` shipped with corrected D4 framing. Email reframed as "Design partnership — follow-up ahead of next call"
  - ~May 6 2026: Follow-up call — awaiting Wednesday 5pm GMT confirmation for May 6. Pre-call prep needed: D1 Chainalysis enrichment live, AHM Verify e2e tested

- Two endpoints positioned as primary value:
  - POST /ahs/{address} — pre-payment trust scoring (the trust gate)
  - POST /v1/outputs via verify.agenthealthmonitor.xyz — post-payment output quality scoring

- AHM Shield SDK integration path proposed (3-line drop-in pattern)
- Walkthrough deck deliverable: SHIPPED May 1 2026 (`AHM-Nevermined-Integration-Walkthrough.pdf`). Slide 4 includes "configurable for swarm and high-throughput contexts" callout — captures Don's earlier swarm-agent feedback that one-size-fits-all routing wouldn't work for multi-agent systems. Per-integrator threshold configuration and bulk allowlist support framed as design-partner-feedback driven, not Don-specific.
- Strategic significance: Only confirmed B2B design partner with live partner API key. May 6 call is the primary commercial inflection point for next 60 days. Conversion would validate the entire suite commercial positioning. Failure forces fundamental rethink of B2B angle.
- Don's state pre-call: travelling Portugal → LA the week of Apr 28; testing status unknown until May 6 call surfaces it
- First noted in backlog: Apr 28 2026 (relationship dates back to Apr 8 2026)

---

## Strategic Positioning

### Standards-layer engagement as primary strategic surface

AHM has emerged in 2026 Q2 as a substantive contributor to multiple ERC-companion specifications, with standards-thread activity tied directly to traffic spikes and design-credibility wins. Pattern recognised across:

- ERC-8183 (Bakugo32): AHM's abstain() architectural argument accepted, Treasury fee design observations adopted (MIN_BUDGET to 1 USDC, dynamic pegging noted, fixed-fee per-evaluation reinforced). Bakugo invited AHM to review INTERFACES.md before finalising Treasury implementation.
- ERC-8210 v2 (JackyWang): AHM is named as reference implementation in v1; Pablo Castejon Espejo (separate person, ETH Magicians username "Pablo / AHM") is shaping v2 directly. Active changelog drafting.
- ERC-1705 (Patrick Nicolas Badoui): AHM is textbook IAttestation implementer. 19 contracts on Base mainnet, 7 on Gnosis Chain, 439 tests passing reference deployment. Patrick has 5 explicit feedback questions on the thread.
- ERC-8239 / ERC-8240 (bransdotcom / Nicopat): ERC-8239 capability registry; ERC-8240 quality scoring layer. Nicopat confirmed (Apr 29) that an ERC-8183 adapter has been built mapping AHM-shaped verdicts into the ERC-8240 24-slot ring buffer. Three-layer composition validated.
- ADW (Oak / Jinn Network): companion to ERC-8004, document-level trust positioning. Composable with AHM's wallet-level scoring.

Strategic implication: Standards-engagement work is AHM's highest-leverage marketing channel (causally tied to traffic spikes per Cloudflare data) AND highest-leverage credibility builder (publicly documented design contributions). Worth treating as a coherent workstream rather than handling each ERC ad hoc.

Action: review whether to formalise a "standards engagement strategy" as an explicit workstream. Decision deferred to next strategic prioritisation conversation.

First noted: Apr 29 2026

### AHM Dimensional Framework as Over-Architecture (standing positioning rule)

- **Hold "trust layer for the agent economy" framing.** Reject "specialist evaluator in a stack" framing in standards-track conversations.
- Applied in: May 2 ThoughtProof PLV draft reply (AHM positions PLV as dimension within AHM's framework); May 1-2 ERC-8239 Patrick reply draft (AHM as score authority, not as peer evaluator in a federation).
- Rationale: AHM's multi-dimensional model (D1/D2/D3 + future D4/D5) is the over-architecture into which external signals compose, not one signal among peers. This framing protects AHM from being commoditised into "one of N evaluators" in ERC-8210/8240 contexts.
- When to apply: any public reply on ETH Magicians, any positioning deck, any partnership framing where AHM's role could be minimised.
- First noted: May 2 2026

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

## Operational

### X / @AHMprotocol account status

- Original @AHMprotocol handle restored Apr 28 2026 after several weeks suspended
- Verified, branded, 57 followers preserved, pinned post intact (April 7 AHM Shield announcement)
- Premium subscription remained active throughout suspension (no chargeback action taken; account is now functional and paid up)
- Strategic decision: X is an auxiliary channel for research and comms, not a primary engagement surface. Empirical engagement was low pre-suspension and platform reliability has shown itself unreliable. Primary public engagement remains ETH Magicians threads and direct outreach.
- @agenttrust (fallback handle used during suspension) running in parallel for now; quiet retirement when convenient.
- First noted: Apr 28 2026.

### Standing Process / Capability Gaps (May 2 2026)

- [ ] **Cross-session context capability gap** — Claude Code has no direct access to Claude.ai conversation history. Manual workaround in place: `prompts/ahm_chat_history_extract_2026-05.md` is the curated cross-session reference artefact. Full Claude.ai data export saved at `C:\Users\Pablo\claude-exports\2026-05-02\` (gitignored). Future export pattern: Settings → Privacy → Export Data → save outside repo → run filter pass to extract AHM-relevant content into `prompts/ahm_chat_archive/` (currently gitignored). Revisit if Claude.ai adds project-level export API.
- [ ] **Technical advisor needed** — Solo founder vs well-resourced multi-person teams in adjacent space (t54 $5M, Parallel $130M, Valory $13.8M). Worth identifying someone in agent-economy/crypto space who can sanity-check protocol-level conversations before they go further. Especially critical for Patrick/ALIA-type proposals involving MoUs, data sharing, or co-publication.
- [ ] **Verification-before-engagement default** — When new counterparties propose data sharing, MoUs, or co-publication, default posture is verify-then-respond rather than respond-then-verify. 48-hour delay is normal counterparty behaviour. Applied to: Patrick/ALIA (May 1), any future inbound.
- [ ] **Async-first posture for protocol coordination** — Keep substantive standards-track coordination on public threads where it belongs; private channels reserved for things that genuinely can't be said publicly. Decline calls until technical depth is matched by preparation or a co-author. Applied to: Patrick/ALIA (declined call), ThoughtProof PLV (public thread preferred).
- [ ] **Public-thread replies don't volunteer schedule status (May 2 2026)** — AHM's positioning ("trust layer for the agent economy") doesn't tolerate self-flagellation in standards forums. Internal commitments not yet shipped get captured in the backlog, not announced as outstanding in public threads. Disclosure is for things that genuinely warrant it (security findings, methodology changes), not for normal scheduling lag. If asked directly about an unshipped commitment in a public thread, the answer is "in progress, scoped, lands alongside [related work]" — present-tense, brief, no apology, said once. First applied: ThoughtProof PLV reply May 2 2026 (confidence-based routing self-disclosure removed from final draft).

### Git Hygiene Note

- Squash-merge workflow leaves branches showing as "unmerged" in `git branch --no-merged` after their content reaches master. Future audits should use content diff (`git diff branch master -- file`), not ancestry checks. Cleanup of post-PR branches should be part of merge workflow itself. First noted: May 2 2026

### Active public commitments — tracker

Public commitments made in ETH Magicians threads, X exchanges, and design partner emails that require follow-through:

- Confidence-based routing build — committed in abstain() architectural reply on ERC-8183 thread. Foundation: extend PR #112 routing policy with confidence_overrides schema. Driver: Job #3 INSUFFICIENT verdict surfaced the gap.
- Per-registry metric label + cross-registry overlap stat on intelligence page — committed to Oak / @tannedoaksprout in 4-tweet thread. Two specific changes to ship: (a) explicit metric definition ("wallets with this registry membership"); (b) overlap stat surfaced on the page with footnote on Olas → ERC-8004 syndication.
- INTERFACES.md review — committed to Bakugo32 in Treasury thread reply. Bakugo is drafting the diff; AHM committed to review before he finalises implementation.
- ERC-8240 alignment dig — committed to Nicopat in ERC-8239 thread reply. Substantive engagement on skill-to-quality link, evidenceHash → evaluator input mapping. Picked up wherever Nicopat directs (8239 or 8240 thread).
- May 6 walkthrough/onboarding deck for Don Gossen (Nevermined) — committed in Apr 15 post-call email.

Track status of each fortnightly. Non-delivery on stated commitments degrades AHM's design-contributor positioning credibility.

First noted: Apr 29 2026

### Standards-track milestones

- **May 2 2026 — ThoughtProof PLV verification of Job #3 completed publicly on EthMag ERC-8183 thread.** AHM provided the verdict-job3.json content with hash verification (keccak256(file) == 0xbe9c3ba2eca135824a330c89b78889dbe0588a365d217d966a929ed59bf50915). ThoughtProof ran their canonical PLV pass and posted: "Result: ALLOW. Cross-model review agreed. AHM's INSUFFICIENT flag was handled correctly as an evidence-boundary, not as an adverse behavioral signal. complete() was process-faithful; reject() would have acted on a verdict AHM itself marked as not defensible." Artifact SHA-256: 3599a5cc80408874a169d6dab7abc4ff36142131ee456bcda95a08cbd8daafa4. AHM acknowledgement reply posted same day with corrected lineage (Bakugo32's role framed as raising a neutral open question, not as position-holder; AHM owned the strategic call; PLV provided independent verification after the fact). Composition pattern (AHM = signal + boundary, PLV = process audit, ERC-8183 = binary primitive at protocol layer) now operationalised with worked example. First external peer-evaluator endorsement of AHM's confidence-boundary methodology. Strategic significance: positions AHM credibility for future investor pitches, design partner conversations, and standards-track engagement.

---

## Commercial / Revenue

### Stripe revenue baseline (as of Apr 28 2026)

- Live Stripe dashboard (account acct_1TGOSnBtIhkG7uE2, pk_live_) shows: gross volume £0.00, net volume £0.00, 0 new customers, no top customers, no payment data for the last 4 weeks
- Despite 8,800+ agents scanned, 14 endpoints live, growing site traffic (4.11K unique visitors / 73K requests over 30 days, accelerating in late April), and substantive standards engagement, AHM has no Stripe-paid customers
- Pricing tiers (£9 Starter / £39 Pro / £99 Unlimited) have produced zero conversions since launch
- Possible explanations (to investigate, not assume):
  - Site traffic is researcher / standards-watcher audience, not buyer audience
  - Pricing model wrong (too high, wrong tiering, wrong unit of consumption)
  - Conversion path broken (no contact form audit pending; CTA placement unverified)
  - Product-market fit at API-key level not yet there; B2B angle (Nevermined) is the right fit
  - Likely some combination of the above

- x402 wallet revenue (Base mainnet) — separate revenue stream, not yet checked. May or may not show actual revenue from agent-paid micropayments.
- Action: check x402 revenue wallet inbound transactions over last 30 days to determine whether AHM has zero revenue full stop, or zero Stripe revenue with x402 activity
- Strategic implication: AHM's traction is currently in attention and design credibility, not revenue. The Nevermined B2B path is materially more important than the Stripe self-serve path until conversion improves on the latter.
- First noted: Apr 28 2026

---

## Adjacent Builders Worth Tracking

### Jinn Network (Oak / @tannedoaksprout) — adjacent builder, ADW spec author

- Project: Jinn Network — decentralised reasoning framework for AI agents. Agentic LAMP stack (LLM + Agents + MCP + Persistence). Built as reasoning layer on top of Olas (Autonolas).
- GitHub: github.com/jinn-network (monorepo + adw-spec + jinn-node + jinn-gemini + shared-skills)
- Authored ERC-companion spec: Agentic Document Web (ADW) — open specification for decentralised agent document identity, discovery, trust, and provenance. Companion to ERC-8004.
- Strategic positioning: ADW handles document-level trust (provenance, attestation, identity); AHM handles wallet-level behavioural trust. Naturally composable, not competing.
- Reclassification (Apr 29 2026): Initial framing as "Olas insider" was overstated. Oak's own Apr 28 reply revealed he's no longer working on Olas core; his Olas knowledge is informed but second-hand. Primary Olas contact is Valory AG (@valoryag) — see below.
- Relationship state: First contact via X reply on registry double-counting question (Apr 28 2026). Substantive technical exchange about Olas → ERC-8004 syndication methodology. Oak directionally answered "close to 100%" syndication ratio and referred to @valoryag for confirmation. Public methodology improvements committed (per-registry metric labelling + cross-registry overlap stat).
- Action: maintain conversation through X reply when natural; engage substantively on ADW spec if it becomes formalised; consider ADW + AHM integration angle once both specs stabilise. No urgent action required.
- First noted: Apr 28 2026

### Valory AG (@valoryag, valory.xyz) — Olas / Pearl ecosystem primary contact

- Company: Valory AG, Zug Switzerland, joined X September 2021, 2,826 followers
- Funding: $13.8M raise led by 1kx (Feb 2025); Olas Accelerator with $1M grants programme; Pearl ("Agent App Store"); currently hiring
- Strategic relevance:
  - Primary contact for Olas (formerly Autonolas) — they are the company behind it
  - Pearl as agent marketplace creates natural surface for AHM trust scoring integration
  - Olas as registry layer creates natural surface for AHM ecosystem coverage

- Initial contact: Apr 29 2026 — public X reply on Oak thread tagging @valoryag with cross-registry overlap methodology question. Low-friction first contact. No reply yet at time of writing.
- Action: if Valory replies on the methodology question, follow up substantively; either way, the relationship is now warmly initiated. Engage further post-Nevermined call (May 6+) to avoid fragmenting attention.
- First noted: Apr 29 2026

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

### Tier 2.5 — Integration Angle Identified, Engagement Pending

- [ ] **Pagga (@obchakevich, 67.1K followers)** — Autonomous OS for crypto company back office; Polygon/Visa/Avalanche/Dune/Artemis trust signals. Natural AHM integration point: pre-transaction check for Pagga's treasury agents before they execute autonomous financial operations. Outreach reply drafted but engagement status unclear. Source: chat history, earlier session
- [ ] **Agent Agora (GitHub agentkit issue #958)** — Offered to share participating agent wallets for bulk analysis. Free data source for trust registry expansion. Follow up when bandwidth allows.
- [ ] **Venice (@venice_ai)** — Privacy-focused AI inference platform; recently x402-integrated. Candidate model provider for AHM Verify v0.2 adjudication panel (adds privacy-preserving inference option). Monitor.
- [ ] **APINow (apinow.fun)** — x402 API marketplace. List AHM endpoints when @AHMprotocol account is active and site presence is strong. Low priority.

### Tier 3 — Monitor and Engage Opportunistically

- [ ] **Sats4AI** — Full AI suite on L402. Different payment rail but same developer community
- [ ] **Lightning Enable** — US government data proxy. Complementary positioning, no overlap
- [ ] **proxy402 / The Ark** — LLM inference proxies. Agents using these are AHM's target users

---

## Content & Marketing (added Mar 13-16)

- [ ] **Demo recording** — Short Loom walkthrough of AHM for hackathon submissions, partnership pitches, cold outreach. Record this week
- [ ] **@BaseHubHB engagement** — weekly Base launch curator, 2.5K+ views per post. Engage every week for inclusion in roundups
- [ ] **Pin a new X post** — Draft and pin a fresh @AHMprotocol post once a meaningful milestone is hit (e.g. first organic payment, OpenServ selection, Coinbase PR merge). Replace the unpinned post with something current
- [ ] **EtherMage / Jesse Pollak X thread reply (drafted, not posted)** — "Agent society" 6-point manifesto thread (replying to Jesse Pollak quoting Marc Andreessen). JayM9x's "who enforces good rewarded, bad punished?" reply is the engagement hook. Draft reframes "enforcement" as economic cost-structure shift, mentions AHM/agenthealthmonitor.xyz descriptively in one line, closes on calibration as harder problem. Post when timing is right (e.g., after a milestone that adds credibility to the claim). First noted: May 2 2026

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

### Session Continuity Shadow Mode — DEPRIORITISED (April 2026)

**Status:** DEPRIORITISED — April 2026

**Reason:** Coverage remained minimal despite two fix attempts (shadow wiring
PR #109, threshold change PR #121). PR #121 (lowering `_MIN_TX_COUNT` from 20
to 10) was reverted after a production regression — session continuity coverage
dropped from 46 agents (0.4%) to 2 agents (0.2%) following the threshold change.
The original design partner use case driving this investigation is no longer
active. D2 is live and functioning with its existing signals. Session continuity
remains architecturally sound but is not worth further investment at this time.
Revisit if a specific commercial use case emerges.

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

### Completed Apr 27 2026

- [x] **Job #3 cycle** — first live ERC-8183 evaluator action — ThoughtProof scored 58/D with INSUFFICIENT confidence; AHM called complete() with verdict JSON hashed on-chain (verdict hash 0xbe9c3ba2..., tx 0x2a33b40e...). Public transparency post on ETH Magicians ERC-8183 thread explaining INSUFFICIENT confidence reasoning. Followed up with abstain() architectural argument (5-point defence of binary terminal states + middleware approach) and Treasury.sol fee design observations (3 concrete points on charging on reject(), 80/20 split, MIN_BUDGET dynamic pegging). Bakugo32 cited Job #3 reasoning as design input for Treasury.sol fee structure. AHM positioning shifting from "evaluator running on the protocol" to "design contributor whose decisions shape the protocol"
- [x] **AHM Intelligence dashboard + Taxonomy v1 publication** — taxonomy POC complete (754 agents classified, 6 anchored categories: Financial, Intelligence & Analytics, Research, Verification, Orchestration, Identity & Trust; 1 latent: Creative; 3 pending: Infrastructure, Commerce, Physical World). Live page at intelligence.agenthealthmonitor.xyz/taxonomy with ANCHORED/LATENT/PENDING badges, methodology disclosure, link to POC summary. Stats footer per category. PRs #137, #138, #139, #141 (taxonomy POC), #142 (backlog evaluator key rotation), and ahm-intelligence PR #1 (live page update) all merged

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
