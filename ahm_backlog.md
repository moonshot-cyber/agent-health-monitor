# AHM Backlog

> Single source of truth for Agent Health Monitor. Update at end of every session.

---

## Current State (as of Mar 16 2026)

- **11 endpoints** live on Base mainnet at agenthealthmonitor.xyz
- **ERC-8004 registered** — agentId #32328 on Base mainnet
- **Nansen integration** — 4 direct API connections (labels, counterparties, PnL, related wallets)
- **Listed on:** Virtuals ACP (11 offerings), x402scan, Bankr Skills, agdp.io, 8004scan.io
- **Stack:** FastAPI, x402 SDK v2, Nansen API, Blockscout API, Base Mainnet, Railway
- **Repo:** github.com/moonshot-cyber/agent-health-monitor

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

---

## P3 — Tech Debt / Frontend Fixes

- [ ] **Proper OG banner (1200x630)** — `generate_og_banner.py` created, interim logo fix live. Complete this week
- [ ] Wash scan composite scoring refinement (see wash_spike_results.md for formula)

---

## P4 — Ecosystem Scanning / Monitoring

- [ ] **ERC-8004 ecosystem** — @quantu_ai, @cascade_fyi, Warden ($4M funded, Messari-backed)
- [ ] **FairScale (@fairscalexyz)** — Solana credibility scoring, track for potential cross-chain angle
- [ ] **PayAI Network** — monitor new analytics dashboard when it ships
- [ ] **Virtuals ACP** — maintain 11 offerings, update as new endpoints ship

---

## Content & Marketing (added Mar 13-16)

- [ ] **Demo recording** — Short Loom walkthrough of AHM for hackathon submissions, partnership pitches, cold outreach. Record this week
- [ ] **@BaseHubHB engagement** — weekly Base launch curator, 2.5K+ views per post. Engage every week for inclusion in roundups
- [ ] **Pin a new X post** — Draft and pin a fresh @AHMprotocol post once a meaningful milestone is hit (e.g. first organic payment, OpenServ selection, Coinbase PR merge). Replace the unpinned post with something current

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
