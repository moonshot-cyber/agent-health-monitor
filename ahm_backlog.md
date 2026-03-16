# AHM Backlog

> Single source of truth for Agent Health Monitor. Update at end of every session.

---

## Current State (as of Mar 16 2026)

- **11 endpoints** live on Base mainnet at agenthealthmonitor.xyz
- **Nansen integration** — 4 direct API connections (labels, counterparties, PnL, related wallets)
- **Listed on:** Virtuals ACP (11 offerings), x402scan, Bankr Skills, agdp.io
- **Stack:** FastAPI, x402 SDK v2, Nansen API, Blockscout API, Base Mainnet, Railway
- **Repo:** github.com/moonshot-cyber/agent-health-monitor

---

## P1 — Active / In-Flight

### Distribution & Partnerships (added Mar 13-16)

- [ ] **OpenServ/SERV Foundry hackathon** — Submitted, awaiting response on selection. If selected, ship on OpenServ within 2 weeks. Prep needed: demo recording (Loom), add `issa-me-sush` as GitHub collaborator on `moonshot-cyber/agent-health-monitor`
- [ ] **KAMIYO (@kamiyoai)** — Singularity platform launching this week, has "hard risk checks" pre-agent-funding. AHM `/risk` and `/ahs` are natural pre-funding health verification layer. Reach out
- [ ] **synthesis.md** — Agent-native hackathon/registry at synthesis.md, registration flow via `curl -s https://synthesis.md/skill.md`. Investigate and register Monday
- [ ] **8004scan.io** — ERC-8004 equivalent of x402scan. Register AHM here for discovery
- [ ] **@BaseHubHB** — Weekly Base launch curator, 2.5K+ views per post. Engage every week to get included in future roundups
- [ ] **FairScale (@fairscalexyz)** — Solana credibility scoring, complementary to AHM on Base. Monitor and engage
- [ ] **Messari x402 integration** — Explore consuming Messari's x402 endpoints (market data, token unlocks, X mindshare) to enrich AHS D3 scoring. Also a co-marketing story
- [x] **PayAI Network** — Already registered as merchant, new analytics dashboard coming. No action needed
- [x] **Coinbase PR #1207** — Logo fix pushed Mar 16, approved & merged
- [ ] **Bankr Skills PR #195** — Bumped Mar 8, chase this week if no response

### ERC-8004 (URGENT — added Mar 13-16)

ERC-8004 deployed on Ethereum mainnet Jan 29 2026. Identity + reputation standard for agents, complements x402. AHM health scoring sits naturally on top of this stack.

- [ ] Monitor ecosystem: **@quantu_ai** (indexing attestations), **@cascade_fyi** (SATI attestations), **Warden** ($4M funded, Messari-backed)
- [ ] Consider ERC-8004 integration angle for AHS — health scores as attestation data

---

## P2 — Product Backlog

### New Endpoints / Features

- [ ] **Messari signal integration** — X mindshare, token unlocks, fundraising data as enrichment layer for AHS D3 or new premium endpoint
- [ ] **B2B customer angle** — Agent credit providers, lending protocols, agent creation platforms are natural AHM customers (not just agents themselves). Develop pitch for this segment
- [ ] **Wash Phase 2** — Token approvals scan + dead contract detection (deferred from wash MVP, see wash_spike_results.md)

### Long-term Product Visions

- [ ] **AHM Shield** — Always-on agent protection middleware. SDK → Proxy → Enterprise Fleet. Business model evolution: pay-per-call → per-transaction → subscription → enterprise per-agent pricing. "Norton/CrowdStrike for the agent economy"
- [ ] **Agent Title Registry** — Ownership transparency modelled on UK Land Registry (proprietorship, property, charges & restrictions)
- [ ] **Agent Certification** — On-chain attestation badges (Gold/Silver/Bronze), 90-day renewal
- [ ] **Agent Power Index** — Comprehensive measure of agent digital footprint and influence
- [ ] **Micro-Utility Portfolio** — 20-50 tiny single-purpose x402 endpoints, separate strategy

### AHS Enhancements

- [ ] AHS D3 infrastructure probing — expand probe coverage (uptime, latency, error rates)
- [ ] Cross-dimensional pattern library — expand beyond Zombie Agent, Cascading Infrastructure Failure, Spam Drain, Gas Hemorrhage
- [ ] Trend tracking improvements — JWT-based score history

---

## P3 — Tech Debt / Frontend Fixes

- [ ] **Update pinned tweet** on @AHMprotocol from "9 endpoints" to 11
- [x] **Fix OG/Open Graph image** on agenthealthmonitor.xyz — ~~currently showing old cached "Wallet Health Analytics" branding~~ Interim fix: updated og:image to self-hosted ahm-logo.png, updated descriptions. Force refresh via Twitter Card Validator (cards-dev.twitter.com/validator) after deploy
- [ ] **Create proper OG banner image** (1200x630 landscape) with current Agent Health Monitor branding and update og:image meta tag — replace interim logo fix
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
