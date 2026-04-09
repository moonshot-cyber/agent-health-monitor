# AHM Verify — Feasibility Spike

Written feasibility summary for a post-transaction output-quality scoring service, architecturally separate from AHM core but part of the AHM product suite. No code or PRs — analysis only.

---

## 1. Positioning & the Core Thesis

AHM today answers **"should we trust this agent before we pay it?"** — pre-transaction counterparty health via D1/D2/D3.

AHM Verify would answer **"did the agent actually deliver what it was paid for?"** — post-transaction deliverable quality via a multi-LLM adjudication panel, anchored to a pre-registered job specification.

The two products together cover the full job lifecycle:

| Phase | Question | Product | Signal |
|---|---|---|---|
| Pre-funding | Can we trust the counterparty? | **AHM core** | AHS (D1/D2/D3) |
| Runtime | Is the agent behaving? | **AHM Shield** | Continuous monitoring |
| Post-delivery | Was the output correct? | **AHM Verify** | D4 verdict (panel) |

This is the "full lifecycle evaluator" positioning already flagged in the backlog's ThoughtProof entry. AHM Verify is the concrete product that makes that positioning real.

---

## 2. The Moat — Why This Is Not Just "Another ThoughtProof"

ThoughtProof does adversarial multi-model critique of a deliverable in isolation — stateless adjudication. Given the same inputs on two different days, ThoughtProof returns roughly the same verdict.

AHM Verify's structural advantage is the **AHM trust registry** — 5,000+ scanned agents with historical AHS, behavioural patterns, wallet history, and registry cross-links. That registry turns stateless adjudication into **stateful, identity-anchored adjudication**. Examples of verdicts the panel cannot produce on its own but that AHM Verify can:

- *"Output looks borderline in isolation, but the provider has AHS 86/B, 706 days active, zero prior DISSENTs in 40 completed jobs → ALLOW with high confidence."*
- *"Output looks technically OK but the provider has AHS 38/E (Zombie Agent pattern), DISSENT on 3 of last 5 jobs, registered this spec 12 seconds after funding → HOLD for human review."*
- *"Provider has no AHM history at all — new wallet, first job → verdict capped at HOLD regardless of output quality."*

Stated as a one-liner: **ThoughtProof scores a deliverable; AHM Verify scores a deliverable-from-this-specific-agent.** The prior behavioural record is part of the input, not an afterthought. That is the moat — it is directly a function of the data AHM core has been accumulating for months, which ThoughtProof cannot replicate without building a competing registry first.

Secondary moat: AHM Verify accumulates its own longitudinal output-quality ledger (verdict history per agent), which in turn feeds back into AHM core as the eventual D4 dimension. This creates a self-reinforcing data loop — more AHM core wallets drive more AHM Verify verdicts, which improve AHM core scoring, which drives more AHM Verify usage.

---

## 3. Architecture Options

### Option A — Fully standalone, zero coupling to AHM core

Separate repo, separate service, separate DB, separate deploy. AHM Verify does not talk to AHM core at all.

- **Pros:** cleanest separation, simplest security model, independent release cadence.
- **Cons:** destroys the moat. Verdicts would be isolated, same as ThoughtProof. Not recommended.

### Option B — Embedded in AHM core as a new endpoint namespace

`/verify/*` endpoints added to the existing FastAPI app, sharing the existing DB and registry.

- **Pros:** fastest to ship, trivial cross-reference to AHS data, one deploy.
- **Cons:** violates the explicit architectural constraint ("different repo, different service, different codebase"). Couples two products that should have independent lifecycles. Makes the AHM core app larger and harder to reason about. **Not recommended.**

### Option C — Standalone service with read-only access to AHM core registry — **RECOMMENDED**

- AHM Verify lives in its own repo (`ahm-verify`), own FastAPI app, own DB, own Railway project.
- AHM Verify calls AHM core as a client via a new internal endpoint `/internal/agent-profile/{address}` (API-key gated, not x402, read-only). Returns a condensed AHM core profile: current AHS, grade, D1/D2/D3 components, known patterns, days active, registries.
- The "stateful adjudication" moat is preserved: every LLM panel call receives the AHM profile as part of its prompt context.
- AHM core remains unaware of AHM Verify. The dependency is one-way.

Tradeoff: one extra HTTP hop per verdict. At 50-500 verdicts/day in early scale, this is negligible. Long-term it can be cached.

**Recommendation: Option C.** Matches the user's stated architectural constraint, preserves the moat, keeps the two products independently deployable.

---

## 4. Data Model

Minimum schema (PostgreSQL, three core tables):

**`job_specs`** — declared intent, registered pre-execution
- `spec_id` (uuid, PK)
- `client_addr` (text) — wallet that registered the spec
- `provider_addr` (text, nullable) — intended provider if known at registration time
- `spec_text` (text) — the task description
- `acceptance_criteria` (jsonb) — structured criteria (e.g. `{must_include: [...], format: "json", max_length: 5000}`)
- `spec_hash` (bytea) — keccak256 of canonicalised spec for tamper evidence
- `registered_at` (timestamptz)
- `erc8183_job_id` (bytea, nullable) — optional on-chain link

**`job_outputs`** — actual delivered artefact
- `output_id` (uuid, PK)
- `spec_id` (uuid, FK → job_specs)
- `delivered_at` (timestamptz)
- `output_text` (text) — or `output_uri` for large/binary artefacts
- `output_hash` (bytea)
- `submitted_by` (enum: `client`, `agent`, `protocol`)
- `submitter_addr` (text)

**`verdicts`** — panel adjudication result
- `verdict_id` (uuid, PK)
- `output_id` (uuid, FK → job_outputs)
- `panel_composition` (jsonb) — `[{model: "claude-opus-4-6", version: "...", weight: 1.0}, ...]`
- `individual_scores` (jsonb) — `[{model, verdict, score, objections, latency_ms}, ...]`
- `aggregate_verdict` (enum: `ALLOW`, `HOLD`, `DISSENT`, `REJECT`, `ESCALATE`)
- `aggregate_score` (int 0-100)
- `confidence` (numeric 0-1)
- `dissent_count` (int)
- `ahm_profile_snapshot` (jsonb) — **the moat column**: snapshot of AHM core profile at verdict time
- `evaluated_at` (timestamptz)

Optional fourth table:

**`agent_quality_history`** — rolling per-agent quality ledger (materialised view over `verdicts` grouped by provider_addr). Feeds the eventual D4 dimension.

### Spec tamper evidence

Two options for ensuring the spec delivered-against matches the spec originally registered:

1. **Off-chain hash commitment** — return `spec_hash` to client on registration; client must supply same hash on output submission. Cheap but trusts AHM Verify as hash custodian.
2. **On-chain hash commitment** — optional path that anchors `spec_hash` to an on-chain registry (cheapest: EAS attestation on Base; cheapest without contracts: `spec_hash` posted in an x402 payment memo field). Adds cost but removes custodian trust.

MVP: off-chain hash. Optional on-chain anchoring as a premium tier.

---

## 5. LLM Adjudication Panel

Architecture revised based on research into ThoughtProof's pot-cli methodology (github.com/ThoughtProof/pot-cli).

### Pipeline

```
Job Spec + Deliverable + AHS Profile
         ↓
4 Generator Models (parallel, independent, different providers)
         ↓
1 Adversarial Critic / Red Team (dedicated role — actively finds flaws)
         ↓
1 Synthesizer (consensus + MDI score + ALLOW/HOLD/REJECT verdict)
         ↓
Epistemic Block (hash-chained JSON, tamper-evident audit trail)
```

### Panel composition (6 roles total)

- **4 Generators** — run in parallel, independently propose a verdict and score:
  - Claude Sonnet
  - GPT-4o
  - Gemini 2.x
  - DeepSeek
- **1 Adversarial Critic (Red Team)**: Claude Opus — explicitly tasked with finding flaws in all 4 generator proposals, not evaluating neutrally
- **1 Synthesizer**: Claude Opus — combines generator proposals with critic findings into final verdict

### Why separate critic role

Simple majority vote is provably inferior to adversarial critique (ThoughtProof benchmark: 10:0). Asking models to verify without adversarial pressure misses fabricated statistics, hallucinated citations, and edge case failures that adversarial critique catches.

### Model Diversity Index (MDI)

Replaces simple agreement/disagreement counting. MDI is a score (0.0–1.0) quantifying the spread of generator verdicts.

**MDI thresholds:**

| MDI | Interpretation | Action |
|---|---|---|
| < 0.3 | High consensus | Proceed with synthesizer verdict |
| 0.3–0.6 | Moderate disagreement | Synthesizer must explicitly address disagreements |
| > 0.6 | High disagreement | Escalate to Deep Mode (3 runs with rotated critic) or HOLD pending human review |

### Deep Mode (high-value jobs)

For jobs above a budget threshold (e.g. >$50 USDC), run 3 full pipeline iterations with rotated critic roles:

- **Run 1:** Generators A+B+C+D → Critic E (Opus)
- **Run 2:** Generators A+B+C+E → Critic D (GPT)
- **Run 3:** Generators A+B+D+E → Critic C (Gemini)

Then **meta-synthesizer** combines all 3 runs. Eliminates single-critic bias.

### Rejection asymmetry

Rejection requires a higher confidence threshold than approval — because `reject()` is terminal and irreversible in ERC-8183.

**Default thresholds:**

| Verdict | MDI requirement | Confidence requirement |
|---|---|---|
| ALLOW | < 0.4 | ≥ 0.65 |
| HOLD | 0.4–0.6 or confidence 0.50–0.65 | — |
| REJECT | < 0.3 (higher bar) | ≥ 0.75 (higher bar) |

### Domain-specific confidence thresholds

| Domain | Confidence floor | Examples |
|---|---|---|
| Low stakes | 0.50 | Chatbot, content generation |
| Default | 0.70 | Data pipelines, analysis |
| High stakes | 0.80+ | Financial execution, code deployment |

### Epistemic Block schema

Hash-chained JSON providing a tamper-evident audit trail for every verdict:

```json
{
  "block_id": "AHM-VERIFY-042",
  "prev_hash": "0x...",
  "job_id": "7",
  "provider_address": "0xa981...",
  "ahs_snapshot": { "score": 58, "grade": "D", "d1": 75, "d2": 50 },
  "proposals": [...],
  "critique": { "model": "claude-opus-4-6", "findings": [...] },
  "synthesis": {
    "verdict": "ALLOW|HOLD|REJECT",
    "score": 0-100,
    "confidence": 0.0-1.0,
    "mdi": 0.0-1.0,
    "reasoning": "..."
  },
  "metadata": {
    "duration_seconds": 45,
    "mode": "standard|deep",
    "domain": "default|financial|medical"
  }
}
```

### AHM differentiation from ThoughtProof

ThoughtProof is general-purpose verification. AHM Verify adds:

- **AHS registry context** in every critic and synthesizer prompt — counterparty history ThoughtProof cannot replicate
- **ERC-8183 native** — verdicts map directly to `complete()`/`reject()` on-chain actions
- **Domain tuning** for agent economy job types specifically

---

## 6. Submission Flow

Three submission models, in order of implementation ease:

### MVP — Client-submitted (pull-based)

Client pays an agent via x402/ERC-8183, then:
1. Client pre-registered a spec with `POST /v1/specs` and got back `spec_id` + `spec_hash`
2. Client receives output from agent
3. Client submits the output to `POST /v1/outputs` with `spec_id` + `output_text`
4. AHM Verify returns `verdict_id`, client polls or webhooks on completion
5. Client decides whether to release escrow / accept deliverable

Simplest — no coordination with the agent or protocol required. Works today for any x402 payment flow.

### Phase 2 — Agent-submitted (push-based, reputation-building)

Agent voluntarily submits its own outputs after every job. Builds a public verdict history. Agents with strong verdict history gain a visible quality badge. Incentive: trust premium on future jobs.

### Phase 3 — Protocol-pulled (ERC-8183 evaluator role)

AHM Verify registers as an ERC-8183 evaluator. When jobs complete on-chain, the protocol routes the deliverable hash + spec hash to AHM Verify automatically. Verdict is submitted on-chain as an attestation. This is the full-lifecycle vision from the backlog's ThoughtProof entry.

Gate Phase 3 on mainnet ERC-8183 volume being measurable — matches the same gating already applied to the backlog's ERC-8183 evaluator monetisation item.

---

## 7. Integration with AHM Core — Standalone vs Composite D4

Two options for how AHM Verify's verdict data flows back into AHS scoring:

### Option 1 — Pure standalone (MVP)

AHM Verify runs independently. AHM core does not consume verdicts. Clients can optionally query both services and combine signals themselves.

### Option 2 — D4 composite (Phase 2)

AHM core adds a new D4 dimension with a rolling window (e.g. last 90 days of verdicts for that wallet). D4 feeds into the AHS composite alongside D1/D2/D3. Requires:

- AHM core reads from AHM Verify's `agent_quality_history` materialised view (or a dedicated `/internal/quality/{address}` endpoint)
- Weight rebalancing across D1/D2/D3/D4 — new AHS model version (`AHS-v3`)
- Coverage gating: wallets with <10 verdicts in the rolling window keep `d4_score = None` and are scored on D1/D2/D3 only, matching how smart-contract wallet scoring already handles `d2_data_source`

**Recommendation:** Ship MVP as Option 1 standalone. Promote to Option 2 when verdict volume supports meaningful rolling averages (likely >500 verdicts/week across active agents). This matches the backlog's existing D4 Output Quality Score entry which was already gated on "D3 live and ERC-8183 mainnet job volume measurable".

---

## 8. x402 Monetisation Options

| Tier | Price | Description |
|---|---|---|
| **Single verdict** | ~$0.25 | 3-model panel, standard prompt, AHM profile cross-ref included |
| **Premium verdict** | ~$1.00 | 5-model expanded panel, escalation path on ESCALATE, faster SLA |
| **Combined report** | ~$0.35 | Verdict + fresh AHM AHS scan bundled (premium moat surface) |
| **Spec registration** | free | Loss-leader — encourages clients to pre-register |
| **Subscription (enterprise)** | $X/month | Unlimited verdicts up to N/day, SLA, API-key gated |
| **Bulk** | $20 / 100 verdicts | Discount tier for high-volume integrators |

Pricing assumptions: 3-model panel at current LLM prices ≈ $0.05-0.12 per verdict raw cost (depends heavily on spec + output token length). Margin at $0.25 = ~50%. Validate against actual LLM costs before launch — this is the single biggest pricing risk.

The **Combined report** is the monetisation expression of the moat — it's the one product neither ThoughtProof nor any other isolated evaluator can offer, and it's worth a deliberate premium.

---

## 9. Naming, Repo, Positioning

### Name options

| Name | Pros | Cons |
|---|---|---|
| **AHM Verify** *(user's suggestion)* | Clear functional verb; "verify output quality" parses immediately | Generic, potentially SEO-weak |
| **AHM Verdict** | Strong, singular, memorable | Emphasises output not action |
| **AHM Witness** | Implies neutrality + attestation; fits on-chain phase 3 | Less obvious functional meaning |
| **AHM Audit** | Rigour connotation | Heavy, compliance baggage, slower cadence implied |

**Recommendation: AHM Verify** — the user already landed on it, it's the clearest description of what the service does, and "Verify" pairs cleanly with "Monitor" (AHM core) and "Shield" (AHM Shield) as a verb trio.

### Repo

`moonshot-cyber/ahm-verify`

Separate repo, separate GitHub Actions CI, separate Railway project. Shares no code with `agent-health-monitor`. Optional: a lightweight shared package `ahm-core-client` if/when the dependency on AHM core becomes non-trivial (not needed for MVP — a single HTTP client function is enough).

### Positioning in the AHM suite

- **AHM core** — agenthealthmonitor.xyz — pre-transaction trust (AHS). "Should I pay this agent?"
- **AHM Shield** — always-on runtime monitoring middleware. "Is this agent still behaving?"
- **AHM Verify** — post-transaction output quality. "Did the agent actually deliver?"

Three verbs: **Monitor**, **Shield**, **Verify**. Three phases: **before**, **during**, **after**. Together: the complete agent-health lifecycle.

Marketing one-liner: *"AHM is the only agent health stack that scores a job before it starts, watches it run, and audits the deliverable after it's done."*

---

## 10. Recommended Approach

1. **Greenfield new repo `ahm-verify`**, Option C architecture (standalone service + read-only AHM core client).
2. **MVP = client-submitted flow only**. No agent self-submission, no on-chain/ERC-8183 integration, no D4 composite feedback.
3. **3-model panel** (Claude Sonnet 4.5 + GPT + Gemini) with the AHM profile injected as context, not as a gate.
4. **PostgreSQL schema** as described above — specs, outputs, verdicts, with `ahm_profile_snapshot` on every verdict row as the moat column.
5. **New AHM core endpoint** `/internal/agent-profile/{address}` (API-key gated, not x402) to serve the profile to AHM Verify. This is the only change required on AHM core side, and it is small and additive.
6. **x402 pricing** at $0.25 per single verdict, $0.35 for combined report, subscription tier TBD.
7. **Pre-launch**: run 50-100 hand-labelled test verdicts to validate panel accuracy and measure real LLM cost per verdict before committing to the $0.25 price point.
8. **Launch channel**: announce alongside ThoughtProof-style thread on ERC-8183 evaluator community, positioning as the "stateful cousin" — explicitly complementary, not competing. Include a combined-report co-demo if ThoughtProof partnership materialises.

---

## 11. MVP Scope

**In scope (ship v0.1):**

- Repo scaffold (FastAPI, Poetry/uv, Railway deploy)
- 3 HTTP endpoints: `POST /v1/specs`, `POST /v1/outputs`, `GET /v1/verdicts/{id}`
- Postgres schema (specs, outputs, verdicts)
- 6-role LLM panel (4 generators + adversarial critic + synthesizer) with structured JSON responses, MDI scoring + aggregation logic
- AHM core profile fetch via `/internal/agent-profile/{address}`
- x402 payment gating at $0.50/verdict standard, $1.50/verdict deep mode
- Spec hash commitment (off-chain, keccak256)
- Basic API-key auth for high-volume clients
- Minimal admin dashboard: recent verdicts table, panel agreement rate, cost-per-verdict
- Test suite with ≥20 hand-labelled ground-truth cases

**Out of scope for v0.1 (deferred to v0.2+):**

- Agent self-submission flow
- ERC-8183 evaluator role registration
- On-chain spec anchoring (EAS attestations)
- D4 composite feedback into AHM core AHS
- Deep Mode (3-run rotated critic pipeline for high-value jobs)
- Subscription / bulk tiers
- Non-text outputs (code analysis, image evaluation, structured-data diff)
- Appeal / re-adjudication mechanism
- Public verdict explorer UI

**Success criteria for v0.1:**

- 50 paid verdicts in first 30 days
- Panel agreement rate ≥85% on test set
- Actual LLM cost per verdict ≤ $0.25 (50% margin at $0.50 list)
- At least one verdict where AHM profile cross-reference materially influenced the outcome (demonstrating the moat lands)

---

## 12. Open Questions & Risks

**Open questions**

- **Non-text outputs:** how do we score delivered code / images / structured JSON where semantic correctness matters more than text match? Probably phase 2, but worth a sketch in the spec.
- **Spec gaming:** what stops an agent from registering a trivially weak spec to guarantee ALLOW? Mitigations: client-registered specs only in MVP (agent cannot register its own spec), AHM profile cross-ref penalises agents with history of gaming, spec quality itself can be panel-scored as a separate signal.
- **Appeal mechanism:** when an agent disputes a HOLD/REJECT, what's the process? Re-run with expanded panel? Human review? Revenue share on successful appeals?
- **Panel drift:** LLM providers change model behaviour over time. Need a recalibration cadence (quarterly?) against a stable ground-truth set.
- **Legal exposure:** expressing opinions about deliverable quality carries liability risk, especially for high-value jobs. Terms of service must make clear AHM Verify is an advisory signal, not a binding adjudication, and disclaim liability for client decisions taken on the basis of verdicts. Worth a lawyer review before launch.
- **AHM core coupling:** what's the SLA contract between AHM Verify and AHM core? If AHM core is down, does AHM Verify degrade gracefully (verdict with null `ahm_profile_snapshot`) or hard-fail?

**Risks**

| Risk | Severity | Mitigation |
|---|---|---|
| LLM cost spike erodes margin | High | Pass-through pricing tier, cost monitoring, panel downsizing fallback |
| Panel disagreement rate too high to be useful | High | Validate on test set before launch; adjust prompts; consider 5-model default if needed |
| ThoughtProof ships a registry-backed feature first | Medium | Move fast on MVP, lean on AHM core's existing 5K+ wallet moat which ThoughtProof cannot rebuild quickly |
| Clients don't pre-register specs | Medium | Make registration free, provide SDK helpers, allow retroactive spec registration with a "speculative" flag |
| Legal liability on high-value verdicts | Medium | TOS disclaimer, verdicts framed as advisory, no binding adjudication claims |
| Gaming via weak specs | Low | Panel scores the spec quality itself; AHM profile cross-ref catches repeat offenders |

---

## 13. Summary

**Yes, this is feasible.** The core technical components (FastAPI service, LLM panel orchestration, Postgres, x402 gating) are all things AHM already knows how to ship. The novel components are the data model (spec/output/verdict triad with AHM profile snapshot) and the prompt engineering of the multi-model panel.

The moat is real and defensible: AHM core's existing registry of scanned agents cannot be replicated by a pure-play evaluator without months of scanning infrastructure. AHM Verify turns that registry from a pre-transaction asset into a bidirectional asset — pre-transaction trust *and* post-transaction adjudication context — which is the specific combined signal no single competitor can currently offer.

Recommended next step after this spike: a second, deeper planning doc on **prompt design for the adjudication panel** + a small hand-labelled test set (20-30 cases) to validate feasibility of the core LLM adjudication quality before any code is written. Prompt quality and panel agreement rate are the single biggest technical risks; everything else is standard service engineering.
