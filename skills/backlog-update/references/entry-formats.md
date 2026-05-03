# Entry Format Examples

Worked examples of every canonical entry shape in `ahm_backlog.md`. Use these as templates rather than inventing new shapes.

## 1. Simple single-line entry

The most common shape. Bolded title, em-dash, body sentence, `First noted` stamp.

```markdown
- [ ] **Per-dimension methodology versioning** — Currently `model_version: "AHS-v1"` is report-level. Patrick / ERC-8240 coordination may require finer per-dimension version tags (e.g. `d1_version: "D1-v2"`, `d2_version: "D2-v1"`) to signal when individual dimension methodologies change without a full composite version bump. Review when ERC-8240 alignment work progresses. First noted: May 2 2026
```

When to use: a discrete idea or task that fits in two or three sentences.

## 2. Single-line entry with source attribution

Adds a parenthetical to `First noted`. Use when the source is non-obvious or a record of provenance matters.

```markdown
- [ ] **Agent Intent Detection** — Measure drift from declared agent purpose to unrelated contract interactions. Build a 30-day baseline behaviour profile per agent, flag deviations. No commitment to build; track for periodic revisit. First noted: May 2 2026 (chat history extract)
```

Other patterns seen in the backlog:

```markdown
... Source: ThoughtProof research, Apr 7 2026
... Source: ARS paper positioning, Apr 8 2026
... Source: chat history
... Driver: Job #3 (Apr 27)
```

## 3. Multi-paragraph entry

For complex items where prose flow matters and sub-bullets would fragment the argument.

```markdown
- [ ] **Alfred Zhang (@Alfredz0x) / OpenPasskey integration angle — relationship status to clarify** — April 3 2026 X exchange (visible in current notifications on restored @AHMprotocol account): Alfred replied substantively to AHM, identifying integration thesis — OpenPasskey HTTP tap logs + AHM on-chain signals as composite intelligence layer for physical-to-onchain payment verification (RIP-7212 / P-256 on Base L2). Alfred published a relevant deep-dive on dev.to (RIP-7212 / OpenPasskey card verification). [...continues for several sentences...] Action: Locate prior email thread with Alfred, locate any AHM repo or local notes on OpenPasskey, decide based on findings. Do not re-engage on X without first reconstructing the state. First noted: Apr 29 2026
```

When to use: relationship items with history, items where the reasoning is the value (not just the action), items where premature compression would lose load-bearing context.

## 4. Sub-bullet entry (heading-style)

The entry is itself a `###` or `####` heading rather than a checkbox bullet. Used when the item is a coherent topic with multiple aspects that need their own bullets. The heading replaces the bolded title; the bullets carry the body.

```markdown
### Bootstrapping Problem — Zero-History Wallet Treatment
- Currently a wallet with no on-chain history scores similarly to one with demonstrably degraded patterns due to D2 behavioural consistency weighting (70%)
- Raised by Bakugo32 (Arc/ERC-8183 protocol team) Apr 9 2026 after Job #7 evaluation — D grade on zero-history wallet triggered reject when HOLD was more appropriate
- Fix: introduce a separate scoring path for zero-history wallets — score as "Unrated" rather than mapping to D/E grade
- Unrated wallets should route to escrow (not reject) by default — client assumes counterparty risk with funds protected
- Review alongside D2 session continuity gate (April 21 2026)
- Job #3 (Apr 27 2026) was the first live test case of this gap. [...]
- Confidence-based routing is the committed next refinement (see new entry under P2 — Product Backlog → New Endpoints / Features). Public commitment made in the abstain() architectural reply on the ERC-8183 thread
```

When to use: standards-track threads, ongoing topics with multi-faceted state (Bootstrapping Problem, ERC-8210 v2 active drafting, Arc Mainnet Migration). Note: heading-style entries don't get a `First noted:` line at the top — instead the date context is embedded in the bullets.

## 5. Sub-bullet entry with numbered proposals + hard constraints

For items with structured external content (a five-axis proposal, a sequenced list of actions). The numbered list lives inside the entry.

```markdown
#### Patrick / Nicopat / ALIA Collaboration Proposal — HOLD Pending Verification
- Patrick (Nicopat on EthMag) sent private DM May 1 with five-axis collaboration proposal:
  1. Cross-evaluator calibration framework co-spec
  2. Confidence propagation in multi-evaluator systems co-research
  3. Bidirectional data exchange under MoU (AHM scan-level signals exchanged for ALIA on-chain detection signals)
  4. 8004×8240 quality bridge co-design
  5. ZK1 privacy-preserving evaluator (longer term)
- Also offering AHM reference in ALIA-published end-of-May/early-June industry report ("mutual credibility" framing)
- **Verification status:** ERC-8239 thread on EthMag confirmed real (URL: ethereum-magicians.org/t/erc-8239-agent-skill-registry/28335) — Pablo is active participant. ALIA entity claims, institutional clients, published report, on-chain detection contracts NOT independently verified.
- **Verification work pending:** contract on-chain verification, DNS check on lockstep.defire.business and catalyst.defire.business, ALIA entity background
- **Hard constraints:**
  - Any signal-level data exchange involving Nevermined-derived signals requires explicit Nevermined consent first
  - No MoU language without prior verification
  - No private-call commitment until verification complete and technical advisor in loop
  - Async-first posture preferred
- First noted: May 1 2026
```

When to use: items with explicit structured content from an external counterparty, items with hard constraints that need to stay visible, holds that require their own verification trail.

## 6. Completed entry (in-place)

Stays in the original section, flipped to `[x]`, with `(completed Mon DD)` after the title. Use when the item belonged to a list of related items where moving it to **Completed** would lose context.

```markdown
- [x] **/ahs/batch endpoint** (completed Mar 31) — POST `/ahs/batch` scores multiple agent wallets in a single API call. Up to 10 wallets per x402 call ($10.00 flat), up to 25 via API key (1 credit/wallet, partial results supported). Concurrent scoring with semaphore-limited RPC. PRs #42, #43, #44. Design partner validated (Alfred Zhang, httppay.xyz)
```

```markdown
- [x] **Smart contract wallet scoring** (completed Mar 24) — When `txlist` returns < 10 outgoing txs, `calculate_ahs()` now falls back to Blockscout V2 `token-transfers` API. D2 uses 4/8 signals (timing regularity, transfer diversity, counterparty breadth, activity gaps) with redistributed weights. Verified: ACP agents now score D2 10-68 (was baseline 50). EOA wallets unaffected (64/64 tests pass). New: `fetch_token_transfers()`, `calculate_d2_score_from_transfers()`, `_calc_token_transfer_diversity_score()`, `AHSResult.d2_data_source` field
```

When to use: feature-list items, AHS Enhancement items, P4 spike items.

## 7. Completed entry (moved to Completed section)

Project-level milestones. Append to the appropriate `### Completed <date>` subsection.

```markdown
### Completed Apr 27 2026

- [x] **Job #3 cycle** — first live ERC-8183 evaluator action — ThoughtProof scored 58/D with INSUFFICIENT confidence; AHM called complete() with verdict JSON hashed on-chain (verdict hash 0xbe9c3ba2..., tx 0x2a33b40e...). Public transparency post on ETH Magicians ERC-8183 thread explaining INSUFFICIENT confidence reasoning. [...]
```

When to use: new endpoints shipping, registrations, security audits, registry scans, major standards-track wins.

## 8. Threat-level competitive intelligence entry

Threat level lives in the heading. Entry includes status, positioning, differentiator, action.

```markdown
### Verdict Protocol (@verdictprotocol, verdict-protocol.xyz) — ELEVATED THREAT
- Joined: March 2026. 123 followers. Whitepaper published. $VRDCT token live on Virtuals.io.
- Positioning: "The trust layer for agent commerce" — six-layer protocol above ERC-8183 [...]
- Direct overlap: Trust & Reputation Index [...] and Evaluator Network [...]
- No overlap: pre-transaction wallet health scoring, zombie detection, AHM Shield runtime middleware
- Key differentiator vs AHM: token-coordinated protocol play [...]
- Threat level: Medium-high long term if they execute. 6-12 month window before they can meaningfully compete on reputation data
- Action: Monitor only. Do not engage publicly. Review positioning if they reach Phase 3+ or follower count exceeds 1K.
- First noted: Apr 8 2026
```

When to use: any new competitor or reclassification.

## 9. Reclassification within Competitive Intelligence

In-place edit preserving history.

```markdown
### AgentProof (@agentproof, agentproof.sh) — Complementary ecosystem partner

- On-chain reputation oracle, 158K+ agents indexed, 21 chains, live oracle
- Reclassified from "ELEVATED THREAT" (April 8 2026) to complementary positioning following deeper review: AgentProof addresses on-chain reputation aggregation; AHM addresses behavioural trust scoring [...]
- The ERC-8183 evaluator collaboration with ThoughtProof (Job #2 and Job #3 cycle) demonstrates the cooperative dynamic [...]
- Strategic monitoring rather than competitive response: track AgentProof's API surface and pricing for ecosystem mapping, but no defensive positioning required.
- First noted: Apr 8 2026 (as threat). Reclassified: Apr 28 2026.
```

Note the dual `First noted` / `Reclassified` stamps — preserve both.

## 10. Active public commitment

Goes in **Operational → Active public commitments — tracker** as a single bullet with the commitment summary.

```markdown
- Confidence-based routing build — committed in abstain() architectural reply on ERC-8183 thread. Foundation: extend PR #112 routing policy with confidence_overrides schema. Driver: Job #3 INSUFFICIENT verdict surfaced the gap.
- INTERFACES.md review — committed to Bakugo32 in Treasury thread reply. Bakugo is drafting the diff; AHM committed to review before he finalises implementation.
```

When to use: anything said publicly on EthMag, Twitter, or to a design partner that requires follow-through. Track status fortnightly per the existing convention.

## Cross-reference style

```markdown
... see new entry under P2 — Product Backlog → New Endpoints / Features
... (cross-references: Bootstrapping Problem (zero-history wallet treatment))
... (links to EAS integration backlog item)
... see `proactive_scan_spike_results.md`
... (see wash_spike_results.md for formula)
```

Plain prose references work; no need for markdown links unless pointing at an external URL.
