# Agent Taxonomy Classification — POC Summary

## TL;DR

A random sample of 1,000 Base mainnet agent wallets achieved **75.7%
classification coverage** across 6 of 10 v1 functional categories, using a
36-contract lookup table and two registry-derived classification paths. ~54% of
classified agents are registry-derived (ACP → Orchestration), meaning the
contract-lookup table alone covers roughly half as many agents as the headline
number implies. See [Findings](#findings) for the full breakdown.

---

## Run History

| Phase | Date | Sample Size | Selection Method | Lookup Contracts | Coverage | Categories Hit |
|-------|------|-------------|-----------------|-----------------|----------|---------------|
| 1 | April 2026 | 200 | Top by tx_count | 25 | 61.5% | 4 |
| 2.0 | April 25, 2026 | 500 | Random | 31 | 75.8% | 4 |
| 2.1 | April 25, 2026 | 1,000 | Random | 34 | 76.6% | 6 |
| 2.2 (post offset-cap fix) | April 25, 2026 | 1,000 | Random | 34 | 75.7% | 6 |

**Why Phase 2.2 is the headline number.** Phase 2.1 reported 76.6% but was
running with a hardcoded `offset=200` cap in the Blockscout API call, which
silently truncated transaction histories for agents with more than 200
transactions. PR #138 bumped the offset to 1,000, producing more complete
transaction data. Phase 2.2 re-ran with the fix and dropped to 75.7% — not
because the taxonomy regressed, but because the fuller transaction data
resolved some spurious matches where an agent's truncated 200-transaction
window happened to over-index on a single classified contract. The lower number
is more honest.

---

## Methodology

1. **Sample selection.** Random sample from agents with `tx_count >= 1` in the
   AHM database (11,000+ Base mainnet agent wallets sourced from on-chain
   registries).

2. **Lookup table.** `scripts/taxonomy_contracts.json` — 36 contract entries
   post-PR #139, covering individual contract anchors plus subcategory signals
   for dex, oracle, and nft_media.

3. **Two classification paths:**
   - **Registry-derived:** ACP registry membership → Orchestration; ERC-8004
     Identity Registry calls → Identity & Trust.
   - **Contract-lookup:** Matched on the top contracts called from the agent's
     transaction history (fetched via Blockscout `txlist` API).

4. **Confidence bucketing:** HIGH if direct lookup match, MEDIUM if partial
   signal, UNCLASSIFIABLE if no signal.

5. **Classification script:** `scripts/classify_agents_taxonomy.py` with
   `offset=1000` (per PR #138).

---

## Findings

### Coverage Composition

~54% of classifications are registry-derived (ACP agents → Orchestration), with
the remainder driven by the contract lookup table. The registry-derived share is
high because ACP itself is large and well-indexed; this overstates how much
"lookup table coverage" we have for non-ACP agents. Distinguishing the two is
important for interpretation.

### Category Anchors Achieved

Six of ten v1 functional categories now have first-anchor contracts:

| Category | Status | Example Anchors |
|----------|--------|----------------|
| Orchestration | Anchored | ACP registry-derived, $VIRTUAL token, Glorb |
| Identity & Trust | Anchored | ERC-8004 registry-derived, Identity Registry |
| Financial | Anchored | Aerodrome, Giza, INFINIT, Vader AI, Gains Network gTrade |
| Verification | Anchored | ProblemManager, AgentCoin, EAS |
| Intelligence & Analytics | Anchored | Ritual Infernet + EIP712Coordinator, Olas Mech Marketplace, CARV |
| Commerce | Anchored | Seaport, SeaDrop |
| Research | Unanchored | — |
| Creative | Latent | Zora 1155 Factory, Botto |
| Infrastructure | Unanchored | — |
| Physical World | Unanchored | — |

*Latent = anchor contracts exist in the lookup table (Zora 1155 Factory, Botto)
but no agents in the Phase 2.2 sample called them.*

The three unanchored categories will likely require either different signal
sources (off-chain identity, EAS attestations, ERC-8239 skill registry once
deployed) or LLM-assisted classification in Phase 3.

### Methodology Limitations

Three known limitations, in priority order:

1. **Offset cap (resolved in PR #138).** The original script used `offset=200`
   in the Blockscout `txlist` call, which silently truncated agents with more
   than 200 transactions. The fix bumped to `offset=1000`. Multi-page
   pagination remains a backlog item.

2. **Wallet-primitive conflation (open issue #140).** The script currently
   outputs UNCLASSIFIABLE for both genuinely unknown agents and agents whose
   only on-chain activity is calling a Gnosis Safe (a wallet primitive, not a
   functional contract). Two of the three top unclassified contracts in the
   post-fix Phase 2 run were Safes. A wallet-only output bucket is planned to
   surface this distinction honestly.

3. **Inner-call tracing not yet implemented.** Safe-mediated agent activity
   hides the actual functional contract behind `execTransaction` calldata. The
   Olas Mech Marketplace discovery in PR #139 demonstrated that tracing through
   inner calldata can recover real classifications. Implementing this
   systematically would convert wallet-only agents back into properly classified
   ones.

---

## Recommendations

1. **Implement wallet-only bucket (issue #140)** — small change, large honesty
   improvement.
2. **Inner-call tracing for Safe-mediated agents** — bigger change, recovers
   classifications rather than just labelling them differently.
3. **Add ERC-20 token transfer analysis** (`tokentx`) for agents with sparse
   `txlist` matches.
4. **Consider Phase 3 LLM classifier** with EAS-verified agents (already
   tracked in `docs/eas-verified-agents.json`) as labelled seed set.

---

## Next Steps for the Public Taxonomy Page

The page at `intelligence.agenthealthmonitor.xyz/taxonomy` will be updated to:

- Mark each of the 10 categories as either **anchored** (with example
  protocols) or **pending an anchor**.
- Include a methodology note linking back to this document.
- Reference issue #140 as the tracked next iteration.

---

## Files

| File | Purpose |
|------|---------|
| `scripts/taxonomy_contracts.json` | 36-contract lookup table (Base mainnet) |
| `scripts/classify_agents_taxonomy.py` | Classification script (DB + Blockscout API) |
| `docs/taxonomy-poc-summary.md` | This summary document |
| `docs/taxonomy-spike-top-unclassified.md` | Spike: top unclassified contracts (PR #137) |
| `docs/taxonomy-spike-postfix-investigation.md` | Spike: post-offset-fix investigation (PR #139) |
| `docs/eas-verified-agents.json` | EAS trust-score-attested agent owners (PR #135) |
