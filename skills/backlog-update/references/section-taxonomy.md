# Section Taxonomy

Read when placing a new entry and the destination isn't obvious. Sections appear in this order in `ahm_backlog.md`.

## Top-level sections (`##`)

### Current State (as of <date>)
**Not for additions.** This is a hand-curated snapshot, refreshed only at major milestones. Don't append routine items here.

### P1 — Active / In-Flight
Items with a deadline, an active dependency, or a chase-by date. Things that should ship in the next session or two. Sub-sections include "Distribution & Partnerships", "Pending Response / Follow-up", and ERC-specific outreach groupings. New P1 entries usually fit one of these existing subsections; only create a new subsection if multiple items share a theme.

### P2 — Product Backlog
The bulk of forward-looking work. Subsections:
- **New Endpoints / Features** — concrete API/product additions (Confidence-based routing, evidence object, agent alerts).
- **Long-term Product Visions** — moonshots and strategic concepts (Shield, Title Registry, Power Index, EAS Integration). Includes `#### Research / Open Concepts (not on build path, periodic revisit)` for ideas worth tracking but not committed (Agent Intent Detection, Decision Quality, Hidden Backdoor).
- **AHS Enhancements** — improvements to the scoring system (D3 probing, pattern library, D4/D5 dimensions, reputation decay).
- **Bootstrapping Problem — Zero-History Wallet Treatment** — its own subsection because it touches multiple dimensions (D2 weighting, routing, confidence).

### P3 — Tech Debt / Frontend Fixes
Subsections:
- **Frontend / UI** — homepage links, nav, docs structure.
- **Other** — analytics, contact form, USDC sweep, daemon fallbacks, configurable routing docs, deprecated/superseded items.

### P4 — Ecosystem Scanning / Monitoring
Watch-this-protocol items, scanning targets. Has nested **Public Agent Economy Health Dashboard (depends on P4)** as the data-product output of scanning.

### Ecosystem & Protocol
Standards-track engagement. Each ERC tends to get its own `###` subsection. Sub-subsections (`####`) are used for thread-specific developments under one ERC (e.g. subjectId Translation Layer Commitment, ERC-8239 Public Reply, Patrick/Nicopat/ALIA Collaboration Proposal — all under ERC-8239 / ERC-8240).

Threads currently tracked: EvaluatorRegistry Metadata Field, x402 "Upto" Scheme, ERC-1705, ERC-8239 / ERC-8240, ERC-8210 v2, Arc Mainnet Migration.

### Phase 3 — Future Products
Specific named products beyond AHM core. Currently houses **AHM Verify — production status to verify**.

### Future Product Concepts
Earlier-stage product ideas (AHM Challenge Protocol, ACP/Virtuals as new registry type).

### AHM Intelligence — Public KPI Dashboard & Agent Taxonomy
Self-contained planning section. Layer 1 (KPI dashboard), Layer 2 (taxonomy), Layer 3 (enriched reports), Data Enrichment Spike, Suggested Sequencing, Taxonomy Category in AHS API Response.

### Market Research — 402index.io Analysis (March 2026)
Frozen analysis section. Don't edit unless rerunning the scan.

### Competitive Intelligence
One `###` subsection per competitor, with threat-level header in the title:
- `(@handle, domain) — ELEVATED THREAT`
- `(@handle, domain) — Complementary ecosystem partner`
- `(@handle, domain) — TIER 3 ECOSYSTEM WATCH`
- `— INVESTIGATION REQUIRED`

Reclassifications go in-place; preserve "originally classified as X, reclassified to Y" rather than overwriting.

### Active Design Partner Relationships
Currently just Nevermined / Don Gossen. Each relationship gets `###` and includes status, sequence of contact, current state, strategic significance.

### Strategic Positioning
Standing rules and framings, not features. Subsections include "Standards-layer engagement as primary strategic surface", "AHM Dimensional Framework as Over-Architecture", "ARS (Agentic Risk Standard) — Positioning Opportunity". Sub-subsection (`####`) "Three backlog items arising from ARS" is an example of how positioning sections can spawn concrete items inline.

### Operational
- **X / @AHMprotocol account status**
- **Standing Process / Capability Gaps** — items about workflow, not product.
- **Git Hygiene Note**
- **Active public commitments — tracker** — list of public commitments with status.

### Commercial / Revenue
Stripe baseline, x402 wallet revenue.

### Adjacent Builders Worth Tracking
Non-competing peers. Currently Jinn Network and Valory AG.

### Partnership & Outreach Targets (from 402index.io market analysis, March 2026)
Tier 1 / Tier 2 / Tier 2.5 / Tier 3 subsections.

### Content & Marketing (added Mar 13-16)
Demo, BaseHub engagement, pin-a-new-X-post, drafted-but-not-posted threads.

### Scheduled Reviews
Time-gated reviews with documented outcomes. Once a review is done, the outcome stays in this section as the historical record.

### Completed
Append-only history. Subsections by date: `### Completed <date>`. Entries in the unsubsectioned top of this section are old / undated milestones; new completions always go into a dated subsection.

## Heading-level conventions

- `##` — top-level section.
- `###` — first-level subsection.
- `####` — second-level subsection (used sparingly, for thread-specific developments under a single ERC, or for groupings within a complex section like AHM Intelligence Layer 2).
- Entries are bullets (`- [ ]` or `- [x]`), not headings. Don't promote an entry to a heading just because it's complex — sub-bullets and multi-paragraph bodies handle that.

## Section-creation rules

Don't create new top-level sections without good reason. The taxonomy is stable for a reason. New work usually fits an existing section or warrants a new `###` subsection within an existing `##`.

If you genuinely need a new top-level section (rare — happens when a whole new workstream emerges), place it before **Operational** and after the most-similar existing section. Discuss with the user first.
