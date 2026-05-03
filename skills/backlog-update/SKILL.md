---
name: backlog-update
description: Reconcile, add, move, dedupe, and complete entries in the AHM backlog (ahm_backlog.md) using the conventions established in PRs #145-148. Use this skill whenever the user wants to update the AHM backlog, add new items, capture session output into the backlog, reconcile a chat history extract or handoff doc against the backlog, mark items complete, move items between priority sections, or audit the backlog for duplicates. Trigger on phrases like "add to backlog", "update backlog", "capture in backlog", "reconcile backlog", "mark complete", "move to P2", "dedupe backlog", or any mention of ahm_backlog.md edits — even if the user does not explicitly invoke the skill.
version: 0.1
last-updated: 2026-05-03
canonical-source: skills/backlog-update/
---

# AHM Backlog Update

Apply the editorial conventions of `ahm_backlog.md` consistently across every reconciliation, addition, move, completion, and dedupe pass. The backlog is dense, hand-curated, and serves as a single source of truth across sessions — so consistency, dedupe discipline, and source attribution matter more than throughput.

## Lockstep check (run first, every session)

This skill exists in two locations: the canonical version in the AHM repo at `skills/backlog-update/` and a synced copy in the AHM project knowledge base. They must match.

**At the start of any backlog session, confirm the version in context matches the repo HEAD:**

- The version + last-updated stamps in this file's frontmatter are the drift indicator.
- If running in Claude Code or with bash access: `git log -1 --format="%ai %h" -- skills/backlog-update/SKILL.md` against the repo. If the latest commit date is newer than `last-updated` in this file, the project knowledge copy is stale.
- If running in plain chat: ask the user "Has the backlog-update skill been updated in the repo since 2026-05-03?" — if yes, request a re-upload before proceeding.

**Mid-session: re-sync after any repo update to the skill.** If a skill change is committed during the session (the user mentions a PR landing, or you observe one in the conversation), the in-context version is now stale. Pause backlog work, request a project-knowledge re-upload, and only resume once the new version is loaded. Do not keep applying conventions from the stale copy after the repo has moved — that's the exact drift the lockstep rule exists to prevent.

**Skill updates only via PR to the repo.** If during a session a refinement to the skill itself is identified, capture it for a separate PR — do not edit the skill mid-session. The skill is infrastructure; treat it like code.

## Two operating modes

Detect from environment which mode applies:

**Filesystem mode (Claude Code, Cowork, or chat with code execution):**
The backlog file is on disk. Read it fresh with `view`. Apply edits with `str_replace`. Run `scripts/dedupe_check.py` before adds. Output is a working tree change the user commits.

**Chat mode (Claude.ai project, no code execution):**
The backlog is loaded as a project file in context. Edits are produced as a structured diff the user copies into the file or applies via PR locally. Dedupe runs by manually scanning the in-context backlog for keyword overlap — the script is unavailable.

The convention rules are the same in both modes; only the mechanics of applying edits differ.

## Detect intent: reconciliation vs single-edit

**Reconciliation mode** — user is feeding a batch (chat history extract, handoff doc, end-of-session notes) and asking which items are missing, which are already covered, which need updating. Output is typically one PR's worth of changes.

**Single-edit mode** — user wants to add one item, mark one complete, move one between sections, or fix one entry. Output is a small targeted edit.

Both follow the same rules; reconciliation just runs them at scale.

## Workflow

1. **Confirm lockstep** (above).
2. **Read the current backlog.** In filesystem mode, `view ahm_backlog.md`. In chat mode, the file is already in context — but verify it's loaded; never operate from memory of a prior session's state.
3. **Identify candidate changes.** In reconciliation mode, list every proposed add/update/move with its target section. In single-edit mode, locate the target entry or section.
4. **Run the dedupe check** before any add.
   - Filesystem mode: `python scripts/dedupe_check.py --backlog ahm_backlog.md --proposed "..."` (or `--batch-file` for many).
   - Chat mode: scan the loaded backlog for keyword overlap on the proposed title and body. Bias toward false positives — surface anything plausibly related.
5. **Apply the entry-format rules** below for every add or update.
6. **Apply edits.**
   - Filesystem mode: surgical `str_replace`. Avoid rewriting whole sections.
   - Chat mode: produce a structured diff block per edit, with section path and the exact text to insert/replace.
7. **Summarise the diff** for the user before claiming done. Format below.

## Section taxonomy (which section does this go in?)

The full section map is in `references/section-taxonomy.md` — read it when placing a new entry and you're not sure where it belongs. Quick mental model:

- **P1 — Active / In-Flight.** Something has actually started or is committed for the next session. Outreach with a deadline, a fix scheduled before a specific event, a chase-by date.
- **P2 — Product Backlog.** Future features, AHS enhancements, long-term visions, research concepts. The bulk of the file.
- **P3 — Tech Debt / Frontend Fixes.** Operational hygiene, doc gaps, UI fixes, analytics infrastructure.
- **P4 — Ecosystem Scanning / Monitoring.** Watch-this-protocol items, scanning targets.
- **Ecosystem & Protocol.** Standards-track engagement (ERC-8183, 8210, 8239/8240, 1705). Each ERC tends to get its own subsection.
- **Phase 3 — Future Products.** Specific named products beyond core (Verify, Shield, Title, Certification).
- **AHM Intelligence.** Layer 1/2/3 dashboard and taxonomy work — its own dedicated structure.
- **Competitive Intelligence.** One subsection per competitor with threat-level header.
- **Active Design Partner Relationships.** Currently just Nevermined / Don Gossen.
- **Strategic Positioning.** Non-feature standing rules and framings.
- **Operational.** X account status, process gaps, public commitments tracker, git hygiene.
- **Commercial / Revenue.** Stripe baseline, revenue tracking.
- **Adjacent Builders Worth Tracking.** Non-competing peers (Jinn, Valory).
- **Partnership & Outreach Targets.** Tier 1/2/2.5/3 outbound.
- **Content & Marketing.** Drafts, demos, scheduled posts.
- **Scheduled Reviews.** Time-gated review items with outcome.
- **Completed.** Append-only; subsectioned by `### Completed <date>`.

## Entry format rules

Read `references/entry-formats.md` for worked examples of each canonical shape. The rules:

### New uncompleted entries

- Open with `- [ ] **<bolded short title>** — <body>`.
- Title is short, scannable, often includes a project/PR/person reference.
- Body is one or more sentences in prose. Multi-paragraph bodies are fine for items with real complexity (Patrick/ALIA, Bootstrapping Problem, Alfred Zhang). Sub-bullets are acceptable when the structure demands them (numbered proposals, hard constraints lists, sequenced actions).
- Close with `First noted: <Mon DD YYYY>`. Every new entry gets one. Add a parenthetical source if the source matters: `First noted: May 2 2026 (chat history extract)`. If the first-noted date can't be established with confidence, use today's date and note the source.
- Do not number entries. Use `- [ ]` checkboxes throughout.

### Completing an entry

- Flip `[ ]` to `[x]`.
- Insert ` (completed Mon DD)` immediately after the bolded title.
- Append a one-sentence outcome to the body if completion meaningfully changed the description (PRs that shipped, test counts, files added). Don't rewrite — append.
- Move the entry to the **Completed** section if it's a project-level milestone (new endpoint, registration, security audit). Leave it in place if it's a small embedded item where context matters.
- For Completed-section appends, group under `### Completed <date>` if multiple items completed the same day; otherwise add a new dated subsection.

### Updating an existing entry

- Prefer surgical `str_replace` (filesystem mode) or a single-line diff (chat mode) on the specific phrase that changed.
- If status changed materially ("drafted" → "shipped"), update the leading characterisation and append a status note rather than rewriting.
- Don't touch `First noted:` — it's a fossil record. If a new development warrants its own entry, create a new one rather than overwriting.

### Cross-references

- Cite sister entries by section and short name: `(see new entry under P2 — Product Backlog → New Endpoints / Features)`.
- Cite PRs as `PR #N` or `PRs #N, #M, #L`.
- Cite jobs as `Job #N`.
- Cite ERCs as `ERC-NNNN`.
- Cite people by their established handle from the backlog (Bakugo32, JackyWang, Don Gossen, Patrick / Nicopat, Alfred Zhang). Don't invent new shortenings.

## Dedupe rules

Mandatory before adding any new entry.

**Filesystem mode:**
```bash
# Single proposed entry
python scripts/dedupe_check.py --backlog /path/to/ahm_backlog.md \
    --proposed "**Title** — body text here"

# Longer proposed entry from a file
python scripts/dedupe_check.py --backlog /path/to/ahm_backlog.md \
    --proposed-file proposal.txt

# Batch (one entry per line, # for comments)
python scripts/dedupe_check.py --backlog /path/to/ahm_backlog.md \
    --batch-file proposals.txt

# Audit existing backlog for internal duplicates
python scripts/dedupe_check.py --backlog /path/to/ahm_backlog.md --audit
```

Exit code 0 = no matches above threshold (safe to add). Exit code 1 = matches found (read them and decide). Default threshold is 0.25 for proposed-vs-existing; raise to ~0.5 for audit mode (--audit auto-sets a 0.5 floor).

**Chat mode:**
Manually scan the in-context backlog. For each proposed entry, identify 3-5 distinctive keywords from the title and body (drop stopwords, keep proper nouns, ERC numbers, project names). Search the backlog for those keywords and surface any entry where 2+ keywords land. Err on the side of surfacing too much.

**Whichever mode, decide per match:**
1. **True duplicate** — the existing entry covers this. Do not add. Update the existing entry if the new info changes anything.
2. **Near-miss / overlap** — there's a related entry but the new one captures a distinct angle. Add the new entry AND insert a cross-reference both directions.
3. **No match** — proceed with add.

Rule of thumb: if two entries would always be worked together or always be retired together, they should be one entry. If they could plausibly be retired or completed independently (e.g. ERC-8210 v1 named-reference status vs ERC-8210 v2 active drafting), they're separate.

PR #148 had to clean up duplicates that PR #147 introduced — running the dedupe check before commit is the entire point.

## Source attribution

Every reconciliation declares its source(s) in the PR description or summary, not buried inside individual entries. Patterns observed:

- "Source: chat history" — items lifted from a Claude.ai conversation extract.
- "Source: ThoughtProof research, Apr 7 2026" — items from a specific external artefact, with date.
- "Source: ARS paper positioning, Apr 8 2026" — items from a specific publication.
- "Driver: Job #3 (Apr 27)" — items emerging from a concrete event.

When in doubt, attribute. The backlog is a long-running record; future-you wants to know where a half-described item came from.

## Holding decisions

Some items shouldn't be added straight away. Common holds:

- **Verification pending** — the entry exists as "HOLD pending verification" with explicit verification work listed. Patrick / ALIA proposal is the canonical example.
- **Decision input, not feature** — items framed as "produce a report covering X" rather than "build X" (D4 fold-in gates report).
- **Do not build yet** — lifecycle gates documented in the entry (D5 Security Posture, AHM Verify D4 fold-in, EAS integration).

Use these labels rather than inventing new ones — they're load-bearing for future-you.

## Things to never do

- Never overwrite `First noted:` dates.
- Never delete a Competitive Intelligence threat-level entry; reclassifications go in-place with both old and new threat levels visible.
- Never edit the **Completed** section other than appending. It's append-only history.
- Never add an item without running dedupe check.
- Never renumber sections (P1/P2/P3/P4 are fixed labels).
- Never expose AHS scoring weights or proprietary methodology details. Scoring weights are local-only, never in the backlog or any committed artefact.
- Never edit the skill itself mid-session. Skill changes go via PR to the repo, then a project knowledge re-upload.

## When to ask vs. proceed

Single-edit mode rarely needs clarification — apply the rules and execute. Reconciliation mode often produces ambiguity (which section? merge or new entry? completion or update?). When ambiguity exists:

- If it's a section-placement question and one option is obviously stronger, just choose. State the choice in the summary.
- If it's a true judgement call (merge vs new entry on a near-miss), ask the user before writing.
- Don't ask whether to add `First noted:` or whether to follow the entry format — those are settled.

## Output: summarise the diff

Before claiming done, give the user a structured summary:

```
Added: <N> items
  - P2 → AHS Enhancements: ThoughtProof PLV composition (cross-ref to D4 backlog item)
  - Operational → Standing Process: cross-session context capability gap

Updated: <N> items
  - P2 → New Endpoints / Features → Confidence-based routing: noted ThoughtProof PLV concurrence

Held: <N> items
  - <reason>: <items>

Deduped: <N> proposed items folded into existing entries
  - <proposed>: merged into <existing>

Source: <attribution>
```

This makes the PR description trivially writeable and gives future-you something to scan when reading commit history.
