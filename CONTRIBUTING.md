# Contributing to AHM

## Facts on public surfaces

Two rules. Both are enforced by `scripts/check_consistency.py`, which runs in
CI and fails the build.

### 1. Changing figures live on the dashboard only

**Never put a scan count, average or rate into static copy — link to the
dashboard instead.**

Anything the nightly scan can change belongs on `/dashboard`, rendered from the
database, and nowhere else:

- agent wallet counts (scanned, registered, per-registry)
- average AHS, D1/D2 averages
- zombie rate, grade distribution
- percentages derived from any of the above ("X% score below Grade B")
- last-scan timestamps

Write qualitative copy and link:

```html
<!-- no -->
<p>AHM has scanned 8,000+ agent wallets. 91% score below Grade B.</p>

<!-- yes -->
<p>AHM scans agent wallets nightly across five registries, and the large
   majority score below Grade B.
   <a href="https://agenthealthmonitor.xyz/dashboard">See live figures</a>.</p>
```

Do **not** swap in a fresher hard-coded number. That is how the last set went
stale: `2,860+` was accurate the day it was written, then sat on the homepage
for four months while the real figure grew past 42,000.

External ecosystem statistics ("40,000 agents registered across protocols") are
not AHM measurements. They need a cited, dated source or they get cut.

### 2. Stable values come from canonical config

**Never hard-code a price, threshold, count or handle — import from
`config/canonical.json`.**

Values that change only on a deliberate release live there: per-endpoint
pricing, endpoint count, batch limits, grade thresholds and labels, routing
bands, dimension weights, the Verify pipeline description and verdict enum, the
registry list, the Twitter handle, the contact email, the roadmap stamp.

```python
# no
ENDPOINT_COUNT = 14
AHS_PRICE = "$1.00"

# yes
import canonical
ENDPOINT_COUNT = canonical.endpoint_count()
AHS_PRICE = os.getenv("AHS_PRICE_USD", canonical.price_for("ahs"))
```

Static HTML cannot import the config, so the checker verifies those surfaces
against it instead. If you change a value in `canonical.json`, run the checker
and fix whatever it flags.

## Distinctions that keep getting collapsed

These caused real, shipped contradictions. Keep them apart:

| Not the same thing | |
|---|---|
| **Grades** (A–F, quality bands) | **Routing actions** (`instant_settle` / `escrow` / `reject`, a payment-gating decision derived from the grade). Routing has no score ranges of its own, and integrators can override the mapping per-policy. |
| **AHS dimensions** (D1, D2, and D3 as an opt-in overlay) | **AHM Verify**, a separate service that scores delivered output. It is not "D4" and does not feed the composite. There is no D4. |
| **Verify verdicts** (`ALLOW` / `HOLD` / `REJECT`) | **On-chain actions** (`complete` / `hold_pending_review` / `reject`) and **routing actions** above. Three vocabularies, three meanings. |
| **x402 batch limit** (10 wallets/call) | **API-key batch limit** (25/request). Both are real. Never state one without the other. |
| **Zombie rate** (CDP pattern classifier, on `/dashboard`) | **Grade E–F share** (a grade-based proxy, on intelligence). Different predicates — do not give them the same label. |

AHM Verify runs on Claude only. The GPT-4o / Gemini / DeepSeek strings in
`prompts.py` are *prompt personas* that diversify reasoning style, not other
vendors' models. Never describe the panel as multi-vendor or "six-model" — it is
six *roles*: four generators, a critic, a synthesiser.

## Running the checks

```bash
python scripts/check_consistency.py                  # local surfaces
python scripts/check_consistency.py --live           # also crawl deployed URLs
python scripts/check_consistency.py --surfaces ../ahm-docs/public
python scripts/check_consistency.py --list-known     # accepted exceptions
```

Adding an exception requires a reason in `ALLOWED_CONTEXTS`. An exception
without a stated reason is how a rule quietly stops working.

## Derived artefacts

Some published files are generated. Regenerate them rather than editing:

- `static/ahm-og-banner.png` — `python generate_og_banner.py`. Deliberately
  carries no count; a committed image cannot re-render from config.
- Both docs PDFs — `python scripts/build_pdfs.py` in the `ahm-docs` repo. They
  render from `public/index.html`, so they cannot disagree with the docs page.
