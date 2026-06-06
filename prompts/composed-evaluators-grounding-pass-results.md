# COMPOSED_EVALUATORS grounding pass — results

Investigation date: 2026-05-30

---

## Task 1: AHS v9 2D-mode weight renormalisation

### Blog post / earlier-work claims

- D3 base weight: 0.20
- Renormalisation: D1 from 0.30 → 0.375, D2 from 0.50 → 0.625
- Weighted sum example: 58.125, smoothed via EMA (α=0.6) to 57

### Actual code (monitor.py)

**Weight assignment** — `monitor.py:2962-2970`:

```python
# 3D mode (agent_url provided → D3 scored)
result.mode = "3D"
result.d1_weight = 0.25
result.d2_weight = 0.45
result.d3_weight = 0.30

# 2D mode (no agent_url → D3 not scored)
result.mode = "2D"
result.d1_weight = 0.30
result.d2_weight = 0.70
result.d3_weight = 0.0
```

**Composite calculation** — `monitor.py:2972-2976`:

```python
if d3_score is not None:
    composite = 0.25 * d1_score + 0.45 * d2_score + 0.30 * d3_score
else:
    composite = 0.30 * d1_score + 0.70 * d2_score
```

**EMA smoothing** — `monitor.py:3002-3022`:

```python
if scan_count == 2:
    temporal = composite * 0.8 + previous_score * 0.2
else:
    alpha = 0.6
    prev_ema = previous_ema if previous_ema is not None else previous_score
    temporal = composite * alpha + prev_ema * (1 - alpha)
```

**Default weights in AHSResult dataclass** — `monitor.py:1656-1658`:

```python
d1_weight: float = 0.30
d2_weight: float = 0.70
d3_weight: float = 0.0
```

### Findings

| Claim | Actual | Match? |
|---|---|---|
| D3 base weight = 0.20 | D3 weight = 0.30 (3D mode) | **MISMATCH** |
| 2D renorm: D1 = 0.375 | D1 = 0.30 | **MISMATCH** |
| 2D renorm: D2 = 0.625 | D2 = 0.70 | **MISMATCH** |
| EMA α = 0.6 | α = 0.6 | Match |
| Smoothed via EMA | EMA for scan_count ≥ 3; scan_count == 2 uses 80/20 blend | Partial match — blog omits the scan_count == 2 special case |

The blog post weights are **wrong**. The code has never used 0.375/0.625 for 2D mode or 0.20 for D3. The actual 2D-mode weights are simply D1=0.30, D2=0.70 (no renormalisation from a 4-weight base — there is no D4 in the code at all).

### Version markers

- `model_version: str = "AHS-v1"` — `monitor.py:1667`
- No "v9", "v2", or any other version string found in code. "AHS-v1" is the only version declaration.

---

## Task 2: Confidence enum and AHS versioning

### Confidence enum

**Defined at** `monitor.py:1692-1709`:

```python
def _ahs_confidence(tx_count: int, history_days: int, has_d3: bool, has_previous: bool) -> str:
    level = 0
    if tx_count >= 100 and history_days >= 7:
        level = 2  # base HIGH
    elif tx_count >= 50 and history_days >= 3:
        level = 1  # base MEDIUM
    elif tx_count >= 10:
        level = 0  # base LOW
    else:
        return "INSUFFICIENT"

    if has_d3:
        level = min(level + 1, 2)
    if has_previous:
        level = min(level + 1, 2)

    return ["LOW", "MEDIUM", "HIGH"][level]
```

**Values**: `HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT` — all uppercase strings, returned directly.

**AHSResult default**: `confidence: str = "LOW"` — `monitor.py:1651`

**API model**: `confidence: str` with description "Score confidence level: high, medium, or low" and examples `["high"]` — `api.py:707`. Note the API description says lowercase, but the actual code returns uppercase. The `AHSReport` model is just a `str` field with no enum validation, so whatever `_ahs_confidence()` returns flows through.

### AHS versioning

- **Code**: `model_version: str = "AHS-v1"` — `monitor.py:1667`
- **API model**: `model_version: str` with description "AHS scoring model version" and examples `["2.1.0"]` — `api.py:714`. The example value "2.1.0" is inconsistent with the actual default "AHS-v1".
- **ALIA Section 3.2**: refers to "AHS v9" — `ahm-alia-section-3.2.md:34`
- **Codebase-wide**: no string "v9", "v2", "v3"…"v8" appears in any code file. "AHS v9" is not a label used anywhere in the codebase.
- **No version-tracking mechanism** beyond the static default string in the dataclass.

### Finding

The paper can correctly state the confidence enum as `HIGH / MEDIUM / LOW / INSUFFICIENT` (uppercase). "AHS v9" has no basis in code — the code declares `AHS-v1`. There is no version increment mechanism.

---

## Task 3: IRiskHook interface and existing hook patterns

### Search results

**IRiskHook**: appears only in `ahm_backlog.md:317`:
> "Open questions on RoleCollusion, behavioural similarity, IIndependenceSignal vs IRiskHook hierarchy — all directly affect how AHM's D1/D2/D3 outputs compose in multi-evaluator scenarios"

This is a conceptual reference in backlog notes about ERC-8210 v2 drafting, not a code definition.

**"hook" in erc8183_worker.py**: line 79 — an ABI tuple field `{"name": "hook", "type": "address"}` in the ERC-8183 contract ABI. This is a Solidity struct field name, not a Python interface.

**All other "hook" references**: webhook-related (Slack/Discord/Stripe alert webhooks in `api.py`, `db.py`, test files). Not related to ERC-8183 hook surfaces.

**No existing definitions found for**: `IRiskHook`, `RiskHook`, `ERC8183Hook`, or any abstract class / Pydantic model / trait matching a hook interface pattern.

### Finding

**No existing IRiskHook found — sketch in the paper will be greenfield.** The only relevant code-side hook reference is the ERC-8183 contract ABI field `hook: address` in `erc8183_worker.py:79`. Any IRiskHook interface proposed in the paper would be a new design, not documentation of existing code.

---

## Task 4: Dimensional decomposition output shape

### GET /ahs/{address} — full scoring endpoint

**Response model**: `AHSResponse` wrapping `AHSReport` — `api.py:720-722`

```python
class AHSResponse(BaseModel):
    status: str           # "ok"
    report: AHSReport     # full report below
```

**AHSReport** — `api.py:703-718`:

```python
class AHSReport(BaseModel):
    address: str                           # wallet address
    agent_health_score: int                # composite 0-100
    grade: str                             # "C — Needs Attention"
    confidence: str                        # "HIGH" / "MEDIUM" / "LOW" / "INSUFFICIENT"
    mode: str                              # "2D" or "3D"
    dimensions: list[AHSDimensionScore]    # per-dimension breakdown
    patterns_detected: list[AHSCrossDimensionalPattern]
    trend: Optional[str]                   # "improving" / "stable" / "declining"
    recommendations: list[str]
    ahs_token: str                         # JWT for temporal tracking
    model_version: str                     # "AHS-v1"
    scan_timestamp: str
    next_scan_recommended: str
    shadow_signals: Optional[AHSShadowSignals]
    routing_recommendation: str            # "instant_settle" / "escrow" / "reject"
```

**AHSDimensionScore** — `api.py:683-687`:

```python
class AHSDimensionScore(BaseModel):
    dimension: str                # "D1: Wallet Hygiene", "D2: Behavioural Patterns", "D3: Infrastructure Health"
    score: int                    # 0-100
    weight: float                 # weight used in composite
    contributing_factors: list    # top scoring factors
```

**Dimension construction** — `api.py:4416-4438`:

D1 and D2 are always included. D3 is appended only if `result.d3_score is not None` (i.e. 3D mode). The `mode` field distinguishes "2D" vs "3D".

### GET /ahs/route/{address} — lightweight trust route

**Response model**: `TrustRouteResponse` — `api.py:755-764`:

```python
class TrustRouteResponse(BaseModel):
    address: str
    agent_health_score: int
    grade: str
    routing_recommendation: str     # "instant_settle" / "escrow" / "reject"
    confidence: str
    scored_at: str
    stale: bool                     # true if cached score > 24h old
    policy_applied: bool            # custom routing policy used
    allowlisted: bool               # address in caller's allowlist
```

This endpoint does **not** include per-dimension scores — it returns only the composite score and routing signal from the most recent cached result.

### Finding

The full `/ahs/{address}` response **does** include per-dimension scores (D1, D2, optionally D3) with individual scores, weights, and contributing factors. The `mode` field explicitly represents "2D" / "3D" framing. There is no "4D" mode in code. The lightweight `/ahs/route/{address}` omits dimensional decomposition entirely.

---

## Task 5: Cross-check against ALIA Section 3.2

### Source

`C:\Users\Pablo\Documents\alia-coauth\ahm-alia-section-3.2.md`

### Mismatches found

| # | ALIA Section 3.2 states | Code actually does | Severity |
|---|---|---|---|
| 1 | D1 = "transaction patterns" | D1 = "Wallet Hygiene" (`api.py:4418`) | **Naming mismatch** — different label, similar underlying concept |
| 2 | D2 = "behavioural diversity" | D2 = "Behavioural Patterns" (`api.py:4424`) | Minor — "diversity" vs "Patterns" |
| 3 | D3 = "cross-registry composition" | D3 = "Infrastructure Health" (`api.py:4434`) | **Significant mismatch** — completely different concept. Code probes agent URL infrastructure (uptime, TLS, response times); ALIA describes cross-registry presence signals |
| 4 | D4 = "task-execution / outcome quality" | **D4 does not exist in code** | **Major mismatch** — ALIA describes a fourth dimension that is not implemented |
| 5 | "AHS v9 at the time of writing" | `model_version = "AHS-v1"` (`monitor.py:1667`) | **Major mismatch** — no v9 exists |
| 6 | "decomposes into four dimensions" | Code implements 2 or 3 dimensions (2D/3D mode) | **Major mismatch** — ALIA claims four, code has at most three |
| 7 | Confidence enum: "HIGH / MEDIUM / LOW / INSUFFICIENT" | Same in code (`monitor.py:1692-1709`) | Match |
| 8 | "limited evidence does not equal adverse evidence" principle | Implemented: `tx_count < 10` → `INSUFFICIENT` rather than a low score | Match |
| 9 | Grades A through F | Implemented: A/B/C/D/E/F (`monitor.py:1677-1689`) | Match |

### Assessment

The ALIA Section 3.2 text has **three significant factual mismatches** with the production code:

1. **D3 description is wrong** — ALIA says "cross-registry composition"; the code implements infrastructure health probing (TLS, uptime, response characteristics). These are entirely different signal sources.

2. **D4 does not exist** — ALIA describes a fourth dimension ("task-execution / outcome quality") that has no implementation in the codebase. The code's mode system is strictly 2D (D1+D2) or 3D (D1+D2+D3).

3. **"AHS v9" is fabricated** — The only version string in code is `"AHS-v1"`. There is no version history, no increment mechanism, and no evidence that any version other than v1 has ever existed.

The confidence enum, grade bands, and "limited evidence ≠ adverse evidence" principle are accurately described.

---

## Summary

| Task | Key finding |
|---|---|
| 1. Weight renormalisation | Blog post weights (0.375/0.625) are **wrong**. Actual 2D: D1=0.30, D2=0.70. Actual 3D: D1=0.25, D2=0.45, D3=0.30. EMA α=0.6 confirmed. |
| 2. Confidence enum | `HIGH / MEDIUM / LOW / INSUFFICIENT` confirmed (uppercase strings). "AHS v9" does not exist — code has `AHS-v1`. |
| 3. IRiskHook | **No existing IRiskHook found.** Only a conceptual mention in backlog notes. Paper sketch is greenfield. |
| 4. Response shape | Full `/ahs/{address}` includes per-dimension scores, weights, factors, mode field. Route endpoint omits dimensions. No 4D mode. |
| 5. ALIA cross-check | Three major mismatches: D3 description wrong, D4 doesn't exist, "AHS v9" fabricated. Confidence and grade semantics match. |
