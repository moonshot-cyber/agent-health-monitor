# Confidence Overrides — Build Spec

**Status:** Draft — awaiting review before implementation
**Branch:** `feat/confidence-overrides-spec`
**Context:** PR #112 shipped configurable routing policies (`PUT /ahs/route/policy`). This spec extends that work with `confidence_overrides`: per-integrator overrides that modify the grade-based routing decision when a specific confidence level is present.

---

## 1. Schema Shape

### Current confidence levels

The AHS scoring pipeline (`monitor.py:1692`, `_ahs_confidence`) produces exactly four confidence levels:

- `HIGH` — tx_count >= 100, history >= 7 days (boosted by D3 or prior scan)
- `MEDIUM` — tx_count >= 50, history >= 3 days
- `LOW` — tx_count >= 10
- `INSUFFICIENT` — fewer than 10 transactions

The `GET /ahs/route/{address}` response returns confidence as a lowercase string (e.g. `"high"`), while `monitor.py` stores it uppercase (`"HIGH"`). The routing endpoint falls back to `"unknown"` when the stored value is missing (`api.py:4481`).

### Proposed JSON shape

**Option A — Flat dict keyed by confidence level:**

```json
{
  "confidence_overrides": {
    "HIGH": "instant_settle",
    "LOW": "escrow"
  }
}
```

Keys are confidence levels (`HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT`). Values are routing actions (`instant_settle`, `escrow`, `reject`). Only levels that differ from the grade-based default need to be specified — omitted levels use the grade-based decision as-is.

**Option B — Nested by grade then confidence:**

```json
{
  "confidence_overrides": {
    "C": {
      "HIGH": "instant_settle",
      "LOW": "reject"
    },
    "D": {
      "HIGH": "escrow"
    }
  }
}
```

Keys are grade letters; values are confidence-level-to-action mappings. Only grades/levels that need overriding are specified.

### Recommendation: Option B (nested by grade)

**Rationale:** Option A applies a blanket confidence override across all grades. This is too coarse — an integrator likely wants "Grade C + HIGH confidence → instant_settle" but would not want "Grade F + HIGH confidence → instant_settle". Option B gives per-grade granularity, which matches the motivating use case from the EthMag thread and avoids dangerous implicit semantics.

### Interaction with existing fields

`confidence_overrides` **layers on top of** the grade-based decision. The evaluation order is:

1. **Allowlist bypass** — if the address is in the allowlist, return `instant_settle` (no further checks).
2. **Grade-based decision** — use `instant_grades` / `escrow_grades` / `reject_grades` to determine the base action.
3. **Confidence override** — if `confidence_overrides` contains an entry for this grade AND this confidence level, replace the base action with the override value.
4. **`escrow_disabled` guard** — if the final action after all overrides is `escrow` but `escrow_disabled` is `true`, downgrade to `reject`.

The `escrow_disabled` guard runs *after* confidence overrides. This means an override that maps to `escrow` will still be blocked if `escrow_disabled` is true. This is deliberate — `escrow_disabled` is a hard constraint that means "never escrow".

### Validation rules

| Rule | Error |
|---|---|
| Confidence keys must be one of `HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT` | 400: "Invalid confidence level '{key}'" |
| Action values must be one of `instant_settle`, `escrow`, `reject` | 400: "Invalid routing action '{value}'" |
| Grade keys (outer) must be one of `A`-`F` | 400: "Invalid grade '{key}' in confidence_overrides" |
| Override grade must appear in one of `instant_grades`, `escrow_grades`, or `reject_grades` | 400: "Grade '{key}' in confidence_overrides not assigned to any routing category" |
| Override action `escrow` is incompatible with `escrow_disabled=true` | 400: "confidence_overrides maps to 'escrow' but escrow_disabled=true" |
| `confidence_overrides` must be an object (dict), not a list or scalar | 400 (Pydantic validation) |
| `confidence_overrides` is optional; default is `null` / empty | No error — grade-based routing applies unchanged |

---

## 2. Database

### Column type

Add a single `confidence_overrides` column of type `TEXT` to the `routing_policies` table, storing JSON-serialised data. This follows the same pattern used for `shadow_signals_json` on the `scans` table (`db.py:256`).

A separate normalised table is not warranted: the data is small (at most 6 grades x 4 confidence levels = 24 entries), is always read and written as a unit, and has no need for individual-row queries.

### Migration strategy

Follow the existing versioned-migration pattern in `db.py:init_db()`:

```python
# v10: Add confidence_overrides column for per-grade confidence-based
# routing overrides.
if current < 10:
    try:
        conn.execute(
            "ALTER TABLE routing_policies "
            "ADD COLUMN confidence_overrides TEXT DEFAULT NULL"
        )
    except sqlite3.OperationalError:
        pass  # Column already exists
```

Bump `_SCHEMA_VERSION` from `9` to `10`.

### Default value for existing rows

`NULL` — treated identically to an empty dict `{}`. When `confidence_overrides` is `NULL` or empty, the decision logic falls through to the grade-based default with no change in behavior. This ensures full backward compatibility for all existing policies.

---

## 3. Decision Logic

### Extended pseudocode for `_trust_routing_with_policy`

Current signature (`api.py:510`):

```python
def _trust_routing_with_policy(
    grade_letter: str,
    policy: dict | None = None,
    is_allowlisted: bool = False,
) -> str:
```

New signature adds `confidence`:

```python
def _trust_routing_with_policy(
    grade_letter: str,
    policy: dict | None = None,
    is_allowlisted: bool = False,
    confidence: str | None = None,
) -> str:
```

Full decision tree:

```
1. if is_allowlisted → return "instant_settle"
2. if policy is None → return _trust_routing(grade_letter)  # hardcoded defaults
3. Determine base action from grade lists:
   a. if grade_letter in instant_grades → base = "instant_settle"
   b. elif not escrow_disabled and grade_letter in escrow_grades → base = "escrow"
   c. else → base = "reject"
4. Apply confidence override:
   a. overrides = parse confidence_overrides from policy (JSON text → dict)
   b. if overrides and grade_letter in overrides:
        grade_overrides = overrides[grade_letter]
        normalised_confidence = (confidence or "").upper()
        if normalised_confidence in grade_overrides:
            base = grade_overrides[normalised_confidence]
5. Enforce escrow_disabled guard:
   a. if base == "escrow" and policy.get("escrow_disabled") → base = "reject"
6. return base
```

### Edge cases

| Scenario | Behavior |
|---|---|
| `confidence` is `None` or `"unknown"` | No override applied — normalised key won't match any valid confidence level. Grade-based default stands. |
| `confidence` is `"INSUFFICIENT"` | Override applied if there is an `INSUFFICIENT` entry for that grade. Useful for integrators who want to explicitly reject insufficient-data subjects. |
| `confidence_overrides` is `NULL` / empty dict | Step 4 is a no-op. Grade-based default stands. |
| Override maps Grade C + HIGH → `instant_settle`, but `escrow_disabled=true` | Override applies (C+HIGH → `instant_settle`). No conflict — `escrow_disabled` guard only catches `escrow` actions. |
| Override maps Grade C + LOW → `escrow`, but `escrow_disabled=true` | Override tries to set `escrow`, but Step 5 downgrades to `reject`. |
| Override targets a grade in `reject_grades` with action `instant_settle` | Allowed — this is the whole point. A Grade D agent with HIGH confidence can be promoted to `instant_settle`. |
| Case mismatch (`high` vs `HIGH`) | Normalise to uppercase before lookup (Step 4a). |

### Call-site update

`GET /ahs/route/{address}` (`api.py:4478`) currently calls:

```python
_trust_routing_with_policy(grade_letter, routing_policy, is_allowlisted)
```

Must pass the confidence from the cached record:

```python
_trust_routing_with_policy(
    grade_letter, routing_policy, is_allowlisted,
    confidence=record.get("confidence"),
)
```

---

## 4. Tests

### New test cases to add

Mirror the `test_escrow_disabled_*` and existing policy test patterns in `tests/test_trust_routing.py`.

#### PUT validation tests (in `TestRoutingPolicyValidation`)

| Test name | Scenario |
|---|---|
| `test_confidence_overrides_valid` | Valid overrides accepted (no error raised) |
| `test_confidence_overrides_invalid_confidence_key` | Key `"SUPER_HIGH"` → 400 |
| `test_confidence_overrides_invalid_action_value` | Value `"hold"` → 400 |
| `test_confidence_overrides_invalid_grade_key` | Outer key `"X"` → 400 |
| `test_confidence_overrides_grade_not_in_policy` | Grade key not in any grade list → 400 |
| `test_confidence_overrides_escrow_action_with_escrow_disabled` | Override maps to `"escrow"` + `escrow_disabled=true` → 400 |
| `test_confidence_overrides_empty_dict_accepted` | `{}` is valid (no-op) |
| `test_confidence_overrides_null_accepted` | `null` / omitted is valid |

#### Persistence round-trip tests (in `TestRoutingPolicyEndpoints`)

| Test name | Scenario |
|---|---|
| `test_put_with_confidence_overrides_roundtrip` | PUT with overrides → GET returns them in the response |
| `test_put_without_confidence_overrides_backward_compat` | PUT without the field → GET returns `null`/empty (existing tests still pass) |

#### Decision logic unit tests (in `TestTrustRoutingWithPolicy`)

| Test name | Scenario |
|---|---|
| `test_confidence_override_promotes_grade` | Grade C (normally escrow) + HIGH confidence → `instant_settle` |
| `test_confidence_override_demotes_grade` | Grade B (normally instant) + LOW confidence → `escrow` |
| `test_confidence_override_to_reject` | Grade C + INSUFFICIENT → `reject` |
| `test_confidence_override_no_match_uses_default` | Override exists for C+HIGH, query is C+MEDIUM → grade-based default |
| `test_confidence_override_empty_dict_no_effect` | Empty overrides → same as no overrides |
| `test_confidence_override_null_no_effect` | `None` overrides → same as no overrides |
| `test_confidence_override_case_insensitive` | `"high"` matches `"HIGH"` entry |
| `test_confidence_override_unknown_confidence_ignored` | `"unknown"` confidence → no override applied |
| `test_confidence_override_with_escrow_disabled_guard` | Override maps to `escrow` + `escrow_disabled` → downgraded to `reject` |
| `test_confidence_override_allowlist_takes_precedence` | Allowlisted address ignores all overrides → `instant_settle` |

#### Integration tests (in `TestRoutingPolicyEndpoints`)

| Test name | Scenario |
|---|---|
| `test_route_applies_confidence_override` | PUT policy with C+HIGH→instant, then GET route for Grade C address with high confidence → `instant_settle`, `policy_applied=true` |
| `test_route_no_confidence_override_match` | PUT policy with C+HIGH→instant, but address has medium confidence → default escrow |

### Existing tests that may need updating

- `TestTrustRoutingWithPolicy` — the `_trust_routing_with_policy` function signature changes (new `confidence` kwarg). Existing calls pass positional or keyword args without `confidence`, so they should continue to work since it defaults to `None`. **No changes needed** unless the function is refactored to require `confidence`.
- `TestRoutingPolicyEndpoints.test_put_and_get_roundtrip` — the GET response shape grows a new field (`confidence_overrides`). The test checks specific fields by key, so it should still pass, but a defensive assertion for the new field would be good.
- `RoutingPolicyResponse` model — gains a new field. Any test that constructs or asserts the full response dict will need updating.

---

## 5. API Surface and Backward Compatibility

### Versioning

No new endpoint version is needed. Adding an optional field (`confidence_overrides`) to the PUT request body is backward-compatible under standard REST conventions:

- Existing clients that omit `confidence_overrides` get the current behavior (field defaults to `None`).
- The GET response adds a new field — existing clients that don't read it are unaffected.

### Updated request model

```python
class RoutingPolicyRequest(BaseModel):
    instant_grades: list[str] = Field(...)
    escrow_grades: list[str] = Field(...)
    reject_grades: list[str] = Field(...)
    escrow_disabled: bool = Field(default=False, ...)
    allowlist: Optional[list[str]] = Field(default=None, ...)
    confidence_overrides: Optional[dict[str, dict[str, str]]] = Field(
        default=None,
        description=(
            "Per-grade confidence-level overrides. Outer keys are grades (A-F), "
            "inner keys are confidence levels (HIGH, MEDIUM, LOW, INSUFFICIENT), "
            "values are routing actions (instant_settle, escrow, reject)."
        ),
        examples=[{"C": {"HIGH": "instant_settle"}, "D": {"HIGH": "escrow"}}],
    )
```

### Updated response model

```python
class RoutingPolicyResponse(BaseModel):
    instant_grades: list[str] = Field(...)
    escrow_grades: list[str] = Field(...)
    reject_grades: list[str] = Field(...)
    escrow_disabled: bool = Field(...)
    allowlist_count: int = Field(...)
    confidence_overrides: Optional[dict[str, dict[str, str]]] = Field(
        default=None,
        description="Active confidence-level overrides (null if none configured)",
    )
    updated_at: str = Field(...)
```

### SDK impact (ahm-shield)

The `AHMClient` in `ahm-shield` (`ahm_shield/client.py`) only calls `GET /ahs/route/{address}` — it does not interact with `PUT /ahs/route/policy` at all. The route response shape is unchanged (routing_recommendation, grade, confidence, etc.). **No SDK changes needed.**

The client does `confidence=float(data.get("confidence", 0))` which casts confidence to float — this is a pre-existing bug (confidence is a string like `"high"`, not a number) but is unrelated to this feature. Worth noting for a separate fix.

---

## 6. Public-Facing Surface

### Documentation (docs.agenthealthmonitor.xyz)

**Out of scope for this build.** The API docs site will need an update to document the new `confidence_overrides` field on the PUT/GET policy endpoints. Flag as a follow-up task after implementation lands.

### EthMag ERC-8183 thread

**Separate item.** Once confidence overrides ship, a follow-up post on the EthMag thread should announce the feature as fulfilling the public commitment. Draft the post after the implementation PR merges.

---

## 7. Implementation Order

### Recommended sequence

1. **Database migration** — Add `confidence_overrides` column to `routing_policies`, bump schema version to 10. This is a prerequisite for everything else and is a safe, isolated change.

2. **Models** — Add `confidence_overrides` field to `RoutingPolicyRequest` and `RoutingPolicyResponse` in `api.py`.

3. **Validation** — Extend `_validate_routing_policy` with the new validation rules (invalid keys, action values, escrow_disabled conflict, etc.).

4. **Persistence** — Update `upsert_routing_policy` in `db.py` to accept and store `confidence_overrides` (JSON serialise on write, deserialise on read). Update `get_routing_policy` return value to include the field.

5. **Decision logic** — Extend `_trust_routing_with_policy` with the `confidence` parameter and override lookup. Update the call site in `GET /ahs/route/{address}`.

6. **PUT handler** — Update `put_routing_policy` to pass `confidence_overrides` through to the DB and include it in the response.

7. **GET handler** — Update `get_routing_policy` endpoint to include `confidence_overrides` in the response.

8. **Tests** — Add all test cases listed in Section 4. Run full suite to confirm no regressions.

### Estimated time per step

| Step | Hours |
|---|---|
| 1. Database migration | 0.5 |
| 2. Models | 0.5 |
| 3. Validation | 1 |
| 4. Persistence | 1 |
| 5. Decision logic | 1 |
| 6-7. PUT/GET handlers | 0.5 |
| 8. Tests | 2 |
| **Total** | **~6.5** |

---

## Open Decisions

All critical design decisions have proposed solutions above. The following are noted as "confirm before implementation":

1. **Case normalisation** — The spec proposes normalising confidence keys to uppercase on input and comparison. Confirm this is acceptable vs. storing as-provided and normalising only at comparison time.

2. **Validation strictness for override grades** — The spec proposes rejecting overrides for grades not present in any grade list. An alternative is to silently ignore them (they'd never match). Proposed: reject with 400 for early feedback.

3. **`INSUFFICIENT` as a valid override target** — The spec allows overriding `INSUFFICIENT` confidence. Confirm that integrators would realistically want this (e.g., "INSUFFICIENT → reject" as a conservative policy).
