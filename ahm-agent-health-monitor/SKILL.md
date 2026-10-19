---
name: ahm-agent-health-monitor
description: On-chain agent health diagnostics, trust scoring, and wallet hygiene analysis via x402 micropayments on Base
version: 1.0.0
metadata:
  openclaw:
    emoji: "🩺"
    homepage: https://agenthealthmonitor.xyz
    requires:
      anyBins:
        - curl
---

# Agent Health Monitor

Agent Health Monitor (AHM) is a pay-per-call trust layer for the agent economy. It scores
EVM wallet addresses (agents, counterparties, funding sources) for risk, health, hygiene
and trustworthiness. Every paid call settles in USDC on Base through the x402 protocol.
You do not need an account, an API key or a subscription.

**Base URL:** `https://agenthealthmonitor.xyz`

## When to use this skill

Use AHM when the user, or an agent you operate, needs to:

- **Check a counterparty before transacting.** "Is this wallet safe to pay?" Run `GET /risk/{address}` as a cheap pre-flight check.
- **Get a risk or health score for an agent.** Use `/risk/premium` for labelled risk, `/health` for a full wallet diagnosis, or `/ahs` for the composite Agent Health Score.
- **Audit wallet hygiene.** Dust, spam tokens, wasted gas, failed-tx patterns and nonce gaps: `POST /wash/{address}`.
- **Assess behavioural or infrastructure health.** The `/ahs` endpoint scores behavioural patterns (D2). If you supply `?agent_url=`, it also probes the agent's live endpoint for infrastructure health (D3).
- **Decide trust routing before an agent-to-agent payment.** `GET /ahs/route/{address}` returns `instant_settle`, `escrow` or `reject` from the latest cached AHS score.
- **Investigate who a wallet deals with.** Use `/counterparties` for top counterparties with labels, and `/network-map` for funders, deployers and multisig relations.

Do **not** use AHM to move funds. Every endpoint is read-only analysis. `/retry` returns
unsigned transactions that the caller must sign, and AHM never touches private keys.

## Payment model (x402)

- **Protocol:** x402, the HTTP `402 Payment Required` flow (x402 v2)
- **Asset:** USDC on Base (`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`)
- **Network:** Base mainnet, CAIP-2 `eip155:8453`
- **Scheme:** `exact`

Flow:

1. Call the endpoint with no payment.
2. The server replies `402 Payment Required`. The payment requirements (amount in USDC base units, `payTo`, asset, network, timeout) come back **base64-encoded JSON in the `payment-required` response header**. The JSON body is empty (`{}`).
3. An x402 client signs a USDC authorisation for that amount and retries the same request with the payment header.
4. The server verifies and settles the payment, then returns the analysis.

Use an x402 client to automate this, such as `x402-fetch`/`x402-axios` (TypeScript) or `x402` (Python).
Without a client, report the 402 requirements to the user and do not try to pay by hand.

### Free testing with a coupon

Coupons skip payment. They are **path-based**, not query parameters:

```
GET /coupon/{slug}/{CODE}/{address}
```

| Paid endpoint | Coupon route |
|---|---|
| `GET /risk/{a}` | `GET /coupon/risk/{CODE}/{a}` |
| `GET /risk/premium/{a}` | `GET /coupon/risk-premium/{CODE}/{a}` |
| `GET /health/{a}` | `GET /coupon/health/{CODE}/{a}` |
| `POST /wash/{a}` | `GET /coupon/wash/{CODE}/{a}` (note: GET) |
| `GET /ahs/{a}` | `GET /coupon/ahs/{CODE}/{a}` (accepts `?agent_url=`) |
| `GET /counterparties/{a}` | `GET /coupon/counterparties/{CODE}/{a}` |
| `GET /network-map/{a}` | `GET /coupon/network-map/{CODE}/{a}` |
| `GET /optimize/{a}` | `GET /coupon/optimize/{CODE}/{a}` |
| `GET /retry/{a}` | `GET /coupon/retry/{CODE}/{a}` |
| `GET /agent/protect/{a}` | `GET /coupon/protect/{CODE}/{a}` |
| `GET /alerts/subscribe/{a}` | `GET /coupon/alerts/{CODE}/{a}` |
| `GET /report-card/{a}` | `GET /coupon/report-card/{CODE}/{a}` |

`/ahs/route` has no coupon route. An invalid code returns `403`. Coupon access is rate-limited to
5 requests per IP per minute and returns `429` past that. You can check a code with
`GET /coupon/validate/{CODE}`, which returns `{"valid": true|false}`.

## Endpoints and pricing

`{address}` must be a `0x`-prefixed, 40-hex-character EVM address.

| Method | Path | Price (USDC) | Description |
|---|---|---|---|
| GET | `/risk/{address}` | $0.001 | Quick 0-100 risk score and a one-line verdict. Built for high-volume pre-flight checks |
| GET | `/ahs/route/{address}` | $0.01 | Trust routing decision (`instant_settle` / `escrow` / `reject`) from the latest cached AHS. Returns 404 if the wallet has never been scored, so call `/ahs` first |
| GET | `/risk/premium/{address}` | $0.05 | Risk score plus Nansen labels, PnL summary and operational health (tx failure rate, nonce gaps) |
| GET | `/counterparties/{address}` | $0.10 | Know Your Counterparty: top counterparties with labels, interaction counts and volume |
| GET | `/network-map/{address}` | $0.10 | Related wallets: funders, deployers and multisig relations. Optional `?chain=` (default `ethereum`) |
| GET | `/health/{address}` | $0.50 | Full wallet health diagnosis: success rate, gas efficiency, nonce health, recommendations |
| POST | `/wash/{address}` | $0.50 | Agent hygiene scan: dust, spam tokens, gas efficiency, failure patterns, nonce gaps |
| GET | `/ahs/{address}` | $1.00 | **Flagship.** Agent Health Score: a composite 0-100 score with dimensions, patterns and routing |
| GET | `/report-card/{address}` | $2.00 | Visual report card with benchmarks |
| GET | `/alerts/subscribe/{address}` | $2.00 / 30 days | Automated monitoring every 6h with webhook alerts |
| GET | `/optimize/{address}` | $5.00 | Gas optimisation report by contract and method |
| GET | `/retry/{address}` | $10.00 | Ready-to-sign, re-optimised replacement transactions for recent failures |
| POST | `/ahs/batch` | $10.00 | Batch AHS: up to 10 wallets per x402 call |
| GET | `/agent/protect/{address}` | $25.00 | Autonomous protection agent that triages and runs the right combination of services |

Free: `GET /retry/preview/{address}`, `GET /agent/protect/preview/{address}`, `GET /api/info`
(the live price list) and `GET /.well-known/x402`.

### Agent Health Score (AHS) dimensions

| Dimension | Default (2D) weight | With `?agent_url=` (3D) |
|---|---|---|
| D1 Wallet Hygiene: dust, spam, gas efficiency, failed tx rate, nonce gaps | 0.30 | 0.25 |
| D2 Behavioural Patterns: tx diversity, retry storms, activity gaps, gas adaptation | 0.70 | 0.45 |
| D3 Infrastructure Health: live probing of the agent's endpoint | not scored | 0.30 |

D3 enters the composite only when you pass a public `http(s)` `agent_url`. Private IPs and
internal hosts are rejected with `400`. The response `mode` field reports `2D` or `3D`.

## Score bands

**Risk score** (`/risk`, `/risk/premium`; higher means riskier):

| Score | `risk_level` |
|---|---|
| 0-24 | `LOW` |
| 25-49 | `MEDIUM` |
| 50-74 | `HIGH` |
| 75-100 | `CRITICAL` |

**Agent Health Score** (`/ahs`; higher means healthier). There are six grades:

| Score | Grade | Label | Default routing |
|---|---|---|---|
| 90-100 | A | Excellent | `instant_settle` |
| 75-89 | B | Good | `instant_settle` |
| 60-74 | C | Needs Attention | `escrow` |
| 40-59 | D | Degraded | `reject` |
| 20-39 | E | Critical | `reject` |
| 0-19 | F | Failing | `reject` |

Integrators can override the routing mapping per policy (`PUT /ahs/route/policy`).

**Cleanliness score** (`/wash`):

| Score | `cleanliness_grade` |
|---|---|
| 90+ | Spotless |
| 70-89 | Clean |
| 50-69 | Needs Attention |
| 30-49 | Dirty |
| <30 | Critical |

## Cross-dimensional AHS patterns

`/ahs` returns `patterns_detected[]`. Each entry has a `name`, `detected`, a `severity` (`info` / `warning` / `critical`) and a `description`:

- **Zombie Agent:** automation that still runs but does nothing useful. High gas efficiency, zero failures, minimal diversity, a single contract.
- **Cascading Infrastructure Failure:** nonce gaps alongside rising failure trends. Infrastructure is degrading under load.
- **Stale Strategy:** long runs of consecutive failures with no change in tx mix or gas, so the agent isn't adapting.
- **Healthy Operator:** good hygiene, adaptive gas, diverse activity, no retry storms and a healthy infrastructure (D3).
- **Gas War Casualty:** volatile gas, a high failure rate and repeated retry storms with poor gas adaptation.
- **Phantom Activity:** the endpoint is live and healthy (D3 ≥ 70) but recent on-chain activity is near zero. Needs `agent_url`.
- **Recovery in Progress:** the recent failure rate is improving against the historical baseline.

## Response shapes

`GET /risk/{address}`:

```json
{ "risk_score": 22, "risk_level": "LOW", "verdict": "Wallet operating normally with no significant issues detected" }
```

`GET /ahs/{address}` returns `{ "status": "ok", "report": {...} }`. The main `report` fields are:

```json
{
  "address": "0x...",
  "agent_health_score": 67,
  "grade": "C",
  "confidence": "high",
  "mode": "2D",
  "dimensions": [{ "dimension": "D1", "score": 72, "weight": 0.3, "contributing_factors": [] }],
  "patterns_detected": [{ "name": "Zombie Agent", "detected": true, "severity": "warning", "description": "..." }],
  "trend": "stable",
  "recommendations": ["..."],
  "routing_recommendation": "escrow",
  "scan_timestamp": "2026-10-19T10:00:00Z",
  "next_scan_recommended": "2026-10-26T10:00:00Z"
}
```

`GET /ahs/route/{address}`:

```json
{ "address": "0x...", "agent_health_score": 67, "grade": "C", "routing_recommendation": "escrow",
  "confidence": "high", "scored_at": "...", "stale": false, "policy_applied": false, "allowlisted": false }
```

`POST /wash/{address}` returns `{ "status": "ok", "report": { "cleanliness_score", "cleanliness_grade", "total_issues",
"issues_by_severity", "dust_tokens", "spam_tokens", "spam_token_list", "gas_efficiency_pct", "wasted_gas_usd",
"failed_tx_patterns", "nonce_gaps", "issues": [...], "recommendations": [...] } }`.

`GET /counterparties/{address}` returns `{ "status", "address", "counterparties": [{ "address", "label",
"interaction_count", "volume_usd", "last_interaction" }], "total_counterparties", "nansen_available" }`.

`GET /network-map/{address}` returns `{ "status", "address", "chain", "related_wallets": [{ "address", "label",
"relation", "chain", "transaction_hash", "block_timestamp" }], "total_related", "nansen_available" }`.

## HTTP usage examples

See the payment requirements. This returns a 402 and needs no wallet:

```bash
curl -si https://agenthealthmonitor.xyz/risk/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae \
  | grep -i '^payment-required:' | cut -d' ' -f2 | base64 -d
```

Free test with a coupon:

```bash
curl -s https://agenthealthmonitor.xyz/coupon/risk/YOURCODE/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae
curl -s "https://agenthealthmonitor.xyz/coupon/ahs/YOURCODE/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae?agent_url=https://my-agent.example.com"
```

Paid call with an x402 client (TypeScript):

```ts
import { wrapFetchWithPayment } from "x402-fetch";
const fetch402 = wrapFetchWithPayment(fetch, walletClient); // a viem wallet holding USDC on Base

const res = await fetch402("https://agenthealthmonitor.xyz/ahs/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae?agent_url=https://my-agent.example.com");
const { report } = await res.json();
console.log(report.agent_health_score, report.grade, report.routing_recommendation);
```

Wash is a POST:

```ts
await fetch402("https://agenthealthmonitor.xyz/wash/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae", { method: "POST" });
```

Pre-flight pattern for an agent about to pay another agent:

```text
1. GET /ahs/route/{payee}   -> instant_settle: pay; escrow: use escrow; reject: abort
   (on 404 "No score available": GET /ahs/{payee} first, then re-route)
2. For a cheap first look at an unknown wallet: GET /risk/{payee}; treat HIGH/CRITICAL as abort.
```

## Errors

| Status | Meaning |
|---|---|
| 402 | Payment required. Read the `payment-required` header |
| 400 / 422 | Invalid address or an invalid `agent_url` |
| 403 | Invalid coupon code |
| 404 | `/ahs/route`: the wallet hasn't been scored yet |
| 429 | Rate limited (coupon routes: 5 per minute per IP) |
| 5xx | Upstream data provider or server error. Retry later |

## Resources

- Site: https://agenthealthmonitor.xyz
- Docs: https://docs.agenthealthmonitor.xyz
- Live price list: https://agenthealthmonitor.xyz/api/info
- Source: https://github.com/moonshot-cyber/agent-health-monitor
- x402: https://www.x402.org
