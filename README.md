# Agent Health Monitor

Trust and health verification for the autonomous agent economy. Before an agent delegates a task, routes a payment, or enters a contract — it needs to know: **is this counterparty solvent, reliable, and operational?** AHM answers that with 13 diagnostic endpoints, 2,860+ agents in the trust registry, and composite scoring across three dimensions.

**Pay how you want:** [x402 protocol](https://x402.org) (USDC on Base, pay-per-call) or [Stripe](https://agenthealthmonitor.xyz) (fiat API key, no wallet required).

## What it does

**Screen** with `/risk` — **profile** with `/risk/premium` — **investigate** with `/counterparties` and `/network-map` — **diagnose** with `/health` — **clean** with `/wash` — **score** with `/ahs` — **batch score** with `/ahs/batch` — **visualise** with `/report-card` — **monitor** with `/alerts` — **fix** with `/optimize` — **retry** with `/retry` — **protect** with `/agent/protect`.

| Endpoint | Price (x402) | Purpose |
|---|---|---|
| `GET /risk/{address}` | $0.01 USDC | Pre-transaction trust check — is this agent safe to interact with? |
| `GET /risk/premium/{address}` | $0.05 USDC | Premium risk score with Nansen labels + PnL profitability summary |
| `GET /counterparties/{address}` | $0.10 USDC | Know Your Counterparty — top interactions enriched with Nansen |
| `GET /network-map/{address}` | $0.10 USDC | Wallet network map — funding, deployer & multisig links via Nansen |
| `GET /health/{address}` | $0.50 USDC | Full health diagnostic with solvency and operational analysis |
| `POST /wash/{address}` | $0.50 USDC | Financial health scan — portfolio quality, efficiency, and failure patterns |
| `GET /ahs/{address}` | $1.00 USDC | Agent Health Score — composite 0-100 blending solvency, behavioural consistency & operational stability |
| `GET /report-card/{address}` | $2.00 USDC | Visual report card with ecosystem benchmarks and shareable PNG |
| `GET /alerts/subscribe/{address}` | $2.00 USDC/month | Automated monitoring — webhook alerts every 6 hours |
| `GET /optimize/{address}` | $5.00 USDC | Operational efficiency report — per-transaction cost optimization |
| `POST /ahs/batch` | $10.00 USDC | Batch AHS scoring — up to 10 wallets per x402 call |
| `GET /retry/{address}` | $10.00 USDC | Retry failed transactions — ready-to-sign replacements |
| `GET /agent/protect/{address}` | $25.00 USDC | Full autonomous protection — triages risk, runs all needed services |

### Fiat pricing (Stripe API key)

No wallet required. Purchase an API key at [agenthealthmonitor.xyz](https://agenthealthmonitor.xyz) and pass it via `X-API-Key` header.

| Plan | Price | Credits |
|---|---|---|
| Starter | $9 | 100 calls (one-time) |
| Pro | $39 | 500 calls (one-time) |
| Unlimited | $99/mo | Unlimited calls (subscription) |

`/ahs/batch` via API key: 1 credit per wallet, up to 25 wallets per request.

### Nansen integrations (4 direct API connections)

- Wallet labels — smart money tags, entity identification (via Corbits proxy)
- Counterparties — top interactions with Nansen-enriched labels (direct)
- PnL summary — realized profit/loss, win rate, top tokens (direct)
- Related wallets — funding, deployer, and multisig connections (direct)

## Trust Registry

2,860+ agent wallets scanned and tracked across multiple discovery sources:

| Source | Description |
|---|---|
| **ACP (Virtuals)** | Automated nightly scans via `acpx.virtuals.io` API |
| **ERC-8004** | On-chain agent registry on Base mainnet (32,700+ registered) |
| **Olas** | Olas protocol agent registry, nightly scans |
| **API** | Wallets scanned via direct API calls |

All scan results are stored in the trust registry database with grade distribution, trend tracking, and ecosystem-wide health statistics available at `/trust-registry` and `/dashboard`.

## Live

| | |
|---|---|
| **Base URL** | `https://agenthealthmonitor.xyz` |
| **Dashboard** | `https://agenthealthmonitor.xyz/dashboard` |
| **Developer UI** | `https://agenthealthmonitor.xyz/app` |
| **API Docs** | `https://agenthealthmonitor.xyz/docs` |
| **Payment** | USDC via x402 protocol or Stripe fiat API key |
| **Network** | Base Mainnet (eip155:8453) |
| **ERC-8004 ID** | #32328 |

## Quick Start

### 1. Health Check ($0.50) — Diagnose the problem

```
GET /health/{wallet_address}
```

```bash
curl https://agenthealthmonitor.xyz/health/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Without an x402 payment header, you'll get a `402 Payment Required` response with payment instructions. An x402-enabled client handles this automatically.

#### Example Response

```json
{
  "status": "ok",
  "report": {
    "address": "0x1234...abcd",
    "is_contract": true,
    "health_score": 62.3,
    "optimization_priority": "HIGH",
    "total_transactions": 847,
    "successful": 761,
    "failed": 86,
    "success_rate_pct": 89.85,
    "total_gas_spent_eth": 0.01872,
    "wasted_gas_eth": 0.00294,
    "estimated_monthly_waste_usd": 14.70,
    "avg_gas_efficiency_pct": 71.2,
    "out_of_gas_count": 12,
    "reverted_count": 74,
    "nonce_gap_count": 3,
    "retry_count": 5,
    "top_failure_type": "reverted",
    "first_seen": "2024-06-12",
    "last_seen": "2026-02-15",
    "recommendations": [
      {
        "category": "reliability",
        "severity": "high",
        "message": "Transaction success rate is 89.85%. 86 of 847 transactions failed. Review contract interactions for revert conditions and add pre-flight simulation via eth_call before submitting."
      }
    ],
    "eth_price_usd": 2500.00,
    "analyzed_at": "2026-02-16T20:02:19Z"
  }
}
```

### 2. Agent Health Score ($1.00) — Composite diagnostic

```
GET /ahs/{wallet_address}
```

```bash
curl https://agenthealthmonitor.xyz/ahs/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Proprietary composite 0-100 diagnostic blending solvency & financial health (D1), behavioural consistency (D2), and operational stability (D3). Scores both EOA and smart contract wallets — smart contract wallets are automatically scored via token transfer analysis.

#### 3D Mode (with infrastructure probing)

```bash
curl "https://agenthealthmonitor.xyz/ahs/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045?agent_url=https://myagent.example.com"
```

#### Example Response

```json
{
  "status": "ok",
  "report": {
    "address": "0xd8dA...96045",
    "agent_health_score": 41,
    "grade": "D — Degraded",
    "confidence": "HIGH",
    "mode": "3D",
    "dimensions": [
      { "dimension": "D1: Wallet Hygiene", "score": 28, "weight": 0.30 },
      { "dimension": "D2: Behavioural Patterns", "score": 55, "weight": 0.50 },
      { "dimension": "D3: Infrastructure Health", "score": 38, "weight": 0.20 }
    ],
    "patterns_detected": [
      {
        "name": "Zombie Agent",
        "detected": true,
        "severity": "critical",
        "description": "Wallet active but infrastructure unresponsive — agent may be running headless with no oversight"
      }
    ],
    "trend": "declining",
    "recommendations": [
      "Restore agent service availability — health endpoint returned 503",
      "Clear 131 dust tokens and 255 spam tokens cluttering wallet"
    ],
    "ahs_token": "eyJhbGciOiJIUzI1NiIs...",
    "model_version": "AHS-v1",
    "scan_timestamp": "2026-03-08T14:30:00Z",
    "next_scan_recommended": "7 days"
  }
}
```

**Modes:**
- **2D** (default) — solvency & financial health + behavioural consistency only
- **3D** (with `agent_url`) — adds operational stability probing (uptime, latency, error rates)

**Cross-dimensional patterns** detected include: Zombie Agent, Cascading Infrastructure Failure, Stale Strategy, Gas War Casualty, Healthy Operator, and more.

**Trend tracking:** pass the returned `ahs_token` (JWT) as a query parameter on subsequent calls to track score changes over time with exponential moving average.

### 3. Batch AHS ($10.00 x402 / 1 credit per wallet via API key)

```
POST /ahs/batch
```

Score multiple agent wallets in a single request. Up to 10 wallets per x402 call ($10.00 flat), up to 25 via API key (1 credit per wallet). Concurrent scoring with rate-limited RPC calls.

```bash
curl -X POST https://agenthealthmonitor.xyz/ahs/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ahm_live_your_key_here" \
  -d '{"addresses": ["0x1234...", "0x5678...", "0x9abc..."]}'
```

#### Example Response

```json
{
  "results": [
    {
      "address": "0x1234...",
      "ahs_score": 72,
      "grade": "C",
      "d1_score": 68,
      "d2_score": 75,
      "pattern": "Healthy Operator",
      "verdict": "No issues detected"
    }
  ],
  "total_addresses": 3,
  "total_scored": 3,
  "credits_used": 3,
  "credits_remaining": 97,
  "errors": []
}
```

### 4. Report Card ($2.00) — Shareable visual report

```
GET /report-card/{wallet_address}
```

Generates a personalised 1200x675 PNG report card showing AHS score, grade, dimension breakdown, and percentile ranking against all 2,860+ scanned agents. Includes a pre-built share URL for X/Twitter.

### 5. Alert Monitoring ($2.00/month) — Stay on top of it

A two-step flow: pay to subscribe, then configure your webhook.

#### Step 1: Subscribe (x402 payment)

```
GET /alerts/subscribe/{wallet_address}
```

Pays $2.00 USDC and activates 30 days of monitoring. If already subscribed, extends by 30 days.

#### Step 2: Configure webhook (free)

```bash
curl -X POST https://agenthealthmonitor.xyz/alerts/configure \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0x1234...abcd",
    "webhook_url": "https://hooks.slack.com/services/T.../B.../xxx",
    "webhook_type": "slack",
    "thresholds": {
      "health_score": 70,
      "failure_rate": 30,
      "waste_usd": 50
    }
  }'
```

**Webhook types**: `"slack"`, `"discord"`, or `"generic"` (raw JSON).

#### Check subscription status (free)

```
GET /alerts/status/{wallet_address}
```

#### Unsubscribe (free)

```
DELETE /alerts/unsubscribe/{wallet_address}
```

### 6. Protection Agent ($25.00) — Autonomous full protection

One endpoint that triages risk and runs the right combination of all services automatically.

```
GET /agent/protect/{wallet_address}
```

#### Free Preview

See the risk level and which services would run before paying:

```
GET /agent/protect/preview/{wallet_address}
```

#### Triage Logic

| Health Score | Risk Level | Services Run |
|---|---|---|
| 90-100 | Low | Health check only. Recommend alerts setup. |
| 70-89 | Medium | Health check + Gas Optimizer |
| 50-69 | High | Health check + Gas Optimizer + RetryBot |
| 0-49 | Critical | All services + urgent issue flagging |

## The Service Funnel

```
  Screen    Profile    Investigate   Diagnose     Clean      Score     Batch     Report    Monitor      Fix       Retry     Protect
  $0.01     $0.05    $0.10 each     $0.50      $0.50      $1.00     $10.00    $2.00    $2.00/mo    $5.00     $10.00     $25.00

GET /risk  /premium  /counterparties  /health  POST /wash  /ahs   POST /ahs/  /report  /alerts/sub /optimize  /retry  /agent/protect
    |          |     /network-map        |          |        |     batch       card       |          |         |         |
    v          v          v              v          v        v        v          v         v          v         v         v
 "Score:    Nansen    Top counter-   Full health  Dust +  Composite  Up to    Visual    Every 6h:  Per-type  Ready-to  Autonomous
  8/100"    labels    parties +      score +      spam +  0-100 AHS  25       report    check &    gas       sign      triage +
            + PnL     network map    gas waste    hygiene 3D infra   wallets  card PNG  alert      limits    EIP-1559  all services
```

Or skip straight to `GET /agent/protect/{address}` ($25) and let the agent figure it out.

**Listed on:** [Virtuals ACP](https://app.virtuals.io) (11 offerings) · [x402scan](https://x402scan.com) · [Bankr Skills](https://bankr.chat) · [agdp.io](https://agdp.io) · [8004scan.io](https://8004scan.io)

## How Payment Works

### x402 (crypto — USDC on Base)

```
Agent                           API Server                    x402 Facilitator
  |                                |                                |
  |  GET /health/0x1234...         |                                |
  |------------------------------->|                                |
  |                                |                                |
  |  402 Payment Required          |                                |
  |  (price: $0.50 USDC, payTo)   |                                |
  |<-------------------------------|                                |
  |                                |                                |
  |  Sign USDC payment             |                                |
  |  GET /health/0x1234...         |                                |
  |  + X-PAYMENT header            |                                |
  |------------------------------->|                                |
  |                                |  Verify + settle payment       |
  |                                |------------------------------->|
  |                                |  Payment confirmed             |
  |                                |<-------------------------------|
  |                                |                                |
  |  200 OK (health report JSON)   |                                |
  |<-------------------------------|                                |
```

Any x402-compatible client (Python, TypeScript, Go) handles the payment flow automatically:

```python
from x402 import x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme

client = x402Client()
client.register("eip155:*", ExactEvmScheme(signer=your_wallet))

# Quick risk screen ($0.01)
risk = client.get("https://agenthealthmonitor.xyz/risk/0x1234...")

# Agent Health Score ($1.00)
ahs = client.get("https://agenthealthmonitor.xyz/ahs/0x1234...")

# Full autonomous protection ($25.00)
protection = client.get("https://agenthealthmonitor.xyz/agent/protect/0x1234...")
```

### Stripe (fiat — no wallet required)

Purchase an API key at [agenthealthmonitor.xyz](https://agenthealthmonitor.xyz), then pass it in the `X-API-Key` header:

```python
import httpx

headers = {"X-API-Key": "ahm_live_your_key_here"}

# Single AHS scan (1 credit)
resp = httpx.get("https://agenthealthmonitor.xyz/ahs/0x1234...", headers=headers)

# Batch AHS scan (1 credit per wallet, up to 25)
resp = httpx.post("https://agenthealthmonitor.xyz/ahs/batch", headers=headers,
                  json={"addresses": ["0x1234...", "0x5678..."]})
```

## Tech Stack

- **FastAPI** — async Python web framework
- **x402 SDK v2** — payment middleware with Bazaar discovery
- **Nansen API** — wallet intelligence (labels, counterparties, PnL, related wallets)
- **Blockscout API** — on-chain transaction data
- **Base Mainnet** — Ethereum L2 (implementation layer for payments and on-chain analysis)
- **Railway** — cloud deployment
- **SQLite (WAL)** — trust registry, scan history, API key management

## Self-Hosting

```bash
git clone https://github.com/moonshot-cyber/agent-health-monitor.git
cd agent-health-monitor
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set PAYMENT_ADDRESS to your wallet
uvicorn api:app --host 0.0.0.0 --port 4021
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PAYMENT_ADDRESS` | Yes | — | Your wallet address to receive USDC |
| `BASESCAN_API_KEY` | No | — | Blockscout API key (increases rate limits) |
| `FACILITATOR_URL` | No | `https://x402.org/facilitator` | x402 facilitator endpoint |
| `STRIPE_SECRET_KEY` | No | — | Stripe secret key for fiat API key system |
| `STRIPE_WEBHOOK_SECRET` | No | — | Stripe webhook signing secret |
| `RISK_PRICE_USD` | No | `$0.01` | Price per quick risk check |
| `PREMIUM_RISK_PRICE_USD` | No | `$0.05` | Price per premium risk + Nansen + PnL |
| `COUNTERPARTY_PRICE_USD` | No | `$0.10` | Price per counterparty analysis |
| `NETWORK_MAP_PRICE_USD` | No | `$0.10` | Price per network map |
| `PRICE_USD` | No | `$0.50` | Price per health check |
| `WASH_PRICE_USD` | No | `$0.50` | Price per wash hygiene scan |
| `AHS_PRICE_USD` | No | `$1.00` | Price per Agent Health Score |
| `REPORT_CARD_PRICE_USD` | No | `$2.00` | Price per report card |
| `AHS_BATCH_PRICE_USD` | No | `$10.00` | Price per batch AHS (up to 10 wallets) |
| `AHS_JWT_SECRET` | No | (generated) | Secret for signing AHS trend-tracking JWT tokens |
| `ALERT_PRICE_USD` | No | `$2.00` | Price per alert subscription (30 days) |
| `OPTIMIZE_PRICE_USD` | No | `$5.00` | Price per gas optimization |
| `RETRY_PRICE_USD` | No | `$10.00` | Price per retry analysis |
| `PROTECT_PRICE_USD` | No | `$25.00` | Price per protection agent run |
| `NANSEN_PAYER_PRIVATE_KEY` | No | — | Private key for Nansen x402 payments |
| `VALID_COUPONS` | No | — | Comma-separated coupon codes for free access |
| `NETWORK` | No | `eip155:8453` | CAIP-2 network ID (Base Mainnet) |
| `PORT` | No | `4021` | Server port |

## CLI Tools

```bash
# Analyze specific wallets
python monitor.py 0xADDRESS1 0xADDRESS2

# Analyze wallets from a file
python monitor.py -f addresses.txt -o report.csv

# Discover agent wallets on Base automatically
python discover.py --min-score 50 --top 30
```

## License

MIT
