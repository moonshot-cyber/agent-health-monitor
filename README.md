# Agent Health Monitor

Pay-per-use API that analyzes Base blockchain agent wallets for transaction failures, gas waste, and optimization opportunities. Powered by [x402 protocol](https://x402.org).

## What it does

**Diagnose** with `/health` — **monitor** with `/alerts` — **fix** with `/optimize` — **retry** with `/retry` — **protect** with `/agent/protect`.

| Endpoint | Price | Purpose |
|---|---|---|
| `GET /health/{address}` | $0.50 USDC | Diagnose wallet health — score, failure rates, gas waste |
| `GET /alerts/subscribe/{address}` | $2.00 USDC/month | Automated monitoring — webhook alerts every 6 hours |
| `GET /optimize/{address}` | $5.00 USDC | Fix the problems — per-transaction-type gas optimization |
| `GET /retry/{address}` | $10.00 USDC | Retry failed transactions — ready-to-sign replacements |
| `GET /retry/preview/{address}` | Free | Preview retryable failure count and estimated savings |
| `GET /agent/protect/{address}` | $25.00 USDC | Full autonomous protection — triages risk, runs all needed services |
| `GET /agent/protect/preview/{address}` | Free | Preview risk level and recommended services |

- Analyzes agent wallet transaction history on Base L2
- Calculates composite health score (0-100)
- Identifies failed transactions and their causes (out-of-gas, reverted, slippage)
- Detects nonce gaps and retry patterns
- Estimates monthly gas waste in USD
- Monitors wallets on a schedule and sends webhook alerts
- Groups transactions by type and calculates optimal gas limits
- Builds optimized retry transactions with corrected gas parameters (non-custodial)
- Autonomous triage: scores risk and runs the right combination of services

## Live API

| | |
|---|---|
| **Base URL** | `https://web-production-a512e.up.railway.app` |
| **Docs** | `https://web-production-a512e.up.railway.app/docs` |
| **Payment** | USDC via x402 protocol |
| **Network** | Base Sepolia (eip155:84532) |

## Quick Start

### 1. Health Check ($0.50) — Diagnose the problem

```
GET /health/{wallet_address}
```

```bash
curl https://web-production-a512e.up.railway.app/health/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
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
      },
      {
        "category": "cost",
        "severity": "high",
        "message": "Estimated $14.70/month wasted on failed transactions (0.002940 ETH total). Implement transaction simulation (eth_call) before submission to avoid paying for failures."
      }
    ],
    "eth_price_usd": 2500.00,
    "analyzed_at": "2026-02-16T20:02:19Z"
  }
}
```

### 2. Alert Monitoring ($2.00/month) — Stay on top of it

A two-step flow: pay to subscribe, then configure your webhook.

#### Step 1: Subscribe (x402 payment)

```
GET /alerts/subscribe/{wallet_address}
```

Pays $2.00 USDC and activates 30 days of monitoring. If already subscribed, extends by 30 days.

#### Step 2: Configure webhook (free)

```bash
curl -X POST https://web-production-a512e.up.railway.app/alerts/configure \
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

**Default thresholds** (all configurable):
- Health score drops below **70**
- Failure rate exceeds **30%**
- Monthly gas waste exceeds **$50**

#### Check subscription status (free)

```
GET /alerts/status/{wallet_address}
```

#### Unsubscribe (free)

```
DELETE /alerts/unsubscribe/{wallet_address}
```

#### Example Alert (Slack)

```json
{
  "text": "*Agent Health Alert* — `0x1234...abcd`\n• *low_health_score*: Health score dropped to 58.2/100 (threshold: 70)\n• *high_gas_waste*: Estimated gas waste is $72.15/month (threshold: $50.00)"
}
```

### 3. Gas Optimization ($5.00) — Fix the problem

```
GET /optimize/{wallet_address}
```

```bash
curl https://web-production-a512e.up.railway.app/optimize/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

#### Example Response

```json
{
  "status": "ok",
  "report": {
    "address": "0x464fc339...e92ad",
    "total_transactions_analyzed": 998,
    "current_monthly_gas_usd": 346.49,
    "optimized_monthly_gas_usd": 275.67,
    "estimated_monthly_savings_usd": 70.82,
    "total_wasted_gas_eth": 0.00515463,
    "total_wasted_gas_usd": 10.05,
    "tx_types": [
      {
        "contract": "0x6bded42c...7891",
        "method_id": "0xb858183f",
        "method_label": "0xb858183f",
        "tx_count": 590,
        "failed_count": 146,
        "failure_rate_pct": 24.7,
        "current_avg_gas_limit": 271186,
        "current_p50_gas_used": 166053,
        "current_p95_gas_used": 197887,
        "optimal_gas_limit": 227570,
        "gas_limit_reduction_pct": 16.1,
        "wasted_gas_eth": 0.00348,
        "wasted_gas_usd": 6.78
      },
      {
        "contract": "0x6bded42c...7891",
        "method_id": "0x38ed1739",
        "method_label": "swapExactTokensForTokens",
        "tx_count": 100,
        "failed_count": 98,
        "failure_rate_pct": 98.0,
        "current_avg_gas_limit": 5000000,
        "current_p50_gas_used": 30996,
        "current_p95_gas_used": 39474,
        "optimal_gas_limit": 45395,
        "gas_limit_reduction_pct": 99.1,
        "wasted_gas_eth": 0.00145,
        "wasted_gas_usd": 2.83
      }
    ],
    "recommendations": [
      "High failure rate (98.0%) on swapExactTokensForTokens to 0x6bded42c...7891. Add eth_call simulation before submitting these transactions.",
      "Gas limits are 99.1% too high for swapExactTokensForTokens. Current avg: 5,000,000, optimal: 45,395. Use eth_estimateGas with a 15% buffer.",
      "Eliminating failed transactions would save ~$70.82/month (0.005155 ETH wasted so far)."
    ],
    "eth_price_usd": 1949.43,
    "analyzed_at": "2026-02-17T13:15:00Z"
  }
}
```

### 4. RetryBot ($10.00) — Retry failed transactions

Non-custodial: returns ready-to-sign EIP-1559 transactions. Your agent signs and submits.

```
GET /retry/{wallet_address}
```

```bash
curl https://web-production-a512e.up.railway.app/retry/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

#### Free Preview

Check how many failures are retryable before paying:

```
GET /retry/preview/{wallet_address}
```

#### How it works

1. Fetches all failed transactions for the address
2. Classifies each failure: `out_of_gas`, `reverted`, `nonce_conflict`, `slippage`
3. Filters to retryable failures only (skips nonce conflicts, contract-level reverts that would fail again)
4. Builds optimized replacements with:
   - Same `to`/`data`/`value` as the original
   - Gas limit: p95 of successful similar txs x 1.2 safety margin
   - Gas price: current base fee + priority fee (live from Base RPC)
5. Returns ready-to-sign transaction objects with cost estimates

#### Example Response

```json
{
  "status": "ok",
  "report": {
    "address": "0x1234...abcd",
    "failed_transactions_analyzed": 50,
    "retryable_count": 12,
    "retry_transactions": [
      {
        "original_tx_hash": "0xabc123...",
        "failure_reason": "out_of_gas",
        "optimized_transaction": {
          "to": "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
          "data": "0x3593564c...",
          "value": "0x0",
          "gas_limit": "0x55730",
          "max_fee_per_gas": "0x5f5e100",
          "max_priority_fee_per_gas": "0x2faf080"
        },
        "estimated_gas_cost_usd": 0.12,
        "confidence": "high"
      }
    ],
    "total_estimated_retry_cost_usd": 1.44,
    "potential_value_recovered_usd": 85.00,
    "analyzed_at": "2026-02-18T12:00:00Z"
  }
}
```

**Confidence levels:**
- `high` — out-of-gas failures (just needed more gas)
- `medium` — slippage reverts on DEX trades (market conditions may have changed)
- `low` — other reverts that used significant gas (root cause unclear)

### 5. Protection Agent ($25.00) — Autonomous full protection

One endpoint that triages risk and runs the right combination of all services automatically.

```
GET /agent/protect/{wallet_address}
```

```bash
curl https://web-production-a512e.up.railway.app/agent/protect/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

#### Free Preview

See the risk level and which services would run before paying:

```
GET /agent/protect/preview/{wallet_address}
```

#### Triage Logic

The protection agent scores the wallet first, then decides what to run:

| Health Score | Risk Level | Services Run |
|---|---|---|
| 90-100 | Low | Health check only. Recommend alerts setup. |
| 70-89 | Medium | Health check + Gas Optimizer |
| 50-69 | High | Health check + Gas Optimizer + RetryBot |
| 0-49 | Critical | All services + urgent issue flagging |

#### Example Response

```json
{
  "status": "ok",
  "report": {
    "address": "0x1234...abcd",
    "risk_level": "high",
    "health_score": 58,
    "services_run": ["health_check", "gas_optimizer", "retry_bot"],
    "summary": {
      "total_issues_found": 15,
      "total_potential_savings_usd": 156.00,
      "retry_transactions_ready": 8,
      "estimated_retry_cost_usd": 2.40
    },
    "health_report": {
      "address": "0x1234...abcd",
      "health_score": 58,
      "optimization_priority": "HIGH",
      "total_transactions": 500,
      "failed": 75,
      "success_rate_pct": 85.0,
      "recommendations": ["..."]
    },
    "gas_optimization": {
      "total_transactions_analyzed": 500,
      "estimated_monthly_savings_usd": 71.00,
      "tx_types": ["..."]
    },
    "retry_transactions": [
      {
        "original_tx_hash": "0xabc...",
        "failure_reason": "out_of_gas",
        "optimized_transaction": {
          "to": "0x...",
          "data": "0x...",
          "value": "0x0",
          "gas_limit": "0x55730",
          "max_fee_per_gas": "0x5f5e100",
          "max_priority_fee_per_gas": "0x2faf080"
        },
        "estimated_gas_cost_usd": 0.30,
        "confidence": "high"
      }
    ],
    "recommended_actions": [
      {
        "priority": 1,
        "action": "Execute retry transactions",
        "potential_value_usd": 85.00,
        "potential_savings_monthly_usd": 0,
        "description": "8 failed transactions can be retried with optimized gas parameters."
      },
      {
        "priority": 2,
        "action": "Apply gas limit optimizations",
        "potential_value_usd": 0,
        "potential_savings_monthly_usd": 71.00,
        "description": "Reduce gas limits on 5 transaction type(s)."
      },
      {
        "priority": 3,
        "action": "Set up monitoring alerts",
        "potential_value_usd": 0,
        "potential_savings_monthly_usd": 0,
        "description": "Subscribe to automated health monitoring at /alerts/subscribe/0x1234...abcd"
      }
    ],
    "analyzed_at": "2026-02-18T15:30:00Z"
  }
}
```

Actions are ranked by potential value recovered (highest dollar amount first).

## The Service Funnel

```
  Diagnose         Monitor           Fix            Retry          Protect
   $0.50         $2.00/month        $5.00          $10.00          $25.00

GET /health    GET /alerts/sub    GET /optimize   GET /retry    GET /agent/protect
     |         POST /configure         |               |              |
     v                |                v               v              v
"Score: 62"     Every 6 hours:   "22 tx types"   "12 retryable"  Runs all needed
"74% success"   check thresholds "limit 99%      "ready-to-sign"  services based
"$70/mo waste"  send alerts       too high"       EIP-1559 txs    on risk level
     |                |          "$70/mo savings"      |              |
     v                v                v               v              v
"You have a     "You'll know      "Here's        "Sign and       "Here's your
 problem."       immediately."     the fix."       submit."        action plan."
```

Or skip straight to `GET /agent/protect/{address}` and let the agent figure it out.

## How Payment Works

This API uses the [x402 protocol](https://docs.cdp.coinbase.com/x402/welcome) for payments:

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

Any x402-compatible client (Python, TypeScript, Go) handles the payment flow automatically. Install `x402[httpx,evm]` and wrap your HTTP client:

```python
from x402 import x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme

client = x402Client()
client.register("eip155:*", ExactEvmScheme(signer=your_wallet))

# Diagnose
health = client.get("https://web-production-a512e.up.railway.app/health/0x1234...")

# Subscribe to alerts
sub = client.get("https://web-production-a512e.up.railway.app/alerts/subscribe/0x1234...")

# Optimize
optimization = client.get("https://web-production-a512e.up.railway.app/optimize/0x1234...")

# Retry failed transactions
retries = client.get("https://web-production-a512e.up.railway.app/retry/0x1234...")

# Full autonomous protection (recommended)
protection = client.get("https://web-production-a512e.up.railway.app/agent/protect/0x1234...")
```

## Tech Stack

- **FastAPI** - async Python web framework
- **x402 SDK v2** - payment middleware with Bazaar discovery
- **Blockscout API** - on-chain transaction data (free, no key required)
- **Base L2** - Ethereum L2 network being analyzed
- **Railway** - cloud deployment

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
| `PRICE_USD` | No | `$0.50` | Price per health check |
| `ALERT_PRICE_USD` | No | `$2.00` | Price per alert subscription (30 days) |
| `OPTIMIZE_PRICE_USD` | No | `$5.00` | Price per gas optimization |
| `RETRY_PRICE_USD` | No | `$10.00` | Price per retry analysis |
| `PROTECT_PRICE_USD` | No | `$25.00` | Price per protection agent run |
| `NETWORK` | No | `eip155:84532` | CAIP-2 network ID |
| `PORT` | No | `4021` | Server port |

## CLI Tools

The repo also includes standalone CLI tools that run without the API:

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
