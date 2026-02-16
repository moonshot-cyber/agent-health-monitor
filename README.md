# Agent Health Monitor

Pay-per-use API that analyzes Base blockchain agent wallets for transaction failures, gas waste, and optimization opportunities. Powered by [x402 protocol](https://x402.org).

## What it does

- Analyzes agent wallet transaction history on Base L2
- Calculates composite health score (0-100)
- Identifies failed transactions and their causes (out-of-gas, reverted)
- Detects nonce gaps and retry patterns
- Estimates monthly gas waste in USD
- Provides actionable optimization recommendations

## Live API

| | |
|---|---|
| **Base URL** | `https://web-production-a512e.up.railway.app` |
| **Docs** | `https://web-production-a512e.up.railway.app/docs` |
| **Pricing** | $0.50 USDC per health check via x402 |
| **Network** | Base Sepolia (eip155:84532) |

## Quick Start

```
GET /health/{wallet_address}
```

### Example Request

```bash
curl https://web-production-a512e.up.railway.app/health/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

Without an x402 payment header, you'll get a `402 Payment Required` response with payment instructions. An x402-enabled client handles this automatically.

### Example Response

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
        "category": "gas_management",
        "severity": "high",
        "message": "12 transactions ran out of gas. Increase gas limit estimates by 20-30% or use eth_estimateGas before submitting."
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

response = client.get("https://web-production-a512e.up.railway.app/health/0x1234...")
print(response.json())
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
