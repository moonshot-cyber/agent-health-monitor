# Agent Health Monitor: OpenClaw plugin

This plugin adds native OpenClaw agent tools for [Agent Health Monitor](https://agenthealthmonitor.xyz).
AHM runs on-chain agent health diagnostics, trust scoring and wallet hygiene analysis, paid per call in USDC on
Base (`eip155:8453`) via x402.

## Install

```bash
openclaw plugins install clawhub:ahm-agent-health-monitor
```

## Tools

Every tool takes `address` (required, `0x` + 40 hex) and `coupon` (optional free-test code).

| Tool | Endpoint | Price | Timeout |
|---|---|---|---|
| `ahm_risk_check` | `GET /risk/{address}` | $0.001 | 30s |
| `ahm_premium_risk` | `GET /risk/premium/{address}` | $0.05 | 30s |
| `ahm_health` | `GET /health/{address}` | $0.50 | 60s |
| `ahm_wash` | `POST /wash/{address}` | $0.50 | 60s |
| `ahm_health_score` | `GET /ahs/{address}` (+ optional `agent_url` for D3) | $1.00 | 60s |
| `ahm_trust_route` | `GET /ahs/route/{address}` (no coupon route) | $0.01 | 30s |
| `ahm_counterparties` | `GET /counterparties/{address}` | $0.10 | 30s |
| `ahm_network_map` | `GET /network-map/{address}` | $0.10 | 30s |

With a `coupon`, the tool calls the path-based coupon route `/coupon/{slug}/{CODE}/{address}` instead of
the paid route.

## Payments

This plugin **does not hold a wallet or sign payments**. When an endpoint answers `402 Payment Required`,
the tool returns the decoded x402 payment requirements (amount, asset, `payTo`, network) so that the agent,
or an x402-capable client, can pay and retry. To test for free, use a coupon code.

## Results

Success:

```json
{ "success": true, "endpoint": "GET /risk/{address}", "price": "$0.001", "data": { "risk_score": 22, "risk_level": "LOW", "verdict": "..." } }
```

Error:

```json
{ "success": false, "error": "PAYMENT_REQUIRED", "message": "...", "details": { "httpStatus": 402, "paymentRequirements": { "x402Version": 2, "accepts": [ ... ] } } }
```

`error` is one of `INVALID_ADDRESS`, `PAYMENT_REQUIRED`, `TIMEOUT`, `RATE_LIMITED` or `SERVER_ERROR`.

## Build

```bash
npm install
npm run build
```

## Links

- Site: https://agenthealthmonitor.xyz
- Docs: https://docs.agenthealthmonitor.xyz
- Repo: https://github.com/moonshot-cyber/agent-health-monitor

License: MIT
