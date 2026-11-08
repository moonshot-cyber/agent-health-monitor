# Agent Health Monitor: OpenClaw skill

Trust layer for the agent economy. This skill teaches your OpenClaw agent to call
[Agent Health Monitor](https://agenthealthmonitor.xyz), which runs on-chain health diagnostics, trust
scoring and wallet hygiene analysis for any EVM address. Every call is paid in USDC on Base via x402.
There are no accounts and no API keys.

## Capabilities

- **Pre-flight risk checks:** a 0-100 risk score and verdict before you pay a counterparty
- **Agent Health Score (AHS):** a composite score across Wallet Hygiene, Behavioural Patterns and optional Infrastructure probing, with an A-F grade and cross-dimensional patterns such as Zombie Agent and Gas War Casualty
- **Trust routing:** `instant_settle` / `escrow` / `reject` before an agent-to-agent payment
- **Wallet hygiene scans:** dust, spam tokens, wasted gas, failure patterns and nonce gaps
- **Counterparty and funding-source investigation:** labelled counterparties and related wallets
- **Gas optimisation, failed-tx retry building, monitoring alerts and a full protection suite**

## Pricing

Pay per call in USDC on Base (`eip155:8453`). Prices run from **$0.001 to $25 per call**. A quick risk check costs $0.001, the
flagship AHS costs $1.00 and the full protection agent costs $25. You can get the live price list free at
[`/api/info`](https://agenthealthmonitor.xyz/api/info). Coupon codes give free test access.

## Install

```bash
openclaw skills install ahm-agent-health-monitor
```

See [SKILL.md](SKILL.md) for endpoints, score bands and payment flow, and [EXAMPLES.md](EXAMPLES.md) for
example conversations.

## Links

- Site: https://agenthealthmonitor.xyz
- Docs: https://docs.agenthealthmonitor.xyz
- Repo: https://github.com/moonshot-cyber/agent-health-monitor
