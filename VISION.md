# AHM Vision: The Trust Layer for the Agent Economy

## The Problem

The autonomous agent economy is growing fast. Agents delegate tasks to other agents, route payments, enter contracts, and manage assets — often without human oversight. But there's no standard way for an agent to answer three critical questions about a counterparty before transacting:

1. **Is it solvent?** Does this agent have the financial health to fulfil its obligations?
2. **Is it reliable?** Does it behave consistently, adapt to failures, and maintain stable patterns?
3. **Is it operational?** Is the infrastructure behind it responsive and available?

Without trust verification, the agent economy runs on reputation, self-reporting, or nothing at all. That's fragile. AHM exists to make trust objective, verifiable, and automatable.

## What AHM Does

AHM is a diagnostic API that scores agent health across three dimensions:

- **D1 — Solvency & Financial Health.** Token portfolio quality, gas efficiency, transaction success rates, financial hygiene. Answers: can this agent afford to operate?
- **D2 — Behavioural Consistency.** Timing regularity, counterparty diversity, adaptation patterns, failure recovery. Answers: does this agent behave predictably?
- **D3 — Operational Stability.** Endpoint availability, response times, error rates. Answers: is the agent's infrastructure actually running?

These combine into the **Agent Health Score (AHS)** — a composite 0-100 diagnostic that any agent can request before deciding whether to trust a counterparty.

## Why On-Chain Signals

AHM scores are derived from on-chain transaction history — not self-reported metrics, not peer reviews, not reputation tokens.

On-chain data has properties that make it uniquely suited for trust verification:

- **Unfalsifiable.** You can't fake a transaction history without spending real money.
- **Sybil-resistant.** Creating fake agents with strong health scores requires genuine on-chain activity over time.
- **Universally accessible.** Any agent can verify any other agent's history without permission.
- **Temporally rich.** Patterns emerge over weeks and months — timing regularity, counterparty diversity, session behaviour — that are hard to game.

Subjective ratings (stars, upvotes, attestations from friends) are trivially gameable. Transaction history is not.

## Pre-Transaction and Post-Transaction Verification

AHM currently serves **pre-transaction trust checks**: before an agent delegates a task or sends a payment, it calls `/risk` or `/ahs` to verify the counterparty.

The next step is closing the loop with **post-transaction verification**: after a job completes, verify that the outcome matches expectations. This creates a trust cycle:

```
Agent A                         AHM                         Agent B
  |                              |                              |
  |  Pre-check: /ahs/B          |                              |
  |----------------------------->|                              |
  |  AHS: 72 (Good)             |                              |
  |<-----------------------------|                              |
  |                              |                              |
  |  Delegates task to B         |                              |
  |----------------------------------------------------->      |
  |                              |                              |
  |  Post-check: verify outcome  |                              |
  |----------------------------->|                              |
  |  Outcome verified            |                              |
  |<-----------------------------|                              |
```

Pre-transaction gating prevents bad interactions. Post-transaction verification builds a record. Together they form a trust loop that improves over time.

## ERC-8183 Evaluator Opportunity

[ERC-8183](https://eips.ethereum.org/EIPS/eip-8183) defines a standard for agent-to-agent commerce: agents advertise capabilities, negotiate terms, and execute jobs. The spec includes an **evaluator** role — a third party that verifies job quality.

AHM is positioned to serve as an ERC-8183 evaluator:

- **Pre-job health gating.** Before an agent accepts a job, the evaluator checks the counterparty's AHS. Below a threshold? Job rejected.
- **Post-job outcome verification.** After the job, the evaluator checks whether the agent's on-chain state changed as expected (e.g., payment sent, contract deployed, tokens transferred).

This is where AHM evolves from a diagnostic tool into trust infrastructure — a service that other protocols call automatically as part of their agent commerce flow.

## Relationship to ERC-8004

[ERC-8004](https://eips.ethereum.org/EIPS/eip-8004) establishes on-chain agent identity — a registry where agents register their address, capabilities, and metadata. AHM is registered as agent #32328.

ERC-8004 answers "who is this agent?" AHM answers "should I trust this agent?" They're complementary layers: identity and health.

## The Implementation Layer

AHM is built on:

- **Base Mainnet** — Ethereum L2 for on-chain data and payment settlement
- **x402 Protocol** — HTTP-native micropayments in USDC, pay-per-call
- **Nansen API** — Wallet intelligence enrichment (labels, counterparties, PnL, related wallets)
- **Stripe** — Fiat API key access for developers without crypto wallets

These are implementation choices, not the product. The product is trust verification. The implementation could change (different chain, different payment rail, different data enrichment) without changing what AHM does.

## Where This Goes

1. **Today:** 13 diagnostic endpoints, 2,860+ agents in the trust registry, pre-transaction health checks.
2. **Next:** Post-transaction outcome verification, ERC-8183 evaluator integration, automated trust loops.
3. **Endgame:** A trust layer where agents prove their health, quality, and legitimacy to each other — automatically, with verifiable evidence, pay-per-verification.

The agent economy needs objective trust. AHM builds it from the only data source that can't lie: the chain itself.
