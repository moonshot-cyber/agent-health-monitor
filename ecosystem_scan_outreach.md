# Ecosystem Scan — Outreach Targets

> Scanned: 2026-03-16 15:37 UTC
> ETH price: $2,290.85

## Results

| Service | Domain | Wallet | AHS Score | Grade | D1 | D2 | Key Finding |
|---------|--------|--------|-----------|-------|----|----|-------------|
| KAMIYO | kamiyo.ai | `0x742d35...bee4` | N/A | No txs on Base |  |  | No Base activity detected |
| PayAI (signer 1) | payai.network | `0xc6699d...cb63` | 51 | D Degraded | 63 | 46 | none |
| PayAI (signer 2) | payai.network | `0xb2bd29...371b` | 61 | C Needs Attention | 62 | 61 | none |
| PayAI (example) | payai.network | `0x209693...287C` | 58 | D Degraded | 75 | 50 | none |
| Daydreams (EVM signer) | daydreams.systems | `0x1363C7...f678` | 56 | D Degraded | 64 | 52 | none |
| Daydreams (facilitator) | daydreams.systems | `0x279e08...4653` | 38 | E Critical | 67 | 40 | {'name': 'Stale Strategy', 'detected': True, 'severity': 'warning', 'description': 'Agent is repeatedly failing on the same contract interaction without adapting. Possible causes: revoked approval, removed liquidity, contract upgrade. Gas price is hardcoded (no adaptation).'} |
| Daydreams (payTo example) | daydreams.systems | `0xb308ed...E429` | 82 | B Good | 89 | 79 | none |
| AgentCard | agentcard.ai | — | — | — | — | — | No x402 wallet found; virtual card issuer, not x402 native |
| OpenServ | openserv.ai | — | — | — | — | — | No x402 wallet found; $SERV token on ETH only, x402 Lite announced |

## Discovery Notes

- **KAMIYO** — Facilitator at `kamiyo.ai/api/v1/x402`, gated (requires API key). Also has Solana address.
- **PayAI** — Facilitator at `facilitator.payai.network` (currently timing out). 5 Base signers found, 14 chains supported.
- **Daydreams** — Facilitator at `facilitator.daydreams.systems` (live). `/supported` endpoint returns structured signer map.
- **AgentCard** — Virtual Visa card issuer for agents. x402 support listed as coming. No wallet discoverable.
- **OpenServ** — $SERV token on Ethereum. x402 Lite announced but no facilitator or wallet addresses public.
- **Discovery method**: x402 Facilitators Watch (facilitators.x402.watch) was the most productive source. `.well-known/x402` returned 404 on all 5 services.
