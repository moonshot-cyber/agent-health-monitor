# Agent Health Monitor: example conversations

Base URL: `https://agenthealthmonitor.xyz`. All addresses are illustrative.

---

## 1. "Is this wallet safe?" → `/risk`

**User:** My agent is about to pay 0x4a9C…e21F for a data job. Is that wallet safe?

**Assistant (thinking):** This is a pre-flight counterparty check, so the cheapest call fits: `GET /risk/{address}` ($0.001).

```http
GET /risk/0x4a9c7d2b11f0e8a3c5d6b7e8f9a0b1c2d3e4e21f
→ 402 Payment Required   (x402 client pays $0.001 USDC on Base and retries)
→ 200 {"risk_score": 18, "risk_level": "LOW", "verdict": "Wallet operating normally with no significant issues detected"}
```

**Assistant:** It looks fine. The risk score is **18/100 (LOW)**, with no significant issues in its
transaction history. For a large payment, I can also run `/ahs/route` to get a settle, escrow or reject
recommendation, or `/risk/premium` ($0.05) to add Nansen labels and PnL.

---

## 2. "Give me a full health check on my agent" → `/ahs` with `agent_url`

**User:** Run a full health check on my trading agent. Its wallet is 0x9f3e…07aa and its API is at https://bot.example.com.

**Assistant (thinking):** The user wants the flagship composite score. They gave an endpoint, so I'll pass it as `agent_url` to turn on D3 Infrastructure (3D mode, weights 25/45/30).

```http
GET /ahs/0x9f3e5b7c1d2a4e6f8091a2b3c4d5e6f7a8b907aa?agent_url=https://bot.example.com
→ 402 → paid $1.00 → 200
{"status":"ok","report":{"agent_health_score":58,"grade":"D","mode":"3D","confidence":"high",
 "dimensions":[{"dimension":"D1","score":71,...},{"dimension":"D2","score":44,...},{"dimension":"D3","score":69,...}],
 "patterns_detected":[{"name":"Gas War Casualty","detected":true,"severity":"warning",...}],
 "routing_recommendation":"reject","recommendations":["Back off retries during gas spikes", "..."]}}
```

**Assistant:** Your agent scores **58/100, grade D (Degraded)** in 3D mode.
- Wallet hygiene (D1) is fine at 71.
- Behaviour (D2) is the weak spot at 44. AHM flagged **Gas War Casualty**: volatile gas, a high failure rate and repeated retry storms.
- Infrastructure (D3) is 69: the endpoint is reachable but not snappy.

At grade D, counterparties that use AHM routing will get `reject`. Fix the retry behaviour first, then
re-score next week.

---

## 3. "Clean up my wallet" → `/wash`

**User:** My agent wallet 0x1b2c…44d0 feels cluttered. Can you clean it up?

**Assistant (thinking):** This is a hygiene scan, `POST /wash/{address}` ($0.50). AHM diagnoses the wallet and recommends actions. It doesn't sign anything.

```http
POST /wash/0x1b2c3d4e5f60718293a4b5c6d7e8f9012a3b44d0
→ 402 → paid $0.50 → 200
{"status":"ok","report":{"cleanliness_score":61,"cleanliness_grade":"Needs Attention","total_issues":9,
 "dust_tokens":14,"spam_tokens":5,"spam_token_list":["FakeUSDT","SCAM-Airdrop"],"gas_efficiency_pct":71.4,
 "wasted_gas_usd":8.2,"nonce_gaps":0,"recommendations":["Revoke approvals for 5 spam tokens","Set tighter gas limits on swap calls"]}}
```

**Assistant:** The cleanliness score is **61 (Needs Attention)**, with 9 issues:
1. **5 spam tokens** (FakeUSDT, SCAM-Airdrop, …). Don't interact with them, and revoke any approvals.
2. **14 dust tokens**, worth very little. Consolidate them or ignore them.
3. **$8.20 of gas wasted** on failed transactions. Tighten gas limits on swap calls.

I can't sign transactions for you. If you want failed transactions re-built for you to sign,
`/retry` ($10) does that, and `/retry/preview` is free.

---

## 4. "Who has this wallet transacted with?" → `/counterparties`

**User:** Who has 0x7e8f…c3b1 been dealing with? I'm vetting them before a partnership.

**Assistant (thinking):** This is counterparty investigation, `GET /counterparties/{address}` ($0.10). If they want to know where the funds came from, `/network-map` ($0.10) is the follow-up.

```http
GET /counterparties/0x7e8f90a1b2c3d4e5f60718293a4b5c6d7e8fc3b1
→ 402 → paid $0.10 → 200
{"status":"ok","address":"0x7e8f…c3b1","total_counterparties":42,"nansen_available":true,
 "counterparties":[{"address":"0x3fC9…","label":"Uniswap V3: Router","interaction_count":311,"volume_usd":182400.0},
                   {"address":"0xA1b2…","label":"Tornado Cash","interaction_count":3,"volume_usd":9800.0}, ...]}
```

**Assistant:** It has **42 counterparties**. Most of its volume goes through Uniswap V3, which is normal for a
trading agent. One flag: **3 interactions with a Tornado Cash-labelled address (~$9.8k)**. That alone doesn't
prove anything bad, but it's worth asking about. Do you want me to run `/network-map` to see who funded or
deployed this wallet?
