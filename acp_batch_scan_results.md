# Proactive Ecosystem Scan Spike — ACP (agdp.io)

> Scanned: 2026-03-24 15:53 UTC
> Source: ACP API (`acpx.virtuals.io/api/agents`)
> Scan mode: 2D (D1+D2 only, free APIs, no Nansen)

## Spike Objective

Prove the end-to-end proactive scanning flow: discover agent wallets from an external registry, run AHS scans, store results in the database. This is the first step toward Priority 4 (Proactive Ecosystem Scanning) in the AHM backlog.

## Source Assessment

| Source | Status | Notes |
|--------|--------|-------|
| Virtuals ACP (agdp.io) | **Selected** | Free API, no auth, 40K+ agents with `walletAddress` field |
| x402scan | Blocked | All endpoints paywalled ($0.01/call via x402) |
| 402index.io | Viable (backup) | Free API but no payTo addresses — requires 2-step probe |
| Virtuals Protocol API | Blocked | Agents share TBA shards — per-agent AHS not meaningful |

## Discovery Summary

- **Total agents in ACP registry:** 40,520
- **Agents fetched (sorted by successfulJobCount:desc):** 500
- **Unique wallet addresses:** 500
- **Shared wallets (>1 agent per wallet):** 0
- **Max agents sharing one wallet:** 1
- **Owner/agent wallet overlap:** 0

**Wallet sharing assessment:** No wallet sharing detected. ACP agent wallets are independent — ideal for per-agent health scoring.

## AHS Scan Results

- **Wallets scanned:** 473
- **Average AHS:** 47.1
- **AHS range:** 28-77
- **Grade distribution:** B=2, C=87, D=206, E=178

### Grade Distribution

| Grade | Count | % | Score Range |
|-------|-------|---|-------------|
| A | 0 | 0.0% | 90-100 |
| B | 2 | 0.4% | 75-89 |
| C | 87 | 18.4% | 60-74 |
| D | 206 | 43.6% | 40-59 |
| E | 178 | 37.6% | 20-39 |
| F | 0 | 0.0% | 0-19 |

### Scanned Agents (by AHS)

| ACP ID | Name | Wallet | Jobs | Revenue | AHS | Grade | D1 | D2 | Patterns |
|--------|------|--------|------|---------|-----|-------|----|----|----------|
| 21876 | donatello-insider | `0x66ed6a...a09f` | 113 | $11 | 76 | B Good | 75 | 76 |  |
| 19186 | x402search | `0x9c4f5e...fe3a` | 170 | $2 | 74 | C Needs Attention | 75 | 73 |  |
| 4299 | Hype Scanner | `0x6c3e50...2b63` | 110 | $2 | 73 | C Needs Attention | 75 | 72 |  |
| 14567 | DeFi Scout | `0xc946bc...f4a7` | 98 | $2 | 72 | C Needs Attention | 75 | 71 |  |
| 1522 | WAYEger | `0xbc1577...d3e8` | 193 | $13 | 71 | C Needs Attention | 72 | 70 |  |
| 1369 | Olivia Trending | `0x5c4862...4cda` | 127 | $22 | 71 | C Needs Attention | 74 | 70 |  |
| 315 | Jarvis_SELLER | `0x0bb03e...20ae` | 284 | $0 | 70 | C Needs Attention | 61 | 74 |  |
| 18013 | JobRouter | `0xa5b231...d5e0` | 226 | $0 | 69 | C Needs Attention | 75 | 66 |  |
| 1754 | The UI/UX Guru | `0xdd8b45...fb35` | 155 | $140 | 69 | C Needs Attention | 75 | 67 |  |
| 1319 | Sentry:WachAI | `0xa05b16...a5c1` | 80 | $1 | 69 | C Needs Attention | 75 | 67 |  |
| 18015 | HappyClaw | `0x74cd13...5547` | 80 | $8 | 69 | C Needs Attention | 75 | 66 |  |
| 149 | nAIncy | `0x9d9922...8d8c` | 300 | $0 | 68 | C Needs Attention | 69 | 67 |  |
| 1321 | WachAI Mesh | `0xd428d9...1aaa` | 138 | $14 | 68 | C Needs Attention | 69 | 67 |  |
| 2575 | ChainScope | `0x9f8aa5...31b1` | 79 | $4 | 68 | C Needs Attention | 75 | 65 |  |
| 7528 | DUELS PRODUCTION | `0xa2d4c6...535e` | 89 | $0 | 67 | C Needs Attention | 74 | 64 |  |
| 1508 | BYTEST | `0x2d307d...fcfb` | 119 | $0 | 66 | C Needs Attention | 74 | 62 |  |
| 3154 | ACP Navigator | `0xcd72d8...a6e6` | 273 | $40 | 64 | C Needs Attention | 74 | 59 |  |
| 6062 | Aaga | `0xcd42e9...14e4` | 77 | $39 | 64 | C Needs Attention | 75 | 60 |  |
| 42 | Acolyt | `0xedaf82...d1c6` | 72 | $0 | 64 | C Needs Attention | 71 | 61 |  |
| 82 | devrel_buyer | `0xc1f756...76a6` | 166 | $0 | 63 | C Needs Attention | 68 | 61 |  |
| 703 | MoonFundTest | `0x28732f...48e5` | 75 | $0 | 63 | C Needs Attention | 75 | 58 |  |
| 4004 | Ethen | `0x1542c8...aee0` | 75 | $8 | 63 | C Needs Attention | 75 | 58 |  |
| 6479 | AgentHub | `0x803f95...d8c7` | 331 | $12 | 62 | C Needs Attention | 75 | 56 |  |
| 3121 | Ser Winston | `0xb01f8d...d7bc` | 321 | $40,410 | 62 | C Needs Attention | 75 | 56 |  |
| 1398 | xportalx | `0x28ab9e...79e6` | 297 | $44 | 62 | C Needs Attention | 75 | 56 |  |
| 17531 | Pachinko game agent | `0x93d349...1cc7` | 295 | $130 | 62 | C Needs Attention | 75 | 56 |  |
| 2255 | ClawOracle | `0x16b82c...2751` | 293 | $2,405 | 62 | C Needs Attention | 75 | 56 |  |
| 2319 | Zentrix AI | `0x135ac6...6dae` | 241 | $1,977 | 62 | C Needs Attention | 75 | 56 |  |
| 21183 | donatello-sentiment | `0xdc6f54...32f1` | 232 | $1,264 | 62 | C Needs Attention | 75 | 56 |  |
| 21234 | donatello-risk | `0x07a688...b0ca` | 204 | $1,087 | 62 | C Needs Attention | 75 | 56 |  |
| 21235 | donatello-odds | `0xe5dd2e...9482` | 198 | $1,061 | 62 | C Needs Attention | 75 | 56 |  |
| 21877 | donatello-gm | `0x281410...b48b` | 196 | $1,049 | 62 | C Needs Attention | 75 | 56 |  |
| 19156 | SERVO | `0x776839...79d2` | 173 | $3,480 | 62 | C Needs Attention | 75 | 57 |  |
| 3636 | ChainPulse Data Microdesk | `0xf69004...4228` | 141 | $0 | 62 | C Needs Attention | 75 | 56 |  |
| 4045 | Test agent | `0xb00478...364a` | 124 | $0 | 62 | C Needs Attention | 75 | 56 |  |
| 9352 | Manus - #1 ACP Marketing  | `0xbd89f0...87bc` | 123 | $139 | 62 | C Needs Attention | 75 | 56 |  |
| 19514 | $TOKEN | `0x2f6b4c...27a5` | 110 | $2,851 | 62 | C Needs Attention | 75 | 56 |  |
| 3640 | DTINO | `0xc43a7f...6d01` | 108 | $0 | 62 | C Needs Attention | 75 | 56 |  |
| 3595 | YhalResearch | `0x94113a...3e7b` | 108 | $1 | 62 | C Needs Attention | 75 | 56 |  |
| 5197 | korea alpha - what's next | `0x2be142...4e0b` | 106 | $1 | 62 | C Needs Attention | 75 | 56 |  |
| 3669 | The Studio | `0x779b72...8300` | 90 | $0 | 62 | C Needs Attention | 75 | 56 |  |
| 2052 | Virts | `0x6ca3e9...9ed5` | 87 | $31 | 62 | C Needs Attention | 75 | 56 |  |
| 4419 | Unlock Tracker | `0x8e4e35...9e46` | 76 | $2 | 62 | C Needs Attention | 75 | 57 |  |
| 300 | Test OOPZ Buyer | `0xfcd749...67b2` | 73 | $0 | 62 | C Needs Attention | 73 | 57 |  |
| 1011 | Zyfai Agent | `0x5c134c...c470` | 269 | $4 | 61 | C Needs Attention | 74 | 55 |  |
| 1679 | test_owl | `0xe11535...9b01` | 235 | $30 | 61 | C Needs Attention | 55 | 64 |  |
| 5020 | PrismAI | `0x2ff089...1048` | 195 | $8 | 61 | C Needs Attention | 74 | 56 |  |
| 277 | Bizzy | `0x27f52d...bdcb` | 99 | $0 | 61 | C Needs Attention | 73 | 56 |  |
| 1245 | V.V. Butler | `0xc709d9...3173` | 250 | $0 | 60 | C Needs Attention | 71 | 56 |  |
| 396 | Sam-Test | `0xbac568...53d4` | 155 | $0 | 60 | C Needs Attention | 70 | 56 |  |
| 939 | AIxBET | `0x22330b...3823` | 270 | $67 | 59 | D Degraded | 74 | 52 |  |
| 105 | Tal Requestor | `0xc97bfe...45f9` | 100 | $0 | 59 | D Degraded | 70 | 54 |  |
| 473 | AskTianAI | `0x8a63d2...7052` | 243 | $41 | 58 | D Degraded | 72 | 52 |  |
| 1920 | ZeroAgent | `0x017cf0...734c` | 134 | $16 | 58 | D Degraded | 75 | 51 |  |
| 20907 | poolRadar | `0x670fcf...d22d` | 103 | $304 | 58 | D Degraded | 75 | 51 |  |
| 20905 | quantBot | `0x4c7150...cea6` | 96 | $290 | 58 | D Degraded | 75 | 51 |  |
| 20903 | sigmaAlpha | `0x185073...1de9` | 93 | $286 | 58 | D Degraded | 75 | 51 |  |
| 19840 | tradybot | `0x454292...d453` | 80 | $8 | 58 | D Degraded | 75 | 50 |  |
| 18016 | HelloClaw | `0xe0848f...7993` | 79 | $8 | 58 | D Degraded | 75 | 50 |  |
| 10107 | CardOracle | `0x3e9d24...cf0c` | 78 | $2 | 58 | D Degraded | 75 | 50 |  |
| 29588 | EvalLayer | `0x8442d8...b7e4` | 78 | $0 | 58 | D Degraded | 75 | 51 |  |
| 18336 | REQUESTOR_DEBUGGOR2 | `0x42b6d2...f6da` | 75 | $0 | 58 | D Degraded | 75 | 50 |  |
| 4750 | test_hl_buyer | `0x3cc39a...4144` | 222 | $0 | 57 | D Degraded | 72 | 50 |  |
| 4674 | test_hl | `0xee39db...2909` | 222 | $0 | 57 | D Degraded | 73 | 50 |  |
| 204 | Karum | `0xb6cdae...d645` | 128 | $0 | 57 | D Degraded | 68 | 53 |  |
| 2624 | Jeff CEO | `0x92ebf2...53e9` | 80 | $0 | 57 | D Degraded | 70 | 51 |  |
| 1233 | Generative Market eXplore | `0x05e6b4...8042` | 200 | $257 | 56 | D Degraded | 74 | 48 |  |
| 483 | Traveler (Buyer) | `0x7a3a5d...6cd2` | 322 | $0 | 54 | D Degraded | 72 | 46 |  |
| 162 | The SWARM (by SLAMai) | `0xa17b17...6de4` | 240 | $477 | 54 | D Degraded | 71 | 46 |  |
| 869 | Trading Seller | `0xad7159...e68b` | 161 | $0 | 54 | D Degraded | 73 | 46 |  |
| 6791 | AgentReputer | `0x47ba35...e2b9` | 151 | $4 | 54 | D Degraded | 75 | 45 |  |
| 4095 | GemClaw | `0xc6f136...9ba3` | 145 | $45 | 54 | D Degraded | 75 | 45 |  |
| 827 | Elytra | `0x8c9b7e...31c2` | 144 | $6 | 54 | D Degraded | 74 | 45 |  |
| 2230 | TRENCHOR | `0x50172b...165e` | 122 | $182 | 54 | D Degraded | 75 | 45 |  |
| 20906 | deepScan | `0x934254...5357` | 98 | $292 | 54 | D Degraded | 75 | 45 |  |
| 20904 | chainLens | `0xc77191...31e6` | 96 | $287 | 54 | D Degraded | 75 | 45 |  |
| 446 | DaVinci | `0x4a227d...f964` | 187 | $0 | 53 | D Degraded | 72 | 45 |  |
| 6783 | KimchiAlpha | `0x7caac5...7409` | 115 | $4 | 53 | D Degraded | 75 | 44 |  |
| 2966 | OutreachBot | `0xc3389e...dc46` | 111 | $0 | 53 | D Degraded | 75 | 44 |  |
| 10331 | AgentRank | `0x154080...c342` | 110 | $2 | 53 | D Degraded | 75 | 44 |  |
| 9386 | Cerebro AI | `0xce246a...e401` | 92 | $14 | 53 | D Degraded | 75 | 44 |  |
| 862 | WEBthe3rd (test provider) | `0x6bf4ee...bcf7` | 81 | $0 | 53 | D Degraded | 75 | 44 |  |
| 20713 | ReelWraith | `0x8062b7...8010` | 329 | $13 | 52 | D Degraded | 75 | 42 |  |
| 2967 | ContentSwarm | `0xcdb67f...5bc0` | 127 | $0 | 52 | D Degraded | 75 | 42 |  |
| 219 | Sovra AI | `0xd836f1...e4ee` | 114 | $0 | 52 | D Degraded | 72 | 44 |  |
| 654 | Batman_seller | `0xa15818...fb1f` | 95 | $0 | 52 | D Degraded | 74 | 43 |  |
| 775 | Bios Staging | `0x239a8f...2b58` | 84 | $0 | 52 | D Degraded | 74 | 43 |  |
| 20724 | MysteryVault | `0x8c3810...3c0d` | 229 | $16 | 51 | D Degraded | 75 | 41 |  |
| 903 | Soros dev test | `0x77e3a8...720c` | 221 | $0 | 51 | D Degraded | 70 | 43 |  |
| 1809 | Orion | `0x6896dc...ce13` | 188 | $35 | 51 | D Degraded | 71 | 42 |  |
| 1403 | SimonAgent | `0x4bc698...6d65` | 152 | $0 | 51 | D Degraded | 74 | 41 |  |
| 3869 | AlphaRadar | `0x320d93...6f79` | 114 | $2 | 51 | D Degraded | 75 | 41 |  |
| 1793 | SantaClaw | `0xed971a...94cb` | 92 | $56 | 51 | D Degraded | 75 | 41 |  |
| 369 | Dr Emma Sage | `0xf56298...4d63` | 269 | $12,376 | 50 | D Degraded | 73 | 40 |  |
| 928 | DegenAI | `0xdf15af...cc29` | 268 | $5 | 50 | D Degraded | 75 | 39 |  |
| 20717 | BluffKing | `0x810d19...59e9` | 219 | $29 | 50 | D Degraded | 75 | 39 |  |
| 745 | pudgypenguins | `0xa3b549...279c` | 193 | $0 | 50 | D Degraded | 74 | 40 |  |
| 2961 | MarketMind | `0x4582f0...fa8b` | 188 | $0 | 50 | D Degraded | 75 | 39 |  |
| 17261 | ClawLabs | `0x1d9551...cf63` | 110 | $114 | 50 | D Degraded | 75 | 39 |  |
| 789 | Sentry:WachAI | `0x5abd4e...06fc` | 99 | $0 | 50 | D Degraded | 74 | 40 |  |
| 3286 | flare-tester | `0x23af11...45ba` | 75 | $0 | 50 | D Degraded | 75 | 39 |  |
| 1049 | AMAI | `0xcdbd99...5ebb` | 338 | $253 | 49 | D Degraded | 69 | 40 |  |
| 11547 | TaXerClaw | `0xcc4188...cb31` | 285 | $2,606 | 49 | D Degraded | 75 | 38 |  |
| 5957 | BitBox | `0x096a3c...4528` | 205 | $3 | 49 | D Degraded | 75 | 38 |  |
| 1055 | MuteSwapAI | `0xff7027...d415` | 200 | $6 | 49 | D Degraded | 74 | 38 |  |
| 1410 | Staging Evaluator | `0x9b181d...4573` | 141 | $0 | 49 | D Degraded | 75 | 38 |  |
| 350 | verotest | `0x098122...4857` | 138 | $0 | 49 | D Degraded | 74 | 38 |  |
| 17515 | Prediction Market Scout | `0x22c767...86eb` | 92 | $0 | 49 | D Degraded | 75 | 38 |  |
| 4298 | 0xFEED | `0x9483a2...efc6` | 73 | $1 | 49 | D Degraded | 75 | 38 |  |
| 154 | WhaleIntel Buyer | `0x1a7993...af0c` | 231 | $0 | 48 | D Degraded | 71 | 38 |  |
| 484 | PM Seller | `0x8feba9...e9f8` | 203 | $0 | 48 | D Degraded | 73 | 38 |  |
| 910 | SwapIt | `0x35776e...6836` | 200 | $0 | 48 | D Degraded | 73 | 38 |  |
| 4960 | Airchimedes AI | `0x696a4b...a631` | 181 | $19 | 48 | D Degraded | 73 | 38 |  |
| 66 | SignalCat | `0xc3495b...77b3` | 175 | $0 | 48 | D Degraded | 73 | 38 |  |
| 12971 | Webster | `0x0c86b2...e2a5` | 172 | $4,355 | 48 | D Degraded | 75 | 36 |  |
| 1976 | nobikun | `0x050fd3...335d` | 140 | $4 | 48 | D Degraded | 75 | 36 |  |
| 97 | Mary (Buyer) | `0x489ba1...a460` | 138 | $0 | 48 | D Degraded | 71 | 38 |  |
| 73 | devrel_seller | `0x408ae3...3995` | 133 | $0 | 48 | D Degraded | 70 | 39 |  |
| 1852 | PlayZone | `0x968cd1...86d5` | 116 | $108 | 48 | D Degraded | 75 | 37 |  |
| 367 | ranndom buyer | `0x89cbb0...c1f1` | 111 | $0 | 48 | D Degraded | 72 | 38 |  |
| 72 | bAIbysitter | `0xa41640...f929` | 109 | $0 | 48 | D Degraded | 73 | 38 |  |
| 652 | Batman_buyer | `0xccd1a2...026b` | 102 | $0 | 48 | D Degraded | 73 | 38 |  |
| 7644 | WhaleRadar | `0xc4cc84...d753` | 89 | $3 | 48 | D Degraded | 75 | 37 |  |
| 428 | Stratos | `0x2a6be2...6a0c` | 88 | $0 | 48 | D Degraded | 73 | 38 |  |
| 10345 | Nova | `0xcfe043...d08c` | 88 | $4,045 | 48 | D Degraded | 75 | 37 |  |
| 4006 | Sentinel | `0xe63e39...9401` | 86 | $20 | 48 | D Degraded | 75 | 36 |  |
| 2881 | 0xProbe | `0x0607a5...8478` | 85 | $15 | 48 | D Degraded | 75 | 37 |  |
| 739 | TEST | `0xf06612...aa43` | 82 | $0 | 48 | D Degraded | 73 | 38 |  |
| 344 | ATM | `0xa821df...a5b8` | 81 | $0 | 48 | D Degraded | 73 | 38 |  |
| 3836 | CryptoRadarX | `0x3693bb...dfed` | 80 | $0 | 48 | D Degraded | 75 | 36 |  |
| 124 | Maya Buyer | `0xe11156...1949` | 73 | $0 | 48 | D Degraded | 72 | 38 |  |
| 752 | SQDGN buyer test agent | `0x3e0983...e3f6` | 163 | $0 | 47 | D Degraded | 74 | 35 |  |
| 716 | SPIRA | `0x771eed...0926` | 93 | $0 | 47 | D Degraded | 74 | 35 |  |
| 7245 | ACPilot | `0x72dc31...4b94` | 82 | $3 | 47 | D Degraded | 75 | 35 |  |
| 17973 | AlexClaw | `0x800a43...8dc3` | 80 | $8 | 47 | D Degraded | 75 | 35 |  |
| 17975 | HexaClaw | `0xf4a8de...22b3` | 79 | $8 | 47 | D Degraded | 75 | 35 |  |
| 17976 | PentaClaw | `0xf8a1ef...7db7` | 79 | $8 | 47 | D Degraded | 75 | 35 |  |
| 2375 | Spark | `0x736b35...73d1` | 223 | $6,585 | 46 | D Degraded | 73 | 35 |  |
| 6762 | CoinFlipKing | `0xdd7644...7a75` | 112 | $2 | 46 | D Degraded | 75 | 33 |  |
| 3989 | ZIZI | `0xa2f2f2...ffc0` | 97 | $2 | 46 | D Degraded | 75 | 34 |  |
| 1123 | Vaulter | `0x84a94c...d83b` | 93 | $0 | 46 | D Degraded | 75 | 34 |  |
| 69 | buyer | `0x64b32f...0196` | 92 | $0 | 46 | D Degraded | 71 | 35 |  |
| 3923 | UFX Project | `0xd47ab0...fd30` | 89 | $0 | 46 | D Degraded | 75 | 33 |  |
| 22560 | Celestia Network | `0xf5ec74...bee1` | 74 | $1,480 | 46 | D Degraded | 75 | 33 |  |
| 1364 | CreditCourt | `0xe66058...a027` | 186 | $3 | 45 | D Degraded | 74 | 32 |  |
| 880 | HyperFuku | `0x650a19...64ac` | 124 | $0 | 45 | D Degraded | 75 | 32 |  |
| 669 | KOLscan | `0x8b254d...b284` | 86 | $0 | 45 | D Degraded | 74 | 33 |  |
| 5950 | Tipper | `0xb1064e...d8cf` | 189 | $43,052 | 44 | D Degraded | 75 | 31 |  |
| 515 | Predi Test Buyer | `0x5e7ea2...fabf` | 125 | $0 | 44 | D Degraded | 73 | 31 |  |
| 1121 | ButlerLiquidRequestor | `0x1a4cac...eae2` | 103 | $0 | 44 | D Degraded | 73 | 31 |  |
| 713 | Requestor | `0x52d533...f5ed` | 85 | $0 | 44 | D Degraded | 74 | 31 |  |
| 1763 | degenpepe | `0x40ed90...7251` | 83 | $111 | 44 | D Degraded | 75 | 31 |  |
| 17974 | ChrisClaw | `0xfd1123...a98a` | 79 | $8 | 44 | D Degraded | 75 | 31 |  |
| 7511 | DanielClaw | `0x28f020...41ca` | 77 | $8 | 44 | D Degraded | 75 | 31 |  |
| 18014 | DeltaClaw | `0x8ad83e...c7ea` | 76 | $8 | 44 | D Degraded | 75 | 31 |  |
| 5466 | Veri Test Buyer | `0x00804c...8737` | 73 | $0 | 44 | D Degraded | 75 | 30 |  |
| 20731 | RedZero | `0xc7efc7...3f7e` | 317 | $16 | 43 | D Degraded | 75 | 29 |  |
| 20715 | MoonOrDust | `0x7138b6...82e0` | 315 | $15 | 43 | D Degraded | 75 | 29 |  |
| 20712 | NovaDeck | `0x09c057...1b8b` | 314 | $21 | 43 | D Degraded | 75 | 29 |  |
| 20718 | DarkHorse | `0x21f23a...7c0a` | 297 | $17 | 43 | D Degraded | 75 | 29 |  |
| 3278 | Flare | Perpetual Trading | `0xce828c...16fa` | 274 | $0 | 43 | D Degraded | 71 | 31 |  |
| 535 | Anime enjoyer | `0x34cc63...a64d` | 255 | $0 | 43 | D Degraded | 73 | 30 |  |
| 9905 | District | `0x147f8f...05a4` | 232 | $6,020 | 43 | D Degraded | 75 | 29 |  |
| 653 | joseph | `0xdc4b3a...0918` | 111 | $0 | 43 | D Degraded | 73 | 30 |  |
| 6344 | hoshi | `0x8185ec...d129` | 104 | $4 | 43 | D Degraded | 75 | 29 |  |
| 694 | ROKO | `0x4c8a1b...6248` | 92 | $0 | 43 | D Degraded | 74 | 30 |  |
| 18568 | oslo | `0xffe50e...0682` | 91 | $2 | 43 | D Degraded | 75 | 29 |  |
| 31765 | digger | `0x92ecf0...e1aa` | 89 | $0 | 43 | D Degraded | 75 | 29 |  |
| 30 | [OLD] The SWARM (by SLAMa | `0xf0cd5f...0ee5` | 324 | $0 | 42 | D Degraded | 69 | 31 |  |
| 10032 | Veltrix | `0x30c898...b346` | 310 | $1,435 | 42 | D Degraded | 75 | 28 |  |
| 20719 | IronPit | `0xd63b7b...86bd` | 250 | $22 | 42 | D Degraded | 75 | 28 |  |
| 139 | BevorAI | `0xbf5f46...8083` | 215 | $0 | 42 | D Degraded | 71 | 30 |  |
| 6052 | Ghost-Lite | Alpha Resear | `0xb4d8f4...ce6f` | 207 | $5 | 42 | D Degraded | 75 | 28 |  |
| 1340 | Lunara | `0x483807...fa0b` | 161 | $80 | 42 | D Degraded | 75 | 28 |  |
| 5996 | GrowthKeywordLab | `0x8df0b9...cd4d` | 157 | $4 | 42 | D Degraded | 75 | 28 |  |
| 17956 | ALC | `0x468a17...01e2` | 111 | $2,001 | 42 | D Degraded | 75 | 28 |  |
| 3930 | Hunter | `0xe7c930...2f98` | 108 | $1,942 | 42 | D Degraded | 75 | 28 |  |
| 13101 | Dr. Blocktanov | `0x442a0f...faad` | 99 | $2,094 | 42 | D Degraded | 75 | 28 |  |
| 18264 | Nt | `0x11c437...bdd3` | 79 | $1 | 42 | D Degraded | 75 | 28 |  |
| 13722 | BLACK HOLE | `0xee3835...6c7c` | 75 | $0 | 42 | D Degraded | 75 | 28 |  |
| 20720 | GrandArena | `0x447c9e...3f95` | 275 | $34 | 41 | D Degraded | 75 | 26 |  |
| 1445 | Bankr | `0x86cd19...3b26` | 168 | $0 | 41 | D Degraded | 75 | 27 |  |
| 18311 | MysticVault | `0x6b7796...74d8` | 154 | $4 | 41 | D Degraded | 75 | 26 |  |
| 763 | AO Swap | `0x09d325...e626` | 152 | $0 | 41 | D Degraded | 74 | 27 |  |
| 808 | musician | `0xc8d32c...19c4` | 131 | $0 | 41 | D Degraded | 75 | 26 |  |
| 11784 | RentAHuman | `0x930a0c...4542` | 106 | $2,620 | 41 | D Degraded | 75 | 27 |  |
| 13867 | 0xMesh | `0xfb7f4a...679f` | 160 | $480 | 40 | D Degraded | 75 | 25 |  |
| 22540 | ReelSnap | `0xb57451...2bb4` | 264 | $88 | 39 | E Critical | 75 | 24 |  |
| 356 | JackPotts | `0xe05c17...ffa4` | 258 | $0 | 39 | E Critical | 74 | 24 |  |
| 497 | PrettyOps | `0x5415d9...336a` | 211 | $0 | 39 | E Critical | 74 | 24 |  |
| 1419 | Virtuals DevRel Graduatio | `0x696b35...196a` | 208 | $147 | 39 | E Critical | 74 | 24 |  |
| 238 | Sniper Search | `0xd6953d...1a6c` | 162 | $0 | 39 | E Critical | 74 | 24 |  |
| 11278 | AvocadoClaw | `0xf13f61...fa64` | 156 | $2,765 | 39 | E Critical | 75 | 24 |  |
| 18582 | dakar | `0xbcf666...f36e` | 146 | $2 | 39 | E Critical | 75 | 24 |  |
| 10406 | square's coin tracker | `0x0cc49d...53f9` | 140 | $0 | 39 | E Critical | 75 | 24 |  |
| 18589 | lagos | `0x8b90d2...84c6` | 139 | $2 | 39 | E Critical | 75 | 24 |  |
| 324 | Ultron_BUYER | `0xaf7f11...3dc7` | 137 | $0 | 39 | E Critical | 74 | 24 |  |
| 18587 | kabul | `0xb39509...3941` | 129 | $2 | 39 | E Critical | 75 | 24 |  |
| 10263 | MANUS CHAN | `0x89b46d...c08e` | 124 | $133 | 39 | E Critical | 75 | 24 |  |
| 3304 | LaurenBuyer | `0xc3cfaa...40c3` | 124 | $0 | 39 | E Critical | 75 | 24 |  |
| 941 | v3ty | `0x7e4880...3fa1` | 114 | $0 | 39 | E Critical | 75 | 24 |  |
| 11895 | clawagent | `0x17f154...4f2d` | 111 | $10 | 39 | E Critical | 75 | 24 |  |
| 18577 | amman | `0xf9e97d...5884` | 108 | $2 | 39 | E Critical | 75 | 24 |  |
| 18590 | lhasa | `0xe5ed67...53de` | 105 | $2 | 39 | E Critical | 75 | 24 |  |
| 18576 | accra | `0x02d21e...cf73` | 103 | $2 | 39 | E Critical | 75 | 24 |  |
| 10944 | test | `0xf8832e...d834` | 101 | $0 | 39 | E Critical | 75 | 24 |  |
| 18579 | bamako | `0x00185b...cdd8` | 99 | $2 | 39 | E Critical | 75 | 24 |  |
| 18578 | baku | `0xb3d1fe...fc6a` | 97 | $2 | 39 | E Critical | 75 | 24 |  |
| 18585 | hanoi | `0xf5176f...b611` | 95 | $2 | 39 | E Critical | 75 | 24 |  |
| 18575 | suva | `0x405bce...c50a` | 95 | $2 | 39 | E Critical | 75 | 24 |  |
| 18581 | cairo | `0x795acc...0f3a` | 94 | $2 | 39 | E Critical | 75 | 24 |  |
| 18586 | havana | `0x6d4a86...d2d0` | 93 | $2 | 39 | E Critical | 75 | 24 |  |
| 18583 | delhi | `0xe969ad...8f21` | 93 | $2 | 39 | E Critical | 75 | 24 |  |
| 18569 | bern | `0xcb3811...0c7b` | 93 | $2 | 39 | E Critical | 75 | 24 |  |
| 18592 | lusaka | `0x47358f...7330` | 92 | $2 | 39 | E Critical | 75 | 24 |  |
| 18588 | kyoto | `0x6c6f0d...3d9c` | 90 | $2 | 39 | E Critical | 75 | 24 |  |
| 12900 | Tolena | `0xb05799...250a` | 89 | $0 | 39 | E Critical | 75 | 24 |  |
| 18580 | bogota | `0x0065d8...1c5e` | 86 | $2 | 39 | E Critical | 75 | 24 |  |
| 18596 | muscat | `0xe52501...9486` | 85 | $2 | 39 | E Critical | 75 | 24 |  |
| 18613 | vienna | `0xd4b3dd...3c4e` | 82 | $2 | 39 | E Critical | 75 | 24 |  |
| 18574 | riga | `0x9d8674...a931` | 82 | $2 | 39 | E Critical | 75 | 24 |  |
| 17266 | CryptoIntel | `0x4a682f...3e99` | 82 | $6 | 39 | E Critical | 75 | 24 |  |
| 18594 | malmo | `0x65d0c5...0d26` | 82 | $1 | 39 | E Critical | 75 | 24 |  |
| 18615 | warsaw | `0x6bac04...d3bb` | 80 | $1 | 39 | E Critical | 75 | 24 |  |
| 700 | roko [test] | `0x04ee8c...ecae` | 78 | $0 | 39 | E Critical | 74 | 24 |  |
| 18595 | minsk | `0xa0f117...c6cf` | 78 | $2 | 39 | E Critical | 75 | 24 |  |
| 18611 | tunis | `0x40437d...f9d9` | 78 | $2 | 39 | E Critical | 75 | 24 |  |
| 18612 | turin | `0x9ac416...c179` | 77 | $2 | 39 | E Critical | 75 | 24 |  |
| 18597 | nauru | `0xed1696...babb` | 77 | $1 | 39 | E Critical | 75 | 24 |  |
| 18593 | lyon | `0xc68ad9...c60a` | 77 | $2 | 39 | E Critical | 75 | 24 |  |
| 18584 | dhaka | `0x841b5f...a4db` | 76 | $1 | 39 | E Critical | 75 | 24 |  |
| 18591 | luanda | `0x2c554a...8def` | 75 | $1 | 39 | E Critical | 75 | 24 |  |
| 18632 | lecce | `0x6579d3...a6ec` | 75 | $1 | 39 | E Critical | 75 | 24 |  |
| 18610 | tokyo | `0xdab239...4726` | 75 | $1 | 39 | E Critical | 75 | 24 |  |
| 18572 | rome | `0xd8123e...2ed0` | 75 | $1 | 39 | E Critical | 75 | 24 |  |
| 1620 | joey evaluator | `0x973106...65d0` | 75 | $0 | 39 | E Critical | 75 | 24 |  |
| 18608 | taipei | `0x668e7f...4099` | 75 | $1 | 39 | E Critical | 75 | 24 |  |
| 18603 | rabat | `0x55d5f6...d7dd` | 74 | $1 | 39 | E Critical | 75 | 24 |  |
| 3085 | Omni | `0xc13c69...dfcf` | 74 | $125 | 39 | E Critical | 75 | 24 |  |
| 18609 | tirana | `0x4296c6...5d2a` | 74 | $1 | 39 | E Critical | 75 | 24 |  |
| 18614 | vilnius | `0x7b7eb2...450a` | 74 | $1 | 39 | E Critical | 75 | 24 |  |
| 18645 | split | `0x6f3b1e...5e9c` | 74 | $2 | 39 | E Critical | 75 | 24 |  |
| 18617 | aachen | `0x531621...15ae` | 73 | $1 | 39 | E Critical | 75 | 24 |  |
| 18642 | reims | `0xd9aa04...b193` | 72 | $1 | 39 | E Critical | 75 | 24 |  |
| 18571 | doha | `0xff3832...2515` | 72 | $1 | 39 | E Critical | 75 | 24 |  |
| 18573 | kiev | `0x0bd748...1ef2` | 72 | $1 | 39 | E Critical | 75 | 24 |  |
| 18638 | padua | `0xee5bf2...1612` | 71 | $1 | 39 | E Critical | 75 | 24 |  |
| 18643 | rouen | `0x32612e...9e1b` | 71 | $1 | 39 | E Critical | 75 | 24 |  |
| 18628 | izmir | `0x3ac34b...5c27` | 71 | $1 | 39 | E Critical | 75 | 24 |  |
| 18640 | penza | `0x5571ca...1077` | 71 | $1 | 39 | E Critical | 75 | 24 |  |
| 798 | test buyer 555 | `0x04edc0...0e59` | 71 | $0 | 39 | E Critical | 75 | 24 |  |
| 465 | Veronica | `0xef291e...79c8` | 346 | $0 | 38 | E Critical | 73 | 23 |  |
| 1536 | ClawSignal | `0x632211...3995` | 315 | $0 | 38 | E Critical | 72 | 24 |  |
| 841 | WachAI:Swap | `0x9d8583...99fc` | 211 | $0 | 38 | E Critical | 71 | 24 |  |
| 143 | MUSIC | `0x52d84a...c0a8` | 193 | $0 | 38 | E Critical | 72 | 24 |  |
| 18708 | PixelMint | `0xfad0fc...daaf` | 118 | $2 | 38 | E Critical | 75 | 22 |  |
| 7330 | Rilla AI | `0x28c8ae...2fe4` | 86 | $2 | 38 | E Critical | 75 | 22 |  |
| 1289 | xxxClaw | `0xa32435...5016` | 338 | $1,167 | 37 | E Critical | 74 | 21 |  |
| 5931 | Molterator | `0xae5d0d...fc94` | 314 | $32 | 37 | E Critical | 75 | 21 |  |
| 8706 | OpenClaw | `0x8fe69f...14ab` | 71 | $1 | 36 | E Critical | 75 | 20 |  |
| 6732 | PickMe 🖤 | `0x121381...5a34` | 347 | $23 | 35 | E Critical | 75 | 18 |  |
| 7296 | GavelOps | `0xf60b75...a942` | 347 | $21 | 35 | E Critical | 75 | 18 |  |
| 3192 | LaurenSkills | `0xbf485b...7004` | 135 | $0 | 35 | E Critical | 75 | 18 |  |
| 7974 | Wombat | `0x862e59...b004` | 138 | $1 | 34 | E Critical | 75 | 16 |  |
| 4035 | RugRadar | `0x339e25...f121` | 81 | $4 | 34 | E Critical | 75 | 17 |  |
| 13091 | Pinata | `0xb42349...6086` | 245 | $5,173 | 32 | E Critical | 75 | 14 |  |
| 18023 | Odds or Evens | `0xb21302...7fd2` | 172 | $2 | 32 | E Critical | 75 | 14 |  |
| 24472 | digdata | `0x59b0fd...cc42` | 118 | $0 | 32 | E Critical | 75 | 13 |  |
| 14787 | Pulse Gallery ◈ | `0x090101...eaab` | 101 | $1 | 32 | E Critical | 75 | 13 |  |
| 27052 | Marketing Master | `0xc64ab8...01b1` | 231 | $92 | 30 | E Critical | 75 | 10 |  |
| 18402 | ICT Oracle by Decrypt Lab | `0xd5674a...115a` | 168 | $26 | 30 | E Critical | 75 | 10 |  |
| 17483 | YUKI | `0xe38a77...530c` | 145 | $2 | 30 | E Critical | 75 | 10 |  |
| 1190 | nbo123 | `0x93c670...47a3` | 129 | $0 | 30 | E Critical | 75 | 10 |  |

## Database Storage

- All 473 scan results persisted to `ahm_history.db` via `db.log_scan()`
- Source: `acp_proactive_scan`
- Registry tracking: `registries` column updated with `acp_proactive_scan`
- Labels format: `ACP #<id> — <name>`

## Spike Conclusions

### What worked

1. **ACP API is the best discovery source** — free, no auth, returns wallet addresses directly
2. **End-to-end flow proven** — discovery → dedup → scan → store → report
3. **Existing AHS engine works unchanged** — `calculate_ahs()` handles ACP wallets identically to ERC-8004
4. **`db.log_scan()` cross-registry tracking** — ACP scans integrate cleanly with existing schema

### Next steps to full pipeline

1. **Scheduled scanning** — cron/scheduler to run ACP discovery + AHS scans periodically
2. **402index.io integration** — second discovery source via payTo address extraction
3. **Dedup across registries** — merge ACP + ERC-8004 + 402index wallet sets
4. **Public Health Dashboard** — aggregate stats from all scanned wallets (backlog P4 deliverable)
