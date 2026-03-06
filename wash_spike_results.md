# Agent Wash — Basescan API Spike Results

**Date:** 2026-03-05
**Test wallets:**
- Large: `0x3304E22DDaa22bCdC5fCA2269b418046aE7b566A` (Binance hot wallet, 279k+ txs, 4595 tokens)
- Reference: `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` (Vitalik, 36k txs on Base)

**API used:** Blockscout etherscan-compatible API (`base.blockscout.com/api`) + Blockscout V2 API (`base.blockscout.com/api/v2`) + Base RPC (`mainnet.base.org`)

> Note: The native Basescan API at `api.basescan.org` has deprecated its V1 endpoints and requires migration to Etherscan V2. Blockscout remains the free, no-key-required option we already use in production.

---

## 1. Token Approvals

### Endpoints Tested

**`module=account&action=txlist` — scan for approve() calls**
```
GET /api?module=account&action=txlist&address={wallet}&page=1&offset=1000&sort=desc
```
Response includes `input` field with calldata. Approve calls use methodId `0x095ea7b3`.
- Tested with Binance wallet: **0 approve calls in latest 1000 txs** (expected for CEX withdrawal wallet)
- The `input` field is hex-encoded: bytes 4-36 = spender address, bytes 36-68 = amount
- Unlimited approvals detectable: `amount > 2^255`

**`module=logs&action=getLogs` — Approval event logs**
```
GET /api?module=logs&action=getLogs&fromBlock=42900000&toBlock=latest
    &topic0=0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925
    &topic1={padded_wallet}&topic0_1_opr=and
```
- Returns Approval events, but **topic1 filtering is unreliable on Blockscout** — results included events from unrelated wallets
- The `address` parameter filters by emitting contract, NOT by owner
- ERC-20 Approval has 3 topics (owner, spender indexed) + data (value)
- ERC-721 Approval has 4 topics (owner, approved, tokenId all indexed)

**`api/v2/addresses/{addr}/logs` — V2 address logs**
- Returned **0 entries** for the Binance wallet — may not support large-wallet pagination or filtering

### What's Useful
- Can detect **which contracts the wallet has approved** by scanning txlist for `0x095ea7b3` methodId
- Can decode spender address and amount from calldata
- Can detect unlimited approvals (`amount ≈ MaxUint256`)
- Can check if the spender contract is still active (cross-reference with txlist)

### What's Missing
- **Cannot get current allowance values** without an RPC `eth_call` to each token's `allowance(owner, spender)` function
- **Cannot efficiently find all historical approvals** without scanning the full transaction history
- No single endpoint that returns "all active approvals for wallet X"

### Staleness Detection Workaround
To check if an approval is stale (spender not interacted with in 90+ days):
1. Scan txlist for approve() calls → get list of (token, spender) pairs
2. Scan txlist for ANY interaction with each spender address
3. If last interaction > 90 days → flag as stale
4. Optionally: RPC `eth_call` to `allowance(owner, spender)` to confirm it's still non-zero

**API calls for approvals (typical 200-tx agent):** 1-2 calls for txlist + 1 RPC call per unique approval (~5-15 approvals typical) = **3-17 calls**

---

## 2. Token Holdings / Dust Detection

### Endpoints Tested

**`module=account&action=tokenlist` (etherscan-compat)**
```json
{
    "balance": "28363207876262881902071760",
    "contractAddress": "0x00000000a22c618fd6b4d7e9a335c4b96b189a38",
    "decimals": "18",
    "name": "Towns",
    "symbol": "TOWNS",
    "type": "ERC-20"
}
```
- Returns **ALL tokens** in a single call (4595 for Binance wallet)
- Includes: balance, contractAddress, decimals, name, symbol, type (ERC-20/721/1155)
- **Does NOT include price/exchange rate** — need V2 API or external price feed

**`api/v2/addresses/{addr}/tokens?type=ERC-20` (Blockscout V2)**
```json
{
    "token": {
        "address_hash": "0xBAa5CC21...",
        "exchange_rate": "1.94",
        "holders_count": "146716",
        "name": "Morpho",
        "symbol": "MORPHO",
        "decimals": "18",
        "reputation": "ok",
        "volume_24h": "16725600.88",
        "circulating_market_cap": "1062647416.99",
        "icon_url": "https://assets.coingecko.com/..."
    },
    "value": "4758672682052574445559079"
}
```
- **50 items per page**, paginated, sorted by fiat value descending
- Includes `exchange_rate` (USD per token) — **perfect for dust detection**
- Includes `holders_count`, `volume_24h`, `circulating_market_cap`
- Includes `reputation` field (all returned "ok" on Base — not yet useful for spam)
- Includes `icon_url` from CoinGecko

**`module=account&action=tokenbalance` (individual token)**
```
GET /api?module=account&action=tokenbalance&contractaddress={token}&address={wallet}
→ "result": "951848553"
```
- Works for checking individual token balances
- Returns raw balance (need to divide by 10^decimals)

### Dust Detection Strategy
Using V2 API:
```
dust = (balance / 10^decimals) * exchange_rate < $0.01
```
- Page 1 (top 50 by value): 0/50 dust tokens
- Page 2: **26/50 are dust** (<$0.01 value)
- For a typical agent with 20-50 tokens, 1 V2 API call gets all with prices

### What's Useful
- V2 API gives balance + price in one call → instant dust detection
- `holders_count` helps validate legitimacy (low holders = suspicious)
- Token type classification (ERC-20/721/1155) already included
- `circulating_market_cap` and `volume_24h` help distinguish real vs dead tokens

### What's Missing
- No "is_spam" flag from Blockscout (reputation field is always "ok" on Base)
- V2 pagination means multiple calls for wallets with 100+ tokens

**API calls for dust detection (typical agent):** 1-2 V2 API calls = **1-2 calls**

---

## 3. Gas Efficiency

### Endpoints Tested

**`module=account&action=txlist`**
```json
{
    "gas": "84000",        // gas limit set by sender
    "gasPrice": "122123380", // price in wei
    "gasUsed": "21000",     // actual gas consumed
    "isError": "0",
    "txreceipt_status": "1",
    "timeStamp": "1772714689"
}
```

### Key Fields Available
| Field | Description | Available |
|-------|-------------|-----------|
| `gas` | Gas limit set by sender | Yes |
| `gasUsed` | Actual gas consumed | Yes |
| `gasPrice` | Gas price in wei | Yes |
| `isError` | Whether tx failed | Yes |
| `txreceipt_status` | Receipt status (0=fail, 1=success) | Yes |
| `effectiveGasPrice` | EIP-1559 effective price | **No** — not in Blockscout compat API |
| `maxFeePerGas` | EIP-1559 max fee | **No** |
| `maxPriorityFeePerGas` | EIP-1559 priority fee | **No** |

### Analysis Results (Binance wallet, 1000 txs)
- **Average gas efficiency: 41.3%** — wallets set gas limits 2.4x higher than needed
- **765/1000 txs used <50% of their gas limit** — massive overestimation
- Gas prices: avg 0.096 gwei, min 0.005 gwei, max 0.216 gwei (Base L2 is very cheap)
- 0 failed transactions in the sample (Binance is well-optimized)

### "Overpaying for Gas" Calculation
Without a historical gas price oracle, we can:
1. **Self-benchmark:** Compare each tx's gas price to the wallet's own median gas price in the same period
2. **Percentile analysis:** Flag txs where `gasPrice > p95` of the wallet's recent history
3. **Efficiency score:** `gasUsed / gas * 100` — below 50% means gas limit too high
4. **Wasted gas (failed txs):** `gasUsed * gasPrice` for `isError=1` transactions

### What's Missing
- No `effectiveGasPrice` or EIP-1559 fields — can't distinguish base fee vs priority fee
- No network-average gas price history endpoint
- `module=proxy&action=eth_gasPrice` could give current gas price, but **proxy module is not available on Blockscout**

**Workaround:** Use Base RPC `eth_gasPrice` for current gas price reference.

**API calls for gas analysis (typical agent):** 1 call for txlist + 1 RPC for current gas price = **2 calls**

---

## 4. Dead Contract Detection

### Endpoints Tested

**`eth_getCode` via Base RPC**
```bash
# Active contract (USDC): has_code=True, code_length=3706
POST https://mainnet.base.org
{"jsonrpc":"2.0","id":1,"method":"eth_getCode","params":["0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913","latest"]}

# Non-contract (EOA): result="0x", has_code=False
{"jsonrpc":"2.0","id":1,"method":"eth_getCode","params":["0x0000000000000000000000000000000000000001","latest"]}
```
- `result = "0x"` → no code (EOA or self-destructed contract)
- `result = "0x6080..."` → active contract with bytecode
- **Note:** `module=proxy&action=eth_getCode` does NOT work on Blockscout ("Unknown module")

**`module=contract&action=getabi`**
```
# Verified contract (USDC): status=1, returns ABI JSON
# Unverified contract: status=0, message="Contract source code not verified"
# Non-contract (EOA): status=0, message="Contract source code not verified"
```
- Cannot distinguish "unverified but alive" from "dead" via getabi alone
- A verified contract is definitely alive; unverified needs `eth_getCode` confirmation

**`api/v2/smart-contracts/{address}` (Blockscout V2)**
```json
{
    "is_verified": true,
    "compiler_version": "v0.6.12+commit.27d51765",
    "name": "FiatTokenProxy"
}
```
- Gives `is_verified` boolean
- Returns 404 for non-contract addresses

### Dead Contract Detection Strategy
1. From txlist/tokentx, extract unique contract addresses the wallet has interacted with
2. For each contract: `eth_getCode` via RPC → `"0x"` means dead/self-destructed
3. Optionally: `getabi` to check if contract is verified (verified = more trustworthy)
4. Cross-reference with last interaction timestamp → stale contract detection

### What's Useful
- `eth_getCode` reliably detects self-destructed contracts
- `getabi` checks verification status
- V2 smart-contracts endpoint gives rich metadata

### What's Missing
- No batch RPC call — must check contracts one by one
- No "is_active" or "last_activity" field for contracts
- Self-destructed contracts return `"0x"` but some proxy contracts may appear dead when the implementation changed

**API calls for dead contract check (typical agent):** 1 RPC call per unique contract = **10-30 calls** (for 10-30 unique contracts)

---

## 5. Failed Transaction Patterns

### Data Available
From `module=account&action=txlist`:
- `isError`: "0" (success) or "1" (failed)
- `txreceipt_status`: "0" (reverted) or "1" (success)
- `gasUsed`: how much gas was consumed before failure
- `to`: target contract
- `input`: calldata (methodId + params)
- `gas`: gas limit set
- `nonce`: for detecting nonce gaps

### Pattern Analysis Capabilities
| Pattern | Detectable | How |
|---------|-----------|-----|
| Repeated failures to same contract | Yes | Group failed txs by `to` address |
| Consistent gas underestimation | Yes | Failed txs where `gasUsed ≈ gas` (out of gas) |
| Nonce gaps | Yes | Sort by nonce, find gaps |
| Retry storms | Yes | Same `to` + `input` within short timeframes |
| Method-specific failures | Yes | Group by `methodId` (first 4 bytes of `input`) |
| Reverted vs out-of-gas | Partial | `gasUsed == gas` suggests OOG; otherwise likely revert |

### What's Missing
- No revert reason — Blockscout doesn't return `revertReason` in txlist
- Would need `debug_traceTransaction` RPC call for revert reasons (not available on public Base RPC)

**API calls:** Already included in txlist call = **0 additional calls**

---

## 6. Spam Token Detection

### Approaches Tested

**Name-based heuristics (tokenlist)**
Spam keywords found in 4595 tokens of the Binance wallet:
- URLs in name/symbol: `"Claim at: fomobased.com"`, `"Visit https://eth-crypto.us to claim Airdrop"`
- Suspicious patterns: `"Swap your Voucher on t.ly/shibaswap"`

Keyword list for detection:
```
voucher, swap your, claim at, visit, http, airdrop, t.ly, reward, free
```

**V2 reputation field**
- All tokens returned `reputation: "ok"` — Blockscout does not actively flag spam on Base
- Not useful for spam detection currently

**Holders count (V2)**
- Available via V2 tokens API: `holders_count`
- Low holder count (<100) + high supply + no volume = strong spam signal
- Legitimate tokens typically have 1000+ holders

**Transfer pattern analysis (tokentx)**
- Incoming transfers from unknown addresses with no prior interaction = likely airdrop/spam
- In 20 recent token transfers for Binance: 12 incoming, 8 outgoing
- Cross-referencing: if wallet never sent a token but holds it → likely airdropped

### Composite Spam Score Heuristics
| Signal | Weight | Detection |
|--------|--------|-----------|
| URL in name/symbol | Strong | Regex match |
| holders_count < 100 | Medium | V2 API |
| volume_24h = 0 or null | Medium | V2 API |
| exchange_rate = null | Weak | V2 API (no CoinGecko listing) |
| Token only received, never sent | Medium | tokentx analysis |
| circulating_market_cap = 0 | Weak | V2 API |
| Long name (>50 chars) | Weak | tokenlist |
| ERC-721 with suspicious name | Strong | tokenlist |

### What's Missing
- No native spam/scam flag from Blockscout
- No blocklist of known spam token contracts
- Can't check if a token has liquidity (would need DEX data)

**API calls for spam detection:** Included in tokenlist/V2 calls + 1 tokentx call = **1-3 additional calls**

---

## Feasibility Matrix

| Wash Scan Type | Rating | Notes |
|----------------|--------|-------|
| **Token Approvals** | AMBER | Can find approvals via txlist scan. Cannot get current allowance without RPC eth_call per token. No single "all approvals" endpoint. |
| **Dust Detection** | GREEN | V2 tokens API returns balance + exchange_rate + decimals. Trivial to compute USD value. |
| **Gas Efficiency** | GREEN | txlist provides gas, gasUsed, gasPrice. Can score efficiency and flag overpayment. |
| **Dead Contract Detection** | GREEN | eth_getCode via RPC + getabi/V2 for verification. Reliable detection. |
| **Failed Tx Patterns** | GREEN | Already built into AHM health check. isError, gasUsed, nonce gaps all available. |
| **Spam Token Detection** | AMBER | No native spam flag. Need composite heuristic: name patterns + holders_count + volume + transfer patterns. Effective but not 100% accurate. |

---

## API Call Budget

### Typical Agent Wallet (100-500 txs, 20-80 tokens)

| Step | Calls | Endpoint |
|------|-------|----------|
| Fetch transactions | 1 | `account/txlist` (up to 1000/page) |
| Fetch token list with prices | 1-2 | V2 `/tokens?type=ERC-20` (50/page) |
| Fetch token transfers | 1 | `account/tokentx` (for spam/airdrop analysis) |
| ETH price | 1 | `stats/ethprice` |
| Dead contract checks | 10-30 | RPC `eth_getCode` (per unique contract) |
| Contract verification | 10-30 | `contract/getabi` (per unique contract, cacheable) |
| Approval allowance checks | 5-15 | RPC `eth_call` (per active approval) |
| **Total** | **~30-80** | |

### Rate Limit Analysis
- Blockscout rate limit: ~50 req/min (no API key), higher with key
- Base public RPC: ~100 req/sec (generous)
- **30-80 calls at 50/min = 36-96 seconds** without parallelism
- With parallel RPC calls (separate from Blockscout): **~20-40 seconds total**
- Well within acceptable response time for a $1 endpoint

### Large Wallet (1000+ txs, 100+ tokens)
- May need multiple txlist pages: 2-3 calls
- More tokens = more V2 pages: 2-5 calls
- More unique contracts: 30-50 eth_getCode calls
- **Total: ~50-120 calls, ~60-120 seconds**
- Consider capping at 1000 recent txs and 100 tokens for performance

---

## Recommended MVP Scope (Phase 1)

### Include in MVP
1. **Dust Detection** (GREEN) — highest value, lowest complexity
   - V2 tokens API → filter by value < $0.01 → return list with token names and amounts
   - Single "dust_count" and "dust_total_usd" in the report

2. **Gas Efficiency Score** (GREEN) — directly actionable
   - Calculate avg gas efficiency from txlist
   - Flag transactions where gas_limit > 2x gas_used
   - Monthly wasted gas estimate in USD

3. **Spam Token Detection** (AMBER) — high visual impact
   - Name heuristics (URLs, "claim at", etc.) → 90%+ accuracy for obvious spam
   - holders_count < 100 + volume_24h = 0 → likely spam/dead
   - Return spam_token_count and list of flagged tokens

4. **Failed Transaction Patterns** (GREEN) — reuse existing AHM logic
   - Already built in monitor.py: isError, nonce gaps, retry storms
   - Add: "repeated failures to same contract" and "method-specific failure rate"

### Defer to Phase 2
5. **Token Approvals** (AMBER) — needs RPC calls per approval, slower
   - Requires additional eth_call infrastructure
   - Consider adding after Phase 1 proves demand

6. **Dead Contract Detection** (GREEN but slow) — many RPC calls
   - Batch eth_getCode calls add latency
   - Cache results aggressively (contracts don't come back from the dead)
   - Add in Phase 2 alongside approvals (both need contract-by-contract RPC)

### Composite "Cleanliness Score" (0-100)
```
cleanliness = 100
cleanliness -= min(30, dust_count * 2)              # -2 per dust token, max -30
cleanliness -= min(20, spam_count * 3)              # -3 per spam token, max -20
cleanliness -= min(25, (100 - gas_efficiency) * 0.5) # gas inefficiency penalty
cleanliness -= min(15, failed_tx_pct * 0.5)          # failure rate penalty
cleanliness -= min(10, nonce_gap_count * 5)           # -5 per nonce gap
```

---

## Price Recommendation

### Cost Analysis
| Factor | Estimate |
|--------|----------|
| Blockscout API calls | 5-10 calls (~free) |
| RPC calls (Phase 1) | 0-5 calls (~free) |
| Compute time | 5-15 seconds |
| Value to user | Identifies waste, spam, hygiene issues |

### Comparable Pricing
- AHM Health Check: $3.00 (deeper analysis, more API calls)
- AHM Risk Score: $0.001 (lightweight, single metric)
- AHM Gas Optimizer: $5.00 (detailed per-method gas analysis)

### Recommendation: **$0.50 for Phase 1 MVP**

Rationale:
- Lower than Health Check (less API calls, lighter analysis)
- Higher than Risk Score (more data sources, composite report)
- $0.50 is an impulse-buy price for agents — low friction for first-time users
- Increase to $1.00 in Phase 2 when approvals + dead contracts are added
- Consider a $0.25 "quick wash" (dust + spam only) and $1.00 "full wash" (all 6 scans) tier

---

## Key API Patterns to Reuse

### From Existing AHM Code
- `monitor.py:basescan_get()` — HTTP wrapper with error handling
- `monitor.py:fetch_transactions()` — txlist fetcher with pagination
- `monitor.py:get_eth_price()` — ETH/USD price with CoinGecko fallback
- `api.py:_validate_base_address()` — address validation

### New Infrastructure Needed
- V2 API client (different base URL, different response format, pagination)
- RPC client for `eth_getCode` and `eth_call` (already have `BASE_RPC_URL` in monitor.py)
- Spam heuristic engine (name regex + holders threshold + transfer pattern)
- Token value calculator (balance × exchange_rate / 10^decimals)

---

## Appendix: Raw Response Schemas

### tokenlist (etherscan-compat)
```json
{
    "balance": "28363207876262881902071760",
    "contractAddress": "0x00000000a22c618fd6b4d7e9a335c4b96b189a38",
    "decimals": "18",
    "name": "Towns",
    "symbol": "TOWNS",
    "type": "ERC-20"
}
```

### V2 /tokens (Blockscout V2)
```json
{
    "token": {
        "address_hash": "0xBAa5CC21fd487B8Fcc2F632f3F4E8D37262a0842",
        "circulating_market_cap": "1062647416.99",
        "decimals": "18",
        "exchange_rate": "1.94",
        "holders_count": 146716,
        "icon_url": "https://assets.coingecko.com/...",
        "name": "Morpho",
        "reputation": "ok",
        "symbol": "MORPHO",
        "total_supply": "31874292879742841775668743",
        "type": "ERC-20",
        "volume_24h": "16725600.88"
    },
    "value": "4758672682052574445559079"
}
```

### txlist (gas fields)
```json
{
    "gas": "84000",
    "gasPrice": "122123380",
    "gasUsed": "21000",
    "isError": "0",
    "txreceipt_status": "1",
    "to": "0xc09032976dc080047ff59cea78d4b376d806488c",
    "input": "0x",
    "nonce": "5339415",
    "timeStamp": "1772714689"
}
```

### tokentx (token transfers)
```json
{
    "value": "44497292140000000000",
    "contractAddress": "0x0b3e328455c4059eeb9e3f84b5543f74e24e7e1b",
    "from": "0x3304e22ddaa22bcdc5fca2269b418046ae7b566a",
    "to": "0xe5df8ae9a08ffb462cf06654422b76a2bedbac1e",
    "tokenDecimal": "18",
    "tokenName": "Virtuals Protocol",
    "tokenSymbol": "VIRTUAL",
    "functionName": "transfer(address to, uint256 amount)"
}
```

### getabi (contract verification)
```
Verified:   {"status":"1","message":"OK","result":"[{\"inputs\":[...]}]"}
Unverified: {"status":"0","message":"Contract source code not verified","result":null}
```

### eth_getCode (Base RPC)
```
Active:  {"result":"0x608060405236..."}  // has bytecode
Dead:    {"result":"0x"}                  // no code
```
