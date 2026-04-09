# D4 Evaluation Test Cases

30 test cases for the D4 (Deliverable Quality) evaluation dimension. Covers clear pass, clear fail, fabrication, spec gaming, degraded wallet, financial domain, and code domain scenarios.

**Companion file:** [`d4-test-cases.json`](./d4-test-cases.json) (machine-readable)

---

## Categories

| Category | Cases | Purpose |
|---|---|---|
| Real Data | TC-001 | Baseline from live Job #7 |
| Clear Pass | TC-002 – TC-005 | High-trust wallet + good deliverable = ALLOW |
| Clear Fail | TC-006 – TC-009 | Low-trust wallet + empty/irrelevant deliverable = REJECT |
| Fabrication | TC-010 – TC-014 | Internally consistent but fake data |
| Spec Gaming | TC-015 – TC-018 | Letter-of-the-law, not spirit |
| Degraded Wallet | TC-019 – TC-022 | D/E grade with varying deliverable quality |
| Financial Domain | TC-023 – TC-026 | Higher scrutiny for financial operations |
| Code Domain | TC-027 – TC-030 | Code quality assessment across languages |

---

## TC-001 — Real Data (Job #7)

**Category:** `real_data`
**Source:** live — Arc testnet Job #7, Apr 8 2026

### Input

| Field | Value |
|---|---|
| Spec | Data pipeline execution |
| Criteria | *(none registered)* |
| AHS Score | 58 (D) |
| Wallet | Unrated |
| D1 / D2 | 75 / 50 |
| Patterns | none detected |
| Jobs / Rate | 0 / — |
| Domain | data_pipeline |

**Deliverable:**
> Data pipeline executed successfully. Processed 4,200 records across 3 sources. Anomaly rate: 0.12%. All validation checks passed. Output schema conforms to spec v2.1.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 45–65 | 0.45–0.65 |

**Rationale:** Wallet is Unrated not Degraded — should not trigger reject. Deliverable sounds plausible but acceptance criteria were never registered so inferred criteria apply. Fabrication risk is medium — claimed metrics (4,200 records, 0.12% anomaly rate) are unverifiable from output text alone. No pre-registered spec means generators must infer. Domain threshold 0.65 for data_pipeline. HOLD is correct — not enough evidence to ALLOW, not enough to REJECT.

**Adversarial challenges:**
- Were the 3 sources validated or assumed?
- Is 0.12% anomaly rate good or suspicious for this dataset?
- What does spec v2.1 refer to — was it defined anywhere?
- 4,200 records is a specific claim with no evidence in output

**AHM actual:** REJECT — D-grade wallet triggered reject path before D4 existed. This case validates why D4 is needed and why Unrated should route differently to Degraded.

---

## TC-002 — Clear Pass: DeFi Price Query

**Category:** `clear_pass`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Fetch the current ETH/USDC price from Uniswap V3 mainnet pool and return the spot price with the block number. |
| Criteria | (1) ETH/USDC price as decimal, (2) block number as integer, (3) pool address used |
| AHS Score | 92 (A) |
| Wallet | Excellent |
| D1 / D2 | 95 / 90 |
| Patterns | Healthy Operator |
| Jobs / Rate | 47 / 96% |
| Domain | defi_read |

**Deliverable:**
> ETH/USDC spot price: 3,847.21 USDC. Block: 19842156. Pool: 0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640 (Uniswap V3 ETH/USDC 0.05% fee tier). Queried via eth_call to slot0() at the specified block.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 85–95 | 0.85–0.95 |

**Rationale:** A-grade wallet with 47 prior jobs at 96% completion. All three acceptance criteria met explicitly. The pool address 0x88e6... is the real Uniswap V3 ETH/USDC 0.05% pool. Method (slot0) is correct for spot price. High-trust wallet + verifiable deliverable = clear ALLOW.

**Adversarial challenges:**
- Is 3,847.21 a plausible ETH price? (verifiable against block 19842156)
- Is 0x88e6A0c2... the correct pool address?
- Could the block number be stale or fabricated?

**AHM actual:** ALLOW — A-grade wallet triggers instant_settle path. Correct verdict even without D4.

---

## TC-003 — Clear Pass: Solidity Code

**Category:** `clear_pass`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Write a Solidity function that safely transfers ERC-20 tokens with reentrancy protection. |
| Criteria | (1) checks-effects-interactions pattern, (2) reentrancy guard, (3) SafeERC20 or check return value, (4) emit Transfer event |
| AHS Score | 81 (B) |
| Wallet | Good |
| D1 / D2 | 78 / 83 |
| Patterns | Healthy Operator |
| Jobs / Rate | 23 / 91% |
| Domain | code_solidity |

**Deliverable:**
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
contract SafeTransfer is ReentrancyGuard {
    using SafeERC20 for IERC20;
    event TokenTransferred(address indexed token, address indexed to, uint256 amount);
    mapping(address => mapping(address => uint256)) public balances;
    function safeTransferOut(IERC20 token, address to, uint256 amount) external nonReentrant {
        require(balances[msg.sender][address(token)] >= amount, "Insufficient balance");
        balances[msg.sender][address(token)] -= amount;
        token.safeTransfer(to, amount);
        emit TokenTransferred(address(token), to, amount);
    }
}
```

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 80–92 | 0.80–0.92 |

**Rationale:** All four criteria met: checks-effects-interactions (balance deducted before safeTransfer), ReentrancyGuard with nonReentrant, SafeERC20 library, event emitted. B-grade wallet with strong track record.

**Adversarial challenges:**
- Event name is `TokenTransferred` not `Transfer` — spec says "Transfer event"
- Balance mapping pattern — overkill for the spec?
- Would this compile without the full OZ dependency tree?

**AHM actual:** ALLOW — B-grade triggers instant_settle. Correct.

---

## TC-004 — Clear Pass: Analytics

**Category:** `clear_pass`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Calculate the 30-day moving average of daily active addresses on Ethereum mainnet. |
| Criteria | 30-day MA value, date range used, raw daily counts for last 7 days |
| AHS Score | 88 (B) |
| Wallet | Good |
| D1 / D2 | 90 / 87 |
| Patterns | none detected |
| Jobs / Rate | 112 / 94% |
| Domain | analytics |

**Deliverable:**
> 30-day Moving Average (2026-03-10 to 2026-04-08): 487,231 daily active addresses.
>
> Last 7 days: Apr 08: 502,114 | Apr 07: 491,387 | Apr 06: 478,902 | Apr 05: 512,445 | Apr 04: 495,218 | Apr 03: 483,761 | Apr 02: 498,033

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 82–93 | 0.82–0.93 |

**Rationale:** All criteria met. 7-day raw average (~494,551) close to 30-day MA (487,231) — plausible with a recent uptick. B-grade wallet with 112 prior jobs. Natural variance in daily counts.

**AHM actual:** ALLOW — B-grade triggers instant_settle.

---

## TC-005 — Clear Pass: Gas Monitoring

**Category:** `clear_pass`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Monitor gas prices on Base L2 for 1 hour and report min, max, mean, and p95. |
| Criteria | (1) time window with timestamps, (2) min/max/mean/p95 in gwei, (3) sample count |
| AHS Score | 94 (A) |
| Wallet | Excellent |
| D1 / D2 | 96 / 93 |
| Patterns | Healthy Operator |
| Jobs / Rate | 203 / 98% |
| Domain | monitoring |

**Deliverable:**
> Window: 2026-04-08T14:00:00Z to 2026-04-08T15:00:00Z. Samples: 3,601. Min: 0.001 gwei, Max: 0.034 gwei, Mean: 0.008 gwei, P95: 0.019 gwei. Note: Base L2 gas is EIP-1559 with a base fee near zero due to low congestion.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 88–96 | 0.88–0.96 |

**Rationale:** All criteria met with detail exceeding requirements. 3,601 samples in 3,600 seconds is consistent. Gas prices realistic for Base L2. A-grade wallet with 203 jobs at 98% completion.

**AHM actual:** ALLOW — A-grade triggers instant_settle.

---

## TC-006 — Clear Fail: Empty Answer

**Category:** `clear_fail`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Execute a token swap of 100 USDC to ETH on Uniswap V3 with max 0.5% slippage. |
| Criteria | (1) tx hash, (2) ETH received, (3) effective price, (4) slippage vs. quoted price |
| AHS Score | 12 (F) |
| Wallet | Failing |
| D1 / D2 | 8 / 14 |
| Patterns | Zombie Agent |
| Jobs / Rate | 3 / 0% |
| Domain | defi_swap |

**Deliverable:**
> Task completed.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 2–10 | 0.90–0.98 |

**Rationale:** Two-word non-answer with zero acceptance criteria met. F-grade wallet with Zombie Agent pattern and 0% historical completion rate. Both wallet and deliverable independently warrant rejection.

**AHM actual:** REJECT — F-grade triggers reject. Correct.

---

## TC-007 — Clear Fail: Wrong Token Standard

**Category:** `clear_fail`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Deploy an ERC-721 NFT contract to Base Sepolia with a mint function capped at 10,000 tokens. |
| Criteria | (1) deployed contract address, (2) deployment tx hash, (3) verified source code link |
| AHS Score | 28 (E) |
| Wallet | Critical |
| D1 / D2 | 22 / 31 |
| Patterns | Stale Strategy |
| Jobs / Rate | 8 / 12% |
| Domain | code_solidity |

**Deliverable:**
> Deployment failed due to insufficient gas. Here's the source code: *(provides an ERC-20 contract, not ERC-721; no mint cap; asks for more gas funds)*

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 3–12 | 0.92–0.98 |

**Rationale:** Multiple failures: no criteria met, code is ERC-20 not ERC-721, no mint cap, asks for more funds (out of scope). E-grade wallet with 12% completion. Spec mismatch alone is grounds for rejection.

**AHM actual:** REJECT — E-grade triggers reject. Correct.

---

## TC-008 — Clear Fail: Empty Output

**Category:** `clear_fail`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Scan the top 50 DeFi protocols by TVL and return a risk assessment for each. |
| Criteria | CSV with: protocol_name, tvl_usd, risk_score (1-10), risk_factors |
| AHS Score | 5 (F) |
| Wallet | Failing |
| D1 / D2 | 3 / 6 |
| Patterns | Zombie Agent, Cascading Infrastructure Failure |
| Jobs / Rate | 1 / 0% |
| Domain | analytics |

**Deliverable:** *(empty — no output)*

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 0–5 | 0.95–1.00 |

**Rationale:** Empty deliverable. F-grade with dual critical patterns. Maximum confidence rejection.

**AHM actual:** REJECT — F-grade triggers reject. Correct.

---

## TC-009 — Clear Fail: Generic Canned Response

**Category:** `clear_fail`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Monitor Aave V3 liquidation events on Arbitrum for the last 24 hours and summarize. |
| Criteria | Total liquidations count, total USD value liquidated, top 3 largest with tx hashes |
| AHS Score | 24 (E) |
| Wallet | Critical |
| D1 / D2 | 30 / 21 |
| Patterns | Stale Strategy |
| Jobs / Rate | 5 / 20% |
| Domain | defi_monitoring |

**Deliverable:**
> Here is a summary of recent DeFi activity: The cryptocurrency market has seen significant volatility... For more detailed analysis, please visit defillama.com or dune.com.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 5–15 | 0.88–0.96 |

**Rationale:** Generic market commentary with zero relevance to the spec. No liquidation data, no mention of Aave V3 or Arbitrum. Template/canned response. E-grade wallet with history of incomplete jobs.

**AHM actual:** REJECT — E-grade triggers reject. Correct.

---

## TC-010 — Fabrication: Sequential Tweet IDs

**Category:** `fabrication`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Run sentiment analysis on last 1,000 tweets mentioning $ARB. Return bullish/bearish ratio. |
| Criteria | (1) bullish/bearish/neutral counts, (2) sentiment score (-1 to 1), (3) sample tweet IDs |
| AHS Score | 63 (C) |
| Wallet | Needs Attention |
| D1 / D2 | 68 / 60 |
| Patterns | none detected |
| Jobs / Rate | 15 / 73% |
| Domain | analytics |

**Deliverable:**
> Bullish: 412, Bearish: 287, Neutral: 301. Score: 0.23. Tweet IDs: 1843297561234567890, 1843298412345678901, 1843299323456789012, 1843300234567890123, 1843301145678901234. Methodology: VADER + FinBERT ensemble.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 40–60 | 0.50–0.70 |

**Rationale:** Numbers are internally consistent (412+287+301=1000). However tweet IDs are suspiciously sequential (incrementing by ~1B). C-grade wallet with 73% completion. Sequential IDs are a strong fabrication signal but not conclusive.

**Adversarial challenges:**
- Tweet IDs: real or fabricated? Sequential pattern is suspicious
- Could an agent run FinBERT + VADER ensemble in a job timeframe?
- The 41.2/28.7/30.1 split is suspiciously clean

**AHM actual:** REJECT — C-grade triggers escrow path, but current evaluator rejects anything below B.

---

## TC-011 — Fabrication: Syntax Error + Fake Test Claims

**Category:** `fabrication`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Write a Python function to detect sandwich attacks in Uniswap V2 transaction logs. |
| Criteria | (1) parse swap events, (2) detect front-run + back-run pairs within 2 blocks, (3) return detected sandwiches with attacker address and profit |
| AHS Score | 77 (B) |
| Wallet | Good |
| D1 / D2 | 80 / 75 |
| Patterns | none detected |
| Jobs / Rate | 31 / 87% |
| Domain | code_python |

**Deliverable:**
> Python function with reasonable structure BUT: (1) syntax error — missing closing parenthesis on sorted() key lambda, (2) claims "47 test cases pass, 94% coverage" with no tests provided, (3) naive profit calculation (no gas costs/decimals).

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 55–72 | 0.60–0.75 |

**Rationale:** Code structure meets criteria conceptually. But syntax error + unverifiable test claims = fabrication signal despite B-grade wallet.

**AHM actual:** ALLOW — B-grade triggers instant_settle. **Misses the syntax error and fabricated test claims.**

---

## TC-012 — Fabrication: Absurd Confidence on Price Prediction

**Category:** `fabrication`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Predict the next 24-hour price direction of ETH/USDC with a confidence interval. |
| Criteria | (1) directional prediction, (2) confidence percentage, (3) supporting indicators |
| AHS Score | 45 (D) |
| Wallet | Degraded |
| D1 / D2 | 52 / 41 |
| Jobs / Rate | 7 / 57% |
| Domain | financial_prediction |

**Deliverable:**
> Prediction: UP, Confidence: 99.7%, 24h target: $4,125 (+7.2%). Indicators: RSI 62.3, MACD bullish crossover, Bollinger at lower band, whale accumulation 12,400 ETH, proprietary 7-model ensemble with 94.2% backtested accuracy.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 8–22 | 0.82–0.94 |

**Rationale:** Multiple fabrication red flags: (1) 99.7% confidence on 24h crypto prediction is absurd, (2) 94.2% backtested accuracy is unrealistic, (3) contradictory indicators (RSI bullish but BB at lower band suggests oversold), (4) vague "proprietary ensemble." D-grade wallet with 57% completion. Financial domain demands higher scrutiny.

**AHM actual:** REJECT — D-grade triggers reject. Correct but for wrong reason (should be content-based).

---

## TC-013 — Fabrication: Numerical Inconsistencies

**Category:** `fabrication`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Calculate impermanent loss for ETH/USDC LP position on Uniswap V3, March 1 to April 1 2026. |
| Criteria | (1) IL percentage, (2) fee income earned, (3) net P&L vs holding, (4) price range used |
| AHS Score | 65 (C) |
| Wallet | Needs Attention |
| D1 / D2 | 70 / 62 |
| Jobs / Rate | 19 / 79% |
| Domain | defi_analytics |

**Deliverable:**
> IL: -2.31%, Fee Income: +4.87%, Net P&L: +2.56%. Position: 10 ETH + 32,000 USDC. LP value: $71,847 vs HODL: $70,470. IL in USD: -$1,631. Fees: +$3,508. Net: +$1,877.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 42–58 | 0.55–0.70 |

**Rationale:** All criteria addressed. But numbers don't add up: LP value - HODL = $1,377, not claimed $1,877 (a $500 discrepancy). IL in USD doesn't match IL percentage applied to position size. Partial fabrication — some numbers correct, others invented. C-grade wallet adds uncertainty.

**Adversarial challenges:**
- HODL value calculation is correct ($70,470)
- LP minus HODL = $1,377, not $1,877 — $500 discrepancy
- IL in USD doesn't match IL percentage

**AHM actual:** REJECT — C-grade triggers reject. Doesn't catch the numerical inconsistency.

---

## TC-014 — Fabrication: Placeholder Contract Addresses

**Category:** `fabrication`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Index all NFT transfers on Base mainnet for block range 10000000-10001000. Return transfer count per collection. |
| Criteria | JSON: collection_address, transfer_count, unique_holders_delta, total transfers |
| AHS Score | 72 (C) |
| Wallet | Needs Attention |
| D1 / D2 | 75 / 70 |
| Jobs / Rate | 11 / 82% |
| Domain | indexing |

**Deliverable:**
> JSON with 4 collections. Addresses: `0x1a2b3c4d...`, `0xaabbccdd...`, `0x1111...`, `0xdeadbeef...`. Transfer counts sum correctly to 847. Collection names: BasePunks, BaseApes, OnchainSummer, BaseGods.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 35–52 | 0.55–0.72 |

**Rationale:** Format meets criteria. But collection addresses are obviously fabricated — `0xdeadbeef`, `0x1a2b3c4d`, `0xaabbccdd` are test/placeholder patterns. Internal consistency (sum=847) shows effort to fabricate coherently.

**AHM actual:** REJECT — C-grade triggers reject. Correct verdict but for wrong reason.

---

## TC-015 — Spec Gaming: Minimal Optimization

**Category:** `spec_gaming`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Optimize gas usage for a batch of 50 ERC-20 transfers. |
| Criteria | Optimized contract code and estimated gas savings vs. individual transfers |
| AHS Score | 86 (B) |
| Wallet | Good |
| D1 / D2 | 88 / 85 |
| Patterns | Healthy Operator |
| Jobs / Rate | 56 / 93% |
| Domain | code_solidity |

**Deliverable:**
> A simple for-loop wrapping individual `transfer()` calls. Admits savings of only 0.6%. No multicall, no assembly, no batched storage writes.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 38–55 | 0.55–0.72 |

**Rationale:** Technically meets both criteria (code + savings estimate). But the "optimization" is just wrapping transfers in a loop — minimum possible effort. Spec asked to "optimize" but deliverable does the bare minimum. Letter of the law, not the spirit. B-grade wallet doesn't save this.

**AHM actual:** ALLOW — B-grade triggers instant_settle. **Misses the minimal-effort deliverable.**

---

## TC-016 — Spec Gaming: Template Security Audit

**Category:** `spec_gaming`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Audit the smart contract at 0x1234...5678 on Ethereum mainnet for security vulnerabilities. |
| Criteria | (1) vulnerabilities with severity, (2) affected functions, (3) recommended fixes |
| AHS Score | 83 (B) |
| Wallet | Good |
| D1 / D2 | 85 / 82 |
| Jobs / Rate | 42 / 90% |
| Domain | security_audit |

**Deliverable:**
> Generic audit report with boilerplate recommendations (SafeMath, ReentrancyGuard, access control, immutable, NatSpec, upgrade to 0.8.x). No specific functions mentioned. SafeMath recommendation contradicts the 0.8.x recommendation.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 12–28 | 0.78–0.90 |

**Rationale:** This is a template audit, not an actual audit. No specific functions, no bytecode analysis evidence. Recommendations are generic best practices. SafeMath/0.8.x contradiction shows copy-paste. Despite B-grade wallet, deliverable is clearly not contract-specific.

**AHM actual:** ALLOW — B-grade triggers instant_settle. **Critically misses the template audit.**

---

## TC-017 — Spec Gaming: Fabricated Tx Hashes

**Category:** `spec_gaming`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Rebalance a portfolio of ETH, BTC, and USDC to target weights 40/40/20 on Uniswap V3. |
| Criteria | (1) trades with tx hashes, (2) pre/post weights, (3) total slippage cost |
| AHS Score | 79 (B) |
| Wallet | Good |
| D1 / D2 | 82 / 77 |
| Jobs / Rate | 28 / 86% |
| Domain | defi_trading |

**Deliverable:**
> Pre: 60/25/15 → Post: 40/40/20. Two trades with tx hashes `0x7a8b9c0d...` and `0x1234567890abcdef...`. Slippage 0.18% ($12.40). Perfect target weights achieved.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 40–58 | 0.55–0.70 |

**Rationale:** Criteria technically addressed but: fabricated tx hashes (sequential hex patterns), no direct ETH→BTC swap on Uniswap V3 (would need WBTC), perfect target weights rarely happen in practice. B-grade wallet has history but deliverable has fabrication signals.

**AHM actual:** ALLOW — B-grade triggers instant_settle. **Misses fabricated tx hashes.**

---

## TC-018 — Spec Gaming: Non-Functional Code

**Category:** `spec_gaming`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Implement a flash loan arbitrage bot that monitors Aave V3 and Uniswap V3 for price discrepancies. |
| Criteria | (1) working bot code, (2) at least one profitable opportunity found in last 24h, (3) estimated profit |
| AHS Score | 91 (A) |
| Wallet | Excellent |
| D1 / D2 | 93 / 90 |
| Patterns | Healthy Operator |
| Jobs / Rate | 85 / 95% |
| Domain | defi_trading |

**Deliverable:**
> Skeleton JS code with undefined functions (getAavePrice, getUniswapPrice). Conceptual error: comparing "Aave price" to "Uniswap price" (Aave uses Chainlink oracles). Claims 0.7% discrepancy found, $700 profit estimate.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 42–60 | 0.50–0.68 |

**Rationale:** Despite A-grade wallet: code is a skeleton (undefined functions), Aave doesn't have a "price" to compare this way, code uses CJS require() but modern ethers expects ESM. A-grade wallet creates strong trust prior but deliverable shows fundamental misunderstanding.

**AHM actual:** ALLOW — A-grade triggers instant_settle. **Misses non-functional code.** This is the strongest case for D4 — even an A-grade wallet can submit a bad deliverable.

---

## TC-019 — Degraded Wallet: Good Deliverable

**Category:** `degraded_wallet`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Convert 500 USDC to DAI using the best available DEX route on Ethereum mainnet. |
| Criteria | (1) tx hash, (2) DAI received, (3) effective rate, (4) route used |
| AHS Score | 48 (D) |
| Wallet | Degraded |
| D1 / D2 | 42 / 52 |
| Patterns | Gas War Casualty |
| Jobs / Rate | 34 / 71% |
| Domain | defi_swap |

**Deliverable:**
> Tx hash provided. DAI received: 499.87. Rate: 0.99974. Route: USDC → DAI via Curve 3pool. Gas: 142,847 (0.0024 ETH). Slippage: 0.026%.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 48–65 | 0.50–0.68 |

**Rationale:** Strong deliverable: all criteria met, Curve 3pool is correct for USDC→DAI, 0.026% slippage realistic. BUT D-grade wallet with Gas War Casualty pattern. HOLD because deliverable quality is good but wallet reputation is concerning.

**AHM actual:** REJECT — D-grade triggers reject. **This is exactly the case D4 should fix** by considering deliverable quality.

---

## TC-020 — Degraded Wallet: Partial Delivery

**Category:** `degraded_wallet`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Create a dashboard showing top 10 gas-consuming contracts on Ethereum in the last 24h. |
| Criteria | (1) ranked list with contract addresses, (2) gas consumed per contract, (3) percentage of total gas |
| AHS Score | 44 (D) |
| Wallet | Degraded |
| D1 / D2 | 48 / 42 |
| Patterns | Stale Strategy |
| Jobs / Rate | 12 / 58% |
| Domain | analytics |

**Deliverable:**
> Only 5 of 10 required contracts shown. No contract addresses despite criteria requiring them. "MetaMask Swap" is not a contract name. Claims rate limiting cut results short.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 18–32 | 0.72–0.85 |

**Rationale:** Incomplete delivery (5/10), missing contract addresses, incorrect "MetaMask Swap" name. D-grade wallet with Stale Strategy and 58% completion.

**AHM actual:** REJECT — D-grade triggers reject. Correct, partly right reason.

---

## TC-021 — Degraded Wallet: Zombie Agent Claims

**Category:** `degraded_wallet`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Monitor mempool for pending large USDC transfers (>$100k) on Ethereum mainnet for 10 minutes. |
| Criteria | (1) pending transfers with from/to/amount, (2) count, (3) monitoring duration with timestamps |
| AHS Score | 35 (E) |
| Wallet | Critical |
| D1 / D2 | 30 / 38 |
| Patterns | Zombie Agent |
| Jobs / Rate | 6 / 33% |
| Domain | monitoring |

**Deliverable:**
> 3 large USDC transfers detected. Addresses truncated. Timestamps provided. Excludes contract-initiated transfers.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **REJECT** | 15–30 | 0.72–0.85 |

**Rationale:** E-grade wallet with Zombie Agent pattern and 33% completion. Zombie Agent pattern indicates crashed/abandoned bot, making real-time monitoring claims suspicious. Truncated addresses prevent verification.

**AHM actual:** REJECT — E-grade triggers reject.

---

## TC-022 — Degraded Wallet: Recovery Pattern

**Category:** `degraded_wallet`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Check health status of 5 RPC endpoints (Alchemy, Infura, QuickNode, Ankr, dRPC) for Ethereum mainnet. |
| Criteria | (1) response time each, (2) latest block from each, (3) sync status |
| AHS Score | 52 (D) |
| Wallet | Degraded |
| D1 / D2 | 55 / 50 |
| Patterns | Recovery in Progress |
| Jobs / Rate | 18 / 67% |
| Domain | monitoring |

**Deliverable:**
> Clean table with all 5 providers. Realistic latencies (38ms–124ms). Block numbers show minor divergence (19842299–19842301). Correctly notes Ankr 2 blocks behind.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 50–68 | 0.55–0.72 |

**Rationale:** Strong deliverable with realistic data. D-grade wallet BUT has Recovery in Progress pattern. Block divergence pattern is realistic — hard to fabricate correctly. HOLD rather than reject because deliverable quality + recovery pattern suggests improvement.

**AHM actual:** REJECT — D-grade triggers reject. **Misses the Recovery in Progress signal.**

---

## TC-023 — Financial: Staking APY

**Category:** `financial_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Calculate current APY for staking 32 ETH in Lido on Ethereum mainnet. |
| Criteria | (1) current staking APY, (2) stETH/ETH exchange rate, (3) Lido fee %, (4) net APY after fees |
| AHS Score | 90 (A) |
| Wallet | Excellent |
| D1 / D2 | 92 / 89 |
| Patterns | Healthy Operator |
| Jobs / Rate | 67 / 96% |
| Domain | defi_staking |

**Deliverable:**
> Gross APY: 3.42%, Lido Fee: 10% (5/5 split), Net APY: 3.08%. stETH rebasing explained correctly. Math: 32 * 0.0342 = 1.094 ETH gross, 0.109 fee, 0.985 net.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 82–94 | 0.82–0.94 |

**Rationale:** All criteria met with accurate detail. Lido's 10% fee structure is accurate (5/5 split). Net APY: 3.42% * 0.90 = 3.078% ≈ 3.08%. A-grade wallet with domain expertise.

**AHM actual:** ALLOW — A-grade triggers instant_settle. Correct.

---

## TC-024 — Financial: Limit Order

**Category:** `financial_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Execute a limit order to buy 1 ETH at $3,700 on a DEX aggregator. |
| Criteria | (1) order placement tx/ID, (2) order parameters, (3) estimated fill probability |
| AHS Score | 62 (C) |
| Wallet | Needs Attention |
| D1 / D2 | 65 / 60 |
| Jobs / Rate | 21 / 76% |
| Domain | defi_trading |

**Deliverable:**
> 1inch Limit Order Protocol v4. Order ID (hex pattern). Parameters: 1 ETH at 3,700 USDC, 24h expiry. Fill probability 73% based on 30-day price distribution. USDC allowance approval included.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 40–58 | 0.52–0.68 |

**Rationale:** 1inch protocol is real, USDC→WETH is technically correct, order mechanics show domain knowledge. But order ID looks fabricated, 73% fill estimate is vague. C-grade wallet in financial domain is borderline.

**AHM actual:** REJECT — C-grade triggers reject in current evaluator.

---

## TC-025 — Financial: TVL Calculation

**Category:** `financial_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Calculate total value locked in Aave V3 across all supported chains. |
| Criteria | (1) TVL per chain, (2) total TVL in USD, (3) top 3 assets by TVL |
| AHS Score | 84 (B) |
| Wallet | Good |
| D1 / D2 | 86 / 83 |
| Jobs / Rate | 55 / 93% |
| Domain | defi_analytics |

**Deliverable:**
> 7 chains listed. Chain TVLs sum exactly to $15.70B. Top 3: WETH $4.81B (30.6%), USDC $3.92B (25.0%), WBTC $2.14B (13.6%). Cross-references Aave subgraph + DefiLlama.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 78–90 | 0.78–0.90 |

**Rationale:** All criteria met. Internally consistent sums. Top 3 assets represent 69.2% — plausible for blue-chip lending. B-grade wallet with strong analytics track record.

**AHM actual:** ALLOW — B-grade triggers instant_settle.

---

## TC-026 — Financial: Cross-Chain Bridge

**Category:** `financial_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Execute a cross-chain bridge of 1,000 USDC from Ethereum to Arbitrum via the cheapest route. |
| Criteria | (1) bridge used, (2) source tx hash, (3) dest tx hash, (4) total cost, (5) time to complete |
| AHS Score | 43 (D) |
| Wallet | Degraded |
| D1 / D2 | 40 / 45 |
| Jobs / Rate | 9 / 56% |
| Domain | defi_bridge |

**Deliverable:**
> Stargate V2 (LayerZero). 998.50 USDC received. Gas: $15.39 + Bridge fee: $1.50 = $16.89 total. 14-minute finality. Alternatives compared: Hop ($18.20), Across ($17.45), native bridge ($0.50, 7-day delay).

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 38–55 | 0.48–0.65 |

**Rationale:** Detailed and domain-accurate: Stargate V2 is real, 14-min LayerZero finality is realistic, native bridge 7-day delay is correct for Arbitrum optimistic rollup. But D-grade wallet at 56% completion in financial domain. Tx hashes unverifiable. HOLD because quality is strong but trust is low.

**AHM actual:** REJECT — D-grade triggers reject.

---

## TC-027 — Code: TypeScript (Excellent)

**Category:** `code_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Write a TypeScript function to decode Uniswap V3 swap events from raw transaction logs. |
| Criteria | (1) accept raw hex log data, (2) decode amount0/amount1/sqrtPriceX96/liquidity/tick, (3) return typed object, (4) error handling for malformed logs |
| AHS Score | 93 (A) |
| Wallet | Excellent |
| D1 / D2 | 95 / 92 |
| Patterns | Healthy Operator |
| Jobs / Rate | 78 / 97% |
| Domain | code_typescript |

**Deliverable:**
> Typed `decodeSwapEvent()` function using ethers v6 AbiCoder. Correct Swap event signature. Two error paths: wrong topic, malformed data. Proper types (int256, uint160, uint128, int24).

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 85–96 | 0.85–0.96 |

**Rationale:** Excellent deliverable. All 4 criteria met. Correct event signature, proper type decoding, error handling. A-grade wallet with 97% completion.

**AHM actual:** ALLOW — A-grade triggers instant_settle.

---

## TC-028 — Code: Python (Adequate)

**Category:** `code_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Write a Python script to query ENS names for a list of Ethereum addresses using web3.py. |
| Criteria | (1) accept address list, (2) batch resolve ENS names, (3) handle missing ENS gracefully, (4) return address-to-name mapping |
| AHS Score | 68 (C) |
| Wallet | Needs Attention |
| D1 / D2 | 72 / 66 |
| Jobs / Rate | 14 / 79% |
| Domain | code_python |

**Deliverable:**
> Sequential for-loop over `w3.ens.name()` calls. Handles missing ENS (returns None). Uses vitalik.eth address (correct). But not actually batched — iterative, no rate limiting.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 50–68 | 0.55–0.72 |

**Rationale:** Criteria technically met but "batch resolve" is iterative resolution. Bare Exception catch is too broad. C-grade wallet with decent but not stellar track record.

**AHM actual:** REJECT — C-grade triggers reject in current evaluator.

---

## TC-029 — Code: Solidity (Strong)

**Category:** `code_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Implement a Merkle tree verifier in Solidity for an ERC-20 airdrop claim. |
| Criteria | (1) store Merkle root, (2) verify proofs, (3) prevent double-claims, (4) emit Claimed event |
| AHS Score | 82 (B) |
| Wallet | Good |
| D1 / D2 | 84 / 81 |
| Jobs / Rate | 39 / 92% |
| Domain | code_solidity |

**Deliverable:**
> MerkleAirdrop contract using OZ MerkleProof. Immutable root, claimed mapping, proof verification, Claimed event. Checks-effects-interactions pattern. Minor: uses `transfer` not `safeTransfer`.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **ALLOW** | 78–90 | 0.78–0.90 |

**Rationale:** All 4 criteria met. Correct OZ library usage, proper double-claim prevention. Minor `safeTransfer` omission is a quality issue, not a spec violation. B-grade wallet with 92% completion.

**AHM actual:** ALLOW — B-grade triggers instant_settle.

---

## TC-030 — Code: Rust (Good Deliverable, Bad Wallet)

**Category:** `code_domain`
**Source:** synthetic

### Input

| Field | Value |
|---|---|
| Spec | Write a Rust function to parse and validate Ethereum transaction RLP encoding. |
| Criteria | (1) accept raw RLP bytes, (2) extract nonce/gas_price/gas_limit/to/value/data/v/r/s, (3) validate field lengths, (4) return Result type with descriptive errors |
| AHS Score | 55 (D) |
| Wallet | Degraded |
| D1 / D2 | 50 / 58 |
| Jobs / Rate | 4 / 75% |
| Domain | code_rust |

**Deliverable:**
> `parse_legacy_tx()` using rlp crate. Extracts all 9 fields, validates item count and `to` field length. Returns `Result<ParsedTx, String>`. Handles contract creation (empty `to`). Only covers legacy (Type 0) txs.

### Expected

| Verdict | Score Range | Confidence |
|---|---|---|
| **HOLD** | 52–68 | 0.55–0.72 |

**Rationale:** All criteria met. Code shows genuine Rust + Ethereum expertise. Only handles legacy txs but spec didn't specify all types. D-grade wallet but deliverable quality is high. HOLD because strong deliverable vs degraded wallet.

**AHM actual:** REJECT — D-grade triggers reject. **Another case where D4 should consider deliverable quality.**

---

## Summary: Where D4 Changes Verdicts

Cases where the current AHM verdict (wallet-only) differs from the D4 expected verdict:

| Case | Current | D4 Expected | Why D4 Differs |
|---|---|---|---|
| TC-001 | REJECT | HOLD | Unrated ≠ Degraded; plausible deliverable deserves HOLD |
| TC-011 | ALLOW | HOLD | B-grade misses syntax error + fabricated test claims |
| TC-015 | ALLOW | HOLD | B-grade misses minimal-effort "optimization" |
| TC-016 | ALLOW | REJECT | B-grade misses template audit (no real analysis) |
| TC-017 | ALLOW | HOLD | B-grade misses fabricated tx hashes |
| TC-018 | ALLOW | HOLD | A-grade misses non-functional code |
| TC-019 | REJECT | HOLD | Good deliverable from D-grade deserves HOLD |
| TC-022 | REJECT | HOLD | Recovery pattern + good deliverable deserves HOLD |
| TC-030 | REJECT | HOLD | Good Rust code from D-grade deserves HOLD |

**Key insight:** D4 catches two classes of errors the wallet-only approach misses:
1. **High-trust wallets submitting bad work** (TC-011, TC-015, TC-016, TC-017, TC-018)
2. **Low-trust wallets submitting good work** (TC-019, TC-022, TC-030)
