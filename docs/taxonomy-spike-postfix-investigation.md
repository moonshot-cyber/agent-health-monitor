# Taxonomy Spike: Post-Fix Investigation (offset=1000)

**Date:** 2026-04-26
**Branch:** `data/taxonomy-postfix-spike`
**Context:** After PR #138 bumped the Blockscout txlist offset from 200 to 1000,
Phase 2 classification was re-run. Three high-volume unclassified contracts
surfaced. This spike investigates each for taxonomy addition.

---

## Contract #1: `0x13fa4b947877c4b67b91aecddcf10d6e24b33ee1` (986 interactions)

### Identification

- **Contract Name:** GnosisSafe (via GnosisSafeProxy)
- **Version:** Safe v1.3.0
- **Compiler:** Solidity >=0.7.0 <0.9.0
- **Verified:** Yes
- **Proxy:** Yes (GnosisSafeProxy → GnosisSafe singleton)

### What It Does

This is a **Gnosis Safe multisig wallet** on Base, operated by a single EOA
(`0xf46bcdae...`). All 986 interactions are `execTransaction` calls from that
one address — textbook AI agent wallet pattern.

### Method Breakdown

| Method | Selector | Count | % |
|--------|----------|-------|---|
| `execTransaction` | `0x6a761202` | ~950 | 96.3% |
| Unknown | `0xf74e481b` | ~32 | 3.2% |
| `transfer` (ERC-20) | `0xa9059cbb` | ~2 | 0.2% |
| `multiSend` | `0x8d80ff0a` | ~1 | 0.1% |
| Other | — | ~3 | 0.2% |

100% inbound. Single caller. Token holdings include STAR (Star by Virtuals),
suggesting a Virtuals Protocol ecosystem agent.

### DefiLlama

No match for this address.

### Proposed Action

**Do NOT add to taxonomy.** Same rationale as the 199-200 cluster in the
previous spike (PR #137): Gnosis Safe is generic wallet infrastructure. The
meaningful classification signal is in what the Safe *calls*, not the Safe
itself. The single-caller pattern confirms this is an agent's wallet, not a
protocol endpoint.

---

## Contract #2: `0xc440e601e22429c8f93a65548a746f015dda26d2` (616 interactions)

### Identification

- **Contract Name:** GnosisSafe (minimal proxy)
- **Implementation:** GnosisSafe v1.3.0 (`0xd9Db270c...`)
- **Compiler:** Solidity 0.7.6
- **Verified:** Yes (implementation verified)
- **Proxy:** Yes (master_copy minimal proxy)

### What It Does

Another Gnosis Safe wallet, but this one is identifiable as an **Olas
(Autonolas) AI Mech agent service**. All 616 external transactions are
`execTransaction` calls from a single operator EOA
(`0xe0a6f3fc...`).

The inner calldata reveals the Safe is calling the **Olas Mech Marketplace**
contract (`0xb55fadf1f0bb1de99c13301397c7b67fde44f6b1`) on Base:

| Inner Method | Selector | Purpose |
|-------------|----------|---------|
| `deliverToMarketplace` | `0x6f6885bb` | Delivering AI compute results to Mech Marketplace |
| `request` | `0xf6938b09` | Requesting AI compute from other Mechs |

The Safe holds 50 OLAS tokens. The dual presence of `deliverToMarketplace`
(provider) and `request` (consumer) indicates a composite agent that both
provides and consumes AI computation services.

### DefiLlama

No match for this Safe address. Olas protocol itself is tracked on DefiLlama.

### Proposed Action

**Do NOT add this Safe address to taxonomy** — it is an agent wallet, not a
protocol endpoint.

**Add the Olas Mech Marketplace contract instead:**
`0xb55fadf1f0bb1de99c13301397c7b67fde44f6b1` as `Intelligence & Analytics
Agents` with label "Olas Mech Marketplace (decentralized AI compute marketplace)".
This is the protocol endpoint that agents interact with to request and deliver
AI computation results. Any agent calling `deliverToMarketplace` or `request`
on this contract is performing intelligence/analytics work.

---

## Contract #3: `0x6cd5ac19a07518a8092eeffda4f1174c72704eeb` (614 interactions)

### Identification

- **Contract Name:** GNSMultiCollatDiamond (via TransparentUpgradeableProxy)
- **Protocol:** Gains Network (gTrade)
- **Compiler:** Solidity 0.8.9 (OpenZeppelin v4.8.3)
- **Verified:** Yes
- **Proxy:** Yes (EIP-1967 Transparent Upgradeable Proxy)
- **Architecture:** EIP-2535 Diamond Standard (multi-facet)
- **Official Docs:** Listed at [docs.gains.trade/contract-addresses/base-mainnet](https://docs.gains.trade/what-is-gains-network/contract-addresses/base-mainnet)

### What It Does

Gains Network's **core leveraged trading diamond** on Base. Supports synthetic
leveraged trading on crypto, forex, stocks, and commodities with multiple
collateral types (USDC, DAI, etc.).

### Method Breakdown (131 sampled transactions)

| Method | Selector | Count | % | Category |
|--------|----------|-------|---|----------|
| `setPairDepthBands` | `0x88be5e4f` | 45 | 34% | Admin/Config |
| `delegatedTradingAction` | `0x737b84cd` | 22 | 17% | Trade Execution |
| `triggerOrderWithSignatures` | `0xc7e2b2a9` | 16 | 12% | Trade Execution |
| `setBorrowingPairParamsArray` | `0xd4b813f2` | 14 | 11% | Admin/Config |
| `setBorrowingPairFeePerBlockCapArray` | `0x92748a7d` | 11 | 8% | Admin/Config |
| `openTrade` | `0x5bfcc4f8` | 10 | 8% | Trade Execution |
| `updateSl` | `0xb5d9e9d0` | 7 | 5% | Trade Management |
| `multicall` | `0xac9650d8` | 2 | 2% | Batching |
| `setTradingDelegate` | `0x604755cf` | 2 | 2% | Access Control |
| `increasePositionSize` | `0x24058ad3` | 1 | 1% | Trade Management |
| `closeTradeMarket` | `0x36ce736b` | 1 | 1% | Trade Execution |

100% inbound. Mix of keeper/bot admin calls (~53%) and user-facing trading
operations (~38%). Dominant caller `0xe72dfec4...` is likely a Gains Network
keeper bot.

### DefiLlama

Gains Network is tracked on DefiLlama. This specific contract address was not
returned by the `/protocols` endpoint but is officially documented by the
protocol.

### Proposed Action

**Add to taxonomy** as `Financial Agents` with label "Gains Network gTrade
(leveraged trading)". High confidence — the contract is the official core
trading diamond, verified source, officially documented address, and all
methods are financial trading operations.

---

## Summary of Recommendations

| Contract | Identity | Action | Category |
|----------|----------|--------|----------|
| `0x13fa4b94...` | Gnosis Safe wallet | NOT added — generic infra | N/A |
| `0xc440e601...` | Gnosis Safe wallet (Olas Mech agent) | NOT added — agent wallet | N/A |
| `0xb55fadf1...` | Olas Mech Marketplace (discovered via #2) | **Add to taxonomy** | Intelligence & Analytics Agents |
| `0x6cd5ac19...` | Gains Network gTrade Diamond | **Add to taxonomy** | Financial Agents |

## Key Takeaway

2 of 3 high-volume unclassified contracts were Gnosis Safe wallets — the same
pattern identified in the PR #137 spike. This reinforces the recommendation
from that spike to implement Safe trace-through logic: instead of classifying
the Safe address, trace through `execTransaction` calls to identify the target
contracts being called (as done manually here for the Olas Mech agent).

The Olas Mech Marketplace discovery is a bonus: it was not one of the three
input contracts but was found by tracing through the Safe's inner calls. This
validates the trace-through approach as a classification strategy.
