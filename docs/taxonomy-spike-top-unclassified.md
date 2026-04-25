# Taxonomy Spike: Top Unclassified Contracts

**Date:** 2026-04-25
**Branch:** `data/taxonomy-new-sources`
**Context:** Phase 2 taxonomy classification ran on 500 random Base mainnet agent
wallets. 75.8% were classified, but 121 agents (24.2%) remain unclassified. This
spike investigates the top unclassified contracts by interaction count.

---

## Contract #1: `0x00005ea00ac477b1030ce78506496e8c2de24bf5` (594 interactions)

### Identification

- **Contract Name:** SeaDrop
- **Protocol:** OpenSea
- **Compiler:** Solidity v0.8.17
- **Verified:** Yes (source code on Blockscout, 45,656 chars)
- **Proxy:** No

### What It Does

SeaDrop is OpenSea's smart contract for conducting **primary NFT drops** on
EVM-compatible blockchains. It supports public drops, Merkle-tree allowlists,
server-signed mints, and token-gated drops. The contract is deployed at the same
address across 17+ EVM chains (Ethereum, Base, Arbitrum, Polygon, etc.).

Key functions observed: `mintPublic`, `mintAllowList`, `mintSigned`,
`mintAllowedTokenHolder`, `updatePublicDrop`, `updateAllowList`.

### Method Breakdown (1,000 sampled transactions via Blockscout)

| Method | Selector | Count | % |
|--------|----------|-------|---|
| `mintPublic` | `0x161ac21f` | 987 | 98.7% |
| `mintSigned` | `0x4b61cd6f` | 13 | 1.3% |

All 1,000 transactions are **inbound** (agents calling SeaDrop), zero outbound.
Zero seller-side methods (`updatePublicDrop`, `updateAllowList`,
`withdrawProceeds`, etc.) were observed.

### DefiLlama

No direct match for this address. OpenSea appears under "NFT Marketplace"
category but without contract-level address mapping.

### Proposed Action

**Add to taxonomy** as `Commerce Agents` with label "SeaDrop (OpenSea NFT Drops)".
The 100% buyer-side activity (mintPublic/mintSigned, all inbound) indicates
automated NFT purchasing rather than autonomous artistry. This is the first
contract anchor in the previously-empty Commerce category, alongside the existing
Seaport (OpenSea) entry.

---

## Contract #2: `0xd85ee50da419cc5af83a1e70a91d5c630b8c650a` (503 interactions)

### Identification

- **Contract Name:** Unverified (no source code, no ABI on Blockscout)
- **Protocol:** Ritual Network (Infernet)
- **Bytecode:** Empty (0 bytes at current block -- contract may have been
  self-destructed or upgraded)
- **Created:** 2024-06-05 by `0xED55E260cD9Ec62815Cf6fDd75C5020Da3B062D0`

### What It Does

All observed transactions call `deliverCompute(uint32 subscriptionId, uint32
deliveryInterval, bytes input, bytes output, bytes proof, address nodeWallet)`
with method selector `0xc509543d`. This is part of the **Ritual Infernet**
decentralized AI compute network.

Ritual Infernet allows on-chain smart contracts to request off-chain AI/ML
compute workloads. Infernet nodes execute the compute and deliver results back
on-chain via `deliverCompute`. The decoded output from one sample transaction
contained `"hello, world!"` -- consistent with Ritual's
[Hello World tutorial](https://learn.ritual.net/examples/hello_world).

The official Ritual Coordinator on Base is `EIP712Coordinator` at
`0x8D871Ef2826ac9001fB2e33fDD6379b6aaBF449c` (verified). This address
(`0xd85ee50...`) appears to be an older or auxiliary Ritual deployment, possibly
a previous version of the Coordinator or a consumer/subscription contract.

### DefiLlama

No direct match for this address in the DefiLlama protocols list.

### Proposed Action

**Add to taxonomy** as `Intelligence & Analytics Agents` with label
"Ritual Infernet (decentralized AI compute)". Agents calling `deliverCompute`
are delivering AI inference results on-chain, which fits the Intelligence &
Analytics category.

Also add the verified Coordinator (`0x8d871ef2826ac9001fb2e33fdd6379b6aabf449c`)
for broader coverage.

---

## The 199-200 Interaction Cluster (16 contracts)

### Sample Contracts Investigated

| Address | ContractName | Source Hash | Verified |
|---------|-------------|-------------|----------|
| `0x0322e066...` | GnosisSafeProxy | `13a8cd723b5a4fa5` | Yes |
| `0xbb96a7b6...` | GnosisSafe | `744af6600164990c` | Yes |
| `0x31d5ef02...` | GnosisSafeProxy | `13a8cd723b5a4fa5` | Yes |

### Findings

1. **Same template confirmed.** The two GnosisSafeProxy contracts share identical
   source code (SHA-256 prefix: `13a8cd723b5a4fa5`). The third is a GnosisSafe
   implementation contract. All transactions observed are `execTransaction` calls
   -- standard Safe multisig operations.

2. **These are Gnosis Safe (now Safe) multisig wallets.** They are generic
   smart-contract wallets used by agents (and humans) to hold assets and execute
   transactions with multi-signature approval. They are infrastructure, not
   protocol-specific.

3. **Why they cluster at exactly 199-200 interactions:** The classification script
   (`classify_agents_taxonomy.py`, line 73) sets `"offset": 200` in the
   Blockscout API params. This limits the API to return **at most 200
   transactions** per agent wallet. For agents whose entire 200-transaction
   window is dominated by interactions with a single Safe contract, the count
   maxes out at 199-200. The uniform count is an **artifact of the pagination
   cap**, not a real signal.

### Script Cap Evidence

```python
# classify_agents_taxonomy.py, line 67-73
params = {
    "module": "account",
    "action": "txlist",
    "address": address,
    "sort": "desc",
    "offset": 200,   # <-- THIS IS THE CAP
    "page": 1,
}
```

The `offset` parameter in the Blockscout API controls results-per-page, not a
database offset. Setting it to 200 means at most 200 transactions are returned.
Only page 1 is fetched, so agents with >200 transactions have their history
truncated.

### Proposed Action

- **Do NOT add Safe contracts to taxonomy.** Gnosis Safe is generic wallet
  infrastructure. An agent using a Safe tells us nothing about the agent's
  purpose -- the meaningful interactions are whatever the Safe *calls*, not the
  Safe itself. Adding Safe contracts would incorrectly classify agents.

- **Fix the pagination cap.** Increase `offset` to 500 or implement pagination
  (fetching multiple pages) to get a more complete transaction history. This
  would naturally eliminate the artificial 199-200 cluster and improve
  classification coverage.

- **Consider flagging Safe interactions.** Instead of classifying, the script
  could trace through Safe `execTransaction` calls to identify the *target*
  contracts being called, which would be the actual classification signal.

---

## Summary of Actions Taken

| Contract | Identity | Action | Category |
|----------|----------|--------|----------|
| `0x00005ea0...` | SeaDrop (OpenSea) | Added to taxonomy | Commerce Agents |
| `0xd85ee50d...` | Ritual Infernet (AI compute) | Added to taxonomy | Intelligence & Analytics Agents |
| `0x8d871ef2...` | Ritual EIP712Coordinator | Added to taxonomy | Intelligence & Analytics Agents |
| Cluster (Safe) | Gnosis Safe multisig wallets | NOT added -- generic infra | N/A |

## Recommendations

1. **Increase API pagination** in `classify_agents_taxonomy.py` to fetch more
   than 200 transactions per agent (offset=500 or multi-page).
2. **Add Safe trace-through logic** to classify agents based on what their Safe
   wallets call, not the Safe address itself.
3. **Re-run classification** after adding the new taxonomy entries to measure
   coverage improvement.
