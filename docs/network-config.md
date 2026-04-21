# ERC-8183 Network Configuration

## Active Test Environment

| Field              | Value                                        |
|--------------------|----------------------------------------------|
| Network            | Base Sepolia                                 |
| Chain ID           | 84532                                        |
| RPC                | https://sepolia.base.org                     |
| AgentJobManager    | `0x892e7e77BC8DBc7E97E16E8e7DcF3783aFbB3A19` |
| AHM Provider Wallet| `0x35eeDdcbE5E1AE01396Cb93Fc8606cE4C713d7BC` |
| Explorer           | https://sepolia.basescan.org                 |

## Inactive Networks

### Arc Testnet

Arc testnet contracts exist (`ARC_CONTRACT_ADDRESS`, `ARC_RPC_URL` in
`erc8183_worker.py`) but are **not the active test environment** for
ERC-8183 job submissions. The Arc deployment was used for early
integration testing of the evaluator worker loop and is retained for
reference only.

| Field              | Value                                        |
|--------------------|----------------------------------------------|
| Network            | Arc Testnet                                  |
| Chain ID           | 5042002                                      |
| RPC                | https://rpc.testnet.arc.network              |
| AgenticCommerce    | Set via `ARC_CONTRACT_ADDRESS` env var        |
| Evaluator Key      | `ARC_EVALUATOR_KEY` env var                   |

## Notes

- The `ARC_EVALUATOR_KEY` private key is shared between both networks
  (same wallet `0x35eeDdcbE5E1AE01396Cb93Fc8606cE4C713d7BC`).
- All ERC-8183 job submissions (e.g., `submit()`, `complete()`) should
  target **Base Sepolia** unless explicitly stated otherwise.
- The `erc8183_worker.py` background loop currently points at Arc testnet.
  It will need updating when the evaluator role moves to Base Sepolia.
