#!/usr/bin/env python3
"""Register Agent Health Monitor on the ERC-8004 Identity Registry (Base mainnet)."""

import json
import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# --- Config ---
BASE_RPC = "https://mainnet.base.org"
REGISTRY_ADDRESS = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
AGENT_URI = "https://agenthealthmonitor.xyz/.well-known/agent-registration.json"

# Minimal ABI: register(string agentURI) and Registered event
REGISTRY_ABI = [
    {
        "type": "function",
        "name": "register",
        "inputs": [{"name": "agentURI", "type": "string"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "event",
        "name": "Registered",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": False},
            {"name": "agentURI", "type": "string", "indexed": False},
            {"name": "owner", "type": "address", "indexed": False},
        ],
    },
]


def main():
    private_key = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
    if not private_key:
        print("[!] NANSEN_PAYER_PRIVATE_KEY not set in .env")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(BASE_RPC))
    if not w3.is_connected():
        print("[!] Cannot connect to Base mainnet RPC")
        sys.exit(1)

    account = w3.eth.account.from_key(private_key)
    print(f"[*] Wallet:   {account.address}")
    print(f"[*] Registry: {REGISTRY_ADDRESS}")
    print(f"[*] agentURI: {AGENT_URI}")

    # Check balance
    balance = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance, "ether")
    print(f"[*] Balance:  {balance_eth:.6f} ETH")
    if balance == 0:
        print("[!] Zero ETH balance — cannot pay gas")
        sys.exit(1)

    registry = w3.eth.contract(
        address=w3.to_checksum_address(REGISTRY_ADDRESS),
        abi=REGISTRY_ABI,
    )

    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    tx = registry.functions.register(AGENT_URI).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "maxFeePerGas": w3.eth.gas_price * 2,
        "maxPriorityFeePerGas": w3.to_wei(0.001, "gwei"),
        "chainId": 8453,
    })

    # Estimate gas
    gas_estimate = w3.eth.estimate_gas(tx)
    tx["gas"] = int(gas_estimate * 1.2)
    gas_cost_eth = w3.from_wei(tx["gas"] * tx["maxFeePerGas"], "ether")
    print(f"[*] Gas est:  {gas_estimate} (padded to {tx['gas']})")
    print(f"[*] Max cost: {gas_cost_eth:.6f} ETH")

    # Sign and send
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"\n[+] Tx sent:  {tx_hash.hex()}")
    print(f"[*] Waiting for confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    print(f"[+] Status:   {'SUCCESS' if receipt.status == 1 else 'FAILED'}")
    print(f"[+] Block:    {receipt.blockNumber}")
    print(f"[+] Gas used: {receipt.gasUsed}")

    if receipt.status != 1:
        print("[!] Transaction reverted")
        sys.exit(1)

    # Parse Registered event to get agentId
    logs = registry.events.Registered().process_receipt(receipt)
    if logs:
        agent_id = logs[0]["args"]["agentId"]
        owner = logs[0]["args"]["owner"]
        print(f"\n[+] agentId:  {agent_id}")
        print(f"[+] Owner:    {owner}")
        print(f"[+] Basescan: https://basescan.org/tx/{tx_hash.hex()}")
    else:
        print("[!] No Registered event found in logs — check tx manually")
        print(f"[+] Basescan: https://basescan.org/tx/{tx_hash.hex()}")


if __name__ == "__main__":
    main()
