#!/usr/bin/env python3
"""Outreach scan — AHS scores for partnership target wallets."""

import time
from datetime import datetime, timezone
from monitor import calculate_ahs, fetch_tokens_v2, fetch_transactions, get_eth_price

DELAY = 2.0

TARGETS = [
    # (label, address, notes)
    ("KAMIYO", "0x742d35cc6634c0532925a3b844bc9e7595f0bee4", "EVM signer from facilitator registry"),
    ("PayAI (signer 1)", "0xc6699d2aada6c36dfea5c248dd70f9cb0235cb63", "Base facilitator signer"),
    ("PayAI (signer 2)", "0xb2bd29925cbbcea7628279c91945ca5b98bf371b", "Base facilitator signer"),
    ("PayAI (example)", "0x209693Bc6afc0C5328bA36FaF03C514EF312287C", "Docs example recipient"),
    ("Daydreams (EVM signer)", "0x1363C7Ff51CcCE10258A7F7bddd63bAaB6aAf678", "Facilitator EVM signer"),
    ("Daydreams (facilitator)", "0x279e08f711182c79Ba6d09669127a426228a4653", "Facilitator operational"),
    ("Daydreams (payTo example)", "0xb308ed39d67D0d4BAe5BC2FAEF60c66BBb6AE429", "GitHub nanoservice example"),
]

def main():
    print(f"[*] Outreach Scan — {len(TARGETS)} addresses")
    eth_price = get_eth_price()
    print(f"[+] ETH: ${eth_price:,.2f}\n")

    results = []
    for i, (label, addr, notes) in enumerate(TARGETS, 1):
        print(f"[{i}/{len(TARGETS)}] {label}: {addr[:10]}...{addr[-6:]}")
        try:
            txs = fetch_transactions(addr)
            print(f"  Txs: {len(txs)}")
            time.sleep(DELAY)
            tokens = fetch_tokens_v2(addr, max_pages=3)
            print(f"  Tokens: {len(tokens)}")
            time.sleep(DELAY)

            if not txs:
                print(f"  [!] No transactions — skipping AHS")
                results.append((label, addr, notes, "N/A", "No txs on Base", "", "", "", "No Base activity detected"))
                continue

            ahs = calculate_ahs(address=addr.lower(), tokens=tokens, transactions=txs, eth_price=eth_price)
            patterns = "; ".join(
                p.get("pattern", str(p)) if isinstance(p, dict) else str(p)
                for p in ahs.patterns_detected
            ) if ahs.patterns_detected else "none"

            print(f"  AHS: {ahs.agent_health_score} ({ahs.grade} — {ahs.grade_label}) | D1={ahs.d1_score} D2={ahs.d2_score}")
            results.append((label, addr, notes, ahs.agent_health_score, f"{ahs.grade} {ahs.grade_label}",
                           ahs.d1_score, ahs.d2_score, ahs.confidence, patterns))
        except Exception as e:
            print(f"  [!] Error: {e}")
            results.append((label, addr, notes, "ERROR", str(e)[:50], "", "", "", ""))

    # Write markdown
    with open("ecosystem_scan_outreach.md", "w") as f:
        f.write("# Ecosystem Scan — Outreach Targets\n\n")
        f.write(f"> Scanned: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"> ETH price: ${eth_price:,.2f}\n\n")

        f.write("## Results\n\n")
        f.write("| Service | Domain | Wallet | AHS Score | Grade | D1 | D2 | Key Finding |\n")
        f.write("|---------|--------|--------|-----------|-------|----|----|-------------|\n")

        domain_map = {
            "KAMIYO": "kamiyo.ai",
            "PayAI": "payai.network",
            "Daydreams": "daydreams.systems",
            "AgentCard": "agentcard.ai",
            "OpenServ": "openserv.ai",
        }

        for label, addr, notes, ahs, grade, d1, d2, conf, finding in results:
            domain = next((v for k, v in domain_map.items() if k in label), "—")
            addr_short = f"`{addr[:8]}...{addr[-4:]}`"
            key = finding if finding else notes
            f.write(f"| {label} | {domain} | {addr_short} | {ahs} | {grade} | {d1} | {d2} | {key} |\n")

        # Add not-found services
        f.write(f"| AgentCard | agentcard.ai | — | — | — | — | — | No x402 wallet found; virtual card issuer, not x402 native |\n")
        f.write(f"| OpenServ | openserv.ai | — | — | — | — | — | No x402 wallet found; $SERV token on ETH only, x402 Lite announced |\n")

        f.write("\n## Discovery Notes\n\n")
        f.write("- **KAMIYO** — Facilitator at `kamiyo.ai/api/v1/x402`, gated (requires API key). Also has Solana address.\n")
        f.write("- **PayAI** — Facilitator at `facilitator.payai.network` (currently timing out). 5 Base signers found, 14 chains supported.\n")
        f.write("- **Daydreams** — Facilitator at `facilitator.daydreams.systems` (live). `/supported` endpoint returns structured signer map.\n")
        f.write("- **AgentCard** — Virtual Visa card issuer for agents. x402 support listed as coming. No wallet discoverable.\n")
        f.write("- **OpenServ** — $SERV token on Ethereum. x402 Lite announced but no facilitator or wallet addresses public.\n")
        f.write("- **Discovery method**: x402 Facilitators Watch (facilitators.x402.watch) was the most productive source. `.well-known/x402` returned 404 on all 5 services.\n")

    print(f"\n[+] Written: ecosystem_scan_outreach.md")

if __name__ == "__main__":
    main()
