"""
ERC-8183 Worker — Arc Testnet Evaluator for Agent Health Monitor

Background task that watches for JobSubmitted events on the AgenticCommerce
contract (Arc testnet) where the evaluator is the AHM relayer address.
On each event, calls AHM /ahs/{provider} to score the provider, then
executes complete() or reject() on-chain based on the AHS grade.

Runs as an asyncio background task inside the AHM FastAPI lifespan,
alongside alert_monitor, rescan_loop, and the APScheduler jobs.

Environment variables:
    ARC_CONTRACT_ADDRESS  — AgenticCommerce proxy on Arc testnet (required)
    ARC_EVALUATOR_KEY     — Private key for the evaluator wallet (required)
    ARC_RPC_URL           — Arc testnet RPC (default: https://rpc.testnet.arc.network)
    ARC_CHAIN_ID          — Arc testnet chain ID (default: 5042002)
    ARC_PASS_GRADES       — Comma-separated passing grades (default: A,B)
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from functools import partial

import httpx
from web3 import Web3

logger = logging.getLogger("ahm")

# ---------------------------------------------------------------------------
# Configuration (read once at import, validated at startup)
# ---------------------------------------------------------------------------

ARC_RPC_URL = os.getenv("ARC_RPC_URL", "https://rpc.testnet.arc.network")
ARC_CHAIN_ID = int(os.getenv("ARC_CHAIN_ID", "5042002"))
ARC_CONTRACT_ADDRESS = os.getenv("ARC_CONTRACT_ADDRESS", "")
ARC_EVALUATOR_KEY = os.getenv("ARC_EVALUATOR_KEY", "")
ARC_PASS_GRADES = set(
    g.strip() for g in os.getenv("ARC_PASS_GRADES", "A,B").split(",") if g.strip()
)

POLL_INTERVAL = int(os.getenv("ARC_POLL_INTERVAL", "4"))

# ---------------------------------------------------------------------------
# Minimal ABI — only the entries we need
# ---------------------------------------------------------------------------

COMMERCE_ABI = [
    {
        "type": "function",
        "name": "complete",
        "inputs": [
            {"name": "jobId", "type": "uint256"},
            {"name": "reason", "type": "bytes32"},
            {"name": "optParams", "type": "bytes"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "getJob",
        "inputs": [{"name": "jobId", "type": "uint256"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "id", "type": "uint256"},
                    {"name": "client", "type": "address"},
                    {"name": "provider", "type": "address"},
                    {"name": "evaluator", "type": "address"},
                    {"name": "description", "type": "string"},
                    {"name": "budget", "type": "uint256"},
                    {"name": "expiredAt", "type": "uint256"},
                    {"name": "status", "type": "uint8"},
                    {"name": "hook", "type": "address"},
                ],
            }
        ],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "reject",
        "inputs": [
            {"name": "jobId", "type": "uint256"},
            {"name": "reason", "type": "bytes32"},
            {"name": "optParams", "type": "bytes"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "event",
        "name": "JobSubmitted",
        "inputs": [
            {"name": "jobId", "type": "uint256", "indexed": True},
            {"name": "provider", "type": "address", "indexed": True},
            {"name": "deliverable", "type": "bytes32", "indexed": False},
        ],
        "anonymous": False,
    },
]

# ---------------------------------------------------------------------------
# Runtime stats (exposed via /erc8183/status)
# ---------------------------------------------------------------------------

_stats = {
    "running": False,
    "jobs_evaluated_today": 0,
    "jobs_evaluated_total": 0,
    "last_evaluated_at": None,
    "today_date": None,
    "evaluator_address": None,
}


def get_status() -> dict:
    """Return current worker status for the /erc8183/status endpoint."""
    return {
        "running": _stats["running"],
        "contract_address": ARC_CONTRACT_ADDRESS,
        "evaluator_address": _stats["evaluator_address"],
        "chain_id": ARC_CHAIN_ID,
        "rpc_url": ARC_RPC_URL,
        "pass_grades": sorted(ARC_PASS_GRADES),
        "jobs_evaluated_today": _stats["jobs_evaluated_today"],
        "jobs_evaluated_total": _stats["jobs_evaluated_total"],
        "last_evaluated_at": _stats["last_evaluated_at"],
    }


def _bump_eval_count():
    """Increment evaluation counters and reset daily count if date changed."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _stats["today_date"] != today:
        _stats["today_date"] = today
        _stats["jobs_evaluated_today"] = 0
    _stats["jobs_evaluated_today"] += 1
    _stats["jobs_evaluated_total"] += 1
    _stats["last_evaluated_at"] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Health check via AHM /ahs endpoint (localhost with internal key)
# ---------------------------------------------------------------------------

def _check_provider_health(provider_address: str) -> dict:
    """Call AHM /ahs/{address} using localhost + X-Internal-Key."""
    # Import here to avoid circular import at module level
    ahm_port = os.getenv("PORT", "4021")
    url = f"http://127.0.0.1:{ahm_port}/ahs/{provider_address}"

    headers = {}
    internal_key = os.getenv("INTERNAL_API_KEY", "")
    if internal_key:
        headers["X-Internal-Key"] = internal_key

    try:
        resp = httpx.get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("ERC-8183 worker: AHM returned %d for %s", resp.status_code, provider_address)
        return {"error": f"status_{resp.status_code}", "status_code": resp.status_code}
    except Exception as e:
        logger.error("ERC-8183 worker: AHM request failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# On-chain transaction helper (sync, runs in executor)
# ---------------------------------------------------------------------------

def _send_tx(w3, contract_fn, private_key, label=""):
    """Build, sign, send a transaction and wait for receipt."""
    account = w3.eth.account.from_key(private_key)
    tx = contract_fn.build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 500_000,
        "gasPrice": w3.eth.gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    status = "OK" if receipt["status"] == 1 else "REVERTED"
    logger.info("ERC-8183 [%s] %s  tx=%s  gas=%d", status, label, tx_hash.hex()[:18], receipt["gasUsed"])
    return receipt


# ---------------------------------------------------------------------------
# Evaluate a single job (sync, runs in executor)
# ---------------------------------------------------------------------------

def _evaluate_job(w3, commerce, job_id: int, provider: str):
    """Score a provider via AHM and call complete() or reject() on-chain."""
    logger.info("ERC-8183 evaluating job #%d  provider=%s", job_id, provider)

    result = _check_provider_health(provider)
    if "error" in result:
        logger.warning("ERC-8183 job #%d: AHM error %s — skipping", job_id, result["error"])
        return

    report = result.get("report", {})
    ahs_score = report.get("agent_health_score", 0)
    grade_raw = report.get("grade", "F — Unknown")
    grade_letter = grade_raw.split(" ")[0] if " " in grade_raw else grade_raw
    confidence = report.get("confidence", "UNKNOWN")

    logger.info(
        "ERC-8183 job #%d: AHS=%d grade=%s confidence=%s",
        job_id, ahs_score, grade_letter, confidence,
    )

    reason_text = f"AHS {ahs_score} grade {grade_letter}"
    reason_bytes32 = Web3.keccak(text=reason_text)

    if grade_letter in ARC_PASS_GRADES:
        logger.info("ERC-8183 job #%d: COMPLETE (grade %s in pass list)", job_id, grade_letter)
        try:
            _send_tx(
                w3,
                commerce.functions.complete(job_id, reason_bytes32, b""),
                ARC_EVALUATOR_KEY,
                f"complete(job={job_id})",
            )
        except Exception as e:
            logger.error("ERC-8183 job #%d complete() failed: %s", job_id, e)
    else:
        logger.info("ERC-8183 job #%d: REJECT (grade %s not in pass list)", job_id, grade_letter)
        try:
            _send_tx(
                w3,
                commerce.functions.reject(job_id, reason_bytes32, b""),
                ARC_EVALUATOR_KEY,
                f"reject(job={job_id})",
            )
        except Exception as e:
            logger.error("ERC-8183 job #%d reject() failed: %s", job_id, e)

    _bump_eval_count()


# ---------------------------------------------------------------------------
# Main async loop (started via asyncio.create_task in lifespan)
# ---------------------------------------------------------------------------

def can_start() -> bool:
    """Check whether required env vars are present."""
    missing = []
    if not ARC_CONTRACT_ADDRESS:
        missing.append("ARC_CONTRACT_ADDRESS")
    if not ARC_EVALUATOR_KEY:
        missing.append("ARC_EVALUATOR_KEY")
    if missing:
        logger.warning(
            "ERC-8183 worker disabled: missing env vars %s", ", ".join(missing)
        )
        return False
    return True


async def erc8183_worker_loop():
    """Background loop: poll for JobSubmitted events on Arc testnet.

    All blocking web3/httpx calls are offloaded to the default executor.
    """
    # Startup delay to let the API finish initializing
    await asyncio.sleep(10)

    loop = asyncio.get_running_loop()

    try:
        w3 = Web3(Web3.HTTPProvider(ARC_RPC_URL))
        connected = await loop.run_in_executor(None, lambda: w3.is_connected())
        if not connected:
            logger.error("ERC-8183 worker: cannot connect to RPC at %s", ARC_RPC_URL)
            return

        chain_id = await loop.run_in_executor(None, lambda: w3.eth.chain_id)
        logger.info("ERC-8183 worker: connected to chain %d via %s", chain_id, ARC_RPC_URL)

        evaluator_account = w3.eth.account.from_key(ARC_EVALUATOR_KEY)
        evaluator_address = evaluator_account.address
        _stats["evaluator_address"] = evaluator_address
        _stats["running"] = True

        commerce = w3.eth.contract(
            address=Web3.to_checksum_address(ARC_CONTRACT_ADDRESS),
            abi=COMMERCE_ABI,
        )

        start_block = await loop.run_in_executor(None, lambda: w3.eth.block_number)
        last_block = start_block

        logger.info(
            "ERC-8183 worker: watching from block %d  contract=%s  evaluator=%s  pass_grades=%s",
            start_block, ARC_CONTRACT_ADDRESS, evaluator_address, ARC_PASS_GRADES,
        )

        while True:
            try:
                current_block = await loop.run_in_executor(None, lambda: w3.eth.block_number)

                if current_block > last_block:
                    from_blk = last_block + 1
                    last_block = current_block

                    events = await loop.run_in_executor(
                        None,
                        lambda: commerce.events.JobSubmitted.create_filter(
                            from_block=from_blk,
                            to_block=current_block,
                        ).get_all_entries(),
                    )

                    for event in events:
                        job_id = event["args"]["jobId"]
                        provider = event["args"]["provider"]

                        logger.info(
                            "ERC-8183 JobSubmitted: job=%d provider=%s block=%d",
                            job_id, provider, event["blockNumber"],
                        )

                        # Check if we are the evaluator
                        job = await loop.run_in_executor(
                            None,
                            lambda jid=job_id: commerce.functions.getJob(jid).call(),
                        )
                        job_evaluator = job[3]  # evaluator field

                        if job_evaluator.lower() == evaluator_address.lower():
                            logger.info("ERC-8183 job #%d: we are the evaluator, scoring...", job_id)
                            await loop.run_in_executor(
                                None,
                                partial(_evaluate_job, w3, commerce, job_id, provider),
                            )
                        else:
                            logger.debug(
                                "ERC-8183 job #%d: evaluator is %s, not us — skipping",
                                job_id, job_evaluator,
                            )

                await asyncio.sleep(POLL_INTERVAL)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("ERC-8183 worker poll error: %s", e)
                await asyncio.sleep(POLL_INTERVAL)

    except asyncio.CancelledError:
        logger.info("ERC-8183 worker: shutting down")
    except Exception as e:
        logger.error("ERC-8183 worker: fatal error: %s", e)
    finally:
        _stats["running"] = False
