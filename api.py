#!/usr/bin/env python3
"""
Agent Health Monitor - x402 API Service

REST API providing Base blockchain agent health reports,
payable via x402 protocol (USDC on Base).

Usage:
    uvicorn api:app --host 0.0.0.0 --port 4021

Environment variables:
    PAYMENT_ADDRESS  - Your wallet address to receive USDC (required)
    BASESCAN_API_KEY - Blockscout API key for higher rate limits (optional)
    FACILITATOR_URL  - x402 facilitator endpoint (optional)
    PRICE_USD        - Price per health check, e.g. "$0.50" (optional)
    NETWORK          - CAIP-2 network ID (optional, default: Base mainnet)
    PORT             - Server port (optional, default: 4021)
"""

import asyncio
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.server import x402ResourceServer

from monitor import (
    analyze_address,
    fetch_transactions,
    get_eth_price,
    is_contract_address,
    optimize_gas,
)

load_dotenv()

# -- Configuration -----------------------------------------------------------

PAYMENT_ADDRESS = os.environ.get("PAYMENT_ADDRESS", "")
if not PAYMENT_ADDRESS:
    raise RuntimeError(
        "PAYMENT_ADDRESS environment variable is required. "
        "Set it to your wallet address that will receive USDC payments."
    )

FACILITATOR_URL = os.getenv("FACILITATOR_URL", "https://x402.org/facilitator")
BASESCAN_API_KEY = os.getenv("BASESCAN_API_KEY", "")
PRICE = os.getenv("PRICE_USD", "$0.50")
OPTIMIZE_PRICE = os.getenv("OPTIMIZE_PRICE_USD", "$5.00")
# x402.org facilitator supports Base Sepolia (eip155:84532).
# For Base mainnet (eip155:8453), use the CDP facilitator:
#   FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
NETWORK = os.getenv("NETWORK", "eip155:84532")  # Base Sepolia (testnet)
PORT = int(os.getenv("PORT", "4021"))

ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


# -- Response Models ---------------------------------------------------------

class Recommendation(BaseModel):
    category: str
    severity: str  # "critical", "high", "medium", "info"
    message: str


class HealthReport(BaseModel):
    address: str
    is_contract: bool
    health_score: float
    optimization_priority: str
    total_transactions: int
    successful: int
    failed: int
    success_rate_pct: float
    total_gas_spent_eth: float
    wasted_gas_eth: float
    estimated_monthly_waste_usd: float
    avg_gas_efficiency_pct: float
    out_of_gas_count: int
    reverted_count: int
    nonce_gap_count: int
    retry_count: int
    top_failure_type: str
    first_seen: str
    last_seen: str
    recommendations: list[Recommendation]
    eth_price_usd: float
    analyzed_at: str


class HealthResponse(BaseModel):
    status: str
    report: HealthReport


class TransactionTypeOptimization(BaseModel):
    contract: str
    method_id: str
    method_label: str
    tx_count: int
    failed_count: int
    failure_rate_pct: float
    current_avg_gas_limit: int
    current_p50_gas_used: int
    current_p95_gas_used: int
    optimal_gas_limit: int
    gas_limit_reduction_pct: float
    wasted_gas_eth: float
    wasted_gas_usd: float


class GasOptimizationReport(BaseModel):
    address: str
    total_transactions_analyzed: int
    current_monthly_gas_usd: float
    optimized_monthly_gas_usd: float
    estimated_monthly_savings_usd: float
    total_wasted_gas_eth: float
    total_wasted_gas_usd: float
    tx_types: list[TransactionTypeOptimization]
    recommendations: list[str]
    eth_price_usd: float
    analyzed_at: str


class OptimizeResponse(BaseModel):
    status: str
    report: GasOptimizationReport


# -- Recommendations Engine --------------------------------------------------

def generate_recommendations(health) -> list[Recommendation]:
    """Generate actionable optimization recommendations from health data."""
    recs = []

    if health.total_transactions == 0:
        recs.append(Recommendation(
            category="activity",
            severity="info",
            message="No outgoing transactions found for this address.",
        ))
        return recs

    # High failure rate
    if health.success_rate_pct < 90:
        severity = "critical" if health.success_rate_pct < 70 else "high"
        recs.append(Recommendation(
            category="reliability",
            severity=severity,
            message=(
                f"Transaction success rate is {health.success_rate_pct}%. "
                f"{health.failed} of {health.total_transactions} transactions failed. "
                "Review contract interactions for revert conditions and "
                "add pre-flight simulation via eth_call before submitting."
            ),
        ))

    # Out of gas
    if health.out_of_gas_count > 0:
        recs.append(Recommendation(
            category="gas_management",
            severity="high",
            message=(
                f"{health.out_of_gas_count} transactions ran out of gas. "
                "Increase gas limit estimates by 20-30% or use eth_estimateGas "
                "before submitting. Consider dynamic gas limit adjustment "
                "based on calldata complexity."
            ),
        ))

    # Gas efficiency too low (over-estimating limits)
    if 0 < health.avg_gas_efficiency_pct < 40:
        recs.append(Recommendation(
            category="gas_efficiency",
            severity="medium",
            message=(
                f"Average gas utilization is {health.avg_gas_efficiency_pct}% of limit. "
                "Gas limits are set too high. Use tighter estimates via "
                "eth_estimateGas with a 10-15% buffer."
            ),
        ))

    # Gas efficiency too high (under-estimating limits)
    if health.avg_gas_efficiency_pct > 85:
        recs.append(Recommendation(
            category="gas_efficiency",
            severity="medium",
            message=(
                f"Average gas utilization is {health.avg_gas_efficiency_pct}% of limit. "
                "Gas limits are dangerously tight, likely causing out-of-gas failures. "
                "Add a 20% buffer to estimated gas."
            ),
        ))

    # Nonce gaps
    if health.nonce_gap_count > 0:
        recs.append(Recommendation(
            category="nonce_management",
            severity="high",
            message=(
                f"Detected {health.nonce_gap_count} nonce gaps in transaction sequence. "
                "This causes stuck pending transactions. Implement proper nonce "
                "tracking with pending transaction awareness."
            ),
        ))

    # Nonce retries
    if health.retry_count > 0:
        recs.append(Recommendation(
            category="nonce_management",
            severity="medium",
            message=(
                f"Detected {health.retry_count} nonce reuse events (retries/replacements). "
                "Excessive retries signal gas price estimation issues. "
                "Consider EIP-1559 dynamic fee estimation."
            ),
        ))

    # Wasted gas cost
    if health.estimated_monthly_waste_usd > 10:
        severity = "critical" if health.estimated_monthly_waste_usd > 100 else "high"
        recs.append(Recommendation(
            category="cost",
            severity=severity,
            message=(
                f"Estimated ${health.estimated_monthly_waste_usd:.2f}/month wasted on "
                f"failed transactions ({health.wasted_gas_eth:.6f} ETH total). "
                "Implement transaction simulation (eth_call) before submission "
                "to avoid paying for failures."
            ),
        ))

    # Reverted transactions
    if health.reverted_count > 0 and health.top_failure_type == "reverted":
        recs.append(Recommendation(
            category="contract_interaction",
            severity="high",
            message=(
                f"{health.reverted_count} transactions reverted. Common causes: "
                "insufficient token approval, slippage too low, deadline expired, "
                "or invalid state. Add pre-checks for allowances and balances "
                "before executing swaps/transfers."
            ),
        ))

    # Healthy agent
    if health.health_score >= 85 and not recs:
        recs.append(Recommendation(
            category="general",
            severity="info",
            message=(
                f"Health score {health.health_score}/100. "
                "This agent is operating well. Continue monitoring for regressions."
            ),
        ))

    return recs


# -- FastAPI App -------------------------------------------------------------

app = FastAPI(
    title="Agent Health Monitor API",
    description=(
        "Analyzes Base blockchain agent wallets for transaction failures, "
        "gas inefficiency, and nonce issues. Returns actionable health reports. "
        "Payments via x402 protocol (USDC on Base)."
    ),
    version="1.1.0",
)


# -- x402 Payment Middleware -------------------------------------------------

facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=FACILITATOR_URL))
server = x402ResourceServer(facilitator)
server.register(NETWORK, ExactEvmServerScheme())

x402_routes = {
    "GET /health/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Base blockchain agent wallet health analyzer. "
            "Returns health score, gas efficiency, failure analysis, "
            "and optimization recommendations for any Base L2 wallet address."
        ),
        extensions={
            "bazaar": {
                "info": {
                    "input": {
                        "type": "http",
                        "method": "GET",
                        "queryParams": {
                            "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                        },
                    },
                    "output": {
                        "type": "json",
                        "example": {
                            "status": "ok",
                            "report": {
                                "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                                "is_contract": True,
                                "health_score": 82.5,
                                "optimization_priority": "MEDIUM",
                                "total_transactions": 150,
                                "successful": 138,
                                "failed": 12,
                                "success_rate_pct": 92.0,
                                "total_gas_spent_eth": 0.00342,
                                "wasted_gas_eth": 0.00041,
                                "estimated_monthly_waste_usd": 1.23,
                                "avg_gas_efficiency_pct": 65.4,
                                "out_of_gas_count": 2,
                                "reverted_count": 10,
                                "nonce_gap_count": 0,
                                "retry_count": 1,
                                "top_failure_type": "reverted",
                                "first_seen": "2024-03-01",
                                "last_seen": "2025-06-15",
                                "recommendations": [
                                    {
                                        "category": "reliability",
                                        "severity": "high",
                                        "message": "Transaction success rate is 92.0%...",
                                    },
                                ],
                                "eth_price_usd": 2500.0,
                                "analyzed_at": "2026-02-16T12:00:00Z",
                            },
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "report": {
                                    "type": "object",
                                    "properties": {
                                        "address": {"type": "string", "description": "Wallet address analyzed"},
                                        "is_contract": {"type": "boolean", "description": "Whether address is a smart contract"},
                                        "health_score": {"type": "number", "description": "Composite health score 0-100"},
                                        "optimization_priority": {"type": "string", "description": "CRITICAL, HIGH, MEDIUM, or LOW"},
                                        "total_transactions": {"type": "integer", "description": "Total outgoing transactions"},
                                        "successful": {"type": "integer"},
                                        "failed": {"type": "integer"},
                                        "success_rate_pct": {"type": "number", "description": "Success rate percentage"},
                                        "total_gas_spent_eth": {"type": "number"},
                                        "wasted_gas_eth": {"type": "number", "description": "ETH wasted on failed transactions"},
                                        "estimated_monthly_waste_usd": {"type": "number"},
                                        "avg_gas_efficiency_pct": {"type": "number"},
                                        "out_of_gas_count": {"type": "integer"},
                                        "reverted_count": {"type": "integer"},
                                        "nonce_gap_count": {"type": "integer"},
                                        "retry_count": {"type": "integer"},
                                        "top_failure_type": {"type": "string"},
                                        "first_seen": {"type": "string"},
                                        "last_seen": {"type": "string"},
                                        "recommendations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "category": {"type": "string"},
                                                    "severity": {"type": "string"},
                                                    "message": {"type": "string"},
                                                },
                                            },
                                            "description": "Actionable optimization recommendations",
                                        },
                                        "eth_price_usd": {"type": "number"},
                                        "analyzed_at": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    ),
    "GET /optimize/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=OPTIMIZE_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Gas optimization service for Base L2 agent wallets. "
            "Analyzes transaction types, calculates optimal gas limits, "
            "identifies wasted gas, and estimates monthly savings."
        ),
        extensions={
            "bazaar": {
                "info": {
                    "input": {
                        "type": "http",
                        "method": "GET",
                        "queryParams": {
                            "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                        },
                    },
                    "output": {
                        "type": "json",
                        "example": {
                            "status": "ok",
                            "report": {
                                "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                                "total_transactions_analyzed": 150,
                                "current_monthly_gas_usd": 45.20,
                                "optimized_monthly_gas_usd": 38.50,
                                "estimated_monthly_savings_usd": 6.70,
                                "total_wasted_gas_eth": 0.00215,
                                "total_wasted_gas_usd": 5.37,
                                "tx_types": [
                                    {
                                        "contract": "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
                                        "method_id": "0x3593564c",
                                        "method_label": "execute (Universal Router)",
                                        "tx_count": 85,
                                        "failed_count": 8,
                                        "failure_rate_pct": 9.4,
                                        "current_avg_gas_limit": 350000,
                                        "current_p50_gas_used": 185000,
                                        "current_p95_gas_used": 245000,
                                        "optimal_gas_limit": 281750,
                                        "gas_limit_reduction_pct": 19.5,
                                        "wasted_gas_eth": 0.00145,
                                        "wasted_gas_usd": 3.62,
                                    },
                                ],
                                "recommendations": [
                                    "Gas limits are 19.5% too high for execute (Universal Router).",
                                    "Eliminating failed transactions would save ~$6.70/month.",
                                ],
                                "eth_price_usd": 2500.0,
                                "analyzed_at": "2026-02-17T12:00:00Z",
                            },
                        },
                    },
                },
            },
        },
    ),
}

app.add_middleware(PaymentMiddlewareASGI, routes=x402_routes, server=server)


# -- Routes ------------------------------------------------------------------

@app.get("/")
async def root():
    """Service info and pricing."""
    return {
        "service": "Agent Health Monitor",
        "version": "1.1.0",
        "network": "Base L2",
        "endpoints": {
            "GET /health/{address}": f"{PRICE} USDC — wallet health diagnosis",
            "GET /optimize/{address}": f"{OPTIMIZE_PRICE} USDC — gas optimization report",
        },
        "payment": "x402 protocol (USDC on Base)",
        "docs": "/docs",
    }


@app.get("/up")
async def up():
    """Unpaid liveness probe for load balancers."""
    return {"status": "ok"}


@app.get("/health/{address}", response_model=HealthResponse)
async def get_health_report(address: str):
    """
    Analyze a Base wallet address and return a health report.

    Requires x402 payment ($0.50 USDC on Base).

    - Fetches transaction history from Blockscout
    - Calculates success rate, gas efficiency, nonce health
    - Computes composite health score (0-100)
    - Generates specific optimization recommendations
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Run blocking I/O from monitor.py in thread pool
    loop = asyncio.get_running_loop()

    eth_price, transactions, is_contract = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(is_contract_address, address)),
    )

    health = await loop.run_in_executor(
        None, partial(analyze_address, address, transactions, eth_price),
    )
    health.is_contract = is_contract

    recommendations = generate_recommendations(health)

    report = HealthReport(
        **{k: v for k, v in asdict(health).items()},
        recommendations=recommendations,
        eth_price_usd=eth_price,
        analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    return HealthResponse(status="ok", report=report)


@app.get("/optimize/{address}", response_model=OptimizeResponse)
async def get_optimization_report(address: str):
    """
    Analyze a Base wallet and return a gas optimization report.

    Requires x402 payment ($5.00 USDC on Base).

    - Groups transactions by type (contract + method)
    - Calculates optimal gas limits per type
    - Identifies wasted gas from failed transactions
    - Estimates monthly savings with before/after comparison
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    loop = asyncio.get_running_loop()

    eth_price, transactions = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
    )

    optimization = await loop.run_in_executor(
        None, partial(optimize_gas, address, transactions, eth_price),
    )

    report = GasOptimizationReport(
        address=optimization.address,
        total_transactions_analyzed=optimization.total_transactions_analyzed,
        current_monthly_gas_usd=optimization.current_monthly_gas_usd,
        optimized_monthly_gas_usd=optimization.optimized_monthly_gas_usd,
        estimated_monthly_savings_usd=optimization.estimated_monthly_savings_usd,
        total_wasted_gas_eth=optimization.total_wasted_gas_eth,
        total_wasted_gas_usd=optimization.total_wasted_gas_usd,
        tx_types=[
            TransactionTypeOptimization(**{
                "contract": t.contract,
                "method_id": t.method_id,
                "method_label": t.method_label,
                "tx_count": t.tx_count,
                "failed_count": t.failed_count,
                "failure_rate_pct": t.failure_rate_pct,
                "current_avg_gas_limit": t.current_avg_gas_limit,
                "current_p50_gas_used": t.current_p50_gas_used,
                "current_p95_gas_used": t.current_p95_gas_used,
                "optimal_gas_limit": t.optimal_gas_limit,
                "gas_limit_reduction_pct": t.gas_limit_reduction_pct,
                "wasted_gas_eth": t.wasted_gas_eth,
                "wasted_gas_usd": t.wasted_gas_usd,
            })
            for t in optimization.tx_types
        ],
        recommendations=optimization.recommendations,
        eth_price_usd=eth_price,
        analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    return OptimizeResponse(status="ok", report=report)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
