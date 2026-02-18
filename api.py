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
    RETRY_PRICE_USD  - Price per retry analysis, e.g. "$10.00" (optional)
    PROTECT_PRICE_USD - Price per protection agent run, e.g. "$25.00" (optional)
    NETWORK          - CAIP-2 network ID (optional, default: Base mainnet)
    PORT             - Server port (optional, default: 4021)
"""

import asyncio
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional

import httpx as httpx_client
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
    analyze_retryable_transactions,
    fetch_transactions,
    get_eth_price,
    is_contract_address,
    optimize_gas,
    run_protection_agent,
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
ALERT_PRICE = os.getenv("ALERT_PRICE_USD", "$2.00")
RETRY_PRICE = os.getenv("RETRY_PRICE_USD", "$10.00")
PROTECT_PRICE = os.getenv("PROTECT_PRICE_USD", "$25.00")
# x402.org facilitator supports Base Sepolia (eip155:84532).
# For Base mainnet (eip155:8453), use the CDP facilitator:
#   FACILITATOR_URL=https://api.cdp.coinbase.com/platform/v2/x402
NETWORK = os.getenv("NETWORK", "eip155:84532")  # Base Sepolia (testnet)
CDP_API_KEY_ID = os.getenv("CDP_API_KEY_ID", "")
# Railway may store PEM with literal \n — convert to real newlines
CDP_API_KEY_SECRET = os.getenv("CDP_API_KEY_SECRET", "").replace("\\n", "\n")
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


# -- Retry Models -----------------------------------------------------------

class OptimizedTransaction(BaseModel):
    to: str
    data: str
    value: str
    gas_limit: str
    max_fee_per_gas: str
    max_priority_fee_per_gas: str


class RetryTransactionItem(BaseModel):
    original_tx_hash: str
    failure_reason: str
    optimized_transaction: OptimizedTransaction
    estimated_gas_cost_usd: float
    confidence: str


class RetryReport(BaseModel):
    address: str
    failed_transactions_analyzed: int
    retryable_count: int
    retry_transactions: list[RetryTransactionItem]
    total_estimated_retry_cost_usd: float
    potential_value_recovered_usd: float
    analyzed_at: str


class RetryResponse(BaseModel):
    status: str
    report: RetryReport


class RetryPreviewResponse(BaseModel):
    status: str
    address: str
    failed_transactions_analyzed: int
    retryable_count: int
    total_estimated_retry_cost_usd: float
    potential_value_recovered_usd: float
    message: str


# -- Protection Agent Models -------------------------------------------------

class ProtectionActionItem(BaseModel):
    priority: int
    action: str
    description: str
    potential_value_usd: float = 0.0
    potential_savings_monthly_usd: float = 0.0


class ProtectionSummaryModel(BaseModel):
    total_issues_found: int
    total_potential_savings_usd: float
    retry_transactions_ready: int
    estimated_retry_cost_usd: float


class ProtectionReport(BaseModel):
    address: str
    risk_level: str
    health_score: float
    services_run: list[str]
    summary: ProtectionSummaryModel
    health_report: HealthReport
    gas_optimization: Optional[GasOptimizationReport] = None
    retry_transactions: Optional[list[RetryTransactionItem]] = None
    recommended_actions: list[ProtectionActionItem]
    analyzed_at: str


class ProtectionResponse(BaseModel):
    status: str
    report: ProtectionReport


class ProtectionPreviewResponse(BaseModel):
    status: str
    address: str
    risk_level: str
    health_score: float
    services_recommended: list[str]
    estimated_issues: int
    message: str


# -- Alert Models & Store ----------------------------------------------------

@dataclass
class Subscription:
    address: str
    webhook_url: Optional[str] = None
    webhook_type: str = "generic"  # "generic", "slack", "discord"
    subscribed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    health_score_threshold: float = 70.0
    failure_rate_threshold: float = 30.0
    waste_usd_threshold: float = 50.0
    last_check_at: Optional[datetime] = None
    last_alert_at: Optional[datetime] = None
    alerts_sent: int = 0


# In-memory subscription store (MVP — lost on restart)
subscriptions: dict[str, Subscription] = {}

logger = logging.getLogger("alert_service")


class AlertThresholds(BaseModel):
    health_score: Optional[float] = None
    failure_rate: Optional[float] = None
    waste_usd: Optional[float] = None


class ConfigureRequest(BaseModel):
    address: str
    webhook_url: str
    webhook_type: str = "generic"
    thresholds: Optional[AlertThresholds] = None


class SubscriptionStatus(BaseModel):
    address: str
    active: bool
    webhook_configured: bool
    webhook_type: str
    subscribed_at: str
    expires_at: str
    thresholds: AlertThresholds
    last_check_at: Optional[str]
    last_alert_at: Optional[str]
    alerts_sent: int


def _sub_to_status(sub: Subscription) -> SubscriptionStatus:
    now = datetime.now(timezone.utc)
    return SubscriptionStatus(
        address=sub.address,
        active=now < sub.expires_at,
        webhook_configured=sub.webhook_url is not None,
        webhook_type=sub.webhook_type,
        subscribed_at=sub.subscribed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        expires_at=sub.expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        thresholds=AlertThresholds(
            health_score=sub.health_score_threshold,
            failure_rate=sub.failure_rate_threshold,
            waste_usd=sub.waste_usd_threshold,
        ),
        last_check_at=sub.last_check_at.strftime("%Y-%m-%dT%H:%M:%SZ") if sub.last_check_at else None,
        last_alert_at=sub.last_alert_at.strftime("%Y-%m-%dT%H:%M:%SZ") if sub.last_alert_at else None,
        alerts_sent=sub.alerts_sent,
    )


async def send_webhook(sub: Subscription, alerts: list[dict]):
    """Send alert via webhook. Fire-and-forget."""
    if not sub.webhook_url:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body: dict

    if sub.webhook_type == "slack":
        lines = [f"*Agent Health Alert* — `{sub.address}`"]
        for a in alerts:
            lines.append(f"• *{a['type']}*: {a['message']}")
        body = {"text": "\n".join(lines)}
    elif sub.webhook_type == "discord":
        lines = [f"**Agent Health Alert** — `{sub.address}`"]
        for a in alerts:
            lines.append(f"• **{a['type']}**: {a['message']}")
        body = {"content": "\n".join(lines)}
    else:
        body = {
            "address": sub.address,
            "timestamp": now,
            "alerts": alerts,
        }

    try:
        async with httpx_client.AsyncClient(timeout=10.0) as client:
            await client.post(sub.webhook_url, json=body)
    except Exception as e:
        logger.warning("Webhook delivery failed for %s: %s", sub.address, e)


async def check_and_alert(sub: Subscription):
    """Run health check on a subscription and send alerts if thresholds breached."""
    loop = asyncio.get_running_loop()

    try:
        eth_price, transactions = await asyncio.gather(
            loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
            loop.run_in_executor(None, partial(fetch_transactions, sub.address, BASESCAN_API_KEY)),
        )

        health = await loop.run_in_executor(
            None, partial(analyze_address, sub.address, transactions, eth_price),
        )
    except Exception as e:
        logger.warning("Health check failed for %s: %s", sub.address, e)
        return

    sub.last_check_at = datetime.now(timezone.utc)

    if health.total_transactions == 0:
        return

    alerts = []

    if health.health_score < sub.health_score_threshold:
        alerts.append({
            "type": "low_health_score",
            "message": f"Health score dropped to {health.health_score}/100 (threshold: {sub.health_score_threshold})",
            "value": health.health_score,
        })

    failure_rate = 100 - health.success_rate_pct
    if failure_rate > sub.failure_rate_threshold:
        alerts.append({
            "type": "high_failure_rate",
            "message": f"Failure rate is {failure_rate:.1f}% (threshold: {sub.failure_rate_threshold}%)",
            "value": failure_rate,
        })

    if health.estimated_monthly_waste_usd > sub.waste_usd_threshold:
        alerts.append({
            "type": "high_gas_waste",
            "message": f"Estimated gas waste is ${health.estimated_monthly_waste_usd:.2f}/month (threshold: ${sub.waste_usd_threshold:.2f})",
            "value": health.estimated_monthly_waste_usd,
        })

    if alerts:
        await send_webhook(sub, alerts)
        sub.last_alert_at = datetime.now(timezone.utc)
        sub.alerts_sent += len(alerts)


async def alert_monitor_loop():
    """Background loop: check all active subscriptions every 6 hours."""
    while True:
        await asyncio.sleep(6 * 3600)
        now = datetime.now(timezone.utc)
        active = [
            s for s in list(subscriptions.values())
            if now < s.expires_at and s.webhook_url
        ]
        logger.info("Alert monitor: checking %d active subscriptions", len(active))
        for sub in active:
            try:
                await check_and_alert(sub)
            except Exception as e:
                logger.warning("Alert check error for %s: %s", sub.address, e)
            await asyncio.sleep(2)  # rate limit between checks


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(alert_monitor_loop())
    logger.info("Alert monitor background task started")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Agent Health Monitor API",
    description=(
        "Analyzes Base blockchain agent wallets for transaction failures, "
        "gas inefficiency, and nonce issues. Returns actionable health reports. "
        "Payments via x402 protocol (USDC on Base)."
    ),
    version="1.5.0",
    lifespan=lifespan,
)


# -- x402 Payment Middleware -------------------------------------------------


_cdp_logger = logging.getLogger("cdp_auth")


def _build_cdp_jwt(uri: str = "") -> str:
    """Build a CDP-signed JWT for facilitator API authentication."""
    import jwt as pyjwt

    payload = {
        "sub": CDP_API_KEY_ID,
        "iss": "cdp",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
    }
    if uri:
        payload["uri"] = uri
    # Pass PEM string directly to PyJWT — it handles key parsing internally
    token = pyjwt.encode(
        payload, CDP_API_KEY_SECRET, algorithm="ES256",
        headers={"kid": CDP_API_KEY_ID, "nonce": secrets.token_hex()},
    )
    _cdp_logger.info("CDP JWT generated successfully (kid=%s)", CDP_API_KEY_ID)
    return token


def _cdp_create_headers() -> dict[str, dict[str, str]]:
    """Generate CDP auth headers for all facilitator endpoints."""
    token = _build_cdp_jwt()
    auth = {"Authorization": f"Bearer {token}"}
    return {"verify": auth, "settle": auth, "supported": auth}


if CDP_API_KEY_ID and CDP_API_KEY_SECRET:
    _cdp_logger.info("CDP auth enabled (key_id=%s, secret_len=%d)", CDP_API_KEY_ID, len(CDP_API_KEY_SECRET))
    facilitator = HTTPFacilitatorClient({
        "url": FACILITATOR_URL,
        "create_headers": _cdp_create_headers,
    })
else:
    _cdp_logger.info("CDP auth NOT configured — using unauthenticated facilitator")
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
    "GET /alerts/subscribe/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=ALERT_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Subscribe to automated wallet health monitoring. "
            "Checks wallet health every 6 hours and sends webhook alerts "
            "when health score, failure rate, or gas waste thresholds are breached. "
            "$2.00 USDC for 30 days of monitoring."
        ),
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
    "GET /agent/protect/[address]": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=PROTECT_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Protection Agent: Autonomous orchestrator that triages wallet risk "
            "and runs the appropriate combination of health check, gas optimization, "
            "and retry bot services. Returns a unified report with prioritized "
            "actions ranked by potential value recovered."
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
                                "risk_level": "high",
                                "health_score": 58,
                                "services_run": ["health_check", "gas_optimizer", "retry_bot"],
                                "summary": {
                                    "total_issues_found": 15,
                                    "total_potential_savings_usd": 156.00,
                                    "retry_transactions_ready": 8,
                                    "estimated_retry_cost_usd": 2.40,
                                },
                                "recommended_actions": [
                                    {
                                        "priority": 1,
                                        "action": "Execute retry transactions",
                                        "potential_value_usd": 85.00,
                                        "description": "8 failed transactions can be retried with optimized gas",
                                    },
                                ],
                                "analyzed_at": "2026-02-18T12:00:00Z",
                            },
                        },
                    },
                },
            },
        },
    ),
    "GET /retry/[address]": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=RETRY_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "RetryBot: Non-custodial failed transaction retry service. "
            "Analyzes failed transactions, classifies failure reasons, "
            "and returns optimized ready-to-sign replacement transactions "
            "with corrected gas parameters. Agent signs and submits."
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
                                "failed_transactions_analyzed": 50,
                                "retryable_count": 12,
                                "retry_transactions": [
                                    {
                                        "original_tx_hash": "0xabc123...",
                                        "failure_reason": "out_of_gas",
                                        "optimized_transaction": {
                                            "to": "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
                                            "data": "0x3593564c...",
                                            "value": "0x0",
                                            "gas_limit": "0x55730",
                                            "max_fee_per_gas": "0x5f5e100",
                                            "max_priority_fee_per_gas": "0x2faf080",
                                        },
                                        "estimated_gas_cost_usd": 0.12,
                                        "confidence": "high",
                                    },
                                ],
                                "total_estimated_retry_cost_usd": 1.44,
                                "potential_value_recovered_usd": 85.00,
                                "analyzed_at": "2026-02-18T12:00:00Z",
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
        "version": "1.5.0",
        "network": "Base L2",
        "endpoints": {
            "GET /health/{address}": f"{PRICE} USDC — wallet health diagnosis",
            "GET /alerts/subscribe/{address}": f"{ALERT_PRICE} USDC/month — automated monitoring & webhook alerts",
            "GET /optimize/{address}": f"{OPTIMIZE_PRICE} USDC — gas optimization report",
            "GET /retry/{address}": f"{RETRY_PRICE} USDC — optimized retry transactions for recent failures",
            "GET /retry/preview/{address}": "free — preview retryable failure count and estimated savings",
            "GET /agent/protect/{address}": f"{PROTECT_PRICE} USDC — autonomous protection agent (runs all services)",
            "GET /agent/protect/preview/{address}": "free — preview risk level and recommended services",
        },
        "payment": "x402 protocol (USDC on Base)",
        "docs": "/docs",
    }


@app.get("/up")
async def up():
    """Unpaid liveness probe for load balancers."""
    return {"status": "ok"}


@app.get("/debug/config")
async def debug_config():
    """Show non-secret config for debugging x402 setup."""
    cdp_configured = bool(CDP_API_KEY_ID and CDP_API_KEY_SECRET)
    pem_ok = False
    jwt_ok = False
    pem_header = ""
    pem_lines = 0
    pem_load_ok = False
    if cdp_configured:
        pem_ok = CDP_API_KEY_SECRET.startswith("-----BEGIN")
        pem_lines = CDP_API_KEY_SECRET.count("\n")
        # Show the first line (header only, not secret data)
        pem_header = CDP_API_KEY_SECRET.split("\n")[0] if "\n" in CDP_API_KEY_SECRET else CDP_API_KEY_SECRET[:40]
        # Show line lengths to diagnose PEM structure issues
        pem_line_lengths = [len(line) for line in CDP_API_KEY_SECRET.split("\n")]
        # Check for carriage returns or other hidden chars
        has_cr = "\r" in CDP_API_KEY_SECRET
        # Test JWT generation (passes PEM string directly to PyJWT)
        try:
            _build_cdp_jwt()
            jwt_test = True
        except Exception as e:
            jwt_test = f"{type(e).__name__}: {e}"
        # Check library versions
        import sys
        try:
            import cryptography
            crypto_ver = cryptography.__version__
        except Exception:
            crypto_ver = "unknown"
        try:
            import jwt as pyjwt_mod
            pyjwt_ver = pyjwt_mod.__version__
        except Exception:
            pyjwt_ver = "unknown"
        # Try loading key with cryptography directly (more verbose error)
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            pk = load_pem_private_key(CDP_API_KEY_SECRET.encode(), password=None)
            key_info = f"OK: {type(pk).__name__}, key_size={pk.key_size}"
        except Exception as e:
            import traceback
            key_info = traceback.format_exc().split("\n")[-3:]
        try:
            _build_cdp_jwt()
            jwt_ok = True
        except Exception as e:
            jwt_ok = f"{type(e).__name__}: {e}"
    return {
        "version": "1.5.0",
        "network": NETWORK,
        "facilitator_url": FACILITATOR_URL,
        "payment_address": PAYMENT_ADDRESS,
        "cdp_auth": {
            "configured": cdp_configured,
            "key_id": CDP_API_KEY_ID[:8] + "..." if CDP_API_KEY_ID else "",
            "pem_header": pem_header,
            "pem_newline_count": pem_lines,
            "pem_line_lengths": pem_line_lengths if cdp_configured else [],
            "pem_has_cr": has_cr if cdp_configured else False,
            "secret_length": len(CDP_API_KEY_SECRET),
            "jwt_generation": jwt_test if cdp_configured else False,
            "python_version": sys.version,
            "cryptography_version": crypto_ver if cdp_configured else "",
            "pyjwt_version": pyjwt_ver if cdp_configured else "",
            "key_load_detail": key_info if cdp_configured else "",
        },
    }


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


# -- Protection Agent Endpoints -----------------------------------------------

@app.get("/agent/protect/preview/{address}", response_model=ProtectionPreviewResponse)
async def protection_preview(address: str):
    """
    Free preview of protection agent analysis.

    Shows wallet risk level, health score, and which services
    would be run without executing the full analysis.
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

    health = await loop.run_in_executor(
        None, partial(analyze_address, address, transactions, eth_price),
    )

    score = health.health_score
    if score >= 90:
        risk_level = "low"
        services = ["health_check"]
    elif score >= 70:
        risk_level = "medium"
        services = ["health_check", "gas_optimizer"]
    elif score >= 50:
        risk_level = "high"
        services = ["health_check", "gas_optimizer", "retry_bot"]
    else:
        risk_level = "critical"
        services = ["health_check", "gas_optimizer", "retry_bot"]

    estimated_issues = health.failed + health.nonce_gap_count + health.retry_count

    message = (
        f"Risk level: {risk_level.upper()}. Health score: {score}/100. "
        f"{estimated_issues} issue(s) detected. "
        f"Protection agent will run: {', '.join(services)}. "
        f"Pay {PROTECT_PRICE} USDC via GET /agent/protect/{address} "
        f"for full analysis with prioritized action plan."
    )

    return ProtectionPreviewResponse(
        status="ok",
        address=address,
        risk_level=risk_level,
        health_score=score,
        services_recommended=services,
        estimated_issues=estimated_issues,
        message=message,
    )


@app.get("/agent/protect/{address}", response_model=ProtectionResponse)
async def get_protection_report(address: str):
    """
    Autonomous protection agent — full wallet analysis and action plan.

    Requires x402 payment ($25.00 USDC on Base).

    Triages wallet risk and runs the appropriate services:
    - Score 90-100: Low risk — health report + recommend alerts
    - Score 70-89: Medium risk — health + gas optimization
    - Score 50-69: High risk — health + gas optimization + retry bot
    - Score 0-49: Critical — all services, urgent flagging

    Returns a unified report with all results and prioritized actions
    ranked by potential value recovered.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()
    loop = asyncio.get_running_loop()

    eth_price, transactions, is_contract = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(is_contract_address, address)),
    )

    protection = await loop.run_in_executor(
        None, partial(run_protection_agent, address, transactions, eth_price),
    )
    protection.health.is_contract = is_contract

    # Build health report with recommendations
    recommendations = generate_recommendations(protection.health)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    health_report = HealthReport(
        **{k: v for k, v in asdict(protection.health).items()},
        recommendations=recommendations,
        eth_price_usd=eth_price,
        analyzed_at=now_str,
    )

    # Build gas optimization report if available
    gas_report = None
    if protection.gas_optimization:
        opt = protection.gas_optimization
        gas_report = GasOptimizationReport(
            address=opt.address,
            total_transactions_analyzed=opt.total_transactions_analyzed,
            current_monthly_gas_usd=opt.current_monthly_gas_usd,
            optimized_monthly_gas_usd=opt.optimized_monthly_gas_usd,
            estimated_monthly_savings_usd=opt.estimated_monthly_savings_usd,
            total_wasted_gas_eth=opt.total_wasted_gas_eth,
            total_wasted_gas_usd=opt.total_wasted_gas_usd,
            tx_types=[
                TransactionTypeOptimization(
                    contract=t.contract,
                    method_id=t.method_id,
                    method_label=t.method_label,
                    tx_count=t.tx_count,
                    failed_count=t.failed_count,
                    failure_rate_pct=t.failure_rate_pct,
                    current_avg_gas_limit=t.current_avg_gas_limit,
                    current_p50_gas_used=t.current_p50_gas_used,
                    current_p95_gas_used=t.current_p95_gas_used,
                    optimal_gas_limit=t.optimal_gas_limit,
                    gas_limit_reduction_pct=t.gas_limit_reduction_pct,
                    wasted_gas_eth=t.wasted_gas_eth,
                    wasted_gas_usd=t.wasted_gas_usd,
                )
                for t in opt.tx_types
            ],
            recommendations=opt.recommendations,
            eth_price_usd=eth_price,
            analyzed_at=now_str,
        )

    # Build retry transactions list if available
    retry_items = None
    if protection.retry_analysis and protection.retry_analysis.retryable_count > 0:
        retry_items = [
            RetryTransactionItem(
                original_tx_hash=rt.original_tx_hash,
                failure_reason=rt.failure_reason,
                optimized_transaction=OptimizedTransaction(**rt.optimized_transaction),
                estimated_gas_cost_usd=rt.estimated_gas_cost_usd,
                confidence=rt.confidence,
            )
            for rt in protection.retry_analysis.retry_transactions
        ]

    report = ProtectionReport(
        address=protection.address,
        risk_level=protection.risk_level,
        health_score=protection.health_score,
        services_run=protection.services_run,
        summary=ProtectionSummaryModel(
            total_issues_found=protection.summary.total_issues_found,
            total_potential_savings_usd=protection.summary.total_potential_savings_usd,
            retry_transactions_ready=protection.summary.retry_transactions_ready,
            estimated_retry_cost_usd=protection.summary.estimated_retry_cost_usd,
        ),
        health_report=health_report,
        gas_optimization=gas_report,
        retry_transactions=retry_items,
        recommended_actions=[
            ProtectionActionItem(
                priority=a.priority,
                action=a.action,
                description=a.description,
                potential_value_usd=a.potential_value_usd,
                potential_savings_monthly_usd=a.potential_savings_monthly_usd,
            )
            for a in protection.recommended_actions
        ],
        analyzed_at=now_str,
    )

    return ProtectionResponse(status="ok", report=report)


# -- RetryBot Endpoints ------------------------------------------------------

@app.get("/retry/preview/{address}", response_model=RetryPreviewResponse)
async def retry_preview(address: str):
    """
    Free preview of retryable failed transactions.

    Shows count of retryable failures and estimated savings
    without returning the full optimized transactions.
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

    analysis = await loop.run_in_executor(
        None, partial(analyze_retryable_transactions, address, transactions, eth_price),
    )

    message = (
        f"Found {analysis.retryable_count} retryable failures out of "
        f"{analysis.failed_transactions_analyzed} failed transactions. "
        f"Pay {RETRY_PRICE} USDC via GET /retry/{address} to get optimized "
        f"ready-to-sign retry transactions."
        if analysis.retryable_count > 0
        else f"Analyzed {analysis.failed_transactions_analyzed} failed transactions. "
        f"No retryable failures found."
    )

    return RetryPreviewResponse(
        status="ok",
        address=address,
        failed_transactions_analyzed=analysis.failed_transactions_analyzed,
        retryable_count=analysis.retryable_count,
        total_estimated_retry_cost_usd=analysis.total_estimated_retry_cost_usd,
        potential_value_recovered_usd=analysis.potential_value_recovered_usd,
        message=message,
    )


@app.get("/retry/{address}", response_model=RetryResponse)
async def get_retry_transactions(address: str):
    """
    Analyze failed transactions and return optimized retry transactions.

    Requires x402 payment ($10.00 USDC on Base).

    Non-custodial: returns ready-to-sign transaction objects.
    The agent signs and submits the transactions themselves.

    - Fetches failed transactions from Blockscout
    - Classifies failure reasons: out_of_gas, reverted, nonce_conflict, slippage
    - Filters to retryable failures only
    - Builds optimized replacements with corrected gas parameters
    - Returns ready-to-sign EIP-1559 transaction objects
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

    analysis = await loop.run_in_executor(
        None, partial(analyze_retryable_transactions, address, transactions, eth_price),
    )

    report = RetryReport(
        address=analysis.address,
        failed_transactions_analyzed=analysis.failed_transactions_analyzed,
        retryable_count=analysis.retryable_count,
        retry_transactions=[
            RetryTransactionItem(
                original_tx_hash=rt.original_tx_hash,
                failure_reason=rt.failure_reason,
                optimized_transaction=OptimizedTransaction(**rt.optimized_transaction),
                estimated_gas_cost_usd=rt.estimated_gas_cost_usd,
                confidence=rt.confidence,
            )
            for rt in analysis.retry_transactions
        ],
        total_estimated_retry_cost_usd=analysis.total_estimated_retry_cost_usd,
        potential_value_recovered_usd=analysis.potential_value_recovered_usd,
        analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    return RetryResponse(status="ok", report=report)


# -- Alert Endpoints ---------------------------------------------------------

@app.get("/alerts/subscribe/{address}")
async def subscribe_alerts(address: str):
    """
    Subscribe a wallet to automated health monitoring (30 days).

    Requires x402 payment ($2.00 USDC on Base).

    After payment, configure your webhook via POST /alerts/configure.
    Health checks run every 6 hours and send alerts when thresholds are breached.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()
    now = datetime.now(timezone.utc)

    existing = subscriptions.get(address)
    if existing and now < existing.expires_at:
        # Renew: extend 30 days from current expiry
        existing.expires_at += timedelta(days=30)
        sub = existing
    else:
        sub = Subscription(address=address)
        subscriptions[address] = sub

    return {
        "status": "ok",
        "message": "Subscription active. Configure your webhook via POST /alerts/configure.",
        "subscription": _sub_to_status(sub),
    }


@app.post("/alerts/configure")
async def configure_alerts(req: ConfigureRequest):
    """
    Configure webhook and thresholds for an active alert subscription.

    No payment required — must have an active subscription from GET /alerts/subscribe/{address}.
    """
    address = req.address.lower()
    if not ADDRESS_RE.match(req.address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {req.address}")

    sub = subscriptions.get(address)
    if not sub:
        raise HTTPException(status_code=404, detail="No subscription found. Subscribe first via GET /alerts/subscribe/{address}.")

    now = datetime.now(timezone.utc)
    if now >= sub.expires_at:
        raise HTTPException(status_code=410, detail="Subscription expired. Renew via GET /alerts/subscribe/{address}.")

    if req.webhook_type not in ("generic", "slack", "discord"):
        raise HTTPException(status_code=400, detail="webhook_type must be 'generic', 'slack', or 'discord'.")

    sub.webhook_url = req.webhook_url
    sub.webhook_type = req.webhook_type

    if req.thresholds:
        if req.thresholds.health_score is not None:
            sub.health_score_threshold = req.thresholds.health_score
        if req.thresholds.failure_rate is not None:
            sub.failure_rate_threshold = req.thresholds.failure_rate
        if req.thresholds.waste_usd is not None:
            sub.waste_usd_threshold = req.thresholds.waste_usd

    return {
        "status": "ok",
        "message": "Webhook configured. Alerts will be sent when thresholds are breached.",
        "subscription": _sub_to_status(sub),
    }


@app.get("/alerts/status/{address}")
async def alert_status(address: str):
    """Check the status of an alert subscription."""
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()
    sub = subscriptions.get(address)
    if not sub:
        raise HTTPException(status_code=404, detail="No subscription found for this address.")

    return {"status": "ok", "subscription": _sub_to_status(sub)}


@app.delete("/alerts/unsubscribe/{address}")
async def unsubscribe_alerts(address: str):
    """Remove an alert subscription."""
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()
    if address not in subscriptions:
        raise HTTPException(status_code=404, detail="No subscription found for this address.")

    del subscriptions[address]
    return {"status": "ok", "message": f"Subscription removed for {address}."}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
