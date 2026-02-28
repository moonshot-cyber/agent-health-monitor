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
    INTERNAL_API_KEY - Secret key for ACP bridge bypass (optional, auto-generated if not set)
"""

import asyncio
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Optional

import httpx as httpx_client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme, register_exact_evm_client
from x402.server import x402ResourceServer

from x402 import x402Client as x402PayerClient
from x402.mechanisms.evm.signers import EthAccountSigner
from x402.http.clients.httpx import x402HttpxClient
from eth_account import Account as EthAccount

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

FACILITATOR_URL = os.getenv("FACILITATOR_URL", "https://facilitator.payai.network")
BASESCAN_API_KEY = os.getenv("BASESCAN_API_KEY", "")
PRICE = os.getenv("PRICE_USD", "$0.50")
OPTIMIZE_PRICE = os.getenv("OPTIMIZE_PRICE_USD", "$5.00")
ALERT_PRICE = os.getenv("ALERT_PRICE_USD", "$2.00")
RETRY_PRICE = os.getenv("RETRY_PRICE_USD", "$10.00")
PROTECT_PRICE = os.getenv("PROTECT_PRICE_USD", "$25.00")
RISK_PRICE = os.getenv("RISK_PRICE_USD", "$0.001")
PREMIUM_RISK_PRICE = os.getenv("PREMIUM_RISK_PRICE_USD", "$0.05")
NANSEN_PAYER_PRIVATE_KEY = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
NETWORK = os.getenv("NETWORK", "eip155:8453")  # Base mainnet
VALID_COUPONS = set(c.strip().upper() for c in os.getenv("VALID_COUPONS", "").split(",") if c.strip())
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PORT = int(os.getenv("PORT", "4021"))

# -- Internal API Key (for ACP bridge bypass) --------------------------------
# If not set, generate a secure random key at startup
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    INTERNAL_API_KEY = secrets.token_urlsafe(32)
    logging.warning(
        "INTERNAL_API_KEY not set. Generated random key: %s. "
        "Set INTERNAL_API_KEY in .env to persist across restarts.",
        INTERNAL_API_KEY
    )

ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# -- Nansen x402 Client (for premium risk endpoint) -------------------------
NANSEN_API_URL = "https://nansen.api.corbits.dev/api/beta/profiler/address/labels"
NANSEN_BALANCES_URL = "https://nansen.api.corbits.dev/api/beta/profiler/address/balances"
_nansen_x402_client = None
if NANSEN_PAYER_PRIVATE_KEY:
    try:
        _nansen_account = EthAccount.from_key(NANSEN_PAYER_PRIVATE_KEY)
        _nansen_signer = EthAccountSigner(_nansen_account)
        _nansen_x402_client = x402PayerClient()
        register_exact_evm_client(_nansen_x402_client, _nansen_signer, networks=[NETWORK])
        logging.info("Nansen x402 client initialized (payer: %s)", _nansen_account.address)
    except Exception as e:
        logging.warning("Failed to initialize Nansen x402 client: %s", e)


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


class NansenLabel(BaseModel):
    label: str
    category: Optional[str] = None
    definition: Optional[str] = None


class TokenBalance(BaseModel):
    chain: str
    symbol: str
    name: str
    amount: float
    usd_value: float


class HealthResponse(BaseModel):
    status: str
    report: HealthReport
    nansen_labels: list[NansenLabel] = []
    nansen_available: bool = False
    token_balances: list[TokenBalance] = []
    total_portfolio_usd: float = 0.0


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


# -- Risk Score Model --------------------------------------------------------

class RiskResponse(BaseModel):
    risk_score: int
    risk_level: str
    verdict: str


class PremiumRiskResponse(BaseModel):
    risk_score: int
    risk_level: str
    verdict: str
    nansen_labels: list[NansenLabel]
    nansen_available: bool


# -- Nansen Helper -----------------------------------------------------------

async def fetch_nansen_labels(address: str) -> list[dict] | None:
    """Fetch wallet labels from Nansen via x402-paid API call."""
    if not _nansen_x402_client:
        logging.debug("Nansen labels skipped: x402 client not initialized")
        return None
    body = {
        "parameters": {"chain": "all", "address": address},
        "pagination": {"page": 1, "recordsPerPage": 100},
    }
    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(30.0)) as http:
            response = await http.post(NANSEN_API_URL, json=body)
        logging.info("Nansen labels response: status=%s length=%s", response.status_code, len(response.content))
        if response.status_code == 200:
            data = response.json()
            result = data if isinstance(data, list) else data.get("labels", data.get("data", []))
            logging.info("Nansen labels parsed: %d items", len(result) if isinstance(result, list) else 0)
            return result
        logging.warning("Nansen labels non-200: status=%s body=%s", response.status_code, response.text[:500])
        return None
    except Exception as e:
        logging.warning("Nansen labels call failed: %s", e, exc_info=True)
        return None


async def fetch_nansen_balances(address: str) -> list[dict] | None:
    """Fetch token balances from Nansen via x402-paid API call."""
    if not _nansen_x402_client:
        logging.debug("Nansen balances skipped: x402 client not initialized")
        return None
    body = {
        "parameters": {
            "chain": "all",
            "walletAddresses": [address],
            "suspiciousFilter": "on",
        },
        "pagination": {"page": 1, "recordsPerPage": 100},
    }
    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(30.0)) as http:
            response = await http.post(NANSEN_BALANCES_URL, json=body)
        logging.info("Nansen balances response: status=%s length=%s", response.status_code, len(response.content))
        if response.status_code == 200:
            data = response.json()
            result = data if isinstance(data, list) else data.get("balances", data.get("data", []))
            logging.info("Nansen balances parsed: %d items", len(result) if isinstance(result, list) else 0)
            return result
        logging.warning("Nansen balances non-200: status=%s body=%s", response.status_code, response.text[:500])
        return None
    except Exception as e:
        logging.warning("Nansen balances call failed: %s", e, exc_info=True)
        return None


async def fetch_nansen_enrichment(address: str) -> tuple[list[dict] | None, list[dict] | None]:
    """Fetch labels AND balances through a single x402 session.

    Running two x402-paid calls concurrently from the same payer client
    causes nonce conflicts (both try to sign a payment simultaneously).
    This function serialises the two calls through one HTTP session so
    only one x402 payment flow is active at a time.
    """
    if not _nansen_x402_client:
        logging.debug("Nansen enrichment skipped: x402 client not initialized")
        return None, None

    labels_result: list[dict] | None = None
    balances_result: list[dict] | None = None

    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(30.0)) as http:
            # --- Labels ---
            try:
                labels_body = {
                    "parameters": {"chain": "all", "address": address},
                    "pagination": {"page": 1, "recordsPerPage": 100},
                }
                resp = await http.post(NANSEN_API_URL, json=labels_body)
                logging.info("Nansen labels response: status=%s length=%s", resp.status_code, len(resp.content))
                if resp.status_code == 200:
                    data = resp.json()
                    labels_result = data if isinstance(data, list) else data.get("labels", data.get("data", []))
                    logging.info("Nansen labels parsed: %d items", len(labels_result) if isinstance(labels_result, list) else 0)
                else:
                    logging.warning("Nansen labels non-200: status=%s body=%s", resp.status_code, resp.text[:500])
            except Exception as e:
                logging.warning("Nansen labels call failed: %s", e, exc_info=True)

            # --- Balances ---
            try:
                balances_body = {
                    "parameters": {
                        "chain": "all",
                        "walletAddresses": [address],
                        "suspiciousFilter": "on",
                    },
                    "pagination": {"page": 1, "recordsPerPage": 100},
                }
                resp = await http.post(NANSEN_BALANCES_URL, json=balances_body)
                logging.info("Nansen balances response: status=%s length=%s", resp.status_code, len(resp.content))
                if resp.status_code == 200:
                    data = resp.json()
                    balances_result = data if isinstance(data, list) else data.get("balances", data.get("data", []))
                    logging.info("Nansen balances parsed: %d items", len(balances_result) if isinstance(balances_result, list) else 0)
                else:
                    logging.warning("Nansen balances non-200: status=%s body=%s", resp.status_code, resp.text[:500])
            except Exception as e:
                logging.warning("Nansen balances call failed: %s", e, exc_info=True)

    except Exception as e:
        logging.warning("Nansen enrichment session failed: %s", e, exc_info=True)

    return labels_result, balances_result


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


# -- FastAPI App with Lifecycle & Custom Middleware --------------------------

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
    version="1.6.0",
    lifespan=lifespan,
)


# -- Custom Middleware to Bypass x402 on Internal Calls ----------------------

class InternalKeyBypassMiddleware:
    """
    ASGI middleware that skips x402 payment when X-Internal-Key header is valid.

    Must be added AFTER PaymentMiddlewareASGI so it wraps the outside
    (last added = outermost in Starlette's middleware stack).

    When the key matches, we call self.app.app — the inner app that
    PaymentMiddlewareASGI wraps — skipping the payment check entirely.
    """
    def __init__(self, app):
        self.app = app  # This is PaymentMiddlewareASGI

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check for internal API key in headers
        headers = dict(scope.get("headers", []))
        internal_key = headers.get(b"x-internal-key", b"").decode()

        if internal_key and internal_key == INTERNAL_API_KEY:
            # Valid key — skip x402 by jumping past PaymentMiddlewareASGI
            # to the inner app it wraps (self.app.app)
            inner_app = getattr(self.app, "app", self.app)
            await inner_app(scope, receive, send)
        else:
            # No valid key — normal flow through x402
            await self.app(scope, receive, send)


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
                        "discoverable": True,
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
        extensions={
            "bazaar": {
                "info": {
                    "input": {
                        "type": "http",
                        "method": "GET",
                        "discoverable": True,
                        "queryParams": {
                            "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                        },
                    },
                    "output": {
                        "type": "json",
                        "example": {
                            "status": "ok",
                            "message": "Subscription active. Configure your webhook via POST /alerts/configure.",
                            "subscription": {
                                "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                                "active": True,
                                "webhook_configured": False,
                                "webhook_type": "generic",
                                "expires_at": "2026-03-22T12:00:00Z",
                                "alerts_sent": 0,
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
                        "discoverable": True,
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
                        "discoverable": True,
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
    "GET /risk/premium/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=PREMIUM_RISK_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Premium risk score enriched with Nansen wallet intelligence labels. "
            "Includes smart money tags, entity identification, and behavioral signals."
        ),
    ),
    "GET /risk/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=RISK_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Quick risk score for agent pre-flight checks. "
            "Returns a 0-100 risk score and one-line verdict. "
            "Designed for high-volume, low-latency use."
        ),
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
                        "discoverable": True,
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

# Add internal key bypass AFTER x402 — last added = outermost = runs first.
# When valid X-Internal-Key is present, skips x402 entirely.
app.add_middleware(InternalKeyBypassMiddleware)


# -- Routes ------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the landing page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/.well-known/x402")
async def x402_discovery(request: Request):
    """
    x402 discovery document — lists all paid endpoints with payment
    requirements and bazaar metadata so crawlers like x402scan can
    auto-register the service.
    """
    base_url = str(request.base_url).rstrip("/")
    # Railway terminates TLS at the proxy — ensure discovery URLs use https
    if base_url.startswith("http://") and "railway.app" in base_url:
        base_url = "https://" + base_url[7:]
    endpoints = []

    for route_pattern, config in x402_routes.items():
        parts = route_pattern.split(" ", 1)
        method = parts[0] if len(parts) == 2 else "GET"
        path = parts[1] if len(parts) == 2 else parts[0]

        accepts = config.accepts if isinstance(config.accepts, list) else [config.accepts]
        payment_options = []
        for opt in accepts:
            payment_options.append({
                "scheme": opt.scheme,
                "network": opt.network,
                "pay_to": opt.pay_to,
                "price": opt.price,
            })

        endpoint_entry = {
            "url": f"{base_url}{path}",
            "method": method,
            "description": config.description,
            "mime_type": config.mime_type,
            "payment": payment_options,
        }

        if config.extensions and "bazaar" in config.extensions:
            endpoint_entry["bazaar"] = config.extensions["bazaar"]

        endpoints.append(endpoint_entry)

    return {
        "x402": True,
        "version": "1.0",
        "service": "Agent Health Monitor",
        "description": (
            "Pay-per-use API that analyzes Base blockchain agent wallets "
            "for transaction failures, gas waste, and optimization opportunities."
        ),
        "facilitator": FACILITATOR_URL,
        "endpoints": endpoints,
    }


@app.get("/api/info")
async def api_info():
    """Service info and pricing."""
    return {
        "service": "Agent Health Monitor",
        "version": "1.7.0",
        "network": "Base L2",
        "endpoints": {
            "GET /risk/{address}": f"{RISK_PRICE} USDC — quick risk score for pre-flight checks",
            "GET /risk/premium/{address}": f"{PREMIUM_RISK_PRICE} USDC — premium risk score with Nansen wallet labels",
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


@app.get("/coupon/validate/{code}")
async def validate_coupon(code: str):
    """Check if a coupon code is valid."""
    return {"valid": code.strip().upper() in VALID_COUPONS}


def _require_coupon(code: str):
    """Validate a coupon code or raise 403."""
    if code.strip().upper() not in VALID_COUPONS:
        raise HTTPException(status_code=403, detail="Invalid coupon code")


@app.get("/coupon/health/{code}/{address}")
async def coupon_health(code: str, address: str):
    _require_coupon(code)
    return await get_health_report(address)


@app.get("/coupon/optimize/{code}/{address}")
async def coupon_optimize(code: str, address: str):
    _require_coupon(code)
    return await get_optimization_report(address)


@app.get("/coupon/retry/{code}/{address}")
async def coupon_retry(code: str, address: str):
    _require_coupon(code)
    return await get_retry_transactions(address)


@app.get("/coupon/protect/{code}/{address}")
async def coupon_protect(code: str, address: str):
    _require_coupon(code)
    return await get_protection_report(address)


@app.get("/coupon/alerts/{code}/{address}")
async def coupon_alerts(code: str, address: str):
    _require_coupon(code)
    return await subscribe_alerts(address)


@app.get("/coupon/risk-premium/{code}/{address}")
async def coupon_premium_risk(code: str, address: str):
    _require_coupon(code)
    return await get_premium_risk_score(address)


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


# Chat rate limiting: 10 messages per IP per hour
_chat_rate: dict[str, list[float]] = {}
CHAT_RATE_LIMIT = 10
CHAT_RATE_WINDOW = 3600  # seconds


def _check_chat_rate(ip: str):
    import time
    now = time.time()
    timestamps = _chat_rate.get(ip, [])
    timestamps = [t for t in timestamps if now - t < CHAT_RATE_WINDOW]
    if len(timestamps) >= CHAT_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    timestamps.append(now)
    _chat_rate[ip] = timestamps


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """AI chat about wallet analysis results."""
    _check_chat_rate(request.client.host)
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Chat not configured")

    import anthropic

    user_content = req.message
    if req.context:
        user_content = (
            f"The user ran a wallet health analysis. Results:\n"
            f"{__import__('json').dumps(req.context, indent=2)}\n\n"
            f"User question: {req.message}"
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=(
            "You are a blockchain analyst assistant for Agent Health Monitor, "
            "a pay-per-call wallet intelligence service on Base mainnet. Be concise, technical, and helpful.\n\n"
            "Available services you should recommend based on the user's results:\n"
            "- /health ($0.50 USDC) — wallet health score with risk analysis\n"
            "- /optimize ($5.00 USDC) — gas optimization report with per-transaction-type savings\n"
            "- /retry ($10.00 USDC) — analyzes failed transactions and returns ready-to-sign retry transactions\n"
            "- /alerts ($2.00 USDC) — subscribe to automated health monitoring with webhook alerts\n"
            "- /agent/protect ($25.00 USDC) — full autonomous protection suite (runs all services)\n\n"
            "When analyzing results, recommend the most relevant paid endpoint as a next step:\n"
            "- High gas waste or low efficiency → recommend /optimize ($5)\n"
            "- Many reverted/failed transactions → recommend /retry ($10)\n"
            "- Score below 70 or multiple issues → recommend /agent/protect ($25)\n"
            "- Ongoing risk or volatile wallet → recommend /alerts ($2)\n\n"
            "End every response with a clear call to action referencing the specific endpoint and its cost. "
            "Example: 'Run /optimize ($5 USDC) to get a detailed gas savings breakdown per transaction type.'"
        ),
        messages=[{"role": "user", "content": user_content}],
    )

    reply = response.content[0].text if response.content else "Unable to get response."
    return {"reply": reply}


@app.get("/up")
async def up():
    """Unpaid liveness probe for load balancers."""
    return {"status": "ok"}


@app.get("/risk/premium/{address}", response_model=PremiumRiskResponse)
async def get_premium_risk_score(address: str):
    """
    Premium risk score enriched with Nansen wallet intelligence labels.

    Requires x402 payment ($0.05 USDC on Base) OR valid X-Internal-Key header.

    Returns a 0-100 risk score plus Nansen smart money tags,
    entity labels, and behavioral signals for the wallet.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()
    loop = asyncio.get_running_loop()

    # Run risk analysis + Nansen fetch in parallel
    risk_task = asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
    )
    nansen_task = fetch_nansen_labels(address)

    (eth_price, transactions), nansen_raw = await asyncio.gather(risk_task, nansen_task)

    health = await loop.run_in_executor(
        None, partial(analyze_address, address, transactions, eth_price),
    )

    # Derive risk score (same logic as /risk)
    risk_score = max(0, min(100, 100 - int(health.health_score)))

    if risk_score >= 75:
        risk_level = "CRITICAL"
    elif risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Build verdict
    failure_rate = 100 - health.success_rate_pct
    signals = []
    if failure_rate > 20:
        signals.append("high failure rate")
    elif failure_rate > 5:
        signals.append("elevated failure rate")
    if health.out_of_gas_count > 0:
        signals.append("out-of-gas errors")
    if health.nonce_gap_count > 0:
        signals.append("nonce gaps")
    if health.reverted_count > 5:
        signals.append("suspicious contract interactions")
    elif health.reverted_count > 0:
        signals.append("contract reverts")
    if health.estimated_monthly_waste_usd > 50:
        signals.append("significant gas waste")

    if health.total_transactions == 0:
        verdict = "No transaction history found"
    elif not signals:
        verdict = "Wallet operating normally with no significant issues detected"
    else:
        verdict = f"{risk_level.capitalize()} - {' with '.join(signals)} detected"

    # Format Nansen labels
    nansen_labels = []
    nansen_available = nansen_raw is not None
    if nansen_raw:
        for item in nansen_raw:
            if isinstance(item, dict):
                nansen_labels.append(NansenLabel(
                    label=item.get("label", item.get("name", str(item))).replace("\u200b", "").replace("\u00e2\u20ac\u2039", "").replace("\ufeff", "").replace("\u200c", "").replace("\u200d", "").strip(),
                    category=item.get("category"),
                    definition=item.get("definition"),
                ))

    return PremiumRiskResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        verdict=verdict,
        nansen_labels=nansen_labels,
        nansen_available=nansen_available,
    )


@app.get("/risk/{address}", response_model=RiskResponse)
async def get_risk_score(address: str):
    """
    Quick risk score for agent pre-flight checks.

    Requires x402 payment ($0.001 USDC on Base) OR valid X-Internal-Key header.

    Returns a 0-100 risk score derived from transaction failure rate,
    wallet age, and contract interaction patterns. Designed for
    high-volume, low-latency use.
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

    # Derive risk score (inverse of health: 100 = max risk, 0 = no risk)
    risk_score = max(0, min(100, 100 - int(health.health_score)))

    # Classify risk level
    if risk_score >= 75:
        risk_level = "CRITICAL"
    elif risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Build one-line verdict from dominant signals
    failure_rate = 100 - health.success_rate_pct
    signals = []
    if failure_rate > 20:
        signals.append("high failure rate")
    elif failure_rate > 5:
        signals.append("elevated failure rate")
    if health.out_of_gas_count > 0:
        signals.append("out-of-gas errors")
    if health.nonce_gap_count > 0:
        signals.append("nonce gaps")
    if health.reverted_count > 5:
        signals.append("suspicious contract interactions")
    elif health.reverted_count > 0:
        signals.append("contract reverts")
    if health.estimated_monthly_waste_usd > 50:
        signals.append("significant gas waste")

    if health.total_transactions == 0:
        verdict = "No transaction history found"
    elif not signals:
        verdict = "Wallet operating normally with no significant issues detected"
    else:
        verdict = f"{risk_level.capitalize()} - {' with '.join(signals)} detected"

    return RiskResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        verdict=verdict,
    )


@app.get("/health/{address}", response_model=HealthResponse)
async def get_health_report(address: str):
    """
    Analyze a Base wallet address and return a health report.

    Requires x402 payment ($0.50 USDC on Base) OR valid X-Internal-Key header.

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

    # Run blocking I/O and Nansen enrichment in parallel.
    # Nansen labels + balances share a single x402 session (serialised)
    # to avoid payer-nonce conflicts from concurrent x402 payment flows.
    loop = asyncio.get_running_loop()
    (eth_price, transactions, is_contract), (nansen_raw, balances_raw) = await asyncio.gather(
        asyncio.gather(
            loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
            loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
            loop.run_in_executor(None, partial(is_contract_address, address)),
        ),
        fetch_nansen_enrichment(address),
    )

    logging.info(
        "/health [%s] nansen_raw=%s balances_raw=%s",
        address,
        type(nansen_raw).__name__ if not isinstance(nansen_raw, list) else f"list[{len(nansen_raw)}]",
        type(balances_raw).__name__ if not isinstance(balances_raw, list) else f"list[{len(balances_raw)}]",
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

    # Format Nansen labels
    nansen_labels = []
    nansen_available = nansen_raw is not None
    if nansen_raw:
        for item in nansen_raw:
            if isinstance(item, dict):
                nansen_labels.append(NansenLabel(
                    label=item.get("label", item.get("name", str(item))).replace("\u200b", "").replace("\u00e2\u20ac\u2039", "").replace("\ufeff", "").replace("\u200c", "").replace("\u200d", "").strip(),
                    category=item.get("category"),
                    definition=item.get("definition"),
                ))

    # Format token balances
    token_balances = []
    if balances_raw:
        for item in balances_raw:
            if isinstance(item, dict):
                token_balances.append(TokenBalance(
                    chain=item.get("chain", ""),
                    symbol=item.get("symbol", ""),
                    name=item.get("name", ""),
                    amount=float(item.get("tokenAmount", 0)),
                    usd_value=float(item.get("usdValue", 0)),
                ))
    total_portfolio_usd = sum(tb.usd_value for tb in token_balances)

    logging.info(
        "/health [%s] result: nansen_available=%s labels=%d balances=%d portfolio=$%.2f",
        address, nansen_available, len(nansen_labels), len(token_balances), total_portfolio_usd,
    )

    return HealthResponse(
        status="ok",
        report=report,
        nansen_labels=nansen_labels,
        nansen_available=nansen_available,
        token_balances=token_balances,
        total_portfolio_usd=total_portfolio_usd,
    )


@app.get("/optimize/{address}", response_model=OptimizeResponse)
async def get_optimization_report(address: str):
    """
    Analyze a Base wallet and return a gas optimization report.

    Requires x402 payment ($5.00 USDC on Base) OR valid X-Internal-Key header.

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

    Requires x402 payment ($25.00 USDC on Base) OR valid X-Internal-Key header.

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

    Requires x402 payment ($10.00 USDC on Base) OR valid X-Internal-Key header.

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
