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
    STRIPE_SECRET_KEY   - Stripe secret key for fiat API key system (optional)
    STRIPE_WEBHOOK_SECRET - Stripe webhook signing secret (optional)
"""

import asyncio
import hmac
import logging
import os
import re
import secrets
import time

# Configure logging at module level so it works under both
# `python api.py` and `uvicorn api:app` (Railway deploy path).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

import db as scan_db
import erc8183_worker
from generate_report_card import generate_report_card
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from ipaddress import ip_address
import pathlib
from typing import Annotated, Optional
from urllib.parse import quote as url_quote, urlparse

import hashlib

import httpx as httpx_client
import jwt
import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
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
    analyze_wash,
    calculate_ahs,
    fetch_tokens_v2,
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
COUNTERPARTY_PRICE = os.getenv("COUNTERPARTY_PRICE_USD", "$0.10")
NETWORK_MAP_PRICE = os.getenv("NETWORK_MAP_PRICE_USD", "$0.10")
WASH_PRICE = os.getenv("WASH_PRICE_USD", "$0.50")
AHS_PRICE = os.getenv("AHS_PRICE_USD", "$1.00")
AHS_BATCH_PRICE = os.getenv("AHS_BATCH_PRICE_USD", "$10.00")
ROUTE_PRICE = os.getenv("ROUTE_PRICE_USD", "$0.01")
REPORT_CARD_PRICE = os.getenv("REPORT_CARD_PRICE_USD", "$2.00")
AHS_JWT_SECRET = os.getenv("AHS_JWT_SECRET", secrets.token_urlsafe(32))
NANSEN_PAYER_PRIVATE_KEY = os.getenv("NANSEN_PAYER_PRIVATE_KEY", "")
ENDPOINT_COUNT = 14  # single source of truth — update here when adding endpoints
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
        "INTERNAL_API_KEY not set. Generated random key (not shown in logs). "
        "Set INTERNAL_API_KEY in .env to persist across restarts."
    )

# -- Stripe Configuration ---------------------------------------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Reusable Annotated type for {address} path parameters across all endpoints.
# Using Annotated keeps address as a required positional arg (no default value),
# so direct calls like `await get_risk_score(address)` from coupon endpoints still work.
WalletAddress = Annotated[str, Path(
    description="Ethereum wallet address (0x-prefixed, 40 hex chars)",
    examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"],
)]


# -- Client IP Extraction ---------------------------------------------------

def get_client_ip(request: Request) -> str:
    """Extract real client IP. Railway's Envoy proxy sets X-Envoy-External-Address."""
    return (
        request.headers.get("X-Envoy-External-Address")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )


# -- Rate Limiter (slowapi) -------------------------------------------------

limiter = Limiter(key_func=get_client_ip)

# -- Nansen API Routing ------------------------------------------------------
# Labels & balances: only available via Corbits x402 proxy (Nansen returns 401
# for these endpoints when called directly — no native x402 support yet).
# Counterparties & pnl-summary: call Nansen DIRECTLY — the Corbits proxy 500s
# on these, but api.nansen.ai serves them natively via x402.
# NOTE: Direct Nansen rate limits are stricter (5 req/s, 60 req/min) vs
#       Corbits (20 req/s, 300 req/min).  Keep calls sequential.
NANSEN_CORBITS_URL = "https://nansen.api.corbits.dev/api/beta"
NANSEN_DIRECT_URL = "https://api.nansen.ai/api/v1"

NANSEN_API_URL = NANSEN_CORBITS_URL + "/profiler/address/labels"
NANSEN_BALANCES_URL = NANSEN_CORBITS_URL + "/profiler/address/balances"
NANSEN_COUNTERPARTIES_URL = NANSEN_DIRECT_URL + "/profiler/address/counterparties"
NANSEN_PNL_SUMMARY_URL = NANSEN_DIRECT_URL + "/profiler/address/pnl-summary"
NANSEN_RELATED_WALLETS_URL = NANSEN_DIRECT_URL + "/profiler/address/related-wallets"
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
    category: str = Field(description="Issue category (gas, nonce, failure, balance)", examples=["gas"])
    severity: str = Field(description="Severity level: critical, high, medium, or info", examples=["high"])
    message: str = Field(description="Actionable recommendation text", examples=["Gas waste detected — 12% of gas spent on reverted transactions"])


class HealthReport(BaseModel):
    address: str = Field(description="Wallet address analysed", examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"])
    is_contract: bool = Field(description="Whether the address is a smart contract", examples=[False])
    health_score: float = Field(description="Composite health score 0-100", examples=[74.5])
    optimization_priority: str = Field(description="Priority level: low, medium, high, critical", examples=["medium"])
    total_transactions: int = Field(description="Total transactions in analysis window", examples=[1542])
    successful: int = Field(description="Number of successful transactions", examples=[1480])
    failed: int = Field(description="Number of failed transactions", examples=[62])
    success_rate_pct: float = Field(description="Transaction success rate as percentage", examples=[95.98])
    total_gas_spent_eth: float = Field(description="Total gas spent in ETH", examples=[0.284])
    wasted_gas_eth: float = Field(description="Gas wasted on failed transactions in ETH", examples=[0.019])
    estimated_monthly_waste_usd: float = Field(description="Projected monthly USD waste at current rate", examples=[14.30])
    avg_gas_efficiency_pct: float = Field(description="Average gas usage vs gas limit percentage", examples=[67.3])
    out_of_gas_count: int = Field(description="Transactions that failed due to out-of-gas", examples=[3])
    reverted_count: int = Field(description="Transactions that reverted", examples=[59])
    nonce_gap_count: int = Field(description="Detected nonce gaps", examples=[0])
    retry_count: int = Field(description="Duplicate-nonce retry transactions detected", examples=[2])
    top_failure_type: str = Field(description="Most common failure reason", examples=["execution reverted"])
    first_seen: str = Field(description="ISO 8601 timestamp of earliest transaction", examples=["2024-01-15T08:30:00Z"])
    last_seen: str = Field(description="ISO 8601 timestamp of most recent transaction", examples=["2025-04-05T14:22:00Z"])
    recommendations: list[Recommendation] = Field(description="Prioritised list of actionable recommendations")
    eth_price_usd: float = Field(description="ETH/USD price used for calculations", examples=[3245.50])
    analyzed_at: str = Field(description="ISO 8601 timestamp of this analysis", examples=["2025-04-06T10:00:00Z"])


class NansenLabel(BaseModel):
    label: str = Field(description="Nansen wallet label", examples=["Smart Money"])
    category: Optional[str] = Field(default=None, description="Label category", examples=["behavioral"])
    definition: Optional[str] = Field(default=None, description="Human-readable label definition", examples=["Wallet that consistently trades profitably"])


class TokenBalance(BaseModel):
    chain: str = Field(description="Chain identifier", examples=["base"])
    symbol: str = Field(description="Token ticker symbol", examples=["USDC"])
    name: str = Field(description="Full token name", examples=["USD Coin"])
    amount: float = Field(description="Token balance amount", examples=[1250.75])
    usd_value: float = Field(description="USD value of the balance", examples=[1250.75])


class Counterparty(BaseModel):
    address: str
    label: Optional[str] = None
    interaction_count: int = 0
    volume_usd: float = 0.0
    last_interaction: Optional[str] = None


class CounterpartyResponse(BaseModel):
    status: str
    address: str
    counterparties: list[Counterparty] = []
    total_counterparties: int = 0
    nansen_available: bool = False


class RelatedWallet(BaseModel):
    address: str
    label: Optional[str] = None
    relation: str
    chain: str
    transaction_hash: Optional[str] = None
    block_timestamp: Optional[str] = None


class RelatedWalletsResponse(BaseModel):
    status: str
    address: str
    chain: str
    related_wallets: list[RelatedWallet] = []
    total_related: int = 0
    nansen_available: bool = False


class HealthResponse(BaseModel):
    status: str = Field(description="Response status", examples=["ok"])
    report: HealthReport = Field(description="Full health analysis report")
    nansen_labels: list[NansenLabel] = Field(default=[], description="Nansen wallet intelligence labels, if available")
    nansen_available: bool = Field(default=False, description="Whether Nansen enrichment was successful")
    token_balances: list[TokenBalance] = Field(default=[], description="Token balances across chains")
    total_portfolio_usd: float = Field(default=0.0, description="Total portfolio value in USD", examples=[4820.50])


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
    risk_score: int = Field(description="Risk score from 0 (safe) to 100 (dangerous)", examples=[32])
    risk_level: str = Field(description="Risk classification: LOW, MEDIUM, HIGH, or CRITICAL", examples=["LOW"])
    verdict: str = Field(description="One-line human-readable risk assessment", examples=["Low-risk wallet with normal transaction patterns"])


class PnlTokenSummary(BaseModel):
    token_symbol: str
    chain: str
    realized_pnl: float
    realized_roi: float

class PnlSummary(BaseModel):
    realized_pnl_usd: float = 0.0
    realized_pnl_percent: float = 0.0
    win_rate: float = 0.0
    traded_token_count: int = 0
    traded_times: int = 0
    top_tokens: list[PnlTokenSummary] = []

class OperationalHealth(BaseModel):
    tx_failure_rate_1hr: float = 0.0
    tx_failure_rate_24hr: float = 0.0
    total_txs_1hr: int = 0
    total_txs_24hr: int = 0
    nonce_gaps_detected: bool = False
    volume_anomaly: bool = False
    health_status: str = "unknown"  # healthy, degraded, critical, unknown
    health_detail: str = ""


class PremiumRiskResponse(BaseModel):
    risk_score: int
    risk_level: str
    verdict: str
    nansen_labels: list[NansenLabel]
    nansen_available: bool
    pnl_summary: Optional[PnlSummary] = None
    pnl_available: bool = False
    operational_health: Optional[OperationalHealth] = None


# -- Wash Models ------------------------------------------------------------

class WashIssue(BaseModel):
    category: str = Field(description="Issue category: dust, spam, gas, failed_tx, or nonce", examples=["dust"])
    severity: str = Field(description="Issue severity: low, medium, or high", examples=["medium"])
    description: str = Field(description="Human-readable description of the issue", examples=["14 dust tokens worth < $0.01 each cluttering wallet"])
    action: str = Field(description="Recommended cleanup action", examples=["Consolidate or discard dust tokens to reduce clutter"])
    estimated_savings: Optional[str] = Field(default=None, description="Potential savings from fixing this issue", examples=["$0.42/month in gas"])


class WashReport(BaseModel):
    address: str = Field(description="Wallet address scanned", examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"])
    cleanliness_score: int = Field(description="Cleanliness score 0 (dirty) to 100 (spotless)", examples=[72])
    cleanliness_grade: str = Field(description="Letter grade A-F", examples=["C"])
    total_issues: int = Field(description="Total hygiene issues found", examples=[8])
    issues_by_severity: dict = Field(description="Issue count by severity level", examples=[{"high": 1, "medium": 3, "low": 4}])
    dust_tokens: int = Field(description="Number of dust tokens (< $0.01 value)", examples=[14])
    dust_total_usd: float = Field(description="Total USD value of dust tokens", examples=[0.03])
    spam_tokens: int = Field(description="Number of suspected spam/airdrop tokens", examples=[5])
    spam_token_list: list = Field(description="Names of detected spam tokens", examples=[["FakeUSDT", "SCAM-Airdrop"]])
    gas_efficiency_pct: float = Field(description="Gas usage efficiency percentage", examples=[71.4])
    gas_efficiency_grade: str = Field(description="Gas efficiency letter grade", examples=["C"])
    wasted_gas_usd: float = Field(description="USD value of gas wasted on failed txs", examples=[8.20])
    failed_tx_count_24hr: int = Field(description="Failed transactions in the last 24 hours", examples=[3])
    failed_tx_patterns: list = Field(description="Detected failure pattern names", examples=[["repeated_revert"]])
    nonce_gaps: int = Field(description="Number of nonce gaps detected", examples=[0])
    issues: list[WashIssue] = Field(description="Detailed list of hygiene issues found")
    recommendations: list[str] = Field(description="Prioritised cleanup recommendations", examples=[["Revoke approvals for 5 spam tokens", "Set tighter gas limits on swap calls"]])
    scan_timestamp: str = Field(description="ISO 8601 timestamp of this scan", examples=["2025-04-06T10:00:00Z"])
    next_wash_recommended: str = Field(description="Recommended next scan date", examples=["2025-04-13T10:00:00Z"])


class WashResponse(BaseModel):
    status: str = Field(description="Response status", examples=["ok"])
    report: WashReport = Field(description="Full hygiene scan report")


# -- Trust Routing Helper ----------------------------------------------------

def _trust_routing(grade_letter: str) -> str:
    """Map AHS grade letter to a payment routing recommendation.

    A/B → instant_settle (trusted, low-risk agent)
    C   → escrow (moderate risk, hold funds until confirmed)
    D/E/F → reject (high risk, do not transact)
    """
    if grade_letter in ("A", "B"):
        return "instant_settle"
    if grade_letter == "C":
        return "escrow"
    return "reject"


_VALID_GRADES = frozenset({"A", "B", "C", "D", "E", "F"})


def _trust_routing_with_policy(
    grade_letter: str,
    policy: dict | None = None,
    is_allowlisted: bool = False,
) -> str:
    """Map AHS grade to routing recommendation using an integrator's custom policy.

    If is_allowlisted is True, always returns 'instant_settle' (bypass).
    If policy is None, falls back to the default hardcoded behavior.
    """
    if is_allowlisted:
        return "instant_settle"
    if policy is None:
        return _trust_routing(grade_letter)

    instant = set(g.strip() for g in policy.get("instant_grades", "A,B").split(",") if g.strip())
    escrow = set(g.strip() for g in policy.get("escrow_grades", "C").split(",") if g.strip())

    if grade_letter in instant:
        return "instant_settle"
    if not policy.get("escrow_disabled") and grade_letter in escrow:
        return "escrow"
    return "reject"


def _get_x402_payer(request: Request) -> str | None:
    """Extract the payer wallet address from an x402 payment payload (if any)."""
    payload = getattr(getattr(request, "state", None), "payment_payload", None)
    if payload is None:
        return None
    raw = payload.payload  # dict
    auth = raw.get("authorization") or raw.get("permit2Authorization") or {}
    return auth.get("from")


import re as _re
_ETH_ADDR_RE = _re.compile(r"^0x[0-9a-fA-F]{40}$")


def _resolve_policy_owner(request: Request) -> tuple[str, str | None]:
    """Resolve the policy owner ID and caller address from the request.

    API key callers: owner_id = key_hash, caller_address = None
    x402 callers: owner_id = lowercased payer wallet, caller_address = same

    Returns (owner_id, caller_address).
    Raises HTTPException 401 if neither auth path is present.
    """
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        return fiat_key["key_hash"], None
    payer = _get_x402_payer(request)
    if payer:
        return payer.lower(), payer.lower()
    raise HTTPException(status_code=401, detail="X-API-Key or x402 payment required")


def _validate_routing_policy(
    body: "RoutingPolicyRequest",
    caller_address: str | None,
) -> None:
    """Validate a routing policy request. Raises HTTPException 400 on invalid input."""
    all_grades = body.instant_grades + body.escrow_grades + body.reject_grades

    # All grades must be valid letters
    for g in all_grades:
        if g not in _VALID_GRADES:
            raise HTTPException(400, f"Invalid grade '{g}'. Must be one of A, B, C, D, E, F")

    # No duplicate grades across categories
    if len(all_grades) != len(set(all_grades)):
        raise HTTPException(400, "Grade appears in multiple categories")

    # All 6 grades must be covered
    if set(all_grades) != _VALID_GRADES:
        missing = _VALID_GRADES - set(all_grades)
        raise HTTPException(400, f"All 6 grades must be assigned. Missing: {', '.join(sorted(missing))}")

    # escrow_disabled consistency
    if body.escrow_disabled and body.escrow_grades:
        raise HTTPException(400, "escrow_disabled=true but escrow_grades is non-empty. Move escrow grades to reject_grades.")
    if not body.escrow_disabled and not body.escrow_grades:
        raise HTTPException(400, "escrow_grades is empty but escrow_disabled=false. Set escrow_disabled=true for binary mode.")

    # Allowlist validation
    if body.allowlist is not None:
        if len(body.allowlist) > 1000:
            raise HTTPException(400, "Allowlist exceeds maximum of 1000 addresses")

        for addr in body.allowlist:
            if not _ETH_ADDR_RE.match(addr):
                raise HTTPException(400, f"Invalid Ethereum address in allowlist: {addr}")

            # Self-allowlist restriction
            if caller_address and addr.lower() == caller_address.lower():
                raise HTTPException(400, "Cannot add your own wallet address to the allowlist")

            # Grade floor check: only C or above
            record = scan_db.get_latest_ahs_for_address(addr)
            if record is None:
                raise HTTPException(
                    400,
                    f"Address {addr} has no AHS score. Only addresses with Grade C or above can be allowlisted.",
                )
            grade_letter = (record["latest_grade"] or "F").split()[0]
            if grade_letter not in ("A", "B", "C"):
                raise HTTPException(
                    400,
                    f"Address {addr} has Grade {grade_letter}. Only Grade C or above can be allowlisted.",
                )


# -- AHS Models --------------------------------------------------------------

class AHSDimensionScore(BaseModel):
    dimension: str
    score: int
    weight: float
    contributing_factors: list

class AHSCrossDimensionalPattern(BaseModel):
    name: str = Field(description="Pattern identifier", examples=["Zombie Agent"])
    detected: bool = Field(description="Whether this pattern was detected", examples=[True])
    severity: str = Field(description="Pattern severity: info, warning, or critical", examples=["warning"])
    description: str = Field(description="Human-readable explanation", examples=["Wallet shows signs of abandoned automation"])

class AHSShadowSignals(BaseModel):
    session_continuity_score: Optional[int] = None
    abrupt_sessions: int = 0
    budget_exhaustion_count: int = 0
    total_sessions: int = 0
    avg_session_length: float = 0.0
    shadow_patterns: list[dict] = []

class AHSReport(BaseModel):
    address: str = Field(description="Wallet address scored", examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"])
    agent_health_score: int = Field(description="Composite AHS score 0-100", examples=[67])
    grade: str = Field(description="Letter grade A-F", examples=["C"])
    confidence: str = Field(description="Score confidence level: high, medium, or low", examples=["high"])
    mode: str = Field(description="Scoring mode: 2D (wallet + behavioural) or 3D (+ infrastructure)", examples=["2D"])
    dimensions: list[AHSDimensionScore] = Field(description="Per-dimension score breakdown")
    patterns_detected: list[AHSCrossDimensionalPattern] = Field(description="Cross-dimensional anomaly patterns detected")
    trend: Optional[str] = Field(default=None, description="Score trend vs previous scan: improving, stable, or declining", examples=["stable"])
    recommendations: list[str] = Field(description="Prioritised improvement recommendations", examples=[["Reduce revert rate below 5% to improve D1 score", "Investigate Zombie Agent pattern"]])
    ahs_token: str = Field(description="JWT token for temporal trend tracking across scans", examples=["eyJhbGciOiJIUzI1NiIs..."])
    model_version: str = Field(description="AHS scoring model version", examples=["2.1.0"])
    scan_timestamp: str = Field(description="ISO 8601 timestamp of this scan", examples=["2025-04-06T10:00:00Z"])
    next_scan_recommended: str = Field(description="Recommended next scan date", examples=["2025-04-13T10:00:00Z"])
    shadow_signals: Optional[AHSShadowSignals] = Field(default=None, description="Shadow signal analysis (session continuity, budget exhaustion)")
    routing_recommendation: str = Field(description="Trust-based routing signal: instant_settle, escrow, or reject", examples=["escrow"])

class AHSResponse(BaseModel):
    status: str = Field(description="Response status", examples=["ok"])
    report: AHSReport = Field(description="Full Agent Health Score report")


# -- AHS Batch models -------------------------------------------------------

class AHSBatchRequest(BaseModel):
    addresses: list[str]
    page: int = 1
    page_size: int = 10

class AHSBatchResultItem(BaseModel):
    address: str
    ahs_score: int
    grade: str
    d1_score: int
    d2_score: int
    pattern: str
    verdict: str
    routing_recommendation: str

class AHSBatchResponse(BaseModel):
    results: list[AHSBatchResultItem]
    page: int
    page_size: int
    total_addresses: int
    total_scored: int
    credits_used: int
    credits_remaining: Optional[int] = None
    errors: list[str]


# -- Trust Route models ------------------------------------------------------

class TrustRouteResponse(BaseModel):
    address: str = Field(description="Wallet address queried", examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"])
    agent_health_score: int = Field(description="Most recent AHS score 0-100", examples=[67])
    grade: str = Field(description="Letter grade A-F", examples=["C"])
    routing_recommendation: str = Field(description="Trust-based routing signal: instant_settle, escrow, or reject", examples=["escrow"])
    confidence: str = Field(description="Score confidence level", examples=["high"])
    scored_at: str = Field(description="ISO 8601 timestamp of the cached score", examples=["2025-04-06T10:00:00Z"])
    stale: bool = Field(description="True if the cached score is older than 24 hours", examples=[False])
    policy_applied: bool = Field(default=False, description="True if a custom routing policy was used for this recommendation")
    allowlisted: bool = Field(default=False, description="True if this address is in the caller's allowlist (instant_settle bypass)")


# -- Routing Policy models ---------------------------------------------------

class RoutingPolicyRequest(BaseModel):
    instant_grades: list[str] = Field(description="Grades that map to instant_settle", examples=[["A", "B"]])
    escrow_grades: list[str] = Field(description="Grades that map to escrow", examples=[["C"]])
    reject_grades: list[str] = Field(description="Grades that map to reject", examples=[["D", "E", "F"]])
    escrow_disabled: bool = Field(default=False, description="If true, escrow grades fall through to reject (binary mode)")
    allowlist: Optional[list[str]] = Field(default=None, description="Wallet addresses to bypass routing (always instant_settle). Max 1000.")


class RoutingPolicyResponse(BaseModel):
    instant_grades: list[str] = Field(description="Grades that map to instant_settle")
    escrow_grades: list[str] = Field(description="Grades that map to escrow")
    reject_grades: list[str] = Field(description="Grades that map to reject")
    escrow_disabled: bool = Field(description="Binary mode — no escrow tier")
    allowlist_count: int = Field(description="Number of addresses in the allowlist")
    updated_at: str = Field(description="ISO 8601 timestamp of last policy update")


# -- Report Card models ------------------------------------------------------

class EcosystemComparison(BaseModel):
    average_ahs: Optional[float] = Field(default=None, description="Ecosystem-wide average AHS score", examples=[58.3])
    percentile_rank: int = Field(description="This wallet's percentile rank in the ecosystem (0-100)", examples=[72])
    grade_distribution: dict = Field(description="Grade distribution across all scored agents", examples=[{"A": 8, "B": 22, "C": 45, "D": 18, "F": 7}])
    total_agents_scored: int = Field(description="Total agents in the comparison pool", examples=[1247])

class ReportCardDimension(BaseModel):
    dimension: str = Field(description="Dimension name (D1 Wallet Hygiene, D2 Behavioural, D3 Infrastructure)", examples=["D1 Wallet Hygiene"])
    score: Optional[int] = Field(default=None, description="Dimension score 0-100", examples=[84])
    weight: float = Field(description="Weight of this dimension in the composite score", examples=[0.5])

class ReportCardReport(BaseModel):
    address: str = Field(description="Wallet address scored", examples=["0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"])
    agent_health_score: int = Field(description="Composite AHS score 0-100", examples=[67])
    grade: str = Field(description="Letter grade A-F", examples=["C"])
    confidence: str = Field(description="Score confidence: high, medium, or low", examples=["high"])
    mode: str = Field(description="Scoring mode: 2D or 3D", examples=["2D"])
    dimensions: list[ReportCardDimension] = Field(description="Per-dimension score breakdown")
    patterns_detected: list[AHSCrossDimensionalPattern] = Field(description="Cross-dimensional anomaly patterns detected")
    recommendations: list[str] = Field(description="Prioritised improvement actions", examples=[["Reduce revert rate below 5%"]])
    ecosystem_comparison: EcosystemComparison = Field(description="How this wallet compares to the ecosystem")
    image_url: str = Field(description="URL to the generated report card PNG image", examples=["https://agenthealthmonitor.xyz/static/report-cards/0xde0b...7bae.png"])
    share_url: str = Field(description="Shareable URL for this report card", examples=["https://agenthealthmonitor.xyz/report-card/0xde0b...7bae"])

class ReportCardResponse(BaseModel):
    status: str = Field(description="Response status", examples=["ok"])
    report: ReportCardReport = Field(description="Full report card with ecosystem benchmarks")


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


def _clean_label(text: str) -> str:
    """Strip zero-width chars and emoji from Nansen labels.

    Nansen prefixes labels with emoji (e.g. '🏦 Binance: Deposit') which
    crash on Windows/cp1252 consoles and Railway logs.  Remove all chars
    outside the Basic Multilingual Plane (> U+FFFF) plus known zero-width
    codepoints, then collapse whitespace.
    """
    cleaned = []
    for ch in text:
        if ord(ch) > 0xFFFF:        # emoji / supplementary planes
            continue
        if ch in "\u200b\u200c\u200d\ufeff":  # zero-width chars
            continue
        cleaned.append(ch)
    return " ".join("".join(cleaned).split()).strip()


async def fetch_nansen_counterparties(address: str) -> list[dict] | None:
    """Fetch top counterparties from Nansen DIRECTLY via x402-paid API call.

    Uses api.nansen.ai (not Corbits proxy) with flat request body format.
    The direct endpoint returns 402 → x402 client pays → 200 with data.
    """
    if not _nansen_x402_client:
        logging.debug("Nansen counterparties skipped: x402 client not initialized")
        return None
    now = datetime.now(timezone.utc)
    body = {
        "address": address,
        "chain": "all",
        "date": {
            "from": (now - timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "pagination": {"page": 1, "per_page": 25},
    }
    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(45.0)) as http:
            response = await http.post(NANSEN_COUNTERPARTIES_URL, json=body)
        logging.info("Nansen counterparties response: status=%s length=%s", response.status_code, len(response.content))
        if response.status_code == 200:
            data = response.json()
            logging.info("Nansen counterparties raw response: %s", str(data)[:1000])
            result = data if isinstance(data, list) else data.get("data", data.get("counterparties", []))
            logging.info("Nansen counterparties parsed: %d items", len(result) if isinstance(result, list) else 0)
            return result
        logging.warning("Nansen counterparties non-200: status=%s body=%s", response.status_code, response.text[:500])
        return None
    except Exception as e:
        logging.warning("Nansen counterparties call failed: %s", e, exc_info=True)
        return None


async def fetch_nansen_pnl(address: str) -> dict | None:
    """Fetch PnL summary from Nansen DIRECTLY via x402-paid API call.

    Uses api.nansen.ai (not Corbits proxy) with flat request body format.
    Must be called sequentially (not concurrent) with other Nansen x402 calls
    to avoid payer nonce conflicts.
    """
    if not _nansen_x402_client:
        logging.debug("Nansen PnL skipped: x402 client not initialized")
        return None
    now = datetime.now(timezone.utc)
    body = {
        "address": address,
        "chain": "all",
        "date": {
            "from": (now - timedelta(days=180)).strftime("%Y-%m-%d"),
            "to": now.strftime("%Y-%m-%d"),
        },
    }
    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(45.0)) as http:
            response = await http.post(NANSEN_PNL_SUMMARY_URL, json=body)
        logging.info("Nansen PnL response: status=%s length=%s", response.status_code, len(response.content))
        if response.status_code == 200:
            data = response.json()
            logging.info("Nansen PnL raw response: %s", str(data)[:1000])
            return data
        logging.warning("Nansen PnL non-200: status=%s body=%s", response.status_code, response.text[:500])
        return None
    except Exception as e:
        logging.warning("Nansen PnL call failed: %s", e, exc_info=True)
        return None


VALID_NANSEN_CHAINS = frozenset({
    "arbitrum", "avalanche", "base", "bitcoin", "bnb", "ethereum",
    "iotaevm", "linea", "mantle", "monad", "near", "optimism",
    "plasma", "polygon", "ronin", "scroll", "sei", "solana",
    "sonic", "starknet", "sui", "ton", "tron",
})


async def fetch_nansen_related_wallets(address: str, chain: str = "ethereum") -> list[dict] | None:
    """Fetch related wallets from Nansen DIRECTLY via x402-paid API call.

    Uses api.nansen.ai (not Corbits proxy) with flat request body format.
    NOTE: This endpoint does NOT support chain="all" — a specific chain
    is required.  Defaults to "ethereum".
    """
    if not _nansen_x402_client:
        logging.debug("Nansen related-wallets skipped: x402 client not initialized")
        return None
    body = {
        "address": address,
        "chain": chain,
    }
    try:
        async with x402HttpxClient(_nansen_x402_client, timeout=httpx_client.Timeout(45.0)) as http:
            response = await http.post(NANSEN_RELATED_WALLETS_URL, json=body)
        logging.info("Nansen related-wallets response: status=%s length=%s", response.status_code, len(response.content))
        if response.status_code == 200:
            data = response.json()
            logging.info("Nansen related-wallets raw response: %s", str(data)[:1000])
            result = data if isinstance(data, list) else data.get("data", [])
            logging.info("Nansen related-wallets parsed: %d items", len(result) if isinstance(result, list) else 0)
            return result
        logging.warning("Nansen related-wallets non-200: status=%s body=%s", response.status_code, response.text[:500])
        return None
    except Exception as e:
        logging.warning("Nansen related-wallets call failed: %s", e, exc_info=True)
        return None


# -- Transaction Failure Metrics (Operational Health) -------------------------

def get_tx_failure_metrics(address: str, transactions: list[dict]) -> OperationalHealth:
    """Calculate transaction failure metrics from Basescan transaction list.

    Analyses recent transactions to determine operational health based on
    revert rates, nonce gaps, and volume anomalies.
    """
    if not transactions:
        return OperationalHealth(
            health_status="unknown",
            health_detail="Insufficient transaction history",
        )

    now_ts = datetime.now(timezone.utc).timestamp()
    one_hour_ago = now_ts - 3600
    twenty_four_hours_ago = now_ts - 86400

    # Filter to outgoing transactions from this address
    outgoing = [
        tx for tx in transactions
        if tx.get("from", "").lower() == address.lower()
    ]

    if not outgoing:
        return OperationalHealth(
            health_status="unknown",
            health_detail="No outgoing transactions found",
        )

    # Split into time windows
    txs_1hr = []
    txs_24hr = []
    for tx in outgoing:
        try:
            tx_ts = int(tx.get("timeStamp", 0))
        except (ValueError, TypeError):
            continue
        if tx_ts >= one_hour_ago:
            txs_1hr.append(tx)
        if tx_ts >= twenty_four_hours_ago:
            txs_24hr.append(tx)

    # Count failures: isError == "1" or txreceipt_status == "0"
    def is_failed(tx: dict) -> bool:
        return tx.get("isError") == "1" or tx.get("txreceipt_status") == "0"

    failed_1hr = sum(1 for tx in txs_1hr if is_failed(tx))
    failed_24hr = sum(1 for tx in txs_24hr if is_failed(tx))

    total_1hr = len(txs_1hr)
    total_24hr = len(txs_24hr)

    rate_1hr = (failed_1hr / total_1hr) if total_1hr > 0 else 0.0
    rate_24hr = (failed_24hr / total_24hr) if total_24hr > 0 else 0.0

    # Nonce gap detection: check for gaps in the nonce sequence
    nonce_gaps = False
    if len(outgoing) >= 2:
        nonces = []
        for tx in outgoing:
            try:
                nonces.append(int(tx.get("nonce", 0)))
            except (ValueError, TypeError):
                continue
        if nonces:
            nonces_sorted = sorted(set(nonces))
            if len(nonces_sorted) >= 2:
                expected_count = nonces_sorted[-1] - nonces_sorted[0] + 1
                if expected_count > len(nonces_sorted):
                    nonce_gaps = True

    # Volume anomaly: compare last-hour tx rate to historical average
    volume_anomaly = False
    if len(outgoing) >= 10:
        try:
            first_ts = int(outgoing[0].get("timeStamp", 0))
            last_ts = int(outgoing[-1].get("timeStamp", 0))
            history_span_hrs = max((last_ts - first_ts) / 3600, 1.0)
            avg_txs_per_hr = len(outgoing) / history_span_hrs
            if avg_txs_per_hr > 0:
                # Spike: >3x average, or drop: <0.1x average (with at least
                # 1hr of history and non-trivial average)
                if total_1hr > avg_txs_per_hr * 3:
                    volume_anomaly = True
                elif avg_txs_per_hr >= 1.0 and total_1hr < avg_txs_per_hr * 0.1:
                    volume_anomaly = True
        except (ValueError, TypeError):
            pass

    # Determine health status
    details = []
    if total_1hr == 0 and total_24hr == 0:
        health_status = "unknown"
        health_detail = "No recent transactions to evaluate"
    else:
        if rate_1hr > 0.20 or rate_24hr > 0.20:
            health_status = "critical"
            if rate_1hr > 0.20:
                details.append(f"{rate_1hr:.0%} revert rate in last hour")
            if rate_24hr > 0.20:
                details.append(f"{rate_24hr:.0%} revert rate in last 24 hours")
        elif rate_1hr > 0.05 or rate_24hr > 0.05 or nonce_gaps:
            health_status = "degraded"
            if rate_1hr > 0.05:
                details.append(f"{rate_1hr:.0%} revert rate in last hour")
            if rate_24hr > 0.05:
                details.append(f"{rate_24hr:.0%} revert rate in last 24 hours")
            if nonce_gaps:
                details.append("nonce gaps detected")
        else:
            health_status = "healthy"

        # Upgrade to critical if volume anomaly + high reverts
        if volume_anomaly and (rate_1hr > 0.05 or rate_24hr > 0.05):
            health_status = "critical"
            details.append("transaction volume anomaly with elevated reverts")
        elif volume_anomaly:
            if health_status == "healthy":
                health_status = "degraded"
            details.append("transaction volume anomaly detected")

        if health_status == "healthy":
            health_detail = "Operations normal — low revert rate across all windows"
        else:
            health_detail = "; ".join(details) if details else f"{health_status} operational health"

    return OperationalHealth(
        tx_failure_rate_1hr=round(rate_1hr, 4),
        tx_failure_rate_24hr=round(rate_24hr, 4),
        total_txs_1hr=total_1hr,
        total_txs_24hr=total_24hr,
        nonce_gaps_detected=nonce_gaps,
        volume_anomaly=volume_anomaly,
        health_status=health_status,
        health_detail=health_detail,
    )


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


def _is_safe_webhook_url(url: str) -> bool:
    """Reject URLs targeting private/internal networks (SSRF prevention)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Block cloud metadata, loopback, and private ranges
        try:
            addr = ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return False
        except ValueError:
            # hostname is a domain name, not an IP — block obvious patterns
            lower = hostname.lower()
            if lower in ("localhost", "metadata.google.internal"):
                return False
            if lower.endswith(".internal") or lower.endswith(".local"):
                return False
        return True
    except Exception:
        return False


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
    import db as _db
    _db.init_db()

    # Verify critical static assets exist at startup
    _log = logging.getLogger("ahm")
    og_img = pathlib.Path(__file__).parent / "static" / "ahm-og-banner.png"
    if og_img.is_file():
        _log.info("Static OG image verified: %s (%d bytes)", og_img, og_img.stat().st_size)
    else:
        _log.error("OG image MISSING at %s — social previews will be broken", og_img)

    # One-time backfill: populate Zombie Agent patterns for low-D2 scans
    try:
        filled = _db.backfill_zombie_patterns()
        if filled:
            logger.info("Backfilled %d Zombie Agent patterns on startup", filled)
    except Exception:
        logger.exception("Zombie pattern backfill failed (non-fatal)")
    alert_task = asyncio.create_task(alert_monitor_loop())
    rescan_task = asyncio.create_task(rescan_loop())

    # ERC-8183 evaluator worker (Arc testnet)
    erc8183_task = None
    if erc8183_worker.can_start():
        erc8183_task = asyncio.create_task(erc8183_worker.erc8183_worker_loop())

    # -- APScheduler: nightly scans --
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_acp_scan,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="acp_nightly_scan",
        name="ACP Nightly Scan",
        coalesce=True,
        misfire_grace_time=3600,  # fire if <=1hr late (e.g. restart at 02:30)
        max_instances=1,
    )
    scheduler.add_job(
        run_olas_scan,
        trigger=CronTrigger(hour=2, minute=30, timezone="UTC"),
        id="olas_nightly_scan",
        name="Olas Nightly Scan",
        coalesce=True,
        misfire_grace_time=3600,
        max_instances=1,
    )
    scheduler.add_job(
        run_celo_scan,
        trigger=CronTrigger(hour=2, minute=45, timezone="UTC"),
        id="celo_nightly_scan",
        name="Celo Nightly Scan",
        coalesce=True,
        misfire_grace_time=3600,
        max_instances=1,
    )
    scheduler.add_job(
        run_arc_scan,
        trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="arc_nightly_scan",
        name="Arc Nightly Scan",
        coalesce=True,
        misfire_grace_time=3600,
        max_instances=1,
    )
    scheduler.add_job(
        run_erc8004_scan,
        trigger=CronTrigger(hour=3, minute=15, timezone="UTC"),
        id="erc8004_nightly_scan",
        name="ERC-8004 Base Nightly Scan",
        coalesce=True,
        misfire_grace_time=3600,
        max_instances=1,
    )
    scheduler.start()
    app.state.scheduler = scheduler

    bg_names = "alert_monitor, rescan_loop, acp_scheduler, olas_scheduler, celo_scheduler, arc_scheduler, erc8004_scheduler"
    if erc8183_task:
        bg_names += ", erc8183_worker"
    logger.info("Background tasks started: %s", bg_names)
    yield

    # -- Shutdown --
    scheduler.shutdown(wait=False)
    alert_task.cancel()
    rescan_task.cancel()
    if erc8183_task:
        erc8183_task.cancel()
    tasks_to_await = [alert_task, rescan_task]
    if erc8183_task:
        tasks_to_await.append(erc8183_task)
    for t in tasks_to_await:
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Agent Health Monitor API",
    description=(
        "Wallet intelligence API for autonomous agents on Base L2.\n\n"
        "Analyzes transaction failures, gas inefficiency, nonce issues, counterparty risk, "
        "and behavioural patterns. Returns actionable health reports, risk scores, "
        "gas optimization plans, and ready-to-sign retry transactions.\n\n"
        "**Payment:** All paid endpoints accept x402 protocol (USDC on Base) or a "
        "fiat-purchased API key via the `X-API-Key` header.\n\n"
        "**Free preview endpoints** are available for `/retry/preview` and `/agent/protect/preview`.\n\n"
        "**Coupon access** — partners can use coupon codes to access any paid endpoint "
        "without x402 payment via the `/coupon/{action}/{code}/{address}` routes."
    ),
    version="1.8.0",
    contact={
        "name": "AHM Support",
        "url": "https://agenthealthmonitor.xyz",
    },
    license_info={
        "name": "Proprietary",
    },
    openapi_tags=[
        {
            "name": "Scoring & Risk",
            "description": "Risk scoring, counterparty analysis, and wallet network mapping. "
            "Fast pre-flight checks from $0.001 to deep Nansen-enriched analysis at $0.10.",
        },
        {
            "name": "Health & Hygiene",
            "description": "Wallet health diagnosis, hygiene scans, composite Agent Health Score (AHS), "
            "and visual report cards. Core analysis endpoints from $0.50 to $2.00.",
        },
        {
            "name": "Optimization",
            "description": "Gas optimization reports and failed-transaction retry bot. "
            "Returns per-transaction-type savings and ready-to-sign EIP-1559 retry payloads.",
        },
        {
            "name": "Protection",
            "description": "Autonomous protection agent that triages risk and runs the appropriate "
            "combination of health, optimization, and retry services automatically.",
        },
        {
            "name": "Alerts",
            "description": "Subscribe wallets to automated health monitoring with webhook alerts. "
            "Health checks run every 6 hours; alerts fire when configurable thresholds are breached.",
        },
        {
            "name": "Coupon Access",
            "description": "Partner coupon routes — access any paid endpoint for free with a valid coupon code. "
            "Rate-limited to 5 requests per IP per minute.",
        },
        {
            "name": "Discovery & Info",
            "description": "Service discovery, pricing info, ecosystem statistics, and x402/ERC-8004 "
            "well-known documents for automated agent registration.",
        },
        {
            "name": "Billing",
            "description": "Stripe checkout webhooks and API key management for fiat-paid access.",
        },
        {
            "name": "Admin",
            "description": "Internal admin endpoints protected by X-Internal-Key header. "
            "Security activity logs, trust registry, and registry scan triggers.",
        },
        {
            "name": "Utility",
            "description": "Health checks and AI chat assistant.",
        },
    ],
    lifespan=lifespan,
)


# -- CORS -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Public API — allow any origin
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


# -- Security Headers Middleware --------------------------------------------

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if "railway.app" in request.headers.get("host", ""):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# -- Custom Middleware to Bypass x402 on Internal Calls ----------------------

class AddressValidationMiddleware:
    """
    ASGI middleware that rejects invalid Ethereum addresses early,
    before the x402 middleware processes the request.

    Checks any path segment that looks like an address parameter
    in known endpoint patterns. Returns 400 JSON for invalid addresses.
    """
    _ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
    # Path prefixes that take an {address} parameter
    # Sorted by length descending so longer/more-specific prefixes
    # (e.g. /risk/premium/) match before shorter ones (e.g. /risk/).
    _ADDRESS_PREFIXES = tuple(sorted([
        "/health/", "/risk/", "/risk/premium/", "/counterparties/",
        "/network-map/", "/wash/", "/ahs/", "/ahs/route/", "/optimize/",
        "/retry/", "/retry/preview/", "/agent/protect/",
        "/agent/protect/preview/", "/alerts/subscribe/",
        "/alerts/status/", "/alerts/unsubscribe/", "/report-card/",
    ], key=len, reverse=True))
    # Exact paths under address prefixes that are NOT address endpoints
    _SKIP_PATHS = frozenset({"/ahs/batch", "/ahs/route/policy"})

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip non-address endpoints that share a prefix (e.g. /ahs/batch)
        if path in self._SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        # Check direct endpoint patterns: /prefix/{address}
        for prefix in self._ADDRESS_PREFIXES:
            if path.startswith(prefix) and len(path) > len(prefix):
                address_part = path[len(prefix):].split("/")[0].split("?")[0]
                if not self._ADDRESS_RE.match(address_part):
                    await self._send_400(send)
                    return
                break

        # Check coupon endpoint patterns: /coupon/{action}/{code}/{address}
        if path.startswith("/coupon/") and path.count("/") >= 4:
            parts = path.strip("/").split("/")
            if len(parts) >= 4:
                address_part = parts[3].split("?")[0]
                if address_part and not self._ADDRESS_RE.match(address_part):
                    await self._send_400(send)
                    return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_400(send):
        import json
        body = json.dumps({
            "error": "Invalid address format",
            "detail": "Address must be a valid 0x-prefixed 40-character hex string",
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


class SecurityMonitorMiddleware:
    """ASGI middleware that observes all responses and detects suspicious patterns.

    Must be the outermost middleware (last added via add_middleware).

    Detects:
    1. Endpoint enumeration: 10+ distinct paths from same IP in 60s
    2. High 4xx rate: >80% of last 20 requests are 4xx
    3. 402 without payment: 3+ 402s with no subsequent 200 in 5min
    """

    _ENUM_THRESHOLD = 10
    _ENUM_WINDOW = 60
    _4XX_THRESHOLD = 0.8
    _4XX_SAMPLE = 20
    _402_THRESHOLD = 3
    _402_WINDOW = 300
    _COOLDOWN = 300  # Don't re-log same event type for same IP within 5 min

    def __init__(self, app):
        self.app = app
        self._requests: dict[str, list[tuple[float, str, int]]] = {}
        self._402_ips: dict[str, list[float]] = {}
        self._event_cooldown: dict[str, float] = {}
        self._cleanup_counter = 0

    @staticmethod
    def _extract_ip(scope: dict) -> str:
        """Extract client IP from ASGI scope headers."""
        headers = dict(scope.get("headers", []))
        ip = headers.get(b"x-envoy-external-address", b"").decode()
        if ip:
            return ip
        xff = headers.get(b"x-forwarded-for", b"").decode()
        if xff:
            return xff.split(",")[0].strip()
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        ip = self._extract_ip(scope)
        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if status_code is not None:
            self._record_request(ip, path, status_code)

    def _record_request(self, ip: str, path: str, status_code: int):
        now = time.time()

        if ip not in self._requests:
            self._requests[ip] = []
        self._requests[ip].append((now, path, status_code))

        # Trim to 5 minutes and 100 entries max
        cutoff = now - 300
        entries = [(t, p, s) for t, p, s in self._requests[ip] if t > cutoff]
        if len(entries) > 100:
            entries = entries[-100:]
        self._requests[ip] = entries

        # Pattern 1: Endpoint enumeration (10+ distinct paths in 60s)
        window_entries = [(t, p, s) for t, p, s in entries if now - t < self._ENUM_WINDOW]
        distinct = set(p for _, p, _ in window_entries)
        if len(distinct) >= self._ENUM_THRESHOLD:
            self._log_event(
                "endpoint_enumeration", ip,
                f"{len(distinct)} distinct paths in {self._ENUM_WINDOW}s",
            )

        # Pattern 2: High 4xx rate (>80% of last 20 requests)
        recent = entries[-self._4XX_SAMPLE:]
        if len(recent) >= self._4XX_SAMPLE:
            n_4xx = sum(1 for _, _, s in recent if 400 <= s < 500)
            if n_4xx / len(recent) > self._4XX_THRESHOLD:
                self._log_event(
                    "high_4xx_rate", ip,
                    f"{n_4xx}/{len(recent)} requests returned 4xx",
                )

        # Pattern 3: 402 without payment (3+ 402s, no 200, within 5min)
        if status_code == 402:
            if ip not in self._402_ips:
                self._402_ips[ip] = []
            self._402_ips[ip].append(now)
        elif status_code == 200 and ip in self._402_ips:
            self._402_ips.pop(ip, None)

        if ip in self._402_ips:
            window = [t for t in self._402_ips[ip] if now - t < self._402_WINDOW]
            self._402_ips[ip] = window
            if len(window) >= self._402_THRESHOLD:
                self._log_event(
                    "402_without_payment", ip,
                    f"{len(window)} unpaid 402 responses in {self._402_WINDOW}s",
                )
                self._402_ips.pop(ip, None)

        # Periodic cleanup
        self._cleanup_counter += 1
        if self._cleanup_counter >= 500:
            self._cleanup_counter = 0
            self._cleanup()

    def _log_event(self, event_type: str, ip: str, details: str):
        key = f"{event_type}:{ip}"
        now = time.time()
        if key in self._event_cooldown and now - self._event_cooldown[key] < self._COOLDOWN:
            return
        self._event_cooldown[key] = now
        try:
            import db as _db
            _db.log_security_event(event_type, ip, details)
        except Exception:
            pass

    def _cleanup(self):
        now = time.time()
        cutoff = now - 300
        stale = [ip for ip, reqs in self._requests.items()
                 if not reqs or reqs[-1][0] < cutoff]
        for ip in stale:
            del self._requests[ip]
        stale_402 = [ip for ip, ts in self._402_ips.items()
                     if not ts or max(ts) < cutoff]
        for ip in stale_402:
            del self._402_ips[ip]
        stale_cd = [k for k, t in self._event_cooldown.items() if now - t > self._COOLDOWN]
        for k in stale_cd:
            del self._event_cooldown[k]


class InternalKeyBypassMiddleware:
    """
    ASGI middleware that skips x402 payment when X-Internal-Key header is valid.

    Must be added AFTER PaymentMiddlewareASGI so it wraps the outside
    (last added = outermost in Starlette's middleware stack).

    When the key matches, we walk the middleware chain to find
    PaymentMiddlewareASGI and jump past it to the inner app,
    regardless of how many middleware layers sit between us.
    """
    def __init__(self, app):
        self.app = app
        # Walk the middleware chain once at startup to find the app
        # on the other side of PaymentMiddlewareASGI.
        inner = app
        while inner is not None:
            if isinstance(inner, PaymentMiddlewareASGI):
                inner = getattr(inner, "app", inner)
                break
            inner = getattr(inner, "app", None)
        self._bypass_app = inner or app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check for internal API key in headers
        headers = dict(scope.get("headers", []))
        internal_key = headers.get(b"x-internal-key", b"").decode()

        if internal_key and hmac.compare_digest(internal_key, INTERNAL_API_KEY):
            # Valid key — skip x402 by jumping past PaymentMiddlewareASGI
            await self._bypass_app(scope, receive, send)
        else:
            # No valid key — normal flow through x402
            await self.app(scope, receive, send)


class ApiKeyBypassMiddleware:
    """
    ASGI middleware that skips x402 payment when a valid X-API-Key header is present.

    Validates the key exists and is active (hash lookup only — lightweight).
    Full validation (calls_remaining decrement, usage logging) happens inside
    each endpoint handler to keep accounting per-endpoint.
    """
    # Paths where X-API-Key can bypass x402 payment
    _FIAT_PATHS = (
        "/risk/premium/", "/risk/", "/counterparties/", "/network-map/",
        "/health/", "/wash/", "/ahs/", "/optimize/", "/retry/",
        "/report-card/", "/alerts/subscribe/", "/agent/protect/",
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only apply to paid endpoint paths
        is_fiat_path = any(path.startswith(p) for p in self._FIAT_PATHS)
        if not is_fiat_path:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        api_key = headers.get(b"x-api-key", b"").decode()

        if api_key and api_key.startswith("ahm_live_"):
            # Looks like an AHM API key — validate hash exists and is active
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            import db as _db
            conn = _db.get_connection()
            try:
                row = conn.execute(
                    "SELECT 1 FROM api_keys WHERE key_hash = ? AND is_active = 1",
                    (key_hash,),
                ).fetchone()
            finally:
                conn.close()

            if row:
                # Valid key — skip x402 by jumping to inner app
                inner_app = getattr(self.app, "app", self.app)
                await inner_app(scope, receive, send)
                return

        # No valid API key — normal flow through x402
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
            "Premium risk score enriched with Nansen wallet intelligence labels, "
            "PnL profitability summary, and operational health metrics "
            "(transaction failure rates, nonce gaps, volume anomalies)."
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
                            "risk_score": 35,
                            "risk_level": "MEDIUM",
                            "verdict": "Medium - elevated failure rate with contract reverts detected",
                            "nansen_labels": [
                                {"label": "Smart Money", "category": "smart_money"},
                            ],
                            "nansen_available": True,
                            "pnl_summary": {
                                "realized_pnl_usd": 12500.0,
                                "win_rate": 0.65,
                                "traded_token_count": 18,
                            },
                            "pnl_available": True,
                            "operational_health": {
                                "tx_failure_rate_1hr": 0.02,
                                "tx_failure_rate_24hr": 0.05,
                                "health_status": "healthy",
                            },
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "risk_score": {"type": "integer", "description": "Risk score 0-100"},
                                "risk_level": {"type": "string", "description": "CRITICAL, HIGH, MEDIUM, or LOW"},
                                "verdict": {"type": "string", "description": "One-line risk assessment"},
                                "nansen_labels": {"type": "array", "description": "Nansen wallet intelligence labels"},
                                "nansen_available": {"type": "boolean"},
                                "pnl_summary": {"type": "object", "description": "Profit/loss summary from Nansen"},
                                "pnl_available": {"type": "boolean"},
                                "operational_health": {"type": "object", "description": "Tx failure rates and health status"},
                            },
                        },
                    },
                },
            },
        },
    ),
    "GET /counterparties/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=COUNTERPARTY_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Know Your Counterparty report showing top wallets and contracts "
            "the address interacts with most, enriched with Nansen labels."
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
                            "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                            "counterparties": [
                                {
                                    "address": "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
                                    "label": "Uniswap: Universal Router",
                                    "interaction_count": 85,
                                    "volume_usd": 125000.0,
                                    "last_interaction": "2026-02-20",
                                },
                            ],
                            "total_counterparties": 25,
                            "nansen_available": True,
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "address": {"type": "string", "description": "Queried wallet address"},
                                "counterparties": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "address": {"type": "string"},
                                            "label": {"type": "string"},
                                            "interaction_count": {"type": "integer"},
                                            "volume_usd": {"type": "number"},
                                            "last_interaction": {"type": "string"},
                                        },
                                    },
                                    "description": "Top counterparties ranked by interaction count",
                                },
                                "total_counterparties": {"type": "integer"},
                                "nansen_available": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        },
    ),
    "GET /network-map/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=NETWORK_MAP_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Wallet Network Map showing related wallets (first funders, deployers, "
            "multisig co-signers) enriched with Nansen labels."
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
                            "chain": "ethereum",
                        },
                    },
                    "output": {
                        "type": "json",
                        "example": {
                            "status": "ok",
                            "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                            "chain": "ethereum",
                            "related_wallets": [
                                {
                                    "address": "0x1234567890abcdef1234567890abcdef12345678",
                                    "label": "First Funder",
                                    "relation": "first_funder",
                                    "chain": "ethereum",
                                },
                            ],
                            "total_related": 5,
                            "nansen_available": True,
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "address": {"type": "string", "description": "Queried wallet address"},
                                "chain": {"type": "string", "description": "Blockchain queried"},
                                "related_wallets": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "address": {"type": "string"},
                                            "label": {"type": "string"},
                                            "relation": {"type": "string"},
                                            "chain": {"type": "string"},
                                        },
                                    },
                                    "description": "Related wallets via funding, deployment, or multisig links",
                                },
                                "total_related": {"type": "integer"},
                                "nansen_available": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        },
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
                            "risk_score": 22,
                            "risk_level": "LOW",
                            "verdict": "Wallet operating normally with no significant issues detected",
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "risk_score": {"type": "integer", "description": "Risk score 0-100 (0=safe, 100=critical)"},
                                "risk_level": {"type": "string", "description": "CRITICAL, HIGH, MEDIUM, or LOW"},
                                "verdict": {"type": "string", "description": "One-line human-readable risk assessment"},
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
    "GET /ahs/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=AHS_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Agent Health Score: Proprietary composite 0-100 diagnostic across wallet "
            "hygiene, behavioural patterns, and infrastructure health. Premium agent analysis."
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
                                "agent_health_score": 72,
                                "grade": "Needs Attention",
                                "confidence": "HIGH",
                                "mode": "2D",
                                "dimensions": [
                                    {"dimension": "wallet_hygiene", "score": 85, "weight": 0.30, "contributing_factors": ["Wallet hygiene is healthy"]},
                                    {"dimension": "behavioural_patterns", "score": 63, "weight": 0.70, "contributing_factors": ["Repeated failures: 6 consecutive"]},
                                ],
                                "patterns_detected": [
                                    {"name": "Stale Strategy", "detected": True, "severity": "warning", "description": "Agent is repeatedly failing on the same contract interaction without adapting."},
                                ],
                                "recommendations": ["Investigate repeated failures", "Enable dynamic gas pricing"],
                                "model_version": "AHS-v1",
                                "next_scan_recommended": "7 days",
                            },
                        },
                    },
                },
            },
        },
    ),
    "GET /ahs/route/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=ROUTE_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Trust routing signal for agent wallets. Returns a lightweight "
            "routing recommendation (instant_settle / escrow / reject) based "
            "on the most recent AHS grade."
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
                            "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                            "agent_health_score": 67,
                            "grade": "C",
                            "routing_recommendation": "escrow",
                            "confidence": "high",
                            "scored_at": "2026-04-22T10:00:00Z",
                            "stale": False,
                            "policy_applied": False,
                            "allowlisted": False,
                        },
                    },
                },
            },
        },
    ),
    "PUT /ahs/route/policy": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price="$0.01",
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Configure custom trust routing policy. Set grade-to-action mappings "
            "and manage an address allowlist for instant_settle bypass."
        ),
    ),
    "POST /ahs/batch": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=AHS_BATCH_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "AHS Batch: Score multiple agent wallets in a single call. "
            "Up to 10 wallets per x402 payment ($10.00), or up to 25 via API key."
        ),
        extensions={
            "bazaar": {
                "info": {
                    "input": {
                        "type": "http",
                        "method": "POST",
                        "discoverable": True,
                        "body": {
                            "addresses": [
                                "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                                "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
                            ],
                        },
                    },
                    "output": {
                        "type": "json",
                        "example": {
                            "results": [
                                {
                                    "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                                    "ahs_score": 72,
                                    "grade": "C",
                                    "d1_score": 85,
                                    "d2_score": 63,
                                    "pattern": "Stale Strategy",
                                    "verdict": "Needs Attention",
                                    "routing_recommendation": "escrow",
                                },
                                {
                                    "address": "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae",
                                    "ahs_score": 88,
                                    "grade": "B",
                                    "d1_score": 91,
                                    "d2_score": 86,
                                    "pattern": "Healthy",
                                    "verdict": "Good",
                                    "routing_recommendation": "instant_settle",
                                },
                            ],
                            "page": 1,
                            "page_size": 25,
                            "total_addresses": 2,
                            "total_scored": 2,
                            "credits_used": 2,
                            "errors": [],
                        },
                    },
                },
            },
        },
    ),
    "GET /report-card/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=REPORT_CARD_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Agent Report Card: Visual health report card with ecosystem benchmarks. "
            "Generates a shareable 1200x675 PNG image with AHS score, dimension breakdown, "
            "ecosystem comparison, and detected patterns."
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
                                "agent_health_score": 72,
                                "grade": "C — Needs Attention",
                                "dimensions": [
                                    {"dimension": "D1: Wallet Hygiene", "score": 68, "weight": 0.30},
                                    {"dimension": "D2: Behavioural Patterns", "score": 55, "weight": 0.70},
                                ],
                                "ecosystem_comparison": {
                                    "average_ahs": 59.3,
                                    "percentile_rank": 65,
                                    "total_agents_scored": 4552,
                                },
                                "image_url": "/static/report-cards/0xd8da...6045.png",
                            },
                        },
                    },
                },
            },
        },
    ),
    "POST /wash/*": RouteConfig(
        accepts=[
            PaymentOption(
                scheme="exact",
                pay_to=PAYMENT_ADDRESS,
                price=WASH_PRICE,
                network=NETWORK,
            ),
        ],
        mime_type="application/json",
        description=(
            "Agent Wash: Recurring hygiene scan for blockchain agent wallets. "
            "Detects dust tokens, spam tokens, gas inefficiency, failed transaction "
            "patterns, and nonce gaps. Returns a cleanliness score and prioritised "
            "cleanup recommendations."
        ),
        extensions={
            "bazaar": {
                "info": {
                    "input": {
                        "type": "http",
                        "method": "POST",
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
                                "cleanliness_score": 72,
                                "cleanliness_grade": "Clean",
                                "total_issues": 8,
                                "dust_tokens": 5,
                                "spam_tokens": 2,
                                "gas_efficiency_pct": 68.5,
                                "gas_efficiency_grade": "Good",
                                "failed_tx_count_24hr": 1,
                                "nonce_gaps": 0,
                                "recommendations": [
                                    "Clear 5 dust tokens to declutter wallet",
                                    "2 spam tokens detected — consider blocking future airdrops",
                                    "Looking good — schedule your next wash in 30 days",
                                ],
                                "next_wash_recommended": "30 days",
                            },
                        },
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "report": {
                                    "type": "object",
                                    "properties": {
                                        "cleanliness_score": {"type": "integer", "description": "0-100 cleanliness score"},
                                        "cleanliness_grade": {"type": "string", "description": "Spotless/Clean/Needs Attention/Dirty/Critical"},
                                        "dust_tokens": {"type": "integer"},
                                        "spam_tokens": {"type": "integer"},
                                        "gas_efficiency_pct": {"type": "number"},
                                        "recommendations": {"type": "array", "items": {"type": "string"}},
                                        "next_wash_recommended": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    ),
}

app.add_middleware(PaymentMiddlewareASGI, routes=x402_routes, server=server)

# Add API key bypass AFTER x402 — X-API-Key (Stripe fiat customers) skips x402.
app.add_middleware(ApiKeyBypassMiddleware)

# Add internal key bypass AFTER API key bypass — last added = outermost = runs first.
# When valid X-Internal-Key is present, skips x402 entirely.
app.add_middleware(InternalKeyBypassMiddleware)

# Add address validation as outermost middleware — rejects invalid addresses
# before x402 or internal key bypass can process them.
app.add_middleware(AddressValidationMiddleware)

# Security monitor: outermost — observes all response status codes and
# detects endpoint enumeration, high 4xx rates, and 402 without payment.
app.add_middleware(SecurityMonitorMiddleware)

# Rate limiter state (slowapi reads app.state.limiter for decorator checks)
app.state.limiter = limiter


# -- Global Exception Handler ------------------------------------------------

logger = logging.getLogger("ahm")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — log full traceback, return generic error."""
    import traceback
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path, exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Return JSON 429 and log to security_events."""
    ip = get_client_ip(request)
    try:
        scan_db.log_security_event("rate_limit_exceeded", ip, str(exc.detail))
    except Exception:
        pass
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )


# -- API Key (Fiat) Access Helpers ------------------------------------------

def validate_fiat_request(request: Request) -> dict | None:
    """Check for X-API-Key header and validate.

    Returns the key record dict if valid, None if no X-API-Key header present.
    Raises HTTPException 401/429 if the key is present but invalid or exhausted.
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None

    record = scan_db.validate_api_key(api_key)
    if record is None:
        # Key present but invalid — could be expired, inactive, or wrong
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    if record["calls_remaining"] is not None and record["calls_remaining"] <= 0:
        raise HTTPException(status_code=429, detail="API key calls exhausted")

    return record


def consume_api_key(
    record: dict,
    endpoint: str,
    wallet: str | None = None,
    request: Request | None = None,
) -> None:
    """Decrement calls and log usage for a validated API key.

    When an X-Partner-Id header is present on the request, attribute the
    call to that partner in usage logs.
    """
    key_hash = record["key_hash"]
    # Resolve partner_id: explicit header > key's own partner_id
    partner_id = None
    if request:
        partner_id = request.headers.get("X-Partner-Id")
    if not partner_id:
        partner_id = record.get("partner_id")
    scan_db.decrement_api_key(key_hash)
    scan_db.log_api_key_usage(key_hash, endpoint, wallet, partner_id=partner_id)


# -- Routes ------------------------------------------------------------------

STATIC_DIR = pathlib.Path(__file__).parent / "static"


@app.api_route("/", methods=["GET", "HEAD"], tags=["Discovery & Info"])
async def root():
    """Serve the marketing homepage."""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"service": "Agent Health Monitor", "docs": "/docs", "info": "/api/info"})


@app.get("/app", tags=["Discovery & Info"])
async def app_page():
    """Serve the developer tool (formerly the homepage)."""
    app_file = STATIC_DIR / "app.html"
    if app_file.is_file():
        return FileResponse(app_file)
    raise HTTPException(status_code=404, detail="App page not found")


@app.get("/shield", tags=["Discovery & Info"])
async def shield_page():
    """Serve the Shield SDK landing page."""
    return FileResponse(STATIC_DIR / "shield.html")


@app.get("/verify", tags=["Discovery & Info"])
async def verify_page():
    """Serve the AHM Verify landing page."""
    return FileResponse(STATIC_DIR / "verify.html")


@app.get("/roadmap", tags=["Discovery & Info"])
async def roadmap():
    """Serve the roadmap page."""
    return FileResponse(STATIC_DIR / "ahm-roadmap.html")


@app.get("/.well-known/agent-registration.json", tags=["Discovery & Info"])
async def agent_registration():
    """ERC-8004 agent registration document."""
    return FileResponse(STATIC_DIR / "agent-registration.json", media_type="application/json")


@app.get("/.well-known/agent.json", tags=["Discovery & Info"])
async def a2a_agent_card():
    """A2A Agent Card — public discovery endpoint per A2A protocol spec."""
    return FileResponse(
        pathlib.Path(__file__).parent / "docs" / "ecosystem" / "agent-card.json",
        media_type="application/json",
    )


@app.get("/.well-known/402index-verify.txt", tags=["Discovery & Info"])
async def verify_402index():
    """402index.io domain verification token."""
    return Response(
        content="d259ab24f09ce339920742368a801a04794b8b4c7c9d72bdedb39ffd88881ed5",
        media_type="text/plain",
    )


@app.get("/.well-known/x402", tags=["Discovery & Info"])
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
    # Also force https for custom domain behind Railway proxy
    if base_url.startswith("http://") and request.headers.get("x-forwarded-proto") == "https":
        base_url = "https://" + base_url[7:]

    endpoints = []
    # x402scan-compatible resources array (flat list of endpoint URLs)
    resources = []

    for route_pattern, config in x402_routes.items():
        parts = route_pattern.split(" ", 1)
        method = parts[0] if len(parts) == 2 else "GET"
        path = parts[1] if len(parts) == 2 else parts[0]

        resource_url = f"{base_url}{path}"
        resources.append(resource_url)

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
            "url": resource_url,
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
        "version": 1,
        "service": "Agent Health Monitor",
        "description": (
            "Pay-per-use API that analyzes Base blockchain agent wallets "
            "for transaction failures, gas waste, and optimization opportunities."
        ),
        "facilitator": FACILITATOR_URL,
        # x402scan v2 discovery: flat resources array for fan-out probing
        "resources": resources,
        # Extended endpoint metadata (Bazaar-compatible)
        "endpoints": endpoints,
    }


@app.get("/api/info", tags=["Discovery & Info"])
async def api_info():
    """Service info and pricing."""
    return {
        "service": "Agent Health Monitor",
        "version": "1.8.0",
        "network": "Base L2",
        "endpoints": {
            "GET /risk/{address}": f"{RISK_PRICE} USDC — quick risk score for pre-flight checks",
            "GET /risk/premium/{address}": f"{PREMIUM_RISK_PRICE} USDC — premium risk score with Nansen labels, PnL summary + operational health (tx failure rate)",
            "GET /counterparties/{address}": f"{COUNTERPARTY_PRICE} USDC — top counterparties enriched with Nansen labels",
            "GET /network-map/{address}": f"{NETWORK_MAP_PRICE} USDC — related wallets (funders, deployers, multisig) with Nansen labels",
            "GET /health/{address}": f"{PRICE} USDC — wallet health diagnosis",
            "POST /wash/{address}": f"{WASH_PRICE} USDC — agent hygiene scan (dust, spam, gas efficiency, failure patterns)",
            "GET /ahs/{address}": f"{AHS_PRICE} USDC — Agent Health Score (composite 0-100 index across wallet hygiene, behavioural patterns, and infrastructure health)",
            "POST /ahs/batch": f"{AHS_BATCH_PRICE} USDC — batch AHS scoring (up to 10 wallets per x402 call, up to 25 via API key at {AHS_PRICE}/wallet)",
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


@app.get("/api/ecosystem-stats", tags=["Discovery & Info"])
async def ecosystem_stats():
    """Public aggregate ecosystem stats for the dashboard. No auth required."""
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, scan_db.get_ecosystem_dashboard_stats)
    stats["endpoint_count"] = ENDPOINT_COUNT
    return stats


@app.get("/scan/quality", tags=["Discovery & Info"])
async def scan_quality():
    """Batch quality history for the last 30 days. No auth required."""
    loop = asyncio.get_running_loop()
    history = await loop.run_in_executor(None, scan_db.get_batch_quality_history)
    return {"batches": history}


@app.get("/dashboard", tags=["Discovery & Info"])
async def dashboard():
    """Serve the public ecosystem health dashboard."""
    dash_file = STATIC_DIR / "dashboard.html"
    if dash_file.is_file():
        return FileResponse(dash_file)
    raise HTTPException(status_code=404, detail="Dashboard not found")


# -- Endpoint Marketing Pages ------------------------------------------------

ENDPOINT_PAGES = {
    "ahs": {
        "name": "Agent Health Score",
        "slug": "ahs",
        "price": "$1.00",
        "tagline": "The definitive health score for autonomous agents",
        "description": "A composite 0-100 score blending wallet hygiene, behavioural patterns, and infrastructure health. Detects anomalies including Zombie Agent, Cascading Failure, and Phantom Activity patterns.",
        "who_its_for": ["Agent operators", "DeFi protocols", "Agent marketplaces", "x402 providers"],
        "what_you_get": ["Composite AHS score 0-100", "Grade A-F with label", "D1 wallet hygiene score", "D2 behavioural score", "Cross-dimensional pattern detection", "JWT temporal token for trend tracking"],
        "sample_output": {
            "agent_health_score": 67,
            "grade": "C",
            "grade_label": "Needs Attention",
            "d1_wallet_hygiene": 84,
            "d2_behavioural": 52,
            "patterns_detected": ["Healthy Operator"],
            "confidence": "high",
            "mode": "2D",
        },
        "ecosystem_stat": "avg_ahs",
        "ecosystem_label": "avg AHS across {total} agents scanned",
        "cta": "Score Your Agent — $1.00 USDC",
    },
    "wash": {
        "name": "Agent Wash",
        "slug": "wash",
        "price": "$0.50",
        "tagline": "Deep hygiene scan for agent wallets",
        "description": "Scans for dust accumulation, spam token exposure, failed transaction patterns, and gas efficiency. Returns a cleanliness score 0-100 with actionable findings.",
        "who_its_for": ["Agent operators", "Wallet managers", "Pre-deployment checks"],
        "what_you_get": ["Cleanliness score 0-100", "Dust token count and value", "Spam exposure rating", "Failed tx pattern analysis", "Gas efficiency score", "Actionable recommendations"],
        "sample_output": {
            "cleanliness_score": 78,
            "dust_tokens": 3,
            "dust_value_usd": 0.42,
            "spam_exposure": "low",
            "failed_tx_rate_pct": 2.1,
            "gas_efficiency": "good",
        },
        "ecosystem_stat": "avg_d1",
        "ecosystem_label": "avg hygiene score across {total} agents",
        "cta": "Scan Your Agent — $0.50 USDC",
    },
    "risk": {
        "name": "Agent Risk Score",
        "slug": "risk",
        "price": "$0.01",
        "tagline": "Instant risk assessment for any agent wallet",
        "description": "A fast, lightweight risk score for autonomous agents. Perfect as a pre-transaction check before routing payments or approving interactions.",
        "who_its_for": ["Payment routers", "Agent frameworks", "Pre-flight checks", "High-frequency workflows"],
        "what_you_get": ["Risk score 0-100", "Risk level classification", "Key risk flags", "Sub-second response time"],
        "sample_output": {
            "risk_score": 23,
            "risk_level": "low",
            "flags": [],
            "recommendation": "Safe to transact",
        },
        "ecosystem_stat": "zombie_pct",
        "ecosystem_label": "of agents flagged with critical patterns",
        "cta": "Check Risk — $0.01 USDC",
    },
    "report-card": {
        "name": "Agent Report Card",
        "slug": "report-card",
        "price": "$2.00",
        "tagline": "See how your agent ranks against the ecosystem",
        "description": "A personalised visual report card showing your agent's AHS score, grade, dimension breakdown, and percentile ranking against all scanned agents. Includes a shareable PNG image.",
        "who_its_for": ["Agent operators", "Builders shipping agents", "Anyone curious about their agent's health"],
        "what_you_get": ["Personalised AHS score and grade", "Ecosystem percentile rank", "D1 + D2 dimension breakdown", "Pattern detection results", "Shareable 1200x675 PNG image", "One-click Share on X"],
        "sample_output": {
            "agent_health_score": 72,
            "grade": "C — Needs Attention",
            "percentile_rank": 16,
            "patterns_detected": ["Zombie Agent"],
            "image_url": "/static/report-cards/0xabcd...efgh.png",
        },
        "ecosystem_stat": "total",
        "ecosystem_label": "agents scanned — see how you compare",
        "cta": "Get Your Report Card — $2.00 USDC",
    },
}


@app.get("/api/endpoint-info/{slug}", tags=["Discovery & Info"])
async def endpoint_info(slug: str):
    """Public endpoint metadata for marketing pages. No auth required."""
    page = ENDPOINT_PAGES.get(slug)
    if not page:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {slug}")

    # Attach live ecosystem stat
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, scan_db.get_ecosystem_dashboard_stats)
    total = stats.get("total_scanned", 0)

    stat_key = page.get("ecosystem_stat", "")
    if stat_key == "avg_ahs":
        stat_value = str(stats.get("avg_ahs", 0))
    elif stat_key == "avg_d1":
        stat_value = str(stats.get("avg_d1", 0))
    elif stat_key == "zombie_pct":
        zombie = (stats.get("pattern_distribution") or {}).get("Zombie Agent", 0)
        pct = round(zombie / total * 100) if total > 0 else 0
        stat_value = f"{pct}%"
    elif stat_key == "total":
        stat_value = f"{total:,}"
    else:
        stat_value = str(total)

    label = page.get("ecosystem_label", "").replace("{total}", f"{total:,}")

    return {
        **page,
        "ecosystem_value": stat_value,
        "ecosystem_label": label,
        "all_slugs": list(ENDPOINT_PAGES.keys()),
    }


@app.get("/endpoints/{slug}", tags=["Discovery & Info"])
async def endpoint_page(slug: str):
    """Serve the endpoint marketing page for any valid slug."""
    if slug not in ENDPOINT_PAGES:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {slug}")
    ep_file = STATIC_DIR / "endpoints.html"
    if ep_file.is_file():
        return FileResponse(ep_file)
    raise HTTPException(status_code=404, detail="Endpoint page not found")


# Coupon validation rate limiting: 5 attempts per IP per minute
_coupon_rate: dict[str, list[float]] = {}
COUPON_RATE_LIMIT = 5
COUPON_RATE_WINDOW = 60  # seconds


def _check_coupon_rate(ip: str):
    import time
    now = time.time()
    timestamps = _coupon_rate.get(ip, [])
    timestamps = [t for t in timestamps if now - t < COUPON_RATE_WINDOW]
    if len(timestamps) >= COUPON_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many coupon validation attempts")
    timestamps.append(now)
    _coupon_rate[ip] = timestamps
    # Periodic cleanup
    if len(_coupon_rate) > 1000:
        stale = [k for k, v in _coupon_rate.items() if not v or now - v[-1] > COUPON_RATE_WINDOW]
        for k in stale:
            del _coupon_rate[k]


@app.get("/coupon/validate/{code}", tags=["Coupon Access"])
async def validate_coupon(code: str, request: Request):
    """Check if a coupon code is valid. Rate-limited to 5 attempts per IP per minute."""
    _check_coupon_rate(request.client.host)
    return {"valid": code.strip().upper() in VALID_COUPONS}


def _require_coupon(code: str):
    """Validate a coupon code or raise 403."""
    if code.strip().upper() not in VALID_COUPONS:
        raise HTTPException(status_code=403, detail="Invalid coupon code")


# Coupon access rate limiting: 5 requests per IP per minute (separate from validation)
_coupon_access_rate: dict[str, list[float]] = {}
COUPON_ACCESS_RATE_LIMIT = 5
COUPON_ACCESS_RATE_WINDOW = 60  # seconds


def _check_coupon_access_rate(ip: str):
    import time
    now = time.time()
    timestamps = _coupon_access_rate.get(ip, [])
    timestamps = [t for t in timestamps if now - t < COUPON_ACCESS_RATE_WINDOW]
    if len(timestamps) >= COUPON_ACCESS_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many coupon access requests")
    timestamps.append(now)
    _coupon_access_rate[ip] = timestamps
    if len(_coupon_access_rate) > 1000:
        stale = [k for k, v in _coupon_access_rate.items() if not v or now - v[-1] > COUPON_ACCESS_RATE_WINDOW]
        for k in stale:
            del _coupon_access_rate[k]


@app.get("/coupon/risk/{code}/{address}", tags=["Coupon Access"])
async def coupon_risk(code: str, address: WalletAddress, request: Request):
    """Quick risk score via coupon — mirrors GET /risk/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_risk_score(address)


@app.get("/coupon/health/{code}/{address}", tags=["Coupon Access"])
async def coupon_health(code: str, address: WalletAddress, request: Request):
    """Wallet health diagnosis via coupon — mirrors GET /health/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_health_report(address)


@app.get("/coupon/optimize/{code}/{address}", tags=["Coupon Access"])
async def coupon_optimize(code: str, address: WalletAddress, request: Request):
    """Gas optimization report via coupon — mirrors GET /optimize/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_optimization_report(address)


@app.get("/coupon/retry/{code}/{address}", tags=["Coupon Access"])
async def coupon_retry(code: str, address: WalletAddress, request: Request):
    """Retry bot via coupon — mirrors GET /retry/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_retry_transactions(address)


@app.get("/coupon/protect/{code}/{address}", tags=["Coupon Access"])
async def coupon_protect(code: str, address: WalletAddress, request: Request):
    """Full protection agent via coupon — mirrors GET /agent/protect/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_protection_report(address)


@app.get("/coupon/alerts/{code}/{address}", tags=["Coupon Access"])
async def coupon_alerts(code: str, address: WalletAddress, request: Request):
    """Alert subscription via coupon — mirrors GET /alerts/subscribe/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await subscribe_alerts(address)


@app.get("/coupon/risk-premium/{code}/{address}", tags=["Coupon Access"])
async def coupon_premium_risk(code: str, address: WalletAddress, request: Request):
    """Premium risk score via coupon — mirrors GET /risk/premium/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_premium_risk_score(address)


@app.get("/coupon/counterparties/{code}/{address}", tags=["Coupon Access"])
async def coupon_counterparties(code: str, address: WalletAddress, request: Request):
    """Counterparty analysis via coupon — mirrors GET /counterparties/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_counterparties(address)


@app.get("/coupon/network-map/{code}/{address}", tags=["Coupon Access"])
async def coupon_network_map(code: str, address: WalletAddress, request: Request, chain: str = "ethereum"):
    """Network map via coupon — mirrors GET /network-map/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_network_map(address, chain=chain)


@app.get("/coupon/wash/{code}/{address}", tags=["Coupon Access"])
async def coupon_wash(code: str, address: WalletAddress, request: Request):
    """Hygiene scan via coupon — mirrors POST /wash/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_wash_report(address)


@app.get("/coupon/ahs/{code}/{address}", tags=["Coupon Access"])
async def coupon_ahs(code: str, address: WalletAddress, request: Request):
    """Agent Health Score via coupon — mirrors GET /ahs/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_ahs_report(address, request)


@app.get("/coupon/report-card/{code}/{address}", tags=["Coupon Access"])
async def coupon_report_card(code: str, address: WalletAddress, request: Request):
    """Visual report card via coupon — mirrors GET /report-card/{address}."""
    _check_coupon_access_rate(request.client.host)
    _require_coupon(code)
    return await get_report_card(address, request)


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


# Chat rate limiting: 10 messages per IP per hour
_chat_rate: dict[str, list[float]] = {}
CHAT_RATE_LIMIT = 10
CHAT_RATE_WINDOW = 3600  # seconds


_chat_rate_cleanup_counter = 0

def _check_chat_rate(ip: str):
    global _chat_rate_cleanup_counter
    import time
    now = time.time()
    timestamps = _chat_rate.get(ip, [])
    timestamps = [t for t in timestamps if now - t < CHAT_RATE_WINDOW]
    if len(timestamps) >= CHAT_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    timestamps.append(now)
    _chat_rate[ip] = timestamps
    # Periodic eviction: every 100 calls, purge stale IPs
    _chat_rate_cleanup_counter += 1
    if _chat_rate_cleanup_counter >= 100:
        _chat_rate_cleanup_counter = 0
        stale = [k for k, v in _chat_rate.items() if not v or now - v[-1] > CHAT_RATE_WINDOW]
        for k in stale:
            del _chat_rate[k]


@app.post("/chat", tags=["Utility"])
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


@app.get("/up", tags=["Utility"])
async def up():
    """Unpaid liveness probe for load balancers."""
    return {"status": "ok"}


@app.get("/risk/premium/{address}", response_model=PremiumRiskResponse, tags=["Scoring & Risk"])
@limiter.limit("60/minute")
async def get_premium_risk_score(address: WalletAddress, request: Request):
    """
    Premium risk score enriched with Nansen wallet intelligence, PnL data,
    and operational health metrics (transaction failure rate analysis).

    Requires x402 payment ($0.05 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Returns a 0-100 risk score plus Nansen smart money tags,
    entity labels, behavioral signals, PnL summary, and operational
    health (1hr/24hr revert rates, nonce gaps, volume anomalies).
    PnL data adjusts the risk score: profitable wallets get up to -10,
    unprofitable wallets get up to +10. Operational health adjusts:
    degraded +5, critical +10.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "risk/premium", address, request=request)
    loop = asyncio.get_running_loop()

    # Run risk analysis in parallel with Nansen labels (Corbits, safe to overlap)
    risk_task = asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
    )
    nansen_task = fetch_nansen_labels(address)

    (eth_price, transactions), nansen_raw = await asyncio.gather(risk_task, nansen_task)

    health = await loop.run_in_executor(
        None, partial(analyze_address, address, transactions, eth_price),
    )

    # Nansen PnL — sequential AFTER labels to avoid x402 nonce conflicts
    # (labels goes through Corbits, PnL goes direct to api.nansen.ai;
    #  different facilitators so technically safe, but keep sequential
    #  since both use the same payer wallet/signer)
    pnl_raw = await fetch_nansen_pnl(address)

    # Derive risk score (same logic as /risk)
    risk_score = max(0, min(100, 100 - int(health.health_score)))

    # Adjust risk score based on PnL profitability (up to +/-10 points)
    pnl_summary: Optional[PnlSummary] = None
    pnl_available = pnl_raw is not None
    if pnl_raw:
        realized_pnl = float(pnl_raw.get("realized_pnl_usd", 0))
        # Scale: $10k+ profit → full -10, $10k+ loss → full +10, linear between
        pnl_magnitude = min(abs(realized_pnl) / 10_000.0, 1.0)
        pnl_adjustment = int(pnl_magnitude * 10)
        if realized_pnl > 0:
            risk_score = max(0, risk_score - pnl_adjustment)
        elif realized_pnl < 0:
            risk_score = min(100, risk_score + pnl_adjustment)

        top_tokens = []
        for tok in pnl_raw.get("top5_tokens", []):
            if isinstance(tok, dict):
                top_tokens.append(PnlTokenSummary(
                    token_symbol=_clean_label(tok.get("token_symbol", "")),
                    chain=tok.get("chain", ""),
                    realized_pnl=float(tok.get("realized_pnl", 0)),
                    realized_roi=float(tok.get("realized_roi", 0)),
                ))

        pnl_summary = PnlSummary(
            realized_pnl_usd=realized_pnl,
            realized_pnl_percent=float(pnl_raw.get("realized_pnl_percent", 0)),
            win_rate=float(pnl_raw.get("win_rate", 0)),
            traded_token_count=int(pnl_raw.get("traded_token_count", 0)),
            traded_times=int(pnl_raw.get("traded_times", 0)),
            top_tokens=top_tokens,
        )

    # Operational health from transaction failure metrics
    op_health = get_tx_failure_metrics(address, transactions)

    # Adjust risk score based on operational health
    if op_health.health_status == "critical":
        risk_score = min(100, risk_score + 10)
    elif op_health.health_status == "degraded":
        risk_score = min(100, risk_score + 5)

    # Recompute risk level after PnL + operational health adjustments
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
    if op_health.health_status in ("degraded", "critical"):
        signals.append(f"operational health {op_health.health_status}")

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
                    label=_clean_label(item.get("label", item.get("name", str(item)))),
                    category=item.get("category"),
                    definition=item.get("definition"),
                ))

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="risk_premium",
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        risk_score=risk_score,
        response_data={"risk_score": risk_score, "risk_level": risk_level, "verdict": verdict},
    ))
    return PremiumRiskResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        verdict=verdict,
        nansen_labels=nansen_labels,
        nansen_available=nansen_available,
        pnl_summary=pnl_summary,
        pnl_available=pnl_available,
        operational_health=op_health,
    )


@app.get("/counterparties/{address}", response_model=CounterpartyResponse, tags=["Scoring & Risk"])
@limiter.limit("60/minute")
async def get_counterparties(address: WalletAddress, request: Request):
    """
    Know Your Counterparty — top wallets/contracts this address interacts with.

    Requires x402 payment ($0.10 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Returns the top counterparties ranked by interaction count, enriched
    with Nansen labels where available.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "counterparties", address, request=request)

    raw = await fetch_nansen_counterparties(address)

    logging.info(
        "/counterparties [%s] raw=%s",
        address,
        type(raw).__name__ if not isinstance(raw, list) else f"list[{len(raw)}]",
    )

    counterparties = []
    nansen_available = raw is not None
    if raw:
        for item in raw:
            if isinstance(item, dict):
                # Nansen returns labels as an array; join into a single string
                raw_labels = item.get("counterparty_address_label", item.get("labels", []))
                if isinstance(raw_labels, list) and raw_labels:
                    label = ", ".join(
                        _clean_label(str(l))
                        for l in raw_labels if l
                    ) or None
                elif isinstance(raw_labels, str) and raw_labels:
                    label = _clean_label(raw_labels) or None
                else:
                    label = item.get("label", item.get("name"))

                counterparties.append(Counterparty(
                    address=item.get("counterparty_address", item.get("address", "")),
                    label=label,
                    interaction_count=int(item.get("interaction_count", item.get("interactionCount", 0))),
                    volume_usd=float(item.get("total_volume_usd", item.get("volumeUsd", item.get("volume", 0)))),
                    last_interaction=item.get("last_interaction_date", item.get("lastInteraction", item.get("last_interaction"))),
                ))

    logging.info(
        "/counterparties [%s] result: nansen_available=%s count=%d",
        address, nansen_available, len(counterparties),
    )

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="counterparties",
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        response_data={"total_counterparties": len(counterparties)},
    ))
    return CounterpartyResponse(
        status="ok",
        address=address,
        counterparties=counterparties,
        total_counterparties=len(counterparties),
        nansen_available=nansen_available,
    )


@app.get("/network-map/{address}", response_model=RelatedWalletsResponse, tags=["Scoring & Risk"])
@limiter.limit("60/minute")
async def get_network_map(address: WalletAddress, request: Request, chain: str = "ethereum"):
    """
    Wallet Network Map — related wallets linked by funding, deployment, or multisig.

    Requires x402 payment ($0.10 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Returns wallets connected to this address via first-funder relationships,
    contract deployments, and multisig co-signer links, enriched with Nansen labels.

    Query params:
        chain: blockchain to query (default: ethereum). Does NOT support "all".
               Valid: arbitrum, avalanche, base, bnb, ethereum, polygon, solana, etc.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    chain = chain.lower().strip()
    if chain not in VALID_NANSEN_CHAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chain '{chain}'. Valid options: {', '.join(sorted(VALID_NANSEN_CHAINS))}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "network-map", address, request=request)

    raw = await fetch_nansen_related_wallets(address, chain=chain)

    logging.info(
        "/network-map [%s] chain=%s raw=%s",
        address, chain,
        type(raw).__name__ if not isinstance(raw, list) else f"list[{len(raw)}]",
    )

    related_wallets = []
    nansen_available = raw is not None
    if raw:
        for item in raw:
            if isinstance(item, dict):
                label_raw = item.get("address_label", item.get("label"))
                label = _clean_label(str(label_raw)) if label_raw else None

                related_wallets.append(RelatedWallet(
                    address=item.get("address", ""),
                    label=label,
                    relation=item.get("relation", "unknown"),
                    chain=item.get("chain", chain),
                    transaction_hash=item.get("transaction_hash"),
                    block_timestamp=item.get("block_timestamp"),
                ))

    logging.info(
        "/network-map [%s] result: nansen_available=%s count=%d",
        address, nansen_available, len(related_wallets),
    )

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="network_map",
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        response_data={"total_related": len(related_wallets)},
    ))
    return RelatedWalletsResponse(
        status="ok",
        address=address,
        chain=chain,
        related_wallets=related_wallets,
        total_related=len(related_wallets),
        nansen_available=nansen_available,
    )


@app.get("/risk/{address}", response_model=RiskResponse, tags=["Scoring & Risk"])
@limiter.limit("60/minute")
async def get_risk_score(address: WalletAddress, request: Request):
    """
    Quick risk score for agent pre-flight checks.

    Requires x402 payment ($0.001 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

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

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "risk", address, request=request)

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

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="risk",
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        risk_score=risk_score,
        response_data={"risk_score": risk_score, "risk_level": risk_level, "verdict": verdict},
    ))
    return RiskResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        verdict=verdict,
    )


@app.get("/health/{address}", response_model=HealthResponse, tags=["Health & Hygiene"])
@limiter.limit("60/minute")
async def get_health_report(address: WalletAddress, request: Request):
    """
    Analyze a Base wallet address and return a health report.

    Requires x402 payment ($0.50 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

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

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "health", address, request=request)

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
                    label=_clean_label(item.get("label", item.get("name", str(item)))),
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

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="health",
        scan_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        health_score=report.health_score if hasattr(report, "health_score") else None,
        response_data={"status": "ok", "total_portfolio_usd": total_portfolio_usd},
    ))
    return HealthResponse(
        status="ok",
        report=report,
        nansen_labels=nansen_labels,
        nansen_available=nansen_available,
        token_balances=token_balances,
        total_portfolio_usd=total_portfolio_usd,
    )


@app.post("/wash/{address}", response_model=WashResponse, tags=["Health & Hygiene"])
@limiter.limit("60/minute")
async def get_wash_report(address: WalletAddress, request: Request):
    """
    Agent Wash: Hygiene scan for Base wallet addresses.

    Requires x402 payment ($0.50 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Scans for:
    - Dust tokens (< $0.01 value)
    - Spam tokens (URL names, low holders, zero volume)
    - Gas efficiency (gasUsed/gas ratio analysis)
    - Failed transaction patterns (repeated failures, retry storms)
    - Nonce gaps

    Returns a cleanliness score (0-100) with prioritised cleanup recommendations.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "wash", address, request=request)

    loop = asyncio.get_running_loop()

    # Parallel fetch: tokens (V2 API), transactions (etherscan-compat), ETH price
    eth_price, transactions, tokens = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_tokens_v2, address)),
    )

    wash = await loop.run_in_executor(
        None, partial(analyze_wash, address, tokens, transactions, eth_price),
    )

    report = WashReport(
        address=wash.address,
        cleanliness_score=wash.cleanliness_score,
        cleanliness_grade=wash.cleanliness_grade,
        total_issues=wash.total_issues,
        issues_by_severity=wash.issues_by_severity,
        dust_tokens=wash.dust_tokens,
        dust_total_usd=wash.dust_total_usd,
        spam_tokens=wash.spam_tokens,
        spam_token_list=wash.spam_token_list,
        gas_efficiency_pct=wash.gas_efficiency_pct,
        gas_efficiency_grade=wash.gas_efficiency_grade,
        wasted_gas_usd=wash.wasted_gas_usd,
        failed_tx_count_24hr=wash.failed_tx_count_24hr,
        failed_tx_patterns=wash.failed_tx_patterns,
        nonce_gaps=wash.nonce_gaps,
        issues=[WashIssue(**i) for i in wash.issues],
        recommendations=wash.recommendations,
        scan_timestamp=wash.scan_timestamp,
        next_wash_recommended=wash.next_wash_recommended,
    )

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="wash",
        scan_timestamp=wash.scan_timestamp,
        cleanliness_score=wash.cleanliness_score,
        response_data={"cleanliness_score": wash.cleanliness_score, "total_issues": wash.total_issues},
    ))
    return WashResponse(status="ok", report=report)


# -- Agent Health Score (AHS) Endpoint --------------------------------------

@app.get("/ahs/{address}", response_model=AHSResponse, tags=["Health & Hygiene"])
@limiter.limit("60/minute")
async def get_ahs_report(address: WalletAddress, request: Request):
    """
    Agent Health Score: Composite 0-100 index for on-chain agent wallets.

    Requires x402 payment ($1.00 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Three dimensions:
    - D1 Wallet Hygiene (dust, spam, gas efficiency, failure rate, nonce gaps)
    - D2 Behavioural Patterns (repeated failures, gas adaptation, timing, diversity, retry storms)
    - D3 Infrastructure Health (optional — provide ?agent_url= to enable 3D mode)

    Plus cross-dimensional pattern detection (Zombie Agent, Cascading Failure, etc.)
    and temporal scoring via X-AHS-Previous JWT header.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "ahs", address, request=request)

    loop = asyncio.get_running_loop()

    # Optional agent_url for D3 probing
    agent_url = request.query_params.get("agent_url")
    if agent_url:
        if not _is_safe_webhook_url(agent_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid agent_url. Must be a public https/http URL (no private IPs or internal hosts).",
            )

    # Parse previous AHS JWT token for temporal scoring
    previous_score = None
    previous_ema = None
    scan_count = 1
    prev_token = request.headers.get("X-AHS-Previous")
    if prev_token:
        try:
            payload = jwt.decode(prev_token, AHS_JWT_SECRET, algorithms=["HS256"])
            if payload.get("address", "").lower() == address:
                previous_score = payload.get("score")
                previous_ema = payload.get("ema")
                scan_count = payload.get("scan_count", 1) + 1
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            pass  # Ignore invalid tokens, proceed without temporal data

    # Parallel fetch: tokens, transactions, ETH price
    eth_price, transactions, tokens = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_tokens_v2, address)),
    )

    # Run scoring engine
    result = await loop.run_in_executor(
        None,
        partial(
            calculate_ahs,
            address=address,
            tokens=tokens,
            transactions=transactions,
            eth_price=eth_price,
            agent_url=agent_url,
            previous_score=previous_score,
            previous_ema=previous_ema,
            scan_count=scan_count,
        ),
    )

    # Build dimension scores list
    dimensions = [
        AHSDimensionScore(
            dimension="D1: Wallet Hygiene",
            score=result.d1_score,
            weight=result.d1_weight,
            contributing_factors=result.d1_top_factors,
        ),
        AHSDimensionScore(
            dimension="D2: Behavioural Patterns",
            score=result.d2_score,
            weight=result.d2_weight,
            contributing_factors=result.d2_top_factors,
        ),
    ]
    if result.d3_score is not None:
        dimensions.append(
            AHSDimensionScore(
                dimension="D3: Infrastructure Health",
                score=result.d3_score,
                weight=result.d3_weight,
                contributing_factors=result.d3_top_factors,
            )
        )

    # Build patterns list
    patterns = [
        AHSCrossDimensionalPattern(
            name=p["name"],
            detected=p["detected"],
            severity=p["severity"],
            description=p["description"],
        )
        for p in result.patterns_detected
    ]

    # Generate signed JWT token for temporal scoring
    token_payload = {
        "address": address,
        "score": result.agent_health_score,
        "ema": result.temporal_score if result.temporal_score is not None else float(result.agent_health_score),
        "scan_count": scan_count,
        "ts": result.scan_timestamp,
        "exp": int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp()),
    }
    ahs_token = jwt.encode(token_payload, AHS_JWT_SECRET, algorithm="HS256")

    # Extract shadow signals for monitoring
    sigs = result._signals
    shadow = AHSShadowSignals(
        session_continuity_score=sigs.get("session_continuity_score"),
        abrupt_sessions=sigs.get("abrupt_sessions", 0),
        budget_exhaustion_count=sigs.get("budget_exhaustion_count", 0),
        total_sessions=sigs.get("total_sessions", 0),
        avg_session_length=sigs.get("avg_session_length", 0.0),
        shadow_patterns=sigs.get("shadow_patterns", []),
    )

    report = AHSReport(
        address=address,
        agent_health_score=result.agent_health_score,
        grade=f"{result.grade} — {result.grade_label}",
        confidence=result.confidence,
        mode=result.mode,
        dimensions=dimensions,
        patterns_detected=patterns,
        trend=result.trend,
        routing_recommendation=_trust_routing(result.grade),
        recommendations=result.recommendations,
        ahs_token=ahs_token,
        model_version=result.model_version,
        scan_timestamp=result.scan_timestamp,
        next_scan_recommended=result.next_scan_recommended,
        shadow_signals=shadow,
    )

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="ahs",
        scan_timestamp=result.scan_timestamp,
        ahs_score=result.agent_health_score,
        grade=result.grade, grade_label=result.grade_label,
        confidence=result.confidence, mode=result.mode,
        d1_score=result.d1_score, d2_score=result.d2_score,
        d3_score=result.d3_score, cdp_modifier=result.cdp_modifier,
        patterns=[{"name": p.get("name", ""), "severity": p.get("severity", ""),
                   "description": p.get("description", ""), "modifier": p.get("modifier")}
                  for p in result.patterns_detected] if result.patterns_detected else None,
        tx_count=result.tx_count, history_days=result.history_days,
        response_data={"agent_health_score": result.agent_health_score, "grade": result.grade,
                       "d1_score": result.d1_score, "d2_score": result.d2_score},
        shadow_signals=shadow.model_dump(),
    ))
    return AHSResponse(status="ok", report=report)


# ── AHS Routing Policy endpoints ──────────────────────────────────────────
# IMPORTANT: These must be registered BEFORE /ahs/route/{address} so FastAPI
# does not capture "policy" as an address parameter.


@app.get("/ahs/route/policy", response_model=RoutingPolicyResponse, tags=["Health & Hygiene"])
@limiter.limit("60/minute")
async def get_routing_policy(request: Request):
    """Return the caller's current routing policy configuration.

    Requires X-API-Key or x402 payment for authentication. Returns default
    thresholds if no custom policy has been configured.
    """
    owner_id, _ = _resolve_policy_owner(request)

    policy = await asyncio.get_running_loop().run_in_executor(
        None, scan_db.get_routing_policy, owner_id,
    )
    allowlist = await asyncio.get_running_loop().run_in_executor(
        None, scan_db.get_routing_allowlist, owner_id,
    )

    if policy is None:
        # Return defaults
        return RoutingPolicyResponse(
            instant_grades=["A", "B"],
            escrow_grades=["C"],
            reject_grades=["D", "E", "F"],
            escrow_disabled=False,
            allowlist_count=len(allowlist),
            updated_at="",
        )

    return RoutingPolicyResponse(
        instant_grades=[g.strip() for g in policy["instant_grades"].split(",") if g.strip()],
        escrow_grades=[g.strip() for g in policy["escrow_grades"].split(",") if g.strip()],
        reject_grades=[g.strip() for g in policy["reject_grades"].split(",") if g.strip()],
        escrow_disabled=bool(policy["escrow_disabled"]),
        allowlist_count=len(allowlist),
        updated_at=policy["updated_at"],
    )


@app.put("/ahs/route/policy", response_model=RoutingPolicyResponse, tags=["Health & Hygiene"])
@limiter.limit("20/minute")
async def put_routing_policy(body: RoutingPolicyRequest, request: Request):
    """Create or update the caller's routing policy.

    Requires X-API-Key or x402 payment ($0.01). Validates that:
    - All 6 grades (A-F) are assigned to exactly one category
    - escrow_disabled consistency (no escrow grades when disabled)
    - Allowlist addresses are valid, max 1000
    - No self-allowlisting (caller's own address)
    - Allowlisted addresses must have AHS Grade C or above
    """
    owner_id, caller_address = _resolve_policy_owner(request)
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "ahs/route/policy", request=request)

    _validate_routing_policy(body, caller_address)

    instant_str = ",".join(body.instant_grades)
    escrow_str = ",".join(body.escrow_grades)
    reject_str = ",".join(body.reject_grades)

    policy = await asyncio.get_running_loop().run_in_executor(
        None,
        scan_db.upsert_routing_policy,
        owner_id, instant_str, escrow_str, reject_str, body.escrow_disabled,
    )

    allowlist_count = 0
    if body.allowlist is not None:
        allowlist_count = await asyncio.get_running_loop().run_in_executor(
            None, scan_db.set_routing_allowlist, owner_id, body.allowlist,
        )
    else:
        existing = await asyncio.get_running_loop().run_in_executor(
            None, scan_db.get_routing_allowlist, owner_id,
        )
        allowlist_count = len(existing)

    return RoutingPolicyResponse(
        instant_grades=body.instant_grades,
        escrow_grades=body.escrow_grades,
        reject_grades=body.reject_grades,
        escrow_disabled=body.escrow_disabled,
        allowlist_count=allowlist_count,
        updated_at=policy["updated_at"],
    )


# ── AHS Trust Route endpoint ──────────────────────────────────────────────


@app.get("/ahs/route/{address}", response_model=TrustRouteResponse, tags=["Health & Hygiene"])
@limiter.limit("60/minute")
async def get_trust_route(address: WalletAddress, request: Request):
    """Lightweight trust routing signal from the most recent cached AHS score.

    Returns a routing recommendation (instant_settle / escrow / reject) without
    re-running the full AHS scoring pipeline.  Useful for payment gateways and
    agent orchestrators that need a fast trust check.
    """
    # Fiat API key path — decrement + attribute to partner
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "ahs/route", address, request=request)

        # Enforce Shield subscription quota (if the key has one)
        key_hash = fiat_key["key_hash"]
        sub = await asyncio.get_running_loop().run_in_executor(
            None, scan_db.get_shield_subscription, key_hash,
        )
        if sub:
            if sub["calls_used_this_period"] >= sub["call_quota"]:
                raise HTTPException(
                    status_code=429,
                    detail="Shield call quota exceeded for this billing period",
                )
            await asyncio.get_running_loop().run_in_executor(
                None, scan_db.increment_shield_usage, key_hash,
            )

    # Resolve caller's routing policy (if any)
    routing_policy = None
    is_allowlisted = False
    owner_id = None
    if fiat_key:
        owner_id = fiat_key["key_hash"]
    else:
        payer = _get_x402_payer(request)
        if payer:
            owner_id = payer.lower()

    if owner_id:
        routing_policy = await asyncio.get_running_loop().run_in_executor(
            None, scan_db.get_routing_policy, owner_id,
        )
        if routing_policy:
            is_allowlisted = await asyncio.get_running_loop().run_in_executor(
                None, scan_db.is_address_allowlisted, owner_id, address,
            )

    record = await asyncio.get_running_loop().run_in_executor(
        None, scan_db.get_latest_ahs_for_address, address,
    )
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No score available — call /ahs/{address} first",
        )

    scored_at = record["last_scanned_at"] or ""
    stale = False
    if scored_at:
        try:
            scored_dt = datetime.fromisoformat(scored_at.replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - scored_dt).total_seconds() > 86400
        except (ValueError, TypeError):
            stale = True

    grade_letter = (record["latest_grade"] or "F").split()[0]
    return TrustRouteResponse(
        address=record["address"],
        agent_health_score=record["latest_ahs"],
        grade=record["latest_grade"],
        routing_recommendation=_trust_routing_with_policy(
            grade_letter, routing_policy, is_allowlisted,
        ),
        confidence=record.get("confidence") or "unknown",
        scored_at=scored_at,
        stale=stale,
        policy_applied=routing_policy is not None,
        allowlisted=is_allowlisted,
    )


# ── AHS Batch endpoint ─────────────────────────────────────────────────────

_BATCH_SEMAPHORE = asyncio.Semaphore(3)  # max 3 concurrent RPC calls


@app.post("/ahs/batch", response_model=AHSBatchResponse, tags=["Health & Hygiene"])
@limiter.limit("20/minute")
async def get_ahs_batch(body: AHSBatchRequest, request: Request):
    """
    Batch Agent Health Score: score multiple wallets in a single call.

    Accepts up to 25 wallet addresses. Pricing:
    - **x402 path:** $10.00 USDC flat for up to 10 wallets per call.
    - **API key path:** 1 credit per wallet scored. Supports partial results
      if credits are insufficient (scores as many as credits allow).

    Results include the same AHS scoring as the single GET /ahs/{address}
    endpoint — composite 0-100 score with D1/D2 dimension breakdown,
    pattern detection, and recommendations.
    """
    # ── Validate & normalise input ──────────────────────────────────────
    if not body.addresses:
        raise HTTPException(status_code=400, detail="addresses array is required and must not be empty")

    page_size = max(1, min(25, body.page_size))
    page = max(1, body.page)

    # Validate all addresses upfront (before normalisation, so errors show raw input)
    stripped = [a.strip() for a in body.addresses]
    invalid = [a for a in stripped if not ADDRESS_RE.match(a)]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address(es): {', '.join(invalid[:5])}",
        )

    # Deduplicate (case-insensitive) while preserving original form
    seen: set[str] = set()
    unique: list[str] = []
    for a in stripped:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    addresses = unique
    total_addresses = len(addresses)

    # Paginate
    start = (page - 1) * page_size
    page_addrs = addresses[start:start + page_size]

    if not page_addrs:
        return AHSBatchResponse(
            results=[], page=page, page_size=page_size,
            total_addresses=total_addresses, total_scored=0,
            credits_used=0, errors=["Page out of range"],
        )

    # ── Payment logic ───────────────────────────────────────────────────
    fiat_key = validate_fiat_request(request)
    errors: list[str] = []
    credits_remaining: int | None = None

    if fiat_key:
        # API key path: cap to available credits if needed
        available = fiat_key.get("calls_remaining")
        if available is not None:
            if available < len(page_addrs):
                scored_count = max(0, available)
                page_addrs = page_addrs[:scored_count]
                if not page_addrs:
                    return AHSBatchResponse(
                        results=[], page=page, page_size=page_size,
                        total_addresses=total_addresses, total_scored=0,
                        credits_used=0, credits_remaining=0,
                        errors=["Insufficient credits: 0 remaining"],
                    )
                errors.append(
                    f"Partial results: only {len(page_addrs)} of "
                    f"{total_addresses} addresses scored ({available} credits remaining)"
                )
    else:
        # x402 path: payment already settled by middleware for $10 (up to 10 wallets)
        if len(page_addrs) > 10:
            page_addrs = page_addrs[:10]
            errors.append(
                "x402 batch limited to 10 wallets per call. "
                "Use an API key (X-API-Key header) for batches of up to 25."
            )

    # ── Score wallets concurrently ──────────────────────────────────────
    loop = asyncio.get_running_loop()

    async def _score_one(addr: str) -> tuple[str, object | None, str | None]:
        """Score a single wallet, respecting the concurrency semaphore."""
        async with _BATCH_SEMAPHORE:
            try:
                eth_price, txs, tokens = await asyncio.gather(
                    loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
                    loop.run_in_executor(None, partial(fetch_transactions, addr, BASESCAN_API_KEY)),
                    loop.run_in_executor(None, partial(fetch_tokens_v2, addr)),
                )
                result = await loop.run_in_executor(
                    None,
                    partial(
                        calculate_ahs,
                        address=addr,
                        tokens=tokens,
                        transactions=txs,
                        eth_price=eth_price,
                    ),
                )
                return addr, result, None
            except Exception as exc:
                return addr, None, f"{addr}: {exc}"

    raw = await asyncio.gather(*[_score_one(a) for a in page_addrs])

    # ── Assemble results ────────────────────────────────────────────────
    results: list[AHSBatchResultItem] = []
    scored = 0

    for addr, result, err in raw:
        if err or result is None:
            if err:
                errors.append(err)
            continue
        scored += 1

        # Primary detected pattern
        primary_pattern = "No Pattern"
        if result.patterns_detected:
            for p in result.patterns_detected:
                if p.get("detected"):
                    primary_pattern = p["name"]
                    break

        results.append(AHSBatchResultItem(
            address=addr,
            ahs_score=result.agent_health_score,
            grade=f"{result.grade} — {result.grade_label}",
            d1_score=result.d1_score,
            d2_score=result.d2_score,
            pattern=primary_pattern,
            verdict=result.recommendations[0] if result.recommendations else "No issues detected",
            routing_recommendation=_trust_routing(result.grade),
        ))

        # Fire-and-forget: persist scan to DB
        loop.run_in_executor(None, lambda a=addr, r=result: scan_db.log_scan(
            address=a, endpoint="ahs",
            scan_timestamp=r.scan_timestamp,
            ahs_score=r.agent_health_score,
            grade=r.grade, grade_label=r.grade_label,
            confidence=r.confidence, mode=r.mode,
            d1_score=r.d1_score, d2_score=r.d2_score,
            d3_score=r.d3_score, cdp_modifier=r.cdp_modifier,
            patterns=[{"name": p.get("name", ""), "severity": p.get("severity", ""),
                       "description": p.get("description", ""), "modifier": p.get("modifier")}
                      for p in r.patterns_detected] if r.patterns_detected else None,
            tx_count=r.tx_count, history_days=r.history_days,
            response_data={"agent_health_score": r.agent_health_score, "grade": r.grade,
                           "d1_score": r.d1_score, "d2_score": r.d2_score},
        ))

    # ── Consume API key credits ─────────────────────────────────────────
    if fiat_key and scored > 0:
        key_hash = fiat_key["key_hash"]
        batch_partner_id = request.headers.get("X-Partner-Id") or fiat_key.get("partner_id")
        for i, item in enumerate(results):
            scan_db.decrement_api_key(key_hash)
            scan_db.log_api_key_usage(key_hash, "ahs_batch", item.address, partner_id=batch_partner_id)
        available = fiat_key.get("calls_remaining")
        credits_remaining = max(0, available - scored) if available is not None else None

    return AHSBatchResponse(
        results=results,
        page=page,
        page_size=page_size,
        total_addresses=total_addresses,
        total_scored=scored,
        credits_used=scored,
        credits_remaining=credits_remaining,
        errors=errors,
    )


# ── Report Card endpoint ────────────────────────────────────────────────────

@app.get("/report-card/{address}", response_model=ReportCardResponse, tags=["Health & Hygiene"])
@limiter.limit("10/minute")
async def get_report_card(address: WalletAddress, request: Request):
    """
    Agent Report Card: Visual health report card with ecosystem benchmarks.

    Requires x402 payment ($2.00 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

    Runs a full AHS scan, pulls ecosystem comparison data, and generates
    a personalised 1200x675 PNG report card image. Returns JSON with scores,
    benchmarks, and a URL to the generated image.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Ethereum address format: {address}",
        )

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "report-card", address, request=request)

    loop = asyncio.get_running_loop()

    # Parallel fetch: tokens, transactions, ETH price (same as AHS)
    eth_price, transactions, tokens = await asyncio.gather(
        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_transactions, address, BASESCAN_API_KEY)),
        loop.run_in_executor(None, partial(fetch_tokens_v2, address)),
    )

    # Run scoring engine + fetch ecosystem stats in parallel
    result, eco_stats = await asyncio.gather(
        loop.run_in_executor(
            None,
            partial(
                calculate_ahs,
                address=address,
                tokens=tokens,
                transactions=transactions,
                eth_price=eth_price,
            ),
        ),
        loop.run_in_executor(None, scan_db.get_trust_registry_stats),
    )

    # Generate report card image
    png_bytes = await loop.run_in_executor(
        None,
        partial(
            generate_report_card,
            address=address,
            ahs_score=result.agent_health_score,
            grade=result.grade,
            grade_label=result.grade_label,
            d1_score=result.d1_score,
            d2_score=result.d2_score,
            d3_score=result.d3_score,
            mode=result.mode,
            patterns=result.patterns_detected,
            recommendations=result.recommendations,
            ecosystem=eco_stats,
        ),
    )

    # Save PNG to static directory
    img_filename = f"{address}.png"
    img_path = STATIC_DIR / "report-cards" / img_filename
    await loop.run_in_executor(None, lambda: img_path.write_bytes(png_bytes))
    image_url = f"/static/report-cards/{img_filename}"

    # Calculate percentile rank
    percentiles = eco_stats.get("baseline_calibration", {}).get("score_percentiles", {})
    pct_rank = _report_card_percentile(result.agent_health_score, percentiles)

    # Build share URL
    grade_full = f"{result.grade} — {result.grade_label}"
    share_text = (
        f"My agent scored {result.agent_health_score}/100 ({grade_full}) "
        f"on @AHM_xyz Report Card — Top {100 - pct_rank}% of agents on Base"
    )
    share_url = f"https://x.com/intent/tweet?text={url_quote(share_text)}"

    # Build dimension list
    dimensions = [
        ReportCardDimension(dimension="D1: Wallet Hygiene", score=result.d1_score, weight=result.d1_weight),
        ReportCardDimension(dimension="D2: Behavioural Patterns", score=result.d2_score, weight=result.d2_weight),
    ]
    if result.d3_score is not None:
        dimensions.append(
            ReportCardDimension(dimension="D3: Infrastructure Health", score=result.d3_score, weight=result.d3_weight)
        )

    # Build patterns list
    patterns = [
        AHSCrossDimensionalPattern(
            name=p["name"], detected=p["detected"],
            severity=p["severity"], description=p["description"],
        )
        for p in result.patterns_detected
    ]

    eco_comparison = EcosystemComparison(
        average_ahs=eco_stats.get("average_ahs"),
        percentile_rank=pct_rank,
        grade_distribution=eco_stats.get("grade_distribution", {}),
        total_agents_scored=eco_stats.get("summary", {}).get("total_unique_addresses", 0),
    )

    report = ReportCardReport(
        address=address,
        agent_health_score=result.agent_health_score,
        grade=grade_full,
        confidence=result.confidence,
        mode=result.mode,
        dimensions=dimensions,
        patterns_detected=patterns,
        recommendations=result.recommendations,
        ecosystem_comparison=eco_comparison,
        image_url=image_url,
        share_url=share_url,
    )

    # Log scan to DB
    loop.run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="report-card",
        scan_timestamp=result.scan_timestamp,
        ahs_score=result.agent_health_score,
        grade=result.grade, grade_label=result.grade_label,
        confidence=result.confidence, mode=result.mode,
        d1_score=result.d1_score, d2_score=result.d2_score,
        d3_score=result.d3_score, cdp_modifier=result.cdp_modifier,
        patterns=[{"name": p.get("name", ""), "severity": p.get("severity", ""),
                   "description": p.get("description", ""), "modifier": p.get("modifier")}
                  for p in result.patterns_detected] if result.patterns_detected else None,
        tx_count=result.tx_count, history_days=result.history_days,
        response_data={"agent_health_score": result.agent_health_score, "grade": result.grade,
                       "d1_score": result.d1_score, "d2_score": result.d2_score,
                       "image_url": image_url},
    ))

    return ReportCardResponse(status="ok", report=report)


def _report_card_percentile(score: int, percentiles: dict) -> int:
    """Estimate percentile rank from p10/p25/p50/p75/p90 percentile dict."""
    if not percentiles:
        return 50
    checkpoints = [
        (percentiles.get("p10", 0), 10),
        (percentiles.get("p25", 0), 25),
        (percentiles.get("p50", 0), 50),
        (percentiles.get("p75", 0), 75),
        (percentiles.get("p90", 0), 90),
    ]
    if score <= checkpoints[0][0]:
        return max(1, int(10 * score / max(checkpoints[0][0], 1)))
    if score >= checkpoints[-1][0]:
        return min(99, 90 + int(10 * (score - checkpoints[-1][0]) / max(100 - checkpoints[-1][0], 1)))
    for i in range(len(checkpoints) - 1):
        s1, p1 = checkpoints[i]
        s2, p2 = checkpoints[i + 1]
        if s1 <= score <= s2:
            if s2 == s1:
                return p1
            return int(p1 + (p2 - p1) * (score - s1) / (s2 - s1))
    return 50


@app.get("/optimize/{address}", response_model=OptimizeResponse, tags=["Optimization"])
@limiter.limit("60/minute")
async def get_optimization_report(address: WalletAddress, request: Request):
    """
    Analyze a Base wallet and return a gas optimization report.

    Requires x402 payment ($5.00 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

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

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "optimize", address, request=request)

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

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="optimize",
        scan_timestamp=report.analyzed_at,
        response_data={"total_savings_usd": optimization.total_potential_savings_usd},
    ))
    return OptimizeResponse(status="ok", report=report)


# -- Protection Agent Endpoints -----------------------------------------------

@app.get("/agent/protect/preview/{address}", response_model=ProtectionPreviewResponse, tags=["Protection"])
async def protection_preview(address: WalletAddress):
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


@app.get("/agent/protect/{address}", response_model=ProtectionResponse, tags=["Protection"])
@limiter.limit("10/minute")
async def get_protection_report(address: WalletAddress, request: Request):
    """
    Autonomous protection agent — full wallet analysis and action plan.

    Requires x402 payment ($25.00 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

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

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "protect", address, request=request)

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

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="protect",
        scan_timestamp=now_str,
        health_score=protection.health_score,
        response_data={"risk_level": protection.risk_level, "health_score": protection.health_score,
                       "total_issues_found": protection.summary.total_issues_found},
    ))
    return ProtectionResponse(status="ok", report=report)


# -- RetryBot Endpoints ------------------------------------------------------

@app.get("/retry/preview/{address}", response_model=RetryPreviewResponse, tags=["Optimization"])
async def retry_preview(address: WalletAddress):
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


@app.get("/retry/{address}", response_model=RetryResponse, tags=["Optimization"])
@limiter.limit("60/minute")
async def get_retry_transactions(address: WalletAddress, request: Request):
    """
    Analyze failed transactions and return optimized retry transactions.

    Requires x402 payment ($10.00 USDC on Base) OR valid X-API-Key / X-Internal-Key header.

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

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "retry", address, request=request)

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

    asyncio.get_running_loop().run_in_executor(None, lambda: scan_db.log_scan(
        address=address, endpoint="retry",
        scan_timestamp=report.analyzed_at,
        response_data={"retryable_count": analysis.retryable_count,
                       "total_estimated_retry_cost_usd": analysis.total_estimated_retry_cost_usd},
    ))
    return RetryResponse(status="ok", report=report)


# -- Alert Endpoints ---------------------------------------------------------

@app.get("/alerts/subscribe/{address}", tags=["Alerts"])
@limiter.limit("60/minute")
async def subscribe_alerts(address: WalletAddress, request: Request):
    """
    Subscribe a wallet to automated health monitoring (30 days).

    Requires x402 payment ($2.00 USDC on Base) OR valid X-API-Key header.

    After payment, configure your webhook via POST /alerts/configure.
    Health checks run every 6 hours and send alerts when thresholds are breached.
    """
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()

    # Fiat API key path
    fiat_key = validate_fiat_request(request)
    if fiat_key:
        consume_api_key(fiat_key, "alerts/subscribe", address, request=request)
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


@app.post("/alerts/configure", tags=["Alerts"])
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

    if not _is_safe_webhook_url(req.webhook_url):
        raise HTTPException(status_code=400, detail="Invalid webhook URL. Must be a public https/http URL (no private IPs or internal hosts).")

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


@app.get("/alerts/status/{address}", tags=["Alerts"])
async def alert_status(address: WalletAddress):
    """Check the status of an alert subscription."""
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()
    sub = subscriptions.get(address)
    if not sub:
        raise HTTPException(status_code=404, detail="No subscription found for this address.")

    return {"status": "ok", "subscription": _sub_to_status(sub)}


@app.delete("/alerts/unsubscribe/{address}", tags=["Alerts"])
async def unsubscribe_alerts(address: WalletAddress):
    """Remove an alert subscription."""
    if not ADDRESS_RE.match(address):
        raise HTTPException(status_code=400, detail=f"Invalid Ethereum address format: {address}")

    address = address.lower()
    if address not in subscriptions:
        raise HTTPException(status_code=404, detail="No subscription found for this address.")

    del subscriptions[address]
    return {"status": "ok", "message": f"Subscription removed for {address}."}


# -- Stripe Webhook & API Key Endpoints --------------------------------------

@app.post("/stripe/webhook", tags=["Billing"])
async def stripe_webhook(request: Request):
    """
    Unified Stripe webhook handler for all payment events.

    Handles:
    - checkout.session.completed — issues an API key (core) or creates
      a Shield subscription (when metadata.product == "shield")
    - customer.subscription.updated — updates Shield subscription status
    - customer.subscription.deleted — cancels Shield subscription
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Stripe webhooks not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    obj = event["data"]["object"]

    # ── checkout.session.completed ──
    if event_type == "checkout.session.completed":
        metadata = obj.get("metadata", {})

        # Shield subscription checkout
        if metadata.get("product") == "shield":
            tier = metadata.get("tier", "starter")
            api_key_hash = metadata.get("api_key_hash", "")
            stripe_customer_id = obj.get("customer")
            stripe_subscription_id = obj.get("subscription")

            if not api_key_hash:
                logger.error("Shield webhook: no api_key_hash in metadata for session %s", obj.get("id"))
                return {"status": "error", "detail": "Missing api_key_hash"}

            scan_db.create_shield_subscription(
                api_key_hash=api_key_hash,
                tier=tier,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
            )
            logger.info("SHIELD SUBSCRIPTION CREATED: tier=%s key=%s… stripe_sub=%s",
                         tier, api_key_hash[:12], stripe_subscription_id)
            return {"status": "ok", "event": "subscription_created"}

        # Core API key checkout (default)
        email = obj.get("customer_email") or obj.get("customer_details", {}).get("email", "")
        stripe_customer_id = obj.get("customer")

        tier = metadata.get("tier", "starter")
        key_type = metadata.get("type", "payg")
        calls = int(metadata.get("calls", "100"))

        if not email:
            logger.error("Stripe webhook: no email found in session %s", obj.get("id"))
            return {"status": "error", "detail": "No customer email in session"}

        raw_key = scan_db.create_api_key(
            customer_email=email,
            stripe_customer_id=stripe_customer_id,
            key_type=key_type,
            tier=tier,
            calls_total=calls,
        )

        # Log prominently — email delivery not yet implemented
        logger.info(
            "API KEY ISSUED: %s for %s (tier=%s, type=%s, calls=%d)",
            raw_key, email, tier, key_type, calls,
        )
        return {"status": "ok", "message": f"API key issued for {email}"}

    # ── Shield subscription updated (plan change, renewal) ──
    if event_type == "customer.subscription.updated":
        stripe_sub_id = obj.get("id")
        status = obj.get("status", "active")
        status_map = {
            "active": "active",
            "past_due": "past_due",
            "unpaid": "past_due",
            "trialing": "active",
            "paused": "paused",
        }
        our_status = status_map.get(status, status)
        scan_db.update_shield_subscription_status(stripe_sub_id, our_status)
        return {"status": "ok", "event": "subscription_updated"}

    # ── Shield subscription cancelled/deleted ──
    if event_type == "customer.subscription.deleted":
        stripe_sub_id = obj.get("id")
        scan_db.update_shield_subscription_status(stripe_sub_id, "cancelled")
        return {"status": "ok", "event": "subscription_cancelled"}

    return {"status": "ignored", "event_type": event_type}


class TestWebhookRequest(BaseModel):
    email: str
    tier: str = "starter"
    key_type: str = "payg"
    calls: int = 100


@app.post("/stripe/webhook/test", tags=["Billing"])
async def stripe_webhook_test(req: TestWebhookRequest, request: Request):
    """
    Test endpoint: simulate a Stripe checkout.session.completed event.

    Protected by X-Internal-Key header. For development/testing only.
    Generates a real API key for the given email without requiring Stripe.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_key = scan_db.create_api_key(
        customer_email=req.email,
        stripe_customer_id=None,
        key_type=req.key_type,
        tier=req.tier,
        calls_total=req.calls,
    )

    logger.info(
        "TEST API KEY ISSUED: %s for %s (tier=%s, type=%s, calls=%d)",
        raw_key, req.email, req.tier, req.key_type, req.calls,
    )

    return {
        "status": "ok",
        "api_key": raw_key,
        "email": req.email,
        "tier": req.tier,
        "type": req.key_type,
        "calls": req.calls,
    }


# -- Shield Subscription Endpoints -------------------------------------------

# Shield tier -> (Stripe price lookup, agent_slots, call_quota, price_cents)
_SHIELD_TIERS = {
    "starter":    {"agent_slots": 5,   "call_quota": 10_000,      "price_cents": 2900},
    "pro":        {"agent_slots": 50,  "call_quota": 100_000,     "price_cents": 9900},
    "enterprise": {"agent_slots": 999, "call_quota": 999_999_999, "price_cents": 29900},
}


class ShieldSubscribeRequest(BaseModel):
    tier: str = Field(description="Shield tier: starter, pro, or enterprise")
    success_url: str = Field(default="https://agenthealthmonitor.xyz/shield?subscribed=true",
                             description="URL to redirect after successful checkout")
    cancel_url: str = Field(default="https://agenthealthmonitor.xyz/shield",
                            description="URL to redirect if checkout is cancelled")


@app.post("/shield/subscribe", tags=["Billing"])
async def shield_subscribe(body: ShieldSubscribeRequest, request: Request):
    """Create a Stripe Checkout session for a Shield subscription tier.

    Requires a valid X-API-Key header. The subscription will be linked to
    the API key used for authentication.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    tier_info = _SHIELD_TIERS.get(body.tier)
    if not tier_info:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier '{body.tier}'. Valid: {', '.join(_SHIELD_TIERS)}",
        )

    # Require an API key to link the subscription
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    record = scan_db.validate_api_key(api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    key_hash = record["key_hash"]

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": tier_info["price_cents"],
                    "recurring": {"interval": "month"},
                    "product_data": {
                        "name": f"AHM Shield — {body.tier.title()}",
                        "description": (
                            f"Up to {tier_info['agent_slots']} agents, "
                            f"{tier_info['call_quota']:,} route checks/mo"
                        ),
                    },
                },
                "quantity": 1,
            }],
            metadata={
                "product": "shield",
                "tier": body.tier,
                "api_key_hash": key_hash,
            },
            customer_email=record["customer_email"],
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )
    except stripe.StripeError as e:
        logger.error("Stripe session creation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to create checkout session")

    return {
        "status": "ok",
        "checkout_url": session.url,
        "session_id": session.id,
    }



@app.get("/api/key/status", tags=["Billing"])
async def api_key_status(request: Request):
    """
    Check the status of an API key.

    Requires X-API-Key header. Returns tier, remaining calls, and key metadata.
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    record = scan_db.validate_api_key(api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    return {
        "tier": record["tier"],
        "calls_remaining": record["calls_remaining"],
        "calls_total": record["calls_total"],
        "type": record["type"],
        "created_at": record["created_at"],
        "is_active": bool(record["is_active"]),
    }


# -- Partner Usage Endpoint ---------------------------------------------------

@app.get("/partners/{partner_id}/usage", tags=["Billing"])
async def partner_usage(partner_id: str, request: Request):
    """Usage report for a Shield reseller partner.

    Protected by X-Internal-Key header. Returns call count, wholesale cost,
    and billing period for the given partner_id.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    days = min(int(request.query_params.get("days", "30")), 365)

    loop = asyncio.get_running_loop()
    usage = await loop.run_in_executor(
        None, lambda: scan_db.get_partner_usage(partner_id, days=days)
    )
    return usage


# -- Security Activity Endpoint -----------------------------------------------

@app.get("/security/activity", tags=["Admin"])
async def security_activity(request: Request):
    """Recent security events (rate limits, suspicious patterns).

    Protected by X-Internal-Key header. Not accessible via x402 payment.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    event_type = request.query_params.get("type")
    limit = min(int(request.query_params.get("limit", "100")), 500)

    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(
        None, lambda: scan_db.get_security_events(limit=limit, event_type=event_type)
    )
    return {"events": events, "total": len(events)}


# -- Trust Registry Endpoint -------------------------------------------------

@app.get("/trust-registry", tags=["Admin"])
async def trust_registry(request: Request):
    """
    Aggregated scan statistics for the AHM trust layer.

    Protected by X-Internal-Key header. Not accessible via x402 payment.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, scan_db.get_trust_registry_stats)
    return stats


# -- Agent Profile (internal, for ahm-verify) --------------------------------

@app.get("/internal/agent-profile/{address}", tags=["Admin"])
async def internal_agent_profile(address: str, request: Request):
    """Return AHS profile context for a single agent address.

    Protected by X-Internal-Key header. Used by ahm-verify to fetch
    agent scoring context for the adjudication panel.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden")

    loop = asyncio.get_running_loop()
    profile = await loop.run_in_executor(
        None, lambda: scan_db.get_agent_profile(address)
    )

    if profile is None:
        return {"address": address.lower(), "status": "unrated"}

    return profile


# -- ACP Scan Trigger & Status -----------------------------------------------

@app.post("/acp-scan/trigger", tags=["Admin"])
async def trigger_acp_scan(request: Request):
    """Manually trigger the ACP nightly scan.

    Protected by X-Internal-Key header. Returns immediately;
    scan runs in the background.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _acp_scan_lock.locked():
        raise HTTPException(status_code=409, detail="ACP scan already running")

    asyncio.create_task(run_acp_scan())
    return {"status": "ok", "message": "ACP scan triggered"}


@app.get("/acp-scan/status", tags=["Admin"])
async def acp_scan_status(request: Request):
    """Check if an ACP scan is currently running.

    Protected by X-Internal-Key header.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    scheduler = request.app.state.scheduler
    job = scheduler.get_job("acp_nightly_scan")
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    return {
        "running": _acp_scan_lock.locked(),
        "next_scheduled_run": next_run,
    }


# -- Olas Scan Trigger & Status ----------------------------------------------

@app.post("/olas-scan/trigger", tags=["Admin"])
async def trigger_olas_scan(request: Request):
    """Manually trigger the Olas nightly scan.

    Protected by X-Internal-Key header. Returns immediately;
    scan runs in the background.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _olas_scan_lock.locked():
        raise HTTPException(status_code=409, detail="Olas scan already running")

    asyncio.create_task(run_olas_scan())
    return {"status": "ok", "message": "Olas scan triggered"}


@app.get("/olas-scan/status", tags=["Admin"])
async def olas_scan_status(request: Request):
    """Check if an Olas scan is currently running.

    Protected by X-Internal-Key header.

    Also queries ``totalSupply()`` on the Olas ServiceRegistryL2 contract
    on Base so the dashboard can render a live saturation figure
    (e.g. "887 / 900") instead of looking stalled when the nightly
    scanner has already caught up to the registry. Any RPC failure is
    caught and surfaced as ``registry_total_supply: null`` rather than
    failing the whole status endpoint.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Match the defensive pattern used by arc_scan_status / celo_scan_status
    # so the endpoint still serves a useful response if the scheduler
    # hasn't attached yet (e.g. during tests, or briefly on cold start).
    next_run = None
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler:
        job = scheduler.get_job("olas_nightly_scan")
        next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    # Live on-chain supply — offloaded to an executor so the Web3 HTTP
    # call doesn't block the event loop.
    registry_total_supply: int | None = None
    try:
        from olas_scan import fetch_registry_total_supply
        loop = asyncio.get_event_loop()
        registry_total_supply = await loop.run_in_executor(
            None, fetch_registry_total_supply,
        )
    except Exception as e:
        logger.warning("olas_scan_status: totalSupply() unavailable: %s", e)

    return {
        "running": _olas_scan_lock.locked(),
        "next_scheduled_run": next_run,
        "registry_total_supply": registry_total_supply,
    }


# -- Arc Scan Trigger & Status -----------------------------------------------

@app.post("/arc-scan/trigger", tags=["Admin"])
async def trigger_arc_scan(request: Request):
    """Manually trigger the Arc nightly scan.

    Protected by X-Internal-Key header. Returns immediately;
    scan runs in the background.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _arc_scan_lock.locked():
        raise HTTPException(status_code=409, detail="Arc scan already running")

    asyncio.create_task(run_arc_scan())
    return {"status": "ok", "message": "Arc scan triggered"}


@app.get("/arc-scan/status", tags=["Admin"])
async def arc_scan_status(request: Request):
    """Check if an Arc scan is currently running.

    Protected by X-Internal-Key header.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    next_run = None
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler:
        job = scheduler.get_job("arc_nightly_scan")
        next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    return {
        "running": _arc_scan_lock.locked(),
        "next_scheduled_run": next_run,
    }


# -- Celo Scan Trigger & Status ----------------------------------------------

@app.post("/celo-scan/trigger", tags=["Admin"])
async def trigger_celo_scan(request: Request):
    """Manually trigger the Celo nightly scan.

    Protected by X-Internal-Key header. Returns immediately;
    scan runs in the background.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _celo_scan_lock.locked():
        raise HTTPException(status_code=409, detail="Celo scan already running")

    asyncio.create_task(run_celo_scan())
    return {"status": "ok", "message": "Celo scan triggered"}


@app.get("/celo-scan/status", tags=["Admin"])
async def celo_scan_status(request: Request):
    """Check if a Celo scan is currently running.

    Protected by X-Internal-Key header.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    next_run = None
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler:
        job = scheduler.get_job("celo_nightly_scan")
        next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    return {
        "running": _celo_scan_lock.locked(),
        "next_scheduled_run": next_run,
    }


# -- ERC-8004 Base Scan Trigger & Status ------------------------------------

@app.post("/erc8004-scan/trigger", tags=["Admin"])
async def trigger_erc8004_scan(request: Request):
    """Manually trigger the ERC-8004 Base-mainnet nightly scan.

    Protected by X-Internal-Key header. Returns immediately;
    scan runs in the background.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _erc8004_scan_lock.locked():
        raise HTTPException(status_code=409, detail="ERC-8004 scan already running")

    asyncio.create_task(run_erc8004_scan())
    return {"status": "ok", "message": "ERC-8004 scan triggered"}


@app.get("/erc8004-scan/status", tags=["Admin"])
async def erc8004_scan_status(request: Request):
    """Check if an ERC-8004 Base-mainnet scan is currently running.

    Protected by X-Internal-Key header.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    next_run = None
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler:
        job = scheduler.get_job("erc8004_nightly_scan")
        next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    return {
        "running": _erc8004_scan_lock.locked(),
        "next_scheduled_run": next_run,
    }


# -- ERC-8183 Worker Status --------------------------------------------------

@app.get("/erc8183/status", tags=["Admin"])
async def erc8183_status(request: Request):
    """Check ERC-8183 evaluator worker status on Arc testnet.

    Protected by X-Internal-Key header.
    """
    internal_key = request.headers.get("X-Internal-Key", "")
    if not INTERNAL_API_KEY or not hmac.compare_digest(internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return erc8183_worker.get_status()


# -- Static file serving (must be after all route definitions) ----------------
# Use absolute path (STATIC_DIR) so serving works regardless of working directory.
os.makedirs(STATIC_DIR / "report-cards", exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- ACP Nightly Scan (replaces Railway cron service) -----------------------


def _cleanup_acp_checkpoint():
    """Remove ACP checkpoint file after a successful scan completion."""
    try:
        from acp_proactive_scan import CHECKPOINT_PATH
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            logger.info("ACP checkpoint cleaned up (scan complete)")
    except Exception as e:
        logger.warning("Failed to clean up ACP checkpoint: %s", e)


_acp_scan_lock = asyncio.Lock()


async def run_acp_scan():
    """Run the nightly ACP proactive scan in a background thread.

    Mirrors cron_acp_scan.py logic but runs inside the web process.
    All sync I/O is offloaded to the default executor via run_in_executor.
    """
    if _acp_scan_lock.locked():
        logger.warning("ACP scan skipped: previous scan still running")
        return

    async with _acp_scan_lock:
        logger.info("ACP nightly scan — START")
        start = time.time()

        max_agents = int(os.getenv("ACP_MAX_AGENTS", "500"))
        max_scans = int(os.getenv("ACP_MAX_SCANS", "100"))
        sort_order = os.getenv("ACP_SORT", "successfulJobCount:desc")
        max_runtime = int(os.getenv("ACP_MAX_RUNTIME", "3600"))

        loop = asyncio.get_running_loop()

        try:
            from acp_proactive_scan import (
                discover_agents, deduplicate_wallets,
                scan_wallets, generate_report,
            )

            # Phase 1: Discovery
            agents, api_stats = await loop.run_in_executor(
                None,
                partial(discover_agents,
                        max_agents=max_agents,
                        sort=sort_order),
            )

            if not agents:
                logger.info("ACP scan: no agents discovered, exiting cleanly")
                return

            # Phase 2: Deduplication
            dedup_stats = await loop.run_in_executor(
                None,
                partial(deduplicate_wallets, agents),
            )

            # Runtime guard
            elapsed = time.time() - start
            if elapsed > max_runtime * 0.8:
                logger.warning(
                    "ACP scan runtime guard: discovery took %.0fs, skipping scan phase",
                    elapsed,
                )
                await loop.run_in_executor(
                    None,
                    partial(generate_report, agents, api_stats, dedup_stats, {}),
                )
                return

            # Phase 3: Scanning (stale-first rotation)
            scan_results = await loop.run_in_executor(
                None,
                partial(scan_wallets, agents,
                        max_scans=max_scans),
            )

            # Phase 4: Report
            await loop.run_in_executor(
                None,
                partial(generate_report, agents, api_stats, dedup_stats, scan_results),
            )

            # Clean up checkpoint on successful completion (mirrors cron_acp_scan.py)
            await loop.run_in_executor(None, _cleanup_acp_checkpoint)

            elapsed = time.time() - start
            logger.info("ACP nightly scan — COMPLETE (%.0fs)", elapsed)

        except Exception:
            elapsed = time.time() - start
            logger.exception("ACP nightly scan — FAILED after %.0fs", elapsed)


# -- Olas Nightly Scan -------------------------------------------------------

_olas_scan_lock = asyncio.Lock()


async def run_olas_scan():
    """Run the nightly Olas protocol scan in a background thread.

    Discovers wallet addresses from the Olas ServiceRegistryL2 on Base,
    then runs AHS scoring on new wallets. All sync I/O is offloaded
    to the default executor via run_in_executor.
    """
    if _olas_scan_lock.locked():
        logger.warning("Olas scan skipped: previous scan still running")
        return

    async with _olas_scan_lock:
        logger.info("Olas nightly scan — START")
        start = time.time()

        max_scans = int(os.getenv("OLAS_MAX_SCANS", "200"))
        loop = asyncio.get_running_loop()

        try:
            from olas_scan import scan_olas_services

            wallets = await loop.run_in_executor(
                None,
                partial(scan_olas_services, max_scans=max_scans),
            )

            elapsed = time.time() - start
            logger.info(
                "Olas nightly scan — COMPLETE (%.0fs, %d wallets discovered)",
                elapsed, len(wallets),
            )

        except Exception:
            elapsed = time.time() - start
            logger.exception("Olas nightly scan — FAILED after %.0fs", elapsed)


# -- Arc Nightly Scan -------------------------------------------------------

_arc_scan_lock = asyncio.Lock()


async def run_arc_scan():
    """Run the nightly Arc protocol scan in a background thread.

    Discovers agent wallet addresses from the ERC-8004 IdentityRegistry
    on Arc testnet, then runs AHS scoring on new wallets. All sync I/O
    is offloaded to the default executor via run_in_executor.
    """
    if _arc_scan_lock.locked():
        logger.warning("Arc scan skipped: previous scan still running")
        return

    async with _arc_scan_lock:
        logger.info("Arc nightly scan — START")
        start = time.time()

        max_scans = int(os.getenv("ARC_MAX_SCANS", "200"))
        loop = asyncio.get_running_loop()

        try:
            from arc_scan import scan_arc_agents

            wallets = await loop.run_in_executor(
                None,
                partial(scan_arc_agents, max_scans=max_scans),
            )

            elapsed = time.time() - start
            logger.info(
                "Arc nightly scan — COMPLETE (%.0fs, %d wallets discovered)",
                elapsed, len(wallets),
            )

        except Exception:
            elapsed = time.time() - start
            logger.exception("Arc nightly scan — FAILED after %.0fs", elapsed)


# -- Celo Nightly Scan ------------------------------------------------------

_celo_scan_lock = asyncio.Lock()


async def run_celo_scan():
    """Run the nightly Celo ERC-8004 scan in a background thread.

    Discovers agent wallet addresses from the ERC-8004 IdentityRegistry
    on Celo mainnet, then runs AHS scoring on new wallets. All sync I/O
    is offloaded to the default executor via run_in_executor.
    """
    if _celo_scan_lock.locked():
        logger.warning("Celo scan skipped: previous scan still running")
        return

    async with _celo_scan_lock:
        logger.info("Celo nightly scan — START")
        start = time.time()

        max_scans = int(os.getenv("CELO_MAX_SCANS", "200"))
        loop = asyncio.get_running_loop()

        try:
            from celo_scan import scan_celo_agents

            wallets = await loop.run_in_executor(
                None,
                partial(scan_celo_agents, max_scans=max_scans),
            )

            elapsed = time.time() - start
            logger.info(
                "Celo nightly scan — COMPLETE (%.0fs, %d wallets discovered)",
                elapsed, len(wallets),
            )

        except Exception:
            elapsed = time.time() - start
            logger.exception("Celo nightly scan — FAILED after %.0fs", elapsed)


# -- ERC-8004 Base Nightly Scan ---------------------------------------------

_erc8004_scan_lock = asyncio.Lock()


async def run_erc8004_scan():
    """Run the nightly ERC-8004 Base-mainnet scan in a background thread.

    Enumerates agents on the canonical ERC-8004 IdentityRegistry at
    0x8004A169FB4a3325136EB29fA0ceB6D2e539a432 (Base mainnet), resolves
    their tokenURI registration docs, and runs AHS scoring on every
    distinct wallet surfaced (owner, agent_wallet, and URI-embedded
    addresses). All sync I/O is offloaded to the default executor via
    run_in_executor.
    """
    if _erc8004_scan_lock.locked():
        logger.warning("ERC-8004 scan skipped: previous scan still running")
        return

    async with _erc8004_scan_lock:
        logger.info("ERC-8004 nightly scan — START")
        start = time.time()

        max_scans = int(os.getenv("ERC8004_MAX_SCANS", "100"))
        loop = asyncio.get_running_loop()

        try:
            from erc8004_scan import scan_erc8004_agents

            wallets = await loop.run_in_executor(
                None,
                partial(scan_erc8004_agents, max_scans=max_scans),
            )

            elapsed = time.time() - start
            logger.info(
                "ERC-8004 nightly scan — COMPLETE (%.0fs, %d wallets scored)",
                elapsed, len(wallets),
            )

        except Exception:
            elapsed = time.time() - start
            logger.exception("ERC-8004 nightly scan — FAILED after %.0fs", elapsed)


# -- Rescan Background Loop --------------------------------------------------

async def rescan_loop():
    """Background loop: re-scan known wallets on their configured interval."""
    await asyncio.sleep(300)  # 5-minute startup delay

    while True:
        try:
            loop = asyncio.get_running_loop()
            wallets_due = await loop.run_in_executor(None, scan_db.get_wallets_due_for_rescan)
            if wallets_due:
                logger.info("Rescan loop: %d wallets due", len(wallets_due))

            for wallet in wallets_due:
                try:
                    eth_price, transactions, tokens = await asyncio.gather(
                        loop.run_in_executor(None, partial(get_eth_price, BASESCAN_API_KEY)),
                        loop.run_in_executor(None, partial(fetch_transactions, wallet["address"], BASESCAN_API_KEY)),
                        loop.run_in_executor(None, partial(fetch_tokens_v2, wallet["address"])),
                    )

                    result = await loop.run_in_executor(None, partial(
                        calculate_ahs,
                        address=wallet["address"],
                        tokens=tokens,
                        transactions=transactions,
                        eth_price=eth_price,
                    ))

                    patterns = None
                    if result.patterns_detected:
                        patterns = [{"name": p.get("name", ""), "severity": p.get("severity", ""),
                                     "description": p.get("description", ""), "modifier": p.get("modifier")}
                                    for p in result.patterns_detected]

                    await loop.run_in_executor(None, lambda: scan_db.log_scan(
                        address=wallet["address"], endpoint="ahs",
                        scan_timestamp=result.scan_timestamp,
                        source="rescan", label=wallet.get("label"),
                        ahs_score=result.agent_health_score,
                        grade=result.grade, grade_label=result.grade_label,
                        confidence=result.confidence, mode=result.mode,
                        d1_score=result.d1_score, d2_score=result.d2_score,
                        d3_score=result.d3_score, cdp_modifier=result.cdp_modifier,
                        patterns=patterns,
                        tx_count=result.tx_count, history_days=result.history_days,
                        response_data={"agent_health_score": result.agent_health_score,
                                       "grade": result.grade},
                    ))
                    logger.info("Rescan %s: AHS=%d/%s", wallet["address"][:10], result.agent_health_score, result.grade)

                except Exception as e:
                    logger.warning("Rescan failed for %s: %s", wallet["address"][:10], e)

                await asyncio.sleep(3)  # Rate limit between scans

        except Exception as e:
            logger.error("Rescan loop error: %s", e)

        await asyncio.sleep(3600)  # Check every hour


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
