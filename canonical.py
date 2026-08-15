"""Canonical facts loader.

Single source of truth for every value that appears on a public AHM surface
and changes only on a deliberate release: prices, endpoint count, batch limits,
grade thresholds, routing bands, dimension weights, the Verify pipeline
description and verdict enum, the registry list, the Twitter handle.

Anything that changes on its own - scan counts, averages, rates, distributions -
is NOT in here. Those live on /dashboard, rendered from the database. See
config/canonical.json -> dynamic_figures.

Usage:
    from canonical import CANONICAL, endpoint_count, grade_for_score

    ENDPOINT_COUNT = endpoint_count()
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CANONICAL_PATH = Path(__file__).parent / "config" / "canonical.json"


@lru_cache(maxsize=1)
def load() -> dict:
    """Load and cache the canonical config."""
    with CANONICAL_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


CANONICAL = load()


# -- Endpoints ---------------------------------------------------------------

def endpoint_count() -> int:
    """Number of billable data endpoints a caller can hit today."""
    return CANONICAL["endpoints"]["count"]


def endpoints() -> list[dict]:
    return CANONICAL["endpoints"]["list"]


def price_for(slug: str) -> str:
    """Canonical list price for an endpoint slug, e.g. 'ahs' -> '$1.00'."""
    for ep in endpoints():
        if ep["slug"] == slug:
            return ep["price"]
    raise KeyError(f"Unknown endpoint slug: {slug}")


def price_env_defaults() -> dict[str, str]:
    """Map of env var name -> canonical default price, for api.py wiring."""
    return {ep["price_env"]: ep["price"] for ep in endpoints()}


# -- Grades ------------------------------------------------------------------

def grade_bands() -> list[dict]:
    return CANONICAL["grades"]["bands"]


def grade_for_score(score: int) -> tuple[str, str]:
    """Return (letter, label) for a 0-100 AHS score.

    Mirrors monitor._ahs_grade. Kept in sync by scripts/check_consistency.py,
    which fails the build if the two ever diverge.
    """
    for band in grade_bands():
        if score >= band["min"]:
            return band["letter"], band["label"]
    last = grade_bands()[-1]
    return last["letter"], last["label"]


def grade_letters() -> list[str]:
    return [b["letter"] for b in grade_bands()]


# -- Routing -----------------------------------------------------------------

def routing_actions() -> list[str]:
    return CANONICAL["routing"]["actions"]


def routing_for_grade(letter: str) -> str:
    """Default routing action for a grade letter. Integrator policies override."""
    for band in CANONICAL["routing"]["default_bands"]:
        if letter in band["grades"]:
            return band["action"]
    return "reject"


# -- Dimensions --------------------------------------------------------------

def weights(mode: str = "2D") -> dict[str, float]:
    """Dimension weights for '2D' (default) or '3D' (agent_url supplied)."""
    return CANONICAL["dimensions"]["modes"][mode]["weights"]


def dimensions_statement() -> str:
    return CANONICAL["dimensions"]["statement"]


# -- Batch -------------------------------------------------------------------

def batch_limits() -> tuple[int, int]:
    """(x402 max wallets, API-key max wallets). Never publish one without the other."""
    b = CANONICAL["batch"]
    return b["x402_max_wallets"], b["api_key_max_wallets"]


def batch_statement() -> str:
    return CANONICAL["batch"]["statement"]


# -- Verify ------------------------------------------------------------------

def verify_verdicts() -> list[str]:
    return CANONICAL["verify"]["verdict_enum"]


def verify_description() -> str:
    return CANONICAL["verify"]["description"]


# -- Registries / social -----------------------------------------------------

def registries() -> list[dict]:
    return CANONICAL["registries"]["scanned"]


def registry_display_order() -> list[str]:
    return CANONICAL["registries"]["display_order"]


def twitter_handle() -> str:
    return CANONICAL["social"]["twitter_handle"]


def contact_email() -> str:
    return CANONICAL["social"]["contact_email"]


def scanned_unit() -> str:
    """The noun for what AHM counts. See canonical.json -> nouns."""
    return CANONICAL["nouns"]["scanned_unit"]
