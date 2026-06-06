"""Crawler / indexer classifier for AHM Crawler Observatory.

Classifies incoming request user-agents and paths to identify
which bots, indexers, and AI crawlers are probing the service.

Configuration is data-driven (PATTERNS list + DISCOVERY_PATHS set),
not hardcoded in logic — edit the tables below to add new entries.
"""

import re
from typing import Optional

# -- UA class enum values ---------------------------------------------------

UA_KNOWN_INDEXER = "known_indexer"
UA_AI_CRAWLER = "ai_crawler"
UA_SCANNER = "scanner"
UA_GENERIC_BOT = "generic_bot"
UA_BROWSER = "browser"
UA_UNKNOWN = "unknown"

# -- UA pattern table -------------------------------------------------------
# Each entry: (compiled regex, ua_class, indexer_name | None)
# Order matters — first match wins.

_PATTERN_DEFS: list[tuple[str, str, Optional[str]]] = [
    # Known agent-economy indexers
    (r"ari-indexer", UA_KNOWN_INDEXER, "ari-indexer"),
    (r"A2A-Registry-TaskProbe", UA_KNOWN_INDEXER, "A2A-Registry-TaskProbe"),
    (r"flows-crawler", UA_KNOWN_INDEXER, "flows-crawler"),
    (r"usemur\.dev", UA_KNOWN_INDEXER, "flows-crawler"),
    (r"402index", UA_KNOWN_INDEXER, "402index"),
    (r"x402scan", UA_KNOWN_INDEXER, "x402scan"),

    # AI crawlers
    (r"GPTBot", UA_AI_CRAWLER, "GPTBot"),
    (r"ChatGPT-User", UA_AI_CRAWLER, "ChatGPT-User"),
    (r"ClaudeBot", UA_AI_CRAWLER, "ClaudeBot"),
    (r"PerplexityBot", UA_AI_CRAWLER, "PerplexityBot"),
    (r"Google-Extended", UA_AI_CRAWLER, "Google-Extended"),
    (r"CCBot", UA_AI_CRAWLER, "CCBot"),
    (r"Bytespider", UA_AI_CRAWLER, "Bytespider"),
    (r"Amazonbot", UA_AI_CRAWLER, "Amazonbot"),
    (r"anthropic-ai", UA_AI_CRAWLER, "anthropic-ai"),
    (r"Applebot", UA_AI_CRAWLER, "Applebot"),
    (r"cohere-ai", UA_AI_CRAWLER, "cohere-ai"),

    # Scanners
    (r"l9scan", UA_SCANNER, "l9scan"),
    (r"l9explore", UA_SCANNER, "l9explore"),
    (r"Nuclei", UA_SCANNER, "Nuclei"),
    (r"zgrab", UA_SCANNER, "zgrab"),
    (r"masscan", UA_SCANNER, "masscan"),

    # Generic bots (broad patterns — keep after specific ones)
    (r"(?i)bot[/\s;)]", UA_GENERIC_BOT, None),
    (r"(?i)spider", UA_GENERIC_BOT, None),
    (r"(?i)crawl", UA_GENERIC_BOT, None),

    # Browsers (catch common browser engine strings)
    (r"Mozilla/.*(?:Chrome|Firefox|Safari|Edge)", UA_BROWSER, None),
]

PATTERNS: list[tuple[re.Pattern, str, Optional[str]]] = [
    (re.compile(regex), cls, name) for regex, cls, name in _PATTERN_DEFS
]

# -- Discovery paths --------------------------------------------------------
# Union of paths we serve AND paths agents probe for (even if we 404 them).
# is_discovery_path matches any of these exactly, or any /.well-known/* path.

DISCOVERY_PATHS_EXACT: set[str] = {
    # Paths we actually serve
    "/.well-known/agent.json",
    "/.well-known/agent-registration.json",
    "/.well-known/x402",
    "/.well-known/x402-services.json",
    "/.well-known/402index-verify.txt",
    "/openapi.json",
    # Paths agents probe for (may 404 — high signal)
    "/a2a",
    "/a2a/.well-known/agent.json",
    "/llms.txt",
    "/ai-plugin.json",
    "/.well-known/ai-plugin.json",
}


def classify(user_agent: str) -> tuple[str, Optional[str]]:
    """Classify a user-agent string.

    Returns (ua_class, indexer_name).  indexer_name is None when
    the class is generic (browser, generic_bot, unknown).
    """
    if not user_agent or not user_agent.strip():
        return UA_GENERIC_BOT, None

    ua = user_agent.strip()

    # Bare "node" or "node-fetch" without a real product token
    if re.match(r"^node(?:-fetch)?(?:/[\d.]+)?$", ua, re.IGNORECASE):
        return UA_GENERIC_BOT, None

    for pattern, cls, name in PATTERNS:
        if pattern.search(ua):
            return cls, name

    return UA_UNKNOWN, None


def is_discovery_path(path: str) -> bool:
    """Return True if the path is a known agent-discovery endpoint.

    Matches exact paths in DISCOVERY_PATHS_EXACT plus any
    /.well-known/* path (catching novel discovery probes).
    """
    if not path:
        return False
    normalised = path.rstrip("/")
    if normalised in DISCOVERY_PATHS_EXACT:
        return True
    if normalised.startswith("/.well-known"):
        return True
    return False
