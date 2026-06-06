"""Unit tests for crawler_classifier — UA classification and discovery paths."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import crawler_classifier as cc


# -- UA classification fixtures ---------------------------------------------

UA_FIXTURES = [
    # (raw user-agent string, expected ua_class, expected indexer_name)

    # Known indexers
    ("ari-indexer/1.0", cc.UA_KNOWN_INDEXER, "ari-indexer"),
    ("A2A-Registry-TaskProbe/0.1 (https://a2a-registry.example)", cc.UA_KNOWN_INDEXER, "A2A-Registry-TaskProbe"),
    ("flows-crawler/2.0 (+https://flows.network)", cc.UA_KNOWN_INDEXER, "flows-crawler"),
    ("Mozilla/5.0 (compatible; usemur.dev crawler)", cc.UA_KNOWN_INDEXER, "flows-crawler"),
    ("402index-bot/1.0", cc.UA_KNOWN_INDEXER, "402index"),
    ("x402scan/0.3", cc.UA_KNOWN_INDEXER, "x402scan"),

    # AI crawlers
    ("Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GPTBot/1.2; +https://openai.com/gptbot)", cc.UA_AI_CRAWLER, "GPTBot"),
    ("ClaudeBot/1.0", cc.UA_AI_CRAWLER, "ClaudeBot"),
    ("PerplexityBot/1.0", cc.UA_AI_CRAWLER, "PerplexityBot"),
    ("Mozilla/5.0 (compatible; Google-Extended)", cc.UA_AI_CRAWLER, "Google-Extended"),
    ("CCBot/2.0 (https://commoncrawl.org/faq/)", cc.UA_AI_CRAWLER, "CCBot"),
    ("Mozilla/5.0 (compatible; Bytespider; spider-feedback@bytedance.com)", cc.UA_AI_CRAWLER, "Bytespider"),
    ("Amazonbot/0.1 (https://developer.amazon.com/support/amazonbot)", cc.UA_AI_CRAWLER, "Amazonbot"),

    # Scanners
    ("l9scan/0.1", cc.UA_SCANNER, "l9scan"),
    ("l9explore/1.0", cc.UA_SCANNER, "l9explore"),
    ("Nuclei - Open-source project (github.com/projectdiscovery/nuclei)", cc.UA_SCANNER, "Nuclei"),
    ("Mozilla/5.0 zgrab/0.x", cc.UA_SCANNER, "zgrab"),

    # Generic bots
    ("", cc.UA_GENERIC_BOT, None),
    ("   ", cc.UA_GENERIC_BOT, None),
    ("node", cc.UA_GENERIC_BOT, None),
    ("node/18.0.0", cc.UA_GENERIC_BOT, None),
    ("node-fetch/2.6.1", cc.UA_GENERIC_BOT, None),
    ("SomeBot/1.0 (http://example.com/bot)", cc.UA_GENERIC_BOT, None),

    # Browsers
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36", cc.UA_BROWSER, None),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15", cc.UA_BROWSER, None),
    ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0", cc.UA_BROWSER, None),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0", cc.UA_BROWSER, None),

    # Unknown
    ("python-requests/2.31.0", cc.UA_UNKNOWN, None),
    ("curl/8.4.0", cc.UA_UNKNOWN, None),
    ("httpx/0.27.0", cc.UA_UNKNOWN, None),
]


@pytest.mark.parametrize("ua,expected_class,expected_name", UA_FIXTURES)
def test_classify(ua, expected_class, expected_name):
    ua_class, indexer_name = cc.classify(ua)
    assert ua_class == expected_class, f"UA {ua!r}: expected class {expected_class}, got {ua_class}"
    assert indexer_name == expected_name, f"UA {ua!r}: expected name {expected_name}, got {indexer_name}"


# -- Discovery path fixtures ------------------------------------------------

DISCOVERY_PATH_FIXTURES = [
    # (path, expected)
    ("/.well-known/agent.json", True),
    ("/.well-known/agent-registration.json", True),
    ("/.well-known/x402", True),
    ("/.well-known/x402-services.json", True),
    ("/.well-known/402index-verify.txt", True),
    ("/openapi.json", True),
    ("/a2a", True),
    ("/a2a/.well-known/agent.json", True),
    ("/llms.txt", True),
    ("/ai-plugin.json", True),
    ("/.well-known/ai-plugin.json", True),
    # Any /.well-known/* — even novel probes
    ("/.well-known/something-new", True),
    ("/.well-known/assetlinks.json", True),
    # Trailing slash normalisation
    ("/.well-known/agent.json/", True),
    ("/a2a/", True),
    # Non-discovery paths
    ("/", False),
    ("/up", False),
    ("/health", False),
    ("/agents/0xabc123", False),
    ("/dashboard", False),
    ("/cdn-cgi/rum", False),
    ("", False),
]


@pytest.mark.parametrize("path,expected", DISCOVERY_PATH_FIXTURES)
def test_is_discovery_path(path, expected):
    result = cc.is_discovery_path(path)
    assert result == expected, f"Path {path!r}: expected {expected}, got {result}"
