#!/usr/bin/env python3
"""Fail the build when a public surface drifts from canonical truth.

Three classes of check:

  1. CHANGING FIGURES OUTSIDE /dashboard — scan counts, averages, rates and
     derived percentages must appear on the dashboard only. Everywhere else
     links to it. A number in static copy is stale the moment the nightly scan
     runs; the only durable fix is for it not to be there.

  2. FORBIDDEN STRINGS — specific claims that were wrong and must not return,
     e.g. "8,000+", "six-model", "GPT-4o", "AHMprotocol", "13 endpoints".

  3. HARD-CODED STABLE VALUES — prices, thresholds, weights, counts and handles
     must come from config/canonical.json, not be retyped into a template.
     Also verifies canonical.json still agrees with the code it describes, so
     the config itself cannot silently drift from the scorer.

Usage:
    python scripts/check_consistency.py                 # local surfaces
    python scripts/check_consistency.py --live          # also crawl deployed URLs
    python scripts/check_consistency.py --surfaces ../ahm-docs/public
    python scripts/check_consistency.py --list-known    # show accepted exceptions

Exit code 0 = clean, 1 = drift found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import canonical  # noqa: E402

CANON = canonical.CANONICAL

# ── Surfaces ────────────────────────────────────────────────────────────────
# The dashboard is the one place changing figures are allowed. Everything else
# is static copy and is held to the no-figures rule.
DASHBOARD_FILES = {"dashboard.html"}

DEFAULT_SURFACE_GLOBS = [
    "static/*.html",
    "README.md",
    "VISION.md",
    "PARTNERSHIPS.md",
]

# Sibling repos, checked when present. Keeps the whole public estate honest
# from one command rather than one checker per repo.
SIBLING_SURFACES = [
    ("../ahm-docs/public", ["*.html", "*.md"]),
    ("../ahm-intelligence/public", ["*.html"]),
]

LIVE_URLS = [
    "https://agenthealthmonitor.xyz/",
    "https://agenthealthmonitor.xyz/roadmap",
    "https://agenthealthmonitor.xyz/app",
    "https://agenthealthmonitor.xyz/shield",
    "https://agenthealthmonitor.xyz/verify",
    "https://agenthealthmonitor.xyz/endpoints/ahs",
    "https://docs.agenthealthmonitor.xyz/",
    "https://intelligence.agenthealthmonitor.xyz/",
]

# ── Rules ───────────────────────────────────────────────────────────────────

# Targets ecosystem-scale population figures, not every number that happens to
# sit next to the word "agents". A comma-grouped number (8,000 / 42,205) or a
# "+" suffix (2,860+) is the shape a scan count takes. "up to 10 wallets" is a
# canonical batch limit and must not trip this.
CHANGING_FIGURE_PATTERNS = [
    (r"\d{1,3}(?:,\d{3})+\+?\s*(?:agents?|wallets?|scanned)\b", "scan/agent count"),
    (r"\b\d+\+\s*(?:agents?|wallets?|scanned)\b", "scan/agent count"),
    (r"average\s+AHS\s*[:\-]?\s*\d", "average AHS with a value"),
    (r"\bAHS\s+(?:of|is)\s+\d+(?:\.\d+)?", "average AHS with a value"),
    (r"zombie\s+(?:agent\s+)?rate\s*[:\-]?\s*\d", "zombie rate with a value"),
    (r"\d+(?:\.\d+)?%\s*(?:score|below|of\s+agents)", "derived percentage claim"),
    (r"\b\d+(?:\.\d+)?%\s+(?:reach|are|have)\s+(?:a\s+)?grade", "grade-share claim"),
]

FORBIDDEN_STRINGS = [
    "8,000+", "8,800+", "2,860+", "4,500+", "4,552", "42,000+",
    "13 endpoints", "13 diagnostic", "11 pay-per-call",
    "GPT-4o", "six-model", "6-model", "DeepSeek",
    "AHMprotocol", "AHM_xyz",
    "61.3", "61.2", "59.3",
    "91% score", "91% of agents",
    "APPROVE/REJECT", "APPROVE or REJECT",
    "40,000 autonomous agents", "over 40,000",
    "four dimensions", "four verifiable dimensions",
]

# Substrings that legitimately contain a forbidden token or a figure-like
# string. Each needs a reason — an exception without one is how a rule dies.
ALLOWED_CONTEXTS = [
    ("_retired_handles", "canonical.json records retired handles so they can be detected"),
    ("_retired_claims", "canonical.json records retired Verify claims for detection"),
    ("check_consistency", "this checker names the strings it forbids"),
    ("There is no APPROVE", "explicit negation in ahm-verify CLAUDE.md"),
    ("no other grade table", "explicit negation in the docs grade section"),
    ("does not exist in the scoring code", "explicit negation of D4"),
    # Only a DATED quotation is excused. A quote is someone else's words and is
    # never edited, but an undated one presents a point-in-time figure as
    # current — so the date is what earns the exemption, not the quotation
    # marks. Adding a quote without a quote-date will fail this check.
    ("quote-date", "attributed third-party quotation, carrying its date"),
    ("of classified", "dated one-off taxonomy sample, not a nightly-scan figure"),
    ("random sample", "dated one-off taxonomy sample, not a nightly-scan figure"),
    ("classification run", "dated one-off taxonomy sample, not a nightly-scan figure"),
]


@dataclass
class Finding:
    surface: str
    line: int
    rule: str
    detail: str
    excerpt: str

    def __str__(self) -> str:
        return (f"  {self.surface}:{self.line}\n"
                f"      {self.rule}: {self.detail}\n"
                f"      > {self.excerpt.strip()[:150]}")


def strip_markup(text: str) -> str:
    """Drop script/style bodies and HTML comments before matching prose.

    Client-side code legitimately contains figure-shaped expressions (it renders
    live values); the rule is about published copy, not the code that fetches it.
    """
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return text


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def excused(line_text: str, full_text: str, idx: int) -> str | None:
    window = full_text[max(0, idx - 300): idx + 300]
    for needle, reason in ALLOWED_CONTEXTS:
        if needle in line_text or needle in window:
            return reason
    return None


def check_text(name: str, raw: str, is_dashboard: bool) -> list[Finding]:
    findings: list[Finding] = []
    prose = strip_markup(raw)
    lines = prose.splitlines()

    if not is_dashboard:
        for pattern, label in CHANGING_FIGURE_PATTERNS:
            for m in re.finditer(pattern, prose, re.I):
                ln = line_of(prose, m.start())
                text = lines[ln - 1] if ln <= len(lines) else ""
                if excused(text, prose, m.start()):
                    continue
                findings.append(Finding(
                    name, ln, "changing figure in static copy", label, m.group(0)))

    for bad in FORBIDDEN_STRINGS:
        for m in re.finditer(re.escape(bad), raw, re.I):
            ln = line_of(raw, m.start())
            text = raw.splitlines()[ln - 1] if ln <= raw.count("\n") + 1 else ""
            if excused(text, raw, m.start()):
                continue
            findings.append(Finding(
                name, ln, "forbidden stale string", repr(bad), text))
    return findings


def extract_grade_bands(path: Path) -> list[tuple[int, str, str]]:
    """Pull the (min, letter, label) ladder out of _ahs_grade by parsing source.

    Returns them in descending threshold order, matching how the function is
    written and how canonical.json lists them. The final bare `return` (the
    else-case, F) is treated as min=0.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_ahs_grade"), None)
    if fn is None:
        raise ValueError("_ahs_grade not found")

    bands: list[tuple[int, str, str]] = []
    for node in fn.body:
        if isinstance(node, ast.If):
            # if score >= N: return "X", "Label"
            cmp_node = node.test
            threshold = cmp_node.comparators[0].value
            letter, label = (e.value for e in node.body[0].value.elts)
            bands.append((threshold, letter, label))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            letter, label = (e.value for e in node.value.elts)
            bands.append((0, letter, label))
    if not bands:
        raise ValueError("no grade bands parsed from _ahs_grade")
    return bands


def check_canonical_against_code() -> list[Finding]:
    """The config must not drift from the code it claims to describe."""
    findings: list[Finding] = []
    src = "config/canonical.json"

    # Read the thresholds out of monitor.py's source rather than importing it.
    # Importing drags in the whole runtime dependency tree, which this check
    # does not need and which is not installed in CI — and a checker that can
    # be defeated by a missing dependency is not a checker.
    try:
        bands = extract_grade_bands(ROOT / "monitor.py")
    except Exception as exc:
        return [Finding(src, 0, "cannot verify grades",
                        f"could not parse _ahs_grade from monitor.py: {exc}", "")]

    canon_bands = [(b["min"], b["letter"], b["label"]) for b in canonical.grade_bands()]
    if bands != canon_bands:
        findings.append(Finding(
            src, 0, "grade table drift",
            f"canonical has {canon_bands}, monitor._ahs_grade has {bands}", ""))

    api = (ROOT / "api.py").read_text(encoding="utf-8")

    listed = len(CANON["endpoints"]["list"])
    if listed != CANON["endpoints"]["count"]:
        findings.append(Finding(
            src, 0, "endpoint count drift",
            f"count says {CANON['endpoints']['count']} but the list has {listed}", ""))

    route_configs = len(re.findall(r'":\s*RouteConfig\(', api))
    expected = listed + len(CANON["endpoints"].get("unlisted_paid_routes", []))
    if route_configs != expected:
        findings.append(Finding(
            src, 0, "endpoint count drift",
            f"api.py registers {route_configs} x402 routes; canonical accounts "
            f"for {expected} ({listed} public + "
            f"{len(CANON['endpoints'].get('unlisted_paid_routes', []))} unlisted)", ""))

    for mode, expect in (("2D", (0.30, 0.70, 0.0)), ("3D", (0.25, 0.45, 0.30))):
        w = canonical.weights(mode)
        if (w["D1"], w["D2"], w["D3"]) != expect:
            findings.append(Finding(
                src, 0, "dimension weight drift",
                f"{mode} weights {w} do not match calculate_ahs {expect}", ""))

    composite = re.search(
        r"composite\s*=\s*0\.25\s*\*\s*d1_score\s*\+\s*0\.45\s*\*\s*d2_score"
        r"\s*\+\s*0\.30\s*\*\s*d3_score", (ROOT / "monitor.py").read_text(encoding="utf-8"))
    if not composite:
        findings.append(Finding(
            src, 0, "dimension weight drift",
            "could not find the 3D composite expression in monitor.py — "
            "weights may have moved", ""))

    return findings


def check_hardcoded_values(surfaces: list[tuple[str, str]]) -> list[Finding]:
    """Stable values must be rendered from canonical, not retyped.

    Only flags values that disagree with canonical — a template repeating the
    correct price is a maintenance smell, but a template repeating a WRONG one
    is a live defect, and that is what blocks a deploy.
    """
    findings: list[Finding] = []
    prices = {ep["slug"]: ep["price"] for ep in canonical.endpoints()}
    handle = canonical.twitter_handle()

    for name, raw in surfaces:
        text = strip_markup(raw)

        # A quoted /risk price that isn't the canonical one.
        for m in re.finditer(r"/risk/\{?address\}?</code>\s*</td>\s*<td>(\$[\d.]+)", raw):
            if m.group(1) != prices["risk"]:
                findings.append(Finding(
                    name, line_of(raw, m.start()), "hard-coded price disagrees with canonical",
                    f"/risk quoted at {m.group(1)}, canonical is {prices['risk']}", m.group(0)))

        # Any @handle that looks like ours but isn't the canonical one.
        # The lookbehind keeps this off email addresses and domains
        # (pablo@agenthealthmonitor.xyz) — we want social handles only.
        for m in re.finditer(r"(?<![\w.])@(AHM\w+|AHM_\w+|agenttrust\w*)", text):
            found = "@" + m.group(1)
            if found.lower() != handle.lower():
                ln = line_of(text, m.start())
                lines = text.splitlines()
                if excused(lines[ln - 1] if ln <= len(lines) else "", text, m.start()):
                    continue
                findings.append(Finding(
                    name, ln, "non-canonical handle",
                    f"{found} — canonical is {handle}", found))

        # Grade bands stated with wrong boundaries.
        for m in re.finditer(r"\bB\b[^\d\n]{0,20}(\d{2})\s*[–\-]\s*(\d{2})", text):
            lo = int(m.group(1))
            if lo not in (75,):
                findings.append(Finding(
                    name, line_of(text, m.start()), "grade band disagrees with canonical",
                    f"Grade B stated as {lo}-{m.group(2)}, canonical is 75-89", m.group(0)))
    return findings


def gather_local(extra_dirs: list[str]) -> list[tuple[str, str, bool]]:
    out: list[tuple[str, str, bool]] = []
    for pattern in DEFAULT_SURFACE_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            out.append((str(p.relative_to(ROOT)), p.read_text(encoding="utf-8", errors="replace"),
                        p.name in DASHBOARD_FILES))

    for rel, globs in SIBLING_SURFACES:
        base = (ROOT / rel).resolve()
        if not base.is_dir():
            continue
        for g in globs:
            for p in sorted(base.rglob(g)):
                out.append((str(p), p.read_text(encoding="utf-8", errors="replace"), False))

    for d in extra_dirs:
        base = Path(d).resolve()
        if not base.is_dir():
            print(f"warning: --surfaces path not found: {d}", file=sys.stderr)
            continue
        for p in sorted(base.rglob("*.html")):
            out.append((str(p), p.read_text(encoding="utf-8", errors="replace"),
                        p.name in DASHBOARD_FILES))
    return out


def gather_live() -> list[tuple[str, str, bool]]:
    """Fetch the deployed surfaces, bypassing the CDN.

    These sites sit behind Cloudflare with a 4-hour TTL. Fetching them plainly
    reads whatever the edge happens to be holding, which makes this check
    worthless in both directions: it reported drift on a homepage that had
    already been fixed, and it would just as happily report a pass while the
    edge served a good page over a broken deploy.

    A unique query string forces a MISS so the response comes from the origin.
    """
    import time
    import urllib.request

    out = []
    bust = str(int(time.time()))
    for url in LIVE_URLS:
        sep = "&" if "?" in url else "?"
        fetch_url = f"{url}{sep}_cc={bust}"
        try:
            req = urllib.request.Request(fetch_url, headers={
                "User-Agent": "ahm-consistency-check",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                cache_status = r.headers.get("cf-cache-status", "")
                if cache_status.upper() == "HIT":
                    print(f"warning: {url} still served from CDN cache "
                          f"(cf-cache-status: {cache_status}) — result may be stale",
                          file=sys.stderr)
                # Report against the clean URL; the buster is a fetch detail.
                out.append((url, r.read().decode("utf-8", "replace"),
                            url.rstrip("/").endswith("/dashboard")))
        except Exception as exc:
            print(f"warning: could not fetch {url}: {exc}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also crawl deployed URLs")
    ap.add_argument("--surfaces", action="append", default=[],
                    help="extra directory of rendered surfaces to check")
    ap.add_argument("--list-known", action="store_true",
                    help="print the accepted exceptions and exit")
    args = ap.parse_args()

    if args.list_known:
        print("Accepted exceptions (context -> reason):")
        for needle, reason in ALLOWED_CONTEXTS:
            print(f"  {needle!r}\n      {reason}")
        return 0

    surfaces = gather_local(args.surfaces)
    if args.live:
        surfaces += gather_live()

    findings = check_canonical_against_code()
    for name, raw, is_dash in surfaces:
        findings += check_text(name, raw, is_dash)
    findings += check_hardcoded_values([(n, r) for n, r, _ in surfaces])

    print(f"Checked {len(surfaces)} surfaces against config/canonical.json\n")
    if not findings:
        print("PASS — no drift found.")
        return 0

    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule, []).append(f)

    for rule, items in sorted(by_rule.items()):
        print(f"{rule.upper()} ({len(items)})")
        for f in items:
            print(f)
        print()

    print(f"FAIL — {len(findings)} issue(s). Changing figures belong on "
          f"/dashboard; stable values belong in config/canonical.json.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
