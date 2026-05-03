#!/usr/bin/env python3
"""
Dedupe check for AHM backlog entries.

Run before adding any new entry to ahm_backlog.md. Surfaces near-matches in the
existing backlog by keyword overlap so the human can decide: true duplicate,
near-miss requiring cross-reference, or no match.

Usage:
    # Check a proposed entry against the backlog
    python dedupe_check.py --backlog path/to/ahm_backlog.md --proposed "ThoughtProof PLV composition concept"

    # Read a longer proposed entry from a file (title + body)
    python dedupe_check.py --backlog path/to/ahm_backlog.md --proposed-file proposal.txt

    # Check several proposed entries at once (one per line in a file)
    python dedupe_check.py --backlog path/to/ahm_backlog.md --batch-file proposals.txt

    # Audit the whole backlog for internal duplicates
    python dedupe_check.py --backlog path/to/ahm_backlog.md --audit

The script is intentionally simple: TF-style keyword overlap, no embeddings. It's a
suggestion tool, not a decision tool. The human reads each match and decides.
"""

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Words common enough to be useless as match signal.
STOPWORDS = set("""
a an and or the of to in on at for by with from as is are was were be been being
this that these those it its they them their there here which who whom whose
not no yes do does did done has have had having can could should would may might
will shall i we you he she his her our your my mine ours yours
about into over under again further then once
all any both each few more most other some such only own same so than too very
just but if because while although though when where why how
am i'm we're you're he's she's it's they're i've we've you've they've
ahm need should after before currently per via etc
""".split())

# Treat hyphenated identifiers, ERC numbers, PR refs, addresses, etc. as single tokens.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]*[A-Za-z0-9]|0x[a-fA-F0-9]+|#\d+")


@dataclass
class Entry:
    """A single backlog entry with enough metadata to surface in dedupe output."""
    section_path: str        # e.g. "P2 — Product Backlog → New Endpoints / Features"
    title: str               # extracted bolded title or heading text
    body: str                # the rest of the entry text
    line_number: int         # for traceback in the source file
    tokens: set = field(default_factory=set)


def tokenize(text: str) -> set:
    """Lowercase, strip stopwords, return token set."""
    text = text.lower()
    raw = TOKEN_RE.findall(text)
    return {t for t in raw if t not in STOPWORDS and len(t) > 1}


def parse_backlog(path: Path) -> list[Entry]:
    """
    Walk the backlog, building an Entry per checkbox bullet and per heading-style entry.

    Section path is tracked by maintaining a stack of current headings at each level.
    """
    entries = []
    section_stack = []  # list of (level, title)
    current_entry_lines = []
    current_entry_meta = None  # (line_no, title) for in-flight entry

    def flush():
        nonlocal current_entry_lines, current_entry_meta
        if current_entry_meta is None:
            return
        line_no, title = current_entry_meta
        body = "\n".join(current_entry_lines).strip()
        section_path = " → ".join(t for _, t in section_stack)
        entry = Entry(
            section_path=section_path,
            title=title,
            body=body,
            line_number=line_no,
            tokens=tokenize(title + " " + body),
        )
        entries.append(entry)
        current_entry_lines = []
        current_entry_meta = None

    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        # Heading
        m = re.match(r"^(#{2,6})\s+(.+)$", line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            # Pop stack to current level
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, heading))
            # A heading might itself be an entry (heading-style entry).
            # We don't auto-create an Entry for a heading — a heading-style entry's
            # body comes from the bullets that follow until the next heading or
            # checkbox-entry.
            continue

        # Checkbox bullet (top-level entry)
        m = re.match(r"^- \[[ x]\]\s+(.+)$", line)
        if m:
            flush()
            rest = m.group(1)
            # Extract bolded title if present
            bold = re.match(r"^\*\*([^*]+)\*\*\s*(?:\(completed [^)]+\))?\s*[—\-]?\s*(.*)$", rest)
            if bold:
                title = bold.group(1).strip()
                body_start = bold.group(2).strip()
            else:
                # Untitled entry: use first ~10 words as pseudo-title
                title = " ".join(rest.split()[:10])
                body_start = rest
            current_entry_meta = (i, title)
            current_entry_lines = [body_start] if body_start else []
            continue

        # Continuation lines (sub-bullet, blank line, or prose continuation)
        if current_entry_meta is not None:
            current_entry_lines.append(line)

    flush()
    return entries


def parse_proposed(text: str) -> tuple[str, str]:
    """Split a proposed entry into (title, body). Accepts:
       - A markdown bullet line: '- [ ] **Title** — body...'
       - A title-prefixed string: 'Title — body...'
       - Just a bare description (whole thing becomes body, title is best-effort)
    """
    text = text.strip()
    # Strip leading checkbox if present
    text = re.sub(r"^- \[[ x]\]\s+", "", text)
    bold = re.match(r"^\*\*([^*]+)\*\*\s*[—\-]?\s*(.*)$", text, re.DOTALL)
    if bold:
        return bold.group(1).strip(), bold.group(2).strip()
    if "—" in text:
        title, body = text.split("—", 1)
        return title.strip(), body.strip()
    return " ".join(text.split()[:10]), text


def overlap_score(a: set, b: set) -> float:
    """Symmetric overlap: |A ∩ B| / min(|A|, |B|). Range 0..1."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def find_matches(proposed_title: str, proposed_body: str, entries: list[Entry], top_n: int = 5):
    """Return top_n entries ranked by token overlap."""
    proposed_tokens = tokenize(proposed_title + " " + proposed_body)
    if not proposed_tokens:
        return []
    scored = []
    for e in entries:
        s = overlap_score(proposed_tokens, e.tokens)
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_n]


def format_match(score: float, e: Entry, max_body: int = 200) -> str:
    body_preview = e.body.replace("\n", " ").strip()
    if len(body_preview) > max_body:
        body_preview = body_preview[:max_body] + "..."
    return (
        f"  [{score:.2f}] line {e.line_number}\n"
        f"        Section: {e.section_path}\n"
        f"        Title:   {e.title}\n"
        f"        Body:    {body_preview}\n"
    )


def check_proposed(proposed_text: str, entries: list[Entry], threshold: float = 0.25) -> int:
    """Run dedupe for a single proposed entry. Returns number of matches above threshold."""
    title, body = parse_proposed(proposed_text)
    print(f"\nProposed:")
    print(f"  Title: {title}")
    if body:
        body_preview = body[:200] + ("..." if len(body) > 200 else "")
        print(f"  Body:  {body_preview}")
    matches = find_matches(title, body, entries, top_n=5)
    above_threshold = [(s, e) for s, e in matches if s >= threshold]
    if not above_threshold:
        print(f"\n  No near-matches above threshold {threshold}. Safe to add.")
        return 0
    print(f"\n  {len(above_threshold)} potential match(es) at score >= {threshold}:")
    for s, e in above_threshold:
        print(format_match(s, e))
    if len(matches) > len(above_threshold):
        weak = [(s, e) for s, e in matches if s < threshold]
        print(f"  {len(weak)} weaker match(es) below threshold (for reference):")
        for s, e in weak[:3]:
            print(format_match(s, e))
    return len(above_threshold)


def audit_internal(entries: list[Entry], threshold: float = 0.5) -> int:
    """Cross-check every entry against every other; report pairs above threshold."""
    print(f"\nAuditing {len(entries)} entries for internal duplicates (threshold {threshold})...\n")
    pairs = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            s = overlap_score(a.tokens, b.tokens)
            if s >= threshold:
                pairs.append((s, a, b))
    pairs.sort(key=lambda x: -x[0])
    if not pairs:
        print("No duplicate pairs above threshold.")
        return 0
    print(f"Found {len(pairs)} suspicious pair(s):\n")
    for s, a, b in pairs[:30]:
        print(f"[{s:.2f}]")
        print(f"  A (line {a.line_number}, {a.section_path})")
        print(f"     {a.title}")
        print(f"  B (line {b.line_number}, {b.section_path})")
        print(f"     {b.title}")
        print()
    if len(pairs) > 30:
        print(f"... and {len(pairs) - 30} more pairs not shown.")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Dedupe check for AHM backlog.")
    parser.add_argument("--backlog", required=True, type=Path, help="Path to ahm_backlog.md")
    parser.add_argument("--proposed", type=str, help="Proposed entry text (single line or quoted)")
    parser.add_argument("--proposed-file", type=Path, help="Path to file containing one proposed entry")
    parser.add_argument("--batch-file", type=Path, help="Path to file with one proposed entry per line")
    parser.add_argument("--audit", action="store_true", help="Audit existing backlog for internal dupes")
    parser.add_argument("--threshold", type=float, default=0.25, help="Match threshold (0..1, default 0.25)")
    args = parser.parse_args()

    if not args.backlog.exists():
        print(f"Backlog not found: {args.backlog}", file=sys.stderr)
        sys.exit(2)

    entries = parse_backlog(args.backlog)
    print(f"Parsed {len(entries)} entries from {args.backlog}")

    modes_used = sum(bool(x) for x in (args.proposed, args.proposed_file, args.batch_file, args.audit))
    if modes_used == 0:
        parser.error("Specify --proposed, --proposed-file, --batch-file, or --audit")
    if modes_used > 1:
        parser.error("Specify only one of --proposed / --proposed-file / --batch-file / --audit")

    if args.audit:
        n = audit_internal(entries, threshold=max(args.threshold, 0.5))
        sys.exit(0 if n == 0 else 1)

    if args.proposed:
        n = check_proposed(args.proposed, entries, threshold=args.threshold)
        sys.exit(0 if n == 0 else 1)

    if args.proposed_file:
        text = args.proposed_file.read_text(encoding="utf-8")
        n = check_proposed(text, entries, threshold=args.threshold)
        sys.exit(0 if n == 0 else 1)

    if args.batch_file:
        any_match = False
        for line in args.batch_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n = check_proposed(line, entries, threshold=args.threshold)
            if n:
                any_match = True
        sys.exit(0 if not any_match else 1)


if __name__ == "__main__":
    main()
