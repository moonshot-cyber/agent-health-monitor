#!/usr/bin/env python3
"""Generate Agent Report Card — personalised 1200x675 PNG.

Dark terminal aesthetic matching generate_health_report_2.py.
Returns raw PNG bytes via io.BytesIO (no disk writes).
"""

import io
import math
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 675

# -- Colour palette (matching site CSS vars / generate_health_report_2.py) --
BG = (3, 6, 9)
BG2 = (7, 13, 18)
PANEL = (13, 31, 45)
BORDER = (14, 58, 82)
CYAN = (0, 245, 255)
CYAN_DIM = (0, 122, 136)
GREEN = (0, 255, 136)
RED = (255, 68, 68)
AMBER = (255, 187, 51)
TEXT = (200, 232, 240)
TEXT_DIM = (74, 122, 138)
TEXT_BRIGHT = (232, 248, 255)

# -- Fonts --
# Try Windows → Linux common monospace paths → Pillow built-in fallback
def _find_font(candidates: list[str]) -> str | None:
    import os
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

MONO = _find_font([
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
])
MONO_BOLD = _find_font([
    "C:/Windows/Fonts/consolab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
])


def _load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to Pillow's built-in default."""
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow <10.1 doesn't support size= on load_default
        return ImageFont.load_default()


# -- Reusable helpers (from generate_health_report_2.py) --

def draw_scanlines(draw, opacity=8):
    for y in range(0, HEIGHT, 3):
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, opacity), width=1)


def draw_grid(draw):
    for x in range(0, WIDTH, 60):
        draw.line([(x, 0), (x, HEIGHT)], fill=(*BORDER, 20), width=1)
    for y in range(0, HEIGHT, 60):
        draw.line([(0, y), (WIDTH, y)], fill=(*BORDER, 20), width=1)


def draw_corner_brackets(draw, margin=24, size=36, width=2):
    corners = [
        (margin, margin, margin + size, margin, margin, margin + size),
        (WIDTH - margin - size, margin, WIDTH - margin, margin, WIDTH - margin, margin + size),
        (margin, HEIGHT - margin, margin + size, HEIGHT - margin, margin, HEIGHT - margin - size),
        (WIDTH - margin - size, HEIGHT - margin, WIDTH - margin, HEIGHT - margin, WIDTH - margin, HEIGHT - margin - size),
    ]
    for x1, y1, x2, y2, x3, y3 in corners:
        draw.line([(x1, y1), (x2, y2)], fill=CYAN_DIM, width=width)
        draw.line([(x1 if x1 == x3 else x2, y1 if y1 == y3 else y2), (x3, y3)], fill=CYAN_DIM, width=width)


def draw_heartbeat(draw, y_center, x_start, x_end, amplitude=14, color=CYAN_DIM):
    points = []
    seg_width = x_end - x_start
    for x in range(x_start, x_end):
        t = (x - x_start) / seg_width
        if 0.42 < t < 0.45:
            y = y_center - amplitude * ((t - 0.42) / 0.03)
        elif 0.45 <= t < 0.48:
            y = y_center - amplitude
        elif 0.48 <= t < 0.50:
            y = y_center - amplitude + amplitude * 2.5 * ((t - 0.48) / 0.02)
        elif 0.50 <= t < 0.53:
            y = y_center + amplitude * 1.5 - amplitude * 1.5 * ((t - 0.50) / 0.03)
        elif 0.53 <= t < 0.56:
            y = y_center - amplitude * 0.3 * math.sin((t - 0.53) / 0.03 * math.pi)
        else:
            y = y_center
        points.append((x, int(y)))
    for offset, alpha in [(2, 40), (1, 80), (0, 180)]:
        glow_color = (*color[:3], alpha)
        shifted = [(px, py + offset) for px, py in points] if offset else points
        for i in range(len(shifted) - 1):
            draw.line([shifted[i], shifted[i + 1]], fill=glow_color, width=2 if offset == 0 else 1)


def draw_bar(draw, x, y, width, height, fill_pct, fill_color, bg_color=PANEL):
    draw.rounded_rectangle([(x, y), (x + width, y + height)], radius=4, fill=bg_color)
    fill_w = max(8, int(width * fill_pct))
    draw.rounded_rectangle([(x, y), (x + fill_w, y + height)], radius=4, fill=fill_color)


def center_text(draw, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((WIDTH - w) // 2, y), text, font=font, fill=fill)


def _grade_color(grade: str):
    """Return colour tuple for a grade letter."""
    g = grade.upper().strip()
    if g == "A":
        return GREEN
    if g == "B":
        return CYAN
    if g == "C":
        return AMBER
    if g in ("D", "E"):
        return RED
    return RED  # F


def _score_color(score: int):
    if score >= 90:
        return GREEN
    if score >= 75:
        return CYAN
    if score >= 60:
        return AMBER
    if score >= 40:
        return (255, 136, 0)
    return RED


def _shorten_address(address: str) -> str:
    if len(address) >= 10:
        return address[:6] + "..." + address[-4:]
    return address


def _percentile_rank(score: int, percentiles: dict) -> int:
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
    # If score is below p10
    if score <= checkpoints[0][0]:
        return max(1, int(10 * score / max(checkpoints[0][0], 1)))
    # If score is above p90
    if score >= checkpoints[-1][0]:
        return min(99, 90 + int(10 * (score - checkpoints[-1][0]) / max(100 - checkpoints[-1][0], 1)))
    # Interpolate
    for i in range(len(checkpoints) - 1):
        s1, p1 = checkpoints[i]
        s2, p2 = checkpoints[i + 1]
        if s1 <= score <= s2:
            if s2 == s1:
                return p1
            return int(p1 + (p2 - p1) * (score - s1) / (s2 - s1))
    return 50


def generate_report_card(
    address: str,
    ahs_score: int,
    grade: str,
    grade_label: str,
    d1_score: int,
    d2_score: int,
    d3_score: int | None,
    mode: str,
    patterns: list[dict],
    recommendations: list[str],
    ecosystem: dict,
) -> bytes:
    """Generate report card PNG and return raw bytes."""
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    # Background gradient
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(BG[0] + (BG2[0] - BG[0]) * t)
        g = int(BG[1] + (BG2[1] - BG[1]) * t)
        b = int(BG[2] + (BG2[2] - BG[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    draw_grid(draw)
    draw_corner_brackets(draw)

    # Load fonts
    fonts = {
        "title": _load_font(MONO_BOLD, 26),
        "subtitle": _load_font(MONO, 14),
        "big_score": _load_font(MONO_BOLD, 80),
        "grade": _load_font(MONO_BOLD, 28),
        "grade_label": _load_font(MONO, 16),
        "dim_title": _load_font(MONO_BOLD, 13),
        "dim_score": _load_font(MONO_BOLD, 30),
        "dim_label": _load_font(MONO, 11),
        "section": _load_font(MONO_BOLD, 13),
        "body": _load_font(MONO, 13),
        "small": _load_font(MONO, 11),
        "footer": _load_font(MONO, 12),
        "corner": _load_font(MONO, 13),
        "bar_label": _load_font(MONO_BOLD, 12),
    }

    gc = _grade_color(grade)

    # ── Top-left / top-right labels ──
    draw.text((50, 34), "// AGENT REPORT CARD", font=fonts["corner"], fill=TEXT_DIM)
    status_text = f"STATUS: {grade}"
    draw.text((WIDTH - 50 - draw.textbbox((0, 0), status_text, font=fonts["corner"])[2], 34),
              status_text, font=fonts["corner"], fill=gc)

    # ── Header: AGENT HEALTH SCORE ──
    center_text(draw, 60, "AGENT HEALTH SCORE", fonts["title"], CYAN)

    # ── Big score number ──
    sc = _score_color(ahs_score)
    score_str = str(ahs_score)
    bbox = draw.textbbox((0, 0), score_str, font=fonts["big_score"])
    bw = bbox[2] - bbox[0]
    bx = (WIDTH - bw) // 2
    by = 88
    # Glow
    for offset in [4, 3, 2, 1]:
        draw.text((bx + offset, by + offset), score_str, font=fonts["big_score"], fill=(*sc[:3], 30))
    draw.text((bx, by), score_str, font=fonts["big_score"], fill=sc)

    # Grade + label
    grade_full = f"Grade: {grade}"
    center_text(draw, 178, grade_full, fonts["grade"], gc)
    center_text(draw, 212, grade_label, fonts["grade_label"], TEXT_DIM)

    # ── Heartbeat line ──
    hb_y = 240
    draw_heartbeat(draw, y_center=hb_y, x_start=50, x_end=WIDTH - 50, amplitude=10, color=CYAN_DIM)

    # ── Dimension score cards ──
    card_w = 300
    card_h = 80
    card_y = 262
    card_gap = 40
    total_cards = 3
    total_w = total_cards * card_w + (total_cards - 1) * card_gap
    card_start_x = (WIDTH - total_w) // 2

    dim_data = [
        ("D1: Wallet Hygiene", d1_score),
        ("D2: Behavioural", d2_score),
        ("D3: Infrastructure", d3_score),
    ]

    for i, (label, score) in enumerate(dim_data):
        cx = card_start_x + i * (card_w + card_gap)
        # Panel background
        draw.rounded_rectangle(
            [(cx, card_y), (cx + card_w, card_y + card_h)],
            radius=6, fill=PANEL, outline=BORDER, width=1,
        )
        # Dimension label
        draw.text((cx + 12, card_y + 8), label, font=fonts["dim_title"], fill=TEXT_DIM)
        # Score value
        if score is not None:
            ds_color = _score_color(score)
            score_text = str(score)
            draw.text((cx + 12, card_y + 30), score_text, font=fonts["dim_score"], fill=ds_color)
            # Mini bar
            bar_x = cx + 12
            bar_y = card_y + 65
            bar_w = card_w - 24
            draw_bar(draw, bar_x, bar_y, bar_w, 6, score / 100.0, ds_color)
        else:
            ns_font = _load_font(MONO, 20)
            draw.text((cx + 12, card_y + 38), "Not scored", font=ns_font, fill=TEXT_DIM)

    # ── Ecosystem Comparison ──
    eco_y = card_y + card_h + 20
    eco_x = 80
    eco_w = WIDTH - 160
    eco_h = 90

    draw.rounded_rectangle(
        [(eco_x, eco_y), (eco_x + eco_w, eco_y + eco_h)],
        radius=6, fill=PANEL, outline=BORDER, width=1,
    )
    draw.text((eco_x + 14, eco_y + 8), "ECOSYSTEM COMPARISON", font=fonts["section"], fill=CYAN_DIM)

    average_ahs = ecosystem.get("average_ahs") or 0
    percentiles = ecosystem.get("baseline_calibration", {}).get("score_percentiles", {})
    total_agents = ecosystem.get("summary", {}).get("total_unique_addresses", 0)
    pct_rank = _percentile_rank(ahs_score, percentiles)

    # Stats row
    stats_y = eco_y + 28
    col_w = eco_w // 3
    stats = [
        (f"Your Score: {ahs_score}", sc),
        (f"Avg: {average_ahs:.0f}", AMBER),
        (f"Rank: Top {100 - pct_rank}%", GREEN if pct_rank >= 50 else AMBER),
    ]
    for i, (txt, color) in enumerate(stats):
        sx = eco_x + 14 + i * col_w
        draw.text((sx, stats_y), txt, font=fonts["body"], fill=color)

    # Comparison bar
    bar_total_y = eco_y + 52
    bar_total_w = eco_w - 28
    # Background bar
    draw_bar(draw, eco_x + 14, bar_total_y, bar_total_w, 14, 1.0, PANEL, bg_color=(*BORDER, 60))
    # Average marker
    if average_ahs > 0:
        avg_x = eco_x + 14 + int(bar_total_w * average_ahs / 100)
        draw.line([(avg_x, bar_total_y - 2), (avg_x, bar_total_y + 16)], fill=AMBER, width=2)
        draw.text((avg_x - 8, bar_total_y + 18), "avg", font=fonts["small"], fill=AMBER)
    # Your score fill
    draw_bar(draw, eco_x + 14, bar_total_y, bar_total_w, 14, ahs_score / 100.0, sc)
    # Agents count
    agents_text = f"{total_agents:,} agents scored"
    at_bbox = draw.textbbox((0, 0), agents_text, font=fonts["small"])
    draw.text((eco_x + eco_w - 14 - (at_bbox[2] - at_bbox[0]), eco_y + eco_h - 18),
              agents_text, font=fonts["small"], fill=TEXT_DIM)

    # ── Patterns detected ──
    pat_y = eco_y + eco_h + 14
    detected = [p for p in patterns if p.get("detected")]
    has_zombie = any(p.get("name", "").lower().startswith("zombie") for p in detected)
    has_healthy = any(p.get("name", "").lower().startswith("healthy") for p in detected)
    if detected:
        pat_text = "Patterns: " + ", ".join(
            f'{p["name"]} ({"detected" if p["detected"] else "clear"})'
            for p in detected[:3]
        )
        pat_color = RED if any(p.get("severity") == "critical" for p in detected) else AMBER
        draw.text((80, pat_y), pat_text, font=fonts["body"], fill=pat_color)
    else:
        draw.text((80, pat_y), "Patterns: None detected", font=fonts["body"], fill=GREEN)

    # ── Tip line ──
    tip_y = pat_y + 20
    if has_zombie:
        tip_text = "Tip: Increase counterparty diversity to improve D2 score"
    elif has_healthy:
        tip_text = "Tip: Maintain current activity patterns"
    elif detected:
        tip_text = "Tip: Review detected patterns and address recommendations"
    else:
        tip_text = "Tip: Run /wash for a detailed hygiene breakdown"
    draw.text((80, tip_y), tip_text, font=fonts["small"], fill=CYAN_DIM)

    # ── Footer ──
    footer_y = tip_y + 38
    short_addr = _shorten_address(address)
    now_str = datetime.now(timezone.utc).strftime("%B %Y")
    footer_line = f"{short_addr}  \u00b7  Base Mainnet  \u00b7  {now_str}"
    center_text(draw, footer_y - 14, footer_line, fonts["footer"], TEXT_DIM)
    center_text(draw, footer_y + 2, "agenthealthmonitor.xyz", fonts["footer"], TEXT_DIM)

    # Small heartbeat in footer
    draw_heartbeat(draw, y_center=footer_y + 10, x_start=WIDTH // 2 - 180,
                   x_end=WIDTH // 2 - 120, amplitude=6, color=CYAN)

    # Scanlines last
    draw_scanlines(draw, opacity=10)

    # Export to bytes
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, "PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    # Standalone test with mock data
    png_bytes = generate_report_card(
        address="0xAbCdEf1234567890AbCdEf1234567890AbCdEfGh",
        ahs_score=72,
        grade="C",
        grade_label="Needs Attention",
        d1_score=68,
        d2_score=55,
        d3_score=None,
        mode="2D",
        patterns=[
            {"name": "Zombie Agent", "detected": True, "severity": "high",
             "description": "Single counterparty with near-zero diversity."},
        ],
        recommendations=["Diversify counterparties", "Enable gas adaptation"],
        ecosystem={
            "average_ahs": 59.3,
            "summary": {"total_unique_addresses": 4552},
            "baseline_calibration": {
                "score_percentiles": {"p10": 18, "p25": 30, "p50": 45, "p75": 62, "p90": 78},
            },
            "grade_distribution": {"A": 0, "B": 3, "C": 12, "D": 45, "E": 38, "F": 8},
        },
    )
    out_path = "test_report_card.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"[+] Test report card saved: {out_path} ({len(png_bytes):,} bytes)")
