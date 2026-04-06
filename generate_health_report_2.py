#!/usr/bin/env python3
"""Generate Agent Economy Health Report #2 — Pattern Analysis.

1200x675 dark terminal aesthetic PNG for sharing on X.
Matches the color palette and style from generate_og_banner.py.
"""

import math
import traceback
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 675
OUT = "health_report_2_march_2026.png"

# -- Colour palette (from site CSS vars) --
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
MONO = "C:/Windows/Fonts/consola.ttf"
MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"


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
        shifted = [(x, y + offset) for x, y in points] if offset else points
        for i in range(len(shifted) - 1):
            draw.line([shifted[i], shifted[i + 1]], fill=glow_color, width=2 if offset == 0 else 1)


def draw_bar(draw, x, y, width, height, fill_pct, fill_color, bg_color=PANEL):
    """Draw a horizontal progress bar with rounded-ish ends."""
    # Background
    draw.rounded_rectangle([(x, y), (x + width, y + height)], radius=4, fill=bg_color)
    # Fill
    fill_w = max(8, int(width * fill_pct))
    draw.rounded_rectangle([(x, y), (x + fill_w, y + height)], radius=4, fill=fill_color)


def draw_stat_card(draw, cx, y, value, label, value_color=CYAN, fonts=None):
    """Draw a stat card centered at cx."""
    card_w, card_h = 300, 100
    x = cx - card_w // 2
    # Panel background
    draw.rounded_rectangle(
        [(x, y), (x + card_w, y + card_h)],
        radius=6, fill=PANEL, outline=BORDER, width=1,
    )
    # Value
    vfont = fonts["stat_value"]
    bbox = draw.textbbox((0, 0), value, font=vfont)
    vw = bbox[2] - bbox[0]
    draw.text((cx - vw // 2, y + 12), value, font=vfont, fill=value_color)
    # Label
    lfont = fonts["stat_label"]
    bbox = draw.textbbox((0, 0), label, font=lfont)
    lw = bbox[2] - bbox[0]
    draw.text((cx - lw // 2, y + 65), label, font=lfont, fill=TEXT_DIM)


def center_text(draw, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text(((WIDTH - w) // 2, y), text, font=font, fill=fill)


def _fetch_ecosystem_stats() -> dict:
    """Fetch live ecosystem stats from the DB, with sensible fallbacks."""
    defaults = {
        "total_scanned": 4552,
        "avg_ahs": 59.3,
        "grade_distribution": {},
        "pattern_distribution": {},
    }
    try:
        import db
        stats = db.get_ecosystem_dashboard_stats()
        if stats and stats.get("total_scanned"):
            return stats
    except Exception:
        traceback.print_exc()
    return defaults


def main():
    # Fetch live stats for the stat cards
    eco = _fetch_ecosystem_stats()
    total_agents = eco.get("total_scanned", 4552)
    avg_ahs = eco.get("avg_ahs", 59.3)
    grades = eco.get("grade_distribution", {})
    total_graded = sum(grades.values()) if grades else total_agents
    grade_a_count = grades.get("A", 0)
    grade_de_count = grades.get("D", 0) + grades.get("E", 0)
    grade_a_pct = round(grade_a_count / total_graded * 100, 1) if total_graded else 0
    grade_de_pct = round(grade_de_count / total_graded * 100) if total_graded else 0
    zombie_count = eco.get("pattern_distribution", {}).get("Zombie Agent", 0)
    zombie_pct = round(zombie_count / total_graded * 100) if total_graded else 0

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
        "title": ImageFont.truetype(MONO_BOLD, 32),
        "subtitle": ImageFont.truetype(MONO, 16),
        "big_stat": ImageFont.truetype(MONO_BOLD, 96),
        "big_label": ImageFont.truetype(MONO_BOLD, 22),
        "big_sub": ImageFont.truetype(MONO, 15),
        "bar_label": ImageFont.truetype(MONO_BOLD, 17),
        "bar_pct": ImageFont.truetype(MONO_BOLD, 17),
        "stat_value": ImageFont.truetype(MONO_BOLD, 36),
        "stat_label": ImageFont.truetype(MONO, 14),
        "footer": ImageFont.truetype(MONO, 15),
        "corner": ImageFont.truetype(MONO, 14),
    }

    # ── Top-left / top-right labels ──
    draw.text((50, 34), "// PATTERN ANALYSIS", font=fonts["corner"], fill=TEXT_DIM)
    draw.text((WIDTH - 230, 34), "STATUS: OPERATIONAL", font=fonts["corner"], fill=GREEN)

    # ── Header ──
    center_text(draw, 60, "AGENT ECONOMY HEALTH REPORT #2", fonts["title"], CYAN)

    # Subtitle
    sub = f"Pattern Analysis \u2014 {total_agents:,} agents \u2014 Base mainnet"
    center_text(draw, 100, sub, fonts["subtitle"], TEXT_DIM)

    # Accent line
    line_y = 125
    draw.line([(WIDTH // 2 - 200, line_y), (WIDTH // 2 + 200, line_y)], fill=BORDER, width=1)

    # ── Main stat: Zombie % ──
    big_val = f"{zombie_pct}%"
    bbox = draw.textbbox((0, 0), big_val, font=fonts["big_stat"])
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # Glow layers
    bx = (WIDTH - bw) // 2
    by = 140
    for offset in [4, 3, 2, 1]:
        draw.text((bx + offset, by + offset), big_val, font=fonts["big_stat"], fill=(*RED[:3], 30))
    draw.text((bx, by), big_val, font=fonts["big_stat"], fill=RED)

    # Label under big stat
    label = "Zombie Agent Patterns Detected"
    center_text(draw, 248, label, fonts["big_label"], TEXT_BRIGHT)

    # Subtitle
    sub2 = "Single counterparty \u00b7 Near-zero diversity \u00b7 Effectively dormant"
    center_text(draw, 278, sub2, fonts["big_sub"], TEXT_DIM)

    # ── Pattern bars ──
    bar_x = 200
    bar_w = 580
    bar_h = 26
    bar_y = 320

    # Zombie Agent bar
    no_pattern_pct = 100 - zombie_pct
    draw.text((bar_x, bar_y - 22), "Zombie Agent", font=fonts["bar_label"], fill=RED)
    draw.text((bar_x + bar_w + 14, bar_y + 2), f"{zombie_pct}%", font=fonts["bar_pct"], fill=RED)
    draw_bar(draw, bar_x, bar_y, bar_w, bar_h, zombie_pct / 100, RED)

    # No pattern bar
    bar_y2 = bar_y + 52
    draw.text((bar_x, bar_y2 - 22), "No Pattern Detected", font=fonts["bar_label"], fill=GREEN)
    draw.text((bar_x + bar_w + 14, bar_y2 + 2), f"{no_pattern_pct}%", font=fonts["bar_pct"], fill=GREEN)
    draw_bar(draw, bar_x, bar_y2, bar_w, bar_h, no_pattern_pct / 100, GREEN)

    # ── Divider ──
    div_y = bar_y2 + 48

    # ── Heartbeat line (subtle, overlaid on divider) ──
    draw_heartbeat(draw, y_center=div_y, x_start=50, x_end=WIDTH - 50, amplitude=10, color=CYAN_DIM)

    # ── Secondary stat cards ──
    card_y = div_y + 18
    card_centers = [WIDTH // 4, WIDTH // 2, 3 * WIDTH // 4]

    draw_stat_card(draw, card_centers[0], card_y, str(avg_ahs), "avg AHS score",
                   value_color=AMBER, fonts=fonts)
    draw_stat_card(draw, card_centers[1], card_y, f"{grade_de_pct}%", "D or E grade",
                   value_color=RED, fonts=fonts)
    draw_stat_card(draw, card_centers[2], card_y, f"{grade_a_pct}%", "A-grade agents",
                   value_color=TEXT_DIM, fonts=fonts)

    # ── Footer ──
    footer_y = HEIGHT - 46
    center_text(draw, footer_y, "agenthealthmonitor.xyz", fonts["footer"], TEXT_DIM)

    # Small heartbeat in footer
    draw_heartbeat(draw, y_center=footer_y + 8, x_start=WIDTH // 2 - 180,
                   x_end=WIDTH // 2 - 120, amplitude=6, color=CYAN)

    # Scanlines last
    draw_scanlines(draw, opacity=10)

    # Save
    img_rgb = img.convert("RGB")
    img_rgb.save(OUT, "PNG", optimize=True)
    print(f"[+] Saved: {OUT} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
