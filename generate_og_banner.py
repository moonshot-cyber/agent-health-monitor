#!/usr/bin/env python3
"""Generate the AHM OG banner image (1200x630) matching site aesthetic."""

from PIL import Image, ImageDraw, ImageFont
import math

WIDTH, HEIGHT = 1200, 630
OUT = "static/ahm-og-banner.png"

# -- Colour palette (from index.html CSS vars) --
BG = (3, 6, 9)           # --bg: #030609
BG2 = (7, 13, 18)        # --bg2: #070d12
PANEL = (13, 31, 45)     # --panel: #0d1f2d
BORDER = (14, 58, 82)    # --border: #0e3a52
CYAN = (0, 245, 255)     # --cyan: #00f5ff
CYAN_DIM = (0, 122, 136) # --cyan-dim: #007a88
GREEN = (0, 255, 136)    # --green: #00ff88
TEXT = (200, 232, 240)    # --text: #c8e8f0
TEXT_DIM = (74, 122, 138) # --text-dim: #4a7a8a
TEXT_BRIGHT = (232, 248, 255)  # --text-bright: #e8f8ff

# -- Fonts --
MONO = "C:/Windows/Fonts/consola.ttf"
MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"


def draw_scanlines(draw, opacity=8):
    """Subtle horizontal scanlines for CRT effect."""
    for y in range(0, HEIGHT, 3):
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, opacity), width=1)


def draw_grid(draw):
    """Faint grid pattern."""
    for x in range(0, WIDTH, 60):
        draw.line([(x, 0), (x, HEIGHT)], fill=(*BORDER, 25), width=1)
    for y in range(0, HEIGHT, 60):
        draw.line([(0, y), (WIDTH, y)], fill=(*BORDER, 25), width=1)


def draw_corner_brackets(draw, margin=30, size=40, width=2):
    """HUD-style corner brackets."""
    corners = [
        (margin, margin, margin + size, margin, margin, margin + size),           # top-left
        (WIDTH - margin - size, margin, WIDTH - margin, margin, WIDTH - margin, margin + size),  # top-right
        (margin, HEIGHT - margin, margin + size, HEIGHT - margin, margin, HEIGHT - margin - size),  # bottom-left
        (WIDTH - margin - size, HEIGHT - margin, WIDTH - margin, HEIGHT - margin, WIDTH - margin, HEIGHT - margin - size),  # bottom-right
    ]
    for x1, y1, x2, y2, x3, y3 in corners:
        draw.line([(x1, y1), (x2, y2)], fill=CYAN_DIM, width=width)
        draw.line([(x1 if x1 == x3 else x2, y1 if y1 == y3 else y2), (x3, y3)], fill=CYAN_DIM, width=width)


def draw_heartbeat(draw, y_center, x_start, x_end, amplitude=20, color=CYAN):
    """Draw an ECG/heartbeat line like the logo."""
    points = []
    seg_width = x_end - x_start
    for x in range(x_start, x_end):
        t = (x - x_start) / seg_width
        # Flat baseline with a spike in the middle
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

    # Draw with glow effect
    for offset, alpha in [(3, 30), (2, 60), (1, 120), (0, 255)]:
        glow_color = (*color[:3], alpha) if offset > 0 else color
        shifted = [(x, y + offset) for x, y in points] if offset else points
        for i in range(len(shifted) - 1):
            draw.line([shifted[i], shifted[i + 1]], fill=glow_color, width=2 if offset == 0 else 1)


def main():
    # Create RGBA image for transparency support in compositing
    img = Image.new("RGBA", (WIDTH, HEIGHT), BG + (255,))
    draw = ImageDraw.Draw(img, "RGBA")

    # Background gradient (top darker, bottom slightly lighter)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(BG[0] + (BG2[0] - BG[0]) * t)
        g = int(BG[1] + (BG2[1] - BG[1]) * t)
        b = int(BG[2] + (BG2[2] - BG[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b, 255))

    # Grid overlay
    draw_grid(draw)

    # Corner brackets
    draw_corner_brackets(draw)

    # Heartbeat line across the image
    draw_heartbeat(draw, y_center=315, x_start=60, x_end=1140, amplitude=25, color=CYAN)

    # -- Text --
    # Load fonts
    title_font = ImageFont.truetype(MONO_BOLD, 64)
    subtitle_font = ImageFont.truetype(MONO, 24)
    tag_font = ImageFont.truetype(MONO, 20)
    url_font = ImageFont.truetype(MONO, 18)

    # "AGENT HEALTH MONITOR" — main title
    title = "AGENT HEALTH"
    title2 = "MONITOR"
    bbox1 = draw.textbbox((0, 0), title, font=title_font)
    bbox2 = draw.textbbox((0, 0), title2, font=title_font)
    tw1 = bbox1[2] - bbox1[0]
    tw2 = bbox2[2] - bbox2[0]

    # Title glow
    for offset in [3, 2, 1]:
        draw.text(((WIDTH - tw1) // 2 + offset, 155 + offset), title, font=title_font, fill=(*CYAN[:3], 40))
        draw.text(((WIDTH - tw2) // 2 + offset, 225 + offset), title2, font=title_font, fill=(*CYAN[:3], 40))

    draw.text(((WIDTH - tw1) // 2, 155), title, font=title_font, fill=CYAN)
    draw.text(((WIDTH - tw2) // 2, 225), title2, font=title_font, fill=TEXT_BRIGHT)

    # Subtitle
    subtitle = "Diagnostics for the Agent Economy"
    bbox_sub = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    sw = bbox_sub[2] - bbox_sub[0]
    draw.text(((WIDTH - sw) // 2, 360), subtitle, font=subtitle_font, fill=TEXT)

    # Accent line under subtitle
    line_y = 395
    line_hw = 180
    draw.line([(WIDTH // 2 - line_hw, line_y), (WIDTH // 2 + line_hw, line_y)], fill=CYAN_DIM, width=1)

    # Tags: "Base  •  x402  •  USDC"
    tags = "Base  \u2022  x402  \u2022  USDC"
    bbox_tag = draw.textbbox((0, 0), tags, font=tag_font)
    tag_w = bbox_tag[2] - bbox_tag[0]
    draw.text(((WIDTH - tag_w) // 2, 415), tags, font=tag_font, fill=CYAN_DIM)

    # "11 pay-per-call endpoints  •  No accounts  •  Just USDC"
    detail = "11 pay-per-call endpoints  \u2022  No accounts  \u2022  Just USDC"
    bbox_det = draw.textbbox((0, 0), detail, font=url_font)
    det_w = bbox_det[2] - bbox_det[0]
    draw.text(((WIDTH - det_w) // 2, 455), detail, font=url_font, fill=TEXT_DIM)

    # URL at bottom
    url = "agenthealthmonitor.xyz"
    bbox_url = draw.textbbox((0, 0), url, font=url_font)
    uw = bbox_url[2] - bbox_url[0]
    draw.text(((WIDTH - uw) // 2, 550), url, font=url_font, fill=TEXT_DIM)

    # Scanlines
    draw_scanlines(draw, opacity=12)

    # -- Small top-left label --
    draw.text((55, 50), "// BLOCKCHAIN INTELLIGENCE", font=url_font, fill=TEXT_DIM)

    # -- Small top-right status --
    draw.text((WIDTH - 220, 50), "STATUS: OPERATIONAL", font=url_font, fill=GREEN)

    # Convert to RGB for PNG save
    img_rgb = img.convert("RGB")
    img_rgb.save(OUT, "PNG", optimize=True)
    print(f"[+] Saved: {OUT} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
