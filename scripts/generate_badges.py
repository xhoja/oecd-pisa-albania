#!/usr/bin/env python3
"""Generate static shields.io-style SVG badges committed to the repo.

Live img.shields.io badges are fetched through GitHub's Camo image proxy and
intermittently fail to load (broken-image icon) when many are requested at once.
Committing static SVGs makes the README render reliably and offline.

Source of truth: the ``BADGES`` list below. Re-run to regenerate:

    python scripts/generate_badges.py

Output: assets/badges/<slug>.svg  (referenced by README.md)
"""
from __future__ import annotations

import re
from pathlib import Path

# (label, message, message-hex-color)
BADGES = [
    ("Python", "3.10+", "3776AB"),
    ("pandas", "latest", "150458"),
    ("NumPy", "latest", "013243"),
    ("SciPy", "latest", "8CAAE6"),
    ("scikit-learn", "latest", "F7931E"),
    ("statsmodels", "latest", "3B5998"),
    ("CatBoost", "headline", "FFCC00"),
    ("LightGBM", "boosters", "9ACD32"),
    ("XGBoost", "boosters", "337AB7"),
    ("Optuna", "nested-CV HPO", "8A2BE2"),
    ("SHAP", "explainability", "FF6F61"),
    ("Altair", "charts", "1F77B4"),
    ("matplotlib", "figures", "11557C"),
    ("Streamlit", "dashboard", "FF4B4B"),
    ("LaTeX", "article report", "008080"),
    ("pytest", "189 passing", "0A9EDC"),
]

LABEL_BG = "555"
HEIGHT = 20
FONT_SIZE = 11
PAD = 6  # horizontal padding each side of a text segment

# Per-character advance widths (px) for 11px Verdana, approximating shields.io.
_NARROW = set("iIltfj:.,;'|!")
_WIDE = set("mwMW@")


def text_width(s: str) -> int:
    w = 0.0
    for c in s:
        if c in _NARROW:
            w += 3.6
        elif c in _WIDE:
            w += 9.5
        elif c.isupper():
            w += 7.4
        else:
            w += 6.4
    return round(w)


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def svg(label: str, message: str, color: str) -> str:
    lw = text_width(label) + PAD * 2
    mw = text_width(message) + PAD * 2
    total = lw + mw
    fg_msg = "000" if luminance(color) > 0.6 else "fff"
    # shadow offset for the subtle emboss shields.io uses
    label_shadow = "010101"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="{HEIGHT}" role="img" aria-label="{label}: {message}">
  <title>{label}: {message}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{total}" height="{HEIGHT}" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="{HEIGHT}" fill="#{LABEL_BG}"/>
    <rect x="{lw}" width="{mw}" height="{HEIGHT}" fill="#{color}"/>
    <rect width="{total}" height="{HEIGHT}" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="{FONT_SIZE}">
    <text x="{lw/2}" y="15" fill="#{label_shadow}" fill-opacity=".3">{label}</text>
    <text x="{lw/2}" y="14">{label}</text>
    <text x="{lw + mw/2}" y="15" fill="#{label_shadow}" fill-opacity=".3">{message}</text>
    <text x="{lw + mw/2}" y="14" fill="#{fg_msg}">{message}</text>
  </g>
</svg>
'''


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "assets" / "badges"
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, message, color in BADGES:
        path = out_dir / f"{slugify(label)}.svg"
        path.write_text(svg(label, message, color), encoding="utf-8")
        print(f"wrote {path.relative_to(out_dir.parent.parent)}")


if __name__ == "__main__":
    main()
