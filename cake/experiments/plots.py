"""Static SVG line charts (no dependencies) for experiment reports.

Follows a validated two-slot categorical palette (blue / orange, light surface),
2px lines, ~10% min-max bands, end markers with a surface ring, a legend plus
direct end labels, and hairline gridlines.  The data also ships as JSON/Markdown
(table view).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step          # the last tick covers the maximum
    n_ticks = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n_ticks)]


def line_band_svg(series: Sequence[Dict], path: str, title: str, subtitle: str = "", xlabel: str = "",
                  ylabel: str = "", reference: Optional[float] = None, reference_label: str = "",
                  xfmt=lambda x: f"{x:g}", yfmt=lambda y: f"{y:.2f}") -> str:
    W, H = 780, 420
    ml, mr, mt, mb = 64, 170, 78, 56
    pw, ph = W - ml - mr, H - mt - mb
    xs = [x for s in series for x in s["x"]]
    ys = [y for s in series for y in (s.get("lo") or s["y"]) + (s.get("hi") or s["y"]) + s["y"]]
    if reference is not None:
        ys.append(reference)
    x0, x1 = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.08 or 0.05
    ticks = _nice_ticks(ylo - pad, yhi + pad)
    y0, y1 = ticks[0], ticks[-1]

    def X(x):
        return ml + (x - x0) / ((x1 - x0) or 1) * pw

    def Y(y):
        return mt + ph - (y - y0) / ((y1 - y0) or 1) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="{ml}" y="26" font-size="16" font-weight="600" fill="{TEXT}">{escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="{ml}" y="45" font-size="12" fill="{TEXT2}">{escape(subtitle)}</text>')
    # legend (always present for >= 2 series)
    lx = ml
    for i, s in enumerate(series):
        c = s.get("color") or SERIES[i]
        out.append(f'<line x1="{lx}" y1="63" x2="{lx + 18}" y2="63" stroke="{c}" stroke-width="2" '
                   f'stroke-linecap="round"/>')
        out.append(f'<text x="{lx + 24}" y="67" font-size="12" fill="{TEXT2}">{escape(s["name"])}</text>')
        lx += 24 + 7.2 * len(s["name"]) + 22
    if reference is not None:
        out.append(f'<line x1="{lx}" y1="63" x2="{lx + 18}" y2="63" stroke="{MUTED}" stroke-width="1"/>')
        out.append(f'<text x="{lx + 24}" y="67" font-size="12" fill="{TEXT2}">{escape(reference_label)}</text>')
    # grid + y ticks
    for t in ticks:
        out.append(f'<line x1="{ml}" y1="{Y(t):.1f}" x2="{ml + pw}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ml - 8}" y="{Y(t) + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">'
                   f'{escape(yfmt(t))}</text>')
    # x ticks
    for t in _nice_ticks(x0, x1, 6):
        if x0 - 1e-9 <= t <= x1 + 1e-9:
            out.append(f'<text x="{X(t):.1f}" y="{mt + ph + 18}" font-size="11" fill="{MUTED}" '
                       f'text-anchor="middle">{escape(xfmt(t))}</text>')
    out.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{AXIS}" stroke-width="1"/>')
    if xlabel:
        out.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 14}" font-size="12" fill="{TEXT2}" '
                   f'text-anchor="middle">{escape(xlabel)}</text>')
    if ylabel:
        out.append(f'<text x="16" y="{mt + ph / 2:.1f}" font-size="12" fill="{TEXT2}" text-anchor="middle" '
                   f'transform="rotate(-90 16 {mt + ph / 2:.1f})">{escape(ylabel)}</text>')
    if reference is not None:
        out.append(f'<line x1="{ml}" y1="{Y(reference):.1f}" x2="{ml + pw}" y2="{Y(reference):.1f}" '
                   f'stroke="{MUTED}" stroke-width="1"/>')
    # bands, lines, end markers
    ends = []
    for i, s in enumerate(series):
        c = s.get("color") or SERIES[i]
        if s.get("lo") and s.get("hi"):
            pts = [(X(x), Y(y)) for x, y in zip(s["x"], s["hi"])] + \
                  [(X(x), Y(y)) for x, y in reversed(list(zip(s["x"], s["lo"])))]
            out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="{c}" '
                       f'fill-opacity="0.10" stroke="none"/>')
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for j, (x, y) in enumerate(zip(s["x"], s["y"])))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round"/>')
        ex, ey = X(s["x"][-1]), Y(s["y"][-1])
        out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        ends.append([ey, ex, s["name"], s["y"][-1]])
    # direct end labels with leader lines when they would collide
    ends.sort()
    placed = []
    for ey, ex, name, val in ends:
        ly = ey
        if placed and ly - placed[-1] < 16:
            ly = placed[-1] + 16
        placed.append(ly)
        tx = ml + pw + 14
        if abs(ly - ey) > 0.5:
            out.append(f'<line x1="{ex + 5:.1f}" y1="{ey:.1f}" x2="{tx - 3:.1f}" y2="{ly:.1f}" stroke="{MUTED}" '
                       f'stroke-width="1"/>')
        out.append(f'<text x="{tx}" y="{ly + 4:.1f}" font-size="12" fill="{TEXT}">{escape(name)} '
                   f'{escape(yfmt(val))}</text>')
    out.append("</svg>")
    svg = "\n".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return svg
