#!/usr/bin/env python3
"""Generate a self-contained SVG mechanism summary figure.

This script intentionally uses only the Python standard library so the release
docs can be regenerated without installing plotting dependencies.
"""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#9AA0A6",
    "light": "#F6F8FA",
    "dark": "#333333",
}

official_scores = [
    ("MARB", 0.003188, 0.002528, 0.003849),
    ("MDHR", 0.002460, 0.002274, 0.002646),
    ("PAAER raw", 0.002514, 0.002392, 0.002636),
    ("RSF e230", 0.002413, 0.002191, 0.002636),
    ("PACE final", 0.001874, 0.001330, 0.002419),
]

axis_controls = [
    ("C0", 1.7925, 4.5583, COLORS["gray"]),
    ("H1", 1.5958, 53.6993, COLORS["sky"]),
    ("R1", 122.2311, 3.6355, COLORS["orange"]),
    ("HR", 1.4985, 3.9690, COLORS["green"]),
    ("H8 mid-late", 1.8194, 5.2569, COLORS["blue"]),
    ("E1b+E2R", 1.2363, 4.5209, COLORS["pink"]),
]

geometry_counts = {
    "heading": {"helpful": 608, "neutral/tie": 135, "harmful": 68},
    "distance": {"helpful": 10, "neutral/tie": 136, "harmful": 665},
}

representation_gaps = [
    ("H8 val", "heading_8bin", 0.609033),
    ("H8 val", "range_abs", 0.257820),
    ("H8 val", "range_signed", 0.589821),
    ("H8 train", "heading_8bin", 0.599402),
    ("H8 train", "range_abs", 0.310556),
    ("H8 train", "range_signed", 0.544177),
    ("Wbounded val", "heading_8bin", 0.348172),
    ("Wbounded val", "range_abs", 0.218947),
    ("Wbounded val", "range_signed", 0.595233),
]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def text(x: float, y: float, value: str, size: int = 14, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{COLORS["dark"]}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#D0D7DE", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 4) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="2"/>'


def write_data_csv() -> None:
    path = OUT_DIR / "mechanism_summary_data.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "metric", "value"])
        for name, final, distance, angle in official_scores:
            writer.writerow(["official_scores", name, "final_score", final])
            writer.writerow(["official_scores", name, "distance_rel_error", distance])
            writer.writerow(["official_scores", name, "angle_rel_error", angle])
        for name, angle, range_mae, _ in axis_controls:
            writer.writerow(["axis_controls", name, "angle_mae", angle])
            writer.writerow(["axis_controls", name, "range_mae", range_mae])
        for axis, counts in geometry_counts.items():
            for regime, count in counts.items():
                writer.writerow(["geometry_axis_counts", axis, regime, count])
        for split, label, gap in representation_gaps:
            writer.writerow(["representation_gaps", split, label, gap])


def panel_frame(x: float, y: float, w: float, h: float, title: str) -> list[str]:
    return [
        rect(x, y, w, h, "#FFFFFF", "#D0D7DE", rx=8),
        text(x + 18, y + 28, title, size=16, weight="700"),
    ]


def plot_official(x: float, y: float, w: float, h: float) -> list[str]:
    out = panel_frame(x, y, w, h, "A. Official-score progression")
    px, py = x + 52, y + 55
    pw, ph = w - 86, h - 100
    ymin, ymax = 0.0016, 0.00335
    out += [line(px, py + ph, px + pw, py + ph), line(px, py, px, py + ph)]
    for frac, label in [(0, "0.0016"), (0.5, "0.0025"), (1, "0.00335")]:
        gy = py + ph - frac * ph
        out.append(line(px, gy, px + pw, gy, color="#E5E7EB"))
        out.append(text(px - 8, gy + 4, label, size=10, anchor="end"))
    bar_w = pw / (len(official_scores) * 1.45)
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["vermillion"]]
    for i, (name, final, _, _) in enumerate(official_scores):
        cx = px + (i + 0.25) * (pw / len(official_scores))
        bh = (final - ymin) / (ymax - ymin) * ph
        out.append(rect(cx, py + ph - bh, bar_w, bh, colors[i], rx=2))
        out.append(text(cx + bar_w / 2, py + ph - bh - 6, f"{final:.4f}", size=10, anchor="middle"))
        out.append(text(cx + bar_w / 2, py + ph + 20, name, size=10, anchor="middle"))
    out.append(text(px + pw / 2, y + h - 12, "final_score (lower is better)", size=11, anchor="middle"))
    return out


def log_map(value: float, min_v: float, max_v: float, start: float, end: float) -> float:
    lv = math.log10(value)
    return start + (lv - math.log10(min_v)) / (math.log10(max_v) - math.log10(min_v)) * (end - start)


def plot_axis_controls(x: float, y: float, w: float, h: float) -> list[str]:
    out = panel_frame(x, y, w, h, "B. Axis-objective controls")
    px, py = x + 62, y + 55
    pw, ph = w - 100, h - 100
    out += [line(px, py + ph, px + pw, py + ph), line(px, py, px, py + ph)]
    for tick in [1, 10, 100]:
        tx = log_map(tick, 0.9, 160, px, px + pw)
        out.append(line(tx, py, tx, py + ph, color="#E5E7EB"))
        out.append(text(tx, py + ph + 17, str(tick), size=10, anchor="middle"))
    for tick in [3, 10, 30]:
        ty = log_map(tick, 2.5, 80, py + ph, py)
        out.append(line(px, ty, px + pw, ty, color="#E5E7EB"))
        out.append(text(px - 8, ty + 4, str(tick), size=10, anchor="end"))
    for name, angle, range_mae, color in axis_controls:
        cx = log_map(angle, 0.9, 160, px, px + pw)
        cy = log_map(range_mae, 2.5, 80, py + ph, py)
        out.append(circle(cx, cy, 6, color))
        out.append(text(cx + 8, cy - 6, name, size=10))
    out.append(text(px + pw / 2, y + h - 12, "angle MAE (log scale)", size=11, anchor="middle"))
    out.append(text(x + 14, py + ph / 2, "range MAE", size=11))
    return out


def plot_geometry_counts(x: float, y: float, w: float, h: float) -> list[str]:
    out = panel_frame(x, y, w, h, "C. Geometry utility is axis-asymmetric")
    px, py = x + 58, y + 55
    pw, ph = w - 90, h - 100
    regimes = ["helpful", "neutral/tie", "harmful"]
    heading_total = sum(geometry_counts["heading"].values())
    distance_total = sum(geometry_counts["distance"].values())
    out += [line(px, py + ph, px + pw, py + ph), line(px, py, px, py + ph)]
    for tick in [0, 25, 50, 75]:
        ty = py + ph - tick / 90 * ph
        out.append(line(px, ty, px + pw, ty, color="#E5E7EB"))
        out.append(text(px - 8, ty + 4, str(tick), size=10, anchor="end"))
    group_w = pw / len(regimes)
    for i, regime in enumerate(regimes):
        hp = 100 * geometry_counts["heading"][regime] / heading_total
        dp = 100 * geometry_counts["distance"][regime] / distance_total
        bx = px + i * group_w + 18
        bw = 24
        out.append(rect(bx, py + ph - hp / 90 * ph, bw, hp / 90 * ph, COLORS["blue"], rx=2))
        out.append(rect(bx + 30, py + ph - dp / 90 * ph, bw, dp / 90 * ph, COLORS["orange"], rx=2))
        out.append(text(bx + 27, py + ph + 18, regime, size=10, anchor="middle"))
    out.append(rect(px + pw - 108, py + 5, 10, 10, COLORS["blue"], rx=1))
    out.append(text(px + pw - 92, py + 15, "heading", size=10))
    out.append(rect(px + pw - 108, py + 22, 10, 10, COLORS["orange"], rx=1))
    out.append(text(px + pw - 92, py + 32, "distance", size=10))
    out.append(text(px + pw / 2, y + h - 12, "axis regime rows (%)", size=11, anchor="middle"))
    return out


def plot_representation(x: float, y: float, w: float, h: float) -> list[str]:
    out = panel_frame(x, y, w, h, "D. Representation true-vs-shuffle gap")
    px, py = x + 58, y + 55
    pw, ph = w - 90, h - 100
    groups = ["H8 val", "H8 train", "Wbounded val"]
    labels = ["heading_8bin", "range_abs", "range_signed"]
    colors = [COLORS["blue"], COLORS["green"], COLORS["pink"]]
    out += [line(px, py + ph, px + pw, py + ph), line(px, py, px, py + ph)]
    for tick in [0, 0.25, 0.5, 0.7]:
        ty = py + ph - tick / 0.72 * ph
        out.append(line(px, ty, px + pw, ty, color="#E5E7EB"))
        out.append(text(px - 8, ty + 4, f"{tick:.2f}", size=10, anchor="end"))
    group_w = pw / len(groups)
    for i, group in enumerate(groups):
        for j, label in enumerate(labels):
            value = next(v for g, l, v in representation_gaps if g == group and l == label)
            bx = px + i * group_w + 18 + j * 20
            bh = value / 0.72 * ph
            out.append(rect(bx, py + ph - bh, 16, bh, colors[j], rx=2))
        out.append(text(px + i * group_w + group_w / 2, py + ph + 18, group, size=10, anchor="middle"))
    lx = px + pw - 132
    for j, label in enumerate(labels):
        out.append(rect(lx, py + 5 + 17 * j, 10, 10, colors[j], rx=1))
        out.append(text(lx + 15, py + 15 + 17 * j, label, size=10))
    out.append(text(px + pw / 2, y + h - 12, "same-minus-diff gap", size=11, anchor="middle"))
    return out


def main() -> None:
    write_data_csv()
    width, height = 1200, 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["light"], rx=0),
        text(width / 2, 36, "PairUAV mechanism evidence behind PACE", size=24, weight="700", anchor="middle"),
    ]
    panel_w, panel_h = 555, 350
    parts += plot_official(35, 65, panel_w, panel_h)
    parts += plot_axis_controls(610, 65, panel_w, panel_h)
    parts += plot_geometry_counts(35, 440, panel_w, panel_h)
    parts += plot_representation(610, 440, panel_w, panel_h)
    parts.append("</svg>")
    (OUT_DIR / "fig_mechanism_summary.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
