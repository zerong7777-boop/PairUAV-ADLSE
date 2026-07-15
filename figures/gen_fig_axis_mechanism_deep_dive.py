#!/usr/bin/env python3
"""Generate paper-facing mechanism deep-dive figures.

The script uses only the Python standard library. It reads the checked-in
B811 validation surface and summary CSVs, then writes four SVG figures plus
compact CSV summaries:

1. axis-conflict quadrant scatter;
2. regime-conditioned heading/range gain heatmaps;
3. representation probe true-vs-shuffle gaps;
4. range-tail failure quantile plot.
"""
from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
SURFACE_CSV = OUT_DIR / "b811_base_geometry_surface.csv"
ANCHOR_CSV = OUT_DIR / "b811_anchor_val_predictions.csv"
MECHANISM_CSV = OUT_DIR / "mechanism_summary_data.csv"

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#8C8C8C",
    "light": "#F6F8FA",
    "dark": "#24292F",
    "muted": "#57606A",
    "border": "#D0D7DE",
    "grid": "#E5E7EB",
    "white": "#FFFFFF",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def text(
    x: float,
    y: float,
    value: str,
    size: int = 12,
    weight: str = "400",
    anchor: str = "start",
    color: str | None = None,
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color or COLORS["dark"]}" '
        f'text-anchor="{anchor}"{transform}>{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 4) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#D0D7DE", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def circle(x: float, y: float, r: float, fill: str, opacity: float = 1.0, stroke: str = "none") -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="0.5"/>'
    )


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{value}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{v:02X}" for v in rgb)


def mix(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex((round(lerp(r1, r2, t)), round(lerp(g1, g2, t)), round(lerp(b1, b2, t))))


def diverging(value: float, limit: float) -> str:
    if limit <= 0:
        return COLORS["white"]
    v = clamp(value / limit, -1.0, 1.0)
    if v >= 0:
        return mix("#F7FBFF", COLORS["blue"], v)
    return mix("#FFF7EC", COLORS["vermillion"], -v)


def panel_bg(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["light"], rx=0),
        text(34, 38, title, size=22, weight="700"),
        text(34, 62, subtitle, size=13, color=COLORS["muted"]),
    ]


def load_surface_rows() -> list[dict[str, str]]:
    return [row for row in read_csv(SURFACE_CSV) if row.get("row_status") == "ok"]


def load_anchor_by_pair() -> dict[str, dict[str, str]]:
    return {row["pair_id"]: row for row in read_csv(ANCHOR_CSV)}


def generate_axis_conflict_quadrant() -> None:
    rows = load_surface_rows()
    out_csv = OUT_DIR / "axis_conflict_quadrant_summary.csv"
    out_svg = OUT_DIR / "fig_axis_conflict_quadrant.svg"

    counts = {
        "both_helpful": 0,
        "heading_helpful_range_harmful": 0,
        "heading_harmful_range_helpful": 0,
        "both_harmful": 0,
    }
    points: list[dict[str, object]] = []
    for row in rows:
        x = f(row, "delta_heading_abs_error")
        y = f(row, "delta_distance_abs_error")
        if x >= 0 and y >= 0:
            regime = "both_helpful"
        elif x >= 0 and y < 0:
            regime = "heading_helpful_range_harmful"
        elif x < 0 and y >= 0:
            regime = "heading_harmful_range_helpful"
        else:
            regime = "both_harmful"
        counts[regime] += 1
        points.append({"pair_id": row["canonical_pair_id"], "heading_gain": x, "range_gain": y, "regime": regime})

    summary_rows = [
        {"regime": name, "count": count, "fraction": f"{count / len(points):.6f}"} for name, count in counts.items()
    ]
    write_csv(out_csv, summary_rows, ["regime", "count", "fraction"])

    width, height = 900, 640
    px, py, pw, ph = 92, 92, 570, 430
    x_min, x_max = -8.0, 45.0
    y_min, y_max = -80.0, 80.0

    def sx(value: float) -> float:
        return px + (clamp(value, x_min, x_max) - x_min) / (x_max - x_min) * pw

    def sy(value: float) -> float:
        return py + ph - (clamp(value, y_min, y_max) - y_min) / (y_max - y_min) * ph

    colors = {
        "both_helpful": COLORS["green"],
        "heading_helpful_range_harmful": COLORS["blue"],
        "heading_harmful_range_helpful": COLORS["orange"],
        "both_harmful": COLORS["vermillion"],
    }
    parts = panel_bg(
        width,
        height,
        "Axis-conflict quadrant on B811 validation surface",
        "Gain = base error minus geometry error. Positive means the geometry branch improves that axis.",
    )
    parts += [rect(px - 10, py - 10, pw + 20, ph + 20, COLORS["white"], COLORS["border"], rx=8)]
    for tick in [-80, -40, 0, 40, 80]:
        ty = sy(tick)
        parts.append(line(px, ty, px + pw, ty, COLORS["grid"]))
        parts.append(text(px - 10, ty + 4, str(tick), size=10, anchor="end", color=COLORS["muted"]))
    for tick in [-5, 0, 10, 20, 30, 40]:
        tx = sx(tick)
        parts.append(line(tx, py, tx, py + ph, COLORS["grid"]))
        parts.append(text(tx, py + ph + 19, str(tick), size=10, anchor="middle", color=COLORS["muted"]))
    parts += [
        line(sx(0), py, sx(0), py + ph, COLORS["dark"], 1.4),
        line(px, sy(0), px + pw, sy(0), COLORS["dark"], 1.4),
        text(px + pw / 2, height - 54, "heading gain: base angle error - geometry angle error", size=12, anchor="middle"),
        text(24, py + ph / 2, "range gain", size=12, rotate=-90, anchor="middle"),
    ]
    for point in points:
        x = float(point["heading_gain"])
        y = float(point["range_gain"])
        clipped = x < x_min or x > x_max or y < y_min or y > y_max
        parts.append(circle(sx(x), sy(y), 3.0 if not clipped else 4.5, colors[str(point["regime"])], 0.46, "#FFFFFF"))
    parts.append(text(px + pw - 4, py + ph + 38, "x-axis clipped at 45 deg gain for readability", size=10, anchor="end", color=COLORS["muted"]))

    lx, ly = 690, 110
    parts += [text(lx, ly, "Quadrant counts", size=15, weight="700")]
    legend_order = [
        ("heading_helpful_range_harmful", "Heading helps, range harms"),
        ("both_helpful", "Both axes help"),
        ("both_harmful", "Both axes harm"),
        ("heading_harmful_range_helpful", "Range helps, heading harms"),
    ]
    for i, (key, label) in enumerate(legend_order):
        y = ly + 34 + i * 62
        count = counts[key]
        frac = count / len(points)
        parts += [
            rect(lx, y - 16, 15, 15, colors[key], rx=2),
            text(lx + 24, y - 3, label, size=12, weight="700"),
            text(lx + 24, y + 18, f"{count}/811 = {frac:.1%}", size=12, color=COLORS["muted"]),
        ]
    parts += [
        text(lx, 430, "Reading", size=15, weight="700"),
        text(lx, 456, "The dominant regime is not", size=12, color=COLORS["muted"]),
        text(lx, 476, "joint improvement. Geometry", size=12, color=COLORS["muted"]),
        text(lx, 496, "usually rescues heading while", size=12, color=COLORS["muted"]),
        text(lx, 516, "hurting range, motivating", size=12, color=COLORS["muted"]),
        text(lx, 536, "axis-asymmetric composition.", size=12, color=COLORS["muted"]),
    ]
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def heading_bin(value: float) -> str:
    edges = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi or (hi == 180 and value <= 180):
            return f"{lo}..{hi}"
    return "out"


def range_abs_bucket(value: float) -> str:
    abs_v = abs(value)
    edges = [0, 25, 50, 75, 100, 125, 150, 10**9]
    labels = ["0..25", "25..50", "50..75", "75..100", "100..125", "125..150", "150+"]
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        if lo <= abs_v < hi:
            return label
    return "150+"


def generate_regime_conditioned_axis_gain() -> None:
    surface = load_surface_rows()
    anchor_by_pair = load_anchor_by_pair()
    out_csv = OUT_DIR / "regime_conditioned_axis_gain.csv"
    out_svg = OUT_DIR / "fig_regime_conditioned_axis_gain.svg"

    h_labels = ["-180..-135", "-135..-90", "-90..-45", "-45..0", "0..45", "45..90", "90..135", "135..180"]
    r_labels = ["0..25", "25..50", "50..75", "75..100", "100..125", "125..150", "150+"]
    cells: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in surface:
        anchor = anchor_by_pair.get(row["canonical_pair_id"])
        if not anchor:
            continue
        hb = heading_bin(f(anchor, "target_heading"))
        rb = range_abs_bucket(f(anchor, "target_distance"))
        cells[(hb, rb)].append((f(row, "delta_heading_abs_error"), f(row, "delta_distance_abs_error")))

    table_rows: list[dict[str, object]] = []
    for rb in r_labels:
        for hb in h_labels:
            vals = cells.get((hb, rb), [])
            if vals:
                h_mean = sum(v[0] for v in vals) / len(vals)
                r_mean = sum(v[1] for v in vals) / len(vals)
            else:
                h_mean = r_mean = 0.0
            table_rows.append(
                {
                    "range_abs_bucket": rb,
                    "heading_bin": hb,
                    "count": len(vals),
                    "mean_heading_gain": f"{h_mean:.6f}",
                    "mean_range_gain": f"{r_mean:.6f}",
                }
            )
    write_csv(out_csv, table_rows, ["range_abs_bucket", "heading_bin", "count", "mean_heading_gain", "mean_range_gain"])

    width, height = 1180, 610
    parts = panel_bg(
        width,
        height,
        "Regime-conditioned axis gain on B811",
        "Cells show mean gain from replacing the base branch with geometry. Positive improves; negative harms.",
    )
    cell_w, cell_h = 58, 44
    top = 128
    lefts = [96, 650]
    titles = [("mean_heading_gain", "Heading gain"), ("mean_range_gain", "Range gain")]
    limits = {"mean_heading_gain": 30.0, "mean_range_gain": 30.0}
    row_by_key = {(r["range_abs_bucket"], r["heading_bin"]): r for r in table_rows}
    for panel_idx, (metric, title_value) in enumerate(titles):
        left = lefts[panel_idx]
        parts.append(text(left, 102, title_value, size=16, weight="700"))
        for j, hb in enumerate(h_labels):
            parts.append(text(left + j * cell_w + cell_w / 2, top - 10, hb.replace("..", " to "), size=8, anchor="middle", color=COLORS["muted"], rotate=-28))
        for i, rb in enumerate(r_labels):
            y = top + i * cell_h
            parts.append(text(left - 10, y + cell_h / 2 + 4, rb, size=10, anchor="end", color=COLORS["muted"]))
            for j, hb in enumerate(h_labels):
                x = left + j * cell_w
                row = row_by_key[(rb, hb)]
                count = int(row["count"])
                value = float(row[metric])
                fill = "#F0F0F0" if count == 0 else diverging(value, limits[metric])
                parts.append(rect(x, y, cell_w - 2, cell_h - 2, fill, COLORS["white"], rx=2))
                label = "n=0" if count == 0 else f"{value:+.1f}"
                parts.append(text(x + cell_w / 2, y + 20, label, size=9, anchor="middle", color=COLORS["dark"]))
                if count:
                    parts.append(text(x + cell_w / 2, y + 34, f"n={count}", size=7, anchor="middle", color=COLORS["muted"]))
        parts.append(text(left + 4 * cell_w, top + len(r_labels) * cell_h + 38, "true heading bin", size=11, anchor="middle"))
        parts.append(text(left - 62, top + len(r_labels) * cell_h / 2, "|true range|", size=11, rotate=-90, anchor="middle"))
    lx, ly = 480, 500
    for k, value in enumerate([-30, -15, 0, 15, 30]):
        parts.append(rect(lx + k * 38, ly, 36, 14, diverging(value, 30), COLORS["white"], rx=1))
        parts.append(text(lx + k * 38 + 18, ly + 30, f"{value:+d}", size=8, anchor="middle", color=COLORS["muted"]))
    parts.append(text(lx + 95, ly - 8, "gain color scale", size=10, anchor="middle", color=COLORS["muted"]))
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def load_representation_gaps() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(MECHANISM_CSV):
        if row.get("section") == "representation_gaps":
            rows.append({"model_split": row["name"], "probe": row["metric"], "gap": f(row, "value")})
    return rows


def generate_representation_probe_summary() -> None:
    rows = load_representation_gaps()
    out_csv = OUT_DIR / "representation_probe_summary.csv"
    out_svg = OUT_DIR / "fig_representation_probe_summary.svg"
    write_csv(out_csv, rows, ["model_split", "probe", "gap"])

    groups = ["H8 val", "H8 train", "Wbounded val"]
    probes = ["heading_8bin", "range_abs", "range_signed"]
    colors = {"heading_8bin": COLORS["blue"], "range_abs": COLORS["green"], "range_signed": COLORS["pink"]}
    values = {(str(r["model_split"]), str(r["probe"])): float(r["gap"]) for r in rows}
    width, height = 820, 520
    px, py, pw, ph = 82, 98, 620, 310
    parts = panel_bg(
        width,
        height,
        "Representation probes: pose-regime readability",
        "True-vs-shuffle gaps from Phase95 probes. Larger gaps mean stronger readable regime structure.",
    )
    parts += [rect(px - 12, py - 12, pw + 24, ph + 24, COLORS["white"], COLORS["border"], rx=8)]
    for tick in [0, 0.2, 0.4, 0.6]:
        y = py + ph - tick / 0.7 * ph
        parts.append(line(px, y, px + pw, y, COLORS["grid"]))
        parts.append(text(px - 8, y + 4, f"{tick:.1f}", size=10, anchor="end", color=COLORS["muted"]))
    group_w = pw / len(groups)
    bar_w = 28
    for i, group in enumerate(groups):
        center = px + i * group_w + group_w / 2
        for j, probe in enumerate(probes):
            value = values[(group, probe)]
            x = center - 48 + j * 34
            h = value / 0.7 * ph
            parts.append(rect(x, py + ph - h, bar_w, h, colors[probe], rx=2))
            parts.append(text(x + bar_w / 2, py + ph - h - 6, f"{value:.2f}", size=9, anchor="middle", color=COLORS["muted"]))
        parts.append(text(center, py + ph + 25, group, size=11, anchor="middle"))
    parts.append(text(px + pw / 2, height - 46, "probe split", size=12, anchor="middle"))
    parts.append(text(26, py + ph / 2, "same-minus-different gap", size=12, rotate=-90, anchor="middle"))
    lx, ly = 608, 116
    for j, probe in enumerate(probes):
        parts.append(rect(lx, ly + 23 * j, 12, 12, colors[probe], rx=1))
        parts.append(text(lx + 18, ly + 11 + 23 * j, probe, size=11))
    parts.append(text(94, 464, "Reading: H8 preserves strong heading and signed-range regime readability; range_abs is weaker but above shuffle.", size=12, color=COLORS["muted"]))
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def generate_tail_failure_quantiles() -> None:
    data = [
        {"run": "HR50", "median": 0.4546, "p95": 7.1223, "p99": 42.9710, "max": 91.2856},
        {"run": "H8 step50k", "median": 0.3897, "p95": 1.1979, "p99": 2.1321, "max": 6.9968},
        {"run": "Tail10", "median": 0.3600, "p95": 1.0446, "p99": 1.2653, "max": 4.3494},
    ]
    out_csv = OUT_DIR / "range_tail_failure_quantiles.csv"
    out_svg = OUT_DIR / "fig_range_tail_failure_quantiles.svg"
    write_csv(out_csv, data, ["run", "median", "p95", "p99", "max"])

    width, height = 840, 520
    px, py, pw, ph = 90, 98, 610, 310
    quantiles = ["median", "p95", "p99", "max"]
    x_positions = [px + i * (pw / (len(quantiles) - 1)) for i in range(len(quantiles))]
    y_min, y_max = 0.3, 120.0

    def sy(value: float) -> float:
        return py + ph - (math.log10(value) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)) * ph

    run_colors = {"HR50": COLORS["vermillion"], "H8 step50k": COLORS["blue"], "Tail10": COLORS["green"]}
    parts = panel_bg(
        width,
        height,
        "Range-tail failure is a separate mechanism",
        "Range absolute-error quantiles on the local proxy surface. Log scale reveals tail risk.",
    )
    parts += [rect(px - 12, py - 12, pw + 24, ph + 24, COLORS["white"], COLORS["border"], rx=8)]
    for tick in [0.3, 1, 3, 10, 30, 100]:
        y = sy(tick)
        parts.append(line(px, y, px + pw, y, COLORS["grid"]))
        parts.append(text(px - 8, y + 4, str(tick), size=10, anchor="end", color=COLORS["muted"]))
    for i, q in enumerate(quantiles):
        x = x_positions[i]
        parts.append(line(x, py, x, py + ph, COLORS["grid"]))
        parts.append(text(x, py + ph + 24, q, size=11, anchor="middle"))
    for run in data:
        name = str(run["run"])
        pts = [(x_positions[i], sy(float(run[q]))) for i, q in enumerate(quantiles)]
        parts.append(polyline(pts, run_colors[name], 2.4))
        for i, q in enumerate(quantiles):
            value = float(run[q])
            x, y = pts[i]
            parts.append(circle(x, y, 5, run_colors[name], 0.95, "#FFFFFF"))
            if q in {"p99", "max"}:
                parts.append(text(x + 8, y - 6, f"{value:.1f}", size=9, color=COLORS["muted"]))
    lx, ly = 596, 122
    for i, run in enumerate(data):
        name = str(run["run"])
        parts.append(line(lx, ly + i * 26, lx + 18, ly + i * 26, run_colors[name], 3))
        parts.append(text(lx + 26, ly + 4 + i * 26, name, size=11))
    parts.append(text(px + pw / 2, height - 46, "range-error quantile", size=12, anchor="middle"))
    parts.append(text(26, py + ph / 2, "absolute range error (log)", size=12, rotate=-90, anchor="middle"))
    parts.append(text(94, 464, "Reading: HR50 has a competitive median but a much worse p99/max tail; Tail10 specifically suppresses that tail.", size=12, color=COLORS["muted"]))
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    generate_axis_conflict_quadrant()
    generate_regime_conditioned_axis_gain()
    generate_representation_probe_summary()
    generate_tail_failure_quantiles()
    print("wrote fig_axis_conflict_quadrant.svg")
    print("wrote fig_regime_conditioned_axis_gain.svg")
    print("wrote fig_representation_probe_summary.svg")
    print("wrote fig_range_tail_failure_quantiles.svg")


if __name__ == "__main__":
    main()
