#!/usr/bin/env python3
"""Generate a lightweight SVG summary of source adaptation evidence.

The script intentionally uses only the Python standard library so the public
artifact can be regenerated without optional plotting dependencies.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "fig_source_adaptation_summary.svg"
CSV_PATH = ROOT / "source_adaptation_summary_data.csv"


SOURCE_POINTS = [
    {
        "name": "Split fusion",
        "family": "axis split",
        "angle_mae": 1.330926,
        "distance_mae": 4.024406,
        "color": "#009E73",
    },
    {
        "name": "Axis decoupled",
        "family": "axis split",
        "angle_mae": 1.7769,
        "distance_mae": 0.9876,
        "color": "#0072B2",
    },
    {
        "name": "MASt3R geom",
        "family": "geometry",
        "angle_mae": 9.5863,
        "distance_mae": 33.2327,
        "color": "#E69F00",
    },
    {
        "name": "DUSt3R geom",
        "family": "geometry",
        "angle_mae": 11.0434,
        "distance_mae": 33.1489,
        "color": "#D55E00",
    },
    {
        "name": "MASt3R+VGGT",
        "family": "hybrid",
        "angle_mae": 11.7800,
        "distance_mae": 7.8554,
        "color": "#CC79A7",
    },
    {
        "name": "RoMa dense",
        "family": "matching",
        "angle_mae": 15.3929,
        "distance_mae": 19.4702,
        "color": "#56B4E9",
    },
    {
        "name": "VGGT geom",
        "family": "geometry",
        "angle_mae": 52.5120,
        "distance_mae": 14.1716,
        "color": "#F0E442",
    },
    {
        "name": "EffLoFTR",
        "family": "matching",
        "angle_mae": 55.3077,
        "distance_mae": 39.8429,
        "color": "#999999",
    },
]

B811_COUNTS = [
    {"axis": "Heading", "better": 694, "worse": 117},
    {"axis": "Distance", "better": 53, "worse": 758},
]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def write_csv() -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "name", "family", "angle_mae", "distance_mae", "axis", "better", "worse"],
        )
        writer.writeheader()
        for row in SOURCE_POINTS:
            writer.writerow(
                {
                    "section": "source_scatter",
                    "name": row["name"],
                    "family": row["family"],
                    "angle_mae": row["angle_mae"],
                    "distance_mae": row["distance_mae"],
                    "axis": "",
                    "better": "",
                    "worse": "",
                }
            )
        for row in B811_COUNTS:
            writer.writerow(
                {
                    "section": "b811_axis_counts",
                    "name": "",
                    "family": "",
                    "angle_mae": "",
                    "distance_mae": "",
                    "axis": row["axis"],
                    "better": row["better"],
                    "worse": row["worse"],
                }
            )


def sx(angle: float) -> float:
    # Plot area: x 70..525, angle 0..60
    return 70 + (angle / 60.0) * 455


def sy(distance: float) -> float:
    # Plot area: y 390..88, distance 0..45, lower is lower on the chart.
    return 390 - (distance / 45.0) * 302


def svg_text(x: float, y: float, text: str, size: int = 13, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#263238">{esc(text)}</text>'
    )


def build_svg() -> str:
    width, height = 1180, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1180" height="520" fill="#ffffff"/>',
        svg_text(40, 42, "Source Adaptation Evidence: Axis-Conditional Utility", 22, "700"),
        svg_text(
            40,
            67,
            "Lower MAE is better. Diagnostic surfaces are internal and not hidden-test leaderboard results.",
            13,
        ),
    ]

    # Panel A background and axes.
    parts.extend(
        [
            '<rect x="36" y="88" width="545" height="350" rx="8" fill="#F8FAFC" stroke="#DDE4EA"/>',
            svg_text(58, 116, "A. Source utility scatter", 16, "700"),
            '<line x1="70" y1="390" x2="525" y2="390" stroke="#455A64" stroke-width="1.2"/>',
            '<line x1="70" y1="88" x2="70" y2="390" stroke="#455A64" stroke-width="1.2"/>',
            svg_text(278, 426, "Heading MAE", 13, "600", "middle"),
            '<text x="22" y="245" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="600" '
            'text-anchor="middle" fill="#263238" transform="rotate(-90 22 245)">Distance MAE</text>',
        ]
    )

    for tick in [0, 15, 30, 45, 60]:
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="390" x2="{x:.1f}" y2="395" stroke="#455A64"/>')
        parts.append(svg_text(x, 410, str(tick), 11, "400", "middle"))
        if tick:
            parts.append(f'<line x1="{x:.1f}" y1="88" x2="{x:.1f}" y2="390" stroke="#DCE3EA" stroke-width="0.8"/>')

    for tick in [0, 15, 30, 45]:
        y = sy(tick)
        parts.append(f'<line x1="65" y1="{y:.1f}" x2="70" y2="{y:.1f}" stroke="#455A64"/>')
        parts.append(svg_text(58, y + 4, str(tick), 11, "400", "end"))
        if tick:
            parts.append(f'<line x1="70" y1="{y:.1f}" x2="525" y2="{y:.1f}" stroke="#DCE3EA" stroke-width="0.8"/>')

    label_offsets = {
        "Split fusion": (8, -10),
        "Axis decoupled": (8, 17),
        "MASt3R geom": (8, -9),
        "DUSt3R geom": (8, 15),
        "MASt3R+VGGT": (8, -10),
        "RoMa dense": (8, -10),
        "VGGT geom": (-92, -10),
        "EffLoFTR": (-70, -10),
    }
    for row in SOURCE_POINTS:
        x, y = sx(row["angle_mae"]), sy(row["distance_mae"])
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{row["color"]}" stroke="#263238" stroke-width="0.8"/>'
        )
        dx, dy = label_offsets[row["name"]]
        parts.append(svg_text(x + dx, y + dy, row["name"], 11, "600"))

    # Panel B bars.
    parts.extend(
        [
            '<rect x="620" y="88" width="520" height="350" rx="8" fill="#FFFBF4" stroke="#E7DCC8"/>',
            svg_text(642, 116, "B. B811 geometry utility by axis", 16, "700"),
            svg_text(642, 141, "Direct lower-error pair counts, base vs geometry source.", 12),
        ]
    )

    bar_x, bar_y = 680, 190
    bar_w, bar_h = 390, 34
    total = 811
    for i, row in enumerate(B811_COUNTS):
        y = bar_y + i * 95
        better_w = bar_w * row["better"] / total
        worse_w = bar_w * row["worse"] / total
        parts.append(svg_text(642, y + 23, row["axis"], 14, "700"))
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{better_w:.1f}" height="{bar_h}" fill="#009E73"/>')
        parts.append(
            f'<rect x="{bar_x + better_w:.1f}" y="{y}" width="{worse_w:.1f}" height="{bar_h}" fill="#D55E00"/>'
        )
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#263238" stroke-width="0.8"/>')
        parts.append(svg_text(bar_x + 8, y + 23, f'better {row["better"]}', 12, "700"))
        parts.append(svg_text(bar_x + bar_w - 8, y + 23, f'worse {row["worse"]}', 12, "700", "end"))

    parts.extend(
        [
            '<rect x="680" y="390" width="13" height="13" fill="#009E73"/>',
            svg_text(700, 401, "geometry lower error", 12),
            '<rect x="850" y="390" width="13" height="13" fill="#D55E00"/>',
            svg_text(870, 401, "geometry higher error", 12),
            svg_text(
                642,
                428,
                "Takeaway: geometry is usually good for heading and unsafe for distance.",
                13,
                "700",
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    write_csv()
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {SVG_PATH}")
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
