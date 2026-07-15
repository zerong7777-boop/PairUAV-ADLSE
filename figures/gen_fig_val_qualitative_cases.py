#!/usr/bin/env python3
"""Build and render PairUAV validation qualitative cases.

The script is intentionally standard-library only. It can:
1. select representative validation cases from an aligned base-vs-geometry
   outcome surface;
2. write a small CSV manifest for reproducible case selection;
3. render an SVG case panel from the manifest and an optional PairUAV
   train_tour image root.

By default, rendered SVGs reference local image files instead of embedding
dataset images. Use --embed-images only for private/paper drafts when the
dataset license permits redistribution of the resulting SVG.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import mimetypes
from pathlib import Path
from typing import Callable


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = OUT_DIR / "val_qualitative_cases.csv"
DEFAULT_OUTPUT = OUT_DIR / "fig_val_qualitative_cases.svg"

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
    "white": "#FFFFFF",
}

MANIFEST_COLUMNS = [
    "case_rank",
    "case_group",
    "case_title",
    "case_rationale",
    "pair_id",
    "view_a_rel",
    "view_b_rel",
    "target_heading_deg",
    "target_distance",
    "base_pred_heading_deg",
    "base_pred_distance",
    "base_heading_abs_error",
    "base_distance_abs_error",
    "geometry_pred_heading_deg",
    "geometry_pred_distance",
    "geometry_heading_abs_error",
    "geometry_distance_abs_error",
    "delta_heading_abs_error",
    "delta_distance_abs_error",
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fnum(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def image_rels(pair_id: str) -> tuple[str, str]:
    scene, pair = pair_id.split("/")
    view_a, view_b = pair.split("_")
    return f"{scene}/image-{int(view_a):02d}.jpeg", f"{scene}/image-{int(view_b):02d}.jpeg"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_by_pair(path: Path | None, split: str | None = None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if split is not None and row.get("split") != split:
            continue
        pair = row.get("pair_id") or row.get("canonical_pair_id")
        if pair:
            out[pair] = row
    return out


def enrich_surface(
    surface_csv: Path,
    base_csv: Path | None,
    geometry_csv: Path | None,
) -> list[dict[str, str]]:
    base_by_pair = read_by_pair(base_csv)
    geom_by_pair = read_by_pair(geometry_csv, split="val")
    rows: list[dict[str, str]] = []
    for row in read_csv(surface_csv):
        if row.get("row_status") != "ok":
            continue
        pair_id = row.get("canonical_pair_id") or row.get("pair_id")
        if not pair_id:
            continue
        view_a_rel, view_b_rel = image_rels(pair_id)
        base = base_by_pair.get(pair_id, {})
        geom = geom_by_pair.get(pair_id, {})
        merged = {
            "pair_id": pair_id,
            "view_a_rel": view_a_rel,
            "view_b_rel": view_b_rel,
            "target_heading_deg": base.get("target_heading") or geom.get("target_heading_deg") or "",
            "target_distance": base.get("target_distance") or geom.get("target_distance") or "",
            "base_pred_heading_deg": base.get("pred_heading") or row.get("base_pred_heading_deg") or "",
            "base_pred_distance": base.get("pred_distance") or row.get("base_pred_distance") or "",
            "base_heading_abs_error": base.get("angle_abs_error") or row.get("base_heading_abs_error") or "",
            "base_distance_abs_error": base.get("distance_abs_error") or row.get("base_distance_abs_error") or "",
            "geometry_pred_heading_deg": geom.get("pred_heading_deg") or row.get("geometry_pred_heading_deg") or "",
            "geometry_pred_distance": geom.get("pred_distance") or row.get("geometry_pred_distance") or "",
            "geometry_heading_abs_error": geom.get("angle_abs_error") or row.get("geometry_heading_abs_error") or "",
            "geometry_distance_abs_error": geom.get("distance_abs_error") or row.get("geometry_distance_abs_error") or "",
            "delta_heading_abs_error": row.get("delta_heading_abs_error") or "",
            "delta_distance_abs_error": row.get("delta_distance_abs_error") or "",
        }
        rows.append(merged)
    return rows


def case_score_axis_conflict(row: dict[str, str]) -> float:
    h_gain = to_float(row, "delta_heading_abs_error")
    d_harm = -to_float(row, "delta_distance_abs_error")
    geom_d = to_float(row, "geometry_distance_abs_error")
    return h_gain + 0.15 * d_harm - 0.2 * geom_d


def case_score_joint(row: dict[str, str]) -> float:
    return to_float(row, "delta_heading_abs_error") + to_float(row, "delta_distance_abs_error")


def case_score_harmful(row: dict[str, str]) -> float:
    return -to_float(row, "delta_heading_abs_error") - to_float(row, "delta_distance_abs_error")


def same_view(row: dict[str, str]) -> bool:
    return row["view_a_rel"] == row["view_b_rel"]


def group_specs() -> list[dict[str, object]]:
    return [
        {
            "group": "axis_conflict",
            "title": "Heading rescued, range harmed",
            "rationale": "Geometry greatly reduces heading error, but worsens distance; motivates axis-decoupled readout.",
            "quota": 2,
            "filter": lambda r: (
                to_float(r, "delta_heading_abs_error") > 3.0
                and to_float(r, "delta_distance_abs_error") < -3.0
                and to_float(r, "base_distance_abs_error") < 5.0
                and to_float(r, "geometry_distance_abs_error") < 30.0
            ),
            "score": case_score_axis_conflict,
        },
        {
            "group": "joint_helpful",
            "title": "Geometry helps both axes",
            "rationale": "A positive control where the geometry branch improves both heading and distance.",
            "quota": 2,
            "filter": lambda r: (
                to_float(r, "delta_heading_abs_error") > 1.0
                and to_float(r, "delta_distance_abs_error") > 0.5
            ),
            "score": case_score_joint,
        },
        {
            "group": "same_view_hallucination",
            "title": "Same-view false motion",
            "rationale": "The two images are identical, yet the geometry branch predicts nonzero motion; motivates legal-state snapping/protection.",
            "quota": 1,
            "filter": lambda r: same_view(r) and to_float(r, "delta_distance_abs_error") < -5.0,
            "score": case_score_harmful,
        },
        {
            "group": "geometry_harmful",
            "title": "Geometry should be gated",
            "rationale": "Base is already accurate, while the geometry branch degrades the prediction.",
            "quota": 1,
            "filter": lambda r: (
                not same_view(r)
                and to_float(r, "delta_heading_abs_error") < -1.0
                and to_float(r, "delta_distance_abs_error") < -5.0
                and to_float(r, "base_distance_abs_error") < 5.0
            ),
            "score": case_score_harmful,
        },
    ]


def select_cases(rows: list[dict[str, str]], max_cases: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for spec in group_specs():
        filt = spec["filter"]
        score = spec["score"]
        assert callable(filt)
        assert callable(score)
        candidates = [r for r in rows if r["pair_id"] not in used and filt(r)]
        candidates.sort(key=score, reverse=True)
        for row in candidates[: int(spec["quota"])]:
            used.add(row["pair_id"])
            out = dict(row)
            out["case_group"] = str(spec["group"])
            out["case_title"] = str(spec["title"])
            out["case_rationale"] = str(spec["rationale"])
            selected.append(out)
            if len(selected) >= max_cases:
                return selected
    return selected


def write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            clean = {key: row.get(key, "") for key in MANIFEST_COLUMNS}
            clean["case_rank"] = str(idx)
            for key in MANIFEST_COLUMNS:
                if key.endswith("_error") or key in {
                    "target_heading_deg",
                    "target_distance",
                    "base_pred_heading_deg",
                    "base_pred_distance",
                    "geometry_pred_heading_deg",
                    "geometry_pred_distance",
                    "delta_heading_abs_error",
                    "delta_distance_abs_error",
                }:
                    clean[key] = fnum(clean[key], digits=6)
            writer.writerow(clean)


def svg_text(x: float, y: float, value: str, size: int = 15, weight: str = "400", color: str | None = None) -> str:
    fill = color or COLORS["dark"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 4) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_href(path: Path, embed_images: bool) -> str:
    if embed_images:
        return data_url(path)
    return path.resolve().as_uri()


def image_panel(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    rel_path: str,
    path: Path | None,
    embed_images: bool,
) -> list[str]:
    out = [svg_rect(x, y, w, h, "#EEF2F7", COLORS["border"], rx=6)]
    if path is not None and path.exists():
        href = image_href(path, embed_images)
        out.append(
            f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'href="{esc(href)}" preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        out.append(svg_text(x + 12, y + h / 2 - 10, "image not embedded", size=12, color=COLORS["muted"]))
        out.append(svg_text(x + 12, y + h / 2 + 12, rel_path, size=11, color=COLORS["muted"]))
    out.append(svg_rect(x, y, 64, 23, "rgba(36,41,47,0.74)", rx=5))
    out.append(svg_text(x + 9, y + 16, label, size=12, weight="700", color="#FFFFFF"))
    return out


def bar_pair(x: float, y: float, label: str, base_value: float, geom_value: float, scale: float) -> list[str]:
    width = 210
    base_w = min(width, max(2.0, base_value / scale * width))
    geom_w = min(width, max(2.0, geom_value / scale * width))
    out = [
        svg_text(x, y, label, size=13, weight="700"),
        svg_text(x + 72, y, "base", size=12, color=COLORS["muted"]),
        svg_rect(x + 113, y - 12, base_w, 11, COLORS["blue"], rx=2),
        svg_text(x + 113 + base_w + 6, y - 2, f"{base_value:.2f}", size=11, color=COLORS["muted"]),
        svg_text(x + 72, y + 18, "geom", size=12, color=COLORS["muted"]),
        svg_rect(x + 113, y + 6, geom_w, 11, COLORS["orange"], rx=2),
        svg_text(x + 113 + geom_w + 6, y + 16, f"{geom_value:.2f}", size=11, color=COLORS["muted"]),
    ]
    return out


def render_svg(manifest: Path, output: Path, dataset_root: Path | None, embed_images: bool) -> None:
    rows = read_csv(manifest)
    row_h = 252
    width = 1320
    height = 90 + row_h * len(rows) + 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, COLORS["light"], rx=0),
        svg_text(34, 42, "Validation qualitative cases: axis-asymmetric geometry effects", size=24, weight="700"),
        svg_text(
            34,
            66,
            "Selected from the B811 validation surface. Lower errors are better; delta is base error minus geometry error.",
            size=14,
            color=COLORS["muted"],
        ),
    ]
    for idx, row in enumerate(rows):
        y = 92 + idx * row_h
        parts.append(svg_rect(24, y - 16, width - 48, row_h - 18, COLORS["white"], COLORS["border"], rx=8))
        parts.append(svg_text(44, y + 8, f"{row['case_rank']}. {row['case_title']}", size=18, weight="700"))
        parts.append(svg_text(44, y + 30, row["case_rationale"], size=13, color=COLORS["muted"]))
        parts.append(svg_text(44, y + 54, f"pair: {row['pair_id']}", size=13, weight="700"))

        path_a = dataset_root / row["view_a_rel"] if dataset_root else None
        path_b = dataset_root / row["view_b_rel"] if dataset_root else None
        parts += image_panel(44, y + 68, 210, 138, "view A", row["view_a_rel"], path_a, embed_images)
        parts += image_panel(270, y + 68, 210, 138, "view B", row["view_b_rel"], path_b, embed_images)

        text_x = 510
        parts.append(svg_text(text_x, y + 68, f"GT heading {fnum(row['target_heading_deg'])} deg, distance {fnum(row['target_distance'])}", size=14))
        parts.append(svg_text(text_x, y + 94, f"Base pred: h={fnum(row['base_pred_heading_deg'])}, d={fnum(row['base_pred_distance'])}", size=13, color=COLORS["blue"]))
        parts.append(svg_text(text_x, y + 117, f"Geom pred: h={fnum(row['geometry_pred_heading_deg'])}, d={fnum(row['geometry_pred_distance'])}", size=13, color=COLORS["orange"]))
        parts += bar_pair(
            text_x,
            y + 154,
            "heading err",
            to_float(row, "base_heading_abs_error"),
            to_float(row, "geometry_heading_abs_error"),
            scale=120.0,
        )
        parts += bar_pair(
            text_x,
            y + 200,
            "range err",
            to_float(row, "base_distance_abs_error"),
            to_float(row, "geometry_distance_abs_error"),
            scale=80.0,
        )

        delta_h = to_float(row, "delta_heading_abs_error")
        delta_d = to_float(row, "delta_distance_abs_error")
        delta_color_h = COLORS["green"] if delta_h > 0 else COLORS["vermillion"]
        delta_color_d = COLORS["green"] if delta_d > 0 else COLORS["vermillion"]
        parts.append(svg_text(1060, y + 105, f"Delta heading: {delta_h:+.2f}", size=15, weight="700", color=delta_color_h))
        parts.append(svg_text(1060, y + 134, f"Delta range: {delta_d:+.2f}", size=15, weight="700", color=delta_color_d))
        parts.append(svg_text(1060, y + 164, row["case_group"], size=13, color=COLORS["muted"]))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-csv", type=Path, help="Aligned base-vs-geometry validation surface CSV.")
    parser.add_argument("--base-csv", type=Path, help="Base prediction CSV with targets.")
    parser.add_argument("--geometry-csv", type=Path, help="Geometry/split-fusion prediction CSV with targets.")
    parser.add_argument("--write-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest", type=Path, help="Existing qualitative case manifest to render.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, help="Path to PairUAV train_tour root.")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--embed-images", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.write_manifest
    if args.surface_csv:
        rows = enrich_surface(args.surface_csv, args.base_csv, args.geometry_csv)
        selected = select_cases(rows, max_cases=args.max_cases)
        write_manifest(selected, args.write_manifest)
        manifest = args.write_manifest
        print(f"wrote manifest: {args.write_manifest} ({len(selected)} cases)")
    if not args.skip_render:
        render_svg(manifest, args.output, args.dataset_root, args.embed_images)
        print(f"wrote figure: {args.output}")


if __name__ == "__main__":
    main()
