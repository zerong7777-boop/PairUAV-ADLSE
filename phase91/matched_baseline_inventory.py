from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .common import DEFAULT_RUN_ROOT, ensure_run_root, write_csv, write_json, write_text


REQUIRED_BASELINES = [
    {
        "baseline_id": "B0",
        "baseline_name": "H0/shared readout",
        "match_keywords": ("h0", "shared"),
    },
    {
        "baseline_id": "B1",
        "baseline_name": "ordinary two-head readout",
        "match_keywords": ("two_head", "two-head", "ordinary"),
    },
    {
        "baseline_id": "B2",
        "baseline_name": "fixed H3 or early split",
        "match_keywords": ("h3", "early"),
    },
    {
        "baseline_id": "B3",
        "baseline_name": "fixed H5 or dual-path split",
        "match_keywords": ("h5", "dual"),
    },
    {
        "baseline_id": "B4",
        "baseline_name": "fixed H8/mid-late or full-H8 readout",
        "match_keywords": ("h8", "mid_late", "mid-late"),
    },
]


def compatibility_label(run_name: str) -> str:
    normalized = run_name.lower()
    if not normalized:
        return "missing"
    if "phase91" in normalized and "matched" in normalized:
        return "rerun_matched"
    if "exact" in normalized and "phase91" in normalized:
        return "imported_exact"
    return "imported_not_comparable"


def _find_imported_row(baseline: dict[str, Any], imported_rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in imported_rows:
        haystack = " ".join(str(row.get(key, "")) for key in ("run_name", "metric_file", "metric_path")).lower()
        if any(keyword in haystack for keyword in baseline["match_keywords"]):
            return row
    return None


def build_baseline_inventory(imported_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in REQUIRED_BASELINES:
        imported = _find_imported_row(baseline, imported_rows)
        run_name = imported.get("run_name", "") if imported else ""
        compatibility = compatibility_label(run_name)
        row = {
            "baseline_id": baseline["baseline_id"],
            "baseline_name": baseline["baseline_name"],
            "status": "available_import" if imported else "missing",
            "compatibility": compatibility,
            "run_name": run_name,
            "metric_file": imported.get("metric_file", "") if imported else "",
            "metric_path": imported.get("metric_path", "") if imported else "",
            "angle_mae_deg": imported.get("angle_mae_deg", "") if imported else "",
            "distance_mae": imported.get("distance_mae", "") if imported else "",
            "angle_rel_error": imported.get("angle_rel_error", "") if imported else "",
            "distance_rel_error": imported.get("distance_rel_error", "") if imported else "",
            "phase91_claim_usable": compatibility in {"rerun_matched", "imported_exact"},
        }
        rows.append(row)
    return rows


def load_imported_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_matched_baseline_inventory(run_root: Path) -> dict[str, Any]:
    ensure_run_root(run_root)
    imported_path = run_root / "manifests" / "available_baselines.csv"
    imported_rows = load_imported_rows(imported_path)
    rows = build_baseline_inventory(imported_rows)
    router_ready = all(bool(row["phase91_claim_usable"]) for row in rows)

    fieldnames = [
        "baseline_id",
        "baseline_name",
        "status",
        "compatibility",
        "run_name",
        "metric_file",
        "metric_path",
        "angle_mae_deg",
        "distance_mae",
        "angle_rel_error",
        "distance_rel_error",
        "phase91_claim_usable",
    ]
    write_csv(run_root / "router_smokes" / "matched_baseline_inventory.csv", rows, fieldnames=fieldnames)
    write_csv(run_root / "router_smokes" / "matched_baseline_metrics.csv", rows, fieldnames=fieldnames)

    payload = {
        "phase": "phase91_task6_matched_baseline_inventory",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "imported_baseline_manifest": str(imported_path),
        "required_baseline_count": len(REQUIRED_BASELINES),
        "phase91_claim_usable_count": sum(1 for row in rows if row["phase91_claim_usable"]),
        "router_ready": router_ready,
        "blocker": None if router_ready else "matched H0/two-head/H3/H5/H8 baseline rows are incomplete; do not make Router-L/Q claims yet.",
        "rows": rows,
    }
    write_json(run_root / "router_smokes" / "matched_baseline_inventory.json", payload)

    md = [
        "# Phase91 Matched Baseline Compatibility",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- imported_baseline_manifest: `{imported_path}`",
        f"- required_baseline_count: {payload['required_baseline_count']}",
        f"- phase91_claim_usable_count: {payload['phase91_claim_usable_count']}",
        f"- router_ready: {payload['router_ready']}",
        "",
        "## Verdict",
        "",
        payload["blocker"] or "All required baselines are matched or exact imports.",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        md.append(
            f"- `{row['baseline_id']}` {row['baseline_name']}: "
            f"`{row['compatibility']}`; source `{row['run_name'] or 'missing'}`"
        )
    md.extend(
        [
            "",
            "Router-L/Q smoke can be implemented after this point, but Router-L/Q claims must not be made until missing matched baselines are rerun or exact-imported.",
        ]
    )
    write_text(run_root / "router_smokes" / "matched_baseline_compatibility.md", "\n".join(md))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase91 matched baseline inventory.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_matched_baseline_inventory(args.run_root.resolve())
    print(json.dumps({"router_ready": payload["router_ready"], "blocker": payload["blocker"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
