from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "phase62_teacher_signal_audit.py"
    spec = importlib.util.spec_from_file_location("phase62_teacher_signal_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_teacher_signal_audit_reports_oracle_and_feature_separation(tmp_path: Path) -> None:
    module = load_script()
    eval_csv = tmp_path / "eval.csv"
    teacher_csv = tmp_path / "teacher.csv"
    hard_manifest = tmp_path / "hard.jsonl"
    write_csv(
        eval_csv,
        [
            {"pair_id": "g/a", "group_id": "g", "rank1_angle_abs_error": 1.0, "rank1_distance_abs_error": 0.1},
            {"pair_id": "g/b", "group_id": "g", "rank1_angle_abs_error": 0.2, "rank1_distance_abs_error": 0.1},
            {"pair_id": "g/c", "group_id": "g", "rank1_angle_abs_error": 0.1, "rank1_distance_abs_error": 0.1},
        ],
    )
    write_csv(
        teacher_csv,
        [
            {
                "pair_id": "g/a",
                "teacher_angle_abs_error": 0.4,
                "abs_heading_delta_deg": 10.0,
                "source_reliable": "True",
            },
            {
                "pair_id": "g/b",
                "teacher_angle_abs_error": 0.5,
                "abs_heading_delta_deg": 2.0,
                "source_reliable": "False",
            },
            {
                "pair_id": "g/c",
                "teacher_angle_abs_error": 0.3,
                "abs_heading_delta_deg": 1.0,
                "source_reliable": "False",
            },
        ],
    )
    hard_manifest.write_text(json.dumps({"pair_id": "g/a"}) + "\n", encoding="utf-8")

    result = module.run_audit(eval_csv, hard_manifest, teacher_csv, tmp_path / "out")
    payload = result["payload"]

    assert payload["summaries"]["all"]["teacher_coverage_rows"] == 3
    assert payload["summaries"]["all"]["teacher_help_rows"] == 1
    assert payload["summaries"]["hard"]["teacher_help_rows"] == 1
    assert payload["summaries"]["all"]["oracle_selective_angle_mae_on_covered"] < payload["summaries"]["all"]["base_angle_mae_on_covered"]
    assert payload["feature_audit"][0]["feature"] == "abs_heading_delta_deg"
    assert Path(result["json_path"]).exists()
    assert Path(result["report_path"]).exists()
