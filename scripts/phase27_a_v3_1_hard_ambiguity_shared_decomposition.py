from pathlib import Path

from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, read_csv_dicts, safe_div, truthy, write_csv_dicts
from scripts.phase27_a_v3_1_shared_outcome_consistency import shared_rows


def hard_ambiguity_shared_specs():
    return [
        ("heading_hard_multi_modal", "baseline_heading_hard", "multi_modal_ambiguous", "multi-hypothesis candidate"),
        ("range_hard_multi_modal", "baseline_range_hard", "multi_modal_ambiguous", "multi-hypothesis candidate"),
        ("joint_hard_tail_unreliable", "baseline_joint_hard", "tail_error_unreliable", "quarantine / analysis-only"),
        ("heading_hard_semantic_conflict", "baseline_heading_hard", "semantic_geometric_conflict", "correspondence diagnostic candidate"),
        ("joint_hard_stress_ambiguous", "baseline_joint_hard", "stress_sensitive_ambiguous", "weak supervision candidate"),
    ]


def compute_hard_ambiguity_shared_decomposition(rows):
    srows = shared_rows(rows)
    total = len(srows)
    output = []
    for subtype, hard_field, ambiguity_field, interpretation in hard_ambiguity_shared_specs():
        hard_count = sum(1 for row in srows if truthy(row.get(hard_field)))
        count = sum(1 for row in srows if truthy(row.get(hard_field)) and truthy(row.get(ambiguity_field)))
        output.append(
            {
                "subtype": subtype,
                "hard_field": hard_field,
                "ambiguity_field": ambiguity_field,
                "shared_count": count,
                "ratio_of_shared_total": safe_div(count, total),
                "ratio_of_shared_hard": safe_div(count, hard_count),
                "interpretation": interpretation,
                "diagnostic_only": "true",
            }
        )
    return output


def _write_report(table, path):
    lines = ["# A-v3.1 Hard/Ambiguity Shared Decomposition", ""]
    for row in table:
        lines.append(f"- {row['subtype']}: {row['shared_count']} ({row['interpretation']})")
    lines.append("")
    lines.append("All subtypes are diagnostic-only.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hard_ambiguity_shared_decomposition(rows, output_dir):
    out = ensure_output_dirs(output_dir)
    table = compute_hard_ambiguity_shared_decomposition(rows)
    write_csv_dicts(
        out / "tables" / "a_v3_1_hard_ambiguity_shared_decomposition.csv",
        table,
        ["subtype", "hard_field", "ambiguity_field", "shared_count", "ratio_of_shared_total", "ratio_of_shared_hard", "interpretation", "diagnostic_only"],
    )
    _write_report(table, out / "reports" / "a_v3_1_hard_ambiguity_shared_decomposition.md")
    return table


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_hard_ambiguity_shared_decomposition(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
