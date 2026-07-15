from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import ensure_dirs, read_csv_dicts, safe_div, truthy, write_csv_dicts


def hard_ambiguity_specs():
    return [
        ("heading_hard_multi_modal", "evidence_sufficient_heading_hard", "multi_modal_ambiguous", "multi-hypothesis candidate"),
        ("heading_hard_semantic_conflict", "evidence_sufficient_heading_hard", "semantic_geometric_conflict", "correspondence diagnostic candidate"),
        ("heading_hard_stress_ambiguous", "evidence_sufficient_heading_hard", "stress_sensitive_ambiguous", "weak supervision candidate"),
        ("heading_hard_tail_unreliable", "evidence_sufficient_heading_hard", "tail_error_unreliable", "quarantine / analysis-only"),
        ("range_hard_multi_modal", "evidence_sufficient_range_hard", "multi_modal_ambiguous", "multi-hypothesis candidate"),
        ("range_hard_semantic_conflict", "evidence_sufficient_range_hard", "semantic_geometric_conflict", "range-specific supervision candidate"),
        ("range_hard_stress_ambiguous", "evidence_sufficient_range_hard", "stress_sensitive_ambiguous", "weak supervision candidate"),
        ("joint_hard_semantic_conflict", "evidence_sufficient_joint_hard", "semantic_geometric_conflict", "correspondence diagnostic candidate"),
        ("joint_hard_stress_ambiguous", "evidence_sufficient_joint_hard", "stress_sensitive_ambiguous", "weak supervision candidate"),
        ("joint_hard_tail_unreliable", "evidence_sufficient_joint_hard", "tail_error_unreliable", "quarantine / analysis-only"),
    ]


def compute_hard_ambiguity_decomposition(rows):
    total = len(rows)
    out = []
    for subtype, hard_field, ambiguity_field, interpretation in hard_ambiguity_specs():
        hard_count = sum(1 for row in rows if truthy(row.get(hard_field)))
        count = sum(
            1 for row in rows if truthy(row.get(hard_field)) and truthy(row.get(ambiguity_field))
        )
        out.append(
            {
                "subtype": subtype,
                "hard_field": hard_field,
                "ambiguity_field": ambiguity_field,
                "count": count,
                "ratio_of_total": safe_div(count, total),
                "ratio_of_hard": safe_div(count, hard_count),
                "interpretation": interpretation,
                "diagnostic_only": "true",
            }
        )
    return out


def _write_report(rows, path):
    lines = ["# A-v3 Hard/Ambiguity Overlap Decomposition", ""]
    for row in rows:
        lines.append(f"- {row['subtype']}: {row['count']} ({row['interpretation']})")
    lines += ["", "All rows are diagnostic-only. No training labels or sampler instructions are produced."]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hard_ambiguity_decomposition(rows, output_dir):
    out = ensure_dirs(output_dir)
    table = compute_hard_ambiguity_decomposition(rows)
    fields = [
        "subtype",
        "hard_field",
        "ambiguity_field",
        "count",
        "ratio_of_total",
        "ratio_of_hard",
        "interpretation",
        "diagnostic_only",
    ]
    write_csv_dicts(out / "tables" / "a_v3_hard_ambiguity_overlap_decomposition.csv", table, fields)
    _write_report(table, out / "reports" / "a_v3_hard_ambiguity_overlap_decomposition.md")
    return table


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_hard_ambiguity_decomposition(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
