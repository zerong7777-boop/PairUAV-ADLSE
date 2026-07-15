"""Reports and route verdicts for A-v3.2a identity audit."""
import json
from pathlib import Path

from scripts.phase27_a_v3_2a_identity_common import ensure_dirs, read_csv_dicts, write_json


def _read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    return read_csv_dicts(p)


def compute_route_verdict(profile_metrics, pairwise_metrics, repair_metrics):
    pairwise = pairwise_metrics.get("pairwise_rows", [])
    repair = repair_metrics.get("repair_candidates", [])
    promotion_rows = [r for r in pairwise if r.get("key_role") in {"promotion_key", "promotion_key_candidate"}]
    clean = [r for r in promotion_rows if r.get("promotion_eligible") == "true"]
    candidate_full_clean = [
        r
        for r in clean
        if {r.get("left_artifact"), r.get("right_artifact")} == {"candidate", "full_dev"}
    ]
    stress_artifacts = {
        name
        for r in pairwise
        for name in (r.get("left_artifact"), r.get("right_artifact"))
        if name and name.startswith("stress")
    }
    candidate_stress_rows = [
        r
        for r in promotion_rows
        if "candidate" in {r.get("left_artifact"), r.get("right_artifact")}
        and (r.get("left_artifact") in stress_artifacts or r.get("right_artifact") in stress_artifacts)
    ]
    candidate_stress_clean_artifacts = {
        r.get("left_artifact") if r.get("left_artifact") in stress_artifacts else r.get("right_artifact")
        for r in candidate_stress_rows
        if r.get("promotion_eligible") == "true"
    }
    nonzero_promotion = [r for r in promotion_rows if int(r.get("intersection_count", 0)) > 0]
    duplicate_blocked = [r for r in promotion_rows if int(r.get("duplicate_blocked_count", 0)) > 0]
    candidate_stress_duplicate_blocked = [
        r for r in candidate_stress_rows if int(r.get("duplicate_blocked_count", 0)) > 0
    ]
    candidate_stress_nonzero = [r for r in candidate_stress_rows if int(r.get("intersection_count", 0)) > 0]
    diagnostic_overlap = [r for r in pairwise if r.get("key_role") in {"diagnostic_key", "forbidden_for_promotion"} and int(r.get("intersection_count", 0)) > 0]
    repair_classes = {r.get("repair_class") for r in repair}

    if candidate_full_clean and stress_artifacts and candidate_stress_clean_artifacts == stress_artifacts:
        return {"verdict": "identity-contract-pass", "reason": "clean_one_to_one_promotion_key_overlap_exists"}
    if candidate_full_clean and candidate_stress_duplicate_blocked:
        return {
            "verdict": "identity-contract-blocked-duplicates",
            "reason": "stress_surfaces_have_duplicate_promotion_identity_despite_candidate_full_overlap",
        }
    if candidate_full_clean and stress_artifacts and not candidate_stress_nonzero:
        return {
            "verdict": "identity-contract-reacquisition-required",
            "reason": "candidate_full_overlap_exists_but_candidate_stress_promotion_overlap_missing",
        }
    if candidate_full_clean and stress_artifacts:
        return {
            "verdict": "identity-contract-reacquisition-required",
            "reason": "candidate_full_overlap_exists_but_not_all_stress_surfaces_are_cleanly_joinable",
        }
    if clean:
        return {"verdict": "identity-contract-pass", "reason": "clean_one_to_one_promotion_key_overlap_exists"}
    if not nonzero_promotion and diagnostic_overlap:
        return {"verdict": "identity-contract-analysis-only", "reason": "overlap_exists_only_under_diagnostic_or_forbidden_keys"}
    if duplicate_blocked:
        return {"verdict": "identity-contract-blocked-duplicates", "reason": "promotion_keys_have_duplicate_blocked_joins"}
    if not nonzero_promotion:
        if "manifest_reacquisition_required" in repair_classes:
            return {"verdict": "identity-contract-reacquisition-required", "reason": "no_promotion_overlap_and_manifest_reacquisition_required"}
        return {"verdict": "identity-contract-blocked-zero-overlap", "reason": "no_allowed_promotion_key_has_overlap"}
    if "namespace_mapping_candidate" in repair_classes:
        return {"verdict": "identity-contract-blocked-namespace-mismatch", "reason": "namespace_mapping_required_before_promotion"}
    if "manifest_reacquisition_required" in repair_classes or "surface_regeneration_required" in repair_classes:
        return {"verdict": "identity-contract-reacquisition-required", "reason": "identity_audit_points_to_reacquisition"}
    return {"verdict": "identity-contract-analysis-only", "reason": "identity_state_inconclusive"}


def _write(path, lines):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_verdict_report(output_dir):
    out = ensure_dirs(output_dir)
    profile = _read_json(out / "metrics" / "identity_profile_metrics.json")
    pairwise = _read_json(out / "metrics" / "pairwise_join_matrix_metrics.json")
    repair = _read_json(out / "metrics" / "identity_repair_candidate_metrics.json")
    verdict = compute_route_verdict(profile, pairwise, repair)
    write_json(out / "metrics" / "a_v3_2a_route_verdict.json", verdict)
    _write(
        out / "reports" / "a_v3_2a_route_verdict.md",
        [
            "# A-v3.2a Route Verdict",
            "",
            f"verdict: `{verdict['verdict']}`",
            f"reason: `{verdict['reason']}`",
            "",
            "No training, finetuning, fuzzy-overlap construction, silent deduplication, sampler, or B/C gate was run.",
        ],
    )
    return verdict


def write_identity_join_audit_report(output_dir):
    out = ensure_dirs(output_dir)
    profile = _read_csv(out / "tables" / "artifact_identity_profile.csv")
    pairwise = _read_csv(out / "tables" / "pairwise_join_matrix.csv")
    lines = ["# A-v3.2a Identity Join Audit Report", "", f"- profile_rows: {len(profile)}", f"- pairwise_rows: {len(pairwise)}", ""]
    for row in pairwise[:20]:
        lines.append(f"- {row['left_artifact']} vs {row['right_artifact']} using {row['key_strategy']}: intersection={row['intersection_count']}, duplicate_blocked={row['duplicate_blocked_count']}, promotion={row['promotion_eligible']}")
    lines += ["", "Validation-only audit. No upstream artifacts were modified."]
    _write(out / "reports" / "identity_join_audit_report.md", lines)


def write_duplicate_resolution_report(output_dir):
    out = ensure_dirs(output_dir)
    dup = _read_csv(out / "tables" / "duplicate_blocked_pairs.csv")
    lines = ["# A-v3.2a Duplicate Resolution Report", "", f"- duplicate_rows: {len(dup)}", ""]
    for row in dup[:30]:
        lines.append(f"- {row['artifact_name']} {row['key_strategy']} {row['join_key']}: rows={row['row_count']} promotion_allowed={row['promotion_allowed']}")
    lines += ["", "Duplicates are blocked, not silently deduplicated."]
    _write(out / "reports" / "duplicate_resolution_report.md", lines)


def write_identity_repair_candidate_report(output_dir):
    out = ensure_dirs(output_dir)
    repair = _read_csv(out / "tables" / "identity_repair_candidates.csv")
    lines = ["# A-v3.2a Identity Repair Candidate Report", ""]
    for row in repair:
        lines.append(f"- {row['repair_class']}: {row['supporting_evidence']} next={row['next_action']} promotion={row['promotion_support']}")
    lines += ["", "Repair candidates are proposals only; no repair was applied."]
    _write(out / "reports" / "identity_repair_candidate_report.md", lines)


def write_all_reports(output_dir):
    write_identity_join_audit_report(output_dir)
    write_duplicate_resolution_report(output_dir)
    write_identity_repair_candidate_report(output_dir)
    return write_route_verdict_report(output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_all_reports(args.output_dir)


if __name__ == "__main__":
    main()
