import json
from pathlib import Path

from scripts.phase27_a_v3_2a_identity_common import read_csv_dicts
from scripts.phase27_a_v3_2a_report import _read_json


def _count(path):
    p = Path(path)
    if not p.exists():
        return 0
    return len(read_csv_dicts(p))


def write_experiment_record(output_dir, input_manifest=None):
    out = Path(output_dir)
    verdict = _read_json(out / "metrics" / "a_v3_2a_route_verdict.json")
    lines = [
        "# Phase27 A-v3.2a Identity Join Audit Experiment Record",
        "",
        "status: `bounded-full-eval-complete`",
        "",
        "## Commands",
        "- `bash scripts/run_phase27_a_v3_2a_identity_join_bounded.sh`",
        "",
        "## Input Artifacts",
        "```json",
        json.dumps(input_manifest or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Contract",
        f"- contract: `{out / 'contract' / 'canonical_pair_identity_contract.md'}`",
        "",
        "## Output Summary",
        f"- artifact_identity_profile_rows: {_count(out / 'tables' / 'artifact_identity_profile.csv')}",
        f"- pairwise_join_matrix_rows: {_count(out / 'tables' / 'pairwise_join_matrix.csv')}",
        f"- duplicate_blocked_rows: {_count(out / 'tables' / 'duplicate_blocked_pairs.csv')}",
        f"- missing_key_rows: {_count(out / 'tables' / 'missing_key_rows.csv')}",
        f"- disjoint_summary_rows: {_count(out / 'tables' / 'disjoint_universe_summary.csv')}",
        f"- repair_candidate_rows: {_count(out / 'tables' / 'identity_repair_candidates.csv')}",
        "",
        "## Route Verdict",
        f"- verdict: {verdict.get('verdict')}",
        f"- reason: {verdict.get('reason')}",
        "",
        "## Next Allowed Action",
        "- If identity contract passes or reacquisition is required, write a separate A-v3.2b shared surface reacquisition spec.",
        "- Do not train or create B/C gates from this audit.",
        "",
        "No training, finetuning, sample weighting, curriculum, oversampling, checkpoint selection, submission packaging, threshold tuning, fuzzy-overlap construction, silent deduplication, sampler, or B/C gate training was run.",
    ]
    path = out / "phase27_a_v3_2a_identity_join_audit_experiment_record.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    manifest = _read_json(args.input_manifest) if args.input_manifest else None
    write_experiment_record(args.output_dir, manifest)


if __name__ == "__main__":
    main()
