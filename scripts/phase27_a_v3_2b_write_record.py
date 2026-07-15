"""Write remote experiment record for Phase27 A-v3.2b."""
import argparse
import json
from pathlib import Path

from scripts.phase27_a_v3_2b_common import read_csv_dicts


def _read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_record(output_dir, input_manifest=None):
    out = Path(output_dir)
    manifest_metrics = _read_json(out / "manifests" / "fixed_shared_pair_manifest_bounded.metrics.json")
    runner = _read_json(out / "metrics" / "runner_capability.json")
    identity = _read_json(out / "metrics" / "shared_surface_identity_metrics.json")
    coverage = _read_json(out / "metrics" / "shared_surface_coverage_bias_metrics.json")
    verdict = _read_json(out / "metrics" / "a_v3_2b_route_verdict.json")
    outcome = _read_json(out / "metrics" / "outcome_reexport_metrics.json")
    inputs = _read_json(input_manifest) if input_manifest else {}
    lines = [
        "# Phase27 A-v3.2b Fixed-Manifest Shared Surface Experiment Record",
        "",
        "status: `bounded-complete`",
        "",
        "## Commands",
        "- `bash scripts/run_phase27_a_v3_2b_bounded.sh`",
        "",
        "## Input Artifacts",
        "```json",
        json.dumps(inputs, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Fixed Manifest",
        f"- row_count: {manifest_metrics.get('row_count')}",
        f"- unique_canonical_pair_id_count: {manifest_metrics.get('unique_canonical_pair_id_count')}",
        f"- manifest_checksum: `{manifest_metrics.get('manifest_checksum')}`",
        "",
        "## Runner Capability",
        f"- decision: `{runner.get('decision')}`",
        f"- capability_status: `{runner.get('capability_status')}`",
        "",
        "## Outcome And Shared Surface",
        f"- baseline_joined_count: {outcome.get('baseline_joined_count')}",
        f"- baseline_missing_count: {outcome.get('baseline_missing_count')}",
        f"- stress_duplicate_blocked_count: {identity.get('stress_duplicate_blocked_count')}",
        f"- stress_source_target_composite_missing_count: {identity.get('stress_source_target_composite_missing_count')}",
        f"- stress_missing_id_count: {identity.get('stress_missing_id_count')}",
        f"- shared_ready_count: {identity.get('shared_ready_count')}",
        f"- shared_coverage_ratio: {coverage.get('shared_coverage_ratio')}",
        "",
        "## Route Verdict",
        f"- verdict: `{verdict.get('verdict')}`",
        f"- reason: `{verdict.get('reason')}`",
        "",
        "## Boundary",
        "No training, finetuning, sample weighting, curriculum, oversampling, checkpoint selection, submission packaging, threshold tuning, fuzzy join, silent deduplication, or B/C gate creation was run.",
        "",
        "## Next Allowed Action",
        "Enter knowledge-review. If verdict is not `shared-surface-ready-for-outcome-audit`, do not run full eval or outcome-consistency audit.",
    ]
    target = out / "phase27_a_v3_2b_fixed_manifest_shared_surface_experiment_record.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest")
    args = parser.parse_args()
    write_record(args.output_dir, args.input_manifest)


if __name__ == "__main__":
    main()

