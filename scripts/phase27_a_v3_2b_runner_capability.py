"""Read-only runner capability audit for fixed-manifest reacquisition."""
import argparse
from pathlib import Path

from scripts.phase27_a_v3_2b_common import ensure_dirs, write_json


FIXED_MANIFEST_ARGS = (
    "--fixed-manifest",
    "--pair-manifest",
    "--manifest-csv",
    "--pair-list",
    "--json-list",
    "--input-manifest",
)
REEXPORT_HINTS = ("--json-root", "--test_dataset", "write_pairuav_devval_outputs", "result.txt")


def audit_runner_capability(repo_root):
    root = Path(repo_root)
    candidates = []
    fixed = []
    reexport = []
    for pattern in ("*.py", "*.sh"):
        for path in root.rglob(pattern):
            rel_parts = path.relative_to(root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if rel_parts[0] in {"tests", "__pycache__"}:
                continue
            if path.name.startswith("phase27_a_v3_2") or path.name.startswith("run_phase27"):
                continue
            rel = str(path.relative_to(root))
            low = rel.lower()
            if not any(k in low for k in ("eval", "test", "infer", "predict", "run", "submit")):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            is_real_python_entrypoint = path.suffix == ".py" and any(k in low for k in ("eval", "infer", "predict"))
            fixed_hits = [k for k in FIXED_MANIFEST_ARGS if k in text] if is_real_python_entrypoint else []
            reexport_hits = [k for k in REEXPORT_HINTS if k in text]
            if fixed_hits or reexport_hits:
                row = {"path": rel, "fixed_manifest_args": ",".join(fixed_hits), "reexport_hints": ",".join(reexport_hits)}
                candidates.append(row)
                if fixed_hits:
                    fixed.append(row)
                if reexport_hits:
                    reexport.append(row)
    if fixed:
        status = "fixed_manifest_runner_available"
        decision = "attempt_fixed_manifest_reacquisition"
    elif reexport:
        status = "reexport_only"
        decision = "blocked_runner_interface"
    else:
        status = "blocked_runner_interface"
        decision = "blocked_runner_interface"
    return {
        "capability_status": status,
        "candidate_entrypoints": candidates[:50],
        "fixed_manifest_argument_evidence": fixed[:20],
        "result_surface_export_evidence": candidates[:20],
        "decision": decision,
    }


def write_report(out, data):
    lines = [
        "# A-v3.2b Runner Capability Report",
        "",
        f"- capability_status: `{data['capability_status']}`",
        f"- decision: `{data['decision']}`",
        f"- candidate_entrypoints: {len(data['candidate_entrypoints'])}",
        "",
        "Read-only audit. No model evaluation, training, finetuning, or result generation was launched.",
    ]
    (out / "reports" / "runner_capability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ensure_dirs(args.output_dir)
    data = audit_runner_capability(args.repo_root)
    write_json(out / "metrics" / "runner_capability.json", data)
    write_report(out, data)


if __name__ == "__main__":
    main()
