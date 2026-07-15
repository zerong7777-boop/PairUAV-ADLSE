import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANDIDATE_TRAINABLE_TOKENS = ("head", "pose", "range", "heading", "proj")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_status_from_json(path):
    data = read_json(path)
    status = str(data.get("status", "")).lower()
    passed = data.get("pass")
    if passed is None:
        passed = data.get("passed")
    ok = status in {"pass", "ready"} or passed is True
    return ok, data


def load_state_dict_keys(checkpoint):
    try:
        import torch
    except Exception as exc:
        return None, f"torch_import_failed: {type(exc).__name__}: {exc}"
    try:
        payload = _trusted_torch_load(torch, checkpoint)
    except Exception as exc:
        return None, f"torch_load_failed: {type(exc).__name__}: {exc}"
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        return None, f"checkpoint_payload_not_state_dict: {type(state).__name__}"
    return sorted(str(key) for key in state.keys()), None


def _trusted_torch_load(torch, checkpoint):
    try:
        return torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint), map_location="cpu")


def candidate_parameter_names(state_keys):
    names = []
    for name in state_keys:
        lower = name.lower()
        if any(token in lower for token in CANDIDATE_TRAINABLE_TOKENS):
            names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description="Audit geometry-policy finetune contract for one-batch smoke readiness.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--policy-csv", required=True)
    parser.add_argument("--policy-audit-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    report = {
        "checkpoint": args.checkpoint,
        "policy_csv": args.policy_csv,
        "policy_audit_json": args.policy_audit_json,
    }
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        report.update({"status": "blocked_checkpoint_load", "reason": "checkpoint_missing"})
        return write_report(args.out_json, report, exit_code=1)

    actual_sha = sha256_file(checkpoint)
    report["checkpoint_sha256"] = actual_sha
    if actual_sha.lower() != str(args.expected_checkpoint_sha256).lower():
        report.update({"status": "blocked_checkpoint_load", "reason": "checkpoint_sha256_mismatch"})
        return write_report(args.out_json, report, exit_code=1)

    audit_path = Path(args.policy_audit_json)
    if not audit_path.is_file():
        report.update({"status": "blocked_policy_audit_failed", "reason": "policy_audit_json_missing"})
        return write_report(args.out_json, report, exit_code=1)
    policy_ok, policy_audit = audit_status_from_json(audit_path)
    report["policy_audit_status"] = policy_audit.get("status")
    if not policy_ok:
        report.update({"status": "blocked_policy_audit_failed", "reason": "policy_audit_status_not_pass"})
        return write_report(args.out_json, report, exit_code=1)

    try:
        from reloc3r.geometry_policy_loss import GeometryPolicyTable
    except Exception as exc:
        report.update({"status": "blocked_checkpoint_load", "reason": f"geometry_policy_table_import_failed: {type(exc).__name__}: {exc}"})
        return write_report(args.out_json, report, exit_code=1)

    try:
        table = GeometryPolicyTable.from_csv(args.policy_csv)
    except Exception as exc:
        report.update({"status": "blocked_policy_audit_failed", "reason": f"policy_csv_load_failed: {type(exc).__name__}: {exc}"})
        return write_report(args.out_json, report, exit_code=1)
    report["policy_entry_count"] = len(table)

    state_keys, error = load_state_dict_keys(checkpoint)
    if error:
        report.update({"status": "blocked_checkpoint_load", "reason": error})
        return write_report(args.out_json, report, exit_code=1)

    candidates = candidate_parameter_names(state_keys)
    report["checkpoint_state_key_count"] = len(state_keys)
    report["candidate_trainable_parameter_count"] = len(candidates)
    report["candidate_trainable_parameter_names"] = candidates
    if not candidates:
        report.update({"status": "blocked_missing_parameter_group", "reason": "no head/pose/range/heading/proj state_dict keys"})
        return write_report(args.out_json, report, exit_code=1)

    report["status"] = "contract_ready_for_one_batch_smoke"
    return write_report(args.out_json, report, exit_code=0)


def write_report(path, report, exit_code):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
