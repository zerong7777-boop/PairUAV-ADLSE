import argparse
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LIKELY_IDENTITY_KEYS = [
    "pair_id",
    "sample_id",
    "canonical_pair_id",
    "group_id",
    "json_id",
    "json_path",
    "instance",
    "scene_id",
]


def import_status(module_name):
    try:
        module = importlib.import_module(module_name)
        return {"module": module_name, "ok": True, "path": getattr(module, "__file__", "")}
    except Exception as exc:
        return {"module": module_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def infer_dataset_contract():
    result = {
        "status": "blocked_missing_pair_identity",
        "live_batch_inspected": False,
        "likely_identity_keys": LIKELY_IDENTITY_KEYS,
        "sample_id_contract": "<group_id>/<json_id>",
        "imports": [
            import_status("reloc3r.datasets.pairuav"),
            import_status("train"),
        ],
        "notes": [],
    }
    try:
        pairuav = importlib.import_module("reloc3r.datasets.pairuav")
        normalize = getattr(pairuav, "_normalize_record", None)
        if normalize is not None:
            result["notes"].append("PairUAV normalizes group_id and json_id; matcher payload uses sample_id='<group_id>/<json_id>'.")
            result["status"] = "identity_contract_inferred"
    except Exception as exc:
        result["notes"].append(f"PairUAV import unavailable for contract inference: {type(exc).__name__}: {exc}")
    result["blocked_reason"] = (
        "No dataset roots were provided to this conservative inspector; live batch construction was intentionally skipped."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect PairUAV batch identity fields without running training.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-batches", type=int, default=1)
    args = parser.parse_args()

    report = infer_dataset_contract()
    report["requested_num_batches"] = int(args.num_batches)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
