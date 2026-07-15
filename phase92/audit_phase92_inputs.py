#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/jgzn/SSD_lexar/RZ/UAVM")
PHASE91 = ROOT / "runs/phase91_polarrel_problem_mechanism_validation_v1/router_smokes"
PHASE92 = ROOT / "runs/phase92_minimum_sufficient_relation_feasibility_v1"
WORKER = ROOT / "external/reloc3r_pairuav"

REQUIRED = {
    "worker": WORKER,
    "train_py": WORKER / "train.py",
    "eval_pairuav_py": WORKER / "eval_pairuav.py",
    "phase92_stage0_verdict": PHASE92 / "manifests/phase92_stage0_verdict.json",
    "phase92_stage1_contract": PHASE92 / "reports/phase92_stage1_msr_smoke_contract.md",
    "phase91_g6_verdict": PHASE91 / "phase91_g6_router_lq_verdict.md",
    "phase91_g6_metrics": PHASE91 / "phase91_g6_all_model_metrics.csv",
    "phase91_g6_pairwise": PHASE91 / "phase91_g6_pairwise_vs_B4_B3.csv",
    "phase91_g6_buckets": PHASE91 / "phase91_g6_gt_bucket_bests.csv",
    "init_checkpoint": ROOT / "runs/explore_axisdecouple_reloc3r_head_v1/checkpoints/reloc3r512_backbone_only_no_pose_head.pth",
    "train_json": ROOT / "runs/devsplit_v1/train_json",
    "val_json": ROOT / "runs/phase56_reloc3r_geometry_consistent_angle_training_v1/surfaces/phase48_4089_fixed/val_json",
    "image_root": ROOT / "official/UAVM_2026/pairUAV/train_tour",
}


def main() -> int:
    rows = []
    for name, path in REQUIRED.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists()})
    fail_count = sum(1 for row in rows if not row["exists"])

    out_dir = PHASE92 / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": "phase92_input_audit",
        "fail_count": fail_count,
        "rows": rows,
        "verdict": "pass" if fail_count == 0 else "fail",
    }
    (out_dir / "phase92_input_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{'OK' if row['exists'] else 'MISSING'} {row['name']} {row['path']}")
    print(f"verdict={payload['verdict']} fail_count={fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

