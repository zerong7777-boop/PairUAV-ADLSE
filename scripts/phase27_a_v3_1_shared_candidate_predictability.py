from pathlib import Path

from scripts.phase27_a_v3_1_coverage_bias_audit import infer_stress_variants
from scripts.phase27_a_v3_1_shared_surface_common import ensure_output_dirs, read_csv_dicts, safe_div, truthy, write_csv_dicts, write_json
from scripts.phase27_a_v3_1_shared_outcome_consistency import shared_rows


CANDIDATE_FIELDS = [
    "evidence_sufficient_candidate",
    "heading_observable_candidate",
    "range_observable_candidate",
    "semantic_geometric_conflict_candidate",
    "local_alignment_needed_candidate",
    "ambiguity_candidate",
    "ordinary_candidate",
    "control_candidate",
]


def candidate_score_fields(rows):
    header = set(rows[0]) if rows else set()
    return [field for field in CANDIDATE_FIELDS if field in header]


def outcome_fields(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    fields = ["baseline_heading_hard", "baseline_range_hard", "baseline_joint_hard"]
    for variant in stress_variants:
        fields.extend([f"stress_{variant}_heading_sensitive", f"stress_{variant}_range_sensitive", f"stress_{variant}_joint_sensitive"])
    return [field for field in fields if rows and field in rows[0]]


def score_value(row, field):
    value = row.get(field)
    if value in (None, ""):
        return None
    low = str(value).strip().lower()
    if low in {"true", "yes", "joined", "1"}:
        return 1.0
    if low in {"false", "no", "unjoined", "0"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return None


def _valid(rows, score_field, outcome_field):
    vals = []
    for row in rows:
        score = score_value(row, score_field)
        if score is None:
            continue
        vals.append((score, truthy(row.get(outcome_field)), row))
    return vals


def auc_rank(rows, score_field, outcome_field):
    valid = _valid(shared_rows(rows), score_field, outcome_field)
    n_pos = sum(1 for _, label, _ in valid if label)
    n_neg = len(valid) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    asc = sorted(valid, key=lambda item: item[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(asc):
        j = i + 1
        while j < len(asc) and asc[j][0] == asc[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_rank_sum += avg_rank * sum(1 for _, label, _ in asc[i:j] if label)
        i = j
    return safe_div(pos_rank_sum - n_pos * (n_pos + 1) / 2.0, n_pos * n_neg)


def precision_recall_at_k_tie_aware(rows, score_field, outcome_field, k):
    valid = sorted(_valid(shared_rows(rows), score_field, outcome_field), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label, _ in valid if label)
    if not valid:
        return {"precision": 0.0, "recall": 0.0, "selected": 0}
    cutoff_index = min(k, len(valid)) - 1
    cutoff_score = valid[cutoff_index][0]
    selected = [item for item in valid if item[0] >= cutoff_score]
    hits = sum(1 for _, label, _ in selected if label)
    return {"precision": safe_div(hits, len(selected)), "recall": safe_div(hits, positives), "selected": len(selected)}


def flag_wide_precision_recall(rows, flag_field, outcome_field):
    srows = shared_rows(rows)
    selected = [row for row in srows if truthy(row.get(flag_field))]
    positives = sum(1 for row in srows if truthy(row.get(outcome_field)))
    hits = sum(1 for row in selected if truthy(row.get(outcome_field)))
    return {"selected": len(selected), "hits": hits, "precision": safe_div(hits, len(selected)), "recall": safe_div(hits, positives)}


def _top_decile_lift(rows, score_field, outcome_field):
    valid = sorted(_valid(shared_rows(rows), score_field, outcome_field), key=lambda item: item[0], reverse=True)
    if not valid:
        return 0.0
    base = safe_div(sum(1 for _, label, _ in valid if label), len(valid))
    top_n = max(1, int(len(valid) * 0.1))
    top = valid[:top_n]
    return safe_div(safe_div(sum(1 for _, label, _ in top if label), len(top)), base)


def compute_shared_candidate_predictability(rows, stress_variants=None):
    stress_variants = stress_variants or infer_stress_variants(rows)
    scores = candidate_score_fields(rows)
    outcomes = outcome_fields(rows, stress_variants)
    pair_metrics = []
    by_group = []
    for score in scores:
        for outcome in outcomes:
            metric = {"score_field": score, "outcome_field": outcome, "shared_rows": len(shared_rows(rows)), "auc": auc_rank(rows, score, outcome), "top_decile_lift": _top_decile_lift(rows, score, outcome)}
            for k in (50, 100, 500, 1000):
                pr = precision_recall_at_k_tie_aware(rows, score, outcome, k)
                metric[f"precision_at_{k}"] = pr["precision"]
                metric[f"recall_at_{k}"] = pr["recall"]
                metric[f"selected_at_{k}"] = pr["selected"]
            metric["flag_wide"] = flag_wide_precision_recall(rows, score, outcome)
            pair_metrics.append(metric)
            for key in sorted(set((row.get("target_key", "") or "missing", row.get("group_id", "") or "missing") for row in shared_rows(rows))):
                subset = [row for row in rows if truthy(row.get("shared_baseline_stress_joined")) and (row.get("target_key", "") or "missing", row.get("group_id", "") or "missing") == key]
                fw = flag_wide_precision_recall(subset, score, outcome)
                by_group.append({"score_field": score, "outcome_field": outcome, "target_key": key[0], "group_id": key[1], **fw})
    useful = [m for m in pair_metrics if m["flag_wide"]["precision"] > 0 and m["top_decile_lift"] > 1.25]
    return {"score_fields": scores, "outcome_fields": outcomes, "pair_metrics": pair_metrics, "best_pairs": sorted(pair_metrics, key=lambda x: (x["top_decile_lift"], x.get("precision_at_100", 0)), reverse=True)[:20], "useful_pair_count": len(useful), "by_target_group": by_group}


def _write_report(metrics, path):
    lines = ["# A-v3.1 Shared Candidate Predictability", "", f"- useful_pair_count: {metrics['useful_pair_count']}"]
    for row in metrics["best_pairs"][:10]:
        lines.append(f"- {row['score_field']} -> {row['outcome_field']}: auc={row['auc']:.4f}, p@100={row['precision_at_100']:.4f}, lift={row['top_decile_lift']:.4f}, flag_precision={row['flag_wide']['precision']:.4f}")
    lines.append("")
    lines.append("Validation-only; no training policy is produced.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shared_candidate_predictability(rows, output_dir):
    out = ensure_output_dirs(output_dir)
    metrics = compute_shared_candidate_predictability(rows)
    by_group = metrics.pop("by_target_group")
    write_json(out / "metrics" / "a_v3_1_shared_candidate_predictability_metrics.json", metrics)
    write_csv_dicts(out / "tables" / "a_v3_1_candidate_predictability_by_target_group.csv", by_group, ["score_field", "outcome_field", "target_key", "group_id", "selected", "hits", "precision", "recall"])
    metrics["by_target_group"] = by_group
    _write_report(metrics, out / "reports" / "a_v3_1_shared_candidate_predictability_report.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_shared_candidate_predictability(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
