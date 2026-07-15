from pathlib import Path

from scripts.phase27_a_v3_validation_extension_common import ensure_dirs, read_csv_dicts, safe_div, truthy, write_csv_dicts, write_json


BASE_SCORE_FIELDS = [
    "evidence_sufficient_candidate",
    "heading_observable_candidate",
    "range_observable_candidate",
    "semantic_geometric_conflict_candidate",
    "local_alignment_needed_candidate",
    "ambiguity_candidate",
    "ordinary_candidate",
    "control_candidate",
]
NUMERIC_SCORE_FIELDS = [
    "evidence_sufficiency_score",
    "heading_observability_score",
    "range_observability_score",
    "semantic_geometric_conflict_score",
    "match_sufficiency_score",
    "control_stability_score",
]
BASE_OUTCOME_FIELDS = [
    "baseline_heading_hard",
    "baseline_range_hard",
    "baseline_joint_hard",
    "stress_heading_sensitive",
    "stress_range_sensitive",
    "stress_joint_sensitive",
    "tail_error_high",
    "READY_CONTROL_PRESERVATION",
]


def candidate_score_fields(rows):
    fields = []
    header = set(rows[0].keys()) if rows else set()
    for field in BASE_SCORE_FIELDS + NUMERIC_SCORE_FIELDS:
        if field in header:
            fields.append(field)
    return fields


def outcome_fields(rows):
    header = set(rows[0].keys()) if rows else set()
    return [field for field in BASE_OUTCOME_FIELDS if field in header]


def _score(row, field):
    value = row.get(field)
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    low = text.lower()
    if low in {"true", "yes", "y"}:
        return 1.0
    if low in {"false", "no", "n"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def _auc_rank(valid):
    n_pos = sum(1 for _, label in valid if label)
    n_neg = len(valid) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    # Rank-sum AUC with average ranks for ties. Ranks ascend by score.
    ascending = sorted(valid, key=lambda item: item[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(ascending):
        j = i + 1
        while j < len(ascending) and ascending[j][0] == ascending[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_rank_sum += avg_rank * sum(1 for _, label in ascending[i:j] if label)
        i = j
    return safe_div(pos_rank_sum - n_pos * (n_pos + 1) / 2.0, n_pos * n_neg)


def _ranked_valid(rows, score_field, outcome_field):
    valid = []
    for row in rows:
        score = _score(row, score_field)
        if score is None:
            continue
        valid.append((score, truthy(row.get(outcome_field))))
    valid.sort(key=lambda item: item[0], reverse=True)
    return valid


def _precision_recall(valid, k):
    top = valid[: min(k, len(valid))]
    positives = sum(1 for _, label in valid if label)
    hits = sum(1 for _, label in top if label)
    return safe_div(hits, len(top)), safe_div(hits, positives)


def _top_decile_lift(valid):
    if not valid:
        return 0.0
    base = safe_div(sum(1 for _, label in valid if label), len(valid))
    n = max(1, int(len(valid) * 0.1))
    top = valid[:n]
    top_rate = safe_div(sum(1 for _, label in top if label), len(top))
    return safe_div(top_rate, base)


def _deciles_for_pair(valid, score_field, outcome_field):
    rows = []
    if not valid:
        return rows
    size = max(1, len(valid) // 10)
    for idx in range(10):
        start = idx * size
        end = len(valid) if idx == 9 else min(len(valid), (idx + 1) * size)
        chunk = valid[start:end]
        if not chunk:
            continue
        positives = sum(1 for _, label in chunk if label)
        rows.append(
            {
                "score_field": score_field,
                "outcome_field": outcome_field,
                "decile": idx + 1,
                "count": len(chunk),
                "positives": positives,
                "positive_rate": safe_div(positives, len(chunk)),
            }
        )
    return rows


def compute_predictability_for_pair(rows, score_field, outcome_field):
    valid = _ranked_valid(rows, score_field, outcome_field)
    metrics = {
        "score_field": score_field,
        "outcome_field": outcome_field,
        "positive_count": sum(1 for _, label in valid if label),
        "valid_score_count": len(valid),
        "auc": _auc_rank(valid),
        "top_decile_lift": _top_decile_lift(valid),
    }
    for k in (50, 100, 500, 1000):
        precision, recall = _precision_recall(valid, k)
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall
    return metrics


def compute_candidate_to_outcome_predictability(rows):
    scores = candidate_score_fields(rows)
    outcomes = outcome_fields(rows)
    pairs = []
    deciles = []
    for score_field in scores:
        for outcome_field in outcomes:
            valid = _ranked_valid(rows, score_field, outcome_field)
            pair = compute_predictability_for_pair(rows, score_field, outcome_field)
            pairs.append(pair)
            deciles.extend(_deciles_for_pair(valid, score_field, outcome_field))
    useful = [
        row
        for row in pairs
        if row["positive_count"] > 0
        and row["top_decile_lift"] > 1.25
        and row["precision_at_100"] > safe_div(row["positive_count"], row["valid_score_count"])
    ]
    return {
        "score_fields": scores,
        "outcome_fields": outcomes,
        "pair_metrics": pairs,
        "best_pairs": sorted(pairs, key=lambda r: (r["top_decile_lift"], r["precision_at_100"]), reverse=True)[:20],
        "useful_pair_count": len(useful),
        "deciles": deciles,
    }


def _write_report(metrics, path):
    lines = ["# A-v3 Candidate-To-Outcome Predictability", ""]
    lines.append(f"- useful_pair_count: {metrics['useful_pair_count']}")
    lines.append("")
    lines.append("## Top Pairs")
    for row in metrics["best_pairs"][:10]:
        lines.append(
            f"- {row['score_field']} -> {row['outcome_field']}: "
            f"auc={row['auc']:.4f}, p@100={row['precision_at_100']:.4f}, lift={row['top_decile_lift']:.4f}"
        )
    lines += ["", "This report is validation-only and does not define a training policy."]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_candidate_to_outcome_predictability(rows, output_dir):
    out = ensure_dirs(output_dir)
    metrics = compute_candidate_to_outcome_predictability(rows)
    deciles = metrics.pop("deciles")
    write_json(out / "metrics" / "a_v3_candidate_to_outcome_predictability_metrics.json", metrics)
    write_csv_dicts(
        out / "tables" / "a_v3_candidate_to_outcome_deciles.csv",
        deciles,
        ["score_field", "outcome_field", "decile", "count", "positives", "positive_rate"],
    )
    metrics["deciles"] = deciles
    _write_report(metrics, out / "reports" / "a_v3_candidate_to_outcome_predictability_report.md")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_candidate_to_outcome_predictability(read_csv_dicts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
