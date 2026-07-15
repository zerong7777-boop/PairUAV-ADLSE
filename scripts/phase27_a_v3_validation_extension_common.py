"""Common utilities for Phase27 A-v3 validation-extension audits."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_dirs(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_dicts(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, data: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def to_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def safe_div(num, den) -> float:
    try:
        den_f = float(den)
        if den_f == 0.0:
            return 0.0
        return float(num) / den_f
    except (TypeError, ValueError):
        return 0.0


def quantiles(values: list[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    def percentile(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    return {
        "p50": round(percentile(0.50), 10),
        "p90": round(percentile(0.90), 10),
        "p95": round(percentile(0.95), 10),
        "p99": round(percentile(0.99), 10),
    }


def rank_rows(rows: list[dict], score_field: str) -> list[dict]:
    def key(row: dict):
        raw = row.get(score_field)
        try:
            score = float(str(raw).strip())
            valid = 1
        except (TypeError, ValueError):
            score = 0.0
            valid = 0
        return (valid, score)

    return sorted(rows, key=key, reverse=True)


def count_true(rows: list[dict], field: str) -> int:
    return sum(1 for row in rows if truthy(row.get(field)))


def group_count(rows: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def precision_recall_at_k(rows: list[dict], score_field: str, outcome_field: str, k: int) -> dict:
    ranked = rank_rows(rows, score_field)
    positives = count_true(rows, outcome_field)
    top = ranked[:k]
    tp = count_true(top, outcome_field)
    return {
        "k": k,
        "tp_at_k": tp,
        "positives": positives,
        "precision": safe_div(tp, len(top)),
        "recall": safe_div(tp, positives),
    }


def auc_pairwise(rows: list[dict], score_field: str, outcome_field: str) -> float:
    positives = [row for row in rows if truthy(row.get(outcome_field))]
    negatives = [row for row in rows if not truthy(row.get(outcome_field))]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        ps = to_float(pos.get(score_field))
        for neg in negatives:
            ns = to_float(neg.get(score_field))
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return safe_div(wins, total)


def decile_table(rows: list[dict], score_field: str, outcome_field: str) -> list[dict]:
    ranked = rank_rows(rows, score_field)
    if not ranked:
        return []
    table = []
    n = len(ranked)
    for decile in range(10):
        start = int(n * decile / 10)
        end = int(n * (decile + 1) / 10)
        bucket = ranked[start:end]
        positives = count_true(bucket, outcome_field)
        table.append({
            "score_field": score_field,
            "outcome_field": outcome_field,
            "decile": decile + 1,
            "n": len(bucket),
            "positives": positives,
            "positive_rate": safe_div(positives, len(bucket)),
        })
    return table
