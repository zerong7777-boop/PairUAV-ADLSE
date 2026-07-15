import csv
import json
import math
from pathlib import Path

import numpy as np


TOKEN_FEATURE_NAMES = [
    "src_x",
    "src_y",
    "dst_x",
    "dst_y",
    "disp_x",
    "disp_y",
    "confidence",
    "rank_norm",
    "src_bin_x",
    "src_bin_y",
    "dst_bin_x",
    "dst_bin_y",
    "translation_residual",
    "similarity_residual",
    "affine_residual",
    "translation_inlier",
    "similarity_inlier",
    "affine_inlier",
]

HYPOTHESIS_NAMES = ["translation", "similarity", "affine"]

HYPOTHESIS_FEATURE_NAMES = [
    "tx",
    "ty",
    "a00",
    "a01",
    "a10",
    "a11",
    "mean_residual",
    "inlier_ratio",
    "valid",
]

GLOBAL_FEATURE_NAMES = [
    "log1p_valid_count",
    "valid_ratio",
    "mean_confidence",
    "max_confidence",
    "std_confidence",
    "spatial_entropy",
    "translation_dx",
    "translation_dy",
    "translation_mean_residual",
    "translation_inlier_ratio",
    "similarity_scale",
    "similarity_rot_cos",
    "similarity_rot_sin",
    "similarity_mean_residual",
    "similarity_inlier_ratio",
    "affine_mean_residual",
    "affine_inlier_ratio",
    "fallback_used",
]


def sample_to_train_match_path(cache_root, sample_id):
    group, pair_id = str(sample_id).split("/", 1)
    left, right = pair_id.split("_", 1)
    return Path(cache_root) / group / f"image-{left}_image-{right}_matches.npz"


def _as_array(record, key, dtype=np.float32):
    if isinstance(record, np.lib.npyio.NpzFile):
        value = record[key]
    else:
        value = record[key]
    return np.asarray(value, dtype=dtype)


def _empty_packet(topk=128, fallback_used=True, reason="empty"):
    return {
        "tokens": np.zeros((int(topk), len(TOKEN_FEATURE_NAMES)), dtype=np.float32),
        "token_mask": np.zeros((int(topk),), dtype=np.float32),
        "hypothesis_features": np.zeros((len(HYPOTHESIS_NAMES), len(HYPOTHESIS_FEATURE_NAMES)), dtype=np.float32),
        "global_stats": np.asarray(
            [0.0] * (len(GLOBAL_FEATURE_NAMES) - 1) + [1.0 if fallback_used else 0.0],
            dtype=np.float32,
        ),
        "fallback_used": np.float32(1.0 if fallback_used else 0.0),
        "raw_counts": {"total_matches": 0, "valid_matches": 0},
        "reason": reason,
    }


def _valid_match_arrays(record_or_npz):
    keypoints0 = _as_array(record_or_npz, "keypoints0", dtype=np.float32)
    keypoints1 = _as_array(record_or_npz, "keypoints1", dtype=np.float32)
    matches = _as_array(record_or_npz, "matches", dtype=np.int64).reshape(-1)
    confidence = _as_array(record_or_npz, "match_confidence", dtype=np.float32).reshape(-1)
    if keypoints0.ndim != 2 or keypoints1.ndim != 2 or keypoints0.shape[-1] < 2 or keypoints1.shape[-1] < 2:
        raise ValueError("Matcher packet has invalid keypoint arrays")
    if confidence.shape[0] < matches.shape[0]:
        padded = np.zeros(matches.shape[0], dtype=np.float32)
        padded[: confidence.shape[0]] = confidence
        confidence = padded
    valid = (matches >= 0) & (matches < len(keypoints1)) & np.isfinite(confidence) & (confidence > 0)
    src = keypoints0[: len(matches)][valid, :2].astype(np.float32, copy=False)
    dst = keypoints1[matches[valid], :2].astype(np.float32, copy=False)
    conf = confidence[valid].astype(np.float32, copy=False)
    return src, dst, conf, int(matches.shape[0])


def _normalize_xy(points, image_size):
    width, height = float(image_size[0]), float(image_size[1])
    denom = np.asarray([max(width, 1.0), max(height, 1.0)], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) / denom


def _spatial_entropy(src_norm, grid_size):
    if src_norm.size == 0:
        return 0.0
    grid_size = int(grid_size)
    xbin = np.clip((src_norm[:, 0] * grid_size).astype(np.int64), 0, grid_size - 1)
    ybin = np.clip((src_norm[:, 1] * grid_size).astype(np.int64), 0, grid_size - 1)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x, y in zip(xbin, ybin):
        counts[int(y), int(x)] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.reshape(-1) / total
    probs = probs[probs > 0]
    denom = math.log(float(grid_size * grid_size)) if grid_size > 1 else 1.0
    return float(-(probs * np.log(probs)).sum() / max(denom, 1e-6))


def _weighted_mean(values, weights):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    total = float(weights.sum())
    if total <= 1e-8:
        return np.asarray(values, dtype=np.float32).mean(axis=0)
    return (np.asarray(values, dtype=np.float32) * weights[:, None]).sum(axis=0) / total


def _hypothesis_row(name, matrix, translation, residuals, threshold, valid):
    row = np.zeros(len(HYPOTHESIS_FEATURE_NAMES), dtype=np.float32)
    if matrix is not None:
        row[2:6] = np.asarray(matrix, dtype=np.float32).reshape(4)
    if translation is not None:
        row[0:2] = np.asarray(translation, dtype=np.float32).reshape(2)
    if valid and residuals.size:
        row[6] = float(np.mean(residuals))
        row[7] = float(np.mean(residuals <= float(threshold)))
        row[8] = 1.0
    return row


def _predict(src, matrix, translation):
    return np.asarray(src, dtype=np.float32) @ np.asarray(matrix, dtype=np.float32).T + np.asarray(
        translation, dtype=np.float32
    )


def _estimate_translation(src, dst, weights):
    disp = dst - src
    translation = _weighted_mean(disp, weights)
    matrix = np.eye(2, dtype=np.float32)
    residuals = np.linalg.norm(_predict(src, matrix, translation) - dst, axis=1)
    return matrix, translation, residuals, True


def _estimate_similarity(src, dst, weights):
    if len(src) < 2:
        return None, None, np.zeros((0,), dtype=np.float32), False
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    src_mean = _weighted_mean(src, weights)
    dst_mean = _weighted_mean(dst, weights)
    x = src - src_mean
    y = dst - dst_mean
    weight_sum = max(float(weights.sum()), 1e-8)
    covariance = (x * weights[:, None]).T @ y / weight_sum
    try:
        u, s, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None, None, np.zeros((0,), dtype=np.float32), False
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    variance = float((weights * np.sum(np.square(x), axis=1)).sum() / weight_sum)
    if variance <= 1e-8:
        return None, None, np.zeros((0,), dtype=np.float32), False
    scale = float(np.sum(s) / variance)
    matrix = (scale * rotation).astype(np.float32)
    translation = (dst_mean - src_mean @ matrix.T).astype(np.float32)
    residuals = np.linalg.norm(_predict(src, matrix, translation) - dst, axis=1)
    return matrix, translation, residuals, True


def _estimate_affine(src, dst, weights):
    if len(src) < 3:
        return None, None, np.zeros((0,), dtype=np.float32), False
    design = np.concatenate([src, np.ones((len(src), 1), dtype=np.float32)], axis=1)
    sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float32).reshape(-1, 1))
    try:
        coeff, _, _, _ = np.linalg.lstsq(design * sqrt_w, dst * sqrt_w, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, np.zeros((0,), dtype=np.float32), False
    matrix = coeff[:2, :].T.astype(np.float32)
    translation = coeff[2, :].astype(np.float32)
    residuals = np.linalg.norm(_predict(src, matrix, translation) - dst, axis=1)
    return matrix, translation, residuals, True


def _hypotheses(src_norm, dst_norm, confidence, residual_threshold):
    estimators = [_estimate_translation, _estimate_similarity, _estimate_affine]
    rows = []
    residual_map = {}
    inlier_map = {}
    for name, estimator in zip(HYPOTHESIS_NAMES, estimators):
        matrix, translation, residuals, valid = estimator(src_norm, dst_norm, confidence)
        rows.append(_hypothesis_row(name, matrix, translation, residuals, residual_threshold, valid))
        if valid and residuals.size:
            residual_map[name] = residuals.astype(np.float32)
            inlier_map[name] = (residuals <= float(residual_threshold)).astype(np.float32)
        else:
            residual_map[name] = np.zeros((len(src_norm),), dtype=np.float32)
            inlier_map[name] = np.zeros((len(src_norm),), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32), residual_map, inlier_map


def build_correspondence_token_packet(
    record_or_npz,
    image_size=(512, 512),
    topk=128,
    grid_size=8,
    residual_threshold=0.035,
):
    """Build a fixed-shape, single-pair correspondence token packet.

    This packet deliberately preserves per-match structure. It is intended for
    a token/set angle specialist, not for another low-dimensional stats MLP.
    """
    topk = int(topk)
    try:
        if isinstance(record_or_npz, (str, Path)):
            path = Path(record_or_npz)
            if not path.is_file():
                return _empty_packet(topk=topk, fallback_used=True, reason="missing_match_file")
            with np.load(path, allow_pickle=True) as data:
                return build_correspondence_token_packet(
                    data,
                    image_size=image_size,
                    topk=topk,
                    grid_size=grid_size,
                    residual_threshold=residual_threshold,
                )
        src, dst, conf, total_matches = _valid_match_arrays(record_or_npz)
    except Exception as exc:
        packet = _empty_packet(topk=topk, fallback_used=True, reason="invalid_match_file")
        packet["error"] = str(exc)
        return packet

    valid_count = int(conf.shape[0])
    if valid_count <= 0 or total_matches <= 0:
        packet = _empty_packet(topk=topk, fallback_used=False, reason="no_valid_matches")
        packet["raw_counts"] = {"total_matches": int(total_matches), "valid_matches": 0}
        return packet

    src_norm = _normalize_xy(src, image_size)
    dst_norm = _normalize_xy(dst, image_size)
    disp = dst_norm - src_norm
    hypothesis_features, residual_map, inlier_map = _hypotheses(
        src_norm,
        dst_norm,
        conf,
        residual_threshold=float(residual_threshold),
    )

    order = np.argsort(-conf)[:topk]
    selected = int(order.shape[0])
    tokens = np.zeros((topk, len(TOKEN_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((topk,), dtype=np.float32)
    if selected:
        src_sel = src_norm[order]
        dst_sel = dst_norm[order]
        disp_sel = disp[order]
        conf_sel = conf[order]
        denom = float(max(valid_count - 1, 1))
        rank_norm = np.arange(selected, dtype=np.float32) / denom
        src_bins = np.clip((src_sel * float(grid_size)).astype(np.int64), 0, int(grid_size) - 1).astype(np.float32)
        dst_bins = np.clip((dst_sel * float(grid_size)).astype(np.int64), 0, int(grid_size) - 1).astype(np.float32)
        bin_denom = float(max(int(grid_size) - 1, 1))
        src_bins /= bin_denom
        dst_bins /= bin_denom
        token_values = np.concatenate(
            [
                src_sel,
                dst_sel,
                disp_sel,
                conf_sel[:, None],
                rank_norm[:, None],
                src_bins,
                dst_bins,
                residual_map["translation"][order, None],
                residual_map["similarity"][order, None],
                residual_map["affine"][order, None],
                inlier_map["translation"][order, None],
                inlier_map["similarity"][order, None],
                inlier_map["affine"][order, None],
            ],
            axis=1,
        )
        tokens[:selected] = np.nan_to_num(token_values, nan=0.0, posinf=0.0, neginf=0.0)
        mask[:selected] = 1.0

    global_values = {
        "log1p_valid_count": float(math.log1p(valid_count)),
        "valid_ratio": float(valid_count / max(total_matches, 1)),
        "mean_confidence": float(conf.mean()),
        "max_confidence": float(conf.max()),
        "std_confidence": float(conf.std()),
        "spatial_entropy": _spatial_entropy(src_norm, grid_size),
        "translation_dx": float(hypothesis_features[0, 0]),
        "translation_dy": float(hypothesis_features[0, 1]),
        "translation_mean_residual": float(hypothesis_features[0, 6]),
        "translation_inlier_ratio": float(hypothesis_features[0, 7]),
        "similarity_scale": float(np.sqrt(max(np.linalg.det(hypothesis_features[1, 2:6].reshape(2, 2)), 0.0))),
        "similarity_rot_cos": float(hypothesis_features[1, 2]),
        "similarity_rot_sin": float(hypothesis_features[1, 4]),
        "similarity_mean_residual": float(hypothesis_features[1, 6]),
        "similarity_inlier_ratio": float(hypothesis_features[1, 7]),
        "affine_mean_residual": float(hypothesis_features[2, 6]),
        "affine_inlier_ratio": float(hypothesis_features[2, 7]),
        "fallback_used": 0.0,
    }
    global_stats = np.asarray([global_values[name] for name in GLOBAL_FEATURE_NAMES], dtype=np.float32)
    return {
        "tokens": tokens,
        "token_mask": mask,
        "hypothesis_features": np.nan_to_num(hypothesis_features, nan=0.0, posinf=0.0, neginf=0.0),
        "global_stats": np.nan_to_num(global_stats, nan=0.0, posinf=0.0, neginf=0.0),
        "fallback_used": np.float32(0.0),
        "raw_counts": {"total_matches": int(total_matches), "valid_matches": int(valid_count)},
        "reason": "ok",
    }


def packet_to_json_row(sample_id, match_path, packet):
    return {
        "sample_id": str(sample_id),
        "match_path": str(match_path),
        "tokens": np.asarray(packet["tokens"], dtype=np.float32).astype(float).tolist(),
        "token_mask": np.asarray(packet["token_mask"], dtype=np.float32).astype(float).tolist(),
        "token_feature_names": list(TOKEN_FEATURE_NAMES),
        "hypothesis_features": np.asarray(packet["hypothesis_features"], dtype=np.float32).astype(float).tolist(),
        "hypothesis_names": list(HYPOTHESIS_NAMES),
        "hypothesis_feature_names": list(HYPOTHESIS_FEATURE_NAMES),
        "global_stats": np.asarray(packet["global_stats"], dtype=np.float32).astype(float).tolist(),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "fallback_used": bool(float(packet["fallback_used"]) >= 0.5),
        "raw_counts": dict(packet.get("raw_counts", {})),
        "reason": str(packet.get("reason", "")),
    }


def read_manifest_rows(path, split=None):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is not None and row.get("split") != split:
                continue
            if "sample_id" in row and row["sample_id"]:
                rows.append({"sample_id": row["sample_id"], "match_path": row.get("match_path", "")})
    return rows


def build_correspondence_token_manifest(
    records_csv,
    cache_root,
    output_jsonl,
    summary_json=None,
    split=None,
    image_size=(512, 512),
    topk=128,
    grid_size=8,
    residual_threshold=0.035,
):
    rows = read_manifest_rows(records_csv, split=split)
    seen = set()
    output_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        match_path = Path(row["match_path"]) if row.get("match_path") else sample_to_train_match_path(cache_root, sample_id)
        packet = build_correspondence_token_packet(
            match_path,
            image_size=image_size,
            topk=topk,
            grid_size=grid_size,
            residual_threshold=residual_threshold,
        )
        output_rows.append(packet_to_json_row(sample_id, match_path, packet))

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    covered = sum(1 for row in output_rows if not row["fallback_used"])
    valid_counts = [int(row["raw_counts"].get("valid_matches", 0)) for row in output_rows]
    summary = {
        "records_csv": str(Path(records_csv)),
        "cache_root": str(Path(cache_root)),
        "output_jsonl": str(output_jsonl),
        "rows": len(output_rows),
        "covered": int(covered),
        "fallback": int(len(output_rows) - covered),
        "coverage_rate": float(covered / len(output_rows)) if output_rows else 0.0,
        "topk": int(topk),
        "grid_size": int(grid_size),
        "token_feature_names": list(TOKEN_FEATURE_NAMES),
        "hypothesis_names": list(HYPOTHESIS_NAMES),
        "hypothesis_feature_names": list(HYPOTHESIS_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "valid_matches_mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
        "valid_matches_p50": float(np.percentile(valid_counts, 50)) if valid_counts else 0.0,
        "valid_matches_p10": float(np.percentile(valid_counts, 10)) if valid_counts else 0.0,
    }
    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_correspondence_token_manifest(path):
    manifest = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest[str(row["sample_id"])] = row
    return manifest
