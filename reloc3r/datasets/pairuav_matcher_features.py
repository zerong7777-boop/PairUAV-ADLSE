import csv
import json
import math
from pathlib import Path

import numpy as np


MATCHER_FEATURE_NAMES = [
    "log1p_match_count",
    "mean_confidence",
    "max_confidence",
    "std_confidence",
    "mean_dx",
    "mean_dy",
    "std_dx",
    "std_dy",
    "mean_disp_norm",
    "max_disp_norm",
    "std_disp_norm",
    "valid_ratio",
    "fallback_used",
]

BSCR_GLOBAL_FEATURE_NAMES = [
    "log1p_match_count",
    "mean_confidence",
    "max_confidence",
    "std_confidence",
    "mean_dx",
    "mean_dy",
    "std_dx",
    "std_dy",
    "mean_disp_norm",
    "std_disp_norm",
    "spatial_entropy",
    "fallback_used",
]

BSCR_GRID_SIZE = 4
BSCR_TOPK = 16
BSCR_SPATIAL_CHANNELS = 4
BSCR_ANCHOR_DIM = 5


def sample_to_match_path(cache_root, sample_id):
    group, pair_id = str(sample_id).split("/", 1)
    left, right = pair_id.split("_", 1)
    return Path(cache_root) / group / f"image-{left}_image-{right}_matches.npz"


def _as_array(record, key, dtype=np.float32):
    if isinstance(record, np.lib.npyio.NpzFile):
        value = record[key]
    else:
        value = record[key]
    return np.asarray(value, dtype=dtype)


def _finite(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _zero_features(fallback_used):
    values = {name: 0.0 for name in MATCHER_FEATURE_NAMES}
    values["fallback_used"] = 1.0 if fallback_used else 0.0
    return values


def extract_matcher_features(record_or_npz, image_width=None, image_height=None):
    try:
        if isinstance(record_or_npz, (str, Path)):
            path = Path(record_or_npz)
            if not path.is_file():
                return _zero_features(fallback_used=True)
            with np.load(path, allow_pickle=True) as data:
                return extract_matcher_features(data, image_width=image_width, image_height=image_height)

        keypoints0 = _as_array(record_or_npz, "keypoints0", dtype=np.float32)
        keypoints1 = _as_array(record_or_npz, "keypoints1", dtype=np.float32)
        matches = _as_array(record_or_npz, "matches", dtype=np.int64).reshape(-1)
        confidence = _as_array(record_or_npz, "match_confidence", dtype=np.float32).reshape(-1)
    except Exception:
        return _zero_features(fallback_used=True)

    if keypoints0.ndim != 2 or keypoints1.ndim != 2 or keypoints0.shape[-1] < 2 or keypoints1.shape[-1] < 2:
        return _zero_features(fallback_used=True)

    total = int(matches.shape[0])
    if total <= 0:
        return _zero_features(fallback_used=False)

    valid = (matches >= 0) & (matches < len(keypoints1)) & np.isfinite(confidence) & (confidence > 0)
    count = int(valid.sum())
    if count <= 0:
        values = _zero_features(fallback_used=False)
        values["valid_ratio"] = 0.0
        return values

    src = keypoints0[: len(matches)][valid, :2]
    dst = keypoints1[matches[valid], :2]
    conf = _finite(confidence[valid])
    disp = dst - src

    width = float(image_width) if image_width else float(max(np.nanmax(keypoints0[:, 0]), np.nanmax(keypoints1[:, 0]), 1.0))
    height = float(image_height) if image_height else float(max(np.nanmax(keypoints0[:, 1]), np.nanmax(keypoints1[:, 1]), 1.0))
    width = max(width, 1.0)
    height = max(height, 1.0)

    dx = _finite(disp[:, 0] / width)
    dy = _finite(disp[:, 1] / height)
    mag = _finite(np.sqrt(np.square(disp[:, 0] / width) + np.square(disp[:, 1] / height)))

    values = _zero_features(fallback_used=False)
    values.update(
        {
            "log1p_match_count": float(math.log1p(count)),
            "mean_confidence": float(conf.mean()) if conf.size else 0.0,
            "max_confidence": float(conf.max()) if conf.size else 0.0,
            "std_confidence": float(conf.std()) if conf.size else 0.0,
            "mean_dx": float(dx.mean()) if dx.size else 0.0,
            "mean_dy": float(dy.mean()) if dy.size else 0.0,
            "std_dx": float(dx.std()) if dx.size else 0.0,
            "std_dy": float(dy.std()) if dy.size else 0.0,
            "mean_disp_norm": float(mag.mean()) if mag.size else 0.0,
            "max_disp_norm": float(mag.max()) if mag.size else 0.0,
            "std_disp_norm": float(mag.std()) if mag.size else 0.0,
            "valid_ratio": float(count / total),
            "fallback_used": 0.0,
        }
    )
    return {name: float(values[name]) for name in MATCHER_FEATURE_NAMES}


def compute_normalization(rows):
    matrix = np.asarray([[float(row["raw_features"][name]) for name in MATCHER_FEATURE_NAMES] for row in rows], dtype=np.float32)
    if matrix.size == 0:
        mean = np.zeros(len(MATCHER_FEATURE_NAMES), dtype=np.float32)
        std = np.ones(len(MATCHER_FEATURE_NAMES), dtype=np.float32)
    else:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-6] = 1.0
    return {
        "feature_names": list(MATCHER_FEATURE_NAMES),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def apply_normalization(feature_dict, stats):
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    raw = np.asarray([float(feature_dict.get(name, 0.0)) for name in MATCHER_FEATURE_NAMES], dtype=np.float32)
    normalized = (raw - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(float).tolist()


def load_feature_manifest(path):
    manifest = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest[str(row["sample_id"])] = row
    return manifest


def feature_tensor_for_sample(sample_id, manifest, feature_dim=None):
    feature_dim = int(feature_dim or len(MATCHER_FEATURE_NAMES))
    row = manifest.get(str(sample_id))
    if row is None:
        return [0.0] * feature_dim, 0.0
    values = row.get("features", [])
    if len(values) != feature_dim:
        values = (list(values) + [0.0] * feature_dim)[:feature_dim]
    mask = 0.0 if bool(row.get("fallback_used", False)) else 1.0
    return [float(v) for v in values], float(mask)


def read_manifest_rows(path, split=None):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is not None and row.get("split") != split:
                continue
            if "sample_id" in row and row["sample_id"]:
                rows.append({"sample_id": row["sample_id"]})
    return rows


def _empty_bscr_packet(fallback_used=True, grid_size=BSCR_GRID_SIZE, topk=BSCR_TOPK):
    return {
        "global_stats": np.zeros(len(BSCR_GLOBAL_FEATURE_NAMES), dtype=np.float32),
        "spatial_bins": np.zeros((int(grid_size), int(grid_size), BSCR_SPATIAL_CHANNELS), dtype=np.float32),
        "topk_anchors": np.zeros((int(topk), BSCR_ANCHOR_DIM), dtype=np.float32),
        "quality_mask": np.asarray([0.0], dtype=np.float32),
        "fallback_used": np.float32(1.0 if fallback_used else 0.0),
        "raw_global_stats": {name: 0.0 for name in BSCR_GLOBAL_FEATURE_NAMES},
    }


def _valid_match_arrays(record_or_npz):
    keypoints0 = _as_array(record_or_npz, "keypoints0", dtype=np.float32)
    keypoints1 = _as_array(record_or_npz, "keypoints1", dtype=np.float32)
    matches = _as_array(record_or_npz, "matches", dtype=np.int64).reshape(-1)
    confidence = _as_array(record_or_npz, "match_confidence", dtype=np.float32).reshape(-1)
    if keypoints0.ndim != 2 or keypoints1.ndim != 2 or keypoints0.shape[-1] < 2 or keypoints1.shape[-1] < 2:
        raise ValueError("Matcher packet has invalid keypoint arrays")
    valid = (matches >= 0) & (matches < len(keypoints1)) & np.isfinite(confidence) & (confidence > 0)
    src = keypoints0[: len(matches)][valid, :2].astype(np.float32, copy=False)
    dst = keypoints1[matches[valid], :2].astype(np.float32, copy=False)
    conf = confidence[valid].astype(np.float32, copy=False)
    return src, dst, conf, int(matches.shape[0])


def _spatial_entropy(counts):
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = np.asarray(counts, dtype=np.float32).reshape(-1) / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    denom = math.log(float(len(counts.reshape(-1)))) if counts.size > 1 else 1.0
    return float(-(probs * np.log(probs)).sum() / max(denom, 1e-6))


def _normalize_xy(points, width, height):
    denom = np.asarray([max(float(width), 1.0), max(float(height), 1.0)], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) / denom


def extract_bscr_packet(record_or_npz, image_size=(512, 512), grid_size=BSCR_GRID_SIZE, topk=BSCR_TOPK):
    """Build structured correspondence evidence without reading labels."""
    try:
        if isinstance(record_or_npz, (str, Path)):
            path = Path(record_or_npz)
            if not path.is_file():
                return _empty_bscr_packet(fallback_used=True, grid_size=grid_size, topk=topk)
            with np.load(path, allow_pickle=True) as data:
                return extract_bscr_packet(data, image_size=image_size, grid_size=grid_size, topk=topk)
        src, dst, conf, total_matches = _valid_match_arrays(record_or_npz)
    except Exception:
        return _empty_bscr_packet(fallback_used=True, grid_size=grid_size, topk=topk)

    grid_size = int(grid_size)
    topk = int(topk)
    width, height = float(image_size[0]), float(image_size[1])
    count = int(conf.shape[0])
    if count <= 0 or total_matches <= 0:
        packet = _empty_bscr_packet(fallback_used=False, grid_size=grid_size, topk=topk)
        packet["quality_mask"] = np.asarray([0.0], dtype=np.float32)
        return packet

    src_norm = _normalize_xy(src, width, height)
    dst_norm = _normalize_xy(dst, width, height)
    disp = dst_norm - src_norm
    mag = np.sqrt(np.square(disp[:, 0]) + np.square(disp[:, 1]))

    xbin = np.clip((src_norm[:, 0] * grid_size).astype(np.int64), 0, grid_size - 1)
    ybin = np.clip((src_norm[:, 1] * grid_size).astype(np.int64), 0, grid_size - 1)
    spatial = np.zeros((grid_size, grid_size, BSCR_SPATIAL_CHANNELS), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.float32)
    for idx in range(count):
        y, x = int(ybin[idx]), int(xbin[idx])
        counts[y, x] += 1.0
        spatial[y, x, 1] += float(conf[idx])
        spatial[y, x, 2] += float(disp[idx, 0])
        spatial[y, x, 3] += float(disp[idx, 1])
    nonzero = counts > 0
    spatial[:, :, 0] = np.log1p(counts)
    for channel in (1, 2, 3):
        values = spatial[:, :, channel]
        values[nonzero] = values[nonzero] / counts[nonzero]
        spatial[:, :, channel] = values

    order = np.argsort(-conf)[:topk]
    anchors = np.zeros((topk, BSCR_ANCHOR_DIM), dtype=np.float32)
    if order.size:
        anchors[: order.size, 0:2] = src_norm[order]
        anchors[: order.size, 2:4] = dst_norm[order]
        anchors[: order.size, 4] = conf[order]

    raw = {
        "log1p_match_count": float(math.log1p(count)),
        "mean_confidence": float(conf.mean()),
        "max_confidence": float(conf.max()),
        "std_confidence": float(conf.std()),
        "mean_dx": float(disp[:, 0].mean()),
        "mean_dy": float(disp[:, 1].mean()),
        "std_dx": float(disp[:, 0].std()),
        "std_dy": float(disp[:, 1].std()),
        "mean_disp_norm": float(mag.mean()),
        "std_disp_norm": float(mag.std()),
        "spatial_entropy": _spatial_entropy(counts),
        "fallback_used": 0.0,
    }
    global_stats = np.asarray([raw[name] for name in BSCR_GLOBAL_FEATURE_NAMES], dtype=np.float32)
    global_stats = np.nan_to_num(global_stats, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "global_stats": global_stats,
        "spatial_bins": np.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "topk_anchors": np.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "quality_mask": np.asarray([1.0], dtype=np.float32),
        "fallback_used": np.float32(0.0),
        "raw_global_stats": raw,
    }


def _bscr_row(sample_id, match_path, image_size=(512, 512), grid_size=BSCR_GRID_SIZE, topk=BSCR_TOPK):
    packet = extract_bscr_packet(match_path, image_size=image_size, grid_size=grid_size, topk=topk)
    return {
        "sample_id": str(sample_id),
        "match_path": str(match_path),
        "global_stats": packet["global_stats"].astype(float).tolist(),
        "spatial_bins": packet["spatial_bins"].astype(float).tolist(),
        "topk_anchors": packet["topk_anchors"].astype(float).tolist(),
        "quality_mask": packet["quality_mask"].astype(float).tolist(),
        "fallback_used": bool(float(packet["fallback_used"]) >= 0.5),
        "raw_global_stats": packet["raw_global_stats"],
    }


def build_bscr_feature_manifest(
    records_csv,
    cache_root,
    output_jsonl,
    stats_json=None,
    split=None,
    image_size=(512, 512),
    grid_size=BSCR_GRID_SIZE,
    topk=BSCR_TOPK,
):
    """Write structured B-SCR packets keyed by sample_id."""
    rows = read_manifest_rows(records_csv, split=split)
    seen = set()
    output_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        output_rows.append(
            _bscr_row(
                sample_id,
                sample_to_match_path(cache_root, sample_id),
                image_size=image_size,
                grid_size=grid_size,
                topk=topk,
            )
        )

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    covered = sum(1 for row in output_rows if not row["fallback_used"])
    summary = {
        "records_csv": str(Path(records_csv)),
        "cache_root": str(Path(cache_root)),
        "output_jsonl": str(output_jsonl),
        "rows": len(output_rows),
        "covered": covered,
        "fallback": len(output_rows) - covered,
        "coverage_rate": covered / len(output_rows) if output_rows else 0.0,
        "global_feature_names": list(BSCR_GLOBAL_FEATURE_NAMES),
        "grid_size": int(grid_size),
        "topk": int(topk),
        "format": "jsonl",
    }
    if stats_json:
        stats_path = Path(stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_bscr_feature_manifest(path):
    manifest = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest[str(row["sample_id"])] = row
    return manifest


def bscr_tensor_for_sample(sample_id, manifest, grid_size=BSCR_GRID_SIZE, topk=BSCR_TOPK):
    row = manifest.get(str(sample_id))
    if row is None:
        packet = _empty_bscr_packet(fallback_used=True, grid_size=grid_size, topk=topk)
        return {
            "bscr_global_stats": packet["global_stats"],
            "bscr_spatial_bins": packet["spatial_bins"],
            "bscr_topk_anchors": packet["topk_anchors"],
            "bscr_quality_mask": packet["quality_mask"],
            "bscr_fallback_used": np.asarray([1.0], dtype=np.float32),
        }
    return {
        "bscr_global_stats": np.asarray(row.get("global_stats", []), dtype=np.float32).reshape(len(BSCR_GLOBAL_FEATURE_NAMES)),
        "bscr_spatial_bins": np.asarray(row.get("spatial_bins", []), dtype=np.float32).reshape(
            int(grid_size), int(grid_size), BSCR_SPATIAL_CHANNELS
        ),
        "bscr_topk_anchors": np.asarray(row.get("topk_anchors", []), dtype=np.float32).reshape(int(topk), BSCR_ANCHOR_DIM),
        "bscr_quality_mask": np.asarray(row.get("quality_mask", [0.0]), dtype=np.float32).reshape(1),
        "bscr_fallback_used": np.asarray([1.0 if bool(row.get("fallback_used", False)) else 0.0], dtype=np.float32),
    }
