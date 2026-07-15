import hashlib
import json
import math
import os.path as osp
import re
from pathlib import Path

import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.datasets.pairuav_matcher_features import (
    BSCR_GLOBAL_FEATURE_NAMES,
    BSCR_GRID_SIZE,
    BSCR_TOPK,
    MATCHER_FEATURE_NAMES,
    bscr_tensor_for_sample,
    feature_tensor_for_sample,
    load_bscr_feature_manifest,
    load_feature_manifest,
)
from reloc3r.datasets.utils.transforms import ImgNorm
from reloc3r.utils.image import imread_cv2


PAIR_JSON_PATTERN = re.compile(r"^\d+_\d+\.json$")


def _extract_int(value):
    match = re.search(r"\d+", str(value))
    if match:
        return int(match.group())
    return float("inf")


def _stable_target_index(value, modulo):
    if int(modulo) <= 0:
        raise ValueError("num_target_groups must be positive")
    text = str(value)
    match = re.search(r"\d+", text)
    if match:
        return np.int64(int(match.group()) % int(modulo))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return np.int64(int(digest[:12], 16) % int(modulo))


def _json_sort_key(json_path):
    path = Path(json_path)
    group_value = path.parent.name
    json_value = path.stem
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        group_value = data.get("group_id", group_value)
        json_value = data.get("json_id", json_value)
    except Exception:
        pass
    return (_extract_int(group_value), str(group_value), _extract_int(json_value), str(json_value))


def _iter_json_paths(root):
    return sorted([p for p in Path(root).rglob("*.json") if p.is_file()], key=_json_sort_key)


def _coerce_float(raw_sample, *keys):
    for key in keys:
        if key in raw_sample and raw_sample[key] is not None:
            return float(raw_sample[key])
    raise KeyError(f"Missing required numeric field. Tried keys={keys}")


def _coerce_str(raw_sample, *keys):
    for key in keys:
        if key in raw_sample and raw_sample[key] is not None:
            return str(raw_sample[key])
    raise KeyError(f"Missing required string field. Tried keys={keys}")


def _normalize_record(json_path, require_labels=True, num_target_groups=4096):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        raw_sample = json.load(handle)

    group_id = str(raw_sample.get("group_id", json_path.parent.name))
    json_id = str(raw_sample.get("json_id", json_path.stem))
    record = {
        "group_id": group_id,
        "target_group_index": _stable_target_index(group_id, num_target_groups),
        "scene_id": str(raw_sample.get("scene_id", f"group_{group_id}")),
        "json_id": json_id,
        "json_path": str(json_path),
        "image_a": _coerce_str(raw_sample, "image_a", "image_a_path"),
        "image_b": _coerce_str(raw_sample, "image_b", "image_b_path"),
    }
    if require_labels:
        record["heading_deg"] = _coerce_float(raw_sample, "heading_deg", "heading_num")
        record["range_value"] = _coerce_float(raw_sample, "range_value", "range_num")
    else:
        record["heading_deg"] = float(raw_sample.get("heading_deg", raw_sample.get("heading_num", 0.0)))
        record["range_value"] = float(raw_sample.get("range_value", raw_sample.get("range_num", 0.0)))
    return record


class PairUAV(BaseStereoViewDataset):
    def __init__(
        self,
        json_root,
        image_root,
        split="train",
        resolution=(512, 384),
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        require_labels=True,
        num_target_groups=4096,
        matcher_feature_manifest=None,
        bscr_feature_manifest=None,
    ):
        super().__init__(split=split, resolution=resolution, transform=transform, aug_crop=aug_crop, seed=seed)
        self.json_root = Path(json_root)
        self.image_root = Path(image_root)
        self.require_labels = require_labels
        self.num_target_groups = int(num_target_groups)
        self.matcher_feature_manifest = str(matcher_feature_manifest) if matcher_feature_manifest else None
        self.bscr_feature_manifest = str(bscr_feature_manifest) if bscr_feature_manifest else None
        self.matcher_feature_dim = len(MATCHER_FEATURE_NAMES)
        self.bscr_global_dim = len(BSCR_GLOBAL_FEATURE_NAMES)
        self.matcher_features_by_id = (
            load_feature_manifest(self.matcher_feature_manifest) if self.matcher_feature_manifest else {}
        )
        self.bscr_features_by_id = load_bscr_feature_manifest(self.bscr_feature_manifest) if self.bscr_feature_manifest else {}
        self.samples = [
            _normalize_record(path, require_labels=require_labels, num_target_groups=self.num_target_groups)
            for path in _iter_json_paths(self.json_root)
        ]
        if not self.samples:
            raise FileNotFoundError(f"No PairUAV json files found under {self.json_root}")

    def __len__(self):
        return len(self.samples)

    def get_stats(self):
        return f"{len(self.samples)} pairs"

    def _resolve_image_path(self, relative_path):
        rel = Path(relative_path)
        candidates = [
            self.image_root / rel,
            self.image_root / rel.name,
            self.image_root / rel.parent.name / rel.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve image path {relative_path}; tried {candidates}")

    @staticmethod
    def _make_intrinsics(image):
        height, width = image.shape[:2]
        focal = float(max(width, height))
        return np.array(
            [
                [focal, 0.0, (width - 1.0) * 0.5],
                [0.0, focal, (height - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _matcher_feature_payload(self, sample):
        sample_id = f"{sample['group_id']}/{sample['json_id']}"
        features, mask = feature_tensor_for_sample(
            sample_id,
            self.matcher_features_by_id,
            feature_dim=self.matcher_feature_dim,
        )
        return {
            "sample_id": sample_id,
            "matcher_features": np.asarray(features, dtype=np.float32),
            "matcher_feature_mask": np.float32(mask),
        }

    def _bscr_feature_payload(self, sample):
        if not self.bscr_feature_manifest:
            return {}
        sample_id = f"{sample['group_id']}/{sample['json_id']}"
        packet = bscr_tensor_for_sample(
            sample_id,
            self.bscr_features_by_id,
            grid_size=BSCR_GRID_SIZE,
            topk=BSCR_TOPK,
        )
        return {key: np.asarray(value, dtype=np.float32) for key, value in packet.items()}

    def _build_view(self, sample, image_key, view_suffix, resolution, rng):
        image_path = self._resolve_image_path(sample[image_key])
        color_image = imread_cv2(str(image_path))
        intrinsics = self._make_intrinsics(color_image)
        color_image, intrinsics = self._crop_resize_if_necessary(
            color_image,
            intrinsics,
            resolution,
            rng=rng,
            info=str(image_path),
        )

        heading_deg = np.float32(sample["heading_deg"])
        heading_rad = np.deg2rad(heading_deg)
        heading_cos = np.float32(math.cos(float(heading_rad)))
        heading_sin = np.float32(math.sin(float(heading_rad)))
        range_value = np.float32(sample["range_value"])

        view = dict(
            img=color_image,
            camera_intrinsics=intrinsics.astype(np.float32),
            dataset="PairUAV",
            label=sample["scene_id"],
            instance=f"{sample['json_id']}_{view_suffix}",
            scene_id=sample["scene_id"],
            group_id=sample["group_id"],
            target_group_index=np.int64(sample["target_group_index"]),
            json_path=sample["json_path"],
            heading_deg=heading_deg,
            heading_cos=heading_cos,
            heading_sin=heading_sin,
            range_value=range_value,
            **self._matcher_feature_payload(sample),
        )
        view.update(self._bscr_feature_payload(sample))
        return view

    def _get_views(self, idx, resolution, rng):
        sample = self.samples[idx]
        return [
            self._build_view(sample, "image_a", "a", resolution, rng),
            self._build_view(sample, "image_b", "b", resolution, rng),
        ]
