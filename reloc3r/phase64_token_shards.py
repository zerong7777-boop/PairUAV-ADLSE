import json
import math
from pathlib import Path

import numpy as np


def wrap_deg(values):
    values = np.asarray(values, dtype=np.float32)
    return (values + 180.0) % 360.0 - 180.0


class Phase64TokenShardDataset:
    """Lazy reader for Phase64 token shards.

    The class intentionally has no torch dependency at import time so shard
    format checks can run in no-GPU environments.
    """

    def __init__(self, manifest_path, to_torch=False, preload=False):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shards = list(self.manifest.get("shards", []))
        self.to_torch = bool(to_torch)
        self.preload = bool(preload)
        self._cache_index = None
        self._cache = None
        self._preloaded = None
        offsets = []
        cursor = 0
        for shard in self.shards:
            rows = int(shard["rows"])
            offsets.append((cursor, cursor + rows))
            cursor += rows
        self._offsets = offsets
        self._length = cursor
        if self.preload:
            self._preloaded = [self._read_shard(index) for index in range(len(self.shards))]

    def __len__(self):
        return self._length

    def _locate(self, index):
        index = int(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for shard_index, (start, end) in enumerate(self._offsets):
            if start <= index < end:
                return shard_index, index - start
        raise IndexError(index)

    def _read_shard(self, shard_index):
        shard_path = Path(self.shards[shard_index]["path"])
        data = np.load(shard_path, allow_pickle=True)
        return {key: np.asarray(data[key]) for key in data.files}

    def _load_shard(self, shard_index):
        if self._preloaded is not None:
            return self._preloaded[shard_index]
        if self._cache_index == shard_index and self._cache is not None:
            return self._cache
        data = self._read_shard(shard_index)
        self._cache_index = shard_index
        self._cache = data
        return data

    def _as_torch(self, sample):
        import torch

        output = {}
        for key, value in sample.items():
            if key in ("sample_id", "match_path"):
                output[key] = value
            elif key in ("valid_matches", "total_matches"):
                output[key] = torch.as_tensor(value, dtype=torch.long)
            else:
                output[key] = torch.as_tensor(value, dtype=torch.float32)
        return output

    def __getitem__(self, index):
        shard_index, local_index = self._locate(index)
        data = self._load_shard(shard_index)
        target_heading = np.float32(data["target_heading"][local_index])
        rank1_heading = np.float32(data["rank1_heading"][local_index])
        residual_target = np.float32(wrap_deg(target_heading - rank1_heading))
        sample = {
            "sample_id": str(data["sample_id"][local_index]),
            "target_heading": target_heading,
            "target_distance": np.float32(data["target_distance"][local_index]),
            "rank1_heading": rank1_heading,
            "rank1_distance": np.float32(data["rank1_distance"][local_index]),
            "rank1_angle_abs_error": np.float32(data["rank1_angle_abs_error"][local_index]),
            "tokens": np.asarray(data["tokens"][local_index], dtype=np.float32),
            "token_mask": np.asarray(data["token_mask"][local_index], dtype=np.float32),
            "hypothesis_features": np.asarray(data["hypothesis_features"][local_index], dtype=np.float32),
            "global_stats": np.asarray(data["global_stats"][local_index], dtype=np.float32),
            "fallback_used": np.float32(data["fallback_used"][local_index]),
            "valid_matches": np.int64(data["valid_matches"][local_index]),
            "total_matches": np.int64(data["total_matches"][local_index]),
            "match_path": str(data["match_path"][local_index]),
            "residual_target": residual_target,
        }
        if self.to_torch:
            return self._as_torch(sample)
        return sample

    def batch_numpy(self, indices):
        samples = [self[index] for index in indices]
        keys = [
            "target_heading",
            "target_distance",
            "rank1_heading",
            "rank1_distance",
            "rank1_angle_abs_error",
            "tokens",
            "token_mask",
            "hypothesis_features",
            "global_stats",
            "fallback_used",
            "valid_matches",
            "total_matches",
            "residual_target",
        ]
        batch = {"sample_id": [sample["sample_id"] for sample in samples], "match_path": [sample["match_path"] for sample in samples]}
        for key in keys:
            batch[key] = np.stack([np.asarray(sample[key]) for sample in samples], axis=0)
        return batch


def validate_manifest(manifest_path):
    dataset = Phase64TokenShardDataset(manifest_path)
    if len(dataset) <= 0:
        raise ValueError("Phase64 shard dataset is empty")
    first = dataset[0]
    required = {
        "tokens",
        "token_mask",
        "hypothesis_features",
        "global_stats",
        "rank1_heading",
        "target_heading",
        "residual_target",
    }
    missing = required.difference(first)
    if missing:
        raise ValueError(f"Missing sample keys: {sorted(missing)}")
    return {
        "rows": len(dataset),
        "tokens_shape": list(first["tokens"].shape),
        "token_mask_shape": list(first["token_mask"].shape),
        "hypothesis_shape": list(first["hypothesis_features"].shape),
        "global_shape": list(first["global_stats"].shape),
        "first_sample_id": first["sample_id"],
        "first_residual_target": float(first["residual_target"]),
    }
