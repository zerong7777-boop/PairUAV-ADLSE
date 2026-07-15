"""Lightweight artifact registry for Phase27 A validation spine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_ARTIFACT_FIELDS = (
    "artifact_id",
    "artifact_kind",
    "path",
    "storage_location",
    "schema_version",
    "key_schema_version",
    "row_count",
    "columns_hash",
    "source_artifacts",
    "generated_by",
    "generated_at",
    "read_only",
    "notes",
)


def compute_columns_hash(columns):
    payload = "\n".join(str(column) for column in sorted(columns))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def artifact_entry(
    artifact_id,
    artifact_kind,
    path,
    schema_version,
    key_schema_version,
    row_count,
    columns,
    source_artifacts,
    generated_by,
    read_only,
    notes="",
):
    path_text = str(path)
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "path": path_text,
        "storage_location": "remote" if path_text.startswith("/media/") else "local_or_relative",
        "schema_version": schema_version,
        "key_schema_version": key_schema_version,
        "row_count": int(row_count),
        "columns_hash": compute_columns_hash(columns),
        "source_artifacts": list(source_artifacts),
        "generated_by": generated_by,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": bool(read_only),
        "notes": notes,
    }


def load_registry(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_registry(path, registry):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False, sort_keys=True)


def validate_registry(registry):
    missing_fields = []
    missing_source_artifacts = []
    for artifact_id, entry in registry.items():
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in entry:
                missing_fields.append(f"{artifact_id}.{field}")
            elif field not in {"notes", "source_artifacts"} and entry[field] in ("", None):
                missing_fields.append(f"{artifact_id}.{field}")
        for source in entry.get("source_artifacts", []):
            if source not in registry:
                missing_source_artifacts.append(f"{artifact_id} -> {source}")
    return {
        "passed": not missing_fields and not missing_source_artifacts,
        "missing_fields": missing_fields,
        "missing_source_artifacts": missing_source_artifacts,
        "artifact_count": len(registry),
    }
