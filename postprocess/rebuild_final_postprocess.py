#!/usr/bin/env python3
"""Rebuild the phase104j final PairUAV postprocess package.

This script reconstructs the final submitted package from the public
postprocess manifest:

1. Build the e230 range stack:
   angle = HR angle
   distance = 0.511 * HR distance + 0.189 * H8 distance + 0.300 * epoch2 distance
2. Snap heading to a 2-degree lattice, normalized to [-180, 180).
3. Snap distance to the nearest support distance observed in the train/dev
   manifest's gt_distance column.

It writes deterministic one-file zip packages and verifies the generated hashes.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_MANIFEST = Path(__file__).with_name("phase104j_final_postprocess_manifest.json")
ENTRY_NAME = "result.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_single_zip_entry(path: Path, entry_name: str = ENTRY_NAME) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if names != [entry_name]:
            raise ValueError(f"{path} must contain exactly {entry_name!r}; got {names!r}")
        return zf.read(entry_name)


def iter_prediction_rows(path: Path, entry_name: str = ENTRY_NAME) -> Iterator[tuple[float, float]]:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if names != [entry_name]:
            raise ValueError(f"{path} must contain exactly {entry_name!r}; got {names!r}")
        with zf.open(entry_name, "r") as f:
            for line_no, raw in enumerate(f, 1):
                parts = raw.split()
                if len(parts) != 2:
                    raise ValueError(f"{path}:{line_no} expected 2 columns, got {raw!r}")
                yield float(parts[0]), float(parts[1])


def count_zip_rows(path: Path, entry_name: str = ENTRY_NAME) -> int:
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(entry_name, "r") as f:
            return sum(1 for _ in f)


def load_distance_support(path: Path, column: str) -> tuple[list[float], int]:
    values: set[float] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no column {column!r}; columns={reader.fieldnames!r}")
        for row in reader:
            rows += 1
            values.add(float(row[column]))
    return sorted(values), rows


def nearest(sorted_values: Sequence[float], value: float) -> float:
    if not sorted_values:
        raise ValueError("empty support set")
    pos = bisect.bisect_left(sorted_values, value)
    if pos <= 0:
        return sorted_values[0]
    if pos >= len(sorted_values):
        return sorted_values[-1]
    left = sorted_values[pos - 1]
    right = sorted_values[pos]
    if abs(right - value) < abs(value - left):
        return right
    return left


def snap_heading(angle: float, lattice_deg: float) -> float:
    snapped = round(angle / lattice_deg) * lattice_deg
    while snapped < -180.0:
        snapped += 360.0
    while snapped >= 180.0:
        snapped -= 360.0
    # Avoid printing "-0.000000".
    if abs(snapped) < 0.5e-6:
        snapped = 0.0
    return snapped


def build_e230_bytes(
    hr_zip: Path,
    h8_zip: Path,
    epoch2_zip: Path,
    weights: dict,
) -> tuple[bytes, int]:
    out = bytearray()
    rows = 0
    sentinel = object()
    for rows, (hr, h8, e2) in enumerate(
        itertools.zip_longest(
            iter_prediction_rows(hr_zip),
            iter_prediction_rows(h8_zip),
            iter_prediction_rows(epoch2_zip),
            fillvalue=sentinel,
        ),
        1,
    ):
        if hr is sentinel or h8 is sentinel or e2 is sentinel:
            raise ValueError(
                "input result row counts differ: "
                f"first mismatch at combined row {rows}"
            )
        hr_angle, hr_dist = hr
        _, h8_dist = h8
        _, e2_dist = e2
        dist = weights["hr"] * hr_dist + weights["h8"] * h8_dist + weights["epoch2"] * e2_dist
        out.extend(f"{hr_angle:.6f} {dist:.6f}\n".encode("ascii"))
    return bytes(out), rows


def build_final_bytes_from_e230(
    e230_bytes: bytes,
    distance_support: Sequence[float],
    heading_lattice_deg: float,
) -> tuple[bytes, int]:
    out = bytearray()
    rows = 0
    for rows, raw in enumerate(e230_bytes.splitlines(), 1):
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"e230 row {rows} expected 2 columns, got {raw!r}")
        angle = snap_heading(float(parts[0]), heading_lattice_deg)
        dist = nearest(distance_support, float(parts[1]))
        out.extend(f"{angle:.6f} {dist:.6f}\n".encode("ascii"))
    return bytes(out), rows


def write_result_zip(
    path: Path,
    result_bytes: bytes,
    date_time: Sequence[int],
    create_system: int,
    external_attr: int,
    compresslevel: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(ENTRY_NAME, tuple(date_time))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = create_system
    info.external_attr = external_attr
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        zf.writestr(info, result_bytes)


def verify_equal(label: str, actual: str, expected: str | None) -> bool:
    if expected is None:
        print(f"{label}: {actual}")
        return True
    ok = actual.lower() == expected.lower()
    status = "OK" if ok else "MISMATCH"
    print(f"{label}: {actual} expected={expected} {status}")
    return ok


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--hr-zip", type=Path, required=True)
    parser.add_argument("--h8-zip", type=Path, required=True)
    parser.add_argument("--epoch2-zip", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--e230-output-zip", type=Path, required=True)
    parser.add_argument("--final-output-zip", type=Path, required=True)
    parser.add_argument("--known-e230-zip", type=Path)
    parser.add_argument("--known-final-zip", type=Path)
    parser.add_argument("--heading-lattice-deg", type=float, default=2.0)
    parser.add_argument(
        "--compresslevel",
        type=int,
        default=None,
        help="zip deflate level. Omit to use Python/zlib default, matching the original packages.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    manifest = load_json(args.manifest)
    expected = manifest.get("expected", {})
    support_cfg = manifest["support"]
    zip_cfg = manifest["zip_metadata"]
    weights = manifest["weights"]

    ok = True
    input_specs = [
        ("hr", args.hr_zip),
        ("h8", args.h8_zip),
        ("epoch2", args.epoch2_zip),
    ]
    for name, path in input_specs:
        ok &= verify_equal(
            f"{name}_zip_sha256",
            sha256_file(path),
            expected.get(f"{name}_zip_sha256"),
        )
        ok &= verify_equal(
            f"{name}_result_txt_sha256",
            sha256_bytes(read_single_zip_entry(path)),
            expected.get(f"{name}_result_txt_sha256"),
        )

    ok &= verify_equal(
        "support_manifest_sha256",
        sha256_file(args.support_manifest),
        expected.get("support_manifest_sha256"),
    )

    distance_support, support_rows = load_distance_support(
        args.support_manifest, support_cfg["distance_column"]
    )
    print(f"support_rows: {support_rows}")
    print(f"support_distance_count: {len(distance_support)}")
    if support_rows != support_cfg.get("expected_support_rows"):
        print(f"support_rows mismatch: expected {support_cfg.get('expected_support_rows')}", file=sys.stderr)
        ok = False
    if len(distance_support) != support_cfg.get("expected_support_distance_count"):
        print(
            "support_distance_count mismatch: "
            f"expected {support_cfg.get('expected_support_distance_count')}",
            file=sys.stderr,
        )
        ok = False

    e230_bytes, e230_rows = build_e230_bytes(args.hr_zip, args.h8_zip, args.epoch2_zip, weights)
    final_bytes, final_rows = build_final_bytes_from_e230(
        e230_bytes, distance_support, args.heading_lattice_deg
    )
    print(f"e230_rows: {e230_rows}")
    print(f"final_rows: {final_rows}")
    if e230_rows != expected.get("row_count") or final_rows != expected.get("row_count"):
        print(f"row_count mismatch: expected {expected.get('row_count')}", file=sys.stderr)
        ok = False

    ok &= verify_equal("e230_result_txt_sha256", sha256_bytes(e230_bytes), expected.get("e230_result_txt_sha256"))
    ok &= verify_equal(
        "final_result_txt_sha256", sha256_bytes(final_bytes), expected.get("final_result_txt_sha256")
    )

    write_result_zip(
        args.e230_output_zip,
        e230_bytes,
        zip_cfg["e230_date_time"],
        int(zip_cfg["create_system"]),
        int(zip_cfg["external_attr"]),
        args.compresslevel,
    )
    write_result_zip(
        args.final_output_zip,
        final_bytes,
        zip_cfg["final_date_time"],
        int(zip_cfg["create_system"]),
        int(zip_cfg["external_attr"]),
        args.compresslevel,
    )

    ok &= verify_equal(
        "e230_result_zip_sha256",
        sha256_file(args.e230_output_zip),
        expected.get("e230_result_zip_sha256"),
    )
    ok &= verify_equal(
        "final_result_zip_sha256",
        sha256_file(args.final_output_zip),
        expected.get("final_result_zip_sha256"),
    )

    if args.known_e230_zip:
        ok &= verify_equal("known_e230_zip_sha256", sha256_file(args.known_e230_zip), expected.get("e230_result_zip_sha256"))
        known_e230_txt = read_single_zip_entry(args.known_e230_zip)
        ok &= verify_equal("known_e230_txt_sha256", sha256_bytes(known_e230_txt), expected.get("e230_result_txt_sha256"))
        print(f"known_e230_txt_equal: {known_e230_txt == e230_bytes}")

    if args.known_final_zip:
        ok &= verify_equal(
            "known_final_zip_sha256", sha256_file(args.known_final_zip), expected.get("final_result_zip_sha256")
        )
        known_final_txt = read_single_zip_entry(args.known_final_zip)
        ok &= verify_equal("known_final_txt_sha256", sha256_bytes(known_final_txt), expected.get("final_result_txt_sha256"))
        print(f"known_final_txt_equal: {known_final_txt == final_bytes}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
