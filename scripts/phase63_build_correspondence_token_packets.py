#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_builder():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "reloc3r" / "datasets" / "pairuav_correspondence_tokens.py"
    spec = importlib.util.spec_from_file_location("pairuav_correspondence_tokens_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_correspondence_token_manifest


def main():
    build_correspondence_token_manifest = _load_builder()
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--residual-threshold", type=float, default=0.035)
    args = parser.parse_args()

    summary = build_correspondence_token_manifest(
        records_csv=args.records_csv,
        cache_root=args.cache_root,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        split=args.split,
        image_size=(args.image_width, args.image_height),
        topk=args.topk,
        grid_size=args.grid_size,
        residual_threshold=args.residual_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
