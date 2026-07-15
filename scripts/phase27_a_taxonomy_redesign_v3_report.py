"""Report writer for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_report(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    metrics = output / "metrics"
    manifests = output / "manifests"
    report_dir = output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = sum(1 for _ in (manifests / "non_leaking_candidate_manifest.csv").open(encoding="utf-8")) - 1
    outcome_rows = sum(1 for _ in (manifests / "validation_only_outcome_attribution_manifest.csv").open(encoding="utf-8")) - 1
    readiness_rows = sum(1 for _ in (manifests / "training_readiness_verdict_manifest.csv").open(encoding="utf-8")) - 1

    ambiguity = _load_json(metrics / "ambiguity_subtype_breakdown.json")
    hard = _load_json(metrics / "heading_range_hard_split.json")
    consistency = _load_json(metrics / "baseline_stress_consistency_audit.json")
    join = _load_json(metrics / "join_coverage_bias_report.json")
    leakage = _load_json(metrics / "leakage_deployability_audit.json")
    boundary = _load_json(metrics / "no_go_training_policy_boundary.json")

    text = f"""# Phase27 A Taxonomy Redesign-v3 Report

Status: `bounded-full-complete`

## Row Counts

- candidate rows: `{candidate_rows}`
- outcome rows: `{outcome_rows}`
- readiness rows: `{readiness_rows}`

## Layer Meaning

- Layer 1 is non-leaking candidate discovery.
- Layer 2 is validation-only outcome attribution.
- Layer 3 is multi-label training-readiness verdict.

## Ambiguity Subtypes

```json
{json.dumps(ambiguity, indent=2, sort_keys=True)}
```

## Heading/Range Hard Split

```json
{json.dumps(hard, indent=2, sort_keys=True)}
```

## Baseline vs Stress Consistency

```json
{json.dumps(consistency, indent=2, sort_keys=True)}
```

## Join Coverage And Bias

```json
{json.dumps(join, indent=2, sort_keys=True)}
```

## Leakage And Deployability

```json
{json.dumps({'passed': leakage.get('passed'), 'violations': leakage.get('violations')}, indent=2, sort_keys=True)}
```

## B/C Boundary

B and C must not consume A-v2 final states. B may only use future approved factorized combinations such as heading-hard + evidence-sufficient + semantic-geometric-conflict + not-low-observable. C may only aggregate v3 factorized axes, not single final states.

## Go/No-Go

```json
{json.dumps(boundary, indent=2, sort_keys=True)}
```

No training, finetuning, sample weighting, curriculum, oversampling, submission packaging, or B/C gate training was run.
"""
    target = report_dir / "phase27_a_taxonomy_redesign_v3_report.md"
    target.write_text(text, encoding="utf-8")
    return target
