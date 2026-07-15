"""Experiment record writer for Phase27 A taxonomy redesign-v3."""
from __future__ import annotations

import json
from pathlib import Path


def write_record(output_dir: str | Path, command: str) -> Path:
    output = Path(output_dir)
    input_manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    boundary = json.loads((output / "metrics" / "no_go_training_policy_boundary.json").read_text(encoding="utf-8"))
    consistency = json.loads((output / "metrics" / "baseline_stress_consistency_audit.json").read_text(encoding="utf-8"))
    hard = json.loads((output / "metrics" / "heading_range_hard_split.json").read_text(encoding="utf-8"))
    join = json.loads((output / "metrics" / "join_coverage_bias_report.json").read_text(encoding="utf-8"))
    text = f"""# exp-20260509 Phase27 A Taxonomy Redesign-v3

Status: `bounded-full-complete`

Remote output:

```text
{output}
```

Command:

```text
{command}
```

## Inputs

```json
{json.dumps(input_manifest, indent=2, sort_keys=True)}
```

## Key Metrics

Heading/range hard split:

```json
{json.dumps(hard, indent=2, sort_keys=True)}
```

Baseline/stress consistency:

```json
{json.dumps(consistency, indent=2, sort_keys=True)}
```

Join coverage:

```json
{json.dumps(join, indent=2, sort_keys=True)}
```

## Verdict

```json
{json.dumps(boundary, indent=2, sort_keys=True)}
```

Training run: `false`

Finetune run: `false`

Submission packaged: `false`

Sample weighting/curriculum/oversampling: `false`
"""
    target = output / "phase27_a_taxonomy_redesign_v3_experiment_record.md"
    target.write_text(text, encoding="utf-8")
    return target
