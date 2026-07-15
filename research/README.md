# Research Diagnostics

This directory contains archived diagnostic scripts used during PACE method
development. They are retained for provenance and paper evidence, not as the
primary reproduction path.

Use the public release paths first:

- exact final-package rebuild: `docs/REPRODUCTION.md`
- method and implementation map: `docs/METHOD.md`
- mechanism evidence: `docs/MECHANISM_INSIGHTS.md`
- source-adaptation evidence: `docs/SOURCE_ADAPTATION.md`

`diagnostics/` contains phase-labeled scripts for checkpoint trajectory,
representation, head/readout, and late-stage policy analysis. Their file names
preserve internal run labels so that historical notes and generated artifacts
remain traceable.

The archived `phase91/` and `phase92/` Python packages remain at the repository
root because their historical tests and commands import them by package name.
