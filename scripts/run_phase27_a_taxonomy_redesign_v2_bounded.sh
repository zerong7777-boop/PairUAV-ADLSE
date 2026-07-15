#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

PROJECT_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM
REPO_ROOT=/media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav
OUT_DIR=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_taxonomy_redesign_v2
DOCS_AI_SUMMARY=/media/jgzn/SSD_lexar/RZ/UAVM/docs/ai/phase27_a_taxonomy_redesign_v2_summary.md

EVIDENCE=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_evidence_state_manifest/manifests/a_evidence_state_manifest_v3_calibrated_v2.csv
FULL_DEV=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_full_dev_joinable_baseline_surface.csv
STRESS_221=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_stress64030429_22181448_joinable_baseline_surface.csv
STRESS_945=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_stress64030429_94572967_joinable_baseline_surface.csv
STRESS_995=/media/jgzn/SSD_lexar/RZ/UAVM/experiments/phase27_a_validation_spine/baseline_surfaces/phase21_stress64030429_99516045_joinable_baseline_surface.csv

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_manifest.csv
SOURCE_REGISTRY=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_source_registry.json
LEAKAGE_AUDIT=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_leakage_audit.json
METRICS_DIR=${OUT_DIR}/metrics
REPORT=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_report.md
RECORD=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_record.md
COMMAND_LOG=${OUT_DIR}/phase27_a_taxonomy_redesign_v2_commands.txt

mkdir -p "${OUT_DIR}" "${METRICS_DIR}" "$(dirname "${DOCS_AI_SUMMARY}")"
: > "${COMMAND_LOG}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"

record_command() {
  printf '%s\n' "$*" >> "${COMMAND_LOG}"
}

record_command "${PYTHON} scripts/phase27_a_taxonomy_redesign_v2_manifest.py --evidence-manifest ${EVIDENCE} --full-dev-surface ${FULL_DEV} --stress-surface ${STRESS_221} --stress-surface ${STRESS_945} --stress-surface ${STRESS_995} --out-manifest ${MANIFEST} --out-source-registry ${SOURCE_REGISTRY} --out-leakage-audit ${LEAKAGE_AUDIT}"
"${PYTHON}" scripts/phase27_a_taxonomy_redesign_v2_manifest.py \
  --evidence-manifest "${EVIDENCE}" \
  --full-dev-surface "${FULL_DEV}" \
  --stress-surface "${STRESS_221}" \
  --stress-surface "${STRESS_945}" \
  --stress-surface "${STRESS_995}" \
  --out-manifest "${MANIFEST}" \
  --out-source-registry "${SOURCE_REGISTRY}" \
  --out-leakage-audit "${LEAKAGE_AUDIT}"

record_command "${PYTHON} scripts/phase27_a_taxonomy_redesign_v2_metrics.py --manifest ${MANIFEST} --out-dir ${METRICS_DIR} --bootstrap-iters 1000 --seed 20260509"
"${PYTHON}" scripts/phase27_a_taxonomy_redesign_v2_metrics.py \
  --manifest "${MANIFEST}" \
  --out-dir "${METRICS_DIR}" \
  --bootstrap-iters 1000 \
  --seed 20260509

record_command "${PYTHON} ${SCRIPT_DIR}/phase27_a_taxonomy_redesign_v2_report.py --evidence-manifest ${EVIDENCE} --full-dev-surface ${FULL_DEV} --stress-surface ${STRESS_221} --stress-surface ${STRESS_945} --stress-surface ${STRESS_995} --manifest ${MANIFEST} --metrics-dir ${METRICS_DIR} --source-registry ${SOURCE_REGISTRY} --leakage-audit ${LEAKAGE_AUDIT} --out-report ${REPORT} --out-summary ${DOCS_AI_SUMMARY}"
"${PYTHON}" "${SCRIPT_DIR}/phase27_a_taxonomy_redesign_v2_report.py" \
  --evidence-manifest "${EVIDENCE}" \
  --full-dev-surface "${FULL_DEV}" \
  --stress-surface "${STRESS_221}" \
  --stress-surface "${STRESS_945}" \
  --stress-surface "${STRESS_995}" \
  --manifest "${MANIFEST}" \
  --metrics-dir "${METRICS_DIR}" \
  --source-registry "${SOURCE_REGISTRY}" \
  --leakage-audit "${LEAKAGE_AUDIT}" \
  --out-report "${REPORT}" \
  --out-summary "${DOCS_AI_SUMMARY}"

record_command "${PYTHON} ${SCRIPT_DIR}/phase27_a_taxonomy_redesign_v2_write_record.py --command-file ${COMMAND_LOG} --evidence-manifest ${EVIDENCE} --full-dev-surface ${FULL_DEV} --stress-surface ${STRESS_221} --stress-surface ${STRESS_945} --stress-surface ${STRESS_995} --manifest ${MANIFEST} --metrics-dir ${METRICS_DIR} --source-registry ${SOURCE_REGISTRY} --leakage-audit ${LEAKAGE_AUDIT} --report ${REPORT} --summary ${DOCS_AI_SUMMARY} --out-record ${RECORD}"
"${PYTHON}" "${SCRIPT_DIR}/phase27_a_taxonomy_redesign_v2_write_record.py" \
  --command-file "${COMMAND_LOG}" \
  --evidence-manifest "${EVIDENCE}" \
  --full-dev-surface "${FULL_DEV}" \
  --stress-surface "${STRESS_221}" \
  --stress-surface "${STRESS_945}" \
  --stress-surface "${STRESS_995}" \
  --manifest "${MANIFEST}" \
  --metrics-dir "${METRICS_DIR}" \
  --source-registry "${SOURCE_REGISTRY}" \
  --leakage-audit "${LEAKAGE_AUDIT}" \
  --report "${REPORT}" \
  --summary "${DOCS_AI_SUMMARY}" \
  --out-record "${RECORD}"

echo "phase27_a_taxonomy_redesign_v2_bounded COMPLETE"
