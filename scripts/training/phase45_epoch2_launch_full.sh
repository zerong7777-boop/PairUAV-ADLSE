set -euo pipefail

export UAVM_ROOT=/root/autodl-tmp/uavm_2026
export RELOC3R_REPO=$UAVM_ROOT/external/reloc3r_pairuav
export OUTPUT_ROOT=$UAVM_ROOT/runs/reloc3r_official_pairuav
export SOURCE_RUN_DIR=$OUTPUT_ROOT/official_metric_full_epoch_v1_20260506_045442
export SOURCE_CKPT_LAST=$SOURCE_RUN_DIR/checkpoint-last.pth
export PYTHON_BIN=/root/miniconda3/bin/python
export RUN_TAG=20260523_215328
export CONTROL_DIR=$OUTPUT_ROOT/phase45_epoch2_resume_control_v1_${RUN_TAG}
export FULL_RUN_NAME=phase45_epoch2_resume_full_v1_${RUN_TAG}
export FULL_RUN_DIR=$OUTPUT_ROOT/$FULL_RUN_NAME

mkdir -p "$FULL_RUN_DIR"
cp -n "$SOURCE_CKPT_LAST" "$FULL_RUN_DIR/checkpoint-last.pth"
stat -c '%n %s %y' "$SOURCE_CKPT_LAST" "$FULL_RUN_DIR/checkpoint-last.pth"

cat > "$CONTROL_DIR/phase45_full.env" <<EOF
RUN_FAMILY=phase45_epoch2_resume_full_v1
RUN_NAME=$FULL_RUN_NAME
PYTHON_BIN=$PYTHON_BIN
UAVM_ROOT=$UAVM_ROOT
OUTPUT_ROOT=$OUTPUT_ROOT
SEED=777
RESOLUTION="(512,384)"
MODEL_EXPR="Reloc3rRelpose(img_size=512, output_mode='pairuav_heading_range')"
CRITERION_EXPR="PairUAVOfficialMetricAwareLoss(heading_weight=1.0, range_weight=1.0, angle_floor_deg=1.0, distance_floor=1.0, absolute_heading_weight=0.05, absolute_range_weight=0.05)"
BATCH_SIZE=8
NUM_WORKERS=8
EPOCHS=2
LR=5e-6
WARMUP_EPOCHS=0
AMP=1
MAX_TRAIN_STEPS=0
STEP_CHECKPOINT_FREQ=2000
EVAL_FREQ=1
PRETRAINED=
EOF

cat "$CONTROL_DIR/phase45_full.env"

cd "$UAVM_ROOT"
nohup bash "$RELOC3R_REPO/scripts/train_pairuav_official_metric_longer_5090.sh" "$CONTROL_DIR/phase45_full.env" > "$FULL_RUN_DIR/launcher.log" 2>&1 &
echo $! > "$FULL_RUN_DIR/train.pid"
cat "$FULL_RUN_DIR/train.pid"
sleep 10
ps -p "$(cat "$FULL_RUN_DIR/train.pid")" -o pid,etime,cmd
