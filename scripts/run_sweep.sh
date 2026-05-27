#!/bin/bash
# HP Sweep: 5 experiments, sequential, 7GPU DDP (GPU 1-7)
# Each experiment: 30 epochs with in-training eval

set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NPROC=7
BASE_CMD="torchrun --nproc_per_node=$NPROC scripts/train.py --config configs/base.yaml"

# Register GPUs
for gpu in 1 2 3 4 5 6 7; do
    mlml-cli set --gpu $gpu --name junyeob --mode Train 2>/dev/null || true
done

echo "=========================================="
echo "HP Sweep Start: $(date)"
echo "=========================================="

# --- Experiment 1: Resume from epoch_30 + cosine restart (30ep period) ---
echo ""
echo "[Exp 1/5] epochs=100 (resume from epoch_30, cosine restart 30ep)"
$BASE_CMD \
    --resume checkpoints/epoch_30.pt \
    --epochs 100 \
    --cosine_restart_epochs 30 \
    --checkpoint_dir checkpoints/sweep_resume100 \
    2>&1 | tee logs/sweep_1_resume100.log
echo "[Exp 1] Done: $(date)"

# --- Experiment 2: lr=3e-4 ---
echo ""
echo "[Exp 2/5] lr=3e-4"
$BASE_CMD \
    --lr 3e-4 \
    --epochs 30 \
    --checkpoint_dir checkpoints/sweep_lr3e4 \
    2>&1 | tee logs/sweep_2_lr3e4.log
echo "[Exp 2] Done: $(date)"

# --- Experiment 3: temperature=0.05 ---
echo ""
echo "[Exp 3/5] temperature=0.05"
$BASE_CMD \
    --temperature 0.05 \
    --epochs 30 \
    --checkpoint_dir checkpoints/sweep_temp005 \
    2>&1 | tee logs/sweep_3_temp005.log
echo "[Exp 3] Done: $(date)"

# --- Experiment 4: batch_size=16 ---
echo ""
echo "[Exp 4/5] batch_size=16"
$BASE_CMD \
    --batch_size 16 \
    --epochs 30 \
    --checkpoint_dir checkpoints/sweep_bs16 \
    2>&1 | tee logs/sweep_4_bs16.log
echo "[Exp 4] Done: $(date)"

# --- Experiment 5: lr=5e-5 ---
echo ""
echo "[Exp 5/5] lr=5e-5"
$BASE_CMD \
    --lr 5e-5 \
    --epochs 30 \
    --checkpoint_dir checkpoints/sweep_lr5e5 \
    2>&1 | tee logs/sweep_5_lr5e5.log
echo "[Exp 5] Done: $(date)"

# Clear GPUs
for gpu in 1 2 3 4 5 6 7; do
    mlml-cli clear --gpu $gpu 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "HP Sweep Complete: $(date)"
echo "=========================================="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in logs/sweep_*.log; do
    name=$(basename $f .log)
    echo "--- $name ---"
    grep -E "(^Epoch|Eval:)" "$f" | tail -5
    echo ""
done
