#!/usr/bin/env bash
set -euo pipefail

LOGFILE="results/run_all.log"
mkdir -p results

# Redirect all output to terminal and log file simultaneously
exec > >(tee -a "$LOGFILE") 2>&1

COMMON="--num-workers 4 --epochs 50 --batch-size 32 --patch-size 128 --patches-per-frame 4"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_train() {
    local mode=$1 dose=$2
    echo ""
    echo "[$(stamp)] TRAINING: mode=${mode} dose=${dose}"
    python scripts/train.py --mode "$mode" --dose "$dose" $COMMON \
        || { echo "[$(stamp)] ERROR: training failed for mode=${mode} dose=${dose}"; exit 1; }
}

# ── Training ────────────────────────────────────────────────────────────────
for dose in low25 low10 low5; do
    run_train single "$dose"
done

for dose in low25 low10 low5; do
    run_train temporal "$dose"
done

# ── Evaluation ──────────────────────────────────────────────────────────────
echo ""
echo "[$(stamp)] EVALUATING: all doses (producing results/eval_summary.csv and results/eval_{dose}.json)"
python scripts/evaluate.py --all-doses --num-workers 4 --batch-size 32 --patch-size 128 --patches-per-frame 4 \
    || { echo "[$(stamp)] ERROR: evaluation failed"; exit 1; }

echo ""
echo "[$(stamp)] DONE — results written to results/"
