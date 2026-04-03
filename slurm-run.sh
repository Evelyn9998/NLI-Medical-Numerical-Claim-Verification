#!/bin/bash
#SBATCH --job-name=cot-calc
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --account=slurm-students
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -e

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p logs results
export HF_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_OFFLINE=1

echo "================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"
echo "================================================"

# ── Install / activate environment ────────────────────────────────────────
if [ ! -f ~/venv/bin/activate ]; then
    python3 -m venv ~/venv
fi
source ~/venv/bin/activate
mkdir -p ~/tmp
TMPDIR=~/tmp pip install -r requirements.txt

# ── Step 0: CSV data check ─────────────────────────────────────────────────
DATA=train_300.csv
echo ""
echo "[Step 0] Checking data file: $DATA"
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found"
    exit 1
fi
echo "  Rows: $(wc -l < "$DATA")"

# ── Step 1: Validate calculators ──────────────────────────────────────────
echo ""
echo "[Step 1] Running calculators.py"
python calculators.py 2>&1 | tee logs/calculators.log
echo "  Done: $(date)"

# ── Step 2: Run CoT + External Calculator pipeline ────────────────────────
echo ""
echo "[Step 2] Running cot_external_calc.py"
python cot_external_calc.py \
    --data "$DATA" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --samples 0 \
    --output "results/cot_calc_train300_${SLURM_JOB_ID}.csv" \
    2>&1 | tee logs/cot_external_calc.log
echo "  Done: $(date)"

echo ""
echo "================================================"
echo "Job finished: $(date)"
echo "Results in:   results/cot_calc_train300_${SLURM_JOB_ID}.csv"
echo "Logs in:      logs/"
echo "================================================"
