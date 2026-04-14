#!/bin/bash
#SBATCH --job-name=cot-calc-mrt
#SBATCH --output=logs/slurm_cot_calc_%j.out
#SBATCH --error=logs/slurm_cot_calc_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python cot_external_calc.py \
    --data        mrt_claim_cleaned.csv \
    --model       "$HOME/models/Llama-3.1-8B-Instruct" \
    --classifier  "$HOME/models/formula_classifier" \
    --max-tokens-step1 512 \
    --max-tokens-step3 768 \
    --start       "${START:-1}" \
    --samples     "${SAMPLES:-0}" \
    --output      "results/cot_clf_mrt_${SLURM_JOB_ID}.csv" \
    --temperature 0.1