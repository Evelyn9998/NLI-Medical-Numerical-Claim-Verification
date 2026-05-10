#!/bin/bash
#SBATCH --job-name=clf-formula-cot-qwen
#SBATCH --output=logs/slurm_clf_formula_cot_qwen_%j.out
#SBATCH --error=logs/slurm_clf_formula_cot_qwen_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python clf_formula_CoT.py \
    --data        mrt_claim_cleaned.csv \
    --model       "$HOME/models/Qwen3-4B" \
    --classifier  "$HOME/models/formula_classifier" \
    --samples     "${SAMPLES:-0}" \
    --output      "results/clf_formula_cot_qwen_${SLURM_JOB_ID}.csv" \
    --start       "${START:-1}" \
    --no-think \
    --formula-data "formula_cleaned.json"
 
