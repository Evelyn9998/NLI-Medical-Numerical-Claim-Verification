#!/bin/bash
#SBATCH --job-name=clf-formula-cot-llama
#SBATCH --output=logs/slurm_clf_formula_cot_Llama_%j.out
#SBATCH --error=logs/slurm_clf_formula_cot_Llama_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python clf_formula_CoT.py \
    --data        mrt_claim_cleaned.csv \
    --model       "$HOME/models/Llama-3.1-8B-Instruct" \
    --classifier  "$HOME/models/formula_classifier" \
    --samples     "${SAMPLES:-0}" \
    --output      "results/clf_formula_cot_llama_${SLURM_JOB_ID}.csv" \
    --start       "${START:-1}" \
    --formula-data "formula_cleaned.json"

    
