#!/bin/bash
#SBATCH --job-name=rag-zero-shot-pot-qwen
#SBATCH --output=logs/slurm_rag_zero_shot_PoT_Qwen_%j.out
#SBATCH --error=logs/slurm_rag_zero_shot_PoT_Qwen_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python rag_zero_shot_PoT.py \
    --data        mrt_claim_cleaned.csv \
    --model       "$HOME/models/Qwen3-4B" \
    --samples     "${SAMPLES:-0}" \
    --output      "results/rag_zero_shot_PoT_qwen_${SLURM_JOB_ID}.csv" \
    --start       "${START:-1}" \
    --rag-k 1 \
    --no-think \
    --formula-db  web_formula.txt

    
