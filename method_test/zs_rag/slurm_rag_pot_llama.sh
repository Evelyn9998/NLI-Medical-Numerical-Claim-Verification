#!/bin/bash
#SBATCH --job-name=rag-zero-shot-pot-llama
#SBATCH --output=logs/slurm_rag_zero_shot_PoT_Llama_%j.out
#SBATCH --error=logs/slurm_rag_zero_shot_PoT_Llama_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python rag_zero_shot_PoT.py \
    --data        mrt_claim_cleaned.csv \
    --model       "$HOME/models/Llama-3.1-8B-Instruct" \
    --samples     "${SAMPLES:-0}" \
    --output      "results/rag_zero_shot_PoT_llama_${SLURM_JOB_ID}.csv" \
    --start       "${START:-1}" \
    --rag-k 1 \
    --formula-db  web_formula.txt

    
