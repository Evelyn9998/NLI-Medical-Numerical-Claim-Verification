#!/bin/bash
#SBATCH --job-name=cot-zs-deepseek
#SBATCH --output=logs/slurm_cot_zs_deepseek_%j.out
#SBATCH --error=logs/slurm_cot_zs_deepseek_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=32G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python zero_shot.py \
    --data mrt_claim_cleaned.csv \
    --model "$HOME/models/DeepSeek-R1-Distill-Qwen-7B" \
    --start "${START:-1}" \
    --samples "${SAMPLES:-0}" \
    --output "results/zs_deepseek_${SLURM_JOB_ID}.csv"