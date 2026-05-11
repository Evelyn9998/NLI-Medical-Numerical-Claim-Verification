#SBATCH --job-name=cot-zs-llama
#SBATCH --output=logs/slurm_cot_zs_llama_%j.out
#SBATCH --error=logs/slurm_cot_zs_llama_%j.err
#SBATCH --account=slurm-students
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -e
mkdir -p logs results

source ~/venv/bin/activate

export TRANSFORMERS_OFFLINE=1

python zero_shot_CoT_ft.py \
    --data mrt_claim_cleaned.csv \
    --model "$HOME/models/Llama-3.1-8B-Instruct" \
    --start "${START:-1}" \
    --samples "${SAMPLES:-0}" \
    --output "results/cot_zs_llama_${SLURM_JOB_ID}.csv"