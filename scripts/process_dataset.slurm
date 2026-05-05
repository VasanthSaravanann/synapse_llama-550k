#!/bin/bash
#SBATCH --job-name=medical_reasoning_process
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=../logs/process_dataset_%j.out
#SBATCH --error=../logs/process_dataset_%j.err

# Load modules
module load cuda/12.1
module load python/3.10

# Activate virtual environment (if applicable)
# source venv/bin/activate

# Set working directory
cd /run/media/sham/AI_/ai-stack/projects/synapse_llama-550k

# Install dependencies
pip install -r requirements.txt

# Run dataset processing
python scripts/main.py --mode process

echo "Dataset processing completed"