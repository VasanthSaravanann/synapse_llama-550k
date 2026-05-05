#!/bin/bash
#SBATCH --job-name=medical_reasoning_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=../logs/train_model_%j.out
#SBATCH --error=../logs/train_model_%j.err

# Load modules
module load cuda/12.1
module load python/3.10

# Activate virtual environment (if applicable)
# source venv/bin/activate

# Set working directory
cd /run/media/sham/AI_/ai-stack/projects/synapse_llama-550k

# Install dependencies
pip install -r requirements.txt

# Install xformers for CUDA 12.1
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and related packages
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Run training with W&B integration
python scripts/main.py --mode train --model-name meta-llama/Llama-2-7b-hf --epochs 3 --batch-size 4 --enable-wandb

echo "Model training completed"