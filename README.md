# Synapse LLaMA-550K Medical Reasoning Dataset

## Overview

This repository contains a combined medical reasoning dataset designed for training large language models in medical reasoning tasks. The dataset combines multiple medical reasoning sources to create a comprehensive training corpus with 550,000+ examples. The project has been refactored from an interactive Jupyter notebook to modular Python scripts that can be executed via SLURM for academic infrastructure.

## Dataset Composition

The dataset is created by strategically blending two medical reasoning datasets:

1. **Ganesh01kumar02reddy/medical-reasoning-processed** (95% of the dataset)
2. **akash2402/OpenMd-medical-reasoning-dataset-lite** (5% of the dataset)

## Repository Structure

```
├── configs/                 # Configuration files
├── data/                   # Data processing scripts
├── models/                 # Model definitions
├── training/               # Training loop implementations
├── evaluation/             # Evaluation scripts
├── edge_deployment/        # Edge deployment configurations
├── scripts/                # Main execution scripts and SLURM jobs
├── utils/                  # Utility functions
├── logs/                   # Log files
├── Dockerfile              # Containerization configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd synapse_llama-550k
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For GPU-accelerated training, install xformers:
   ```bash
   pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install Unsloth framework:
   ```bash
   pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes
   ```

## Usage

### Data Processing

Process the medical reasoning dataset:
```bash
python scripts/main.py --mode process
```

### Training

Train the model with default parameters:
```bash
python scripts/main.py --mode train --model-name meta-llama/Llama-2-7b-hf
```

Train with custom parameters:
```bash
python scripts/main.py --mode train --model-name meta-llama/Llama-2-7b-hf --epochs 3 --batch-size 4
```

### Testing

Run a smoke test to verify the setup:
```bash
python scripts/main.py --mode test
```

## SLURM Execution

Submit jobs using the provided SLURM scripts:

### Process Dataset
```bash
sbatch scripts/process_dataset.slurm
```

### Train Model
```bash
sbatch scripts/train_model.slurm
```

## Weights & Biases Integration

The training pipeline includes built-in Weights & Biases integration for experiment tracking:

```bash
python scripts/main.py --mode train --enable-wandb
```

## Alignment Loss Functions

The training module implements two key alignment loss functions:

1. **Direct Preference Optimization (DPO)** - Directly optimizes the model based on preference data
2. **Reinforcement Learning from Human Feedback (RLHF)** - Uses reinforcement learning with human feedback

Both loss functions mathematically penalize unsafe or unaligned outputs during training.

## Dataset Statistics

- **Total Rows**: 532,740
- **Column Names**: ['instruction', 'output']
- **Format**: Parquet (optimized for high-performance computing)

## Key Features

- **Medical Reasoning Focus**: Specifically designed for training models on medical reasoning tasks
- **Large Scale**: Over 500K training examples
- **High-Quality Alignment**: Carefully aligned instruction-output pairs
- **HPC Optimized**: Stored in Parquet format for efficient loading and processing
- **Strategic Mixing**: Balanced combination of different medical reasoning sources
- **SLURM Compatible**: Designed for academic HPC environments
- **W&B Integration**: Automatic experiment tracking
- **Alignment Loss Functions**: DPO and RLHF implementations

## Applications

This dataset is particularly suitable for:
- Medical reasoning model training
- Clinical decision support systems
- Medical question answering systems
- Healthcare chatbots
- Biomedical research assistance

## License

This dataset is released for research purposes. Please check the licenses of the original datasets for commercial use restrictions.