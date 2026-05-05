# Synapse LLaMA-550K Medical Reasoning Dataset

## Overview

This repository contains a combined medical reasoning dataset designed for training large language models in medical reasoning tasks. The dataset combines multiple medical reasoning sources to create a comprehensive training corpus with 550,000+ examples.

## Dataset Composition

The dataset is created by strategically blending two medical reasoning datasets:

1. **Ganesh01kumar02reddy/medical-reasoning-processed** (95% of the dataset)
2. **akash2402/OpenMd-medical-reasoning-dataset-lite** (5% of the dataset)

## Dataset Processing Pipeline

### 1. Data Loading
```python
mass_ds = load_dataset("Ganesh01kumar02reddy/medical-reasoning-processed", split="train")
brain_ds = load_dataset("akash2402/OpenMd-medical-reasoning-dataset-lite", split="train")
```

### 2. Data Alignment

#### For mass_ds:
- Uses 'user_content' as the instruction
- Combines 'reasoning_content' + 'assistant_content' as the output

#### For brain_ds:
- Maps existing 'instruction' and 'output' columns
- Includes fallback mappings for alternative column names

### 3. Strategic Blending
```python
mixed_ds = interleave_datasets(
    [mass_ds, brain_ds], 
    probabilities=[0.95, 0.05], 
    stopping_strategy="all_exhausted"
)
```

### 4. Export
The final dataset is exported to Parquet format for HPC optimized access:
```python
mixed_ds.to_parquet("medical_reasoning_combined.parquet")
```

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

## Usage

The dataset is ideal for training medical reasoning models and can be used with popular frameworks like Hugging Face Transformers, Unsloth, and other LLM training libraries.

### Example Usage:
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("path/to/medical_reasoning_combined.parquet")

# Access examples
for example in dataset["train"]:
    instruction = example["instruction"]
    output = example["output"]
    # Train your model
```

## Model Training Support

The repository includes code for:
- Installing xformers for CUDA 12.1 optimization
- Setting up Unsloth and related packages
- Configuring gradient checkpointing for memory efficiency
- Running smoke tests to verify training setup

### Training Test Results
Successful manual smoke test with a loss of 3.2988, demonstrating that the dataset is properly formatted for training.

## Dependencies

Key dependencies for working with this dataset:
- datasets (Hugging Face)
- xformers (CUDA 12.1 optimized)
- Unsloth framework
- PyTorch with CUDA support
- transformers library

## Applications

This dataset is particularly suitable for:
- Medical reasoning model training
- Clinical decision support systems
- Medical question answering systems
- Healthcare chatbots
- Biomedical research assistance

## Repository Contents

- `MEDICAL_REASONING_DATASET.md`: Detailed documentation of the dataset creation process
- `notebookf97f3b150d.ipynb`: Jupyter notebook with the complete dataset processing pipeline
- `medical_reasoning_combined.parquet`: The final combined dataset (generated)

## License

This dataset is released for research purposes. Please check the licenses of the original datasets for commercial use restrictions.