# Medical Reasoning Dataset Creation and Processing

This document explains the process of creating a combined medical reasoning dataset from multiple sources, preparing it for machine learning training.

## Overview

The notebook combines two medical reasoning datasets:
1. `Ganesh01kumar02reddy/medical-reasoning-processed`
2. `akash2402/OpenMd-medical-reasoning-dataset-lite`

These datasets are strategically blended with a 95%/5% ratio and saved in Parquet format for HPC optimized access.

## Process Steps

### 1. Dataset Loading
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
- Has fallback mappings for alternative column names

### 3. Strategic Blending
```python
mixed_ds = interleave_datasets(
    [mass_ds, brain_ds], 
    probabilities=[0.95, 0.05], 
    stopping_strategy="all_exhausted"
)
```

### 4. Export
The final dataset is exported to Parquet format:
```python
mixed_ds.to_parquet("medical_reasoning_combined.parquet")
```

## Results
- Total rows: 532,740
- Column names: ['instruction', 'output']

## Dependencies Installation
The notebook also includes steps for installing required dependencies:
1. xformers for CUDA 12.1
2. Unsloth and related packages
3. Force reinstall to sync unsloth and unsloth_zoo

## Model Testing
Includes code for:
1. Clearing GPU memory
2. Retokenizing with smaller sequence length (512)
3. Enabling gradient checkpointing
4. Running a manual smoke test with forward pass

The test was successful with a loss of 3.2988.