"""
Configuration module for the medical reasoning dataset project.
"""

# Dataset configurations
DATASET_CONFIG = {
    "mass_ds": {
        "name": "Ganesh01kumar02reddy/medical-reasoning-processed",
        "split": "train",
        "instruction_column": "user_content",
        "output_columns": ["reasoning_content", "assistant_content"]
    },
    "brain_ds": {
        "name": "akash2402/OpenMd-medical-reasoning-dataset-lite",
        "split": "train",
        "instruction_column": "instruction",
        "output_column": "output"
    },
    "blend_ratios": [0.95, 0.05],
    "stopping_strategy": "all_exhausted"
}

# File paths
OUTPUT_PATH = "medical_reasoning_combined.parquet"
LOG_PATH = "../logs/"

# Model configurations
MODEL_CONFIG = {
    "sequence_length": 512,
    "gradient_checkpointing": True,
    "mixed_precision": True
}

# W&B configurations
WANDB_CONFIG = {
    "project": "medical-reasoning-dataset",
    "entity": None,  # Set to your W&B entity
    "log_interval": 10
}