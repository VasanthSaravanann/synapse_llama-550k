"""
Data processing module for the medical reasoning dataset.
Handles loading, alignment, blending, and exporting of datasets.
"""

import os
import logging
from datasets import load_dataset, interleave_datasets
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_datasets(config):
    """
    Load the medical reasoning datasets from Hugging Face.

    Args:
        config (dict): Configuration dictionary containing dataset information.

    Returns:
        tuple: A tuple containing the loaded datasets (mass_ds, brain_ds).
    """
    logger.info("Loading datasets...")

    mass_ds = load_dataset(
        config["mass_ds"]["name"],
        split=config["mass_ds"]["split"]
    )

    brain_ds = load_dataset(
        config["brain_ds"]["name"],
        split=config["brain_ds"]["split"]
    )

    logger.info(f"Loaded mass_ds with {len(mass_ds)} examples")
    logger.info(f"Loaded brain_ds with {len(brain_ds)} examples")

    return mass_ds, brain_ds

def format_mass_dataset(example, config):
    """
    Format the mass dataset by mapping columns to standardized format.

    Args:
        example (dict): A single example from the dataset.
        config (dict): Configuration dictionary.

    Returns:
        dict: Formatted example with 'instruction' and 'output' keys.
    """
    return {
        "instruction": example[config["mass_ds"]["instruction_column"]],
        "output": f"REASONING: {example[config['mass_ds']['output_columns'][0]]}\n\nRESPONSE: {example[config['mass_ds']['output_columns'][1]]}"
    }

def format_brain_dataset(example, config):
    """
    Format the brain dataset by mapping columns to standardized format.

    Args:
        example (dict): A single example from the dataset.
        config (dict): Configuration dictionary.

    Returns:
        dict: Formatted example with 'instruction' and 'output' keys.
    """
    # Try primary column names first, fallback to alternatives
    instruction = example.get(
        config["brain_ds"]["instruction_column"],
        example.get("input", "")
    )

    output = example.get(
        config["brain_ds"]["output_column"],
        example.get("reasoning", "")
    )

    return {
        "instruction": instruction,
        "output": output
    }

def align_datasets(mass_ds, brain_ds, config):
    """
    Align both datasets to a standardized format.

    Args:
        mass_ds: The mass dataset.
        brain_ds: The brain dataset.
        config (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing the aligned datasets.
    """
    logger.info("Aligning datasets...")

    # Align mass_ds
    mass_ds_aligned = mass_ds.map(
        lambda example: format_mass_dataset(example, config),
        remove_columns=mass_ds.column_names
    )

    # Align brain_ds
    brain_ds_aligned = brain_ds.map(
        lambda example: format_brain_dataset(example, config),
        remove_columns=brain_ds.column_names
    )

    logger.info("Datasets aligned successfully")

    return mass_ds_aligned, brain_ds_aligned

def blend_datasets(datasets, probabilities, stopping_strategy):
    """
    Blend multiple datasets with specified probabilities.

    Args:
        datasets (list): List of datasets to blend.
        probabilities (list): List of probabilities for each dataset.
        stopping_strategy (str): Strategy for stopping iteration.

    Returns:
        Dataset: The blended dataset.
    """
    logger.info("Blending datasets...")

    blended_ds = interleave_datasets(
        datasets,
        probabilities=probabilities,
        stopping_strategy=stopping_strategy
    )

    logger.info(f"Blended dataset created with {len(blended_ds)} examples")

    return blended_ds

def export_to_parquet(dataset, output_path):
    """
    Export the dataset to Parquet format.

    Args:
        dataset: The dataset to export.
        output_path (str): Path to save the Parquet file.
    """
    logger.info(f"Exporting dataset to {output_path}...")

    dataset.to_parquet(output_path)

    logger.info("Dataset exported successfully")

def process_medical_reasoning_dataset(config):
    """
    Main function to process the medical reasoning dataset.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Path to the output Parquet file.
    """
    # Load datasets
    mass_ds, brain_ds = load_datasets(config)

    # Align datasets
    mass_ds_aligned, brain_ds_aligned = align_datasets(mass_ds, brain_ds, config)

    # Blend datasets
    blended_ds = blend_datasets(
        [mass_ds_aligned, brain_ds_aligned],
        config["blend_ratios"],
        config["stopping_strategy"]
    )

    # Export to Parquet
    export_to_parquet(blended_ds, config["OUTPUT_PATH"])

    # Print statistics
    print(f"✅ Processing complete! Total rows: {len(blended_ds)}")
    print(f"📊 Column Names: {blended_ds.column_names}")

    return config["OUTPUT_PATH"]

if __name__ == "__main__":
    from configs.config import DATASET_CONFIG, OUTPUT_PATH

    # Update config with output path
    config = DATASET_CONFIG.copy()
    config["OUTPUT_PATH"] = OUTPUT_PATH

    # Process the dataset
    output_file = process_medical_reasoning_dataset(config)
    print(f"Dataset saved to: {output_file}")