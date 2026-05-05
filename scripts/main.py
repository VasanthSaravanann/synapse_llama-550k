"""
Main execution script for the medical reasoning dataset project.
Orchestrates the entire pipeline from data processing to model training.
"""

import argparse
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import DATASET_CONFIG, OUTPUT_PATH, MODEL_CONFIG, WANDB_CONFIG
from data.dataset_processing import process_medical_reasoning_dataset
from training.trainer import MedicalReasoningTrainer

def main():
    """Main function to run the medical reasoning pipeline."""
    parser = argparse.ArgumentParser(description="Medical Reasoning Dataset Pipeline")
    parser.add_argument("--mode", choices=["process", "train", "test"],
                       default="process", help="Execution mode")
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf",
                       help="Name of the model to use for training")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--enable-wandb", action="store_true",
                       help="Enable Weights & Biases logging")

    args = parser.parse_args()

    # Update config with output path
    config = DATASET_CONFIG.copy()
    config["OUTPUT_PATH"] = OUTPUT_PATH

    if args.mode == "process":
        print("Processing medical reasoning dataset...")
        output_file = process_medical_reasoning_dataset(config)
        print(f"Dataset saved to: {output_file}")

    elif args.mode == "train":
        print("Training medical reasoning model...")

        # Create trainer configuration
        trainer_config = MODEL_CONFIG.copy()
        trainer_config["wandb_enabled"] = args.enable_wandb
        trainer_config["wandb_project"] = WANDB_CONFIG["project"]
        trainer_config["wandb_entity"] = WANDB_CONFIG["entity"]

        # Initialize trainer
        trainer = MedicalReasoningTrainer(trainer_config)

        # Load model and tokenizer
        trainer.load_model_and_tokenizer(args.model_name)

        # Train model (would need actual training data here)
        print(f"Training model: {args.model_name}")
        print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    elif args.mode == "test":
        print("Running smoke test...")

        # Create trainer configuration
        trainer_config = MODEL_CONFIG.copy()
        trainer_config["wandb_enabled"] = args.enable_wandb
        trainer_config["wandb_project"] = WANDB_CONFIG["project"]
        trainer_config["wandb_entity"] = WANDB_CONFIG["entity"]

        # Initialize trainer
        trainer = MedicalReasoningTrainer(trainer_config)

        # Example prompts for testing
        test_prompts = [
            "What are the symptoms of diabetes?",
            "How should a patient with hypertension manage their condition?"
        ]

        # Run smoke test
        loss = trainer.smoke_test(test_prompts)
        print(f"Smoke test completed. Loss: {loss:.4f}")

if __name__ == "__main__":
    main()