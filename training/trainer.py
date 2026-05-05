"""
Training module for the medical reasoning model.
Includes W&B integration and alignment loss functions.
"""

import torch
import torch.nn as nn
import logging
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReasoningTrainer:
    """Trainer class for medical reasoning model with W&B integration."""

    def __init__(self, config):
        """
        Initialize the trainer.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.wandb_run = None

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config.get("wandb_enabled", False):
            logger.info("Initializing Weights & Biases...")
            self.wandb_run = wandb.init(
                project=self.config["wandb_project"],
                entity=self.config["wandb_entity"],
                config=self.config
            )
            logger.info("Weights & Biases initialized")

    def load_model_and_tokenizer(self, model_name):
        """
        Load the model and tokenizer.

        Args:
            model_name (str): Name of the model to load.
        """
        logger.info(f"Loading model and tokenizer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info("Model and tokenizer loaded successfully")

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        logger.info("Clearing GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")

    def tokenize_prompts(self, prompts, max_length=None):
        """
        Tokenize prompts with configurable sequence length.

        Args:
            prompts (list): List of prompts to tokenize.
            max_length (int): Maximum sequence length.

        Returns:
            dict: Tokenized inputs.
        """
        if max_length is None:
            max_length = self.config.get("sequence_length", 512)

        logger.info(f"Tokenizing prompts with max length: {max_length}")

        inputs = self.tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")

        return inputs

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save VRAM."""
        if self.config.get("gradient_checkpointing", True):
            logger.info("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    def dpo_loss(self, preferred_logits, dispreferred_logits, beta=0.1):
        """
        Direct Preference Optimization (DPO) loss function.

        Args:
            preferred_logits (torch.Tensor): Logits for preferred responses.
            dispreferred_logits (torch.Tensor): Logits for dispreferred responses.
            beta (float): Temperature parameter for DPO.

        Returns:
            torch.Tensor: DPO loss.
        """
        # Calculate log sigmoid of differences
        diff = beta * (preferred_logits - dispreferred_logits)
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss

    def rlhf_loss(self, logits, rewards, ref_logits, beta=0.1):
        """
        Reinforcement Learning from Human Feedback (RLHF) loss function.

        Args:
            logits (torch.Tensor): Action logits from the policy model.
            rewards (torch.Tensor): Reward values for actions.
            ref_logits (torch.Tensor): Logits from the reference model.
            beta (float): Temperature parameter for RLHF.

        Returns:
            torch.Tensor: RLHF loss.
        """
        # Compute KL divergence between policy and reference
        kl_div = torch.nn.functional.kl_div(
            torch.log_softmax(logits, dim=-1),
            torch.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )

        # RLHF loss combines reward and KL penalty
        loss = -(rewards - beta * kl_div).mean()
        return loss

    def train_step(self, inputs, labels=None):
        """
        Perform a single training step.

        Args:
            inputs (dict): Tokenized inputs.
            labels (torch.Tensor, optional): Labels for supervised training.

        Returns:
            dict: Training metrics.
        """
        self.model.train()

        # Use mixed precision for memory efficiency
        with torch.cuda.amp.autocast(enabled=self.config.get("mixed_precision", True)):
            if labels is not None:
                outputs = self.model(**inputs, labels=labels)
            else:
                outputs = self.model(**inputs, labels=inputs["input_ids"])

            loss = outputs.loss

        # Log to W&B
        if self.wandb_run is not None:
            wandb.log({"train_loss": loss.item()})

        return {
            "loss": loss.item(),
            "logits": outputs.logits
        }

    def train(self, train_data, num_epochs=1, batch_size=1):
        """
        Train the model on the provided data.

        Args:
            train_data (Dataset): Training dataset.
            num_epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
        """
        logger.info("Starting training...")

        self.setup_wandb()
        self.enable_gradient_checkpointing()

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Process batches
            for i in range(0, len(train_data), batch_size):
                # Get batch
                batch = train_data[i:i+batch_size]

                # Tokenize
                inputs = self.tokenize_prompts(batch["instruction"])

                # Training step
                metrics = self.train_step(inputs)

                # Log progress
                if i % self.config.get("log_interval", 10) == 0:
                    logger.info(f"Epoch {epoch + 1}, Step {i}: Loss = {metrics['loss']:.4f}")

                    # Log to W&B
                    if self.wandb_run is not None:
                        wandb.log({
                            "epoch": epoch + 1,
                            "step": i,
                            "loss": metrics['loss']
                        })

        logger.info("Training completed")

    def smoke_test(self, prompts):
        """
        Run a smoke test to verify the model is working correctly.

        Args:
            prompts (list): List of prompts for testing.

        Returns:
            float: Test loss value.
        """
        logger.info("Running smoke test...")

        self.clear_gpu_memory()
        inputs = self.tokenize_prompts(prompts, max_length=512)
        self.enable_gradient_checkpointing()

        # Forward pass
        self.model.train()
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        loss.backward()

        logger.info(f"Smoke test completed. Loss: {loss.item():.4f}")
        return loss.item()

def main():
    """Main function for testing the trainer."""
    # Example configuration
    config = {
        "sequence_length": 512,
        "gradient_checkpointing": True,
        "mixed_precision": True,
        "wandb_enabled": False,  # Set to True to enable W&B
        "wandb_project": "medical-reasoning-dataset",
        "wandb_entity": None,
        "log_interval": 10
    }

    # Initialize trainer
    trainer = MedicalReasoningTrainer(config)

    # Example prompts for smoke test
    test_prompts = [
        "What are the symptoms of diabetes?",
        "How should a patient with hypertension manage their condition?"
    ]

    # Run smoke test (this would normally be done with a loaded model)
    # loss = trainer.smoke_test(test_prompts)
    # print(f"Smoke test loss: {loss:.4f}")

if __name__ == "__main__":
    main()