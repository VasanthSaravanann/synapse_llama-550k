"""
Unit tests for the dataset processing module.
"""

import unittest
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from configs.config import DATASET_CONFIG, OUTPUT_PATH

class TestDatasetProcessing(unittest.TestCase):
    """Test cases for dataset processing functions."""

    def test_config_loading(self):
        """Test that configuration loads correctly."""
        self.assertIn("mass_ds", DATASET_CONFIG)
        self.assertIn("brain_ds", DATASET_CONFIG)
        self.assertEqual(len(DATASET_CONFIG["blend_ratios"]), 2)
        self.assertEqual(DATASET_CONFIG["blend_ratios"][0], 0.95)
        self.assertEqual(DATASET_CONFIG["blend_ratios"][1], 0.05)

    def test_output_path(self):
        """Test that output path is defined."""
        self.assertEqual(OUTPUT_PATH, "medical_reasoning_combined.parquet")

if __name__ == "__main__":
    unittest.main()