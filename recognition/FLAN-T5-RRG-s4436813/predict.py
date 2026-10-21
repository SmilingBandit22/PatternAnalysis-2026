import argparse
import os
import time
from typing import Dict, List, Optional, Tuple
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataset import (
    IGNORE_INDEX,
    MODEL,
    SOURCE_COL,
    TARGET_COL,
    TASK_PREFIX,
    create_dataloaders,
    load_and_split_dataset,
)
from modules import ClinicalFlanT5, TrainingConstants

# ------------------ Helpers ------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 RRG Model")
    parser.add_argument(
        "--mode",
        default=TrainingConstants.LORA,
        choices=[
            TrainingConstants.ZERO_SHOT,
            TrainingConstants.LORA,
            TrainingConstants.FINETUNE,
        ],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100, help="-1 for full split")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()