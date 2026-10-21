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


def load_eval_model(mode: str, checkpoint_path: Optional[str], device: torch.device) -> ClinicalFlanT5:
    """Initializes model and loads weights if checkpoint is provided."""
    model = ClinicalFlanT5(mode=mode).to(device)

    # Resolve default checkpoint path if not explicitly provided
    if checkpoint_path is None and mode != TrainingConstants.ZERO_SHOT:
        checkpoint_path = os.path.join("checkpoints", f"{mode}.pt")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        is_lora = mode == TrainingConstants.LORA
        model.load_state_dict(state, strict=not is_lora)
    elif mode != TrainingConstants.ZERO_SHOT:
        print(f"No checkpoint found at '{checkpoint_path}'. Evaluating untuned backbone.")

    model.eval()
    return model