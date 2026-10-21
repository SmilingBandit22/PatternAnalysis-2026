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


def load_eval_model(mode: str, checkpoint_path: Optional[str],
                     device: torch.device) -> ClinicalFlanT5:
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


def evaluate_split(model: ClinicalFlanT5, loader: DataLoader,
                    tokeniser: PreTrainedTokenizerBase, device: torch.device,
                      max_samples: int = -1,) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """Generates summaries, computes ROUGE metrics, and measures throughput."""
    rouge = evaluate.load("rouge")
    all_preds: List[str] = list()
    all_refs: List[str] = list()
    all_sources: List[str] = list()

    total_tokens: int = 0
    start_time = time.perf_counter()
    sample_count: int = 0

    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].clone()

        # Generate summary predictions
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                num_beams=2,
            )