import hashlib
import re
from functools import reduce, filter
from typing import Dict, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ----------------- Constants -----------------

# Model parameters & Details
MODEL: str = "google/flan-t5-base"
DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX: str = "Translate this radiology report into plain language " \
"for the patient: "

# Constants
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
IGNORE_INDEX = -100

# String Constants
SOURCE_COL: str = "radiology_report"
TARGET_COL: str = "layman_report"

# ------------------ Helpers ------------------

def normalise_text(text: str) -> str:
    """
    Cleans text to normalise the input to the model.
    """
    transforms: tuple[Callable] = (
        str.lower, 
        lambda s: re.sub(r"[^\w\s]", "", s), # punctuation
        lambda s: re.sub(r"\s+", " ", s), # spaces and newlines,
        str.strip 
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

def hash_text(text: str) -> str:
    """
    Creates a text digest, used for fast deduplication.
    """
    transforms: tuple[Callable, ...] = (
        normalise_text,
        str.encode,
        hashlib.sha256,
        lambda h: h.hexdigest()
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

def load_and_split_dataset(dataset: str = DATASET_NAME) \
      -> Tuple[HFDataset, HFDataset, HFDataset]:
    """
    Loads and splits the dataset.
    """
    # Splits
    raw_dataset: DatasetDict = load_dataset(DATASET_NAME)
    train_split: HFDataset = raw_dataset["train"]
    validation_split: HFDataset = raw_dataset["validation"]

    # Hashes for training purposes
    train_hashes: set[str] = {hash_text(report) for report 
                              in train_split[SOURCE_COL]}
    val_unseen: HFDataset = validation_split.filter(
        lambda row: hash_text(row[SOURCE_COL] not in train_hashes)
    )

    return train_split, validation_split, val_unseen
# -------------- Loading & Splits --------------
