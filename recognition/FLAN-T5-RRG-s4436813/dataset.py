import hashlib
import re
from functools import reduce
from typing import Dict, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ----------------- Constants -----------------

MODEL: str = "google/flan-t5-base"
DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
TASK_PREFIX: str = "Translate this radiology report into plain language " \
"for the patient: "
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
IGNORE_INDEX = -100

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
    transforms: tuple[Callable] = (
        hashlib.sha256,
        normalise_text,
        encode,
        hexdigest,
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

# TODO: this is probably just bloat.
def _resolve_columns():
    """
    Resolves the column names between Hugging Face,
    and this implementation.
    """
    pass

# -------------- Loading & Splits --------------

raw_dataset: DatasetDict = load_dataset(DATASET_NAME)

train_split: HFDataset = raw_dataset["train"]

