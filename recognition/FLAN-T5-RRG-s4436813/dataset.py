import pytorch
import numpy
import re
from functools import reduce

# ------------------ Helpers ------------------

def normalise_text(text: str) -> str:
    """
    Cleans text to normalise the input to the model.
    """
    transforms: tuple[function] = (
        str.lower, 
        lambda s: re.sub(r"[^\w\s]", "", s), # punctuation
        lambda s: re.sub(r"\s+", " ", s), # spaces and newlines,
        str.strip 
    )
    return reduce(lambda acc, fn: fn(acc), transforms, text or "")

# ----------------- Constants -----------------

TASK_PREFIX: str = "Translate this radiology report into plain language" \
"for the patient: "

MODEL: str = "google/flan-t5-base"

# ----------------- Loading & Splits -----------------

dataset = load_dataset