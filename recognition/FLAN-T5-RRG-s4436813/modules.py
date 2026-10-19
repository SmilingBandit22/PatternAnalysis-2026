import math
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
from typing import Dict, Literal, Optional, Tuple

MODEL: str = "google/flan-t5-base"
AdaptationMode = Literal["zero_shot", "lora", "full_finetune"]