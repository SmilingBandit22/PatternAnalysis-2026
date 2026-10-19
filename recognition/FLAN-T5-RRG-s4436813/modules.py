import math
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention
from typing import Dict, Literal, Optional, Tuple

# Project constants

MODEL: str = "google/flan-t5-base"

# Class definitions

class LoRALinear(nn.Module):
    """bottom text"""

    def __init__(self, base_layer: nn.Linear, r: int = 8,
                 alpha: float = 32.0, dropout: float = 0.05):
        self._base_layer = base_layer

        # Freezes the initial base layer, no gradient update
        self._base_layer = self._base_layer
        self._base_layer.weight.requires_grad = False

        # Hyperparameters
        self._r = r
        self._alpha = alpha
        self._dropout = dropout
        self._in_features = self._base_layer.in_features
        self._out_features = self._base_layer.out_features

        # Create the dropout layer
        self._lora_dropout = (nn.Dropout(p=self._dropout) if dropout
                            > 0.0 else nn.Identity())