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
        super().__init__()

        # Freezes the initial base layer, no gradient update
        self._base_layer = self._base_layer
        self._base_layer.weight.requires_grad = False

        # Hyperparameters
        self._r = r
        self._alpha = alpha
        self._dropout = dropout
        self._in_features: int = self._base_layer.in_features
        self._out_features: int = self._base_layer.out_features
        self._scaling: float = alpha / r

        # Create the dropout layer
        self._lora_dropout = (nn.Dropout(p=self._dropout) if dropout
                            > 0.0 else nn.Identity())

        # LoRA linear layers
        self._lora_A: nn.Linear = nn.Linear(self._in_features, self._r, bias=False)
        self._lora_B: nn.Linear = nn.Linear(self._r, self._out_features, bias=False)

        # Produces the Gaussian noise 
        nn.init.normal_(self._lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._lora_B.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b_out = self._base_layer(t)

        # dropout -> LoRA A -> LoRA B
        l_out = self._lora_B(self._lora_A(self._lora_dropout(t)))
        return b_out + l_out

class ClinicalFlanT5(nn.Module):
    """bottom text"""
    def __init__(self, model: str = MODEL, mode: str = "lora",
                 r: int = 8, alpha: float = 0.05):

        super().__init__()
        self._mode: str = mode
        self._r: int  = r
        self._alpha: float = alpha

        self._model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(
                model
            )
        )

    def _inject_lora(self, r: int, alpha: float, dropout: float) -> None:
        for module in self._model.modules():
            pass

