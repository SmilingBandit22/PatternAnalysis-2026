import argparse
import os
import matplotlib
matplotlib.use("Agg")
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import create_dataloaders
from modules import ClinicalFlanT5, TrainingConstants

# ----------------- Constants -----------------

class LearningRates():
    LORA: float = 3e-4
    FINETUNE: float = 3e-5

EPOCHS: int = 0
BATCH_SIZE: int = 8
LR: float = LearningRates.LORA
OUT_DIR: str = ""

# ----------------- Classes -----------------

def parse_args() -> argparse.Namespace:
    """
    Small convenience function for command line parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=TrainingConstants.LORA,
                        choices=[TrainingConstants.LORA, TrainingConstants.FINETUNE])
    parser.add_argument("--max_steps", default=1, type=int)