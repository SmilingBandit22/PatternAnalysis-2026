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

EPOCHS: int = 0
BATCH_SIZE: int = 8
LR: float = 3e-4
OUT_DIR: str = ""