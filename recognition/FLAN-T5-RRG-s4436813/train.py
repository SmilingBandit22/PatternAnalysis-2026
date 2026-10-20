import argparse
import os
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
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


def parse_args() -> argparse.Namespace:
    """
    Small convenience function for command line parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=TrainingConstants.LORA,
                        choices=[TrainingConstants.LORA, TrainingConstants.FINETUNE])
    parser.add_argument("--max_steps", default=1, type=int)
    return parser.parse_args()

# ----------------- Distributed -----------------

# TODO: I don't like the way this is structured.
def setup_distributed() -> tuple[bool, int, torch.device]:
    """
    Returns (distributed, rank, device). Joins the process group under torchrun.
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, dist.get_rank(), torch.device("cuda", local_rank)

# ----------------- Training -----------------

def train_single_epoch(model: nn.Module, loader: DataLoader,
                       optimiser: torch.optim.Optimizer, device: torch.device,
                       max_steps: int = -1) -> float:
    """
    Runs a single epoch of training. Returns the mean training loss.
    """
    total_loss: float = 0.0 
    steps: int = 0
    model.train()

    # Send everything in the batch loader to GPUs
    for batch in loader:
        batch = {
            k: v.to(device) for k, v in batch.items()
        }
        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss: torch.Tensor = outputs.loss
        # Backpropogation and optimisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # Accumulate losses and increment the steps
        total_loss += loss.item()
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break
    return total_loss / max(steps, 1)