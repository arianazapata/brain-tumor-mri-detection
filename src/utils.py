# src/utils.py
import random
import numpy as np
import torch


LABEL_MAP = {"no_tumor": 0, "tumor": 1}
INV_LABEL_MAP = {0: "no_tumor", 1: "tumor"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
