import random

import numpy as np
import torch
from sklearn import metrics


def set_random_state(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not deterministic


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Precision": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Accuracy": float(metrics.accuracy_score(y_true, y_pred)),
    }
