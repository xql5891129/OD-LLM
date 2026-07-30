import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{name}', but CUDA is not available to PyTorch. "
            f"Current torch build: {torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "Install a CUDA-enabled torch wheel, for example: "
            ".\\scripts\\setup_env.ps1 -SkipModel -TorchBackend cu128"
        )
    return torch.device(name)
