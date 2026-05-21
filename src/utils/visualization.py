from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(matrix: np.ndarray, path: str | Path, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    if title:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

