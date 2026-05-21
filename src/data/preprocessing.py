from __future__ import annotations

import numpy as np


def fill_diagonal_zero(od: np.ndarray) -> np.ndarray:
    """Set self OD flow to zero for each time step."""
    od = od.copy()
    idx = np.arange(od.shape[1])
    od[:, idx, idx] = 0.0
    return od

