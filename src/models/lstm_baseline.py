from __future__ import annotations

import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("LSTM baseline will be implemented in the baseline stage.")

