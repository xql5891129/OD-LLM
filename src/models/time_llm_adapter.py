from __future__ import annotations

import torch.nn as nn


class TimeLLMODAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Time-LLM adapter will be implemented in stage 3.")

