from __future__ import annotations

import torch.nn as nn


class STLLMODAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("ST-LLM adapter will be implemented after the main OD-LLM path.")

