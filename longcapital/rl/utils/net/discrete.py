from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from longcapital.utils.constant import MASK_VALUE, NEG_INF
from tianshou.utils.net.common import MLP
from tianshou.utils.net.discrete import Critic
from torch import nn


class MetaActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = action_shape[0]
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)  # type: ignore
        self.softmax_output = softmax_output

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        logits, hidden = self.preprocess(obs, state)
        bsz, ch, d = logits.size(0), logits.size(1), logits.size(2)
        logits = logits.reshape(-1, d)
        logits = self.last(logits)
        logits = logits.view(bsz, ch)
        if self.softmax_output:
            mask = obs.eq(MASK_VALUE).all(2).float()
            logits = (1 - mask) * logits + mask * NEG_INF
            logits = F.softmax(logits, dim=-1)
        return logits, hidden


MetaCritic = Critic
