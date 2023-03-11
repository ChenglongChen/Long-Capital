from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import Critic
from torch import nn


class MetaActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = action_shape[0]
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)  # type: ignore
        self._max = max_action

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
        logits = self._max * torch.tanh(self.last(logits))
        logits = logits.view(bsz, ch)
        return logits, hidden


class MetaCritic(Critic):
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            )
            act = act.unsqueeze(2)
            obs = torch.cat([obs, act], dim=2)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits
