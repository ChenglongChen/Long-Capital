import math
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from tianshou.utils.net.common import Net
from torch import Tensor, nn

EPS = 1e-8
NEG_INF = -1e8
MASK_VALUE = NEG_INF
MAX_POSITIONS = 1000


def get_shape(x: Tensor):
    shape = list(x.size())
    return shape


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_position: int = MAX_POSITIONS,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_position = max_position

        position = torch.arange(max_position).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_position, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Embedding.from_pretrained(embeddings=pe, freeze=True)

    def forward(self, position_ids: Tensor) -> Tensor:
        """
        Args:
            position_ids: Tensor, shape [batch_size, seq_len]
        """
        return self.dropout(self.pe(position_ids))


class Pooling(nn.Module):
    """Pooling over the last but one dimension"""

    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        x: [batch_size, ..., pooling_dim, d_model]
        mask: [batch_size, ..., pooling_dim]
        """

        # reshape
        shape = get_shape(x)
        y = torch.reshape(x, (-1, shape[-2], shape[-1]))
        if mask is None:
            mask = torch.ones(shape[:-1])
        m = torch.reshape(mask.float(), (-1, shape[-2]))

        z = self.pooling(y, m)

        # reshape back to original shape
        shape[-2] = -1
        o = torch.reshape(z, shape[:-1])
        return o

    def pooling(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class MeanPooling(Pooling):
    """Mean Pooling over the last but one dimension"""

    def pooling(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        mask = mask.unsqueeze(2)
        y = (x * mask).sum(1) / (mask.sum(1) + EPS)
        return y


class MaxPooling(Pooling):
    """Max Pooling over the last but one dimension"""

    def pooling(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # check has at least one non-mask value
        valid = mask.sum(1) > 0
        m = mask.unsqueeze(2).transpose(1, 2)
        y = x.transpose(1, 2)
        s = int(m.size(2))
        z = torch.max_pool1d(y + (1 - m) * NEG_INF, s)
        # mask out invalid entry with all mask value
        z[valid] = 0.0
        return z


class AttentionPooling(Pooling):
    """Attention Pooling over the last but one dimension"""

    def __init__(self, d_model, dropout):
        super(AttentionPooling, self).__init__()
        self.d_model = d_model
        self.Q = nn.Linear(d_model, 1)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def pooling(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        v = self.V(x)
        k = self.K(x)
        attn = self.Q(k / (self.d_model**0.5))
        m = mask.unsqueeze(2)
        attn = m * attn + (1 - m) * NEG_INF
        attn = self.dropout(F.softmax(attn, dim=1))
        x = (v * attn * m).sum(1)
        return x


ModuleType = Type[nn.Module]
ArgsType = Union[
    Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]], Sequence[Dict[Any, Any]]
]


class MetaNet(Net):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        attn_pooling=False,
        self_attn=False,
        nhead=2,
        dim_feedforward=256,
        num_layers=6,
        dropout=0.1,
        position_embedding=True,
    ) -> None:
        state_shape = (state_shape[-1],)
        if action_shape != 0:
            action_shape = (1,)
        super(MetaNet, self).__init__(
            state_shape,
            action_shape,
            hidden_sizes,
            norm_layer,
            activation,
            device,
            softmax,
            concat,
            num_atoms,
            dueling_param,
            linear_layer,
        )
        self.self_attn = self_attn
        self.attn_pooling = attn_pooling
        self.position_embedding = position_embedding
        assert not (self.self_attn and self.attn_pooling)
        if self.position_embedding and (self.self_attn or self.attn_pooling):
            self.pos_encoder = PositionalEncoding(
                d_model=self.output_dim,
                dropout=dropout,
            )
        if self.self_attn:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            encoder_norm = nn.LayerNorm(self.output_dim)
            self.attn = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=encoder_norm,
            )
        elif self.attn_pooling:
            self.attn = AttentionPooling(d_model=self.output_dim, dropout=dropout)

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
        bsz, ch, d = obs.size(0), obs.size(1), obs.size(2)
        mask = obs.eq(MASK_VALUE).all(2).float()
        obs = obs.view(-1, d)
        logits, state = super().forward(obs, state, info)
        logits = logits.view(bsz, ch, -1)
        if self.position_embedding:
            position_ids = torch.tile(torch.arange(ch), [bsz, 1])
            logits += self.pos_encoder(position_ids)
        if self.self_attn:
            logits = self.attn(logits, src_key_padding_mask=mask)
        elif self.attn_pooling:
            logits = self.attn(logits, mask=1 - mask)
        return logits, state
