import math
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from longcapital.utils.constant import EPS, MASK_VALUE, NEG_INF
from tianshou.utils.net.common import MLP
from torch import Tensor, nn

ModuleType = Type[nn.Module]


def get_shape(x: Tensor):
    shape = list(x.size())
    return shape


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
        y = (x * (1 - mask)).sum(1) / ((1 - mask).sum(1) + EPS)
        return y


class MaxPooling(Pooling):
    """Max Pooling over the last but one dimension"""

    def pooling(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # check has at least one non-mask value
        valid = mask.sum(1) > 0
        m = mask.unsqueeze(2).transpose(1, 2)
        y = x.transpose(1, 2)
        s = int(m.size(2))
        z = torch.max_pool1d((1 - m) * y + m * NEG_INF, s)
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
        attn = (1 - m) * attn + m * NEG_INF
        attn = self.dropout(F.softmax(attn, dim=1))
        x = (v * attn * (1 - m)).sum(1)
        return x


class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_layers=6,
        nhead=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )
        self.output_dim = d_model

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)


class MetaNet(nn.Module):
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
        use_dueling: bool = False,
        num_atoms: int = 1,
        linear_layer: Type[nn.Linear] = nn.Linear,
        attn_pooling=False,
        self_attn=False,
        nhead=2,
        num_layers=6,
        dropout=0.1,
        position_embedding=True,
    ) -> None:
        super().__init__()
        state_shape = (state_shape[-1],)
        if action_shape != 0:
            action_shape = (1,)
        self.device = device
        self.softmax = softmax
        self.use_dueling = use_dueling
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.mlp = MLP(
            input_dim,
            output_dim,
            hidden_sizes,
            norm_layer,
            activation,
            device,
            linear_layer,
        )
        self.output_dim = self.mlp.output_dim
        self.self_attn = self_attn
        self.attn_pooling = attn_pooling
        self.position_embedding = position_embedding
        assert not (self.self_attn and self.attn_pooling)
        if self.position_embedding:
            self.pe = PositionalEncoding(
                d_model=self.output_dim,
                dropout=dropout,
            )
        if self.self_attn:
            self.attn = Transformer(
                d_model=self.output_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
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
        logits = self.mlp(obs)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        logits = logits.view(bsz, ch, -1)
        if self.position_embedding:
            logits = self.pe(logits)
        if self.self_attn:
            logits = self.attn(logits, src_key_padding_mask=mask)
        elif self.attn_pooling:
            logits = self.attn(logits, mask=mask)
        return logits, state
