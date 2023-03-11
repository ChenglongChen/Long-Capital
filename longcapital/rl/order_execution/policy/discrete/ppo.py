from pathlib import Path
from typing import List, Optional

import gym
import torch
from longcapital.rl.utils.net.common import MetaNet
from qlib.rl.order_execution.policy import Trainer, auto_device, chain_dedup, set_weight
from tianshou.policy import PPOPolicy
from tianshou.utils.net.discrete import Actor, Critic


class PPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [32, 16, 8],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 1.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
    ) -> None:
        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, attn_pooling=True)
        actor = Actor(net, action_space.shape, device=auto_device(net)).to(
            auto_device(net)
        )

        net = MetaNet(
            obs_space.shape,
            action_space.shape,
            hidden_sizes=hidden_sizes,
            attn_pooling=True,
        )
        critic = Critic(net, device=auto_device(net)).to(auto_device(net))

        optimizer = torch.optim.Adam(
            chain_dedup(actor.parameters(), critic.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(
            actor,
            critic,
            optimizer,
            torch.distributions.Categorical,
            discount_factor=discount_factor,
            max_grad_norm=max_grad_norm,
            reward_normalization=reward_normalization,
            eps_clip=eps_clip,
            value_clip=value_clip,
            vf_coef=vf_coef,
            gae_lambda=gae_lambda,
            max_batchsize=max_batch_size,
            deterministic_eval=deterministic_eval,
            observation_space=obs_space,
            action_space=action_space,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))
