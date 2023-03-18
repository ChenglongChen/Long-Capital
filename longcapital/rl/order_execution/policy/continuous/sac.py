from pathlib import Path
from typing import List, Optional

import gym
import numpy as np
import torch
from longcapital.rl.utils.net.common import MetaNet
from longcapital.rl.utils.net.continuous import MetaActorProb, MetaCritic
from qlib.rl.order_execution.policy import Trainer, auto_device, set_weight
from tianshou.policy import SACPolicy


class MetaSAC(SACPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [64, 64, 64],
        n_step: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha_lr: float = 3e-4,
        max_action: float = 1.0,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        unbounded: bool = True,
        conditioned_sigma: bool = True,
        weight_file: Optional[Path] = None,
    ) -> None:
        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActorProb(
            net,
            action_space.shape,
            max_action=max_action,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            device=auto_device(net),
        ).to(auto_device(net))
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        net_c1 = MetaNet(
            obs_space.shape,
            action_space.shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            attn_pooling=True,
        )
        critic1 = MetaCritic(net_c1, device=auto_device(net_c1)).to(auto_device(net_c1))
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)

        net_c2 = MetaNet(
            obs_space.shape,
            action_space.shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            attn_pooling=True,
        )
        critic2 = MetaCritic(net_c2, device=auto_device(net_c2)).to(auto_device(net_c2))
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        if auto_alpha:
            target_entropy = -np.prod(action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=auto_device(net))
            alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)

        super().__init__(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            estimation_step=n_step,
            action_space=action_space,
        )

        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def __str__(self):
        return "MetaSAC"
