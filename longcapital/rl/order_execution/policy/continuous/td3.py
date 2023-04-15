from pathlib import Path
from typing import List, Optional

import gym
import torch
from longcapital.rl.utils.net.common import MetaNet
from longcapital.rl.utils.net.continuous import MetaActor, MetaCritic
from qlib.rl.order_execution.policy import Trainer, auto_device, set_weight
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy


class MetaTD3(TD3Policy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [32, 16, 8],
        reward_normalization: bool = False,
        n_step: int = 3,
        gamma: float = 1.0,
        tau: float = 0.05,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        exploration_noise: float = 0.1,
        max_action: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        action_scaling: bool = False,
        action_bound_method: str = "",
        weight_file: Optional[Path] = None,
        **kwargs,
    ) -> None:
        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActor(
            net, action_space.shape, max_action=max_action, device=auto_device(net)
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

        super().__init__(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            reward_normalization=reward_normalization,
            estimation_step=n_step,
            action_space=action_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )

        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def __str__(self):
        return "MetaTD3"
