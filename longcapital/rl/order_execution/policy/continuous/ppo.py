from pathlib import Path
from typing import List, Optional

import gym
import torch
from longcapital.rl.utils.net.common import MetaNet
from longcapital.rl.utils.net.continuous import MetaActorProb, MetaCritic
from qlib.rl.order_execution.policy import Trainer, auto_device, set_weight
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from torch.distributions import Independent, Normal


class MetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [64, 64, 64],
        lr: float = 1e-4,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 1.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        max_action: float = 1.0,
        weight_file: Optional[Path] = None,
    ) -> None:

        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActorProb(
            net,
            action_space.shape,
            max_action=max_action,
            device=auto_device(net),
        ).to(auto_device(net))

        net = MetaNet(
            obs_space.shape,
            action_space.shape,
            hidden_sizes=hidden_sizes,
            attn_pooling=True,
        )
        critic = MetaCritic(net, device=auto_device(net)).to(auto_device(net))
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # replace DiagGuassian with Independent(Normal) which is equivalent
        # pass *logits to be consistent with policy.forward
        def dist(*logits) -> torch.distributions.Distribution:
            return Independent(Normal(*logits), 1)

        super().__init__(
            actor,
            critic,
            optim,
            dist,
            discount_factor=discount_factor,
            max_grad_norm=max_grad_norm,
            reward_normalization=reward_normalization,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            dual_clip=dual_clip,
            eps_clip=eps_clip,
            value_clip=value_clip,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            max_batchsize=max_batch_size,
            deterministic_eval=deterministic_eval,
            observation_space=obs_space,
            action_space=action_space,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def __str__(self):
        return "MetaPPO"
