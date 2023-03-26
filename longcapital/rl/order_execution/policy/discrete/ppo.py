from pathlib import Path
from typing import Any, List, Optional, Union

import gym
import numpy as np
import torch
from longcapital.rl.utils.distributions import Categorical
from longcapital.rl.utils.net.common import MetaNet
from longcapital.rl.utils.net.discrete import MetaActor, MetaCritic
from qlib.rl.order_execution.policy import Trainer, auto_device, set_weight
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic


class PPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [32, 16, 8],
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
        weight_file: Optional[Path] = None,
    ) -> None:
        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, attn_pooling=True)
        actor = Actor(net, action_space.n, device=auto_device(net)).to(auto_device(net))

        net = MetaNet(
            obs_space.shape,
            action_space.shape,
            hidden_sizes=hidden_sizes,
            attn_pooling=True,
        )
        critic = Critic(net, device=auto_device(net)).to(auto_device(net))
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        super().__init__(
            actor,
            critic,
            optim,
            torch.distributions.Categorical,
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
        return "PPO"


class MetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        softmax_output: bool = True,
        hidden_sizes: List[int] = [32, 16, 8],
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
        weight_file: Optional[Path] = None,
    ) -> None:
        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActor(
            net,
            action_space.shape,
            softmax_output=softmax_output,
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
            return torch.distributions.Independent(Categorical(*logits), 1)

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

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.
        :return: A :class:`~tianshou.data.Batch` which has 4 keys:
            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = torch.argsort(torch.argsort(logits, dim=1), dim=1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:

            def get_rank(index):
                """
                :param index: index for the ranked values, e.g., [2, 0, 1]
                :return: new rank for the original index, e.g., [1, 0, 2]
                """
                bsz, ch = index.size(0), index.size(1)
                _, indices = torch.sort(index, dim=1)
                rank_arr = torch.arange(start=ch - 1, end=-1, step=-1)
                rank_arr = rank_arr.tile([bsz, 1])
                rank = rank_arr.gather(1, indices)
                return rank

            act = get_rank(dist.sample())
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def __str__(self):
        return "MetaPPO"
