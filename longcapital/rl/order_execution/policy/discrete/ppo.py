from pathlib import Path
from typing import Any, List, Optional, Union

import gym
import numpy as np
import torch
from longcapital.rl.utils.distributions import MultivariateHypergeometric
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
        discount_factor: float = 0.95,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
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
        discount_factor: float = 0.95,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
        **kwargs,
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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(-1)
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def __str__(self):
        return "MetaPPO"


class StepByStepMetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        softmax_output: bool = True,
        hidden_sizes: List[int] = [32, 16, 8],
        lr: float = 1e-4,
        discount_factor: float = 0.95,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
        **kwargs,
    ) -> None:

        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActor(
            net,
            [action_space.n],
            softmax_output=softmax_output,
            device=auto_device(net),
        ).to(auto_device(net))

        net = MetaNet(
            obs_space.shape,
            [action_space.n],
            hidden_sizes=hidden_sizes,
            attn_pooling=True,
        )
        critic = MetaCritic(net, device=auto_device(net)).to(auto_device(net))
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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        selected = torch.Tensor(batch.obs[:, :, -1])
        mask_value = 0
        logits = logits * (1 - selected) + mask_value * selected
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(-1)
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def __str__(self):
        return "StepByStepMetaPPO"


class TopkMetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        softmax_output: bool = True,
        hidden_sizes: List[int] = [32, 16, 8],
        lr: float = 1e-4,
        discount_factor: float = 0.95,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
        **kwargs,
    ) -> None:
        self.topk = kwargs.get("topk", 1)

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

        def dist(logits) -> torch.distributions.Distribution:
            return MultivariateHypergeometric(probs=logits, topk=self.topk)

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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = torch.argsort(logits, dim=1, descending=True)[:, : self.topk]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def __str__(self):
        return "TopkMetaPPO"


class MyMultinomial(torch.distributions.Multinomial):
    def entropy(self):
        return torch.tensor(0.0)


class WeightMetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        softmax_output: bool = True,
        hidden_sizes: List[int] = [32, 16, 8],
        lr: float = 1e-4,
        discount_factor: float = 0.95,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.0,
        ent_coef: float = 0.0,
        gae_lambda: float = 0.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        weight_file: Optional[Path] = None,
        total_count: int = 1000,
        **kwargs,
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

        def dist(logits) -> torch.distributions.Distribution:
            return MyMultinomial(total_count=total_count, probs=logits)

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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def __str__(self):
        return "WeightMetaPPO"
