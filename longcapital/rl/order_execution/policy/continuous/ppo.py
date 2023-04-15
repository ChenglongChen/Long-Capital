from pathlib import Path
from typing import Any, Dict, List, Optional

import gym
import torch
import torch.nn.functional as F  # noqa
from longcapital.rl.utils.net.common import MetaNet
from longcapital.rl.utils.net.continuous import MetaActorProb, MetaCritic
from qlib.rl.order_execution.policy import Trainer, auto_device, set_weight
from tianshou.data import Batch, to_torch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from torch import nn
from torch.distributions import Distribution, Independent, Normal


class MetaPPO(PPOPolicy):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        hidden_sizes: List[int] = [32, 16, 8],
        lr: float = 3e-4,
        discount_factor: float = 1.0,
        max_grad_norm: float = 100.0,
        reward_normalization: bool = True,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        dual_clip: float = None,
        eps_clip: float = 0.3,
        value_clip: bool = True,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 1.0,
        max_batch_size: int = 256,
        deterministic_eval: bool = True,
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        sigma_min: float = 1e-8,
        sigma_max: float = 1.0,
        action_scaling: bool = False,
        action_bound_method: str = "",
        weight_file: Optional[Path] = None,
        **kwargs,
    ) -> None:

        net = MetaNet(obs_space.shape, hidden_sizes=hidden_sizes, self_attn=True)
        actor = MetaActorProb(
            net,
            action_space.shape,
            max_action=max_action,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
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
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # replace DiagGuassian with Independent(Normal) which is equivalent
        # pass *logits to be consistent with policy.forward
        def dist(*logits) -> Distribution:
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
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
        )
        if weight_file is not None:
            set_weight(self, Trainer.get_policy_state_dict(weight_file))

    def learn_imitation(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        for step in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                act = self(minibatch).logits[0]  # mu
                act_target = minibatch.info.aux_info["label"]
                act_target = to_torch(
                    act_target, dtype=torch.float32, device=act.device
                )
                loss = F.mse_loss(act, act_target)
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                losses.append(loss.item())
        return {"loss": losses}

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        current_iter = kwargs.get("current_iter")
        imitation_iter = kwargs.get("imitation_iter")
        if (
            (current_iter is not None)
            and (imitation_iter is not None)
            and (current_iter < imitation_iter)
        ):
            return self.learn_imitation(batch, batch_size, repeat, **kwargs)
        else:
            return super(MetaPPO, self).learn(batch, batch_size, repeat, **kwargs)

    def __str__(self):
        return "MetaPPO"
