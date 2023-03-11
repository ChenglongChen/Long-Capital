from typing import Dict, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from gym import spaces
from longcapital.rl.order_execution.state import TradeStrategyState
from longcapital.rl.order_execution.utils import filter_stock, softmax
from longcapital.rl.utils.net.common import MASK_VALUE
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter


class TradeStrategyStateInterpreter(StateInterpreter[TradeStrategyState, np.ndarray]):
    def __init__(self, dim, stock_num=300):
        self.stock_num = stock_num
        self.dim = dim + 1 + 1 + 1
        self.shape = (self.stock_num, self.dim)
        self.empty = np.zeros(self.shape, dtype=np.float32)

    def interpret(self, state: TradeStrategyState) -> np.ndarray:
        if state.feature is None:
            feature = self.empty
        else:
            feature = state.feature.values

        # padding
        if feature.shape[0] < self.stock_num:
            pad_size = self.stock_num - feature.shape[0]
            feature = np.vstack([feature, MASK_VALUE * np.ones((pad_size, self.dim))])

        feature = feature[: self.stock_num, :]
        return np.array(feature, dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0 - np.inf, np.inf, shape=self.shape, dtype=np.float32)


class TopkDropoutStrategyAction(NamedTuple):
    n_drop: int


class TopkDropoutStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, int, TopkDropoutStrategyAction]
):
    def __init__(self, topk: int, n_drop: Optional[int] = None, baseline=False) -> None:
        self.topk = topk
        self.n_drop = n_drop
        self.baseline = baseline

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.topk + 1)

    def interpret(
        self, state: TradeStrategyState, action: int
    ) -> TopkDropoutStrategyAction:
        assert 0 <= action <= self.topk
        n_drop = self.n_drop if self.baseline else int(action)
        return TopkDropoutStrategyAction(n_drop=n_drop)


class TopkDropoutSignalStrategyAction(NamedTuple):
    signal: pd.DataFrame


class TopkDropoutSignalStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, TopkDropoutSignalStrategyAction]
):
    def __init__(self, stock_num, baseline=False, **kwargs) -> None:
        self.stock_num = stock_num
        self.shape = (stock_num,)
        self.baseline = baseline

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(-100, 100, shape=self.shape, dtype=np.float32)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutSignalStrategyAction:
        if state.feature is None:
            return TopkDropoutSignalStrategyAction(signal=None)

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        signal = state.feature[("feature", "signal")][: self.stock_num].copy()
        if not self.baseline:
            signal.loc[:] = action

        return TopkDropoutSignalStrategyAction(signal=signal)


class WeightStrategyAction(NamedTuple):
    target_weight_position: Dict[str, float]


class WeightStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, Dict]
):
    def __init__(
        self, stock_num, topk=6, equal_weight=True, baseline=False, **kwargs
    ) -> None:
        self.stock_num = stock_num
        self.topk = topk
        self.equal_weight = equal_weight
        self.baseline = baseline
        self.shape = (stock_num,)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(-100, 100, shape=self.shape, dtype=np.float32)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> WeightStrategyAction:
        if state.feature is None:
            return WeightStrategyAction(target_weight_position={})

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        # stocks & weights
        stocks = state.feature.index[: self.stock_num]
        signal = state.feature[("feature", "signal")].values
        weights = signal if self.baseline else action

        # filter non-tradable stocks
        stocks, weights = filter_stock(state, stocks, weights)

        if len(stocks) == 0:
            return WeightStrategyAction(target_weight_position={})

        # only select topk
        topk = min(self.topk, len(stocks))
        if topk < len(stocks):
            index = np.argpartition(-weights, topk)[:topk]
            stocks = stocks[index]
            weights = weights[index]

        weights = softmax(weights)

        # assign weight
        target_weight_position = {
            stock: 1.0 / len(stocks) if self.equal_weight else weight
            for stock, weight in zip(stocks, weights)
        }

        return WeightStrategyAction(target_weight_position=target_weight_position)


class TopkActionInterpreter(ActionInterpreter[TradeStrategyState, np.ndarray, Dict]):
    def __init__(self, stock_num, baseline=False, **kwargs) -> None:
        self.stock_num = stock_num
        self.baseline = baseline
        self.shape = stock_num

    @property
    def action_space(self) -> spaces.MultiBinary:
        return spaces.MultiBinary(self.shape)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> WeightStrategyAction:
        if state.feature is None:
            return WeightStrategyAction(target_weight_position={})

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        stocks = state.feature.index[: self.stock_num]
        weights = np.ones(len(stocks)) if self.baseline else action

        # filter non-tradable stocks
        stocks, weights = filter_stock(state, stocks, weights)

        if len(stocks) == 0:
            return WeightStrategyAction(target_weight_position={})

        # select
        index = weights > 0
        stocks, weights = stocks[index], weights[index]

        # assign weight
        target_weight_position = {
            stock: 1.0 / len(stocks) for stock, weight in zip(stocks, weights)
        }

        return WeightStrategyAction(target_weight_position=target_weight_position)
