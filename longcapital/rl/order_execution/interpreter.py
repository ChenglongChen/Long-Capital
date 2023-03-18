import itertools
from typing import Dict, NamedTuple

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
        self.dim = dim
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
    def __init__(self, topk: int, n_drop: int, baseline=False) -> None:
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
    def __init__(
        self, stock_num, signal_key="signal", baseline=False, **kwargs
    ) -> None:
        self.stock_num = stock_num
        self.signal_key = signal_key
        self.baseline = baseline
        self.shape = (stock_num,)

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

        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            signal.loc[:] = action

        return TopkDropoutSignalStrategyAction(signal=signal)


class TopkDropoutSelectionStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, int, TopkDropoutSignalStrategyAction]
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        stock_num: int,
        signal_key="signal",
        baseline=False,
    ) -> None:
        self.topk = topk
        self.n_drop = n_drop
        self.stock_num = stock_num
        self.signal_key = signal_key
        self.baseline = baseline
        sell_combinations = list(itertools.combinations(range(topk), n_drop))
        buy_combinations = list(itertools.combinations(range(topk, stock_num), n_drop))
        self.combinations = list(itertools.product(sell_combinations, buy_combinations))
        self.num_combinations = len(self.combinations)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_combinations)

    def interpret(
        self, state: TradeStrategyState, action: int
    ) -> TopkDropoutSignalStrategyAction:
        assert 0 <= action < self.num_combinations
        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            stock_weight_dict = state.trade_executor.trade_account.current_position.get_stock_weight_dict(
                only_stock=False
            )
            current_position_list = list(stock_weight_dict.keys())
            if len(current_position_list):
                combinations = self.combinations[int(action)]
                sell, buy = combinations[0], combinations[1]
                signal.iloc[list(sell)] = -1000
                signal.iloc[list(buy)] = 1000

        return TopkDropoutSignalStrategyAction(signal=signal)


class WeightStrategyAction(NamedTuple):
    target_weight_position: Dict[str, float]


class WeightStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, Dict]
):
    def __init__(
        self,
        stock_num,
        topk=6,
        equal_weight=True,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        self.stock_num = stock_num
        self.topk = topk
        self.equal_weight = equal_weight
        self.signal_key = signal_key
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
        signal = state.feature[("feature", self.signal_key)].values
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


class DirectSelectionActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, Dict]
):
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


class TopkDropoutDynamicStrategyAction(NamedTuple):
    signal: pd.DataFrame
    topk: int
    n_drop: int


class TopkDropoutDynamicStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, int, TopkDropoutDynamicStrategyAction]
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        stock_num: int,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        self.topk = topk
        self.n_drop = n_drop
        self.signal_key = signal_key
        self.stock_num = stock_num
        self.baseline = baseline
        self.shape = (stock_num,)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(-100, 100, shape=self.shape, dtype=np.float32)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutDynamicStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        topk = self.topk
        n_drop = self.n_drop
        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            hold = np.zeros(self.stock_num, dtype=int)
            index = np.argpartition(-action, topk)[:topk]
            hold[index] = 1
            signal.iloc[:] = hold
            position = state.feature[("feature", "position")][: self.stock_num].copy()
            num_position = int(position.sum())
            if num_position > 0:
                n_drop = int((1 - hold[:num_position]).sum())
            else:
                n_drop = 0

        return TopkDropoutDynamicStrategyAction(signal=signal, topk=topk, n_drop=n_drop)
