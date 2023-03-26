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
    def __init__(self, dim, stock_num):
        self.stock_num = stock_num
        self.dim = dim
        self.shape = (self.stock_num, self.dim)
        self.empty = np.zeros(self.shape, dtype=np.float32)

    def interpret(self, state: TradeStrategyState) -> np.ndarray:
        if state.feature is None:
            feature = self.empty
        else:
            feature = state.feature["feature"].values

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
    signal: pd.DataFrame
    topk: int
    n_drop: int
    hold_thresh: int


class TopkDropoutStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, int, TopkDropoutStrategyAction]
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        hold_thresh: int,
        stock_num: int,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        self.topk = topk
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.signal_key = signal_key
        self.stock_num = stock_num
        self.baseline = baseline


class TopkDropoutDiscreteDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.topk + 1)

    def interpret(
        self, state: TradeStrategyState, action: int
    ) -> TopkDropoutStrategyAction:
        assert 0 <= action <= self.topk
        n_drop = self.n_drop if self.baseline else int(action)
        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        return TopkDropoutStrategyAction(
            signal=signal, topk=self.topk, n_drop=n_drop, hold_thresh=self.hold_thresh
        )


class TopkDropoutDiscreteDynamicSelectionStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        hold_thresh: int,
        stock_num: int,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        super(
            TopkDropoutDiscreteDynamicSelectionStrategyActionInterpreter, self
        ).__init__(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            signal_key=signal_key,
            baseline=baseline,
            **kwargs,
        )
        sell_combinations = list(itertools.combinations(range(topk), n_drop))
        buy_combinations = list(itertools.combinations(range(topk, stock_num), n_drop))
        self.combinations = list(itertools.product(sell_combinations, buy_combinations))
        self.num_combinations = len(self.combinations)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_combinations)

    def interpret(
        self, state: TradeStrategyState, action: int
    ) -> TopkDropoutStrategyAction:
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
        return TopkDropoutStrategyAction(
            signal=signal,
            topk=self.topk,
            n_drop=self.n_drop,
            hold_thresh=self.hold_thresh,
        )


class TopkDropoutContinuousRerankStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=0 - np.inf, high=np.inf, shape=(self.stock_num,), dtype=np.float32
        )

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutStrategyAction:
        if state.feature is None:
            return TopkDropoutStrategyAction(signal=None)

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            signal.loc[:] = action

        return TopkDropoutStrategyAction(
            signal=signal,
            topk=self.topk,
            n_drop=self.n_drop,
            hold_thresh=self.hold_thresh,
        )


class TopkDropoutContinuousRerankDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=0 - np.inf, high=np.inf, shape=(self.stock_num,), dtype=np.float32
        )

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        n_drop = self.n_drop
        hold_thresh = self.hold_thresh
        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            signal.iloc[:] = action
            index = np.argpartition(-action, self.topk)[: self.topk]
            position = state.feature[("feature", "position")][: self.stock_num].copy()
            num_position = int(position.sum())
            if num_position > 0:
                hold = int((index < num_position).sum())
                n_drop = num_position - hold
            else:
                n_drop = 0
            hold_thresh = 1

        return TopkDropoutStrategyAction(
            signal=signal, topk=self.topk, n_drop=n_drop, hold_thresh=hold_thresh
        )


class TopkDropoutDiscreteRerankDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([self.stock_num] * self.stock_num)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        n_drop = self.n_drop
        hold_thresh = self.hold_thresh
        signal = state.feature[("feature", self.signal_key)][: self.stock_num].copy()
        if not self.baseline:
            rerank_index = action
            signal.iloc[rerank_index] = np.arange(self.stock_num - 1, -1, -1)
            position = state.feature[("feature", "position")][: self.stock_num].copy()
            num_position = int(position.sum())
            if num_position > 0:
                hold = int((rerank_index[: self.topk] < num_position).sum())
                n_drop = num_position - hold
            else:
                n_drop = 0
            hold_thresh = 1

        return TopkDropoutStrategyAction(
            signal=signal, topk=self.topk, n_drop=n_drop, hold_thresh=hold_thresh
        )


class WeightStrategyAction(NamedTuple):
    target_weight_position: Dict[str, float]


class WeightStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, Dict]
):
    def __init__(
        self,
        topk,
        stock_num,
        equal_weight=True,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        self.topk = topk
        self.stock_num = stock_num
        self.equal_weight = equal_weight
        self.signal_key = signal_key
        self.baseline = baseline

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=0 - np.inf, high=np.inf, shape=(self.stock_num,), dtype=np.float32
        )

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


class DirectSelectionStrategyActionInterpreter(WeightStrategyActionInterpreter):
    @property
    def action_space(self) -> spaces.MultiBinary:
        return spaces.MultiBinary((self.stock_num,))

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
