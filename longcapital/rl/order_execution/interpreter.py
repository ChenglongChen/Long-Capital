import itertools
from collections.abc import Sequence
from typing import Dict, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
from gym import spaces
from longcapital.rl.order_execution.state import TradeStrategyState
from longcapital.rl.order_execution.utils import (
    filter_fake_stock,
    filter_nontradable_stock,
    softmax,
)
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter


class DynamicBox(spaces.Box):
    def contains(self, x) -> bool:
        from gym import logger

        if not isinstance(x, np.ndarray):
            logger.warn("Casting input x to numpy array.")
            x = np.asarray(x, dtype=self.dtype)

        return bool(
            np.can_cast(x.dtype, self.dtype)
            # ignore shape check
            # and x.shape == self.shape
            and np.all(x >= self.low[: x.shape[0]])
            and np.all(x <= self.high[: x.shape[0]])
        )


class TradeStrategyStateInterpreter(StateInterpreter[TradeStrategyState, np.ndarray]):
    def __init__(self, dim, stock_num=300, **kwargs):
        self.dim = dim
        self.stock_num = stock_num
        self.shape = (self.stock_num, self.dim)

    def interpret(self, state: TradeStrategyState) -> np.ndarray:
        return state.feature["feature"].values.astype(np.float32)

    @property
    def observation_space(self) -> DynamicBox:
        return DynamicBox(0 - np.inf, np.inf, shape=self.shape, dtype=np.float32)


class TopkDropoutStrategyAction(NamedTuple):
    signal: pd.DataFrame
    topk: int
    n_drop: int
    hold_thresh: int
    ready: bool = True


class TopkDropoutStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, int, TopkDropoutStrategyAction]
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        hold_thresh: int,
        stock_num: int,
        rerank_topk=True,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        self.topk = topk
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.rerank_topk = rerank_topk
        self.signal_key = signal_key
        self.stock_num = stock_num
        self.baseline = baseline


class TopkDropoutDiscreteDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete((self.topk + 1) * (self.topk))

    def interpret(
        self, state: TradeStrategyState, action: int
    ) -> TopkDropoutStrategyAction:
        assert 0 <= action < (self.topk + 1) * (self.topk)
        n_drop = self.n_drop if self.baseline else int(action % (self.topk + 1))
        hold_thresh = (
            self.hold_thresh if self.baseline else int(action / (self.topk + 1)) + 1
        )
        topk = state.initial_state.topk
        signal = state.signal
        return TopkDropoutStrategyAction(
            signal=signal, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh
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
        rerank_topk=True,
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
            rerank_topk=rerank_topk,
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
        topk = state.initial_state.topk
        signal = state.signal
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
            topk=topk,
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

        topk = state.initial_state.topk
        signal = state.signal
        if not self.baseline:
            signal.loc[:] = action

        return TopkDropoutStrategyAction(
            signal=signal,
            topk=topk,
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

        topk = state.initial_state.topk
        signal = state.signal
        position = state.position
        n_drop = self.n_drop
        hold_thresh = self.hold_thresh
        if not self.baseline:
            index = np.argpartition(-action, topk)[:topk]
            if self.rerank_topk:
                # only select topk, the rest are sorted by original signal (i.e., baseline)
                signal.iloc[:] = float(np.min(action))
                signal.iloc[index] = action[index]
            else:
                # rerank for the whole stock pool
                signal.iloc[:] = action
            # get dynamic params
            hold_thresh = 1
            num_position = int(position.sum())
            if num_position > 0:
                hold = int((index < num_position).sum())
                n_drop = num_position - hold
            else:
                n_drop = 0
        return TopkDropoutStrategyAction(
            signal=signal, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh
        )


class DynamicMultiDiscrete(spaces.MultiDiscrete):
    def contains(self, x) -> bool:
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool((0 <= x).all() and (x < self.nvec[: x.shape[0]]).all())


class TopkDropoutDiscreteRerankDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    @property
    def action_space(self) -> DynamicMultiDiscrete:
        return DynamicMultiDiscrete([self.stock_num + 1] * self.stock_num)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        topk = state.initial_state.topk
        stock_num = state.initial_state.stock_num
        topk = state.initial_state.topk
        stock_num = state.initial_state.stock_num
        signal = state.signal
        position = state.position
        n_drop = self.n_drop
        hold_thresh = self.hold_thresh
        if not self.baseline:
            if self.rerank_topk:
                # only select topk, the rest are sorted by original signal (i.e., baseline)
                signal.iloc[:] = 0
                signal.iloc[action[:topk]] = np.arange(topk, 0, -1)
            else:
                # rerank for the whole stock pool
                signal.iloc[action] = np.arange(stock_num, 0, -1)
            # get dynamic params
            hold_thresh = 1
            num_position = int(position.sum())
            if num_position > 0:
                hold = int((action[:topk] < num_position).sum())
                n_drop = num_position - hold
            else:
                n_drop = 0
        return TopkDropoutStrategyAction(
            signal=signal,
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
        )


class TopkDropoutStepByStepDiscreteRerankDynamicParamStrategyActionInterpreter(
    TopkDropoutStrategyActionInterpreter
):
    def __init__(
        self,
        topk: int,
        n_drop: int,
        hold_thresh: int,
        stock_num: int,
        rerank_topk=True,
        signal_key="signal",
        baseline=False,
        **kwargs,
    ) -> None:
        super(
            TopkDropoutStepByStepDiscreteRerankDynamicParamStrategyActionInterpreter,
            self,
        ).__init__(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            rerank_topk=rerank_topk,
            signal_key=signal_key,
            baseline=baseline,
            **kwargs,
        )
        self.selected_stock_indices = []

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.stock_num)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> TopkDropoutStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        topk = state.initial_state.topk
        stock_num = state.initial_state.stock_num
        signal = state.signal
        position = state.position
        n_drop = self.n_drop
        hold_thresh = self.hold_thresh
        ready = True
        if not self.baseline:
            ready = False
            self.selected_stock_indices.append(int(action))
            if len(self.selected_stock_indices) == (
                topk if self.rerank_topk else stock_num
            ):
                ready = True
                selected_stock_indices = [
                    s for s in self.selected_stock_indices if s < len(state.feature)
                ]
                signal.iloc[:] = 0
                signal.iloc[selected_stock_indices] = np.arange(
                    len(selected_stock_indices), 0, -1
                )
                # get dynamic params
                hold_thresh = 1
                num_position = int(position.sum())
                if num_position > 0:
                    hold = int(
                        (np.array(selected_stock_indices[:topk]) < num_position).sum()
                    )
                    n_drop = num_position - hold
                else:
                    n_drop = 0
                self.selected_stock_indices.clear()
        state.info["ready"] = ready
        return TopkDropoutStrategyAction(
            signal=signal,
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            ready=ready,
        )


class WeightStrategyAction(NamedTuple):
    signal: pd.DataFrame
    target_weight_position: Dict[str, float]
    ready: bool = True


class WeightStrategyActionInterpreter(
    ActionInterpreter[TradeStrategyState, np.ndarray, Dict]
):
    def __init__(
        self,
        topk,
        stock_num,
        only_tradable,
        equal_weight=False,
        signal_key="signal",
        baseline=False,
        normalize="sum",
        **kwargs,
    ) -> None:
        self.topk = topk
        self.stock_num = stock_num
        self.only_tradable = only_tradable
        self.equal_weight = equal_weight
        self.signal_key = signal_key
        self.baseline = baseline
        self.normalize = normalize
        # for step-by-step action
        self.selected_stock_indices = []

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
        topk = state.initial_state.topk
        stock_num = state.initial_state.stock_num
        signal = state.signal
        stocks = state.feature.index[:stock_num]
        weights = signal.values if self.baseline else action

        # filter fake stocks
        stocks, weights = filter_fake_stock(state, stocks, weights)

        # filter non-tradable stocks
        if self.only_tradable:
            stocks, weights = filter_nontradable_stock(state, stocks, weights)

        if len(stocks) == 0:
            return WeightStrategyAction(target_weight_position={})

        # only select topk
        topk = min(topk, len(stocks))
        if topk < len(stocks):
            index = np.argpartition(-weights, topk)[:topk]
            stocks = stocks[index]
            weights = weights[index]

        # normalize
        if self.normalize == "softmax":
            weights = softmax(weights)
        else:
            weights[weights < 0] = 0
            weights /= weights.sum()

        # select positive weights
        index = weights > 0
        stocks, weights = stocks[index], weights[index]

        # assign weight
        w = 1.0 / topk
        target_weight_position = {
            stock: w if self.equal_weight else weight
            for stock, weight in zip(stocks, weights)
        }

        return WeightStrategyAction(
            signal=signal, target_weight_position=target_weight_position
        )


class StepByStepStrategyActionInterpreter(WeightStrategyActionInterpreter):
    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.stock_num + 1)

    def interpret(
        self, state: TradeStrategyState, action: torch.Tensor
    ) -> WeightStrategyAction:

        if isinstance(action, torch.Tensor):
            action = action.squeeze().detach().numpy()

        topk = state.initial_state.topk
        signal = state.signal
        stocks = []
        ready = False
        if self.baseline:
            selected_stock_indices = np.argpartition(-signal.values, topk)[:topk]
            stocks = state.feature.index[selected_stock_indices]
            ready = True
        else:
            self.selected_stock_indices.append(int(action))
            if len(self.selected_stock_indices) == topk:
                selected_stock_indices = [
                    s
                    for s in self.selected_stock_indices
                    if s < len(state.feature.index)
                ]
                stocks = state.feature.index[selected_stock_indices]
                self.selected_stock_indices.clear()
                ready = True

        weights = [1.0 / topk] * len(stocks)

        # filter fake stocks
        stocks, weights = filter_fake_stock(state, stocks, weights)

        # filter non-tradable stocks
        if self.only_tradable:
            stocks, weights = filter_nontradable_stock(state, stocks, weights)

        # assign weight
        target_weight_position = dict(zip(stocks, weights))

        return WeightStrategyAction(
            signal=signal, target_weight_position=target_weight_position, ready=ready
        )
