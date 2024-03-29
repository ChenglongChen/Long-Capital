import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.aux_info import ImitationLabelCollector
from longcapital.rl.order_execution.buffer import FeatureBuffer
from longcapital.rl.order_execution.interpreter import (
    StepByStepStrategyActionInterpreter,
    TopkDropoutContinuousRerankDynamicParamStrategyActionInterpreter,
    TopkDropoutContinuousRerankStrategyActionInterpreter,
    TopkDropoutDiscreteDynamicParamStrategyActionInterpreter,
    TopkDropoutDiscreteDynamicSelectionStrategyActionInterpreter,
    TopkDropoutDiscreteMultiDynamicParamStrategyActionInterpreter,
    TopkDropoutDiscreteRerankDynamicParamStrategyActionInterpreter,
    TopkDropoutStepByStepDiscreteRerankDynamicParamStrategyActionInterpreter,
    TopkDropoutStrategyAction,
    TradeStrategyStateInterpreter,
    WeightStrategyAction,
    WeightStrategyActionInterpreter,
)
from longcapital.rl.order_execution.policy import continuous, discrete
from longcapital.rl.order_execution.state import (
    TradeStrategyInitialState,
    TradeStrategyState,
)
from longcapital.utils.constant import FAKE_STOCK, MASK_VALUE
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy import TopkDropoutStrategy as TopkDropoutStrategyBase
from qlib.contrib.strategy import WeightStrategyBase
from qlib.data import D
from qlib.rl.aux_info import AuxiliaryInfoCollector
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.strategy.base import BaseStrategy
from qlib.utils.time import epsilon_change
from tianshou.data import Batch
from tianshou.policy import BasePolicy


class BaseTradeStrategy(BaseStrategy):
    state_interpreter: StateInterpreter
    action_interpreter: ActionInterpreter
    baseline_action_interpreter: ActionInterpreter
    aux_info_collector: AuxiliaryInfoCollector
    policy: BasePolicy
    feature_buffer: FeatureBuffer
    position_feature_cols: List
    signal: Any
    raw_signal: pd.DataFrame
    signal_key: str = "signal"
    imitation_label_key: Tuple[str] = ("feature", "signal")
    initial_state: TradeStrategyInitialState
    stock_pool: List[str]
    price_dict: dict

    def action(
        self,
        baseline: bool = False,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> Any:
        """take necessary number of steps before ready for taking step in env"""
        ready = False
        while not ready:
            action = self.one_step_action(
                baseline=baseline,
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
            )
            ready = action.ready
        return action

    def one_step_action(
        self,
        baseline: bool = False,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> Any:
        state = self.get_state(
            pred_start_time=pred_start_time, pred_end_time=pred_end_time
        )
        obs = [{"obs": self.state_interpreter.interpret(state), "info": {}}]
        policy_out = self.policy(Batch(obs))
        if baseline:
            action = self.baseline_action_interpreter.interpret(state, policy_out.act)
        else:
            action = self.action_interpreter.interpret(state, policy_out.act)
        return action

    def get_state(
        self,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> TradeStrategyState:
        feature = self.get_feature(
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
        )
        state = TradeStrategyState(
            trade_executor=self.executor,
            trade_strategy=self,
            initial_state=self.initial_state,
            feature=feature,
            label=self.get_label_from_feature(feature),
            signal=self.get_signal_from_feature(feature),
            position=self.get_position_from_feature(feature),
        )
        return state

    def get_feature(
        self,
        feature=None,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        f = self._get_feature(feature, pred_start_time, pred_end_time)
        self.feature_buffer.add(f)
        return self.feature_buffer.collect()

    def _get_feature(
        self,
        feature=None,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> Union[pd.DataFrame, None]:
        if feature is None:
            if pred_start_time is None or pred_end_time is None:
                pred_start_time, pred_end_time = self.get_pred_start_end_time()
            feature = self.signal.get_signal(
                start_time=pred_start_time, end_time=pred_end_time
            )
        if feature is None:
            return None

        position_df = self.get_position_df(rename_cols=True)
        if position_df is None:
            for k in self.position_feature_cols:
                feature[("feature", k)] = 0
        else:
            feature = pd.merge(feature, position_df, on="instrument", how="left")
            feature.fillna(0, inplace=True)

        # return metrics
        feature[("feature", "return")] = 0
        held = feature[("feature", "position")] == 1
        held_stocks = feature[held].index.tolist()
        count_days = feature[held][("feature", "count_day")]
        for stock, count_day in zip(held_stocks, count_days):
            curr_price = feature.loc[stock][("meta", "price")]
            trade_step, _ = self.trade_calendar.get_range_idx(
                pred_start_time, pred_end_time
            )
            prev_start_time, prev_end_time = self.trade_calendar.get_step_time(
                trade_step, shift=int(count_day)
            )
            prev_price = self.signal.get_signal(prev_start_time, prev_end_time).loc[
                stock
            ][("meta", "price")]
            feature[("feature", "return")].loc[stock] = curr_price / prev_price - 1.0

        # selected flag
        feature[("feature", "selected")] = 0
        if hasattr(self.action_interpreter, "selected_stock_indices"):
            selected_stock_indices = [
                s
                for s in self.action_interpreter.selected_stock_indices
                if s < len(feature)
            ]
            feature[("feature", "selected")].iloc[selected_stock_indices] = 1

        # sort to make sure the ranking distribution is similar across different dates
        feature.sort_values(
            by=[("feature", "position"), ("feature", self.signal_key)],
            ascending=False,
            inplace=True,
        )

        # sample stocks
        stock_pool = self.sample_stocks(stocks=feature.index.tolist())
        feature = feature[feature.index.isin(stock_pool)]

        # padding
        bsz, dim = feature.shape[0], feature.shape[1]
        if bsz < self.initial_state.stock_num:
            pad_size = self.initial_state.stock_num - bsz
            padding = pd.DataFrame(
                MASK_VALUE * np.ones((pad_size, dim)),
                columns=feature.columns,
                index=pd.Index([FAKE_STOCK] * pad_size, name="instrument"),
            )
            feature = pd.concat([feature, padding], axis=0)

        return feature

    def get_label_from_feature(self, feature: pd.DataFrame) -> pd.DataFrame:
        label = feature[self.imitation_label_key][: self.initial_state.stock_num].copy()
        return label

    def get_signal_from_feature(self, feature: pd.DataFrame) -> pd.DataFrame:
        signal = feature[("feature", self.signal_key)][
            : self.initial_state.stock_num
        ].copy()
        return signal

    def get_position_from_feature(self, feature: pd.DataFrame) -> pd.DataFrame:
        position = feature[("feature", "position")][
            : self.initial_state.stock_num
        ].copy()
        return position

    def trade(self, baseline=False) -> pd.DataFrame:
        decision_df = self._trade()
        if baseline:
            decision_df.drop(["direction", "weight", "signal"], axis=1, inplace=True)
            decision_df.columns = [
                "instrument",
                "direction",
                "weight",
                "signal",
                "position",
            ]
            decision_df.sort_values(
                ["position", "signal"], ascending=False, inplace=True
            )
            decision_df.reset_index(drop=True, inplace=True)
        return decision_df

    def _trade(self) -> pd.DataFrame:
        trade_step = self.trade_calendar.get_trade_len() - 1
        pred_start_time = self.trade_calendar.get_step_start_time(trade_step=trade_step)
        pred_end_time = epsilon_change(pred_start_time + pd.Timedelta(days=1))
        # mock fake trade_start_time and trade_end_time as placeholders
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
            trade_step=trade_step, shift=1
        )
        action_dfs = []
        for baseline in [True, False]:
            suffix = "_baseline" if baseline else ""
            action = self.action(
                baseline=baseline,
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
            )
            sell_order_list, buy_order_list = self.generate_trade_decision(
                execute_result=None,
                action=action,
                trade_start_time=trade_start_time,
                trade_end_time=trade_end_time,
                return_decision=False,
            )
            order_list = sell_order_list + buy_order_list

            action_df = action.signal.to_frame().reset_index()
            action_df.columns = ["instrument", "signal"]

            if len(order_list):
                order_df = pd.DataFrame(order_list)[["stock_id", "direction"]]
                order_df.columns = ["instrument", "direction"]
                order_df["direction"][order_df["direction"] == 0] = -1
                action_df = pd.merge(action_df, order_df, on="instrument", how="left")
            else:
                action_df["direction"] = np.nan

            # assign weight
            if hasattr(action, "target_weight_position"):
                weight_df = pd.DataFrame(
                    action.target_weight_position, index=["weight"]
                ).T
                weight_df.index.rename("instrument", inplace=True)
                weight_df.reset_index(inplace=True)
                action_df = pd.merge(action_df, weight_df, on="instrument", how="left")
            else:
                action_df["weight"] = np.nan

            # rename cols
            action_df.columns = [
                "instrument",
                f"signal{suffix}",
                f"direction{suffix}",
                f"weight{suffix}",
            ]
            action_dfs.append(action_df)

        # reformat trade decision into dataframe
        position_df = self.get_position_df()
        position_df.reset_index(inplace=True)

        decision_df = pd.merge(
            pd.merge(action_dfs[0], action_dfs[1], on="instrument", how="left"),
            position_df,
            on="instrument",
            how="left",
        )

        # re-assign weight
        for baseline in [True, False]:
            suffix = "_baseline" if baseline else ""
            if np.isnan(decision_df[f"weight{suffix}"]).all():
                ind = np.logical_or(
                    np.logical_and(
                        decision_df["position"] == 1,
                        decision_df[f"direction{suffix}"] != -1,
                    ),
                    decision_df[f"direction{suffix}"] == 1,
                )
                decision_df[f"weight{suffix}"][ind] = 1.0 / self.topk

        decision_df.sort_values(["position", "signal"], ascending=False, inplace=True)
        decision_df.reset_index(drop=True, inplace=True)
        cols = [
            "instrument",
            "direction",
            "weight",
            "signal",
            "direction_baseline",
            "weight_baseline",
            "signal_baseline",
        ] + self.position_feature_cols
        decision_df = decision_df[cols]
        return decision_df

    def get_position_df(self, rename_cols=False) -> Optional[pd.DataFrame]:
        """[amount, price, weight, count_day, position]"""
        current_position = copy.deepcopy(self.trade_position)
        if "weight" in self.position_feature_cols:
            current_position.update_weight_all()
        position = current_position.position
        for k in ["cash", "now_account_value"]:
            if k in position:
                position.pop(k)
        if len(position):
            position_df = pd.DataFrame(position).T
            position_df.index.rename("instrument", inplace=True)
            position_df["position"] = 1
            for k in self.position_feature_cols:
                if k not in position_df.columns:
                    position_df[k] = 0
            position_df = position_df[self.position_feature_cols]
            if rename_cols:
                position_df.columns = pd.MultiIndex.from_tuples(
                    [("feature", c) for c in position_df.columns]
                )
            return position_df
        return None

    def get_trade_start_end_time(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        trade_step = self.trade_calendar.get_trade_step()
        return self.trade_calendar.get_step_time(trade_step)

    def get_pred_start_end_time(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        trade_step = self.trade_calendar.get_trade_step()
        return self.trade_calendar.get_step_time(trade_step, shift=1)

    def set_initial_state(self):
        self.initial_state = TradeStrategyInitialState(
            start_time=self.start_time,
            end_time=self.end_time,
            topk=self.topk,
            stock_num=self.stock_num,
            stock_sampling_method=self.stock_sampling_method,
        )

    def reset_initial_state(self, initial_state: TradeStrategyInitialState):
        self.initial_state = initial_state
        self.stock_pool = []

    def sample_stocks(self, stocks: List[str]):
        if self.initial_state.stock_sampling_method == "random":
            # select random stocks
            if len(self.stock_pool) == 0:
                instruments = D.list_instruments(
                    instruments=D.instruments("csi300"),
                    start_time=self.trade_calendar.start_time,
                    end_time=self.trade_calendar.end_time,
                    as_list=True,
                )
                self.stock_pool = np.random.choice(
                    instruments, self.initial_state.stock_num
                )
            stock_pool = self.stock_pool
        else:
            # select topk stocks
            stock_pool = stocks[: self.initial_state.stock_num]
        # keep the original order
        return [s for s in stocks if s in stock_pool]

    def reset_policy(self, policy: BasePolicy):
        self.policy = policy


class TopkDropoutStrategy(TopkDropoutStrategyBase, BaseTradeStrategy):
    policy_cls: BasePolicy
    state_interpreter_cls: StateInterpreter = TradeStrategyStateInterpreter
    action_interpreter_cls: ActionInterpreter

    def __init__(
        self,
        *,
        topk,
        dim,
        stock_num,
        only_tradable,
        n_drop,
        hold_thresh,
        stock_sampling_method="topk",
        equal_weight=True,
        start_time=None,
        end_time=None,
        rerank_topk=True,
        signal_key="signal",
        imitation_label_key=("feature", "signal"),
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({"topk": topk if rerank_topk else stock_num})
        state_interpreter_kwargs = kwargs.pop("state_interpreter_kwargs", {})
        action_interpreter_kwargs = kwargs.pop("action_interpreter_kwargs", {})
        super().__init__(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            only_tradable=only_tradable,
            **kwargs,
        )
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.start_time = start_time
        self.end_time = end_time
        self.dim = dim
        self.topk_original = topk
        self.n_drop_original = n_drop
        self.stock_num = stock_num
        self.signal_key = signal_key
        self.imitation_label_key = imitation_label_key
        self.stock_sampling_method = stock_sampling_method
        self.equal_weight = equal_weight
        self.pred_score = None

        self.state_interpreter = self.state_interpreter_cls(
            dim=dim * feature_n_step,
            stock_num=stock_num,
            **state_interpreter_kwargs,
        )
        self.action_interpreter = self.action_interpreter_cls(
            topk=self.topk,
            n_drop=self.n_drop,
            hold_thresh=self.hold_thresh,
            stock_num=stock_num,
            rerank_topk=rerank_topk,
            signal_key=signal_key,
            **action_interpreter_kwargs,
        )
        self.baseline_action_interpreter = self.action_interpreter_cls(
            topk=self.topk,
            n_drop=self.n_drop,
            hold_thresh=self.hold_thresh,
            stock_num=stock_num,
            rerank_topk=rerank_topk,
            signal_key=signal_key,
            baseline=True,
            **action_interpreter_kwargs,
        )
        self.aux_info_collector = ImitationLabelCollector()
        self.policy = self.policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
            **policy_kwargs,
        )
        if checkpoint_path:
            self.policy.eval()
        self.set_initial_state()

    def __str__(self):
        return "TopkDropoutStrategy"

    def get_pred_score(self):
        return self.pred_score

    def prepare_trading_with_action(self, action: TopkDropoutStrategyAction):
        self.pred_score = action.signal[~action.signal.index.isin([FAKE_STOCK])]
        self.topk = action.topk
        self.n_drop = action.n_drop
        self.hold_thresh = action.hold_thresh

    def generate_trade_decision(
        self,
        execute_result=None,
        action: TopkDropoutStrategyAction = None,
        trade_start_time: Optional[pd.Timestamp] = None,
        trade_end_time: Optional[pd.Timestamp] = None,
        return_decision: bool = True,
    ):
        if action is None:
            action = self.action()
        self.prepare_trading_with_action(action)
        return super().generate_trade_decision(
            execute_result,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            return_decision=return_decision,
        )


class TopkDropoutDiscreteDynamicParamStrategy(TopkDropoutStrategy):
    policy_cls = discrete.PPO
    action_interpreter_cls = TopkDropoutDiscreteDynamicParamStrategyActionInterpreter

    def __str__(self):
        return "TopkDropoutDiscreteDynamicParamStrategy"

    def generate_trade_decision(
        self,
        execute_result=None,
        action: TopkDropoutStrategyAction = None,
        trade_start_time: Optional[pd.Timestamp] = None,
        trade_end_time: Optional[pd.Timestamp] = None,
        return_decision: bool = True,
    ):
        if action is None:
            action = self.action()
        topk = action.topk
        action = TopkDropoutStrategyAction(
            signal=action.signal,
            topk=self.topk_original,
            n_drop=action.n_drop,
            hold_thresh=action.hold_thresh,
            ready=action.ready,
        )
        self.prepare_trading_with_action(action)
        sell_order_list, buy_order_list = super(
            TopkDropoutStrategy, self
        ).generate_trade_decision(
            execute_result,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            return_decision=False,
        )
        last = self.trade_position.get_stock_list()
        buy_num = max(len(sell_order_list) + topk - len(last), 0)
        buy_order_list = buy_order_list[:buy_num]
        if return_decision:
            return TradeDecisionWO(sell_order_list + buy_order_list, self)
        else:
            return sell_order_list, buy_order_list


class TopkDropoutDiscreteMultiDynamicParamStrategy(
    TopkDropoutDiscreteDynamicParamStrategy
):
    policy_cls = discrete.MultiPPO
    action_interpreter_cls = (
        TopkDropoutDiscreteMultiDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutDiscreteMultiDynamicParamStrategy"


class TopkDropoutContinuousRerankStrategy(TopkDropoutStrategy):
    policy_cls = continuous.MetaPPO
    action_interpreter_cls = TopkDropoutContinuousRerankStrategyActionInterpreter

    def __str__(self):
        return "TopkDropoutContinuousRerankStrategy"


class TopkDropoutContinuousRerankDynamicParamStrategy(TopkDropoutStrategy):
    policy_cls = continuous.MetaPPO
    action_interpreter_cls = (
        TopkDropoutContinuousRerankDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutContinuousRerankDynamicParamStrategy"


class TopkDropoutDiscreteRerankDynamicParamStrategy(TopkDropoutStrategy):
    policy_cls = discrete.TopkMetaPPO
    action_interpreter_cls = (
        TopkDropoutDiscreteRerankDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutDiscreteRerankDynamicParamStrategy"


class TopkDropoutDiscreteWeightContinuousRerankDynamicParamStrategy(
    TopkDropoutStrategy
):
    policy_cls = discrete.WeightMetaPPO
    action_interpreter_cls = (
        TopkDropoutContinuousRerankDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutDiscreteWeightContinuousRerankDynamicParamStrategy"


class TopkDropoutDiscreteBinaryContinuousRerankDynamicParamStrategy(
    TopkDropoutStrategy
):
    policy_cls = discrete.MultiBinaryMetaPPO
    action_interpreter_cls = (
        TopkDropoutContinuousRerankDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutDiscreteBinaryContinuousRerankDynamicParamStrategy"


class TopkDropoutStepByStepDiscreteRerankDynamicParamStrategy(TopkDropoutStrategy):
    policy_cls = discrete.StepByStepMetaPPO
    action_interpreter_cls = (
        TopkDropoutStepByStepDiscreteRerankDynamicParamStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutStepByStepDiscreteRerankDynamicParamStrategy"


class TopkDropoutDiscreteDynamicSelectionStrategy(TopkDropoutStrategy):
    policy_cls = discrete.PPO
    action_interpreter_cls = (
        TopkDropoutDiscreteDynamicSelectionStrategyActionInterpreter
    )

    def __str__(self):
        return "TopkDropoutDiscreteDynamicSelectionStrategy"


class WeightStrategy(WeightStrategyBase, BaseTradeStrategy):
    policy_cls: BasePolicy = continuous.MetaPPO
    state_interpreter_cls: StateInterpreter = TradeStrategyStateInterpreter
    action_interpreter_cls: ActionInterpreter = WeightStrategyActionInterpreter

    def __init__(
        self,
        *,
        topk,
        dim,
        stock_num,
        only_tradable,
        n_drop=None,  # placeholder
        hold_thresh=None,  # placeholder
        stock_sampling_method="topk",
        equal_weight=True,
        start_time=None,
        end_time=None,
        rerank_topk=True,
        signal_key="signal",
        imitation_label_key=("feature", "signal"),
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        policy_kwargs = kwargs.pop("policy_kwargs", {})
        policy_kwargs.update({"topk": topk if rerank_topk else stock_num})
        state_interpreter_kwargs = kwargs.pop("state_interpreter_kwargs", {})
        action_interpreter_kwargs = kwargs.pop("action_interpreter_kwargs", {})
        super().__init__(**kwargs)
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.start_time = start_time
        self.end_time = end_time
        self.signal_key = signal_key
        self.imitation_label_key = imitation_label_key
        self.topk = topk
        self.stock_num = stock_num
        self.stock_sampling_method = stock_sampling_method
        self.only_tradable = only_tradable
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh

        self.state_interpreter = self.state_interpreter_cls(
            dim=dim * feature_n_step,
            stock_num=stock_num,
            **state_interpreter_kwargs,
        )
        self.action_interpreter = self.action_interpreter_cls(
            topk=topk,
            stock_num=stock_num,
            only_tradable=only_tradable,
            rerank_topk=rerank_topk,
            signal_key=signal_key,
            equal_weight=equal_weight,
            **action_interpreter_kwargs,
        )
        self.baseline_action_interpreter = self.action_interpreter_cls(
            topk=topk,
            stock_num=stock_num,
            only_tradable=only_tradable,
            rerank_topk=rerank_topk,
            signal_key=signal_key,
            equal_weight=equal_weight,
            baseline=True,
            **action_interpreter_kwargs,
        )
        self.aux_info_collector = ImitationLabelCollector()
        self.policy = self.policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
            **policy_kwargs,
        )
        if checkpoint_path:
            self.policy.eval()
        self.set_initial_state()

    def __str__(self):
        return "WeightStrategy"

    def generate_trade_decision(
        self,
        execute_result=None,
        action: WeightStrategyAction = None,
        trade_start_time: Optional[pd.Timestamp] = None,
        trade_end_time: Optional[pd.Timestamp] = None,
        return_decision: bool = True,
    ):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        if trade_start_time is None or trade_end_time is None:
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
                trade_step
            )
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score,
            current=current_temp,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            action=action,
        )
        order_list = (
            self.order_generator.generate_order_list_from_target_weight_position(
                current=current_temp,
                trade_exchange=self.trade_exchange,
                risk_degree=self.get_risk_degree(trade_step),
                target_weight_position=target_weight_position,
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
                trade_start_time=trade_start_time,
                trade_end_time=trade_end_time,
            )
        )
        if return_decision:
            return TradeDecisionWO(order_list, self)
        else:
            return order_list

    def generate_target_weight_position(
        self, score, current, trade_start_time, trade_end_time, action=None
    ):
        if not action:
            action = self.action()
        return action.target_weight_position


class StepByStepStrategy(WeightStrategy):
    policy_cls = discrete.StepByStepMetaPPO
    action_interpreter_cls = StepByStepStrategyActionInterpreter

    def __str__(self):
        return "StepByStepStrategy"


class DiscreteWeightStrategy(WeightStrategy):
    policy_cls = discrete.WeightMetaPPO
    action_interpreter_cls = WeightStrategyActionInterpreter

    def __str__(self):
        return "DiscreteWeightStrategy"


class DiscreteBinaryWeightStrategy(WeightStrategy):
    policy_cls = discrete.MultiBinaryMetaPPO
    action_interpreter_cls = WeightStrategyActionInterpreter

    def __str__(self):
        return "DiscreteBinaryWeightStrategy"
