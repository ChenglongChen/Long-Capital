import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.aux_info import ImitationLabelCollector
from longcapital.rl.order_execution.buffer import FeatureBuffer
from longcapital.rl.order_execution.interpreter import (
    DirectSelectionActionInterpreter,
    TopkDropoutDynamicStrategyActionInterpreter,
    TopkDropoutRerankStrategyActionInterpreter,
    TopkDropoutSelectionStrategyActionInterpreter,
    TopkDropoutSignalStrategyActionInterpreter,
    TopkDropoutStrategyAction,
    TopkDropoutStrategyActionInterpreter,
    TradeStrategyStateInterpreter,
    WeightStrategyAction,
    WeightStrategyActionInterpreter,
)
from longcapital.rl.order_execution.policy import continuous, discrete
from longcapital.rl.order_execution.state import TradeStrategyState
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy import TopkDropoutStrategy as TopkDropoutStrategyBase
from qlib.contrib.strategy import WeightStrategyBase
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
    signal_key: str = "signal"
    imitation_label_key: str = "label"

    def action(
        self,
        baseline: bool = False,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ) -> Any:
        state = TradeStrategyState(
            trade_executor=self.executor,
            trade_strategy=self,
            feature=self.get_feature(
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
            ),
        )
        obs = [{"obs": self.state_interpreter.interpret(state), "info": {}}]
        policy_out = self.policy(Batch(obs))
        if baseline:
            action = self.baseline_action_interpreter.interpret(state, policy_out.act)
        else:
            action = self.action_interpreter.interpret(state, policy_out.act)
        return action

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

        # sort to make sure the ranking distribution is similar across different dates
        feature.sort_values(
            by=[("feature", "position"), ("feature", self.signal_key)],
            ascending=False,
            inplace=True,
        )

        return feature

    def trade(self) -> pd.DataFrame:
        trade_step = self.trade_calendar.get_trade_len() - 1
        pred_start_time = self.trade_calendar.get_step_start_time(trade_step=trade_step)
        pred_end_time = epsilon_change(pred_start_time + pd.Timedelta(days=1))
        # mock fake trade_start_time and trade_end_time as placeholders
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
            trade_step=trade_step, shift=1
        )
        action_dfs = []
        for baseline in [True, False]:
            action = self.action(
                baseline=baseline,
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
            )
            order_list = self.generate_trade_decision(
                execute_result=None,
                action=action,
                trade_start_time=trade_start_time,
                trade_end_time=trade_end_time,
                return_decision=False,
            )

            action_df = action.signal.reset_index()
            action_df.columns = ["instrument", "signal"]

            order_df = pd.DataFrame(order_list)[["stock_id", "direction"]]
            order_df.columns = ["instrument", "direction"]

            action_df = pd.merge(action_df, order_df, on="instrument", how="left")
            if baseline:
                action_df.columns = [
                    "instrument",
                    "signal_baseline",
                    "direction_baseline",
                ]
            else:
                action_df.columns = ["instrument", "signal", "direction"]
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

        decision_df = decision_df.sort_values(["position", "signal"], ascending=False)
        decision_df.reset_index(drop=True, inplace=True)
        cols = [
            "instrument",
            "direction",
            "signal",
            "direction_baseline",
            "signal_baseline",
        ] + self.position_feature_cols
        decision_df = decision_df[cols]
        decision_df["pred_start_time"] = pred_start_time
        return decision_df

    def get_position_df(self, rename_cols=False) -> Optional[pd.DataFrame]:
        """[amount, price, weight, count_day, position]"""
        current_position = copy.deepcopy(self.trade_position)
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


class TopkDropoutStrategy(TopkDropoutStrategyBase, BaseTradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop=None,
        checkpoint_path=None,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=discrete.PPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.policy_cls = policy_cls
        self.signal_key = signal_key
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutStrategyActionInterpreter(
            topk=topk, n_drop=n_drop
        )
        self.baseline_action_interpreter = TopkDropoutStrategyActionInterpreter(
            topk=topk, n_drop=n_drop, baseline=True
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()

    def __str__(self):
        return "TopkDropoutStrategy"

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
        self.n_drop = action.n_drop
        return super().generate_trade_decision(
            execute_result,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
            return_decision=return_decision,
        )


class TopkDropoutSignalStrategy(TopkDropoutStrategyBase, BaseTradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=continuous.MetaPPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutSignalStrategyActionInterpreter(
            stock_num=stock_num, signal_key=signal_key
        )
        self.baseline_action_interpreter = TopkDropoutSignalStrategyActionInterpreter(
            stock_num=stock_num, signal_key=signal_key, baseline=True
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()
        self.pred_score = None

    def __str__(self):
        return "TopkDropoutSignalStrategy"

    def get_pred_score(self):
        return self.pred_score

    def prepare_trading_with_action(self, action: TopkDropoutStrategyAction):
        self.pred_score = action.signal

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


class TopkDropoutSelectionStrategy(TopkDropoutSignalStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=discrete.PPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        super(TopkDropoutSignalStrategy, self).__init__(
            topk=topk, n_drop=n_drop, **kwargs
        )
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutSelectionStrategyActionInterpreter(
            topk=topk, n_drop=n_drop, stock_num=stock_num, signal_key=signal_key
        )
        self.baseline_action_interpreter = (
            TopkDropoutSelectionStrategyActionInterpreter(
                topk=topk,
                n_drop=n_drop,
                stock_num=stock_num,
                signal_key=signal_key,
                baseline=True,
            )
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()
        self.pred_score = None

    def __str__(self):
        return "TopkDropoutSelectionStrategy"


class TopkDropoutDynamicStrategy(TopkDropoutSignalStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop,
        hold_thresh,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=discrete.MetaPPO,
        unbounded=True,
        conditioned_sigma=False,
        max_action=1.0,
        sigma_min=1e-8,
        sigma_max=0.05,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        super(TopkDropoutSignalStrategy, self).__init__(
            topk=topk, n_drop=n_drop, **kwargs
        )
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutDynamicStrategyActionInterpreter(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            signal_key=signal_key,
        )
        self.baseline_action_interpreter = TopkDropoutDynamicStrategyActionInterpreter(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            signal_key=signal_key,
            baseline=True,
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            max_action=max_action,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            imitation_label_key=imitation_label_key,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()
        self.pred_score = None

    def __str__(self):
        return "TopkDropoutDynamicStrategy"

    def prepare_trading_with_action(self, action: TopkDropoutStrategyAction):
        super(TopkDropoutDynamicStrategy, self).prepare_trading_with_action(action)
        self.topk = action.topk
        self.n_drop = action.n_drop
        self.hold_thresh = action.hold_thresh


class TopkDropoutRerankStrategy(TopkDropoutSignalStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop,
        hold_thresh,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=discrete.MetaPPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        checkpoint_path=None,
        **kwargs,
    ):
        super(TopkDropoutSignalStrategy, self).__init__(
            topk=topk, n_drop=n_drop, **kwargs
        )
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutRerankStrategyActionInterpreter(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            signal_key=signal_key,
        )
        self.baseline_action_interpreter = TopkDropoutRerankStrategyActionInterpreter(
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            stock_num=stock_num,
            signal_key=signal_key,
            baseline=True,
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()
        self.pred_score = None

    def __str__(self):
        return "TopkDropoutRerankStrategy"

    def prepare_trading_with_action(self, action: TopkDropoutStrategyAction):
        super(TopkDropoutRerankStrategy, self).prepare_trading_with_action(action)
        self.topk = action.topk
        self.n_drop = action.n_drop
        self.hold_thresh = action.hold_thresh


class WeightStrategy(WeightStrategyBase, BaseTradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        signal_key="signal",
        imitation_label_key="label",
        checkpoint_path=None,
        equal_weight=True,
        policy_cls=continuous.MetaPPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.verbose = verbose
        self.signal_key = signal_key
        self.policy_cls = policy_cls

        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = WeightStrategyActionInterpreter(
            stock_num=stock_num,
            topk=topk,
            signal_key=signal_key,
            equal_weight=equal_weight,
        )
        self.baseline_action_interpreter = WeightStrategyActionInterpreter(
            stock_num=stock_num,
            topk=topk,
            signal_key=signal_key,
            equal_weight=equal_weight,
            baseline=True,
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()

    def __str__(self):
        return "WeightStrategy"

    def generate_trade_decision(
        self, execute_result=None, action: WeightStrategyAction = None
    ):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
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
        return TradeDecisionWO(order_list, self)

    def generate_target_weight_position(
        self, score, current, trade_start_time, trade_end_time, action=None
    ):
        if not action:
            action = self.action()
        return action.target_weight_position


class DirectSelectionStrategy(WeightStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        checkpoint_path=None,
        signal_key="signal",
        imitation_label_key="label",
        policy_cls=discrete.MetaPPO,
        feature_n_step=1,
        position_feature_cols=["count_day"],
        verbose=False,
        **kwargs,
    ):
        super(WeightStrategy, self).__init__(**kwargs)
        self.feature_buffer = FeatureBuffer(size=feature_n_step)
        self.position_feature_cols = position_feature_cols
        self.verbose = verbose
        self.signal_key = signal_key
        self.policy_cls = policy_cls

        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim * feature_n_step, stock_num=stock_num
        )
        self.action_interpreter = DirectSelectionActionInterpreter(stock_num=stock_num)
        self.baseline_action_interpreter = DirectSelectionActionInterpreter(
            stock_num=stock_num, baseline=True
        )
        self.aux_info_collector = ImitationLabelCollector(
            stock_num=stock_num, label_key=imitation_label_key
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()

    def __str__(self):
        return "DirectSelectionStrategy"
