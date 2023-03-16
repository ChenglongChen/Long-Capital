import copy
from pathlib import Path
from typing import Optional

import pandas as pd
import torch.nn.functional as F  # noqa
from longcapital.rl.order_execution.interpreter import (
    TopkActionInterpreter,
    TopkDropoutDynamicStrategyAction,
    TopkDropoutDynamicStrategyActionInterpreter,
    TopkDropoutSelectionStrategyActionInterpreter,
    TopkDropoutSignalStrategyAction,
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
from qlib.utils.time import epsilon_change
from tianshou.data import Batch


class TradeStrategy:
    def action(
        self,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ):
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
        action = self.action_interpreter.interpret(state, policy_out.act)
        return action

    def get_feature(
        self,
        feature=None,
        pred_start_time: Optional[pd.Timestamp] = None,
        pred_end_time: Optional[pd.Timestamp] = None,
    ):
        if feature is None:
            if pred_start_time is None or pred_end_time is None:
                pred_start_time, pred_end_time = self.get_pred_start_end_time()
            feature = self.signal.get_signal(
                start_time=pred_start_time, end_time=pred_end_time
            )

        stock_weight_dict = (
            self.executor.trade_account.current_position.get_stock_weight_dict(
                only_stock=False
            )
        )
        current_position_list = list(stock_weight_dict.keys())
        feature[("feature", "position")] = 0
        feature[("feature", "unhold")] = 1
        feature.loc[
            feature.index.isin(current_position_list), ("feature", "position")
        ] = 1
        feature.loc[
            feature.index.isin(current_position_list), ("feature", "unhold")
        ] = 0

        # sort to make sure the ranking distribution is similar across different dates
        feature.sort_values(
            by=[("feature", "position"), ("feature", self.signal_key)],
            ascending=False,
            inplace=True,
        )

        return feature

    def trade(self):
        trade_step = self.trade_calendar.get_trade_len() - 1
        pred_start_time = self.trade_calendar.get_step_start_time(trade_step=trade_step)
        pred_end_time = epsilon_change(pred_start_time + pd.Timedelta(days=1))
        # mock fake trade_start_time and trade_end_time as placeholders
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
            trade_step=trade_step, shift=1
        )
        action = self.action(
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

        # reformat trade decision into dataframe
        position = copy.deepcopy(self.executor.trade_account.current_position.position)
        position.pop("cash")
        position.pop("now_account_value")
        position = pd.DataFrame(position).T.reset_index()
        position.rename(columns={"index": "instrument"}, inplace=True)
        position["count_day"] = position["count_day"] + 1
        position["position"] = 1

        action = action.signal.reset_index()
        action.columns = ["instrument", "signal"]

        order = pd.DataFrame(order_list)[["stock_id", "direction"]]
        order.columns = ["instrument", "direction"]

        decision = pd.merge(
            pd.merge(action, position, on="instrument", how="left"),
            order,
            on="instrument",
            how="left",
        )

        decision = decision.sort_values(["position", "signal"], ascending=False)
        cols = [
            "instrument",
            "direction",
            "position",
            "count_day",
            "signal",
            "amount",
            "price",
        ]
        decision = decision[cols]
        decision["pred_start_time"] = pred_start_time
        decision["pred_end_time"] = pred_end_time
        return decision

    def get_trade_start_end_time(self):
        trade_step = self.trade_calendar.get_trade_step()
        return self.trade_calendar.get_step_time(trade_step)

    def get_pred_start_end_time(self):
        trade_step = self.trade_calendar.get_trade_step()
        return self.trade_calendar.get_step_time(trade_step, shift=1)


class TopkDropoutStrategy(TopkDropoutStrategyBase, TradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop=None,
        checkpoint_path=None,
        signal_key="signal",
        policy_cls=discrete.PPO,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.policy_cls = policy_cls
        self.signal_key = signal_key
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutStrategyActionInterpreter(
            topk=topk, n_drop=n_drop
        )
        self.baseline_action_interpreter = TopkDropoutStrategyActionInterpreter(
            topk=topk, n_drop=n_drop, baseline=True
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


class TopkDropoutSignalStrategy(TopkDropoutStrategyBase, TradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        n_drop,
        signal_key="signal",
        policy_cls=continuous.MetaTD3,
        checkpoint_path=None,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutSignalStrategyActionInterpreter(
            stock_num=stock_num, signal_key=signal_key
        )
        self.baseline_action_interpreter = TopkDropoutSignalStrategyActionInterpreter(
            stock_num=stock_num, signal_key=signal_key, baseline=True
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

    def prepare_trading_with_action(self, action: TopkDropoutSignalStrategyAction):
        self.pred_score = action.signal

    def generate_trade_decision(
        self,
        execute_result=None,
        action: TopkDropoutSignalStrategyAction = None,
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
        policy_cls=discrete.PPO,
        checkpoint_path=None,
        **kwargs,
    ):
        super(TopkDropoutSignalStrategy, self).__init__(
            topk=topk, n_drop=n_drop, **kwargs
        )
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
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
        signal_key="signal",
        policy_cls=discrete.MetaPPO,
        checkpoint_path=None,
        **kwargs,
    ):
        super(TopkDropoutSignalStrategy, self).__init__(
            topk=topk, n_drop=n_drop, **kwargs
        )
        self.signal_key = signal_key
        self.policy_cls = policy_cls
        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
        )
        self.action_interpreter = TopkDropoutDynamicStrategyActionInterpreter(
            topk=topk, n_drop=n_drop, stock_num=stock_num, signal_key=signal_key
        )
        self.baseline_action_interpreter = TopkDropoutDynamicStrategyActionInterpreter(
            topk=topk,
            n_drop=n_drop,
            stock_num=stock_num,
            signal_key=signal_key,
            baseline=True,
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
        return "TopkDropoutDynamicStrategy"

    def prepare_trading_with_action(self, action: TopkDropoutDynamicStrategyAction):
        super(TopkDropoutDynamicStrategy, self).prepare_trading_with_action(action)
        self.topk = action.topk
        self.n_drop = action.n_drop


class WeightStrategy(WeightStrategyBase, TradeStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        topk,
        signal_key="signal",
        checkpoint_path=None,
        policy_cls=continuous.MetaTD3,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.signal_key = signal_key
        self.policy_cls = policy_cls

        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
        )
        self.action_interpreter = WeightStrategyActionInterpreter(
            stock_num=stock_num, topk=topk, signal_key=signal_key
        )
        self.baseline_action_interpreter = WeightStrategyActionInterpreter(
            stock_num=stock_num, topk=topk, signal_key=signal_key, baseline=True
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


class TopkStrategy(WeightStrategy):
    def __init__(
        self,
        *,
        dim,
        stock_num,
        checkpoint_path=None,
        signal_key="signal",
        policy_cls=discrete.MetaPPO,
        verbose=False,
        **kwargs,
    ):
        super(WeightStrategy, self).__init__(**kwargs)
        self.verbose = verbose
        self.signal_key = signal_key
        self.policy_cls = policy_cls

        self.state_interpreter = TradeStrategyStateInterpreter(
            dim=dim, stock_num=stock_num
        )
        self.action_interpreter = TopkActionInterpreter(stock_num=stock_num)
        self.baseline_action_interpreter = TopkActionInterpreter(
            stock_num=stock_num, baseline=True
        )
        self.policy = policy_cls(
            obs_space=self.state_interpreter.observation_space,
            action_space=self.action_interpreter.action_space,
            weight_file=Path(checkpoint_path) if checkpoint_path else None,
        )
        if checkpoint_path:
            self.policy.eval()

    def __str__(self):
        return "TopkStrategy"
