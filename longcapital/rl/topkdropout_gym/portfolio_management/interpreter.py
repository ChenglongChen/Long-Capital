import numpy as np
import pandas as pd
from gym import spaces
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.signal import SignalWCache
from qlib.rl.interpreter import ActionInterpreter
from qlib.rl.interpreter import StateInterpreter
from qlib.utils import init_instance_by_config

from .simulator import PortfolioManagementState


class TopkDropoutStateInterpreter(StateInterpreter[PortfolioManagementState, np.ndarray]):
    def __init__(self, signal_list, stock_num):
        self.signal_list = signal_list
        self.stock_num = stock_num
        column_num = len(self.signal_list) + 1
        self.shape = ((stock_num * column_num),)
        self.empty_numpy_series = pd.Series([0] * stock_num)

    def get_feature_df(self, state: PortfolioManagementState) -> pd.DataFrame:
        cur_position = state.executor.trade_account.current_position
        position_series = pd.Series(cur_position.get_stock_amount_dict(), dtype=np.float32)

        trade_calendar = state.trade_calendar
        trade_step = trade_calendar.get_trade_step()
        pred_start_time, pred_end_time = trade_calendar.get_step_time(trade_step, shift=1)

        feature_map = {"position": position_series}
        for i in range(len(self.signal_list)):
            singal_name = "signal_{}".format(i + 1)

            pred_score = self.signal_list[i].get_signal(pred_start_time, pred_end_time)
            # On first few points, we might get None as there is no feature input
            if pred_score is None:
                pred_score = self.empty_numpy_series

            feature_map[singal_name] = pred_score

        feature_df = pd.concat(feature_map, axis=1)

        feature_df = feature_df.sort_values(by=['position', "signal_1"], ascending=False)
        return feature_df

    def interpret(self, state: PortfolioManagementState) -> np.ndarray:
        feature_df = self.get_feature_df(state)
        result_array = feature_df.fillna(0).astype('float32').values.flatten()
        result_array.resize(self.shape)
        return result_array

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(0 - np.inf, np.inf, shape=self.shape, dtype=np.float32)


class TopkDropoutActionInterpreter(ActionInterpreter[PortfolioManagementState, int, TradeDecisionWO]):
    def __init__(self, topk: int, pred_scores: pd.DataFrame) -> None:
        self.topk = topk
        signal = SignalWCache(pred_scores)
        strategy = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": signal,
                "topk": topk,
                "n_drop": topk,
            },
        }
        self.trade_strategy = init_instance_by_config(strategy)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.topk + 1)

    def interpret(self, simulator_state: PortfolioManagementState, action: int) -> TradeDecisionWO:
        assert 0 <= action <= self.topk
        self.trade_strategy.reset_common_infra(simulator_state.executor.common_infra)
        self.trade_strategy.reset(level_infra=simulator_state.executor.get_level_infra())

        self.trade_strategy.n_drop = int(action)

        return self.trade_strategy.generate_trade_decision()
