from typing import Dict, NamedTuple

import pandas as pd
import torch.nn.functional as F  # noqa
from qlib.strategy.base import BaseStrategy


class TradeStrategyInitialState(NamedTuple):
    start_time: str
    end_time: str
    topk: int
    stock_num: int = 20
    stock_sampling_method: str = "daily"
    stock_sorting: bool = True
    sample_date: bool = False
    skip_nontradable_start_time: bool = False


class TradeStrategyState(NamedTuple):
    # NOTE:
    # - for avoiding recursive import
    # - typing annotations is not reliable
    from qlib.backtest.executor import BaseExecutor  # pylint: disable=C0415

    trade_executor: BaseExecutor
    trade_strategy: BaseStrategy
    initial_state: TradeStrategyInitialState
    feature: pd.DataFrame
    label: pd.DataFrame
    signal: pd.DataFrame
    position: pd.DataFrame
    info: Dict = {"ready": True}
