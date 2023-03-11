from typing import NamedTuple

import numpy as np
import pandas as pd
import torch.nn.functional as F  # noqa
from qlib.strategy.base import BaseStrategy


class TradeStrategyState(NamedTuple):
    # NOTE:
    # - for avoiding recursive import
    # - typing annotations is not reliable
    from qlib.backtest.executor import BaseExecutor  # pylint: disable=C0415

    trade_executor: BaseExecutor
    trade_strategy: BaseStrategy
    feature: np.ndarray
    trade_start_time: pd.Timestamp
    trade_end_time: pd.Timestamp
    pred_start_time: pd.Timestamp
    pred_end_time: pd.Timestamp


class TradeStrategyInitiateState(NamedTuple):
    start_time: str
    end_time: str
    sample_date: bool
