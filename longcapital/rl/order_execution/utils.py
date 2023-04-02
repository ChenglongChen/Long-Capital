import datetime
from random import randrange

import numpy as np
import scipy
import torch
from longcapital.utils.constant import FAKE_STOCK


def random_daterange(start, end):
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    d = (end - start).days
    i, j = randrange(d), randrange(d)
    s, e = min(i, j), max(i, j)
    end = start + datetime.timedelta(days=e)
    start = start + datetime.timedelta(days=s)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def softmax(x):
    if isinstance(x, np.ndarray):
        return scipy.special.softmax(x, axis=0)
    elif isinstance(x, torch.Tensor):
        return torch.softmax(x.squeeze(), axis=0).numpy()
    else:
        raise ValueError


# filter non-tradable stocks
def filter_stock(state, stocks, weights=None):
    if len(stocks) and (weights is None or len(weights)):
        (
            trade_start_time,
            trade_end_time,
        ) = state.trade_strategy.get_trade_start_end_time()
        tradable = [
            i
            for i, s in enumerate(stocks)
            if state.trade_strategy.trade_exchange.is_stock_tradable(
                stock_id=s,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            and s != FAKE_STOCK
        ]
        if weights is not None:
            return stocks[tradable], weights[tradable]
        else:
            return stocks[tradable]
    if weights is not None:
        return [], []
    else:
        return []
