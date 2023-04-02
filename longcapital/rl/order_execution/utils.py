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
def filter_nontradable_stock(state, stocks, weights=None):
    def get_tradable_index(state, stocks):
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
        ]
        return tradable

    if len(stocks) and len(weights):
        tradable = get_tradable_index(state, stocks)
        return stocks[tradable], weights[tradable]

    return [], []


# filter fake stock
def filter_fake_stock(state, stocks, weights):
    def get_tradable_index(stocks):
        tradable = [i for i, s in enumerate(stocks) if s != FAKE_STOCK]
        return tradable

    if len(stocks) and len(weights):
        tradable = get_tradable_index(stocks)
        return stocks[tradable], weights[tradable]

    return [], []
