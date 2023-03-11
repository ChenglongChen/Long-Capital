import datetime
from random import randrange

import numpy as np
import scipy
import torch


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
def filter_stock(state, stocks, weights):
    tradable = [
        i
        for i, s in enumerate(stocks)
        if state.trade_strategy.trade_exchange.is_stock_tradable(
            stock_id=s, start_time=state.trade_start_time, end_time=state.trade_end_time
        )
    ]
    return stocks[tradable], weights[tradable]
