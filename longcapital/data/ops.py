import numpy as np
import pandas as pd
from scipy import stats


def Eq(series, other):
    return series == other


def Ne(series, other):
    return series != other


def Gt(series, other):
    return series > other


def Ge(series, other):
    return series >= other


def Lt(series, other):
    return series < other


def Le(series, other):
    return series <= other


def If(condition, left, right):
    res = condition.copy()
    if isinstance(left, pd.Series):
        res[condition] = left[condition]
    elif isinstance(left, int) or isinstance(left, float):
        res[condition] = left
    if isinstance(right, pd.Series):
        res[~condition] = right[~condition]
    elif isinstance(right, int) or isinstance(right, float):
        res[~condition] = right
    return res


def Or(left, right):
    return pd.Series(np.logical_or(left, right), left.index)


def And(left, right):
    return pd.Series(np.logical_and(left, right), left.index)


def Greater(left, right):
    return np.maximum(left, right)


def Less(left, right):
    return np.minimum(left, right)


def Ref(series, c):
    return series["feature", c]


def Abs(series):
    return np.abs(series)


def Log(series):
    return np.log(series + 1)


def Sign(series):
    return np.sign(series)


def Delta(series, N=1):
    return series.diff(N)


def Delay(series, N):
    return series.shift(N)


def Rolling(series, N, func):
    res = getattr(series.rolling(N, min_periods=1), func)()
    return res


def Mean(series, N):
    return Rolling(series, N, "mean")


def Std(series, N):
    return Rolling(series, N, "std")


def Max(series, N):
    return Rolling(series, N, "max")


def Min(series, N):
    return Rolling(series, N, "min")


def Sum(series, N):
    return Rolling(series, N, "sum")


def PairRolling(series_left, series_right, N, func):
    series_pair = pd.DataFrame({"left": series_left, "right": series_right})
    res = series_pair.groupby("instrument", group_keys=False).apply(
        lambda x: getattr(x["left"].rolling(N, min_periods=1), func)(x["right"])
    )
    res.sort_index(inplace=True)
    return res


def Cov(series_left, series_right, N):
    return PairRolling(series_left, series_right, N, "cov")


def Corr(series_left, series_right, N):
    return PairRolling(series_left, series_right, N, "corr")


def Rank(series):
    res = series.groupby("datetime", group_keys=False).rank(pct=True)
    res.sort_index(inplace=True)
    return res


def Tsrank(series, N):
    def rank(x, pct=False, ascending=True):
        s = 1 if ascending else -1
        rnk = np.argsort(np.argsort(s * x))
        if pct:
            rnk = rnk / len(x)
        return rnk.values[-1]

    res = series.groupby("instrument", group_keys=False).apply(
        lambda x: x.rolling(N, min_periods=1).apply(lambda y: rank(y, pct=True))
    )
    res.sort_index(inplace=True)
    return res


def Sma(series, n, m):
    new_series = [0] * len(series)
    for i in range(len(series) - 1):
        new_series[i + 1] = (series[i] * m + new_series[i] * (n - m)) / n
    return pd.Series(new_series, series.index)


def _wma(series, N, weights):
    sum_weights = weights.sum()
    res = series.rolling(N).apply(lambda x: np.sum(weights * x.values) / sum_weights)
    return res


def Wma(series, N, decay=0.9):
    weights = decay ** np.arange(0, N)[::-1]
    return _wma(series, N, weights)


def Decaylinear(series, N):
    weights = np.arange(1, N + 1)[::-1]
    return _wma(series, N, weights)


def Sequence(n):
    return np.arange(1, n + 1)


def Regbeta(series, B, n):
    def _Regbeta(x, y):
        res = stats.linregress(x, y)
        return res.slope

    res = series.rolling(n).apply(lambda y: _Regbeta(B, y))
    return res


def Regresi(series, B, n):
    def _Regresi(x, y):
        res = stats.linregress(x, y)
        p = x * res.slope + res.intercept
        resi = ((y - p) ** 2).mean()
        return resi

    res = series.rolling(n).apply(lambda y: _Regresi(B, y))
    return res


def Ret(df):
    return Delay(Ref(df, "close"), -1) / Ref(df, "close") - 1
