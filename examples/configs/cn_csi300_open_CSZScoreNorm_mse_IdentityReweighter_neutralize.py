from longcapital.utils.time import get_diff_date

from qlib.constant import REG_CN, REG_US


EXP_NAME = "long-capital"

# markets
REGION = REG_CN
INSTRUMENTS = "csi300"
DEAL_PRICE = "open"

# params
MODEL_LOSS = "mse"
LABEL_NORM = "CSZScoreNorm"
ENABLE_REWEIGHTER = False
ENABLE_NEUTRALIZE = True
ENABLE_GTJA_ALPHA = False

SEARCH_STRATEGY_PARAMS = True
SEARCH_MODEL_PARAMS = False

# trading
# the day when you have the stock data after close
PRED_DATE = "2023-02-20"
TEST_END_DATE = PRED_DATE
BACKTEST_END_DATE = get_diff_date(PRED_DATE, -1)

# records
STRATEGY_PARAMS_FILE = "../data/params/strategy.json"
MODEL_PARAMS_FILE = "../data/params/model.json"
PERFORMANCE_FILE = "../data/params/performance.json"
REPORT_DF_FOLDER = "../data/report_df"

# model
MODEL_LOSS_KEY_DICT = {
    "mse": "l2",
    "mse_log": "l2",
    "binary": "binary_logloss",
    "lambdarank": "ndcg@5",
}

# market setting for trading
TOPK_LIST = [1, 2, 4, 6, 8, 10]
N_DROP_LIST = [1, 2, 3, 4, 5]

BECHMARK_PARAMS = {
    "csi300": "SH000300",
    "csi500": "SH000905",
    "csi800": "SH000906",
    # https://github.com/microsoft/qlib/issues/720
    "SP500": "^gspc",
    "NASDAQ100": "^ndx",
}

REGION_CONFIG = {
    REG_CN: {
        "benchmark": BECHMARK_PARAMS[INSTRUMENTS],
        "exchange_kwargs": {
            "codes": INSTRUMENTS,
            "freq": "day",
            "trade_unit": 100,
            "limit_threshold": 0.095,
            "deal_price": DEAL_PRICE,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    },
    REG_US: {
        "benchmark": BECHMARK_PARAMS[INSTRUMENTS],
        "exchange_kwargs": {
            "codes": INSTRUMENTS,
            "freq": "day",
            "trade_unit": 1,
            "limit_threshold": None,
            "deal_price": DEAL_PRICE,
            # estimated from moomoo sg
            "open_cost": 0.003,
            "close_cost": 0.005,
            "min_cost": 0
        }
    }
}

# date setting for train, valid, backtest
DATE_CONFIG = {
    REG_CN: {
        "train": {
            "start": "2008-01-01",
            "end": "2016-12-31"
        },
        "valid": {
            "start": "2017-01-01",
            "end": "2018-12-31"
        },
        "test": {
            "start": "2019-01-01",
            "end": TEST_END_DATE
        },
        "backtest": {
            "start": "2019-01-01",
            "end": BACKTEST_END_DATE
        }
    },
    REG_US: {
        "train": {
            "start": "2008-01-01",
            "end": "2016-12-31"
        },
        "valid": {
            "start": "2017-01-01",
            "end": "2018-12-31"
        },
        "test": {
            "start": "2019-01-01",
            "end": TEST_END_DATE
        },
        "backtest": {
            "start": "2019-01-01",
            "end": BACKTEST_END_DATE
        }
    }
}
