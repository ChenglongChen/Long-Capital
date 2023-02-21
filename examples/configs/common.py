import os

from qlib.constant import REG_CN, REG_US

from longcapital.utils.time import get_diff_date

#
EXP_NAME = "long-capital"

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


def get_backtest_config(region=REG_CN, instruments="csi300", deal_price="open"):
    REGION_CONFIG = {
        REG_CN: {
            "benchmark": BECHMARK_PARAMS[instruments],
            "exchange_kwargs": {
                "codes": instruments,
                "freq": "day",
                "trade_unit": 100,
                "limit_threshold": 0.095,
                "deal_price": deal_price,
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            }
        },
        REG_US: {
            "benchmark": BECHMARK_PARAMS[instruments],
            "exchange_kwargs": {
                "codes": instruments,
                "freq": "day",
                "trade_unit": 1,
                "limit_threshold": None,
                "deal_price": deal_price,
                # estimated from moomoo sg
                "open_cost": 0.003,
                "close_cost": 0.005,
                "min_cost": 0
            }
        }
    }
    return REGION_CONFIG[region]


def get_last_date_from_calendar(region=REG_CN):
    file = f"~/.qlib/qlib_data/{region}_data/calendars/day.txt"
    date = os.popen(f"tail -n 1 {file}").read().split("\n")[0]
    return date


def get_date_config(region=REG_CN, pred_date=None):
    if pred_date is None:
        pred_date = get_last_date_from_calendar(region)
    test_end_date = pred_date
    backtest_end_date = get_diff_date(pred_date, -1)
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
                "end": test_end_date
            },
            "backtest": {
                "start": "2019-01-01",
                "end": backtest_end_date
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
                "end": test_end_date
            },
            "backtest": {
                "start": "2019-01-01",
                "end": backtest_end_date
            }
        }
    }
    return DATE_CONFIG[region]
