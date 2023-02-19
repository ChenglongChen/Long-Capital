import json
import os

from pandas import ExcelWriter


def get_params_from_file(file, key):
    if not os.path.exists(file):
        return None
    with open(file, "r") as f:
        params = json.load(f)
    return params.get(key)


def update_params_to_file(file, key, value):
    with open(file, "r") as f:
        params = json.load(f)
    params[key] = value
    with open(file, "w") as f:
        json.dump(params, f, indent=4)


def update_report_df(folder, key, df_dict):
    with ExcelWriter(f"{folder}/{key}.xlsx") as ex:
        for k, df in df_dict.items():
            df.to_excel(ex, sheet_name=k, float_format="%.5f")
