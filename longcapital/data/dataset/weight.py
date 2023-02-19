from typing import Union

import pandas as pd
from qlib.data.dataset.processor import get_group_columns
from qlib.data.dataset.weight import Reweighter


class IdentityReweighter(Reweighter):
    def __init__(self):
        pass

    def __str__(self):
        return "IdentityReweighter"

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        w_s = pd.Series(1.0, index=data.index)
        return w_s


class CSRankNormReweighter(Reweighter):
    def __init__(self, fields_group="label"):
        self.fields_group = fields_group

    def __str__(self):
        return "CSRankNormReweighter"

    def reweight(self, data: Union[pd.DataFrame, pd.Series]):
        cols = get_group_columns(data, self.fields_group)
        w_s = data[cols].groupby("datetime").rank(pct=True).values.flatten()
        return w_s
