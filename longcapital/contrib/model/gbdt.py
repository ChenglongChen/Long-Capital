# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

import lightgbm as lgb
from qlib.contrib.model.gbdt import LGBModel as QlibLGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class LGBModel(QlibLGBModel):
    """LightGBM Model with lambdarank support"""

    def __init__(
        self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs
    ):
        if loss not in {"mse", "binary", "lambdarank"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(
        self, dataset: DatasetH, reweighter=None
    ) -> List[Tuple[lgb.Dataset, str]]:
        ds_l = super(LGBModel, self)._prepare_data(
            dataset=dataset, reweighter=reweighter
        )
        for data, key in ds_l:
            df = dataset.prepare(
                key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
            )
            group = (
                df.index.get_level_values("datetime").value_counts().sort_index().values
            )
            data.set_group(group)
        return ds_l
