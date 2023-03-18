from typing import List

import numpy as np
import pandas as pd


class FeatureBuffer:
    def __init__(self, size):
        assert size >= 1
        self.size = size
        self._buffer = [None] * size
        self._curr = 0

    def add(self, f: pd.DataFrame) -> None:
        self._buffer[self._curr] = f
        self._curr = (self._curr + 1) % self.size

    def collect(self) -> pd.DataFrame:
        curr = (self._curr - 1) % self.size
        res = self._buffer[curr]
        if res is not None:
            index = res.index.tolist()
            columns = res.columns.tolist()
            for i in range(2, self.size + 1):
                curr = (self._curr - i) % self.size
                cols = pd.MultiIndex.from_tuples(
                    [(f, f"{c}_{i - 1}") for f, c in columns]
                )
                if self._buffer[curr] is None:
                    f = self._get_dummy_feature(index=index, columns=cols)
                else:
                    f = self._buffer[curr]
                    f.columns = cols
                res = pd.merge(res, f, on="instrument", how="left")
                res.fillna(0, inplace=True)
        return res

    @staticmethod
    def _get_dummy_feature(index: List, columns: pd.MultiIndex) -> pd.DataFrame:
        df = pd.DataFrame(
            np.zeros((len(index), len(columns)), dtype=float),
            index=index,
            columns=columns,
        )
        df.index.rename("instrument", inplace=True)
        return df
