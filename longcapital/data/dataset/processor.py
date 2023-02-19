import pandas as pd

from qlib.data.dataset.processor import get_group_columns, Processor


class CSBucketizeLabel(Processor):
    """
    Cross Sectional Bucketize.
    """

    def __init__(self, fields_group="label", bucket_size=10):
        self.fields_group = fields_group
        self.bucket_size = bucket_size

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)
        df[cols] = df[cols].groupby("datetime", group_keys=False).apply(
            lambda x: pd.Series(pd.qcut(x.values.flatten(), self.bucket_size, labels=False), index=x.index)
        )
        return df


class GTJAAlpha1(Processor):
    def __call__(self, df):
        df_left = df["feature"]["volume"].log().diff()
        df_right = (df["feature"]["close"] - df["feature"]["open"])/df["feature"]["open"]
        df_left = df_left.groupby("datetime").rank(pct=True)
        df_right = df_right.groupby("datetime").rank(pct=True)
        return df
