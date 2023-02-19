import scipy
import numpy as np
import pandas as pd


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def get_riskest_features(df):
    feature_cols = df["feature"].columns
    # getting the per era correlation of each feature vs the target
    all_feature_corrs = df.groupby(level=["datetime"]).apply(
        lambda d: d["feature"][feature_cols].corrwith(d["label"]["LABEL0"]))

    # find the riskiest features by comparing their correlation vs the target in half 1 and half 2 of training data
    riskiest_features = get_biggest_change_features(all_feature_corrs, 50)
    return riskiest_features


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="datetime"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df.index.get_level_values(era_col).unique()
    computed = []
    for u in unique_eras:
        df_era = df.iloc[df.index.get_level_values(era_col) == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        mask = df_era.columns.get_level_values(-1).isin(neutralizers)
        exposures = df_era.loc[:, mask].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)
