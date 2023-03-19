from sklearn import metrics
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.seasonal import seasonal_decompose


class Fourier(BaseEstimator, TransformerMixin):
    def __init__(self, season="A", order=2):
        self.fourier = CalendarFourier(freq=season, order=order)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        dp = DeterministicProcess(
            index=X_.index,
            constant=True,  # dummy feature for bias (y-intercept)
            order=1,  # trend (order 1 means linear)
            seasonal=True,  # weekly seasonality (indicators)
            additional_terms=[self.fourier],  # annual seasonality (fourier)
            drop=True,  # drop terms to avoid collinearity
        )
        X_ = dp.in_sample()
        return X_


def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), ))


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    frequencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


def moving_stats(series, alpha=1, seasonality=1, periods=1, min_periods=1, window=1, aggfunc="mean", fillna=None):
    size = window if window != -1 else len(series) - 1
    alpha_range = [alpha**i for i in range(0, size)]

    min_required_len = max(min_periods - 1, 0) * seasonality + 1

    def get_required_lags(series):
        return pd.Series(series.values[::-1][:: seasonality])

    def aggregate_window(series):
        tmp_series = get_required_lags(series)
        size = len(tmp_series)
        tmp = tmp_series * alpha_range[-size:]

        if aggfunc == "mdad":
            return tmp.to_frame().agg(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
        else:
            return tmp.agg(aggfunc)

    features = series.shift(periods=periods).rolling(window=seasonality * window if window != -1 else len(series) - 1,
                                                     min_periods=min_required_len).aggregate(aggregate_window)

    if fillna is not None:
        features.fillna(fillna, inplace=True)

    return features


def create_calendar_vars(df):
    """pd.DataFrame input with datetime index"""
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month

    def get_season(month):
        if (month > 11 or month < 3):
            return 0
        elif (month == 3 or month <= 5):
            return 1
        elif (month >= 6 and month < 9):
            return 2
        else:
            return 3

    df["season"] = df["month"].apply(lambda x: get_season(x))

    return df


def calculate_lags_and_stats(df, target, lags_range=None, moving_stats_range=None, periods=1, min_periods=1,
                             aggfunc="mean", seasonality=1):
    """TODO: add ability to add min-max or other complex agg functions"""
    if lags_range is not None:
        for i in lags_range:
            df[f"Lag_{i}"] = target.shift(i)

    if moving_stats_range is not None:
        for i in moving_stats_range:
            df[f"Moving_{aggfunc}_{i}"] = moving_stats(
                target, window=i, periods=periods, min_periods=min_periods, aggfunc=aggfunc,
                seasonality=seasonality,
            )
    return df


def split_time_check(dfs):
    for df in dfs:
        print(df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))


def mean_imputing(df):
    """fill nans with mean of first n nonnull stats"""
    pattern = df.columns.str.contains("Lag_|Moving_")
    features = df.columns[pattern].tolist()
    for feature in features:
        df[feature].fillna(df[feature][df[feature].notnull()].head(
            int(re.findall(r"\d+", feature)[0])
        ).mean(), axis=0, inplace=True)


def decompose(df, column_name, plot=False):
    """
    A function that returns the trend, seasonality and residual captured by applying both multiplicative and
    additive model.
    """
    result_mul = seasonal_decompose(df[column_name], model='multiplicative', extrapolate_trend='freq')
    result_add = seasonal_decompose(df[column_name], model='additive', extrapolate_trend='freq')

    if plot:
        plt.rcParams.update({'figure.figsize': (20, 10)})
        result_mul.plot().suptitle('Multiplicative Decompose', fontsize=30)
        result_add.plot().suptitle('Additive Decompose', fontsize=30)
        plt.show()

    return result_mul, result_add


# TODO: trend extraction - DeterministicProcess, divide from target for GBoosting to fit on residuals, than add to
# predictions
