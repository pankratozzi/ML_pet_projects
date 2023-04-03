from sklearn import metrics
from functools import partial
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.pipeline import Pipeline as sklearn_pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from catboost import CatBoostRegressor, Pool
from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.analysis import plot_forecast
from etna.transforms import TimeSeriesImputerTransform


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


def timeseries_cv_with_lags_and_moving_stats(data, y_data, model, lags_range=None, moving_stats_range=None,
                                             aggfunc="mean", seasonality=1, print_cv_scheme=False,
                                             print_features=False, visualize=True, last_n_train=5,
                                             max_train_size=None, test_size=None, n_splits=3, gap=0,
                                             fillna=None,
                                             metric=partial(metrics.mean_squared_error, squared=False)):
    if min(lags_range) < test_size:
        warnings.warn("The number of lags periods should be equal\n"
                      "or more than forecasting horizon.")

    if min(moving_stats_range) < test_size:
        warnings.warn("The size of moving window should be equal\n"
                      "or more than forecasting horizon.")

    tscv = TimeSeriesSplit(max_train_size=max_train_size, test_size=test_size, n_splits=n_splits, gap=gap)

    metric_list = []
    for cnt, (train_index, test_index) in enumerate(tscv.split(data), 1):
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        y_test_N = y_test.copy(deep=True)

        y_test_N[:] = np.NaN
        tmp_target = pd.concat([y_train, y_test_N])
        concat_data = pd.concat([x_train, x_test])

        if print_cv_scheme:
            print("-"*40)
            print("Train:",
                  [x_train.index[0].strftime("%Y-%m-%d"),
                   x_train.index[-1].strftime("%Y-%m-%d")],
                  "Test:",
                  [x_test.index[0].strftime("%Y-%m-%d"),
                   x_test.index[-1].strftime("%Y-%m-%d")])
            print("\nTotal number of observations: %d" % (len(x_train) + len(x_test)))
            print("Train size: %d" % (len(x_train)))
            print("Test size: %d" % (len(x_test)))

        if print_features:
            print(f"\nDefence:\n\n{tmp_target}\n")

        if lags_range is not None:
            for i in lags_range:
                concat_data[f"Lag_{i}"] = tmp_target.shift(i)

        if moving_stats_range is not None:
            for i in moving_stats_range:
                concat_data[f"Moving_{aggfunc}_{i}"] = moving_stats(tmp_target, window=i, aggfunc=aggfunc,
                                                                    seasonality=seasonality)

        if print_features:
            pattern = concat_data.columns.str.contains("Lag_|Moving_")
            feature = concat_data.columns[pattern]
            print(f"Added features:\n{concat_data[feature]}")

        concat_data = concat_data.sort_index(axis=1)

        x_train = concat_data[:-test_size]
        x_test = concat_data[-test_size:]

        if fillna == "zero":
            x_train = x_train.fillna(0, axis=0)
        if fillna == "mean":
            x_train = x_train.fillna(x_train.mean(), axis=0)

        if model.__class__.__name__ == "CatBoostRegressor":
            cat_indices = np.where(x_train.dtypes == object)[0]
            train_pool = Pool(x_train, y_train, cat_features=cat_indices)
            model.fit(train_pool)
        else:
            model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        predictions = pd.Series(predictions, index=x_test.index)

        metric_value = metric(y_test, predictions)
        metric_list.append(metric_value)

        print(f"\nMetric value: {metric_value:.4f} on {cnt} iteration\n")

        if visualize:
            plt.figure(figsize=(8, 4))
            plt.xticks(rotation=90)
            plt.plot(y_train.iloc[-last_n_train:], label="train dataset")
            plt.plot(predictions, label="forecasts", color="red")
            plt.plot(y_test, label="test dataset", color="green")
            plt.grid()
            plt.legend()
            plt.show()

        metrics_mean = np.mean(metric_list)
        print(f"Mean value of metric: {metrics_mean:.4f}")

        if print_features:
            feature_list = concat_data.columns.tolist()
            print(f"\nFeatures list:\n{feature_list}")


class GroupKFoldAndTimeSeriesSplit:
    def __init__(self, group_n_splits, time_n_splits):
        self.group_n_splits = group_n_splits
        self.time_n_splits = time_n_splits

    def get_n_splits(self, X, y, groups):
        return self.group_n_splits

    def split(self, X, y=None, groups=None):
        group_k_fold = GroupKFold(n_splits=self.group_n_splits)
        for train_index, test_index in group_k_fold.split(X, y, groups=groups):
            tscv = TimeSeriesSplit(n_splits=self.time_n_splits)
            for train_index_t, test_index_t in tscv.split(X, y):
                yield np.intersect1d(train_index, train_index_t), np.intersect1d(test_index, test_index_t)


def differences(series, periods=1):
    diff = []
    for i in range(periods, len(series)):
        value = series[i] - series[i - periods]
        diff.append(value)
    return pd.Series(diff)


def etna_validation_optim(ts, train_start, train_end, valid_start, valid_end, model,
                          horizon, transforms, metrics, optim="minimize"):
    best_score = np.inf if optim == "minimize" else -np.inf
    best_params = None

    for transform in transforms:
        train_ts, valid_ts = ts.train_test_split(train_start=train_start, train_end=train_end,
                                                 test_start=valid_start, test_end=valid_end)

        pipe = Pipeline(model=model, transforms=transform, horizon=horizon)
        pipe.fit(train_ts)

        forecast_ts = pipe.forecast(valid_ts)
        metrics_score = metrics(y_true=y_valid, y_pred=forecast_ts).get("main")
        print(f"Transform: {transform}")
        print(f"{metrics.__class__.__name__}: {metrics_score}")

        if (optim == "minimize" and metrics_score < best_score) or (optim == "maximize" and metrics_score > best_score):
            best_score = metrics_score
            best_params = {'transform': transform}

    print(f"Best transformation set: {best_params}")
    print(f"Best metric {metrics.__class__.__name__}: {best_score}")


def etna_cv_optimize(ts, model, horizon, transforms, n_folds, mode, metrics,
                     refit=True, n_train_samples=10, optim="minimize"):
    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    best_params = None
    best_score = np.inf if optim == "minimize" else -np.inf

    for transform in transforms:
        pipe = Pipeline(model=model, transforms=transform, horizon=horizon)
        df_metrics, _, _ = pipe.backtest(mode=mode, n_folds=n_folds, ts=train_ts, metrics=[metrics],
                                         aggregate_metrics=False, joblib_params=dict(verbose=0))

        metrics_mean = df_metrics[metrics.__class__.__name__].mean()
        metrics_std = df_metrics[metrics.__class__.__name__].std()

        print(f"Transform:\n{transform}")
        print(f"{metrics.__class__.__name__}_mean: {metrics_mean}")
        print(f"{metrics.__class__.__name__}_std: {metrics_std}")

        if (optim == "minimize" and metrics_mean < best_score) or (optim == "maximize" and metrics_mean > best_score):
            best_score = metrics_mean
            best_params = {'transform': transform}

    print(f"Best transformation set:\n{best_params}")
    print(f"Best metric {metrics.__class__.__name__}: {best_score}")

    if refit:
        pipe = Pipeline(model=model, transforms=best_params.get("transform"), horizon=horizon)
        pipe.fit(train_ts)
        forecast_ts = pipe.forecast()

        print(metrics(y_true=test_ts, y_pred=forecast_ts))

        plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=n_train_samples)


def train_and_evaluate(ts, model, transforms, horizon, metrics, print_metrics=True, print_plots=True,
                       n_train_samples=10):
    if not print_plots and n_train_samples is None:
        raise ValueError(f"Set n_train_samples should be set with print_plots=True")

    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    pipe = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipe.fit(train_ts)
    forecast_ts = pipe.forecast()

    segment_metrics = metrics(test_ts, forecast_ts)
    segment_metrics = pd.Series(segment_metrics)

    if print_metrics:
        print(segment_metrics.to_string(), "\n")
        print(f"Mean metric: {np.mean(segment_metrics)}")

    if print_plots:
        plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=n_train_samples)


def etna_impute(df, strategy="mean", window=1, seasonality=1):
    ts = TSDataset(df, freq="D")
    imputer = TimeSeriesImputerTransform(in_column="target", strategy=strategy, window=window,
                                         seasonality=seasonality)
    ts.fit_transform([imputer])
    ts.plot()
    return ts


def get_trend_estimator(df, degree=1, mode="additive"):
    # log transform and detrending
    df = df.copy(deep=True)
    df["target"] = np.log1p(df["target"]) / np.log(10)
    series_len = len(df)
    x = df.index.to_series()
    if isinstance(type(x.dtype), pd.Timestamp):
        raise ValueError(f"Index dtype should be np.datetime64 or datetime.datetime")

    x = x.apply(lambda ts: ts.timestamp())
    x = x.to_numpy().reshape(series_len, 1)

    pipe = sklearn_pipeline([("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                             ("regression", LinearRegression())])

    pipe.fit(x, df["target"].values)
    trend = pipe.predict(x)
    if mode == "additive":
        target = df["target"] - trend
    elif mode == "multiplicative":
        target = df["target"] / trend
    else:
        raise ValueError("Set mode == 'additive' or 'multiplicative'")
    # use pipe to predict test trend (create test timestamp as feature)
    return target, pipe


def inverse_trend(test, pipe, preds=None, mode="additive", get_trend=False):
    # set 'get_trend' to True in case of using trend as feature
    # otherwise return modified predictions (adding trend and exp)
    test = test.copy(deep=True)
    series_len = len(test)
    x = test.index.to_series()
    if isinstance(type(x.dtype), pd.Timestamp):
        raise ValueError(f"Index dtype should be np.datetime64 or datetime.datetime")

    x = x.apply(lambda ts: ts.timestamp())
    x = x.to_numpy().reshape(series_len, 1)

    trend = pipe.predict(x)
    if get_trend:
        return trend

    if mode == "additive":
        preds = np.expm1((preds + trend) * np.log(10))
    elif mode == "multiplicative":
        preds = np.expm1((preds * trend) * np.log(10))
    else:
        raise ValueError("Set mode == 'additive' or 'multiplicative'")

    return preds


def etna_staged_cv_optimize(ts, model, horizon, init_transforms, transforms, n_folds, mode, metrics,
                            refit=True, n_train_samples=15, optim="minimize"):
    train_ts, test_ts = ts.train_test_split(test_size=horizon)
    best_score = np.inf if optim == "minimize" else -np.inf
    best_params = {}

    el_list, transform_list = [], []

    for el in transforms:
        el_list.append(el)
        transform_list.append(el_list.copy())

    for i in range(len(transform_list)):
        transform_list[i] = init_transforms + transform_list[i]

    for transform in transform_list:
        pipe = Pipeline(model=model, transforms=transform, horizon=horizon)
        df_metrics, _, _ = pipe.backtest(mode=mode, n_folds=n_folds, ts=train_ts, metrics=[metrics],
                                         aggregate_metrics=False, joblib_params=dict(verbose=0))

        metrics_mean = df_metrics[metrics.__class__.__name__].mean()
        metrics_std = df_metrics[metrics.__class__.__name__].std()

        print(f"Transforms:\n{transform}")
        print(f"{metrics.__class__.__name__}_mean: {metrics_mean}")
        print(f"{metrics.__class__.__name__}_std: {metrics_std}")

        if (optim == "minimize" and metrics_mean < best_score) or (optim == "maximize" and metrics_mean > best_score):
            best_score = metrics_mean
            best_params = {'transform': transform}

    print(f"Best transform set:\n{best_params}\n")
    print(f"Best {metrics.__class__.__name__} cv: {best_score}\n")

    if refit:
        pipe = Pipeline(model=model, transforms=best_params.get("transform"), horizon=horizon)
        pipe.fit(train_ts)
        forecast_ts = pipe.forecast()

        print(metrics(y_true=test_ts, y_pred=forecast_ts))

        plot_forecast(forecast_ts, test_ts, train_ts, n_train_samples=n_train_samples)


def groupby_moving_stats(df, by, target, min_periods=1, window=4, fillna=0,
                         aggfunc="mean", offset=False, start_date=None):
    if offset and start_date is not None:
        df = df[df.index >= start_date]

    if not isinstance(by, list):
        by = [by]

    df = df.groupby(by)[target].transform(lambda x: x.shift(1).rolling(
        window=window, min_periods=min_periods
    ).agg(aggfunc))

    df.fillna(fillna, inplace=True)
    return df


def expanding_stats(series, min_periods=1, periods=1, aggfunc="mean", fillna=0):
    features = series.shift(periods=periods).expanding(min_periods=min_periods).agg(aggfunc)
    features.fillna(fillna, inplace=True)
    return features


def grouped_stats(df_target, df_calc, var, by, func="mean", fillna=None):
    if not isinstance(by, list):
        by = [by]

    name = f"{var}_by_{by}_{func}"
    grp = df_calc.groupby(by)[[var]].agg(func)
    grp.columns = [name]

    var = pd.merge(df_target[by], grp, left_on=by, right_index=True, how="left")[name]

    if fillna is not None:
        var.fillna(fillna, inplace=True)

    return var


def forecast_trend(train_target, calc_trend_for_test=False, test_size=4, freq="MS"):
    regressor = make_pipeline(PolynomialFeatures(degree=1, include_bias=False),
                              LinearRegression(fit_intercept=False))
    n_timepoints = len(train_target)
    X = np.arange(n_timepoints).reshape(-1, 1)

    regressor.fit(X, train_target)
    coefs = regressor.named_steps["linearregression"].coef_

    train_trend_pred = regressor.predict(X)
    train_trend_pred = pd.Series(train_trend_pred, index=train_target.index)

    if not calc_trend_for_test:
        return train_trend_pred, coefs
    else:
        start = train_trend_pred[-1] + coefs[1]
        count = test_size
        step = coefs[1]
        numbers = []

        for i in range(count):
            numbers.append(start)
            start += step

        test_trend_pred = pd.Series(numbers)
        future_dates = pd.date_range(start=train_target.index[-1],
                                     periods=test_size + 1,
                                     freq=freq,
                                     closed="right")
        test_trend_pred.index = future_dates
        # forecasted trends. Later: subtract or divide it from the original target
        # additive or multiplicative trend
        return train_trend_pred, test_trend_pred


# TODO: trend extraction - DeterministicProcess, divide from target for GBoosting to fit on residuals, than add to
# predictions (check residuals trend)
