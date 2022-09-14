import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from tqdm import tqdm
from scipy import stats
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import clone
import category_encoders as ce

from itertools import combinations, product
import time
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost
from catboost import CatBoostRegressor, Pool
import xgboost as xgb


seed = 42


def rmsle(y_true, y_pred) -> float:
    root_mean_squarred_log_error = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    return root_mean_squarred_log_error


class RMSLE:
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            val = max(approxes[index], 0)
            der1 = math.log1p(targets[index]) - math.log1p(max(0, approxes[index]))
            der2 = -1 / (max(0, approxes[index]) + 1)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class RMSLE_val:
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((math.log1p(max(0, approx[i])) - math.log1p(max(0, target[i])))**2)

        return error_sum, weight_sum


def check_missings(df: pd.DataFrame) -> pd.DataFrame:
    na = df.isnull().sum()
    result = pd.DataFrame({
        "Total": na,
        "Percent": 100*na/df.shape[0],
        "Types": df.dtypes
    })
    result = result[result["Total"] != 0]
    print(f"Total NA-values = {na.sum()}")
    return result.T


def create_numerical_aggs(data: pd.DataFrame,
                          groupby_id: str,
                          aggs: dict,
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None,
                          ) -> pd.DataFrame:
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""

    data_grouped = data.groupby(groupby_id)
    stats = data_grouped.agg(aggs)
    stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]
    stats = stats.reset_index()

    return stats


def create_categorical_aggs(data: pd.DataFrame,
                            groupby_id: str,
                            features: List[str],
                            prefix: Optional[str] = None,
                            suffix: Optional[str] = None,
                            ) -> pd.DataFrame:
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""

    categorical = pd.get_dummies(data[features])
    columns_to_agg = categorical.columns

    categorical[groupby_id] = data[groupby_id]
    data_grouped = categorical.groupby(groupby_id)
    stats = data_grouped.agg({col: ["mean", "sum"] for col in columns_to_agg})
    stats.columns = [f"{prefix}{feature}_{stat}{suffix}".upper() for feature, stat in stats]
    stats.columns = [col.replace("MEAN", "RATIO") for col in stats.columns]
    stats.columns = [col.replace("SUM", "TOTAL") for col in stats.columns]
    stats = stats.reset_index()

    return stats


def calculate_feature_separating_ability(
        features: pd.DataFrame, target: pd.Series, fill_value: float = -9999) -> pd.DataFrame:

    scores = {}
    for feature in features:
        score = roc_auc_score(
            target, features[feature].fillna(fill_value)
        )
        scores[feature] = 2* score - 1

    scores = pd.Series(scores)
    scores = scores.sort_values(ascending=False)

    return scores


def calculate_permutation_importance(estimator,
                                     metric: callable,
                                     x_valid: pd.DataFrame,
                                     y_valid: pd.DataFrame,
                                     maximize: bool = True,
                                     probas: bool = False
                                     ) -> pd.Series:

    def _predict(estimator, x_valid, probas=True):
        if hasattr(estimator, "predict_proba") and probas:
            y_pred = estimator.predict_proba(x_valid)[:, 1]
        else:
            y_pred = estimator.predict(x_valid)

        return y_pred

    y_pred = _predict(estimator, x_valid, probas)
    base_score = metric(y_valid, y_pred)
    scores, delta = {}, {}

    for feature in tqdm(x_valid.columns):
        x_valid_ = x_valid.copy(deep=True)
        dtype = x_valid_[feature].dtype
        np.random.seed(seed)
        x_valid_[feature] = np.random.permutation(x_valid_[feature])
        x_valid_[feature] = x_valid_[feature].astype(dtype)

        y_pred = _predict(estimator, x_valid_, probas)
        feature_score = metric(y_valid, y_pred)

        if maximize:
            delta[feature] = base_score - feature_score
        else:
            delta[feature] = feature_score - base_score

        scores[feature] = feature_score

    scores, delta = pd.Series(scores), pd.Series(delta)
    scores = scores.sort_values(ascending=False)
    delta = delta.sort_values(ascending=False)

    return scores, delta


def make_modify_cross_validation(X: pd.DataFrame,
                                 y: pd.Series,
                                 estimator: object,
                                 metric: callable,
                                 cv_strategy,
                                 error_to_be_outlier: None,
                                 verbose: bool = True,
                                 proba: bool = True,
                                 early: bool = False,
                                 regression: bool = False):
    estimators, fold_train_scores, fold_valid_scores = [], [], []
    oof_predictions = np.zeros(X.shape[0])

    for fold_number, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if early:
            estimator.fit(x_train, y_train,
                          eval_metric="auc",
                          eval_set=[(x_train, y_train), (x_valid, y_valid)],
                          early_stopping_rounds=early,
                          verbose=-1)
        else:
            estimator.fit(x_train, y_train)
        if proba:
            y_train_pred = estimator.predict_proba(x_train)[:, 1]
            y_valid_pred = estimator.predict_proba(x_valid)[:, 1]
        else:
            y_train_pred = estimator.predict(x_train)
            y_valid_pred = estimator.predict(x_valid)

        fold_train_scores.append(metric(y_train, y_train_pred))
        if not error_to_be_outlier:
            fold_valid_scores.append(metric(y_valid, y_valid_pred))
        else:
            if regression:
                mask = ((y_valid - y_valid_pred) / y_valid) < error_to_be_outlier  # for regression task: 0.05
            else:
                mask = y_valid[np.abs(y_valid - metric(y_valid, y_valid_pred)) < error_to_be_outlier].index  # 0.9 for classification with AUC
            fold_valid_scores.append(metric(y_valid.loc[mask], y_valid_pred.loc[mask]))
        oof_predictions[valid_idx] = y_valid_pred

        msg = (
            f"Fold: {fold_number + 1}, train-observations = {len(train_idx)}, "
            f"valid-observations = {len(valid_idx)}\n"
            f"train-score = {round(fold_train_scores[fold_number], 4)}, "
            f"valid-score = {round(fold_valid_scores[fold_number], 4)}"
        )
        if verbose:
            print(msg)
            print("=" * 69)
        if hasattr(estimator, "copy"):
            est = estimator.copy()
            estimators.append(est)
        else:
            estimators.append(estimator)

    if not error_to_be_outlier:
        oof_score = metric(y, oof_predictions)
    else:
        if regression:
            mask = ((y - oof_predictions) / y) < error_to_be_outlier  # for regression task
        else:
            mask = y[np.abs(y - metric(y, oof_predictions)) < error_to_be_outlier].index
        oof_score = metric(y.loc[mask], oof_predictions.loc[mask])

    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions


def reduce_memory_df(df, verbose=False):
    """Reduce memory usage by converting data to more appropriate dtypes"""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:4] != 'uint' and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif str(col_type)[:4] != 'uint':
            df[col] = df[col].astype('category')
    if verbose:
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def create_bootstrap_samples(data: np.array, n_samples: int = 1000) -> np.array:
    bootstrap_idx = np.random.randint(
        low=0, high=len(data), size=(n_samples, len(data))
    )
    return bootstrap_idx


def create_bootstrap_metrics(y_true: np.array,
                             y_pred: np.array,
                             metric: callable,
                             n_samlpes: int = 1000) -> List[float]:
    scores = []

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    bootstrap_idx = create_bootstrap_samples(y_true)
    for idx in bootstrap_idx:
        y_true_bootstrap = y_true[idx]
        y_pred_bootstrap = y_pred[idx]

        score = metric(y_true_bootstrap, y_pred_bootstrap)
        scores.append(score)

    return scores


def calculate_confidence_interval(scores: list, conf_interval: float = 0.95) -> Tuple[float]:
    left_bound = np.percentile(
        scores, ((1 - conf_interval) / 2) * 100
    )
    right_bound = np.percentile(
        scores, (conf_interval + ((1 - conf_interval) / 2)) * 100
    )

    return left_bound, right_bound


def mean_scores(scores, target):
    scores_list = [
        (scores.mean(axis=1), 'AMean score: '),
        (stats.gmean(scores, axis=1), 'GMean score: '),
        (scores.rank().mean(axis=1), 'Rank  score: '),
        (stats.gmean(scores.rank(), axis=1), 'GMean  rank: ')
    ]

    for scores_mean in scores_list:
        score = rmsle(target, scores_mean[0])
        print(f"{scores_mean[1]}{score:.5f}")


def get_encoded(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns


class BestSet(BaseEstimator, TransformerMixin):
    def __init__(self, params, k_features=10, scoring=roc_auc_score):
        self.scoring = scoring
        self.k_features = k_features
        self.params = params

    def fit(self, X, y):
        dim = X.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X, y, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores, subsets = [], []
            for p in tqdm(combinations(self.indices_, r=dim-1), total=dim, leave=False):
                score = self._calc_score(X, y, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        return self
    def transform(self, X):
        best_indices = list(self.subsets_[np.argmax(self.scores_)])
        return X.iloc[:, best_indices]

    def _calc_score(self, X, y, indices):
        fold = KFold(n_splits=5, shuffle=True, random_state=seed)
        _, oof_preds = lightgbm_cross_validation(params=self.params,
                                                 X=X.iloc[:, list(indices)],
                                                 y=y,
                                                 cv=fold,
                                                 verbose=False)
        score = self.scoring(y, oof_preds)
        return score


def lightgbm_cross_validation(params, X, y, cv, categorical=None, rounds=50, verbose=True):
    estimators, folds_scores, train_scores = [], [], []

    if not categorical:
        categorical = "auto"

    oof_preds = np.zeros(X.shape[0])
    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = LGBMRegressor(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmsle",
            verbose=0,
            early_stopping_rounds=rounds
        )
        train_score = model.predict(x_train)
        train_score = rmsle(y_train, train_score)
        oof_preds[valid_idx] = model.predict(x_valid)
        score = rmsle(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {rmsle(y, oof_preds):.5f}")
    return estimators, oof_preds


def catboost_cross_validation(params, X, y, cv, categorical=None, rounds=50):
    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])
    print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")
    if not categorical:
        categorical = [column for column in X.columns if X[column].dtype in ["object", "category"]]

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostRegressor(**params)
        train_pool = Pool(x_train, y_train, cat_features=categorical)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )
        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type='RawFormulaVal')  # Exponent
        train_score = rmsle(y_train, train_score)
        oof_preds[valid_idx] = catboost.CatBoost.predict(model, valid_pool, prediction_type='RawFormulaVal')
        score = rmsle(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    print_scores(folds_scores, train_scores)
    print(f"OOF-score: {rmsle(y, oof_preds):.5f}")
    return estimators, oof_preds


def lightgbm_cross_validation_mean(params, X, y, cv, categorical=None, rounds=50, verbose=True):
    estimators, folds_scores, train_scores = [], [], []

    if not categorical:
        categorical = "auto"

    oof_preds = np.zeros(X.shape[0])
    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = LGBMRegressor(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmsle",
            verbose=0,
            early_stopping_rounds=rounds
        )
        oof_preds[valid_idx] = model.predict(x_valid)
        train_score = model.predict(x_train)
        train_score = rmsle(y_train, train_score)
        score = rmsle(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {rmsle(y, oof_preds):.5f}")
    return estimators, oof_preds, np.mean(folds_scores)


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


def xgboost_cross_validation(params, X, y, cv, categorical=None, rounds=50, verbose=True, num_boost_rounds=600):
    estimators, encoders = [], []
    folds_scores, train_scores = [], []
    oof_preds = np.zeros(X.shape[0])
    X = X.copy(deep=True)

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if categorical:
            encoder = ce.cat_boost.CatBoostEncoder(random_state=42)
            x_train[categorical] = encoder.fit_transform(x_train[categorical].astype("str").fillna("NA"), y_train)
            x_valid[categorical] = encoder.transform(x_valid[categorical].astype("str").fillna("NA"))
            encoders.append(encoder)

        dtrain = xgb.DMatrix(x_train, y_train)
        dvalid = xgb.DMatrix(x_valid, y_valid)

        model = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        maximize=False,
                        num_boost_round=num_boost_rounds,
                        early_stopping_rounds=rounds,
                        evals=[(dtrain, "train"), (dvalid, "valid")],
                        verbose_eval=0,
        )
        train_score = model.predict(dtrain)
        train_score = rmsle(y_train, train_score)
        oof_preds[valid_idx] = model.predict(dvalid)
        score = rmsle(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {rmsle(y, oof_preds):.5f}")

    return estimators, encoders, oof_preds


def cross_validation(model, X, y, cv):
    estimators, folds_scores, train_scores = [], [], []
    oof_preds = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(X=x_train, y=y_train)

        train_score = model.predict(x_train)
        train_score = rmsle(y_train, train_score)
        train_scores.append(round(train_score, 5))

        oof_preds[valid_idx] = model.predict(x_valid)
        score = rmsle(y_valid, oof_preds[valid_idx])
        print(f"Fold {fold + 1}, Train score: {train_score:.5f}, Valid score = {score:.5f}")
        folds_scores.append(round(score, 5))
        estimators.append(clone(model).fit(x_train, y_train))

    print_scores(folds_scores, train_scores)
    return estimators, oof_preds, np.mean(folds_scores)


def check_duplicates_and_constants(df_train, df_test=None, threshold=5e-5):
    low_variance_cols = []
    test_exist = True if df_test is not None else False
    print(f"Initial train shape: {df_train.shape}")
    if test_exist:
        print(f"Initial test shape: {df_test.shape}")

    print(f"Duplicates in train: {df_train.duplicated().sum()}")
    if test_exist:
        print(f"Duplicates in test: {df_test.duplicated().sum()}")

    df_train.drop_duplicates(inplace=True)
    for column in df_train.columns:
        if df_train[column].nunique() < 2:
            end_phrase = "both from train and test sets" if test_exist else "from train set"
            print(f"{column} in train set is constant, removed {end_phrase}.")
            df_train.drop(column, axis=1, inplace=True)
            if test_exist:
                df_test.drop(column, axis=1, inplace=True)
        elif df_train[column].nunique() < 20:
            freqs = df_train[column].value_counts(dropna=True)
            freqs /= len(df_train)
            if freqs.iloc[1] < threshold:
                low_variance_cols.append(column)

    print(f"Final train shape: {df_train.shape}", end=' ')
    if test_exist:
        print(f", test shape: {df_test.shape}.")
    if len(low_variance_cols) > 0:
        print("Check next columns for usability: ", *low_variance_cols)
    return df_train, df_test if test_exist else df_train


def check_split_equality(x_train, x_test, estimator=None, params=None, categorical=None):
    x_inspect = pd.concat([x_train, x_test], axis=0)
    y_inspect = np.hstack((np.ones(x_train.shape[0]), np.zeros(x_test.shape[0])))

    if params is None:
        params = {'max_depth': 4,
                  'n_estimators': 100,
                  'verbose': -1,
                  'is_unbalance': True,
                  'learning_rate': 0.05,
                  'random_seed': 42,
                  'categorical_feature': categorical,
        }
    if estimator is None:
        inspector = LGBMClassifier(**params).fit(x_inspect, y_inspect)
    else:
        inspector = estimator

    cv = cross_val_score(
                         estimator=inspector,
                         X=x_inspect,
                         y=y_inspect,
                         scoring="roc_auc",
                         cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )
    print(f"CV-score: {round(np.mean(cv), 4)}")


def show_correlation_hist(df, columns=None):
    train_correlations = df.corr()
    train_correlations = train_correlations.values.flatten()
    train_correlations = train_correlations[train_correlations != 1]

    if columns is None:
        columns = train.columns.tolist()
        columns.remove(TARGET)

    plt.figure(figsize=(15, 5))
    sns.distplot(train_correlations, color="Blue", label="full dataset")

    plt.xlabel("Correlation values found in train (except 1.0)", size=14)
    plt.title("Are there correlations between features?", size=14)
    plt.legend(loc="best", fontsize=14)
    plt.ylabel("Density", size=14)
    plt.show()


def evaluate_preds(true_values, pred_values, save=False):
    print("R2:\t" + str(round(r2_score(true_values, pred_values), 3)) + "\n" +
          "MAE:\t" + str(round(mean_absolute_error(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(mean_squared_error(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(mean_squared_error(true_values, pred_values), 3)) + "\n"
         )
    print(f"RMSLE:\t{rmsle(true_values, pred_values):.5f}")
    data = pd.DataFrame(data={"true": true_values, "predicted": pred_values})
    plt.figure(figsize=(8, 8))
    sns.regplot(x="true", y="predicted", data=data)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')
    if save:
        if not os.path.exists('./saved'):
            os.makedirs('./saved')
        plt.savefig('./saved/' + 'report.png')
    plt.tight_layout()
    plt.show()
