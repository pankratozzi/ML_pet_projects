import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from tqdm import tqdm
from scipy import stats
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from itertools import combinations, product
import time
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier, Pool
import xgboost as xgb


seed = 42


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
                # y_valid_pred = pd.Series(data=y_valid_pred, index=y_valid.index, name='predictions')
                # outliers = iso.predict(x_valid.select_dtypes(include="number"))
                # mask = y_valid[outliers == 1].index
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
            # oof_predictions = pd.Series(data=oof_predictions, index=y.index, name="oof_predictions")
            # outliers = iso.predict(X.select_dtypes(include="number"))
            # mask = y[outliers == 1].index
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
        score = roc_auc_score(target, scores_mean[0])
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

        model = LGBMClassifier(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            verbose=0,
            early_stopping_rounds=rounds
        )
        train_score = model.predict_proba(x_train)[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds


def catboost_cross_validation(params, X, y, cv, categorical=None, rounds=50):
    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])
    print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostClassifier(**params)
        train_pool = Pool(x_train, y_train, cat_features=categorical)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )
        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type='Probability')[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = catboost.CatBoost.predict(model, valid_pool, prediction_type='Probability')[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    print_scores(folds_scores, train_scores)
    print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds


def lightgbm_cross_validation_group(params, X, y, cv, feature_name, categorical=None, rounds=50, verbose=True):
    estimators, folds_scores, train_scores = [], [], []

    if not categorical:
        categorical = "auto"

    oof_preds = np.zeros(X.shape[0])
    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, X[feature_name])):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        x_train.drop(feature_name, axis=1, inplace=True)
        x_valid.drop(feature_name, axis=1, inplace=True)
        model = LGBMClassifier(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric="auc",
            verbose=0,
            early_stopping_rounds=rounds
        )
        train_score = model.predict_proba(x_train)[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds


def catboost_cross_validation_group(params, X, y, cv, feature_name, categorical=None, rounds=50):
    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])
    print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=X[feature_name])):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostClassifier(**params)
        x_train.drop(feature_name, axis=1, inplace=True)
        x_valid.drop(feature_name, axis=1, inplace=True)
        train_pool = Pool(x_train, y_train, cat_features=categorical)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )
        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type='Probability')[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = catboost.CatBoost.predict(model, valid_pool, prediction_type='Probability')[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    print_scores(folds_scores, train_scores)
    print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
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

        model = LGBMClassifier(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            verbose=0,
            early_stopping_rounds=rounds
        )
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        train_score = model.predict_proba(x_train)[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds, np.mean(folds_scores)


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


def xgboost_cross_validation(params, X, y, cv, categorical=None, rounds=50, verbose=True):
    estimators, encoders = [], {}
    folds_scores, train_scores = [], []
    oof_preds = np.zeros(X.shape[0])

    if categorical:
        for feature in categorical:
            encoder = LabelEncoder()
            X[feature] = encoder.fit_transform(X[feature].astype("str").fillna("NA"))
            encoders[feature] = encoder

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        dtrain = xgb.DMatrix(x_train, y_train)
        dvalid = xgb.DMatrix(x_valid, y_valid)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            maximize=True,
            num_boost_round=10000,
            early_stopping_rounds=rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            verbose_eval=0,
        )
        train_score = model.predict(dtrain)
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = model.predict(dvalid)
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")

    return estimators, encoders, oof_preds


def cross_validation(model, X, y, cv):
    estimators, folds_scores, train_scores = [], [], []
    oof_preds = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(X=x_train, y=y_train)

        train_score = model.predict_proba(x_train)[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        train_scores.append(round(train_score, 5))

        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        print(f"Fold {fold + 1}, Train score: {train_score:.5f}, Valid score = {score:.5f}")
        folds_scores.append(round(score, 5))
        estimators.append(clone(model).fit(x_train, y_train))

    print_scores(folds_scores, train_scores)
    return estimators, oof_preds, np.mean(folds_scores)


def report(y_train, y_train_pred, y_test, y_test_pred, y_train_proba=None, y_test_proba=None):
    """display classification report"""
    print('Train\n', classification_report(y_train, y_train_pred, digits=4))
    print('Test\n', classification_report(y_test, y_test_pred, digits=4))
    if y_train_proba is not None and y_test_proba is not None:
        roc_train, roc_test = roc_auc_score(y_train, y_train_proba), roc_auc_score(y_test, y_test_proba)
        print(f'Train ROC_AUC: {roc_train:.3f}, Test ROC_AUC: {roc_test:.3f}')
        print(f'Train GINI: {(2 * roc_train - 1):.3f}, Test GINI: {(2 * roc_test - 1):.3f}')


# confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(None)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=4)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(fpr, tpr, model_name="", color=None):
    plt.plot(fpr, tpr, label='%s: ROC curve (area = %0.2f)' %
                             (model_name, auc(fpr, tpr)), color=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s: Receiver operating characteristic curve' % model_name)
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(recall, precision, model_name="", color=None):
    plt.plot(recall, precision, label='%s: Precision-Recall curve (area = %0.2f)' %
                                      (model_name, auc(recall, precision)), color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("%s: Precision-Recall curve" % model_name)
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.legend(loc="lower left")
    plt.show()


# calibration probs
def get_best_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fscore = 2 * precision * recall / (precision + recall)
    ix = np.argmax(fscore)
    return thresholds[ix], fscore[ix]


def show_proba_calibration_plots(y_predicted_probs, y_true_labels):
    """display probability calibration results"""
    preds_with_true_labels = np.array(list(zip(y_predicted_probs, y_true_labels)))

    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in np.linspace(0.1, 0.9, 18):
        thresholds.append(threshold)
        precisions.append(precision_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        recalls.append(recall_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))
        f1_scores.append(f1_score(y_true_labels, list(map(int, y_predicted_probs > threshold))))

    scores_table = pd.DataFrame({'f1': f1_scores,
                                 'precision': precisions,
                                 'recall': recalls,
                                 'probability': thresholds}).sort_values('f1', ascending=False).round(3)

    figure = plt.figure(figsize=(15, 5))

    plt1 = figure.add_subplot(121)
    plt1.plot(thresholds, precisions, label='Precision', linewidth=4)
    plt1.plot(thresholds, recalls, label='Recall', linewidth=4)
    plt1.plot(thresholds, f1_scores, label='F1', linewidth=4)
    plt1.set_ylabel('Scores')
    plt1.set_xlabel('Probability threshold')
    plt1.set_title('Probabilities threshold calibration')
    plt1.legend(bbox_to_anchor=(0.25, 0.25))
    plt1.table(cellText=scores_table.values,
               colLabels=scores_table.columns,
               colLoc='center', cellLoc='center', loc='bottom', bbox=[0, -1.3, 1, 1])

    plt2 = figure.add_subplot(122)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 0][:, 0],
              label='Another class', color='royalblue', alpha=1)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 1][:, 0],
              label='Main class', color='darkcyan', alpha=0.8)
    plt2.set_ylabel('Number of examples')
    plt2.set_xlabel('Probabilities')
    plt2.set_title('Probability histogram')
    plt2.legend(bbox_to_anchor=(1, 1))

    plt.show()


def simple_cross_validation(clf, X, y, scoring='f1', cv=5):
    """cross validation"""
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'Меры правильности перекрекстной оценки: {scores}')
    print(f'Точность перекретсной оценки: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
    return scores


def categorical_stats(df, target, alpha=0.05, sample_size=500):
    """chi2 stat test for categorical features"""
    data = df.copy().sample(sample_size)
    columns_to_analize = data.select_dtypes(include=['category', 'object']).columns
    weak_list = []
    for factor in columns_to_analize:
        if factor == target:
            continue
        print(f'{factor}')
        table = pd.crosstab(data[factor], data[target])
        p_value = stats.chi2_contingency(table, correction=False)[1]
        if p_value < alpha:
            print(f'Feature {factor} has statistical impact on target. P-value: {p_value:.6f}')
        else:
            weak_list.append(factor)
    if len(weak_list) > 0:
        print(f'Statistically weak categorical features: ', *weak_list)


def statistic_output(df, target, *columns, alpha=0.05, sample_size=0):
    """statistical test for numerical features"""
    data = df.copy()
    data.drop_duplicates(inplace=True)
    if sample_size == 0:
        sample_size = int(0.05 * len(data))
    for column in columns:
        df_sampled = data[[column, target]].sample(sample_size, random_state=1)
        factor_a = df_sampled.loc[df_sampled[target] == 0][column]
        factor_b = df_sampled.loc[df_sampled[target] == 1][column]
        var_a, var_b = factor_a.var(), factor_b.var()
        _, pvalue = stats.shapiro(df_sampled[column])
        if pvalue >= alpha:
            _, pvalue = stats.ttest_ind(factor_a, factor_b, equal_var=False)
        else:
            if len(factor_a) == 0 or len(factor_b) == 0:
                continue
            _, pvalue = stats.mannwhitneyu(factor_a, factor_b)
        if pvalue < alpha:
            print(f'Factor "{column}" has statistical impact on target (var_a: {var_a:.2f}, var_b: {var_b:.2f}).')
        else:
            print(f'Factor "{column}" does not affect target.')


def categorical_output(df, target, *columns, alpha=0.05, sample_size=0):
    data = df.copy()
    data.drop_duplicates(inplace=True)
    data['id'] = np.arange(len(data)).astype('object')
    if sample_size <= 0:
        sample_size = int(0.05 * len(data))
    for column in columns:
        print(column)
        categories = data[column].unique().tolist()
        for pair in combinations(categories, r=2):
            a, b = pair
            if a != b:
                data_ = data.loc[data[column].isin(pair), ['id', column, target]].sample(sample_size, random_state=seed)
                table = data_.pivot_table(values='id', index=column, columns=target, aggfunc='count').fillna(0)
                try:
                    _, pvalue, _, _ = stats.chi2_contingency(table, correction=False)
                except ValueError:
                    continue
                if pvalue >= alpha:
                    print(f'Categories {a} and {b} can be united. P-value: {pvalue:.4f}')
                else:
                    print(f'Categories {a} and {b} have different frequencies with target, p-value: {pvalue:.4f}.')


def check_duplicates_and_constants(df_train, df_test=None):
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

    print(f"Final train shape: {df_train.shape}", end=' ')
    if test_exist:
        print(f", test shape: {df_test.shape}.")

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


def countplot(df: pd.DataFrame, target: str = None):
    categorical_columns = [column for column in df.columns if df[column].dtype in ["object", "category"]]
    fig_size = (20, len(categorical_columns))
    rows, cols = len(categorical_columns) // 4 + 1, 4
    plt.figure(figsize=fig_size)
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(rows, cols, i)
        plt.title(f'{column}')
        sns.countplot(x=column, hue=target, data=df)
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def boxplot(df: pd.DataFrame, target: str = None):
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    fig_size = (20, len(numerical_columns) // 2)
    rows, cols = len(numerical_columns) // 4 + 1, 4
    plt.figure(figsize=fig_size)
    for idx, column in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, idx)
        sns.boxplot(y=df[column], x=df[target], data=df)
        plt.title(f'{column}')
    plt.tight_layout()
    plt.show()


def histplot(df: pd.DataFrame, target: str = None, bins: int = 30):
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    fig_size = (20, len(numerical_columns) // 2)
    rows, cols = len(numerical_columns) // 4 + 1, 4
    plt.figure(figsize=fig_size)
    for idx, column in enumerate(numerical_columns, 1):
        plt.subplot(rows, cols, idx)
        dist = 'Normal Distribution' if stats.shapiro(df[column].sample(200, random_state=seed))[1] > 0.05 \
            else 'Not normal distribution'
        plt.title(f'{column}: {dist}')
        sns.histplot(data=df, x=column, hue=target, bins=bins, kde=True)
    plt.tight_layout()
    plt.show()


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list, normalize=False) -> None:
        self.columns = columns
        self.normalize = normalize
        self.mappers = {}

    def fit(self, X, y=None):
        X = X.copy(deep=True)
        for column in self.columns:
            X[column] = X[column].astype("str")
            values, counts = np.unique(X[column], return_counts=True)
            if self.normalize:
                counts /= len(X.shape[0])
            mapper = pd.Series(data=counts, index=values)
            self.mappers[column] = mapper
        return self

    def transform(self, X):
        X = X.copy(deep=True)
        for column in self.columns:
            X[column] = X[column].astype("str")
            mapper = self.mappers.get(column)
            X[column] = X[column].map(mapper).astype("category")
        return X


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def make_lags(ts, lags, lead_time=1, feature="_"):
    return pd.concat(
        {
            f'{feature}_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def catboost_cross_validation_group_proc(params,
                                         X,
                                         y,
                                         cv,
                                         feature_name,
                                         preprocess=True,
                                         rounds=200):
    estimators, folds_scores, train_scores, preprocessors = [], [], [], []

    oof_preds = np.zeros(X.shape[0])
    print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=X[feature_name])):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostClassifier(**params)
        x_train.drop(feature_name, axis=1, inplace=True)
        x_valid.drop(feature_name, axis=1, inplace=True)
        if preprocess:
            preprocessor = Preprocessor()
            x_train = preprocessor.fit_transform(x_train)
            x_valid = preprocessor.transform(x_valid)
            preprocessors.append(preprocessor)

        categorical = [column for column in x_train.columns if x_train[column].dtype in ["object", "category"]]
        x_train[categorical] = x_train[categorical].astype("str")
        x_valid[categorical] = x_valid[categorical].astype("str")

        train_pool = Pool(x_train, y_train, cat_features=categorical)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )
        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type='Probability')[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = catboost.CatBoost.predict(model, valid_pool, prediction_type='Probability')[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)
        gc.collect()

    print_scores(folds_scores, train_scores)
    print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds, preprocessors


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, prepare_for_user_split=False):
        self.constants = []
        self.cat_freq_features = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
        self.encoder1 = FrequencyEncoder(self.cat_freq_features)
        self.train_agg_amt_means = {}
        self.train_agg_amt_stds = {}
        self.train_agg_amt_means_d15 = {}
        self.train_agg_amt_stds_d15 = {}
        self.encoder2 = FrequencyEncoder(["P_emaildomain", "R_emaildomain"])
        self.pipe = make_pipeline(StandardScaler(), PCA(n_components=None, random_state=seed))
        self.numerical_columns = []
        self.aggs = {"TransactionAmt": ["mean", "sum", "count", "min", "max"]}
        self.trans_agg_train = None
        self.is_fitted = False
        self.prepare_for_user_split = prepare_for_user_split

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            print(f"Pass returned pandas dataframe with valid columns.")
            return self

        for column in X.columns:
            if X[column].nunique() < 2:
                self.constants.append(column)

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X, columns=[f"feat_{i + 1}" for i in range(X.shape[1])])
            print(f"Pass returned pandas dataframe with valid columns.")
            return X
        else:
            X = X.copy(deep=True)
        X = reduce_memory_df(X)
        X.drop(self.constants, axis=1, inplace=True)

        X['card_hash'] = X.apply(lambda x: self.card_info_hash(x), axis=1)

        X["TransactionDT"] = pd.Timestamp('2017-12-01') + pd.to_timedelta(X["TransactionDT"], unit='s')
        X["TransactionYear"] = X["TransactionDT"].dt.year.astype("category")
        X["TransactionMonth"] = X["TransactionDT"].dt.month.astype("category")
        X["TransactionDayWeek"] = X["TransactionDT"].dt.dayofweek.astype("category")
        X["TransactionHour"] = X["TransactionDT"].dt.hour.astype("category")
        X["TransactionDay"] = X["TransactionDT"].dt.day.astype("category")

        X["card1_2"] = (X["card1"].apply(str) + '_' + X["card2"].apply(str)).astype("category")
        X["card1_2_3_5"] = (X["card1"].apply(str) + '_' + X["card2"].apply(str) + '_' + X["card3"].apply(str) + '_' + X[
            "card5"].apply(str))
        X["card_addr"] = (X["card1_2_3_5"] + '_' + X["addr1"].apply(str) + '_' + X["addr2"].apply(str)).astype(
            "category")
        X["card1_2_3_5"] = X["card1_2_3_5"].astype("category")

        if not self.is_fitted:
            X = self.encoder1.fit_transform(X)

            for feature in self.cat_freq_features:
                self.train_agg_amt_means[feature] = X.groupby(feature)["TransactionAmt"].mean()
                self.train_agg_amt_stds[feature] = X.groupby(feature)["TransactionAmt"].std()
                self.train_agg_amt_means_d15[feature] = X.groupby(feature)["D15"].mean()
                self.train_agg_amt_stds_d15[feature] = X.groupby(feature)["D15"].std()
        else:
            X = self.encoder1.transform(X)

        for feature in self.cat_freq_features:
            X[f"AMT_BY_{feature}_mean"] = X["TransactionAmt"] / X[feature].map(
                self.train_agg_amt_means.get(feature)).astype(float)
            X[f"AMT_BY_{feature}_diff"] = X["TransactionAmt"] - X[feature].map(
                self.train_agg_amt_means.get(feature)).astype(float)
            X[f"AMT_BY_{feature}_std"] = X["TransactionAmt"] / X[feature].map(
                self.train_agg_amt_stds.get(feature)).astype(float)

            X[f"D15_BY_{feature}_mean"] = X["D15"] / X[feature].map(self.train_agg_amt_means_d15.get(feature)).astype(
                float)
            X[f"D15_BY_{feature}_diff"] = X["D15"] - X[feature].map(self.train_agg_amt_means_d15.get(feature)).astype(
                float)
            X[f"D15_BY_{feature}_std"] = X["D15"] / X[feature].map(self.train_agg_amt_stds_d15.get(feature)).astype(
                float)

        X["INT_TransAmt"] = X["TransactionAmt"] // 1
        X["FL_TransAmt"] = X["TransactionAmt"] % 1
        X["TransactionAmt_log"] = np.log(X["TransactionAmt"])

        if not self.is_fitted:
            self.encoder2.fit(X)

        X = self.encoder2.transform(X)

        cols_d = [column for column in X.columns[:393] if column.startswith("D")]
        cols_c = [column for column in X.columns[:393] if column.startswith("C")]
        cols_v = [column for column in X.columns[:393] if column.startswith("V")]

        X["D_mean"] = X[cols_d].mean(axis=1)
        X["D_std"] = X[cols_d].std(axis=1)

        X["C_mean"] = X[cols_c].mean(axis=1)
        X["C_std"] = X[cols_c].std(axis=1)

        X["V_mean"] = X[cols_v].mean(axis=1)
        X["V_std"] = X[cols_v].std(axis=1)

        try:
            check_is_fitted(self.pipe)
        except:
            self.numerical_columns = [col for col in X.columns[2:] if X[col].dtype not in ["object", "category"]]
            # if np.any(np.isinf(X[numerical_columns])):
            #     X[numerical_columns] = X[numerical_columns].replace(np.inf, -1).replace(-np.inf, -1)
            self.pipe.fit(
                X[self.numerical_columns].fillna(0).replace(np.inf, -1).replace(-np.inf, -1))  # were init num_cols

        pca_features = self.pipe.transform(X[self.numerical_columns].fillna(0).replace(np.inf, -1).replace(-np.inf, -1))
        pca_features = pd.DataFrame(data=pca_features[:, :61], columns=[f"PCA_{i}" for i in range(1, 62)],
                                    index=X.index)
        X = pd.concat([X, pca_features], axis=1)

        # apply only train statistics
        if not self.is_fitted:
            self.trans_agg_train = create_numerical_aggs(X, groupby_id="card_hash", aggs=self.aggs,
                                                         suffix="_BY_CARD_HASH")
            self.is_fitted = True
        X = pd.merge(X, self.trans_agg_train, on="card_hash", how='left')

        lags = make_lags(X.sort_values("TransactionDT").groupby("card_hash")["TransactionAmt"], lags=5,
                         feature="TransactionAmt").sort_index()
        X = pd.merge(X, lags, right_index=True, left_index=True)

        if self.prepare_for_user_split:
            X.drop(["TransactionDT", "TransactionID"], axis=1, inplace=True)
        else:
            X.drop(["TransactionDT", "TransactionID", "card_hash"], axis=1, inplace=True)
        gc.collect()
        return X

    @staticmethod
    def card_info_hash(x):
        s = (str(x['card1']) + str(x['card2']) + str(x['card3']) + str(x['card4']) + str(x['card5']) + str(x['card6']) +
             x["ProductCD"])
        h = hashlib.sha256(s.encode('utf-8')).hexdigest()[0:15]
        return h
