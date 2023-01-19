import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

import sklearn.model_selection

from tqdm import tqdm
from scipy import stats
import re
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, classification_report, auc, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.base import clone

from itertools import combinations
import time
import catboost
from catboost import CatBoostClassifier, Pool

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


def _predict(estimator, x_valid, probas=True):
    if hasattr(estimator, "predict_proba") and probas:
        y_pred = estimator.predict_proba(x_valid)[:, 1]
    else:
        y_pred = estimator.predict(x_valid)

    return y_pred


def calculate_permutation_importance(estimator,
                                     metric: callable,
                                     x_valid: pd.DataFrame,
                                     y_valid: pd.DataFrame,
                                     maximize: bool = True,
                                     probas: bool = False
                                     ) -> pd.Series:
    y_pred = _predict(estimator, x_valid, probas)
    base_score = metric(y_valid, y_pred)
    scores, delta = {}, {}

    for feature in tqdm(x_valid.columns):
        x_valid_ = x_valid.copy(deep=True)
        np.random.seed(seed)
        x_valid_[feature] = np.random.permutation(x_valid_[feature])

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


def catboost_cross_validation(X: pd.DataFrame,
                              y: pd.Series,
                              params: dict = None,
                              cv=None,
                              categorical: list = None,
                              textual: list = None,
                              rounds: int = 50,
                              verbose: bool = True,
                              preprocess: object = None,
                              score_fn: callable = roc_auc_score,
                              calculate_ci: bool = False,
                              n_samples: int = 1000,
                              confidence: float = 0.95,
                              seed: int = 42):

    minor_class_counts = y.value_counts(normalize=True).values[-1]

    if cv is None:
        if minor_class_counts >= 0.05:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if params is None:
        if len(X) <= 50_000:
            sub_params = {
                "grow_policy": "SymmetricTree",
                "boosting_type": "Ordered",
                "score_function": "Cosine",
                "depth": 6,
            }
        else:
            sub_params = {
                "grow_policy": "Lossguide",
                "boosting_type": "Plain",
                "score_function": "L2",
                "depth": 16,
                "min_data_in_leaf": 200,
                "max_leaves": 2**16 // 8,
            }
        params = {
            "iterations": 1000,
            "learning_rate": 0.01,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "task_type": "CPU",
            "use_best_model": True,
            "thread_count": -1,
            "silent": True,
            "random_seed": seed,
            "allow_writing_files": False,
            "auto_class_weights": "SqrtBalanced" if minor_class_counts < 0.05 else None,
            "bagging_temperature": 1,
            "max_bin": 255,
            "l2_leaf_reg": 10,
            "subsample": 0.9,
            "bootstrap_type": "MVS",
            "colsample_bylevel": 0.9,
        }
        params.update(sub_params)

    prediction_type = "Probability" if score_fn.__name__ == "roc_auc_score" else "Class"

    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")
        print("Estimating best number of trees.")

    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
            )

        best_iterations.append(model.get_best_iteration())

    best_iteration = int(np.median(best_iterations))  # int(np.mean(best_iterations))
    params["iterations"] = best_iteration

    cv.random_state = seed % 3
    if verbose:
        print(f"Evaluating cross validation with {best_iteration} trees.")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
            )

        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type=prediction_type)
        if prediction_type == "Probability":
            train_score = train_score[:, 1]
        train_score = score_fn(y_train, train_score)

        valid_scores = catboost.CatBoost.predict(model, valid_pool, prediction_type=prediction_type)
        if prediction_type == "Probability":
            valid_scores = valid_scores[:, 1]

        oof_preds[valid_idx] = valid_scores
        score = score_fn(y_valid, oof_preds[valid_idx])

        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        oof_scores = score_fn(y, oof_preds)
        print_scores(folds_scores, train_scores)
        print(f"OOF-score {score_fn.__name__}: {oof_scores:.5f}")
        if calculate_ci:
            bootstrap_scores = create_bootstrap_metrics(y, oof_preds, score_fn, n_samlpes=n_samples)
            left_bound, right_bound = calculate_confidence_interval(bootstrap_scores, conf_interval=confidence)
            print(f"Expected metric value lies between: {left_bound:.5f} and {right_bound:.5f}",
                  f"with confidence of {confidence*100}%")

    return estimators, oof_preds, np.mean(folds_scores)


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


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
    if test_exist:
        return df_train, df_test
    else:
        return df_train


def check_split_equality(x_train, x_test, estimator=None, params=None, categorical=None):
    x_inspect = pd.concat([x_train, x_test], axis=0)
    y_inspect = np.hstack((np.ones(x_train.shape[0]), np.zeros(x_test.shape[0])))

    if params is None:
        params = {'depth': 4,
                  'iterations': 100,
                  'silent': True,
                  'auto_class_weights': None,
                  'learning_rate': 0.1,
                  'random_seed': 42,
                  'cat_features': categorical,
        }
    if estimator is None:
        inspector = CatBoostClassifier(**params)
    else:
        inspector = estimator

    cv = cross_val_score(
                         estimator=inspector,
                         X=x_inspect,
                         y=y_inspect,
                         scoring="roc_auc",
                         cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )
    print(f"CV-score, mean: {round(np.mean(cv), 4)}, std: {round(np.std(cv), 4)}")
