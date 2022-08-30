import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy import stats
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from itertools import combinations
import time
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb


ID_COL = 'APPLICATION_NUMBER'
TARGET= 'TARGET'
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


def make_modify_cross_validation(X: pd.DataFrame,
                                 y: pd.Series,
                                 estimator: object,
                                 metric: callable,
                                 cv_strategy,
                                 error_to_be_outlier: None,
                                 verbose: bool = True,
                                 proba: bool = True,
                                 early: bool = False):
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
            # mask = ((y_valid - y_valid_pred) / y_valid) < error_to_be_outlier  # for regression task: 0.05
            y_valid_pred = pd.Series(data=y_valid_pred, index=y_valid.index, name='predictions')
            outliers = iso.predict(x_valid.select_dtypes(include="number"))
            mask = y_valid[outliers == 1].index
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
        # mask = ((y - oof_predictions) / y) < error_to_be_outlier  # for regression task
        oof_predictions = pd.Series(data=oof_predictions, index=y.index, name="oof_predictions")
        outliers = iso.predict(X.select_dtypes(include="number"))
        mask = y[outliers == 1].index
        oof_score = metric(y.loc[mask], oof_predictions.loc[mask])

    print(f"CV-results train: {round(np.mean(fold_train_scores), 4)} +/- {round(np.std(fold_train_scores), 3)}")
    print(f"CV-results valid: {round(np.mean(fold_valid_scores), 4)} +/- {round(np.std(fold_valid_scores), 3)}")
    print(f"OOF-score = {round(oof_score, 4)}")

    return estimators, oof_score, fold_train_scores, fold_valid_scores, oof_predictions


def reduce_memory_df(df):
    """Reduce memory usage by converting data to more appropriate dtypes"""
    start_mem = df.memory_usage().sum() / 1024 ** 2
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


def fill_nan_by_median_one(train_cleaned):
    for i in train_cleaned.columns:
        train_cleaned[i] = train_cleaned[i].fillna(train_cleaned[i].median())
    return train_cleaned


def prepare_history():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_applications_history = pd.read_csv('applications_history.csv')

    df_applications_history.drop('PREV_APPLICATION_NUMBER', axis=1, inplace=True)

    categorical_feats = df_applications_history.select_dtypes(include="object").columns.tolist()
    df_applications_history = pd.get_dummies(df_applications_history, columns=categorical_feats)

    df_applications_history = df_applications_history.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    df_applications_history = fill_nan_by_median_one(df_applications_history)

    apps = df_applications_history

    aggs = {column: ["max", "sum"] for column in df_applications_history.columns[1:]}

    df_applications_history = create_numerical_aggs(apps, groupby_id=ID_COL, aggs=aggs)

    df_applications_history = fill_nan_by_median_one(df_applications_history)
    """
    print('Original shapes:', df_train.shape, df_test.shape)
    applications_numbers = set(df_applications_history[ID_COL].values)
    train_numbers = set(df_train[ID_COL].values)
    test_numbers  = set(df_test[ID_COL].values)    
    train_in_profiles = list(train_numbers & applications_numbers)
    test_in_profiles = list(test_numbers & applications_numbers)      
    df_train = df_train[(df_train[ID_COL].isin(train_in_profiles))]
    df_test = df_test[(df_test[ID_COL].isin(test_in_profiles))]    
    """
    target = df_train['TARGET']
    df_train = df_train.drop(['TARGET'], axis=1)

    contract_dict = {'Cash': 1, 'Credit Card': 2}
    df_train['NAME_CONTRACT_TYPE'] = df_train['NAME_CONTRACT_TYPE'].map(contract_dict)
    df_test['NAME_CONTRACT_TYPE'] = df_test['NAME_CONTRACT_TYPE'].map(contract_dict)

    df_train = pd.merge(df_train, df_applications_history, how='left', on=ID_COL)
    df_test = pd.merge(df_test, df_applications_history, how='left', on=ID_COL)

    print('Final shapes:', df_train.shape, df_test.shape)

    return df_train, target, df_test


def bki_cr(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('bki.csv', nrows=num_rows)
    bureau, bureau_cat = get_encoded(bureau, nan_as_category)

    bureau.drop(['BUREAU_ID'], axis=1, inplace=True)

    num_aggregations = {
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean']
    }
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']

    bureau_agg = bureau.groupby(ID_COL).agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby(ID_COL).agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on=ID_COL)

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby(ID_COL).agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on=ID_COL)

    return bureau_agg


def get_encoded(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype in ['object', 'category']]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns


def prepare_cp_sec(data: pd.DataFrame,
                   dropna: bool = True,
                   create_features: bool = False,
                   is_category_encode: bool = False) -> pd.DataFrame:
    df = pd.read_csv('client_profile.csv')
    # fill missing values: 0 is reasonable
    # df.fillna(0, inplace=True)

    # deal with extreme outliers
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 50] = df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] <= 50, 'AMT_REQ_CREDIT_BUREAU_QRT'].max() + 1
    df.loc[df['DAYS_ON_LAST_JOB'] > 50000, 'DAYS_ON_LAST_JOB'] = 366
    # df['DAYS_ON_LAST_JOB'] = df['DAYS_ON_LAST_JOB'].replace(365243, np.nan)
    df.loc[df['TOTAL_SALARY'] > 1e+8, 'TOTAL_SALARY'] /= 1000.

    # IQR mark
    """
    df['IS_OUTLIER'] = 0
    for column in df.select_dtypes(include="number").columns:
        q1, q3 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        condition = ((df[column] > upper_bound) | (df[column] < lower_bound))
        df.loc[condition, 'IS_OUTLIER'] = 1
    """

    # transform categorical columns
    df.loc[df['GENDER'] == 'XNA', 'GENDER'] = df['GENDER'].mode()[0]
    df['GENDER'] = df['GENDER'].map({'F': 0, 'M': 1})
    df.loc[df['FAMILY_STATUS'] == 'Unknown', 'FAMILY_STATUS'] = 'Civil marriage'

    df['AGE'] = df['AGE'] / 365.25
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=np.linspace(18, 70, 10), labels=False) + 1

    if create_features:
        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMOUNT_CREDIT'] / (df['AMOUNT_ANNUITY'] + 1)
        df['NEW_STATUS'] = df['GENDER'].map({0: 'F', 1: 'M'}) + df['FAMILY_STATUS'].astype(str)
        df["RATIO_ANNUITY_TO_AGE"] = df["AMOUNT_ANNUITY"] / df["AGE"]
        df["RATIO_CREDIT_TO_AGE"] = df["AMOUNT_CREDIT"] / df["AGE"]
        df["RATIO_SALARY_TO_AGE"] = df["TOTAL_SALARY"] / df["AGE"]
        df["RATIO_AGE_TO_EXPERIENCE"] = df["AGE"] / (df["DAYS_ON_LAST_JOB"] / 365.25 + 1)
        df["RATIO_CAR_TO_EXPERIENCE"] = df["OWN_CAR_AGE"] / (df["DAYS_ON_LAST_JOB"] / 365.25 + 1)
        df["RATIO_CAR_TO_AGE"] = df["OWN_CAR_AGE"] / df["AGE"]
        aggs = {
            "TOTAL_SALARY": ["mean", "max", "min", "count"],
            "AMOUNT_CREDIT": ["mean", "max", "min", "count"],
            "AMOUNT_ANNUITY": ["mean", "max", "min", "count"]
        }

        stat = create_numerical_aggs(df, groupby_id="EDUCATION_LEVEL", aggs=aggs, suffix="_BY_EDUCATION")
        df = df.merge(stat, on='EDUCATION_LEVEL', how='left')
        df["TOTAL_SALARY_TO_MEAN_SALARY_BY_EDUCATION"] = df["TOTAL_SALARY"] / (df["TOTAL_SALARY_MEAN_BY_EDUCATION"] + 1)
        df["DELTA_SALARY_TO_MEAN_SALARY_BY_EDUCATION"] = df["TOTAL_SALARY"] - df["TOTAL_SALARY_MEAN_BY_EDUCATION"]
        df["RATIO_SALARY_TO_AMOUNT_CREDIT"] = df["AMOUNT_CREDIT"] / df["TOTAL_SALARY"]
        df["RATIO_SALARY_TO_AMOUNT_CREDIT_BY_FAMILY"] = df["TOTAL_SALARY"] / (
                    df["AMOUNT_CREDIT"] / (df["FAMILY_SIZE"] + 1))
        df["RATIO_AMOUNT_ANNUITY_TO_SALARY"] = df["AMOUNT_ANNUITY"] / (df["TOTAL_SALARY"] + 1)  # redundant
        df["RATIO_SALARY_TO_PER_FAMILY_SIZE"] = df["TOTAL_SALARY"] / (df["FAMILY_SIZE"] + 1)
        df["FLG_MORE_THAN_30PERCENT_FOR_CREDIT"] = np.where(df["RATIO_AMOUNT_ANNUITY_TO_SALARY"] > 0.3, 1, 0)
        df["EDUCATION_FAMILY_STATUS"] = df["EDUCATION_LEVEL"].apply(str) + " | " + df["FAMILY_STATUS"].apply(str)
        stat = create_numerical_aggs(df, groupby_id="AGE_BIN", aggs=aggs, suffix="_AGE_INTERVAL")
        df = df.merge(stat, on='AGE_BIN', how='left')
        stat = create_numerical_aggs(df, groupby_id="FAMILY_STATUS", aggs=aggs, suffix="_BY_FAMILY_STATUS")
        df = df.merge(stat, on='FAMILY_STATUS', how='left')

        aggs = {"NEW_CREDIT_TO_ANNUITY_RATIO": ["mean"],
                "RATIO_SALARY_TO_AMOUNT_CREDIT": ["mean"]}
        stat = create_numerical_aggs(df, groupby_id=["GENDER", "AGE_BIN"], aggs=aggs, suffix="_GENDER_AGE_BIN")
        df = df.merge(stat, on=["GENDER", "AGE_BIN"], how='left')

        df['EXTERNAL_SCORE_WEIGHTED'] = df['EXTERNAL_SCORING_RATING_1'] * 2 + df['EXTERNAL_SCORING_RATING_2'] + df[
            'EXTERNAL_SCORING_RATING_3'] * 3

        df['EXT_SCORE_1_AMT_CREDIT'] = df['EXTERNAL_SCORING_RATING_1'] * df['AMOUNT_CREDIT']
        df['EXT_SCORE_2_AMT_CREDIT'] = df['EXTERNAL_SCORING_RATING_2'] * df['AMOUNT_CREDIT']
        df['EXT_SCORE_3_AMT_CREDIT'] = df['EXTERNAL_SCORING_RATING_3'] * df['AMOUNT_CREDIT']

        df['EXT_SCORE_1_ANNUITY'] = df['EXTERNAL_SCORING_RATING_1'] * df['AMOUNT_ANNUITY']
        df['EXT_SCORE_2_ANNUITY'] = df['EXTERNAL_SCORING_RATING_2'] * df['AMOUNT_ANNUITY']
        df['EXT_SCORE_3_ANNUITY'] = df['EXTERNAL_SCORING_RATING_3'] * df['AMOUNT_ANNUITY']

        df['SALARY_REGION_POPULATION'] = df['TOTAL_SALARY'] * df['REGION_POPULATION']
        df['SALARY_JOB'] = df['TOTAL_SALARY'] / (df["DAYS_ON_LAST_JOB"] / 365.25 + 1)
        df['CREDIT_JOB'] = df['AMOUNT_CREDIT'] / (df["DAYS_ON_LAST_JOB"] / 365.25 + 1)
        df['ANNUITY_JOB'] = df['AMOUNT_ANNUITY'] / (df["DAYS_ON_LAST_JOB"] / 365.25 + 1)
        funcs = ["min", "max", "mean", "nanmedian", "var"]
        for func in funcs:
            df[f"EXT_SCORES_{func}"] = eval("np.{}".format(func))(df[['EXTERNAL_SCORING_RATING_1',
                                                                      'EXTERNAL_SCORING_RATING_2',
                                                                      'EXTERNAL_SCORING_RATING_3']], axis=1)

    if is_category_encode:
        df, _ = get_encoded(df)

    df = reduce_memory_df(df)
    df = data.merge(df, on=ID_COL, how='left')
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].map({'Cash': 0, 'Credit Card': 1})

    if dropna:
        df.dropna(subset=df.columns.tolist()[3:], how='all', inplace=True)
        # df.set_index(ID_COL, inplace=True)

    return df


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

        model = CatBoostClassifier(cat_features=categorical, **params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=rounds
        )
        train_score = model.predict_proba(x_train)[:, 1]
        train_score = roc_auc_score(y_train, train_score)
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    print_scores(folds_scores, train_scores)
    print(f"OOF-score: {roc_auc_score(y, oof_preds):.5f}")
    return estimators, oof_preds


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


def xgboost_cross_validation(params, X, y, cv, categorical=None, rounds=50, verbose=True, return_valid_mean=False):
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

    return estimators, encoders, oof_preds, np.mean(folds_scores) if return_valid_mean else estimators, encoders, oof_preds


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
