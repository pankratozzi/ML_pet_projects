import gc
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
from scipy import sparse as sp
from scipy import stats
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from itertools import combinations, product
import time
import lightgbm
from lightgbm import LGBMClassifier
import gc
gc.enable()


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
        scores[feature] = 2 * score - 1

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
        score = Gini(target, scores_mean[0])
        print(f"{scores_mean[1]}{score:.5f}")


def lightgbm_cross_validation(params, X, y, cv, test=None, categorical=None, rounds=100, num_boost_round=10000,
                              verbose=0, preprocessor=None, func=np.mean):
    estimators, folds_scores, train_scores = [], [], []

    if not categorical:
        categorical = "auto"

    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros((X.shape[0], cv.get_n_splits()))

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        if isinstance(X, pd.DataFrame):
            x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        else:
            x_train, x_valid = X[train_idx, :], X[valid_idx, :]
            y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocessor is not None:
            x_train = preprocessor.fit_transform(x_train)
            x_valid = preprocessor.transform(x_valid)
            if test is not None:
                test = preprocessor.transform(test)

        dtrain = lightgbm.Dataset(x_train, y_train)
        dvalid = lightgbm.Dataset(x_valid, y_valid, reference=dtrain)

        model = lightgbm.train(params, dtrain,
                               num_boost_round=num_boost_round,
                               valid_sets=dvalid,
                               feval=gini_metric,
                               verbose_eval=verbose,
                               early_stopping_rounds=rounds)

        train_score = model.predict(x_train, num_iteration=model.best_iteration)
        train_score = Gini(y_train, train_score)
        oof_preds[valid_idx] = model.predict(x_valid, num_iteration=model.best_iteration)
        if test is not None:
            test_preds[:, fold] = model.predict(test, num_iteration=model.best_iteration)
        score = Gini(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))
        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {Gini(y, oof_preds):.5f}")

    if test is not None:
        test_preds = func(test_preds, axis=1)
        return estimators, oof_preds, test_preds
    else:
        return estimators, oof_preds


def lightgbm_cross_validation_mean(params, X, y, cv, categorical=None, rounds=100, verbose=0, num_boost_round=10000,
                                   preprocessor=None):
    estimators, folds_scores, train_scores = [], [], []

    if not categorical:
        categorical = "auto"

    oof_preds = np.zeros(X.shape[0])
    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        if isinstance(X, pd.DataFrame):
            x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        else:
            x_train, x_valid = X[train_idx, :], X[valid_idx, :]
            y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocessor is not None:
            x_train = preprocessor.fit_transform(x_train)
            x_valid = preprocessor.transform(x_valid)

        dtrain = lightgbm.Dataset(x_train, y_train)
        dvalid = lightgbm.Dataset(x_valid, y_valid, reference=dtrain)

        model = lightgbm.train(params, dtrain,
                               num_boost_round=num_boost_round,
                               valid_sets=dvalid,
                               feval=gini_metric,
                               verbose_eval=verbose,
                               early_stopping_rounds=rounds)

        oof_preds[valid_idx] = model.predict(x_valid, num_iteration=model.best_iteration)
        train_score = model.predict(x_train, num_iteration=model.best_iteration)
        train_score = Gini(y_train, train_score)
        score = Gini(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        print_scores(folds_scores, train_scores)
        print(f"OOF-score: {Gini(y, oof_preds):.5f}")
    return estimators, oof_preds, np.mean(folds_scores)


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


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


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def Gini(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    return G_pred / G_true


def gini_metric(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.high_cardinality_columns = []
        self.aggs = {
            "ps_car_13": ["mean", "max", "min", "std"],
            "ps_reg_03_sqrt": ["mean", "max", "min", "std"],
            "ps_car_12": ["mean", "max", "min", "std"],
            "ps_reg_02": ["mean", "max", "min"]
        }
        self.agg_03_cat = None
        self.agg_04_cat = None
        self.binary_columns = []
        self.cat_features = []
        self.ind_features = []
        self.le_encoders = []
        self.ohe_encoder = OneHotEncoder()
        self._is_fit = False

    def fit(self, X, y=None):

        for column in X.columns:
            n = X[column].nunique()
            if n > 20:
                self.high_cardinality_columns.append(column)

        for column in X.columns:
            if X[column].nunique() < 3:
                self.binary_columns.append(column)

        self.cat_features = [c for c in X.columns if "cat" in c]
        self.ind_features = [c for c in X.columns if "ind" in c]

        for col in self.cat_features:
            le = LabelEncoder()
            X[col] = le.fit(X[col])
            self.le_encoders.append(le)

        self.ohe_encoder.fit(X[self.cat_features])
        return self

    def transform(self, X):
        X = X.copy(deep=True)
        X["ps_reg_03_sqrt"] = np.nan
        X.loc[X["ps_reg_03"] != -1, "ps_reg_03_sqrt"] = X.loc[X["ps_reg_03"] != -1, 'ps_reg_03'].map(np.sqrt)

        X["ps_car_14_sqrt"] = np.nan
        X.loc[X["ps_car_14"] != -1, "ps_car_14_sqrt"] = X.loc[X["ps_car_14"] != -1, 'ps_car_14'].map(np.sqrt)

        X["bin_sum"] = X[self.binary_columns].sum(axis=1)
        X["bin_mean"] = X[self.binary_columns].mean(axis=1)
        X["bin_std"] = X[self.binary_columns].std(axis=1)

        X["nans"] = (X == -1).sum(axis=1).astype(float)
        X[self.high_cardinality_columns] = X[self.high_cardinality_columns].replace(-1, np.nan)

        if not self._is_fit:
            self.agg_03_cat = create_numerical_aggs(data=X, groupby_id="ps_car_03_cat", aggs=self.aggs,
                                                    suffix="_by_ps_car_03_cat")
            self.agg_04_cat = create_numerical_aggs(data=X, groupby_id="ps_car_04_cat", aggs=self.aggs,
                                                    suffix="_by_ps_car_04_cat")
            self._is_fit = True
        X = X.merge(self.agg_03_cat, on="ps_car_03_cat", how="left")
        X = X.merge(self.agg_04_cat, on="ps_car_04_cat", how="left")
        gc.collect()
        X["ind_total"] = X[self.ind_features].astype(str).sum(axis=1)

        X["ind_total"] = X["ind_total"].astype("category")

        for i, col in enumerate(self.cat_features):
            X[col] = self.le_encoders[i].transform(X[col])
        X_cat = self.ohe_encoder.transform(X[self.cat_features])
        train_features = [feat for feat in X.columns if feat not in self.cat_features + ["ind_total"]]

        X = sp.hstack([X[train_features].values, X_cat]).tocsr()
        del X_cat;
        gc.collect()
        return X
