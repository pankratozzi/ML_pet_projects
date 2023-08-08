import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Optional
import catboost
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns


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


def compute_threshold(y_true, y_score):
    from sklearn.metrics import precision_recall_curve

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.copy(deep=True).values
    if isinstance(y_score, pd.DataFrame):
        y_score = y_score.copy(deep=True).values

    estimated_thresholds = []
    for i in range(y_score.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_score[:, i])
        f_score = 2 * (precision * recall) / (precision + recall)
        max_t = np.argmax(f_score)
        estimated_thresholds.append(thresholds[max_t])

    return estimated_thresholds


def check_split_equality(x_train, x_test, estimator=None, params=None, categorical=None, textual=None):
    from sklearn.model_selection import cross_val_score
    if isinstance(x_train, np.ndarray):
        x_inspect = np.concatenate([x_train, x_test], axis=0)
    else:
        x_inspect = pd.concat([x_train, x_test], axis=0)
    y_inspect = np.hstack((np.ones(x_train.shape[0]), np.zeros(x_test.shape[0])))

    if params is None:
        params = {'depth': 4,
                  'iterations': 100,
                  'silent': True,
                  'auto_class_weights': None,
                  'learning_rate': 0.1,
                  'random_seed': 42,
                  'cat_features': categorical,  # provide cat indices if numpy array else cat names
                  'text_features': textual
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
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    return round(np.mean(cv), 4), round(np.std(cv), 4)


def _predict(estimator, x_valid, probas=True):
    if hasattr(estimator, "predict_proba") and probas:
        y_pred = estimator.predict_proba(x_valid)[:, 1]
    else:
        y_pred = estimator.predict(x_valid)

    return y_pred


def plot_confusion_matrix(y_true,
                          y_score,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          thres=0.5,
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    from itertools import product
    import matplotlib.pyplot as plt

    cmap = plt.cm.Blues if cmap is None else cmap

    cm = confusion_matrix(y_true, y_score >= thres)

    plt.figure(figsize=(5, 5))
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.grid("off")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid()
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

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def create_bootstrap_samples(data: np.array, n_samples: int = 1000) -> np.array:
    bootstrap_idx = np.random.randint(
        low=0, high=len(data), size=(n_samples, len(data))
    )
    return bootstrap_idx


def create_bootstrap_metrics(y_true: np.array,
                             y_pred: np.array,
                             metric: callable,
                             n_samlpes: int = 1000) -> list:
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


def calculate_confidence_interval(scores: list, conf_interval: float = 0.95) -> tuple:
    left_bound = np.percentile(
        scores, ((1 - conf_interval) / 2) * 100
    )
    right_bound = np.percentile(
        scores, (conf_interval + ((1 - conf_interval) / 2)) * 100
    )

    return left_bound, right_bound


def numeric_iv(df, target, nunq=10, eps=0.0001):
    iv_list = []
    df = df.copy(deep=True)

    target = df[target]
    df = df.loc[:, df.apply(pd.Series.nunique) > nunq]
    numerical_cols = df.select_dtypes(include=["number"]).columns

    for var_name in numerical_cols:
        df[var_name] = pd.qcut(df[var_name].values, nunq, duplicates="drop").codes

        freqs = pd.crosstab(df[var_name], target)
        freqs1 = freqs[1] / np.sum(freqs[1]) + eps
        freqs0 = freqs[0] / np.sum(freqs[0]) + eps

        iv = np.sum((freqs1 - freqs0) * np.log(freqs1 / freqs0))
        iv_list.append(iv)

    result = pd.DataFrame({"feature name": numerical_cols, "IV": iv_list})
    result["importance"] = ["Susp. high" if x > 0.5 else "high"
    if x <= 0.5 and x > 0.3 else "middle"
    if x <= 0.3 and x > 0.1 else "weak"
    if x <= 0.1 and x > 0.02 else "useless"
                            for x in result["IV"]]

    return result.sort_values("IV", ascending=False)


def check_woe_quality(df, feature, target, eps=0.0001):
    from sklearn.linear_model import LogisticRegression

    df = df[[feature, target]].copy(deep=True)
    df[feature] = df[feature].astype("str")

    global_mean = df[target].mean()
    freqs = pd.crosstab(df[feature], df[target])
    woe_encoding = np.log((freqs[1] / np.sum(freqs[1]) + eps) / (freqs[0] / np.sum(freqs[0]) + eps))

    df["woe"] = df[feature].map(woe_encoding).fillna(global_mean)
    logreg = LogisticRegression(solver="liblinear").fit(df[["woe"]], df[target])

    intercept = np.round(logreg.intercept_.item(), 3)
    coef = np.round(logreg.coef_.item(), 3)

    percents = df[target].value_counts(normalize=True)
    percents = np.log(percents[1] / percents[0])

    print(f"Bias {intercept} (has to be equal log ration of events to non-events: {percents:.3f}),"
          f"Coef: {coef} (has to be equal 1)")

    iv = np.sum(((freqs[1] / np.sum(freqs[1]) + eps) - (freqs[0] / np.sum(freqs[0]) + eps)) * woe_encoding)
    print(f"Information value: {iv:.6f}")


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


def show_proba_calibration_plots(y_predicted_probs, y_true_labels, average="weighted"):
    preds_with_true_labels = np.array(list(zip(y_predicted_probs, y_true_labels)))

    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in np.linspace(0.1, 0.9, 9):
        thresholds.append(threshold)
        precisions.append(
            precision_score(y_true_labels, list(map(int, y_predicted_probs > threshold)), average=average))
        recalls.append(recall_score(y_true_labels, list(map(int, y_predicted_probs > threshold)), average=average))
        f1_scores.append(f1_score(y_true_labels, list(map(int, y_predicted_probs > threshold)), average=average))

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
              label='Pay class', color='royalblue', alpha=1)
    plt2.hist(preds_with_true_labels[preds_with_true_labels[:, 1] == 1][:, 0],
              label='Default class', color='darkcyan', alpha=0.8)
    plt2.set_ylabel('Number of examples')
    plt2.set_xlabel('Probabilities')
    plt2.set_title('Probability histogram')
    plt2.legend(bbox_to_anchor=(1, 1))

    plt.show()


def plot_precision_recall_curve(y_true, y_score, model_name="", color=None):
    from sklearn.metrics import precision_recall_curve, auc

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, label='%s: Precision-Recall curve (area = %0.2f)' %
                                      (model_name, auc(recall, precision)), color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("%s: Precision-Recall curve" % model_name)
    plt.axis([0.0, 1.0, 0.0, 1.05])
    plt.legend(loc="lower left")
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
    if test_exist:
        return df_train, df_test
    else:
        return df_train


class BestSet(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, k_features=12, scoring=None, test_size=0.3, seed=42):
        self.scoring = scoring if scoring is not None else f1_score
        self.k_features = k_features
        self.test_size = test_size
        self.estimator = clone(estimator)
        self.fit_params = {}
        self.seed = seed

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X, y = X.values, y.values
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.test_size,
                                                            stratify=y,
                                                            random_state=self.seed)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores, subsets = [], []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        best_indices = self.subsets_[np.argmax(self.scores_)]
        return X[:, best_indices]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train, **self.fit_params)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


def print_scores(folds_scores, train_scores):
    print(f"Train score by each fold: {train_scores}")
    print(f"Valid score by each fold: {folds_scores}")
    print(f"Train mean score by each fold:{np.mean(train_scores):.5f} +/- {np.std(train_scores):.5f}")
    print(f"Valid mean score by each fold:{np.mean(folds_scores):.5f} +/- {np.std(folds_scores):.5f}")
    print("*" * 50)


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
                              best_iter: str = "median",
                              not_best_model: bool = False,
                              is_raw: bool = False,
                              threshold: float = 0.5,
                              check_equality: bool = False,
                              seed: int = 42):
    import functools
    if isinstance(score_fn, functools.partial):
        score_fn_name = score_fn.func.__name__
    else:
        score_fn_name = score_fn.__name__

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
                "sampling_frequency": "PerTreeLevel",
                # Frequency to sample weights and objects when building trees, that are used to select objects
                # (Bayesian - strengthen random, Bernoulli - random, MVS - based on gradients size)
                # Bernoulli, MVS - random or gradient based subsample, Bayesian: uses all objects with bayesian
                # bootstrap instead of subsample
            }
        else:
            sub_params = {
                "grow_policy": "Lossguide",
                "boosting_type": "Plain",
                "score_function": "L2",
                "depth": 16,
                "min_data_in_leaf": 200,
                "max_leaves": 2 ** 16 // 8,
                "sampling_frequency": "PerTree",
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
            "random_strength": 1,
        }
        params.update(sub_params)

    if is_raw:
        prediction_type = "RawFormulaVal"
    else:
        prediction_type = "Probability" if score_fn_name == "roc_auc_score" else "Class"

    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")
        print("Estimating best number of trees.")

    best_iterations, equality = [], []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        if check_equality:
            mean, std = check_split_equality(x_train, x_valid, categorical=categorical, textual=textual)
            equality.append((mean, std))

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )

        best_iterations.append(model.get_best_iteration())

    if best_iter == "median":
        best_iteration = int(np.median(best_iterations))
    elif best_iter == "mean":
        best_iteration = int(np.mean(best_iterations))
    else:
        raise NotImplementedError("Set best_iter median or mean")

    params["iterations"] = best_iteration
    if not_best_model:
        params["use_best_model"] = False
        # as we estimated best cv number of trees and want our valid set to be fully independent of training process

    cv.random_state = seed % 3
    if verbose:
        print(f"Evaluating cross validation with {best_iteration} trees.")
        if check_equality:
            means, stds = list(zip(*equality))
            print("Split check on number of tree estimation: ", np.round(np.mean(means), 4), " +/- ",
                  np.round(np.max(stds), 4))

    equality = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        if check_equality:
            mean, std = check_split_equality(x_train, x_valid, categorical=categorical, textual=textual)
            equality.append((mean, std))

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
        )

        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type=prediction_type)
        if is_raw:
            exp = np.exp(train_score)
            train_score = exp / (1 + exp)
            if score_fn_name != "roc_auc_score":
                train_score = (train_score >= threshold).astype(np.uint8)

        if prediction_type == "Probability":
            train_score = train_score[:, 1]
        train_score = score_fn(y_train, train_score)

        valid_scores = catboost.CatBoost.predict(model, valid_pool, prediction_type=prediction_type)
        if prediction_type == "Probability":
            valid_scores = valid_scores[:, 1]

        if is_raw:
            exp = np.exp(valid_scores)
            valid_scores = exp / (1 + exp)
            if score_fn_name != "roc_auc_score":
                valid_scores = (valid_scores >= threshold).astype(np.uint8)

        oof_preds[valid_idx] = valid_scores
        score = score_fn(y_valid, oof_preds[valid_idx])

        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        if check_equality:
            means, stds = list(zip(*equality))
            print("Split check while cross-validating: ", np.round(np.mean(means), 4), " +/- ",
                  np.round(np.max(stds), 4))
        oof_scores = score_fn(y, oof_preds)
        print_scores(folds_scores, train_scores)
        print(f"OOF-score {score_fn_name}: {oof_scores:.5f}")
        if calculate_ci:
            bootstrap_scores = create_bootstrap_metrics(y, oof_preds, score_fn, n_samlpes=n_samples)
            left_bound, right_bound = calculate_confidence_interval(bootstrap_scores, conf_interval=confidence)
            print(f"Expected metric value lies between: {left_bound:.5f} and {right_bound:.5f}",
                  f"with confidence of {confidence * 100}%")

    return estimators, oof_preds, np.mean(folds_scores)


def thresholded_catboost_cross_validation(X: pd.DataFrame,
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
                                          best_iter: str = "median",
                                          not_best_model: bool = False,
                                          is_raw: bool = False,
                                          threshold: float = 0.5,
                                          check_equality: bool = False,
                                          threshold_range: int = 11,
                                          maximize: bool = True,
                                          seed: int = 42):

    def get_best_threshold(y_true, scores, thresholds, maximize=True):
        metrics = []
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(np.uint8)
            metric = score_fn(y_true, y_pred)
            metrics.append(metric)

        idx = np.argmax(metrics) if maximize else np.argmin(metrics)
        return thresholds[idx]

    import functools
    if isinstance(score_fn, functools.partial):
        score_fn_name = score_fn.func.__name__
    else:
        score_fn_name = score_fn.__name__

    minor_class_counts = y.value_counts(normalize=True).values[-1]

    if cv is None:
        if minor_class_counts >= 0.05:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if threshold_range > 2:
        thresholds = np.linspace(0, 1, threshold_range)
        thresholds = np.clip(thresholds, a_min=0.01, a_max=0.99)

    if params is None:
        if len(X) <= 50_000:
            sub_params = {
                "grow_policy": "SymmetricTree",
                "boosting_type": "Ordered",
                "score_function": "Cosine",
                "depth": 6,
                "sampling_frequency": "PerTreeLevel",
                # Frequency to sample weights and objects when building trees, that are used to select objects
                # (Bayesian - strengthen random, Bernoulli - random, MVS - based on gradients size)
                # Bernoulli, MVS - random or gradient based subsample, Bayesian: uses all objects with bayesian
                # bootstrap instead of subsample
            }
        else:
            sub_params = {
                "grow_policy": "Lossguide",
                "boosting_type": "Plain",
                "score_function": "L2",
                "depth": 16,
                "min_data_in_leaf": 200,
                "max_leaves": 2 ** 16 // 8,
                "sampling_frequency": "PerTree",
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
            "random_strength": 1,
        }
        params.update(sub_params)

    if is_raw:
        prediction_type = "RawFormulaVal"
    else:
        prediction_type = "Probability"

    estimators, folds_scores, train_scores = [], [], []

    oof_preds = np.zeros(X.shape[0])
    oof_probs = np.zeros(X.shape[0])
    best_thresholds = []

    if verbose:
        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")
        print("Estimating best number of trees.")

    best_iterations, equality = [], []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        if check_equality:
            mean, std = check_split_equality(x_train, x_valid, categorical=categorical, textual=textual)
            equality.append((mean, std))

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=rounds
        )

        best_iterations.append(model.get_best_iteration())

    if best_iter == "median":
        best_iteration = int(np.median(best_iterations))
    elif best_iter == "mean":
        best_iteration = int(np.mean(best_iterations))
    else:
        raise NotImplementedError("Set best_iter median or mean")

    params["iterations"] = best_iteration
    if not_best_model:
        params["use_best_model"] = False
        # as we estimated best cv number of trees and want our valid set to be fully independent of training process

    cv.random_state = seed % 3
    if verbose:
        print(f"Evaluating cross validation with {best_iteration} trees.")
        if check_equality:
            means, stds = list(zip(*equality))
            print("Split check on number of tree estimation: ", np.round(np.mean(means), 4), " +/- ",
                  np.round(np.max(stds), 4))

    equality = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):

        x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if preprocess is not None:
            x_train = preprocess.fit_transform(x_train, y_train)
            x_valid = preprocess.transform(x_valid)

        if check_equality:
            mean, std = check_split_equality(x_train, x_valid, categorical=categorical, textual=textual)
            equality.append((mean, std))

        train_pool = Pool(x_train, y_train, cat_features=categorical, text_features=textual)
        valid_pool = Pool(x_valid, y_valid, cat_features=categorical, text_features=textual)

        model = CatBoostClassifier(**params).fit(
            train_pool,
            eval_set=valid_pool,
        )

        train_score = catboost.CatBoost.predict(model, train_pool, prediction_type=prediction_type)
        if is_raw:
            exp = np.exp(train_score)
            train_score = exp / (1 + exp)

        if prediction_type == "Probability":
            train_score = train_score[:, 1]

        if score_fn_name != "roc_auc_score":
            train_score = (train_score >= threshold).astype(np.uint8)
        train_score = score_fn(y_train, train_score)

        valid_scores = catboost.CatBoost.predict(model, valid_pool, prediction_type=prediction_type)
        if prediction_type == "Probability":
            valid_scores = valid_scores[:, 1]

        if is_raw:
            exp = np.exp(valid_scores)
            valid_scores = exp / (1 + exp)

        if threshold_range > 2 and score_fn_name != "roc_auc_score":
            best_threshold = get_best_threshold(y_valid, valid_scores, thresholds, maximize=maximize)
            best_thresholds.append(best_threshold)
            oof_preds_part = (valid_scores >= best_threshold).astype(np.uint8)
            oof_probs[valid_idx] = oof_preds_part

        if score_fn_name != "roc_auc_score":
            valid_scores = (valid_scores >= threshold).astype(np.uint8)

        oof_preds[valid_idx] = valid_scores
        score = score_fn(y_valid, oof_preds[valid_idx])

        folds_scores.append(round(score, 5))
        train_scores.append(round(train_score, 5))

        if verbose:
            print(f"Fold {fold + 1}, Train score = {train_score:.5f}, Valid score = {score:.5f}")
        estimators.append(model)

    if verbose:
        if check_equality:
            means, stds = list(zip(*equality))
            print("Split check while cross-validating: ", np.round(np.mean(means), 4), " +/- ",
                  np.round(np.max(stds), 4))
        oof_scores = score_fn(y, oof_preds)
        print_scores(folds_scores, train_scores)
        print(f"OOF-score {score_fn_name}: {oof_scores:.5f}")
        if threshold_range > 2 and score_fn_name != "roc_auc_score":
            oof_scores_th = score_fn(y, oof_probs)
            best_thresholds = [round(t, 2) for t in best_thresholds]
            print(f"Thresholds: {best_thresholds}, OOF-score with thresholds: {oof_scores_th:.5f}")
        if calculate_ci:
            bootstrap_scores = create_bootstrap_metrics(y, oof_preds, score_fn, n_samlpes=n_samples)
            left_bound, right_bound = calculate_confidence_interval(bootstrap_scores, conf_interval=confidence)
            print(f"Expected metric value lies between: {left_bound:.5f} and {right_bound:.5f}",
                  f"with confidence of {confidence * 100}%")

    return estimators, oof_preds, np.mean(folds_scores)


def create_multiple_bootstrap_metrics(y_true: np.array,
                                      y_pred: list,
                                      metric: callable,
                                      n_samples: int = 1000) -> list:
    scores = np.zeros((len(y_pred), n_samples))

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    bootstrap_idx = create_bootstrap_samples(y_true, n_samples)
    for j, idx in enumerate(bootstrap_idx):
        running_scores = np.zeros((len(y_pred)))
        for i, pred in enumerate(y_pred):
            running_scores[i] = metric(y_true[idx], pred[idx])
        scores[:, j] = running_scores

    return scores


def compare_models(oof_preds: list,
                   y_true: np.ndarray,
                   metric: callable,
                   n_train: int,
                   n_test: int,
                   model_names=None,
                   sample: int = 0,
                   conf_level: float = 0.95,
                   rope_interval: list = [-0.01, 0.01],
                   n_samples: int = 3000,
                   verbose: bool = False,
                   correct_bias: bool = False) -> pd.DataFrame:
    from scipy.stats import t, ttest_rel, shapiro
    from itertools import combinations
    from math import factorial

    if model_names is None:
        model_names = [f"model_{i + 1}" for i in range(len(oof_preds))]

    n_comparisons = (factorial(len(oof_preds)) / (factorial(2) * factorial(len(oof_preds) - 2)))
    scores = create_multiple_bootstrap_metrics(y_true, oof_preds, metric, n_samples)
    if sample != 0:
        scores = [np.random.choice(score, size=sample, replace=False) for score in scores]

    df = scores[0].shape[0] - 1

    for i, prediction in enumerate(scores, 1):
        if verbose:
            tmp = pd.Series(data=prediction)
            skew, kurt = tmp.skew(), tmp.kurtosis()  # usually == 3.0
            print(f"Skewness for model {i}: {skew:.4f} (ideal = 0), kurtosis: {kurt:.4f} (kurtosis of normal == 0.0 )")
        p_val = shapiro(prediction)[1]
        if p_val < 0.05:
            print(f"Samples from model {i} are not normally distributed, p-value: {p_val:6f}")
            if correct_bias:
                print(f"Correcting bias for model {i}")
                true_metric = metric(y_true, oof_preds[i-1])
                boot_mean = np.mean(prediction)
                delta_val = np.abs(boot_mean - true_metric)
                scores[i-1] = prediction + delta_val

    def corrected_std(differences, n_train, n_test):
        kr = len(differences)
        corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
        corrected_std = np.sqrt(corrected_var)
        return corrected_std

    def compute_corrected_ttest(differences, df, n_train, n_test):
        mean = np.mean(differences)
        std = corrected_std(differences, n_train, n_test)
        t_stat = mean / std
        p_val = t.sf(np.abs(t_stat), df)
        return t_stat, p_val

    pairwise_ttest, pairwise_bayesian, t_rel = [], [], []

    for model_i, model_k in combinations(range(len(scores)), r=2):
        model_i_scores = scores[model_i]
        model_k_scores = scores[model_k]

        differences = model_i_scores - model_k_scores
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # bonferroni correction
        p_val = 1 if p_val > 1 else p_val

        pairwise_ttest.append([model_names[model_i], model_names[model_k], t_stat, p_val])

        t_post = t(df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test))
        model_i_prob = t_post.cdf(rope_interval[0])  # probability of model 1 worse than 2
        model_k_prob = 1 - t_post.cdf(rope_interval[1])  # probability of model 2 worse than 1

        rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])  # probability of model 1 is equal 2

        cred_interval = list(t_post.interval(conf_level))  # true difference lies inside interval
        pairwise_bayesian.append([model_i_prob, model_k_prob, rope_prob, cred_interval[0], cred_interval[1]])

        tres = ttest_rel(model_i_scores, model_k_scores)
        statistic, p_val = tres[0], tres[1]
        # ci = tres.confidence_interval(confidence_level=0.95)

        t_rel.append([statistic, p_val])

    result = pd.DataFrame(data=np.hstack([pairwise_ttest, pairwise_bayesian, t_rel]),
                          index=[f"compare_{i + 1}" for i in range(int(n_comparisons))],
                          columns=["model_1", "model_2", "t_stat", "corr_p_val", "1_worse_2",
                                   "2_worse_1", "2_equal_1", "ci_lower", "ci_upper", "rel_stat", "non_corr_p_val"])
    result.iloc[:, 2:] = result.iloc[:, 2:].astype("float").apply(lambda x: np.round(x, 4))

    return result


def calculate_feature_separating_ability(
        features: pd.DataFrame, target: pd.Series, fill_value: float = -9999) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    scores = {}
    for feature in features:
        score = roc_auc_score(
            target, features[feature].fillna(fill_value)
        )
        scores[feature] = 2 * score - 1

    scores = pd.Series(scores)
    scores = scores.sort_values(ascending=False)

    return scores


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
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


def calculate_lags_and_stats(df, target, lags_range=None, moving_stats_range=None, periods=1, min_periods=1,
                             aggfunc="mean", seasonality=1):
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