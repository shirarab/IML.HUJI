from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
# import pandas as pd

from IMLearn import BaseEstimator


# from ..utils import split_train_test


# def xcross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
#                    scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
#     """
#     Evaluate metric by cross-validation for given estimator
#
#     Parameters
#     ----------
#     estimator: BaseEstimator
#         Initialized estimator to use for fitting the data
#
#     X: ndarray of shape (n_samples, n_features)
#        Input data to fit
#
#     y: ndarray of shape (n_samples, )
#        Responses of input data to fit to
#
#     scoring: Callable[[np.ndarray, np.ndarray, ...], float]
#         Callable to use for evaluating the performance of the cross-validated model.
#         When called, the scoring function receives the true- and predicted values for each sample
#         and potentially additional arguments. The function returns the score for given input.
#
#     cv: int
#         Specify the number of folds.
#
#     Returns
#     -------
#     train_score: float
#         Average train score over folds
#
#     validation_score: float
#         Average validation score over folds
#     """
#
#     # training set S = {X,y}
#     # learner A = estimator
#     # integer k = cv
#     # scoring function
#
#     split_sets = split_k_sets(X, y, cv)
#     error = []
#     train_score = []
#     validation_score = []
#     for i in range(cv):
#         split_x, split_y = get_split_x_y(split_sets, i, i+1)
#         train_x = np.concatenate(split_x)
#         train_y = np.concatenate(split_y)
#         validate_x = np.array(split_sets[i][0][0])
#         validate_y = np.array(split_sets[i][1])
#         estimator.fit(train_x, train_y)
#         error.append(estimator.loss(validate_x, validate_y))
#         train_score.append(scoring(train_y, estimator.predict(train_x)))
#         validation_score.append(scoring(validate_y, estimator.predict(validate_x)))
#     return np.average(np.array(train_score)), np.average(np.array(validation_score))


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    # training set S = {X,y}
    # learner A = estimator
    # integer k = cv
    # scoring function

    split_x = np.array_split(X, cv)
    split_y = np.array_split(y, cv)

    error = []
    train_score = []
    validation_score = []
    for i in range(cv):
        train_x = np.concatenate(np.array(split_x[:i] + split_x[i + 1:]))
        train_y = np.concatenate(np.array(split_y[:i] + split_y[i + 1:]))
        validate_x = np.array(split_x[i])
        validate_y = np.array(split_y[i])
        estimator.fit(train_x, train_y)
        error.append(estimator.loss(validate_x, validate_y))
        train_score.append(scoring(train_y, estimator.predict(train_x)))
        validation_score.append(scoring(validate_y, estimator.predict(validate_x)))
    return np.average(np.array(train_score)), np.average(np.array(validation_score))

# def get_split_x_y(split_sets, end, start=-1):
#     x, y=[],[]
#     for s in split_sets[:end]:
#         x.append(s[0][0].to_numpy())
#         y.append(s[1].to_numpy())
#     if start == -1:
#         return np.array(x), np.array(y)
#     for s in split_sets[start:]:
#         x.append(s[0][0].to_numpy())
#         y.append(s[1].to_numpy())
#     return np.array(x), np.array(y)
#
#
# def split_k_sets(X, y, cv):
#     """
#     Randomly partition S to k (cv) disjoint subsets
#     """
#     rest_x, rest_y = deepcopy(X), deepcopy(y)
#     split_sets = []
#     for i in range(cv, 0, -1):
#         percentage = i / rest_x.shape[0]
#         s_x, s_y, rest_x, rest_y = split_train_test(pd.DataFrame(rest_x), pd.Series(rest_y), percentage)
#         split_sets.append((s_x, s_y))
#     return split_sets
#
#     # ...|...|...
#     # ...|......
