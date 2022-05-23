from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from ..utils import split_train_test


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

    split_sets = split_k_sets(X, y, cv)
    error = []
    train_score = []
    validation_score = []
    for i in range(cv):
        train_x = np.array(split_sets[:i][0] + split_sets[i + 1:][0])
        train_y = np.array(split_sets[:i][1] + split_sets[i + 1:][1])
        validate_x = np.array(split_sets[i][0])
        validate_y = np.array(split_sets[i][1])
        estimator.fit(train_x, train_y)
        error.append(estimator.loss(validate_x, validate_y))
        train_score.append(1-scoring(train_x, train_y))
        validation_score.append(1-scoring(validate_x, validate_y))
    return np.average(np.array(train_score)), np.average(np.array(validation_score))


def split_k_sets(X, y, cv):
    """
    Randomly partition S to k (cv) disjoint subsets
    """
    rest_x, rest_y = deepcopy(X), deepcopy(y)
    split_sets = []
    for i in range(cv, 0, -1):
        percentage = i / rest_x.shape[0]
        s_x, s_y, rest_x, rest_y = split_train_test(X, y, percentage)
        split_sets.append((s_x, s_y))
    return split_sets

    # ...|...|...
    # ...|......
