from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


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

    train_score = []
    validation_score = []
    for i in range(cv):
        train_x = np.concatenate(np.array(split_x[:i] + split_x[i + 1:]))
        train_y = np.concatenate(np.array(split_y[:i] + split_y[i + 1:]))
        validate_x = np.array(split_x[i])
        validate_y = np.array(split_y[i])
        estimator.fit(train_x, train_y)
        train_score.append(scoring(train_y, estimator.predict(train_x)))
        validation_score.append(scoring(validate_y, estimator.predict(validate_x)))
    return np.average(np.array(train_score)), np.average(np.array(validation_score))
