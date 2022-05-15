from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        min_loss = np.inf
        self.threshold_, self.j_, self.sign_ = 0, 0, 1
        sign_feature_iter = product([-1, 1], range(X.shape[1]))
        for sign, j_feature in sign_feature_iter:
            thr, new_loss = self._find_threshold(X[:, j_feature], y, sign)
            if new_loss < min_loss:
                min_loss = new_loss
                self.threshold_, self.j_, self.sign_ = thr, j_feature, sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        y_hat = np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)
        return y_hat

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        n = values.shape[0]
        indices = np.argsort(values)
        x, y_true = values[indices], labels[indices]

        min_loss = np.inf
        threshold = 0

        new_indices = _get_indices_for_threshold(n, y_true, sign)

        for i in new_indices:  # 32.5344779 time instead of 219.1022538 time
            y_pred = np.concatenate((-sign * np.ones(i), sign * np.ones(n - i)))
            new_loss = np.sum(np.where(np.sign(y_true) != y_pred, np.abs(y_true), 0))
            if new_loss < min_loss:
                min_loss = new_loss
                threshold = x[i]
        return threshold, min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        y_pred = np.sign(self._predict(X))
        return np.sum(np.where(np.sign(y) != y_pred, np.abs(y), 0))


def _get_indices_for_threshold(n, y_true, sign):
    new_indices = []
    good_prev = False
    for i in range(n):
        if np.sign(y_true[i]) == sign and not good_prev:
            new_indices.append(i)
            good_prev = True
        if np.sign(y_true[i]) == -sign:
            good_prev = False
    return np.array(new_indices)
