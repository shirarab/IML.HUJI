from __future__ import annotations
from typing import NoReturn

import sklearn

from IMLearn.base import BaseEstimator
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

N_NEIGHBORS = 10


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self._coef = None
        self._clf = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        self.classifier.fit(X, y)

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
        """
        return self.classifier.predict(X)
        # return self._clf.predict(X)
        # return np.sign(X @ self._coef)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        y_pred = self._predict(X)
        n_samples = y.shape[0]
        # loss is the amount of wrong classifications
        # compare = y_pred * y
        return np.sum(y != y_pred) / n_samples * 100
