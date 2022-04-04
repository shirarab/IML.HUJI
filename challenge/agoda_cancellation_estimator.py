from __future__ import annotations
from typing import NoReturn

import sklearn
from sklearn.linear_model import LinearRegression
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

N_NEIGHBORS = 5


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
        self.classifier = None
        self.regressor = None

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
        self.regressor = LinearRegression()
        boolean_cancel = np.where(y < 0, 0, 1)
        normalized_data = sklearn.preprocessing.normalize(X, axis=0)
        self.classifier.fit(normalized_data, boolean_cancel)
        sub_X = X[y > 0]
        sub_y = y[y > 0]
        self.regressor.fit(sub_X, sub_y)

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
        normalize_x = sklearn.preprocessing.normalize(X, axis=0)
        cancelling_prediction = self.classifier.predict(normalize_x)
        sub_X = X[cancelling_prediction == 1]
        pred_dates = self.regressor.predict(sub_X)
        i = 0
        good_y = []
        for j in range(X.shape[0]):
            if cancelling_prediction[j] == 1:
                good_y.append(pred_dates[i])
                i += 1
            else:
                good_y.append(0)
        return np.array(good_y)

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
        bool_pred = np.where(y_pred > 0, 1, 0)
        n_samples = y.shape[0]
        bool_y = np.where(y > 0, 1, 0)
        # loss is the amount of wrong classifications
        # compare = y_pred * y
        return np.sum(bool_y != bool_pred) / n_samples * 100
