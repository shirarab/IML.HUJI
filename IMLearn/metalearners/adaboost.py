import numpy as np
# from ...base import BaseEstimator
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples, n_features = X.shape
        self.D_ = np.ones(n_samples) / n_samples  # set initial distribution to be uniform
        self.models_, self.weights_ = [], []
        for t in range(self.iterations_):
            wl = self.wl_().fit(X, y * self.D_)  # Invoke base learner
            # wl = self.wl_().fit(X, y)  # Invoke base learner
            self.models_.append(wl)
            y_pred = wl.predict(X)
            # e_t = np.sum(self.D_ @ (y_pred != y).astype(int))
            e_t = np.sum(self.D_[y_pred != y])
            w_t = 0.5 * np.log((1 - e_t) / e_t)
            self.weights_.append(w_t)
            self.D_ *= np.exp(-y * w_t * y_pred)  # Update sample weights
            self.D_ /= np.sum(self.D_)  # Normalize weights
        self.models_ = np.array(self.models_)
        self.weights_ = np.array(self.weights_)

    def _predict(self, X):
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

        return self.partial_predict(X, self.iterations_)

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

        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        # sum_wh = 1 * np.ones(X.shape[0])
        sum_wh = np.sum(self.weights_[t] * self.models_[t].predict(X) for t in range(T))
        return np.sign(sum_wh) # * self.D_
        # sum_wh = 0
        # for t in range(T):
        #     sum_wh += self.weights_[t] * self.models_[t].predict(X)
        # return np.sign(sum_wh)  # * self.D_

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)
        # loss = 0
        # for t in range(T):
        #     loss += self.models_[t]._loss(X, y)
        # return loss
