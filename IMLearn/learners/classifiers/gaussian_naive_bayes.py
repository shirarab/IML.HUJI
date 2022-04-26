from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from .linear_discriminant_analysis import LDA


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples = X.shape[0]
        self.classes_, nk = np.unique(ar=y, return_counts=True)
        self.pi_ = nk / n_samples

        k_idx = {k: (y == k).nonzero()[0] for k in self.classes_}
        mu, var = [], []
        j = 0
        for k, idxs in k_idx.items():
            xi = X[idxs]
            mu_k = np.sum(xi, axis=0) / nk[j]  # same as np.sum(xi) / len(idxs)
            mu.append(mu_k)
            var_k = np.sum((xi - mu_k) ** 2, axis=0) / nk[j]  # todo check: (nk[j] - 1)
            var.append(var_k)
            j += 1
        self.mu_ = np.array(mu)
        self.vars_ = np.array(var)

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

        likeli = self.likelihood(X)  # P(x|y)
        return LDA.get_prediction_helper(X, likeli, self.pi_)  # array of argmax P(x|y)*P(y)/P(X)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = []
        for j, k in enumerate(self.classes_):
            mu_k = self.mu_[j]
            var_k = np.diag(self.vars_[j])
            likelihoods.append(np.array(LDA.gaussian_likelihood_k(X, mu_k, var_k)))
        return np.array(likelihoods).T

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
        from ...metrics import misclassification_error

        return misclassification_error(y, self._predict(X))
