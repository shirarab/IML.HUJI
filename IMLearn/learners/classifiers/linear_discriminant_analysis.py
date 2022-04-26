from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features' covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features' covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples, n_features = X.shape
        self.classes_, nk = np.unique(ar=y, return_counts=True)
        self.pi_ = nk / n_samples

        # k_idx is a dictionary s.t. for each class k the value is an array of all the
        # indexes that it appears at.
        k_idx = {k: (y == k).nonzero()[0] for k in self.classes_}
        mu = []
        sigma = np.zeros((n_features, n_features))
        j = 0
        for k, idxs in k_idx.items():
            xi = X[idxs]
            mu_k = np.sum(xi, axis=0) / nk[j]  # same as np.sum(xi) / len(idxs)
            mu.append(mu_k)
            sigma += ((xi - mu_k).T @ (xi - mu_k))
            j += 1
        self.mu_ = np.array(mu)
        self.cov_ = sigma / (n_samples - len(self.classes_))  # assuming unbiased estimator
        self._cov_inv = inv(self.cov_)

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

        # N(xi|muk,Σ) ------ not N(xi|muk,Σ) * Mult(yi|pik)
        likelihoods = []
        for j, k in enumerate(self.classes_):
            mu_k, pi_k = self.mu_[j], self.pi_[j]
            likelihoods.append(np.array(LDA.gaussian_likelihood_k(X, mu_k, self.cov_)))
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

    @staticmethod
    def gaussian_likelihood_k(X, mu_k, cov):
        n_features = X.shape[1]  # or: n_samples, n_features = X.shape
        denominator = np.sqrt(np.power(2 * np.pi, n_features) * det(cov))  # √((2π)^d*|Σ|)
        x_centered = X - mu_k.T
        k_likelihood = []
        for x in x_centered:
            exp = np.exp(-0.5 * x.T @ inv(cov) @ x)
            k_likelihood.append(exp / denominator)
        return k_likelihood

    @staticmethod
    def get_prediction_helper(X, likeli, pi):
        y_hat = []
        for i, xi in enumerate(X):
            p_xi = pi @ likeli[i]
            max_k = np.argmax(likeli[i] * pi / p_xi)  # argmax P(x|y)*P(y)/P(X)
            y_hat.append(max_k)
        return np.array(y_hat)
