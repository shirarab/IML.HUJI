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

        y_hat = []
        for xi in X:
            y_hat.append(self.__argmax_k(xi))
        return np.array(y_hat)

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

        # N(xi|muk,Σ) * Mult(yi|pi)
        likelihoods = []
        for j, k in enumerate(self.classes_):
            mu_k = self.mu_[j]
            likelihoods.append(LDA.likelihood_k(X, mu_k, self.cov_))
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

    # todo check if can add helper function. if not- the put it as mekunan function of predict.
    def __argmax_k(self, xi):
        max_k = None  # self.classes_[0]
        max_prob = -np.inf
        for i, k in enumerate(self.classes_):
            mu_k, pi_k = self.mu_[i], self.pi_[i]
            ak = self._cov_inv @ mu_k
            bk = np.log(pi_k) - 0.5 * mu_k @ self._cov_inv @ mu_k
            prob = ak.T @ xi + bk  # :=distribution of k
            if prob > max_prob:
                max_prob = prob
                max_k = k
        return max_k

    @staticmethod
    def likelihood_k(X, mu_k, cov):
        n_features = X.shape[1]  # or: n_samples, n_features = X.shape
        denominator = np.sqrt(np.power(2 * np.pi, n_features) * det(cov))  # √((2π)^d*|Σ|)
        x_centered = X - mu_k.T
        k_likelihood = []
        for x in x_centered:
            exp = np.exp(-0.5 * x.T @ inv(cov) @ x)
            k_likelihood.append(exp / denominator)  # todo change: exp * muk / denominator????
        return k_likelihood
