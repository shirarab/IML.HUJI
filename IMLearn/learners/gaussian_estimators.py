from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        # num_samples = X.size  # todo maybe X.shape[0]??
        self.mu_ = np.mean(X)  # X.sum() / num_samples  # (Σx_i)/m
        # x_minus_mu_pow = np.power(X - self.mu_, 2)  # Σ(x_i-μ)^2
        # np.var(X, ddof=1)
        if not self.biased_:
            # todo what happens if num_samples = 1?
            self.var_ = np.var(X, ddof=1)  # x_minus_mu_pow.sum() / (num_samples - 1)  # (Σ(x_i-μ)^2)/(m-1)
        else:
            # todo what happens if num_samples = 0?
            self.var_ = np.var(X, ddof=0)  # x_minus_mu_pow.sum() / num_samples  # (Σ(x_i-μ)^2)/m
        # todo can self.var_ be 0? it can be problematic in pdf func
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        denominator = np.sqrt(2 * np.pi * self.var_)  # √(2πσ^2)
        exp = np.exp(-((X - self.mu_) ** 2) / (2 * self.var_))  # exp(-(x−μ)^2/2σ^2)
        pdfs = exp / denominator  # exp(-(x−μ)^2/2σ^2)/√(2πσ^2)
        return pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # todo can sigma be 0? sigma might be the var (like in the documentation) - check
        num_samples = X.size
        log = np.log(2 * np.pi * sigma ** 2) * num_samples / 2
        inside_exp = np.sum(((X - mu) ** 2) / (2 * sigma ** 2))
        log_likelihood = -log - inside_exp
        return log_likelihood


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()

        # num_samples = X.shape[0]
        self.mu_ = np.mean(X, axis=0)  # (... (Σx_i)/m ...)^T
        # empirical_mean = np.tile(self.mu_, num_samples).reshape(num_samples, self.mu_.size)
        self.cov_ = np.cov(X.T, bias=False)
        # X_centered = X - self.mu_
        # numerator = X_centered.T @ X_centered
        # self.cov_ = numerator / (num_samples - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()

        # todo what happens if num_samples = 0? and if det == 0?
        denominator = np.sqrt(np.power(2 * np.pi, self.mu_.size) * det(self.cov_))  # √((2π)^d*|Σ|)
        X_minus_mu = X - self.mu_
        pdfs = []
        for i in range(X.shape[0]):
            exp = np.exp(-0.5 * X_minus_mu[i, :] @ inv(self.cov_) @ X_minus_mu[i, :])
            pdfs.append(exp / denominator)
        return np.array(pdfs)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # raise NotImplementedError()

        m = X.shape[0]
        d = mu.size
        X_minus_mu = X - mu.T
        first = np.log(2 * np.pi) * m * d / 2
        sign, logdet = slogdet(cov)
        second = sign * logdet * m / 2
        thirds = []
        for i in range(X.shape[0]):
            thirds.append(-0.5 * X_minus_mu[i, :] @ inv(cov) @ X_minus_mu[i, :])
        third = np.sum(np.array(thirds))
        log_likelihood = - first - second + third
        return log_likelihood
