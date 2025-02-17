from typing import Tuple
import numpy as np
import pandas as pd
import sklearn.metrics


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_i = indices[:int(np.ceil(train_proportion * n_samples))],
    test_i = indices[int(np.floor((1 - train_proportion) * n_samples)):]
    train_x = X.T[X.axes[0][train_i]].T
    train_y = y.T[y.axes[0][train_i]].T
    test_x = X.T[X.axes[0][test_i]].T
    test_y = y.T[y.axes[0][test_i]].T
    return train_x, train_y, test_x, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i`
        was found in vector `a` while value `j` vas found in vector `b`
    """

    a_unique, a_counts = np.unique(a, return_counts=True)
    b_unique, b_counts = np.unique(b, return_counts=True)
    conf_matrix = np.zeros((len(a_unique), len(b_unique)))

    for i, au in enumerate(a_unique):
        for j, bu in enumerate(b_unique):
            conf_matrix[au, bu] = a_counts[i]

    return conf_matrix
