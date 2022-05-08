import numpy as np
from typing import Tuple
# from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # ds = DecisionStump().fit(train_X,train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # raise NotImplementedError()
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X[:40], train_y[:40])
    # adaboost.fit(train_X, train_y)
    train_err, test_err = np.zeros(n_learners), np.zeros(n_learners)
    for t in range(n_learners):
        train_err[t] = adaboost.partial_loss(train_X[:40], train_y[:40], t)
        test_err[t] = adaboost.partial_loss(test_X[:40], test_y[:40], t)
        # train_err[t] = adaboost.partial_loss(train_X, train_y, t)
        # test_err[t] = adaboost.partial_loss(test_X, test_y, t)

    x = np.array(range(n_learners))
    fig = go.Figure(layout=go.Layout(title="AdaBoost - Test Error vs Train Error"))
    fig.add_trace(go.Scatter(x=x, y=test_err, mode='lines', name='Test Error'))
    fig.add_trace(go.Scatter(x=x, y=train_err, mode='lines', name='Train Error'))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    raise NotImplementedError()
