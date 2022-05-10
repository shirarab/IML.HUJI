from typing import Tuple
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


# Question 1: Train- and test errors of AdaBoost in noiseless case
def plot_train_test_errors_adaboost(n_learners, test_err, train_err, noise):
    x_range = np.array(range(n_learners))
    fig = go.Figure(layout=go.Layout(title=f"AdaBoost - Test Error vs Train Error (noise={noise})"))
    fig.add_trace(go.Scatter(x=x_range, y=test_err, mode='lines', name='Test Error'))
    fig.add_trace(go.Scatter(x=x_range, y=train_err, mode='lines', name='Train Error'))
    fig.show()


# Question 2: Plotting decision surfaces
def plot_decision_surfaces(T, adaboost, lims, scatter, noise):
    subplot_titles = [f"{t} Classifiers" for t in T]
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)

    for i, t in enumerate(T):
        partial_decision_surface = lambda x: adaboost.partial_predict(x, t)
        fig2.add_traces([decision_surface(partial_decision_surface, lims[0], lims[1], showscale=False),
                         scatter], rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(dict(title=f"Decision Boundries and Test Set (noise={noise})"), showlegend=False)
    fig2.show()


# Question 3: Decision surface of best performing ensemble
def plot_decision_surface_best_ensemble(test_err, adaboost, lims, scatter, noise):
    min_t = np.argmin(test_err)
    min_test_err = test_err[min_t]

    fig3 = make_subplots(rows=1, cols=1)
    partial_decision_surface = lambda x: adaboost.partial_predict(x, min_t)
    fig3.add_traces([decision_surface(partial_decision_surface, lims[0], lims[1], showscale=False),
                     scatter], rows=1, cols=1)
    fig3.update_layout(dict(title=f"Best T={min_t} with Accuracy={1 - min_test_err} (noise={noise})"),
                       showlegend=False)
    fig3.show()


# Question 4: Decision surface with weighted samples
def plot_decision_surface_weights(adaboost, lims, train_X, train_y, noise):
    D = adaboost.D_
    D = D / np.max(D) * 10

    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=train_y, colorscale=[custom[0], custom[-1]], size=D))

    fig4 = make_subplots(rows=1, cols=1)
    fig4.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                     scatter], rows=1, cols=1)
    fig4.update_layout(dict(title=f"Decision Surface with Weighted Samples (noise={noise})"),
                       showlegend=False)
    fig4.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    train_err, test_err = np.zeros(n_learners), np.zeros(n_learners)
    for t in range(n_learners):
        train_err[t] = adaboost.partial_loss(train_X, train_y, t + 1)
        test_err[t] = adaboost.partial_loss(test_X, test_y, t + 1)

    plot_train_test_errors_adaboost(n_learners, test_err, train_err, noise)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T \
           + np.array([-.1, .1])

    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=test_y, colorscale=[custom[0], custom[-1]]))

    plot_decision_surfaces(T, adaboost, lims, scatter, noise)

    # Question 3: Decision surface of best performing ensemble
    plot_decision_surface_best_ensemble(test_err, adaboost, lims, scatter, noise)

    # Question 4: Decision surface with weighted samples
    plot_decision_surface_weights(adaboost, lims, train_X, train_y, noise)


if __name__ == '__main__':
    np.random.seed(0)
    # fast_fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
