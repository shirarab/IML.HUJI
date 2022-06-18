import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, NoReturn

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's y-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recording the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values, weights = [], []

    def cb(**kwargs) -> NoReturn:
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return cb, values, weights


def helper_plot_descent_path(f: BaseModule, module: Type[BaseModule], name: str, eta: float, learning_rate) \
        -> List[np.ndarray]:
    cb, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=learning_rate, callback=cb)
    gd.fit(f=f, X=None, y=None)
    fig = plot_descent_path(module=module, descent_path=np.array(weights), title=f"module: {name}, eta: {eta}")
    fig.show()
    return values


def fixed_convergence(f: BaseModule, fig, min_loss, best_eta, eta, name, values):
    fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="lines+markers", name=name))
    if f.compute_output() < min_loss:
        return f.compute_output(), eta
    return min_loss, best_eta


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_min_loss, l1_best_eta = np.inf, None
    l2_min_loss, l2_best_eta = np.inf, None
    for eta in etas:
        fig = make_subplots()
        fig.update_layout(dict(title=f"Convergence Rate for L1 and L2 (Fixed LR)",
                               xaxis_title="iteration", yaxis_title="values"))
        l1, l2 = L1(init.copy()), L2(init.copy())
        values = helper_plot_descent_path(l1, L1, "L1 (Fixed LR)", eta, FixedLR(eta))
        l1_min_loss, l1_best_eta = fixed_convergence(l1, fig, l1_min_loss, l1_best_eta, eta, "L1", values)
        values = helper_plot_descent_path(l2, L2, "L2 (Fixed LR)", eta, FixedLR(eta))
        l2_min_loss, l2_best_eta = fixed_convergence(l2, fig, l2_min_loss, l2_best_eta, eta, "L2", values)

        fig.show()

    print(f"(Fixed LR) L1: lowest loss={l1_min_loss} with eta={l1_best_eta}")
    print(f"(Fixed LR) L2: lowest loss={l2_min_loss} with eta={l2_best_eta}")


def exponential_optimize_convergence_l1(init: np.ndarray, eta: float, gammas: Tuple[float]):
    fig = make_subplots()
    fig.update_layout(dict(title=f"Convergence Rate for L1 (Exponential LR)",
                           xaxis_title="iteration", yaxis_title="values"))

    min_val, best_gamma = np.inf, None
    for gamma in gammas:
        cb, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=cb)
        f = L1(init.copy())
        gd.fit(f=f, X=None, y=None)
        x = list(range(len(values)))
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=gamma))
        val = f.compute_output()
        if val < min_val:
            min_val, best_gamma = val, gamma

    print(f"(Exponential LR) L1: lowest norm={min_val} with gamma={best_gamma}")
    return fig


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = exponential_optimize_convergence_l1(init, eta, gammas)

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    helper_plot_descent_path(L1(init.copy()), L1, "L1 (Exponential LR)", eta, ExponentialLR(eta, gamma))
    helper_plot_descent_path(L2(init.copy()), L2, "L2 (Exponential LR)", eta, ExponentialLR(eta, gamma))


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()

"""

def helper_compare_fixed_learning_rates(eta: float, f: BaseModule, module: Type[BaseModule], name: str, fig2):
    cb, values, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=FixedLR(eta), callback=cb)
    gd.fit(f=f, X=None, y=None)
    fig = plot_descent_path(module=module, descent_path=np.array(weights), title=f"module: {name}, eta: {eta}")
    fig.show()

    x = list(range(len(values)))
    fig2.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name))
    return f.compute_output()


def xxxcompare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    etas: Tuple[float] = (1, .1, .01, .001)):
    l1_min_loss, l1_best_eta = np.inf, None
    l2_min_loss, l2_best_eta = np.inf, None
    for eta in etas:
        fig2 = make_subplots()
        fig2.update_layout(dict(title=f"Convergence Rate for L1 and L2 (Fixed LR)",
                                xaxis_title="iteration", yaxis_title="values"))
        l1_loss = helper_compare_fixed_learning_rates(eta, L1(init.copy()), L1, "L1 (Fixed LR)", fig2)
        if l1_loss < l1_min_loss:
            l1_min_loss, l1_best_eta = l1_loss, eta
        l2_loss = helper_compare_fixed_learning_rates(eta, L2(init.copy()), L2, "L2 (Fixed LR)", fig2)
        if l2_loss < l2_min_loss:
            l2_min_loss, l2_best_eta = l2_loss, eta
        fig2.show()

    print(f"(Fixed LR) L1: lowest loss={l1_min_loss} with eta={l1_best_eta}")
    print(f"(Fixed LR) L2: lowest loss={l2_min_loss} with eta={l2_best_eta}")

"""
