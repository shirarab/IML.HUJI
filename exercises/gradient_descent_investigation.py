import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, NoReturn

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve, auc

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


def fixed_convergence(f: BaseModule, fig, min_loss, best_eta, eta, values):
    fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="markers", name=eta))
    if f.compute_output() < min_loss:
        return f.compute_output(), eta
    return min_loss, best_eta


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_min_loss, l1_best_eta = np.inf, None
    l2_min_loss, l2_best_eta = np.inf, None
    fig_l1 = make_subplots()
    fig_l1.update_layout(dict(title=f"Convergence Rate for L1 (Fixed LR)",
                              xaxis_title="iteration", yaxis_title="values"))
    fig_l2 = make_subplots()
    fig_l2.update_layout(dict(title=f"Convergence Rate for L2 (Fixed LR)",
                              xaxis_title="iteration", yaxis_title="values"))
    for eta in etas:
        l1, l2 = L1(init.copy()), L2(init.copy())
        values = helper_plot_descent_path(l1, L1, "L1 (Fixed LR)", eta, FixedLR(eta))
        l1_min_loss, l1_best_eta = fixed_convergence(l1, fig_l1, l1_min_loss, l1_best_eta, eta, values)
        values = helper_plot_descent_path(l2, L2, "L2 (Fixed LR)", eta, FixedLR(eta))
        l2_min_loss, l2_best_eta = fixed_convergence(l2, fig_l2, l2_min_loss, l2_best_eta, eta, values)

    fig_l1.show()
    fig_l2.show()

    print(f"(Fixed LR) L1: lowest loss={l1_min_loss:.3f} with eta={l1_best_eta:.3f}")
    print(f"(Fixed LR) L2: lowest loss={l2_min_loss:.3f} with eta={l2_best_eta:.3f}")


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
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines", name=gamma))
        val = f.compute_output()
        if val < min_val:
            min_val, best_gamma = val, gamma

    print(f"(Exponential LR) L1: lowest norm={min_val:.3f} with gamma={best_gamma:.3f}")
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
    helper_plot_descent_path(L1(init.copy()), L1, f"L1 (Exponential LR), gamma: {gamma}",
                             eta, ExponentialLR(eta, gamma))
    helper_plot_descent_path(L2(init.copy()), L2, f"L2 (Exponential LR), gamma: {gamma}",
                             eta, ExponentialLR(eta, gamma))


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


def plot_logistic_convergence_SA_thlds(lr, X_train, y_train):
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_train)
    fpr, tpr, thlds = roc_curve(y_train, proba)

    fig = make_subplots()
    fig.update_layout(dict(title=f"ROC Curve for LR, auc: {auc(fpr, tpr):.3f}",
                           xaxis_title="fpr", yaxis_title="tpr"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines"))
    fig.show()
    return fpr, tpr, thlds


def logistic_regularize_cv(lambdas, penalty, X_train, y_train, X_test, y_test, alpha=.5):
    validation_scores = []
    for l in lambdas:
        solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
        lr = LogisticRegression(penalty=penalty, lam=l, solver=solver)
        ts, vs = cross_validate(lr, X_train, y_train, misclassification_error)
        validation_scores.append(vs)

    l_star = lambdas[int(np.argmin(validation_scores))]
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    lr_star = LogisticRegression(penalty=penalty, lam=l_star, solver=solver)
    lr_star.fit(X_train, y_train)
    test_error = lr_star.loss(X_test, y_test)
    print(f"({penalty}) Best lambda is {l_star} with test error of {test_error:.3f}")


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression()
    fpr, tpr, thlds = plot_logistic_convergence_SA_thlds(lr, X_train, y_train)

    a_star = thlds[np.argmax(tpr - fpr)]
    lr_star = LogisticRegression(alpha=a_star)
    lr_star.fit(X_train, y_train)
    test_error = lr_star.loss(X_test, y_test)
    print(f"(ROC curve) Best alpha is {a_star:.3f} with test error of {test_error:.3f}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    logistic_regularize_cv(lambdas, "l1", X_train, y_train, X_test, y_test)
    logistic_regularize_cv(lambdas, "l2", X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
