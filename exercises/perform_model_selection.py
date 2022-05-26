from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x_true = np.linspace(-1.2, 2, n_samples)
    y_true = f(x_true)
    x = np.random.uniform(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f(x) + epsilon
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3.0)
    train_x, train_y = train_x[0].to_numpy(), train_y.to_numpy()
    test_x, test_y = np.array(test_x[0]), np.array(test_y)

    fig = go.Figure(layout=go.Layout(title=f"Model Selection - Test vs Train (noise={noise})"))
    fig.add_trace(go.Scatter(x=x_true, y=y_true, mode='lines', name='Noiseless Polynom'))
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode='markers', name='Train Set'))
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode='markers', name='Test Set'))
    fig.update_layout(dict(xaxis_title="samples", yaxis_title="results"))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10
    k_degrees = np.array(range(max_degree + 1))
    k_train = np.zeros(max_degree + 1)
    k_validate = np.zeros(max_degree + 1)
    for k in range(11):
        poly_model = PolynomialFitting(k)
        avg_train, avg_validation = cross_validate(poly_model, train_x, train_y, mean_square_error)
        k_train[k] = avg_train
        k_validate[k] = avg_validation
    fig2 = go.Figure(layout=go.Layout(title=f"CV for polynomial fitting with degrees 0,1,...,10 (noise={noise})"))
    fig2.add_trace(go.Scatter(x=k_degrees, y=k_train, mode='lines', name='Train'))
    fig2.add_trace(go.Scatter(x=k_degrees, y=k_validate, mode='lines', name='Validate'))
    fig2.update_layout(dict(xaxis_title="degrees", yaxis_title="values"))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = int(np.argmin(k_validate))
    k_model = PolynomialFitting(k_star).fit(train_x, train_y)
    min_loss = k_model.loss(test_x, test_y)
    print(f"(noise={noise}) k star is {k_star} with test error of {min_loss}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = x[:n_samples], y[:n_samples], x[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.001, 2, n_evaluations)
    ridge_train, ridge_validate = [], []
    lasso_train, lasso_validate = [], []
    for lam in lambdas:
        ridge = RidgeRegression(lam)
        lasso = Lasso(lam)
        r_train, r_validation = cross_validate(ridge, train_x, train_y, mean_square_error)
        l_train, l_validation = cross_validate(lasso, train_x, train_y, mean_square_error)
        ridge_train.append(r_train)
        ridge_validate.append(r_validation)
        lasso_train.append(l_train)
        lasso_validate.append(l_validation)

    subplot_titles = ["Ridge", "Lasso"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles) \
        .add_trace(go.Scatter(x=lambdas, y=ridge_train, mode="lines", name="ridge train"), row=1, col=1) \
        .add_trace(go.Scatter(x=lambdas, y=ridge_validate, mode="lines", name="ridge validate"), row=1, col=1) \
        .add_trace(go.Scatter(x=lambdas, y=lasso_train, mode="lines", name="lasso train"), row=1, col=2) \
        .add_trace(go.Scatter(x=lambdas, y=lasso_validate, mode="lines", name="lasso validate"), row=1, col=2)
    fig.update_layout(dict(title=f"Cross Validation on Ridge and Lasso",
                           xaxis_title="lambdas", yaxis_title="train&validate"))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_lam = lambdas[int(np.argmin(ridge_validate))]
    ridge_model = RidgeRegression(ridge_lam).fit(train_x, train_y)
    ridge_loss = ridge_model.loss(test_x, test_y)

    lasso_lam = lambdas[int(np.argmin(lasso_validate))]
    lasso_model = Lasso(lasso_lam).fit(train_x, train_y)
    lasso_loss = mean_square_error(test_y, lasso_model.predict(test_x))

    lr_model = LinearRegression().fit(train_x, train_y)
    lr_loss = lr_model.loss(test_x, test_y)

    print(f"ridge - best lambda is {ridge_lam} with test error of {ridge_loss}")
    print(f"lasso - best lambda is {lasso_lam} with test error of {lasso_loss}")
    print(f"linear regression - with test error of {lr_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
