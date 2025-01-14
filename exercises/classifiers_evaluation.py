from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable
    and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        x, y = load_dataset(f"./../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        callback = lambda ptn, px_, py_: losses.append(ptn.loss(x, y))
        Perceptron(callback=callback).fit(x, y)

        # Plot figure of loss as function of fitting iteration
        loss_fig = px.line(x=range(1, len(losses) + 1), y=losses)
        loss_fig.update_layout(dict(xaxis_title='Fitting Iteration', yaxis_title="Loss",
                                    title=f"Loss as Function of Fitting Iteration over {n} Dataset"))
        loss_fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 \
        else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def add_cov_ellipses(fig, lda, gnb):
    # Add ellipses depicting the covariances of the fitted Gaussians
    for i, mu in enumerate(gnb.mu_):
        fig.add_trace(get_ellipse(mu, np.diag(gnb.vars_[i])), row=1, col=1)
    for i, mu in enumerate(lda.mu_):
        fig.add_trace(get_ellipse(mu, lda.cov_), row=1, col=2)
    return fig


def add_fitted_means(fig, lda, gnb):
    # Add `X` dots specifying fitted Gaussians' means
    fig.add_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                             mode="markers", marker=dict(color='black', symbol='x')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                             mode="markers", marker=dict(color='black', symbol='x')),
                  row=1, col=2)
    return fig


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        x, y = load_dataset(f"./../datasets/{f}")

        # Fit models and predict over training set
        from IMLearn.metrics import accuracy
        lda = LDA()
        lda.fit(x, y)
        lda_prediction = lda.predict(x)
        lda_accuracy = accuracy(y, lda_prediction)

        gnb = GaussianNaiveBayes()
        gnb.fit(x, y)
        gnb_prediction = gnb.predict(x)
        gnb_accuracy = accuracy(y, gnb_prediction)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right. Plot title should specify dataset used and subplot titles
        # should specify algorithm and accuracy. The subplots are with accuracy and have traces for
        # data-points setting symbols and colors

        n = f.split('.')[0].capitalize()  # dataset name

        subplot_titles = [f"Naive Bayes Prediction with {'%.3f' % gnb_accuracy} accuracy",
                          f"LDA Prediction with {'%.3f' % lda_accuracy} accuracy"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles) \
            .add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers",
                                  marker=dict(color=gnb_prediction, symbol=y * 2)),
                       row=1, col=1) \
            .add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers",
                                  marker=dict(color=lda_prediction, symbol=y * 2)),
                       row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig = add_fitted_means(fig, lda, gnb)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig = add_cov_ellipses(fig, lda, gnb)
        fig.update_layout(dict(title=f"Compare Classifiers Predicted Classes over {n} Dataset"),
                          showlegend=False)
        fig.show()

"""
def quiz():
    # q1
    # S = {(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)}
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    naive_bayes = GaussianNaiveBayes()
    naive_bayes.fit(x, y)
    print(naive_bayes.pi_)
    print(naive_bayes.mu_)

    # q2
    # S = {([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)}
    x = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    naive_bayes2 = GaussianNaiveBayes()
    naive_bayes2.fit(x, y)
    print(naive_bayes2.pi_)
    print(naive_bayes2.mu_)
    print(naive_bayes2.vars_)
"""

if __name__ == '__main__':
    np.random.seed(0)
    # quiz()
    run_perceptron()
    compare_gaussian_classifiers()
