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
        lda_mean = lda.mu_

        naive_bayes = GaussianNaiveBayes()
        naive_bayes.fit(x, y)
        nb_prediction = naive_bayes.predict(x)
        nb_accuracy = accuracy(y, nb_prediction)
        nb_mean = naive_bayes.mu_

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right. Plot title should specify dataset used and subplot titles
        # should specify algorithm and accuracy. The subplots are with accuracy and have traces for
        # data-points setting symbols and colors

        n = f.split('.')[0].capitalize()  # dataset name

        subplot_titles = [f"Naive Bayes Prediction with {'%.3f' % nb_accuracy} accuracy",
                          f"LDA Prediction with {'%.3f' % lda_accuracy} accuracy"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles) \
            .add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=nb_prediction, symbol=y)),
                       row=1, col=1) \
            .add_trace(go.Scatter(x=x[:, 0], y=x[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=lda_prediction, symbol=y)),
                       row=1, col=2)
        fig.update_layout(dict(title=f"Compare Classifiers Predicted Classes over {n} Dataset"))
        fig.show()

        continue

        # print(np.unique(lda_y_hat == y, return_counts=True))
        # lda_lh = lda.likelihood(x)
        # print(lda_lh)
        # print(np.unique(nb_y_hat == y, return_counts=True))
        # nb_lh = naive_bayes.likelihood(x)
        # print(nb_lh)

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
