import time
import numpy as np
import gzip
from typing import Tuple, Callable, List, NoReturn

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    values, weights, grads = [], [], []

    def cb(**kwargs) -> NoReturn:
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        grads.append(kwargs["grad"])

    return cb, values, weights, grads


def init_fit_test_nn_relu_sgd(modules, cb, train_X, train_y, test_X, test_y, neurons_num=-1):
    """questions 5+8"""
    batch_size, max_iter = 256, 10000
    sgd = StochasticGradientDescent(learning_rate=FixedLR(1e-1), max_iter=max_iter, callback=cb, batch_size=batch_size)
    nn = NeuralNetwork(modules, CrossEntropyLoss(), sgd)
    nn.fit(train_X, train_y)
    text = f"Accuracy" if neurons_num < 0 else f"Accuracy (hidden layers-{neurons_num} neurons)"
    print(f"{text}: {accuracy(test_y, nn.predict(test_X))}")
    return nn


def plot_convergence_nn_relu_sgd(values, grads):
    """question 6"""
    fig = make_subplots()
    fig.update_layout(dict(title=f"Network with ReLU activations using SGD Convergence Process",
                           xaxis_title="iteration", yaxis_title="values+grads"))
    x = list(range(len(values)))
    grads_norm = [np.linalg.norm(g) for g in grads]
    fig.add_trace(go.Scatter(x=x, y=values, mode="markers+lines", name="values"))
    fig.add_trace(go.Scatter(x=x, y=grads_norm, mode="markers+lines", name="grads"))
    fig.show()


def plot_confusion_matrix(n_classes, conf_matrix):
    """question 7"""
    fig = make_subplots()
    fig.update_layout(dict(title=f"Test True- vs Predicted Confusion Matrix",
                           xaxis_title="True Labels", yaxis_title="Predicted Labels"))
    xy = list(range(n_classes))
    fig.add_trace(go.Heatmap(x=xy, y=xy, z=conf_matrix))
    fig.show()


def most_least_confused(conf_matrix):
    """question 7"""
    shape = conf_matrix.shape
    conf_matrix_copy = conf_matrix.copy()

    np.fill_diagonal(conf_matrix_copy, 0)
    k = 1
    index_array = np.argsort(conf_matrix_copy, axis=None)
    for i, j in zip(*np.unravel_index(index_array[::-1], shape)):
        if k > 2:
            break
        print(f"{k}) common confusion: {i} and {j}")
        k += 1

    k = 1
    index_array = np.argsort(conf_matrix, axis=None)
    for i, j in zip(*np.unravel_index(index_array, shape)):
        if k > 3:
            break
        print(f"{k}) not common confusion: {i} and {j}")
        k += 1


def plot_confidences():
    pass


def plot_most_least_confident_predictions(nn, test_X, num_images):
    """question 9"""
    d = 784
    pred = nn.compute_prediction(test_X)
    pred = np.apply_along_axis(lambda x: np.max(x), axis=1, arr=pred)
    mc_images_idx = np.argsort(pred)[-num_images:] if num_images == 64 else np.argmax(pred)
    mc_images = test_X[mc_images_idx].reshape(num_images, d)
    lc_images_idx = np.argsort(pred)[:num_images] if num_images == 64 else np.argmin(pred)
    lc_images = test_X[lc_images_idx].reshape(num_images, d)
    plot_images_grid(mc_images, title="most confident").show()
    plot_images_grid(lc_images, title="least confident").show()


def get_time_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    losses, times = [], []

    def cb(**kwargs) -> NoReturn:
        losses.append(np.mean(kwargs["val"]))
        times.append(time.time())

    return cb, losses, times


def get_nn_q10(n_features, n_classes, solver):
    neurons_num = 64
    modules = [
        FullyConnectedLayer(n_features, neurons_num, ReLU()),
        FullyConnectedLayer(neurons_num, neurons_num, ReLU()),
        FullyConnectedLayer(neurons_num, n_classes)
    ]
    nn = NeuralNetwork(modules, CrossEntropyLoss(), solver)
    return nn


def gd_vs_sgd(train_X, train_y, n_features, n_classes):
    """question 10"""
    max_iter = 10000
    gd_cb, gd_losses, gd_times = get_time_recorder_callback()

    gd = GradientDescent(learning_rate=FixedLR(1e-1), tol=1e-10, max_iter=max_iter, callback=gd_cb)
    gd_nn = get_nn_q10(n_features, n_classes, gd)
    gd_nn.fit(train_X, train_y)
    gd_times = np.array(gd_times) - gd_times[0]

    sgd_cb, sgd_losses, sgd_times = get_time_recorder_callback()
    sgd = StochasticGradientDescent(learning_rate=FixedLR(1e-1), tol=1e-10, max_iter=max_iter,
                                    callback=sgd_cb, batch_size=64)
    sgd_nn = get_nn_q10(n_features, n_classes, sgd)
    sgd_nn.fit(train_X, train_y)
    sgd_times = np.array(sgd_times) - sgd_times[0]

    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(dict(title=f"GD vs SGD Running times",
                           xaxis_title="Time", yaxis_title="Loss"))
    fig.add_trace(go.Scatter(x=gd_times, y=np.array(gd_losses),
                             mode="markers+lines", name="GD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sgd_times, y=np.array(sgd_losses),
                             mode="markers+lines", name="SGD"), row=1, col=2)
    fig.show()

    fig2 = make_subplots()
    fig2.update_layout(dict(title=f"GD vs SGD Running times on top of eachother",
                            xaxis_title="Time", yaxis_title="Loss"))
    fig2.add_trace(go.Scatter(x=gd_times, y=np.array(gd_losses), mode="markers+lines", name="GD"))
    fig2.add_trace(go.Scatter(x=sgd_times, y=np.array(sgd_losses), mode="markers+lines", name="SGD"))
    fig2.show()


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    cb, values, weights, grads = get_gd_state_recorder_callback()
    neurons_num = 64
    modules = [
        FullyConnectedLayer(n_features, neurons_num, ReLU()),
        FullyConnectedLayer(neurons_num, neurons_num, ReLU()),
        FullyConnectedLayer(neurons_num, n_classes)
    ]
    nn5 = init_fit_test_nn_relu_sgd(modules, cb, train_X, train_y, test_X, test_y, neurons_num)

    # Plotting convergence process
    plot_convergence_nn_relu_sgd(values, grads)

    # Plotting test true- vs predicted confusion matrix
    conf_matrix = confusion_matrix(test_y, nn5.predict(test_X))
    plot_confusion_matrix(n_classes, conf_matrix)
    most_least_confused(conf_matrix)

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    cb, values, weights, grads = get_gd_state_recorder_callback()
    modules = [FullyConnectedLayer(n_features, n_classes)]
    nn8 = init_fit_test_nn_relu_sgd(modules, cb, train_X, train_y, test_X, test_y)

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    plot_most_least_confident_predictions(nn5, test_X[test_y == 7], 64)

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    gd_vs_sgd(train_X[:2500], train_y[:2500], n_features, n_classes)
