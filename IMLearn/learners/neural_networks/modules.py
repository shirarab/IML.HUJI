import numpy as np
from IMLearn.base.base_module import BaseModule
from IMLearn.metrics.loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network

    Attributes:
    -----------
    input_dim_: int
        Size of input to layer (number of neurons in preceding layer

    output_dim_: int
        Size of layer output (number of neurons in layer_)

    activation_: BaseModule
        Activation function to be performed after integration of inputs and weights

    weights: ndarray of shape (input_dim_, outout_din_)
        Parameters of function with respect to which the function is optimized.

    include_intercept: bool
        Should layer include an intercept or not
    """

    def __init__(self, input_dim: int, output_dim: int, activation: BaseModule = None, include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer

        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer

        output_dim: int
            Size of layer output (number of neurons in layer_)

        activation_: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer

        include_intercept: bool, default=True
            Should layer include an intercept or not

        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """
        super().__init__()

        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.activation_ = Identity() if activation is None else activation
        self.include_intercept_ = include_intercept

        if not include_intercept:
            self.weights_ = np.random.normal(0, 1 / input_dim, (input_dim, output_dim))
        else:
            self.weights_ = np.random.normal(0, 1 / (input_dim + 1), (input_dim + 1, output_dim))

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x: output value of layer at point
        self.weights and given input

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self.weights
        """

        new_x = X.copy()
        if self.include_intercept_:
            new_x = np.insert(new_x, 0, 1, axis=1)
        return self.activation_.compute_output(X=new_x @ self.weights, **kwargs)

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        -------
        output: ndarray of shape (input_dim, n_samples)
            Derivative with respect to self.weights at point self.weights
        """

        new_x = X.copy()
        if self.include_intercept_:
            new_x = np.insert(new_x, 0, 1, axis=1)
        jacobian = self.activation_.compute_jacobian(X=new_x @ self.weights, **kwargs)
        return np.einsum('ki,kj->kij', new_x, jacobian)
        # return diag_compute_jacobian_helper(new_x, jacobian)


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        return np.maximum(0, X)

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to given data

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to compute derivative with respect to

        Returns:
        -------
        output: ndarray of shape (n_samples,)
            Element-wise derivative of ReLU with respect to given data
        """
        n_samples, input_dim = X.shape
        derivative = np.where(X > 0, 1, 0)
        return np.einsum('ij,kj->ikj', derivative, np.identity(input_dim))
        # return diag_compute_jacobian_helper(derivative, np.identity(input_dim))


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """

    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the Cross-Entropy over the Softmax of given data, with respect to every

        CrossEntropy(Softmax(x),e_k) for every sample x

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data for which to compute the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples,)
            cross-entropy loss value of given X and y
        """
        # n_samples = X.shape[0]
        # return np.array([cross_entropy(y[i], softmax(X[i])) for i in range(n_samples)])
        return cross_entropy(y, softmax(X))

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the derivative of the cross-entropy loss function with respect to every given sample

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data with respect to which to compute derivative of the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            derivative of cross-entropy loss with respect to given input
        """

        a = np.zeros(X.shape, dtype=int)
        a[:, y] = 1
        return softmax(X) - a


class Identity(BaseModule):
    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return X

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        n_samples, input_dim = X.shape
        derivative = np.ones((n_samples, input_dim))
        return np.einsum('ij,kj->ikj', derivative, np.identity(input_dim))
        # return diag_compute_jacobian_helper(derivative, np.identity(input_dim))

# def diag_compute_jacobian_helper(out, optimize):
#     return np.einsum('ki,kj->kij', out, optimize)
