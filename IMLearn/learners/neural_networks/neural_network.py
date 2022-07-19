import numpy as np
from typing import List, Union, NoReturn
from IMLearn.base.base_module import BaseModule
from IMLearn.base.base_estimator import BaseEstimator
from IMLearn.desent_methods import StochasticGradientDescent, GradientDescent
from .modules import FullyConnectedLayer


class NeuralNetwork(BaseEstimator, BaseModule):
    """
    Class representing a feed-forward fully-connected neural network

    Attributes:
    ----------
    modules_: List[FullyConnectedLayer]
        A list of network layers, each a fully connected layer with its specified activation function

    loss_fn_: BaseModule
        Network's loss function to optimize weights with respect to

    solver_: Union[StochasticGradientDescent, GradientDescent]
        Instance of optimization algorithm used to optimize network

    pre_activations_:
    """

    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()

        self.modules_ = modules
        self.loss_fn_ = loss_fn
        self.solver_ = solver
        self.pre_activations_ = None
        self.post_activations_ = None

    # region BaseEstimator implementations
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit network over given input data using specified architecture and solver

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.solver_.fit(self, X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given samples using fitted network

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted labels of given samples
        """

        return np.argmax(self.compute_prediction(X), axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates network's loss over given data

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        --------
        loss : float
            Performance under specified loss function
        """

        return self.loss_fn_.compute_output(X=self.compute_prediction(X), y=y)

    # endregion

    # region BaseModule implementations
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network output with respect to modules' weights given input samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        output: ndarray of shape (1,)
            Network's output value including pass through the specified loss function

        Notes
        -----
        Function stores all intermediate values in the `self.pre_activations_` and
        `self.post_activations_` arrays
        """

        return self.loss_fn_.compute_output(X=self.compute_prediction(X), y=y, **kwargs)

    def compute_prediction(self, X: np.ndarray):
        """
        Compute network output (forward pass) with respect to modules' weights given input
        samples, except pass through specified loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        output : ndarray of shape (n_samples, n_classes)
            Network's output values prior to the call of the loss function
        """

        output = X.copy()
        self.pre_activations_ = []
        self.post_activations_ = [output]

        for module in self.modules_:
            o_t = output.copy()
            if module.include_intercept_:
                o_t = np.insert(o_t, 0, 1, axis=1)
            self.pre_activations_.append(o_t @ module.weights)
            output = module.compute_output(X=output)
            self.post_activations_.append(output)
        return output

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network's derivative (backward pass) according to the backpropagation algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        A flattened array containing the gradients of every learned layer.

        Notes
        -----
        Function depends on values calculated in forward pass and stored in
        `self.pre_activations_` and `self.post_activations_`
        """

        self.compute_output(X=X, y=y, **kwargs)

        j = 1
        pre_a_len = len(self.pre_activations_)
        post_a_len = len(self.post_activations_)
        T = len(self.modules_) - j
        pre_a = self.pre_activations_[pre_a_len - j]
        post_a = self.post_activations_[post_a_len - j]
        delta_t = self.loss_fn_.compute_jacobian(X=post_a, y=y, **kwargs)
        jac_t = self.modules_[T].activation_.compute_jacobian(X=pre_a, **kwargs)
        delta_t = np.einsum('kij,kj->ki', jac_t, delta_t)

        N = []

        j += 1
        for module in self.modules_[::-1]:
            pre_a = self.pre_activations_[pre_a_len - j]
            post_a = self.post_activations_[post_a_len - j]
            if module.include_intercept_:
                post_a = np.insert(post_a, 0, 1, axis=1)
            n_to_add = np.einsum('ij,ik->ijk', delta_t, post_a)
            N.append(np.mean(n_to_add, axis=0).T)

            if j > T + 1:
                break
            jac_t = module.activation_.compute_jacobian(X=pre_a)
            weights = module.weights if not module.include_intercept_ else module.weights[1:]
            weights_delta_t = np.einsum('ij,kj->ki', weights, delta_t)
            delta_t = np.einsum('kij,kj->ki', jac_t, weights_delta_t)
            j += 1

        return self._flatten_parameters(N[::-1])

    @property
    def weights(self) -> np.ndarray:
        """
        Get flattened weights vector. Solvers expect weights as a flattened vector

        Returns
        --------
        weights : ndarray of shape (n_features,)
            The network's weights as a flattened vector
        """
        return NeuralNetwork._flatten_parameters([module.weights for module in self.modules_])

    @weights.setter
    def weights(self, weights) -> None:
        """
        Updates network's weights given a *flat* vector of weights. Solvers are expected to update
        weights based on their flattened representation. Function first un-flattens weights and then
        performs weights' updates throughout the network layers

        Parameters
        -----------
        weights : np.ndarray of shape (n_features,)
            A flat vector of weights to update the model
        """
        non_flat_weights = NeuralNetwork._unflatten_parameters(weights, self.modules_)
        for module, weights in zip(self.modules_, non_flat_weights):
            module.weights = weights

    # endregion

    # region Internal methods
    @staticmethod
    def _flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
        """
        Flattens list of all given weights to a single one dimensional vector. To be used when passing
        weights to the solver

        Parameters
        ----------
        params : List[np.ndarray]
            List of differently shaped weight matrices

        Returns
        -------
        weights: ndarray
            A flattened array containing all weights
        """
        return np.concatenate([grad.flatten() for grad in params])

    @staticmethod
    def _unflatten_parameters(flat_params: np.ndarray, modules: List[BaseModule]) -> List[np.ndarray]:
        """
        Performing the inverse operation of "flatten_parameters"

        Parameters
        ----------
        flat_params : ndarray of shape (n_weights,)
            A flat vector containing all weights

        modules : List[BaseModule]
            List of network layers to be used for specifying shapes of weight matrices

        Returns
        -------
        weights: List[ndarray]
            A list where each item contains the weights of the corresponding layer of the network, shaped
            as expected by layer's module
        """
        low, param_list = 0, []
        for module in modules:
            r, c = module.shape
            high = low + r * c
            param_list.append(flat_params[low: high].reshape(module.shape))
            low = high
        return param_list
    # endregion
