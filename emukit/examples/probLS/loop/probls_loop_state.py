# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Tuple

from .noisy_user_function import NoisyUserFunctionWithGradientsResult
from emukit.core.loop import LoopState


class ProbLSLoopState(LoopState):
    """
    Contains the state of the loop, which includes a history of all function evaluations.
    """

    def __init__(self, initial_results: List[NoisyUserFunctionWithGradientsResult], search_direction: np.ndarray,
                 initial_wolfe_probabilities: List[float], alpha0: float = 1e-4, alpha_stats: float = None) -> None:
        """
        :param initial_results: The function results from previous function evaluations
        :param initial_wolfe_probabilities: the wolfe probabilities corresponding to the initial results
        :param search_direction: the search direction, (num_dim, )
        :param alpha0: initial step size
        :param alpha_stats: running average of accepted step sizes (defaults to alpha0)
        """
        super().__init__(initial_results)
        self._wolfe_probabilities = initial_wolfe_probabilities
        self.search_direction = search_direction
        self._alpha0 = alpha0
        self.extrapolation_factor = 1.
        if alpha_stats is None:
            self._alpha_stats = alpha0
        self.beta = self._compute_scaling_factor()

    @property
    def dY(self) -> np.ndarray:
        """
        :return: Noisy gradients for all function evaluations in a 2d array: number of points by input dimensions.
        """
        return np.array([result.dY for result in self.results])

    @property
    def varY(self) -> np.ndarray:
        """
        :return: (Estimated) variances for all function evaluations in a 2d array: number of points by output dimensions.
        """
        return np.array([result.varY for result in self.results])

    @property
    def vardY(self) -> np.ndarray:
        """
        :return: (Estimated) variances for all gradients in a 2d array: number of points by output dimensions.
        """
        return np.array([result.vardY for result in self.results])

    @property
    def wolfe_probabilities(self):
        """
        :return: The Wolfe probabilities in a 2d array: number of points by output dimensions.
        """
        return np.array(self._wolfe_probabilities)

    @wolfe_probabilities.setter
    def wolfe_probabilities(self, values: np.ndarray) -> None:
        self._wolfe_probabilities = values

    @property
    def alpha_stats(self):
        """
        :return: Running average of accepted learning rates. Used as fallback value.
        """
        return self._alpha_stats

    @alpha_stats.setter
    def alpha_stats(self, value: float) -> None:
        """
        :param value:
        :return:
        """
        self._alpha_stats = value

    @property
    def alpha0(self):
        """
        :return: the initial step size
        """
        return self._alpha0

    def get_result_by_index(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        :return: evaluation beloning to idx
        """
        return self.results.X[idx], self.results.Y[idx], self.results.dY[idx], self.results.varY[idx], \
               self.results.vardY[idx], self._wolfe_probabilities[idx]

    def _compute_scaling_factor(self) -> float:
        """
        Computes the scale of the integrated Wiener process.
        :return: The scale of the integrated Wiener process.
        """
        # Todo: check if this gives int as return
        beta = abs(np.dot(self.search_direction, self.vardY[0]))
        return beta


def create_loop_state_probls(x_init: np.ndarray, y_init: float, dy_init: np.ndarray, vary_init: float,
                             vardy_init: np.ndarray, alpha0: float, search_direction: np.ndarray,
                             alpha_stats: float = None) -> ProbLSLoopState:
    """
    Creates a loop state object for probabilistic line search using the provided data.

    :param x_init: x value of initial function evaluation (num_dim, )
    :param y_init: noisy function value of initial function evaluation
    :param dy_init: noisy gradient of initial function evaluation  (n_points, num_dim)
    :param vary_init: (estimated) variance of initial function value  (n_points, 1)
    :param vardy_init: (estimated) variance of initial gradient (n_points, num_dim)
    :param alpha0: the initial step size
    :param search_direction: the search direction (num_dim, ). default to y_init
    :param alpha_stats: the initial value of the running average of accepted step sizes (defaults to alpha0)
    """
    # Todo: this might be too much, since we always initialize with 1 observation only

    if alpha_stats is None:
        alpha_stats = alpha0

    if search_direction is None:
        search_direction = [-y_init]

    # check types
    if not isinstance(y_init, float):
        error_message = "Y should be of type float. Actual type Y {}".format(type(y_init))
        raise TypeError(error_message)

    if not isinstance(vary_init, float):
        error_message = "varY should be of type float. Actual type varY {}".format(type(vary_init))
        raise TypeError(error_message)

    # check dimension and shapes
    if x_init.ndim != 1:
        error_message = "X should be 1d array. Actual dim X {}".format(x_init.ndim)
        raise ValueError(error_message)

    if x_init.shape != dy_init.shape:
        error_message = "X and dY should have the same shape. Actual shapes X {}, dY {}".format(
            x_init.shape, dy_init.shape)
        raise ValueError(error_message)

    if x_init.shape != vardy_init.shape:
        error_message = "X and vardY should have the same shape. Actual shapes X {}, vardY {}".format(
            x_init.shape, vardy_init.shape)
        raise ValueError(error_message)

    if x_init.shape != search_direction.shape:
        error_message = "X and search_direction should have the same shape. Actual shapes X {}, " \
                        "search_direction {}".format(x_init.shape, search_direction.shape)
        raise ValueError(error_message)

    initial_results = [NoisyUserFunctionWithGradientsResult(x_init, np.array([y_init]), dy_init, np.array([vary_init]),
                                                            vardy_init)]
    initial_wolfe_probabilities = [0.]

    return ProbLSLoopState(initial_results, search_direction, initial_wolfe_probabilities, alpha0, alpha_stats)
