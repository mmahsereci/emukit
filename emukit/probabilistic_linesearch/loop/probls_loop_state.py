# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Tuple

from . import ProbLSUserFunctionResult
from ...core.loop import LoopState


class ProbLSLoopState(LoopState):
    """
    Contains the state of the loop, which includes a history of all function evaluations
    """

    def __init__(self, initial_results: List[ProbLSUserFunctionResult], initial_wolfe_probabilities: List[float],
                 alpha0: float, alpha_stats: float) -> None:
        """
        :param initial_results: The function results from previous function evaluations
        :param initial_wolfe_probabilities: the wolfe probabilities corresponding to the initial results
        :param alpha0: initial step size
        :param alpha_stats: running average
        """
        super().__init__(initial_results)
        self._wolfe_probabilities = initial_wolfe_probabilities
        self._alpha0 = alpha0
        self._alpha_stats = alpha_stats

    @property
    def dY(self) -> np.ndarray:
        """
        :return: Gradients for all function evaluations in a 2d array: number of points by input dimensions.
        """
        return np.array([result.dY for result in self.results])

    @property
    def varY(self) -> np.ndarray:
        """
        :return: Estimated Variance for all function evaluations in a 2d array: number of points by output dimensions.
        """
        return np.array([result.varY for result in self.results])

    @property
    def vardY(self) -> np.ndarray:
        """
        :return: Estimate variances outputs for all gradients in a 2d array: number of points by output dimensions.
        """
        return np.array([result.vardY for result in self.results])

    @property
    def wolfe_probabilities(self):
        """
        :return: The Wolfe probabilities in a 2d array: number of points by output dimensions.
        """
        # Todo: check if this is a 2d array
        return np.array(self._wolfe_probabilities)

    def get_entry_by_index(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """:return: evaluation beloning to idx"""
        return self.results.X[idx], self.results.Y[idx], self.results.dY[idx], self.results.varY[idx], \
               self.results.vardY[idx], self._wolfe_probabilities[idx]

    @property
    def alpha_stats(self):
        return self._alpha_stats

    @property
    def alpha0(self):
        return self.alpha0


def create_loop_state_probls(x_init: np.ndarray, y_init: np.ndarray, dy_init:np.ndarray, vary_init:np.ndarray,
                             vardy_init: np.ndarray) -> ProbLSLoopState:
    """
    Creates a loop state object using the provided data

    :param x_init: x values of initial function evaluation.
    :param y_init: function value of initial function evaluation
    :param dy_init: gradient of initial function evaluation
    :param vary_init: estimated variance of initial function value
    :param vardy_init: estimated variance of initial gradient
    """
    # Todo: this might be too much, since we always initialize with 1 obersvation only
    if x_init.shape[0] != y_init.shape[0]:
        error_message = "X and Y should have the same length. Actual length x_init {}, y_init {}".format(
            x_init.shape[0], y_init.shape[0])
        raise ValueError(error_message)

    if x_init.shape[0] != dy_init.shape[0]:
        error_message = "X and dY should have the same length. Actual length x_init {}, dy_init {}".format(
            x_init.shape[0], dy_init.shape[0])
        raise ValueError(error_message)

    if x_init.shape[0] != vary_init.shape[0]:
        error_message = "X and varY should have the same length. Actual length x_init {}, vary_init {}".format(
            x_init.shape[0], vary_init.shape[0])
        raise ValueError(error_message)

    if x_init.shape[0] != vardy_init.shape[0]:
        error_message = "X and vardY should have the same length. Actual length x_init {}, vardy_init {}".format(
            x_init.shape[0], vardy_init.shape[0])
        raise ValueError(error_message)

    initial_results = []
    for x, y, dy, vary, vardy in zip(x_init, y_init, dy_init, vary_init, vardy_init):
        initial_results.append(ProbLSUserFunctionResult(x, y, dy, vary, vardy))

    return ProbLSLoopState(initial_results)
