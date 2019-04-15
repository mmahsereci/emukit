# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List

from . import ProbLSUserFunctionResult
from ...core.loop import LoopState


class ProbLSLoopState(LoopState):
    """
    Contains the state of the loop, which includes a history of all function evaluations
    """

    def __init__(self, initial_results: List[ProbLSUserFunctionResult], initial_wolfe_probabilities: List[float]) \
            -> None:
        """
        :param initial_results: The function results from previous function evaluations
        :param initial_wolfe_probabilities: the wolfe probabilities corresponding to the initial results
        """
        super().__init__(initial_results)
        self._wolfe_probabilities = initial_wolfe_probabilities

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
        :return: The wolfe probabilities in a 2d array: number of points by output dimensions.
        """
        # Todo: check if this is a 2d array
        return np.array(self._wolfe_probabilities)


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
