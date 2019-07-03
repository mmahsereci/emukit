# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Tuple

from .noisy_user_function import NoisyUserFunctionWithGradientsResult
from emukit.core.loop import LoopState


class NoisyUserFunctionWithGradientsLoopState(LoopState):
    """
    Contains the state of the loop, which includes a history of all function evaluations.
    """

    def __init__(self, initial_results: List[NoisyUserFunctionWithGradientsResult]) -> None:
        """
        :param initial_results: The function results from previous function evaluations.
        """
        super().__init__(initial_results)

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

    def get_result_by_index(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: evaluation belonging to idx
        """
        return self.results[idx].X, self.results[idx].Y, self.results[idx].dY, self.results[idx].varY, \
               self.results[idx].vardY


class ProbLSLoopState(NoisyUserFunctionWithGradientsLoopState):

    def __init__(self, initial_results: List[NoisyUserFunctionWithGradientsResult], initial_learning_rates: List[float],
                 search_direction: np.ndarray, alpha0: float = 1e-4, alpha_stats: float = None) -> None:
        """
        :param initial_results: The function results from previous function evaluations. The first in the list needs to
        be the observations of the current location of the optimizer.
        :param initial_learning_rates: Learning rates corresponding to initial results
        :param search_direction: the search direction, (num_dim, )
        :param alpha0: initial step size
        :param alpha_stats: running average of accepted step sizes (defaults to alpha0)
        """
        super().__init__(initial_results)
        self.learning_rates = initial_learning_rates

        self.search_direction = search_direction
        self.alpha0 = alpha0
        self.alpha_stats = alpha_stats
        if self.alpha_stats is None:
            self.alpha_stats = alpha0

        self._f0 = initial_results[0].Y[0]
        self._df0 = initial_results[0].dY
        self._Sigmaf0 = np.clip(initial_results[0].varY[0], 1e-6, np.inf)
        self._Sigmadf0 = np.clip(initial_results[0].vardY, 1e-6, np.inf)
        self._x0 = initial_results[0].X

        self._beta = self._compute_scaling_factor()
        self.sigmaf, self.sigmadf = self._compute_observation_noise()

    def update(self, results: List[NoisyUserFunctionWithGradientsResult], learning_rates: List[float]) -> None:
        """
        :param results: The latest function results since last update
        :param learning_rates: The learning rates corresponding to the results
        """
        if not results:
            raise ValueError("Cannot update state with empty result list.")

        if len(results) != len(learning_rates):
            raise ValueError("Length of results must be equal to length of learning rate. Length are ", len(results),
                             " and ", len(learning_rates), " respectively.")

        self.iteration += 1
        self.results += results
        self.learning_rates += learning_rates

    def _compute_scaling_factor(self) -> float:
        """
        Computes the scale of the integrated Wiener process.
        :return: The scale of the integrated Wiener process.
        """
        beta = abs(np.dot(self.search_direction, self._df0))
        return beta

    def _compute_observation_noise(self):
        sigmaf = np.sqrt(self._Sigmaf0) / (self.alpha0 * self._beta)
        sigmadf = np.sqrt(np.dot(self.search_direction ** 2, self._Sigmadf0)) / self._beta
        return sigmaf, sigmadf

    # Todo: these are being computed each time when called. That is a bit inefficient... If I store and append, then
    #  I'll also need to change the update method.

    @property
    def X_transformed(self):
        """The X values in the scaled GP space"""
        return np.array(self.learning_rates)[:, np.newaxis] / self.alpha0

    @property
    def Y_transformed(self):
        """The function values in the scaled GP space"""
        return (self.Y - self._f0) / (self.alpha0 * self._beta)

    @property
    def dY_transformed(self):
        """The projected gradients in the scaled GP space"""
        res = np.dot(self.dY, self.search_direction) / self._beta
        return res[:, np.newaxis]

    @property
    def stdY_transformed(self):
        """The standard deviations of Y values in the scaled GP space"""
        return  np.sqrt(self.varY) / (self.alpha0 * self._beta)

    @property
    def stddY_transformed(self):
        """The standard deviation of projected gradients in the scaled GP space"""
        res = np.sqrt(np.dot(self.vardY, self.search_direction ** 2)) / self._beta
        return res[:, np.newaxis]

    @property
    def varY_transformed(self):
        """The variances of Y values in the scaled GP space"""
        return self.stdY_transformed ** 2

    @property
    def vardY_transformed(self):
        """The variances of projected gradients in the scaled GP space"""
        return self.stddY_transformed ** 2


def get_next_loop_state(state: ProbLSLoopState, idx: int, extrapolation_factor: float = 1.3) -> ProbLSLoopState:
    """
    Constructs the loop state containing the datapoint corresponding to index idx
    :param state: A ProbLSLoopState object
    :param idx: the index of the datapoint that will be used to create the loop state
    :param extrapolation_factor: The extrapolation factor
    :return: A ProbLSLoopState object
    """
    accepted_result = state.results[idx]
    accepted_learning_rate = state.learning_rates[idx]  # alpha_acc = tt * alpha0
    next_learning_rate = accepted_learning_rate * extrapolation_factor

    gamma = 0.95
    alpha_stats = gamma * state.alpha_stats + (1 - gamma) * accepted_learning_rate

    # reset NEXT initial setp size to average if multiplicative change was larger than 0.01 or 100
    theta_reset = 100.
    if (next_learning_rate < alpha_stats / theta_reset) or (next_learning_rate > alpha_stats * theta_reset):
        next_learning_rate = alpha_stats

    return ProbLSLoopState(initial_results=[accepted_result],
                           initial_learning_rates=[0.],
                           search_direction=-accepted_result.dY,  # this is SGD
                           alpha0=next_learning_rate,
                           alpha_stats=alpha_stats)

