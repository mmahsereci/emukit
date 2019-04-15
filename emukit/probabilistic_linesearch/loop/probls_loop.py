# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union, Callable
import numpy as np

from ...core.loop import OuterLoop
from ...probabilistic_linesearch.models import CubicSplineGP
from . import WolfeThresholdStoppingCondition, ProbLSLoopState, ProbLSUserFunctionResult
from .noisy_user_function import NoisyUserFunctionWithGradientsWrapper
from ...probabilistic_linesearch.acquisitions import WolfeProbability


import logging
_log = logging.getLogger(__name__)


class ProbLineSearch(OuterLoop):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, search_direction: np.ndarray, model: CubicSplineGP, loop_state_init: ProbLSLoopState,
                 acquisition: WolfeProbability = None):
        """
        The loop for vanilla Bayesian Quadrature

        :param search_direction: The search direction (negative noisy gradient) (num_dim, )
        :param model: the cubic spline Gaussian Process model
        :param loop_state_init: The initial state of the loop.
        """
        self.model = model
        self.loop_state = loop_state_init
        self.search_direction = search_direction

        self.beta = self._compute_scaling_factor()

        self._T = np.array([0.])[:, np.newaxis]
        self._Y = np.array([1.])[:, np.newaxis]
        self._dY = np.array([1.])[:, np.newaxis]

        if acquisition is None:
            # Todo: change this to EI time WP, also in init: Union[PW, EI-PW]
            acquisition = WolfeProbability(model)

        # Todo: we do not have point calculator yet
        #
        candidate_point_calculator = SequentialPointCalculator(acquisition)

        super().__init__(candidate_point_calculator, model_updater, loop_state_init)

    def _compute_scaling_factor(self) -> float:
        """
        Computes the scale of the integrated Wiener process.
        :return: The scale of the integrated Wiener process.
        """
        beta = 1.
        return beta

    def _compute_scaled_variances(self) -> Tuple[float, float]:
        """
        Computes the scaled variances of the noisy function values and projected gradients
        :return: scaled variance of function values, scaled variance of projected gradient
        """
        np.inner(self.search_direction)
        sigma_f = 0.
        sigma_df = 0.
        return sigma_f, sigma_df

    def _update_average_statistics(self):
        """Updates alpha_stats"""
        pass

    def get_result(self, alpha_ext: float = 1.3, theta_reset: float = 100.) -> Tuple[float, float, ProbLSLoopState]:
        """
        :param alpha_ext: extrapolation parameter (default 1.3). should be in interval [1.1, 1.3]
        :param theta_reset: safeguard for numerical stability. Defined maximal variability of each step
        :return: accepted step size, Wolfe probability of accepted step, loop state with winning evaluation
        """
        alpha0 = self.loop_state.alpha0

        # Get winning point
        # Todo: get accepted idx from somewhere
        idx_acc = 1
        x, y, dy, vary, vardy, pw = self.loop_state.get_entry_by_index(idx_acc)
        accepted_results = [ProbLSUserFunctionResult(x, y, dy, vary, vardy)]
        # Todo: get accepted step size from somewhere
        alpha_acc = 1.

        # update alpha stats
        gamma = 0.95
        alpha_stats = gamma * self.loop_state.alpha_stats + (1. - gamma) * alpha_acc

        # next step size
        alpha_next = alpha_ext * alpha_acc

        # Safeguard if step size is changed more than 100 %
        if (alpha_next * theta_reset < alpha_stats) or (alpha_next > alpha_stats * theta_reset):
            alpha_next = alpha0
            _log.info("Resetting step size.")

        loop_state = ProbLSLoopState(accepted_results, [0.], alpha_next, alpha_stats)
        return alpha_acc, pw, loop_state

    def run_loop(self, user_function: NoisyUserFunctionWithGradientsWrapper,
                 stopping_condition: WolfeThresholdStoppingCondition = None, context: dict=None) -> None:
        """
        :param user_function: The function that we are emulating
        :param stopping_condition: A threshold on the Wolfe probability is used as stopping criterion.
        """
        if not isinstance(stopping_condition, WolfeThresholdStoppingCondition):
            raise ValueError("Expected stopping_condition to be a WolfeThresholdStoppingCondition instance, "
                             "but received {}".format(type(stopping_condition)))

        if not isinstance(user_function, NoisyUserFunctionWithGradientsWrapper):
            raise ValueError("Expected user_function to be a NoisyUserFunctionWithGradientsWrapper instance, "
                             "but received {}".format(type(user_function)))

        _log.info("Starting outer loop")

        self.loop_start_event(self, self.loop_state)

        while not stopping_condition.should_stop(self.loop_state):
            _log.info("Iteration {}".format(self.loop_state.iteration))

            self._update_models()
            new_x = self.candidate_point_calculator.compute_next_points(self.loop_state, context)
            results = user_function.evaluate(new_x)
            self.loop_state.update(results)
            self.iteration_end_event(self, self.loop_state)

        self._update_models()
        _log.info("Finished outer loop")

