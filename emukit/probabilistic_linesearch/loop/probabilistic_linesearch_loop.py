# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union, Callable
import numpy as np

from ...core.loop import OuterLoop, UserFunction, UserFunctionWrapper
from ...probabilistic_linesearch.models import CubicSplineGP
from . import WolfeThresholdStoppingCondition


import logging
_log = logging.getLogger(__name__)


class ProbLineSearch(OuterLoop):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, model: CubicSplineGP, alpha0: float, f0: float, df0: np.array, search_direction: np.array):
        """

        :param alpha0: initial step
        :param f0: function value at starting point of line search
        :param df0: gradient at starting point of line search (n_dim, 1)
        :param search_direction: The search direction/negative stochastic gradient (n_dim, 1)
        """
        self.model = model
        self.alpha0 = alpha0
        self.search_direction = search_direction

        self.df0 = df0
        self.f0 = f0

        self.beta = self._compute_scaling_factor()

        self._T = np.array([0.])[:, np.newaxis]
        self._Y = np.array([1.])[:, np.newaxis]
        self._dY = np.array([1.])[:, np.newaxis]

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

    def run_loop(self, user_function: Union[UserFunction, Callable], stopping_condition: WolfeThresholdStoppingCondition,
                 context: dict=None) -> None:
        """
        :param user_function: The function that we are emulating
        :param stopping_condition: If integer - a number of iterations to run, if object - a stopping condition object
                                   that decides whether we should stop collecting more points
        :param context: The context is used to force certain parameters of the inputs to the function of interest to
                        have a given value. It is a dictionary whose keys are the parameter names to fix and the values
                        are the values to fix the parameters to.
        """
        if not isinstance(stopping_condition, WolfeThresholdStoppingCondition):
            raise ValueError("Expected stopping_condition to be a WolfeThresholdStoppingCondition instance, "
                             "but received {}".format(type(stopping_condition)))

        if not isinstance(user_function, UserFunction):
            user_function = UserFunctionWrapper(user_function)

        if isinstance(stopping_condition, int):
            stopping_condition = WolfeThresholdStoppingCondition(stopping_condition + self.loop_state.iteration)

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

