# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple
import numpy as np

from emukit.core.loop import OuterLoop
from emukit.core.acquisition.acquisition import Product
from emukit.core.loop.stopping_conditions import FixedIterationsStoppingCondition
from ...probLS.models.cubic_spline_gp import CubicSplineGP
from ...probLS.loop.probls_loop_state import ProbLSLoopState, get_next_loop_state
from ...probLS.acquisitions.probls_acqusitions import WolfeProbability, NoisyExpectedImprovement
from ...probLS.loop.probls_wolfe_conditions import WolfeConditions
from ...probLS.loop.probls_candidate_point_calculator import ProbLSCandidatePointCalculator
from ...probLS.loop.probls_wolfe_threshold_stopping_condition import WolfeThresholdStoppingCondition
from ...probLS.loop.noisy_user_function import NoisyUserFunctionWithGradientsWrapper
from ...probLS.loop.probls_model_updater import ProbLSModelUpdater

import logging
_log = logging.getLogger(__name__)


class ProbLineSearch(OuterLoop):
    """
    The probabilistic line search. Adapts learning rates for stochastic gradient descent.

    Probabilistic Line Searches for Stochastic Optimization
    M. Mahsereci & P. Hennig, NeurIPS 2015, and JMLR 2017
    """

    def __init__(self, loop_state_init: ProbLSLoopState, verbosity: int=0):
        """
        The loop for the probabilistic line search (a bit over-engineered, but it is fun)

        :param loop_state_init: The initial state of the loop.
        :param verbosity: verbosity of the logger (default 0, silence)
        """
        self.model = CubicSplineGP(X=loop_state_init.X_transformed,
                                   Y=loop_state_init.Y_transformed,
                                   dY=loop_state_init.dY_transformed,
                                   varY=loop_state_init.sigmaf ** 2,
                                   vardY=loop_state_init.sigmadf ** 2)

        pw = WolfeProbability(self.model, WolfeConditions())
        nei = NoisyExpectedImprovement(self.model)
        self.acquisition = Product(pw, nei)

        candidate_point_calculator = ProbLSCandidatePointCalculator(acquisition=self.acquisition)
        model_updater = ProbLSModelUpdater(self.model)

        super().__init__(candidate_point_calculator, model_updater, loop_state_init)

        self.stopping_condition_pw = WolfeThresholdStoppingCondition(self.acquisition.acquisition_1.wolfe_condition)
        self.stopping_condition_fix = FixedIterationsStoppingCondition(i_max=5)  # max increase: 5:32, 6:64
        self._verbosity = verbosity

    def run_loop(self, user_function: NoisyUserFunctionWithGradientsWrapper) -> Tuple[ProbLSLoopState, float]:
        """
        This runs the line search.
        :param user_function: The function that we are emulating
        :return: The loop state of the accepted point (can be used as initial loop state for the next line search), the
        accepted learning rate
        """

        tt = 1.
        while True:

            # evaluate function, update loop_state
            self._evaluate(user_function, tt)

            # update model, get wolfe probabilities
            wolfe_probabilities = self._update()

            # check current points for acceptance (checks latest point first)
            should_stop_pw, idx_accept = self.stopping_condition_pw.should_stop(wolfe_probabilities)

            # stop if Wolfe point is found
            if should_stop_pw:
                if self._verbosity > 1:
                    _log.info("Wolfe point found")
                next_loop_state = get_next_loop_state(self.loop_state, idx_accept)
                return next_loop_state, self.model.X[idx_accept, 0] * self.loop_state.alpha0

            # no Wolfe point found -> get next candidate point
            tt, should_stop_uphill = self.candidate_point_calculator.compute_next_points(self.loop_state)

            # stop because we are walking uphill
            if should_stop_uphill:
                if self._verbosity > 0:
                    _log.info("Uphill")
                self._evaluate(user_function, tt)
                next_loop_state = get_next_loop_state(self.loop_state, -1)
                return next_loop_state, self.model.X[-1, 0] * self.loop_state.alpha0

            # stop if fixed iteration reached
            if self.stopping_condition_fix.should_stop(self.loop_state):
                if self._verbosity > 0:
                    _log.info("fixed iter reached")
                self._evaluate(user_function, tt)
                wolfe_probabilities = self._update()
                should_stop_pw, idx_accept = self.stopping_condition_pw.should_stop(wolfe_probabilities)
                if should_stop_pw:
                    if self._verbosity > 0:
                        _log.info("Wolfe point found in last evaluation")
                    next_loop_state = get_next_loop_state(self.loop_state, idx_accept)
                    return next_loop_state, self.model.X[idx_accept, 0] * self.loop_state.alpha0

                idx_accept = self.model.get_index_of_lowest_observed_mean()
                next_loop_state = get_next_loop_state(self.loop_state, idx_accept)
                return next_loop_state, self.model.X[idx_accept, 0] * self.loop_state.alpha0

    def _xfunc(self, tt: float) -> np.ndarray:
        """
        Converts scaled learning rate to parameters
        :param tt: the scaled learning rate
        :return: the location x in parameters space corresponding to tt.
        """
        return self.loop_state.results[0].X + (tt * self.loop_state.alpha0) * self.loop_state.search_direction[:, np.newaxis].T

    def _evaluate(self, user_func: NoisyUserFunctionWithGradientsWrapper, tt: float) -> None:
        """
        Evaluates the user function and
        :param user_func: the user function
        :param tt: the scaled learning rate where we evaluate the function
        """
        # evaluate function
        results = user_func.evaluate(self._xfunc(tt))

        # append result to loop_state
        self.loop_state.update(results=results, learning_rates=[tt * self.loop_state.alpha0])

    def _update(self) -> np.ndarray:
        """
        Updates the model and computes the Wolfe probabilities
        :return: the Wolfe probabilities at observed locations, shape (num_points, )
        """
        # update model
        self.model_updaters[0].update(self.loop_state)

        # wolfe probabilities
        wolfe_probabilities = self.acquisition.acquisition_1.evaluate(self.loop_state.X_transformed)  # this must be pw
        return wolfe_probabilities[:, 0]
