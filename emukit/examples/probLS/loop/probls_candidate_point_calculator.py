# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List

from emukit.core.loop import CandidatePointCalculator
from ...probLS.loop.probls_loop_state import ProbLSLoopState
from ...probLS.acquisitions.probls_acqusitions import WolfeProbability
from ...probLS.models.cubic_spline_gp import CubicSplineGP


class ProbLSCandidatePointCalculator(CandidatePointCalculator):
    """ Computes the next point(s) for function evaluation """

    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: WolfeProbability) -> None:
        """
        :param acquisition: The acquisition function
        """
        # Todo: replace acq in init with union PW, PW+EI etc, and type above with Union[WP, EI+WP]
        self.acquisition = acquisition

    @property
    def model(self) -> CubicSplineGP:
        """The Model"""
        return self.acquisition.model

    def compute_next_points(self, loop_state: ProbLSLoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: will be ignored
        :return: (1 x n_dims) array of next inputs to evaluate the function at
        """
        candidates_minima = self.find_cubic_minima()
        candidates_extrapolation = loop_state.extrapolation_factor * max(self.model.T)
        candidates = np.array(candidates_minima + candidates_extrapolation)

        pwolfe_candidates = self.acquisition.evaluate(candidates)

        idx_max = np.argmax(pwolfe_candidates)
        pwolfe_max = pwolfe_candidates[idx_max]
        candidate_max = candidates[idx_max]



        # Todo: compute mins of mean and one extrapolation point, choose among those the one with best PW+EI
        # Todo: decide of to return x or alpha. Probably x is better for user function
        x = np.array([0.])
        return x

    # ==================================
    # Note: cubic min starts here
    def find_dmu_equal(self, val: float) -> List[float]:
        """
        Finds points where the derivative of the posterior mean equals ``val``
        and the second derivative is positive.

        The posterior mean is a  cubic polynomial in each of the cells"
        ``[t_i, t_i+1]`` where the t_i are the sorted observed ts. For each of
        these cells, returns points with dmu==val the cubic polynomial if it exists
        and happens to lie in that cell.
        """
        # We want to go through the observations from smallest to largest t
        T_sorted = list(self.model.X)
        T_sorted.sort()

        solutions = []

        for t_left, t_right in zip(T_sorted, T_sorted[1:]):
            # Compute the coefficients of the quadratic polynomial dmu/dt in this
            # cell, then call the function minimize_cubic to find the minimizer.
            # If there is one and it falls into the current cell, store it
            a, b, c = self.quadratic_polynomial_coefficients(t_left + 0.5 * (t_right - t_left))
            solutions_cell = self.quadratic_polynomial_solve(a, b, c, val)
            for s in solutions_cell:
                if t_left < s < t_right:
                    solutions.append(s)

        return solutions

    def find_cubic_minima(self) -> List[float]:
        """
        Find the local minimizers of the posterior mean.

        The posterior mean is a cubic polynomial in each of the cells [t_i, t_i+1] where the t_i are the sorted
        observed T. For each of these cells, return the minimizer of the cubic polynomial if it exists and happens to
        lie in that cell.
        """
        return self.find_dmu_equal(0.0)

    def cubic_polynomial_coefficients(self, t: float) -> Tuple[float, float, float, float]:
        """
        :return: the coefficients of the cubic polynomial (piece of the mean of the model) at t.
        """

        # Todo: asserts are bad in code
        assert t not in self.model.X  # at the observations, polynomial is ambiguous

        dmu, d2, d3 = self.model.d1m(t), self.model.d2m(t), self.model.d3m(t)
        a = d3 / 6.0
        b = 0.5 * d2 - 3 * a * t
        c = dmu - 3. * a * t ** 2 - 2. * b * t
        d = self.model.m(t) - a * t ** 3 - b * t ** 2 - c * t

        return a, b, c, d

    def quadratic_polynomial_coefficients(self, t: float) -> Tuple[float, float, float]:
        """
        :return: the coefficients of the quadratic polynomial (piece of the mean of the model) at t.
        """

        # Todo: asserts are bad in code
        assert t not in self.model.X  # at the observations, polynomial is ambiguous

        d1, d2, d3 = self.model.d1m(t), self.model.d2m(t), self.model.d3m(t)
        a = .5 * d3
        b = d2 - d3 * t
        c = d1 - d2 * t + 0.5 * d3 * t ** 2

        return a, b, c

    @staticmethod
    def quadratic_polynomial_solve(a: float, b: float, c: float, val: float) -> List:
        """
        Computes real solutions of f(t) = a*t**2 + b*t + c = val with f''(t)>0.
        :return: list of locations of minima of the quadratic polynomial
        """
        # Check if a is almost zero. If so, solve the remaining linear equation. Note
        # that we return only minimizers, i.e., solutions with f''(t) = b > 0
        if abs(a) < 1e-9:
            if b > 1e-9:
                return [(val - c) / b]
            else:
                return []

        # Compute the term under the square root in pq formula, if it is negative,
        # there is no real solution
        det = b ** 2 - 4. * a * (c - val)
        if det < 0:
            return []

        # Otherwise, compute the two roots
        s = np.sqrt(det)
        r1 = (-b - np.sign(a) * s) / (2. * a)
        r2 = (-b + np.sign(a) * s) / (2. * a)

        # Return the one with f''(t) = 2at + b > 0, or []
        if 2 * a * r1 + b > 0:
            return [r1]
        elif 2 * a * r2 + b > 0:
            return [r2]
        else:
            return []
