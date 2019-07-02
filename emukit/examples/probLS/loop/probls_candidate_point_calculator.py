# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List, Union

from emukit.core.loop import CandidatePointCalculator
from ...probLS.loop.probls_loop_state import ProbLSLoopState
from emukit.core.acquisition.acquisition import Product
from ...probLS.models.cubic_spline_gp import CubicSplineGP


class ProbLSCandidatePointCalculator(CandidatePointCalculator):
    """ Computes the next point for function evaluation """

    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: Product, extrapolation_factor: float = 1.) -> None:
        """
        :param acquisition: The acquisition function, product of Wolfe probability and noisy EI
        """
        self.acquisition = acquisition
        self._extrapolation_factor = extrapolation_factor

    @property
    def model(self) -> CubicSplineGP:
        """The GP Model"""
        return self.acquisition.acquisition_1.model

    @property
    def extrapolation_factor(self) -> float:
        """The extrapolation factor"""
        return self._extrapolation_factor

    def _increase_extrapolation_factor(self, factor: float = 2.) -> None:
        """
        Increase the extrapolation factor by a multiplicative factor
        :param factor: The multiplicative factor
        :return:
        """
        if factor < 1.:
            raise ValueError("Factor can not be smaller than 1., ", factor, " given.")

        self._extrapolation_factor = factor * self._extrapolation_factor

    def compute_next_points(self, loop_state: ProbLSLoopState, context: dict=None) -> Tuple[float, bool]:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: will be ignored
        :return: next learning rate to evaluate the function at, and bool of we were not sucessfull
        """
        cand_minima, success = self._compute_cubic_minima_candidates()

        if not success:
            return cand_minima[0], True

        cand_extrapolation = self._compute_extrapolation_candidate()

        # wolfe probability for all candidates
        candidates = np.array(cand_minima + cand_extrapolation)
        pwei_candidates = self.acquisition.evaluate(candidates[:, np.newaxis])

        # get the point with the highest acquisition value
        idx_max = np.argmax(pwei_candidates)
        tt_max = candidates[idx_max]

        if tt_max == cand_extrapolation[0]:
            self._increase_extrapolation_factor()

        return tt_max, False

    def _compute_extrapolation_candidate(self) -> List[float]:
        """
        Compute the extrapolation candidate
        :return: List containing the location of the extrapolation candidate
        """
        tt = self.model.X.max() + self._extrapolation_factor
        return [tt]

    def _compute_cubic_minima_candidates(self) -> Tuple[List[float], bool]:
        """
        Find the local minimizers of the posterior mean.

        The posterior mean is a cubic polynomial in each of the cells [t_i, t_i+1] where the t_i are the sorted
        observed T. For each of these cells, return the minimizer of the cubic polynomial if it exists and happens to
        lie in that cell.
        :return: A list of candidates, bool if the search was successful
        """
        T_sorted = list(self.model.X[:, 0])
        T_sorted.sort()

        T_min = []
        M_min = []
        S_min = []

        for n, t_left, t_right in zip(range(len(T_sorted)), T_sorted, T_sorted[1:]):
            t_cubmin = self._compute_cubic_minimum_in_cell(t_left, t_right)

            if t_cubmin is not None:
                if t_left < t_cubmin < t_right:
                    T_min.append(t_cubmin)
                    M_min.append(self.model.m(t_cubmin))
                    S_min.append(self.model.V(t_cubmin))
            else:
                # most likely uphill?
                if (n == 0) and (self.model.d1m(0) > 0):
                    r = 0.01
                    tt = r * (t_left + t_right)
                    return [tt], False

        return T_min, True

    def _compute_cubic_minimum_in_cell(self, t_left: float, t_right: float) -> Union[float, None]:
        """
        Finds points where the derivative of the posterior mean equals ``val``
        and the second derivative is positive. To find minima, val=0.

        The posterior mean is a  cubic polynomial in each of the cells"
        ``[t_i, t_i+1]`` where the t_i are the sorted observed ts. For each of
        these cells, returns points with dmu==val the cubic polynomial if it exists
        and happens to lie in that cell.
        """

        t = t_left + 1e-6 * (t_right - t_left)
        d1mt = self.model.d1m(t)
        d2mt = self.model.d2m(t)
        d3mt = self.model.d3m(t)

        a = 0.5 * d3mt
        b = d2mt - t * d3mt
        c = d1mt - d2mt * t + 0.5 * d3mt * t **2

        # third derivative is almost zero -> essentially a quadratic, single extremum
        if abs(d3mt) < 1e-9:
            return -(d1mt - t * d2mt) / d2mt

        # roots are complex, no extremum
        lamb = b ** 2 - 4 * a * c
        if lamb < 0:
            return None

        # compute the two possible roots
        lr = (- b - np.sign(a) * np.sqrt(lamb)) / (2 * a)
        rr = (- b - np.sign(a) * np.sqrt(lamb)) / (2 * a)

        # distance to left and right root
        dtl = lr - t
        dtr = rr - t

        # left and right cubic value
        cvl = d1mt * dtl + 0.5 * d2mt * dtl **2 + (d3mt * dtl ** 3) / 6
        cvr = d1mt * dtr + 0.5 * d2mt * dtr **2 + (d3mt * dtr ** 3) / 6

        # find minimum of the two values
        if cvl < cvr:
            return lr
        else:
            return rr

