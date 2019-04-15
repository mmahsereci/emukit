# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging


_log = logging.getLogger(__name__)


class WolfeConditions:
    def __init__(self, c1: float = 0., c2: float = 0.5, cw: float = 0.3):
        """
        The probabilistic Wolfe conditions.

        For parameters c1, c2 see: Wolfe. Convergence conditions for ascent methods. SIAM Review, pages 226–235, 1969.
        For parameter cw see: Mahsereci & Hennig, Probabilistic Line Searches for Stochastic Optimization,
        NeurIPS vol. 28 (2015), pp. 181–189.

        :param c1: The parameter for the sufficient decrease condition (Armijo condition)
        :param c2: The parameter for the curvature conditions
        :param cw: Acceptance threshold for Wolfe conditions
        """
        self._check_parameter_validity(c1, c2, cw)
        self._c1 = c1
        self._c2 = c2
        self._cw = cw

    @property
    def c1(self):
        return self._c1

    @property
    def c2(self):
        return self._c2

    @property
    def cw(self):
        return self._cw

    def set_parameters(self, c1: float = None, c2: float = None, cw: float = None) -> None:
        """
        Sets new Wolfe parameters and Wolfe threshold and checks their validity.

        :param c1: The parameter for the sufficient decrease condition (Armijo condition)
        :param c2: The parameter for the curvature conditions
        :param cw: Acceptance threshold for Wolfe conditions
        """
        if c1 is None:
            c1 = self._c1
        if c2 is None:
            c2 = self._c2
        if cw is None:
            cw = self._cw

        self._check_parameter_validity(c1, c2, cw)
        self._c1 = c1
        self._c2 = c2
        self._cw = cw

    @staticmethod
    def _check_parameter_validity(c1: float, c2: float, cw: float) -> None:
        """
        Checks the validity of the wolfe parameters

        :param c1: The parameter for the sufficient decrease condition (Armijo condition)
        :param c2: The parameter for the curvature conditions
        :param cw: Acceptance threshold for Wolfe conditions
        """

        # check if all numbers are in (half-)open interval between 0 and 1.
        if not 0. <= c1 < 1.:
            raise ValueError('Wolfe parameter c1 must be in half-open interval [0, 1). Given value is ' + str(c1) + '.')
        if not 0. < c2 <= 1.:
            raise ValueError('Wolfe parameter c2 must be in half-open interval (0, 1]. Given value is ' + str(c2) + '.')
        if not 0. < cw < 1.:
            raise ValueError('Wolfe threshold cw must be in open interval (0, 1). Given value is ' + str(cw) + '.')

        # check if c1 and c2 are such that for convex functions, a Wolfe point can be found.
        if c1 >= c2:
            raise ValueError('Wolfe parameter c2 must be larger than parameter c1. Given values are ' + str(c1)
                             + ' and ' + str(c2) + ' (c1 and c2 respectively).')
        if 0. < cw < 0.3:
            _log.warning('Wolfe threshold cw is less than 0.3 (' + str(cw) + '). Although this might not necessarily '
                         'cause problems, the recommended range is the interval [0.3, 0.8] with a default of 0.3.')
        if 0.8 < cw < 1:
            _log.warning('Wolfe threshold cw is greater than 0.8 (' + str(cw) + '). Although this might not '
                         'necessarily cause problems, the line search might be unnecessarily inefficient. The '
                         'recommended range is the open interval [0.3, 0.8] with a default of 0.3.')

