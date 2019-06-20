# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.stats import norm
from typing import Tuple

from emukit.core.acquisition import Acquisition
from ...probLS.models.cubic_spline_gp import CubicSplineGP
from ...probLS.loop.wolfe_conditions import WolfeConditions
from .bivariate_normal_integral import compute_bivariate_normal_integral


class EIPWAcquisition():
    """Product of EI and PW"""
    pass


class NoisyExpectedImprovement(Acquisition):
    """Expected improvement for noisy data. The difference to the standard expeceted improvoment is that the current
    best guess is the lowest mean predictor at observed locations instead of the observations itself.
    """

    def __init__(self, model: CubicSplineGP):
        """
        :param model: A Cubic spline Gaussian process model
        """
        self.model = model

    def evaluate(self, x: np.ndarray, current_best: float = None) -> np.ndarray:
        """
        Expected improvement at position x under the current GP model and noisy observations.

        :param x: scalar input location
        :param current_best: value the expected improvement uses to compare against. Default is the current best minimal
        posterior mean at observed locations x.
        :return: expected improvement at x
        """

        if current_best is None:
            current_best = min(self.model.m(tt) for tt in self.model.X.squeeze())

        # mean, variance = self.model.m(x), self.model.V(x)
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        z = (current_best - mean) / standard_deviation
        pdf = norm.pdf(z)
        cdf = norm.cdf(z)

        improvement = standard_deviation * (z * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Gradients not implemented for this acquisition function")

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


class WolfeProbability(Acquisition):

    def __init__(self, model: CubicSplineGP, wolfe_conditions: WolfeConditions = None):
        """
        This acquisition computes for a given input the probability that the Wolfe conditions are fulfilled.

        :param model: A Cubic spline Gaussian process model
        :param wolfe_conditions: The probabilistic Wolfe conditions
        """
        self.model = model
        if WolfeConditions is None:
            self._wolfe_conditions = WolfeConditions()
        else:
            self._wolfe_conditions = wolfe_conditions

    @property
    def wolfe_condition(self) -> WolfeConditions:
        """The probabilistic Wolfe conditions"""
        return self._wolfe_conditions

    @property
    def has_gradients(self) -> bool:
        """
        Whether acquisition value has analytical gradient calculation available.
        :return: True if gradients are available
        """
        return False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function.
        :param x: (n_points x 1) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        results = np.zeros(x.shape)
        for i, t in enumerate(x[:, 0]):
            results[i, 0] = self._evaluate_single_point(t)
        return results

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Gradients not implemented for this acquisition function")

    def _evaluate_single_point(self, t: float) -> float:
        c1 = self.wolfe_condition.c1
        c2 = self.wolfe_condition.c2

        m0 = self.model.m(0.)
        dm0 = self.model.dm(0.)
        V0 = self.model.V(0.)
        Vd0 = self.model.Vd(0.)
        dVd0 = self.model.dVd(0.)

        mt = self.model.m(t)
        dmt = self.model.dm(t)
        Vt = self.model.V(t)
        Vdt = self.model.Vd(t)
        dVdt = self.model.dVd(t)

        # marginal for Armijo condition
        ma = m0 - mt + c1 * t * dm0
        Vaa = V0 + (c1 * t) ** 2 * dVd0 + Vt + 2 * (c1 * t * (Vd0 - Vd0f(t)) - V0f(t))

        # marginal for curvature condition
        mb = dmt - c2 * dm0
        Vbb = c2 ^ 2 * dVd0 - 2 * c2 * Vd0df(t) + dVdt

        # covariance between conditions
        Vab = -c2 * (Vd0 + c1 * t * dVd0) + V0df(t) + c2 * Vd0f(t) + c1 * t * Vd0df(t) - Vd(t)

        # deterministic evaluations
        if (Vaa < 1e-9) and (Vbb < 1e-9):
            # Todo: is that bool or float?
            pwolfe = (ma >= 0) * (mb >= 0)
            return pwolfe

        # joint probability
        rho = Vab / np.sqrt(Vaa * Vbb)
        if Vaa <= 0. or Vbb <= 0.:
            pwolfe = 0
            return pwolfe
        x_low = -ma / np.sqrt(Vaa)
        x_up  = np.inf
        y_low = -mb / np.sqrt(Vbb)
        y_up = (2 * c2 * (abs(dm0) + 2 * np.sqrt(dVd0)) - mb) / np.sqrt(Vbb)
        pwolfe = compute_bivariate_normal_integral(x_low, x_up, y_low, y_up, rho)
        return pwolfe

    def _evaluate_single_point2(self, t: float) -> float:
        """
        Evaluates the acquisition function.
        :param t: point at which to calculate acquisition function values
        :return: acquisition function value at point t
        """

        # Compute mean and covariance matrix of the two Wolfe quantities a and b
        # (equations (11) to (13) in [1]).
        c1 = self.wolfe_condition.c1
        c2 = self.wolfe_condition.c2

        m0 = self.model.m(0.)
        dm0 = self.model.dm(0.)
        V0 = self.model.V(0.)
        Vd0 = self.model.Vd(0.)
        dVd0 = self.model.dVd(0.)

        mu = self.model.m(t)
        dm = self.model.dm(t)
        V = self.model.V(t)
        dVd = self.model.dVd(t)
        Cov0t = self.model.Cov_0(t)
        dCov0t = self.model.dCov_0(t)
        Covd0t = self.model.Covd_0(t)

        ma = m0 - mu + c1 * t * dm0
        # Todo: is dCov0t correct here?
        Vaa = V0 + dVd0 * (c1 * t) ** 2 + V + 2. * c1 * t * (Vd0 - dCov0t) - 2. * Cov0t
        mb = dm - c2 * dm0

        # Todo: this is wrong
        Vbb = c2 ** 2 * dVd0 - 2 * c2 + dVd

        # Very small variances can cause numerical problems. Safeguard against
        # this with a deterministic evaluation of the Wolfe conditions.
        if Vaa < 1e-9 or Vbb < 1e-9:
            pwolfe = 1. if ma >= 0. and mb >= 0. else 0.
            return pwolfe

        Vab = Covd0t + c1 * t * self.model.dCovd_0(t) - self.model.Vd(t)

        # Compute correlation factor and integration bounds for adjusted p_Wolfe
        # and return the result of the bivariate normal integral.
        rho = Vab / np.sqrt(Vaa * Vbb)
        al = -ma / np.sqrt(Vaa)
        bl = (self.df_lo - mb) / np.sqrt(Vbb)
        bu = (self.df_hi - mb) / np.sqrt(Vbb)
        pwolfe = compute_bivariate_normal_integral(al, np.inf, bl, np.inf, rho)
        return pwolfe
