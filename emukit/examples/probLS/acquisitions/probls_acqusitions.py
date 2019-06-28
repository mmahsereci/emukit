# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.stats import norm
from typing import Tuple

from emukit.core.acquisition import Acquisition
from ...probLS.models.cubic_spline_gp import CubicSplineGP
from ...probLS.interfaces.models import IModelWithObservedGradientsAndNoise
from ...probLS.loop.probls_wolfe_conditions import WolfeConditions
from .bivariate_normal_integral import compute_bivariate_normal_integral


class NoisyExpectedImprovement(Acquisition):
    """Expected improvement for noisy data. The difference to the standard expected improvement is that the current
    best guess is the lowest mean predictor at observed locations instead of the observations itself.
    """

    def __init__(self, model: IModelWithObservedGradientsAndNoise):
        """
        :param model: The mode; an instance of IModelWithObservedGradientsAndNoise
        """
        self.model = model

    def evaluate(self, x: np.ndarray, current_best: float = None) -> np.ndarray:
        """
        Expected improvement at position x under the current GP model and noisy observations.

        :param x: scalar input location, shape (num_dat, 1)
        :param current_best: value the expected improvement uses to compare against. Default is the current best minimal
        posterior mean at observed locations X.
        :return: expected improvement at x
        """

        if current_best is None:
            # minimal value of GP mean at observed locations
            mean_obs, _ = self.model.predict(self.model.X)
            current_best = mean_obs.min()

        mean, variance = self.model.predict(x)
        variance = np.clip(variance, 0, np.inf)
        standard_deviation = np.sqrt(variance)

        z = (current_best - mean) / standard_deviation
        improvement = standard_deviation * (z * norm.cdf(z) + norm.pdf(z))
        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Gradients not implemented for this acquisition function")

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


class WolfeProbability(Acquisition):

    def __init__(self, model: CubicSplineGP, wolfe_conditions: WolfeConditions):
        """
        This acquisition computes for a given input the probability that the Wolfe conditions are fulfilled.

        :param model: A Cubic spline Gaussian process model
        :param wolfe_conditions: The probabilistic Wolfe conditions
        """
        self.model = model
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

        dm0 = self.model.d1m(0.)
        Vd0 = self.model.Vd(0.)
        dVd0 = self.model.dVd(0.)

        Vd0ft = self.model.Vd0f(t)
        Vd0dft = self.model.Vd0df(t)

        # marginal for Armijo condition
        ma = self.model.m(0.) - self.model.m(t) + c1 * t * dm0
        Vaa = self.model.V(0.) + (c1 * t) ** 2 * dVd0 + self.model.V(t) + 2 * (c1 * t * (Vd0 - Vd0ft)
                                                                               - self.model.V0f(t))

        # marginal for curvature condition
        mb = self.model.d1m(t) - c2 * dm0
        Vbb = c2 ** 2 * dVd0 - 2 * c2 * Vd0dft + self.model.dVd(t)

        # covariance between conditions
        Vab = - c2 * (Vd0 + c1 * t * dVd0) + c2 * Vd0ft + self.model.V0df(t) + c1 * t * Vd0dft - self.model.Vd(t)

        # very small variance -> deterministic evaluations
        if (Vaa < 1e-9) and (Vbb < 1e-9):
            pwolfe = int((ma >= 0) * (mb >= 0))
            return pwolfe

        # zero or negative variances (maybe sth went wrong?)
        if (Vaa <= 0) or (Vbb <= 0):
            return 0

        # joint probability (everything is alright)
        rho = Vab / np.sqrt(Vaa * Vbb)
        rho = np.clip(rho, -1, 1)  # numerical stability

        low_a = - ma / np.sqrt(Vaa)
        up_a = np.inf
        low_b = - mb / np.sqrt(Vbb)
        up_b = (2 * c2 * (abs(dm0) + 2 * np.sqrt(dVd0)) - mb) / np.sqrt(Vbb)

        pwolfe = compute_bivariate_normal_integral(low_a, up_a, low_b, up_b, rho)
        return pwolfe
