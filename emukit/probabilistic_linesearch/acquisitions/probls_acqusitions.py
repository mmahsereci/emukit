# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.stats import norm
from typing import List, Tuple

from ...core.acquisition import Acquisition
from ...probabilistic_linesearch.models import CubicSplineGP
from ...probabilistic_linesearch.loop import WolfeConditions


class WolfeProbability(Acquisition):

    def __init__(self, model: CubicSplineGP, wolfe_conditions: WolfeConditions = None):
        """
        This acquisition computes for a given input the probability that the Wolfe conditions are fulfilled.

        :param model: A Cubic spline Gaussian process model
        :param wolfe_conditions: The probabilistic Wolfe conditions
        """
        # Todo: add return types
        # Model is in scaled space
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
        Vbb = c2 ** 2 * dVd0 - 2 * c2 *
        + dVd

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


def compute_bivariate_normal_integral(x_low: float, x_upp: float, y_low: float, y_upp: float, rho: float) -> float:
    """
    Computes probabilities p(x_low < x < x_upp and y_low < y < y_upp) of bivariate Gaussian with correlation rho.

    :param x_low: 1st lower bound of the integral
    :param x_upp: 1st upper bound of the integral
    :param y_low: 2nd lower bound of the integral
    :param y_upp: 2nd upper bound of the integral
    :param rho: correlation of the 2D Gaussian
    :return: integral value
        """
    p = compute_bivariate_normal_integral_unbounded(x_low, y_low, rho) \
        - compute_bivariate_normal_integral_unbounded(x_upp, y_low, rho) \
        - compute_bivariate_normal_integral_unbounded(x_low, y_upp, rho) \
        + compute_bivariate_normal_integral_unbounded(x_upp, y_upp, rho)

    integral_value = max(0., min(p, 1.))
    return integral_value


def compute_bivariate_normal_integral_unbounded(x_low: float, y_low: float, rho: float) -> float:
    """
    Computes the probability p(x > x_low and y > y_low) of a bivariate Gaussian with correlation rho.

    :param x_low: 1st lower bound of the integral
    :param y_low: 2nd lower bound of the integral
    :param rho: correlation of the 2D Gaussian

    Ported from matlab code by Alan Genz:

    Alan Genz, Department of Mathematics
    Washington State University, Pullman, Wa 99164-3113
    Email : alangenz@wsu.edu

    Based on: Drezner, Z and G.O. Wesolowsky, (1989), On the computation of the bivariate normal integral, Journal of
    Statist. Comput. Simul. 35, pp. 101-107

    Original copyright note:

    Copyright (C) 2013, Alan Genz,  All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided the following conditions are met:
      1. Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
      2. Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in
         the documentation and/or other materials provided with the
         distribution.
      3. The contributor name(s) may not be used to endorse or promote
         products derived from this software without specific prior
         written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    # clip for numerical stability
    rho = np.clip(rho, -1, 1)

    # upper as well as lower integral bounds are infty.
    if np.isposinf(x_low) or np.isposinf(y_low):
        return 0.
    elif np.isneginf(x_low):
        # integral over whole domain
        if np.isneginf(y_low):
            return 1.
        else:
            return norm.cdf(-y_low)
    elif np.isneginf(y_low):
        return norm.cdf(-x_low)
    elif rho == 0:
        return norm.cdf(-x_low) * norm.cdf(-y_low)

    tp = 2 * np.pi
    h = x_low
    k = y_low
    hk = h * k
    bvn = 0.

    if np.abs(rho) < 0.3:
        w, x = GaussLegendreParams['6']
    elif np.abs(rho) < 0.75:
        w, x = GaussLegendreParams['12']
    else:
        w, x = GaussLegendreParams['20']

    w = np.tile(w, 2)
    x = np.concatenate((1 - x, 1 + x))

    if np.abs(rho) < 0.925:
        hs = 0.5 * (h * h + k * k)
        asr = 0.5 * np.arcsin(rho)
        sn = np.sin(asr * x)
        bvn = np.dot(w, np.exp((sn * hk - hs) / (1. - sn ** 2)))
        bvn = bvn * asr / tp + norm.cdf(-h) * norm.cdf(-k)
    else:
        if rho < 0.:
            k = -k
            hk = -hk
        if np.abs(rho) < 1:
            ass = 1 - rho ** 2
            a = np.sqrt(ass)
            bs = (h - k) ** 2
            asr = - 0.5 * (bs / ass + hk)
            c = (4. - hk) / 8.
            d = (12. - hk) / 80.
            if asr > -100.:
                bvn = a * np.exp(asr) * (1. - c * (bs - ass) * (1. - d * bs) / 3. + c * d * ass ** 2)
            if hk > -100.:
                b = np.sqrt(bs)
                sp = np.sqrt(tp) * norm.cdf(-b / a)
                bvn = bvn - np.exp(-0.5 * hk) * sp * b * (1. - c * bs * (1. - d * bs) / 3.)
            a = 0.5 * a
            xs = (a * x) ** 2
            asr = - 0.5 * (bs / xs + hk)
            idx = np.where(asr > -100)[0]  # find( asr > -100 )
            xs = xs[idx]
            sp = 1 + c * xs * (1. + 5. * d * xs)
            rs = np.sqrt(1. - xs)
            ep = np.exp(-0.5 * hk * xs / (1. + rs) ** 2) / rs
            bvn = (a * np.dot(np.exp(asr[idx]) * (sp - ep), w[idx]) - bvn) / tp
        if rho > 0:
            bvn = bvn + norm.cdf(-max(h, k))
        elif h >= k:
            bvn = -bvn
        else:
            if h < 0:
                L = norm.cdf(k) - norm.cdf(h)
            else:
                L = norm.cdf(-h) - norm.cdf(-k)
            bvn = L - bvn

    integral_value = max(0., min(1, bvn))
    return integral_value


# Gauss Legendre weights and points, key indicates n
GaussLegendreParams = {'6': (np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904]),
                             np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])),
                       '12': (np.array([.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                                        0.2031674267230659, 0.2334925365383547, 0.2491470458134029]),
                              np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                                        0.5873179542866171, 0.3678314989981802, 0.1252334085114692])),
                       '20': (np.array([.01761400713915212, .04060142980038694, .06267204833410906,
                                        .08327674157670475, 0.1019301198172404, 0.1181945319615184,
                                        0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                                        0.1527533871307259]),
                              np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                                        0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                                        0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                                        0.07652652113349733]))
                       }