# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.stats import norm


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
