# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List
from scipy.linalg import lu_solve

from ...core.interfaces import IModel


class CubicSplineGP(IModel):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, T: np.ndarray, Y: np.ndarray, dY: np.ndarray, sigmaf: float,
                 sigmadf: float):
        """
        All values are in the scaled space.

        :param T: locations of datapoints (num_dat, 1)
        :param Y: function value at starting point of line search
        :param dY: gradient at starting point of line search (n_dim, 1)
        :param sigmaf: The variance of the noisy function values
        :param sigmaf: The variance of the noisy projected gradients
        """
        # Todo: cubic splice GP needs to do the scaling such tah loop only sees the original values
        self.sigmaf = sigmaf
        self.sigmadf = sigmadf
        self.kernel = IntegratedWienerKernel()

        self._T = T
        self._Y = Y
        self._dY = dY

        # Kernel matrices
        self.K = None
        self.Kd = None
        self.dKd = None

        # Gram matrix and pre-computed "weights" of the GP
        self.G = None
        self.w = None

    @property
    def X(self):
        return self._T

    @property
    def Y(self):
        return self._Y

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        self._T = X
        self._Y = Y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

    def _infer(self):
        """Computes the weights of the Gaussian process."""
        pass

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError


class IntegratedWienerMean:

    # TODO: check the return types
    def mu(self, t: float) -> float:
        """Posterior mean of f at ``t``."""
        # Compute kernel vector (k and kd) of the query t and the observations T
        # Then perform inner product with the pre-computed GP weights
        T = np.array(self.ts)
        kvec = np.concatenate([self.k(t, T), self.kd(t, T)])

        return np.dot(self.w, kvec)[0]

    def dmu(self, t: float) -> float:
        """Evaluate first derivative of the posterior mean of df at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        T = np.array(self.ts)
        kvec = np.concatenate([self.kd(T, t), self.dkd(t, T)])

        return np.dot(self.w, kvec)[0]

    def d2mu(self, t: float) -> float:
        """Evaluate 2nd derivative of the posterior mean of f at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        T = np.array(self.ts)
        kvec = np.concatenate([self.d2k(t, T), self.d2kd(t, T)])

        return np.dot(self.w, kvec)[0]

    def d3mu(self, t: float) -> float:
        """Evaluate 3rd derivative of the posterior mean of f at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        T = np.array(self.ts)
        kvec = np.concatenate([self.d3k(t, T), np.zeros(self.N)])

        return np.dot(self.w, kvec)[0]

    def find_dmu_equal(self, val):
        """Finds points where the derivative of the posterior mean equals ``val``
        and the second derivative is positive.

        The posterior mean is a  cubic polynomial in each of the cells"
        ``[t_i, t_i+1]`` where the t_i are the sorted observed ts. For each of
        these cells, returns points with dmu==val the cubic polynomial if it exists
        and happens to lie in that cell."""

        # We want to go through the observations from smallest to largest t
        ts_sorted = list(self.ts)
        ts_sorted.sort()

        solutions = []

        for t1, t2 in zip(ts_sorted, ts_sorted[1:]):
            # Compute the coefficients of the quadratic polynomial dmu/dt in this
            # cell, then call the function minimize_cubic to find the minimizer.
            # If there is one and it falls into the current cell, store it
            a, b, c = self.quadratic_polynomial_coefficients(t1 + 0.5 * (t2 - t1))
            solutions_cell = quadratic_polynomial_solve(a, b, c, val)
            for s in solutions_cell:
                if s > t1 and s < t2:
                    solutions.append(s)

        return solutions

    def find_cubic_minima(self):
        """Find the local minimizers of the posterior mean.

        The posterior mean is a  cubic polynomial in each of the cells"
        [t_i, t_i+1] where the t_i are the sorted observed ts. For each of these
        cells, return the minimizer of the cubic polynomial if it exists and
        happens to lie in that cell."""

        return self.find_dmu_equal(0.0)


class IntegratedWienerKernel:

    def __init__(self, theta: float = 1.0, offset: float = 10.0):
        """
        The integrated Wiener kernel.

        :param theta: scale of the kernel
        :param offset: offset of the kernel to the left in inputs space
        """
        self.offset = offset
        self.theta = theta

    def solve_G(self, b):
        """
        Solve ``Gx=b`` where ``G`` is the Gram matrix of the GP.

        Uses the internally-stored LU decomposition of ``G`` computed in
        ``gp.update()``.
        """

        return lu_solve((self.LU, self.LU_piv), b, check_finite=True)

    def k(self, x, y):
        """
        Kernel function.
        """
        mi = self.offset + np.minimum(x, y)
        return self.theta ** 2 * (mi ** 3 / 3.0 + 0.5 * np.abs(x - y) * mi ** 2)

    def kd(self, x, y):
        """
        Derivative of kernel function, 1st derivative w.r.t. right argument.
        """
        xx = x + self.offset
        yy = y + self.offset
        return self.theta ** 2 * np.where(x < y, 0.5 * xx ** 2, xx * yy - 0.5 * yy ** 2)

    def dkd(self, x, y):
        """
        Derivative of kernel function,  1st derivative w.r.t. both arguments.
        """
        xx = x + self.offset
        yy = y + self.offset
        return self.theta ** 2 * np.minimum(xx, yy)

    def d2k(self, x, y):
        """
        Derivative of kernel function,  2nd derivative w.r.t. left argument.
        """
        return self.theta ** 2 * np.where(x < y, y - x, 0.)

    def d3k(self, x, y):
        """
        Derivative of kernel function,  3rd derivative w.r.t. left argument.
        """
        return self.theta ** 2 * np.where(x < y, -1., 0.)

    def d2kd(self, x, y):
        """Derivative of kernel function,  2nd derivative w.r.t. left argument,
        1st derivative w.r.t. right argument."""
        return self.theta ** 2 * np.where(x < y, 1., 0.)

    def V(self, t: float):
        """Posterior variance of f at ``t``."""
        # Compute the needed k vector
        T = np.array(self.ts)
        kvec = np.concatenate([self.k(t, T), self.kd(t, T)])
        ktt = self.k(t, t)

        return ktt - np.dot(kvec, self.solve_G(kvec))

    def Vd(self, t: float):
        """Posterior co-variance of f and df at ``t``."""
        T = np.array(self.ts)
        ktT = self.k(t, T)
        kdtT = self.kd(t, T)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kdtt = self.kd(t, t)
        kvec_a = np.concatenate([ktT, kdtT])
        kvec_b = np.concatenate([dktT, dkdtT])

        return kdtt - np.dot(kvec_a, self.solve_G(kvec_b))

    def dVd(self, t: float):
        """Posterior variance of df at ``t``"""
        T = np.array(self.ts)
        dkdtt = self.dkd(t, t)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kvec = np.concatenate([dktT, dkdtT])

        return dkdtt - np.dot(kvec, self.solve_G(kvec))

    def Cov_0(self, t: float):
        """Posterior co-variance of f at 0. and ``t``."""
        T = np.array(self.ts)
        k0t = self.k(0., t)
        k0T = self.k(0., T)
        kd0T = self.kd(0., T)
        ktT = self.k(t, T)
        kdtT = self.kd(t, T)
        kvec_a = np.concatenate([k0T, kd0T])
        kvec_b = np.concatenate([ktT, kdtT])

        return k0t - np.dot(kvec_a, self.solve_G(kvec_b))

    def Covd_0(self, t: float):
        """Posterior co-variance of f at 0. and df at ``t``."""
        # !!! I changed this in line_search new, Covd_0 <-> dCov_0
        T = np.array(self.ts)
        kd0t = self.kd(0., t)
        k0T = self.k(0., T)
        kd0T = self.kd(0., T)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kvec_a = np.concatenate([k0T, kd0T])
        kvec_b = np.concatenate([dktT, dkdtT])

        return kd0t - np.dot(kvec_a, self.solve_G(kvec_b))

    def dCov_0(self, t: float):
        """Posterior co-variance of df at 0. and f at ``t``."""
        # !!! I changed this in line_search new, Covd_0 <-> dCov_0
        T = np.array(self.ts)
        dk0t = self.kd(t, 0.)
        dk0T = self.kd(T, 0.)
        dkd0T = self.dkd(0., T)
        ktT = self.k(t, T)
        kdtT = self.kd(t, T)
        kvec_a = np.concatenate([dk0T, dkd0T])
        kvec_b = np.concatenate([ktT, kdtT])

        return dk0t - np.dot(kvec_a, self.solve_G(kvec_b))

    def dCovd_0(self, t: float):
        """Posterior co-variance of df at 0. and ``t``."""
        T = np.array(self.ts)
        dkd0t = self.dkd(0., t)
        dk0T = self.kd(T, 0.)
        dkd0T = self.dkd(0., T)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kvec_a = np.concatenate([dk0T, dkd0T])
        kvec_b = np.concatenate([dktT, dkdtT])

        return dkd0t - np.dot(kvec_a, self.solve_G(kvec_b))


def quadratic_polynomial_solve(a: float, b: float, c: float, val: float) -> List:
    """Computes *real* solutions of f(t) = a*t**2 + b*t + c = val with f''(t)>0.

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
