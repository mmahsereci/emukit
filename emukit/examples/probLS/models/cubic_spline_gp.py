# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lu_solve, lu_factor, solve_triangular
from typing import Tuple, Union

from ..interfaces.models import INoisyModelWithGradients


class CubicSplineGP(INoisyModelWithGradients):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: float, vardY: float, offset: float = 10.0):
        """
        All values are in the scaled space.

        :param X: 1D locations of datapoints (num_dat, 1). The first entry must be the current location of the optimizer
        :param Y: noisy function value at starting point of line search (num_dat, 1)
        :param dY: noisy gradient at starting point of line search (num_dat, 1)
        :param varY: The variance of the noisy function values
        :param vardY: The variance of the noisy projected gradients
        :param offset: offset of the kernel to the left in inputs space
        """

        # kernel parameters
        self._offset = offset

        # Kernel matrices
        self._K = None
        self._Kd = None
        self._dKd = None

        # Gram matrix and pre-computed weights of the GP
        self._G = None
        self._A = None
        self._L = None

        # Observation counter and arrays to store observations
        self.N = X.shape[0]
        self._T = X
        self._Y = Y
        self._dY = dY

        # Note: this is sigma squared in the original paper
        if not isinstance(varY, float):
            raise TypeError('varY must be float. ', type(varY), ' given.')
        if not isinstance(vardY, float):
            raise TypeError('vardY must be float. ', type(vardY), ' given.')

        self._varY = np.array(self.N * [varY])
        self._vardY = np.array(self.N * [vardY])

        self._update()

    @property
    def X(self):
        return self._T

    @property
    def Y(self):
        return self._Y

    @property
    def dY(self):
        return self._dY

    @property
    def varY(self):
        return self._varY

    @property
    def vardY(self):
        return self._vardY

    def optimize(self) -> None:
        """The line search does not optimize its hyper, but sets them once at the beginning of the loop."""
        pass

    def set_data(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: float, vardY: float) -> None:
        """
        Sets training data in model

        :param X: new points (num_points, 1)
        :param Y: noisy function values at new points X, (num_points, 1)
        :param dY: noisy gradients  at new points X, (num_points, 1)
        :param varY: The variance of the noisy function values
        :param vardY: The variance of the noisy projected gradients
        """

        self._T = X
        self._Y = Y
        self._dY = dY
        self.N = X.shape[0]

        # Note: this is sigma squared in the original paper
        if not isinstance(varY, float):
            raise TypeError('varY must be float. ', type(varY), ' given.')
        if not isinstance(vardY, float):
            raise TypeError('vardY must be float. ', type(vardY), ' given.')

        self._varY = np.array(self.N * [varY])
        self._vardY = np.array(self.N * [vardY])

        self._update()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance if function values for given points

        :param X: array of shape (n_points x 1) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x 1)
        """
        # Todo: this predicts the marginal of function values. Do we also need derivatives?
        means = np.zeros(X.shape)
        variances = np.zeros(X.shape)
        for i in range(X.shape[0]):
            means[i, 0], variances[i, :] = self.m(X[i, 0]), self.V(X[i, 0])

        return means, variances

    def _update(self):
        """
        Computes and stores all needed matrices for the posterior: kernel matrices, Gram matrix, Cholesky, and weights.
        """

        N = self.N

        # Kernel matrices
        self._K = np.zeros((N, N))
        self._Kd = np.zeros((N, N))
        self._dKd = np.zeros((N, N))
        for i in range(N):
            self._K[i, :] = self.k(self._T[i], self._T)
            self._Kd[i, :] = self.kd(self._T[i], self._T)
            self._dKd[i, :] = self.dkd(self._T[i], self._T)
        self._K = 0.5 * (self._K + self._K.T)
        self._dKd = 0.5 * (self._dKd + self._dKd.T)

        # Gram matrix
        self._G = np.zeros((int(2 * N), int(2 * N)))
        self._G[:N, :N] = self._K + np.diag(self._varY)
        self._G[N:, N:] = self._dKd + np.diag(self._vardY)
        self._G[:N, N:] = self._Kd
        self._G[N:, :N] = self._Kd.T
        self._G = 0.5 * (self._G + self._G.T)

        # Cholesky and weights
        self._L = np.linalg.cholesky(self._G)

        tmp = solve_triangular(self._L, np.vstack([self._Y, self._dY])[:, 0], lower=True)
        self._A = solve_triangular(self._L.T, tmp, lower=False)  # shape (2 N,)

    # === posterior means start here ===
    def m(self, t: float) -> float:
        """
        The marginal posterior mean of function values
        :param t: location where mean is evaluated
        :return: the posterior mean at t
        """
        kvec = np.hstack([self.k(t, self._T), self.kd(t, self._T)])[:, 0]
        return (kvec * self._A).sum()

    def dm(self, t: float) -> float:
        """
        First derivative of posterior mean
        :param t: location where mean is evaluated
        :return: First derivative of posterior mean at t
        """
        """Evaluate first derivative of the posterior mean of df at ``t``."""
        kvec = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)])[:, 0]
        return (kvec * self._A).sum()

    def d2m(self, t: float) -> float:
        """
        Second derivative of posterior mean
        :param t: location where mean is evaluated
        :return: Second derivative of posterior mean at t
        """
        """Evaluate 2nd derivative of the posterior mean of f at ``t``."""
        kvec = np.hstack([self.ddk(t, self._T), self.ddkd(t, self._T)])[:, 0]
        return (kvec * self._A).sum()

    def d3m(self, t: float) -> float:
        """
        Third derivative of posterior mean
        :param t: location where mean is evaluated
        :return: Third derivative of posterior mean at t
        """
        kvec = np.hstack([self.dddk(t, self._T), np.zeros((1, self.N))])[:, 0]
        return (kvec * self._A).sum()

    # === prior covariances start here ===
    def k(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        The kernel function
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix k(a, b), array with shape (1, num_points), or float
        """
        ab_min = self._offset + np.minimum(a, b)
        result = ab_min ** 3 / 3.0 + 0.5 * np.abs(a - b) * ab_min ** 2
        return result.T

    def kd(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function once derived w.r.t. second argument.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d/db k(a, b), array with shape (1, num_points), or float
        """
        aa = a + self._offset
        bb = b + self._offset
        result = np.where(a < b, 0.5 * aa ** 2, aa * bb - 0.5 * bb ** 2)
        return result.T

    def dk(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function once derived w.r.t. first argument.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d/da k(a, b), array with shape (1, num_points), or float
        """
        aa = a + self._offset
        bb = b + self._offset
        result = np.where(a > b, 0.5 * bb ** 2, aa * bb - 0.5 * aa ** 2)
        return result.T

    def dkd(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function once derived w.r.t. both arguments.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d^2/dadb k(a, b), array with shape (1, num_points), or float
        """
        aa = a + self._offset
        bb = b + self._offset
        return np.minimum(aa, bb).T

    def ddk(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function twice derived w.r.t. first argument.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d^2/da^2 k(a, b), array with shape (1, num_points), or float
        """
        return np.where(a <= b, b - a, 0.).T

    def dddk(self, a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function thrice derived w.r.t. first argument.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d^3/da^3 k(a, b), array with shape (1, num_points), or float
        """
        return np.where(a <= b, -1., 0.).T

    def ddkd(self,  a: float, b: Union[np.ndarray, np.float]) -> Union[np.ndarray, np.float]:
        """
        kernel function twice derived w.r.t. first argument, and once derived w.r.t. second argument.
        :param a: left argument of kernel function
        :param b: right argument of kernel function, array with shape (num_points, 1), or float
        :return: kernel matrix d^3/da^2db k(a, b), array with shape (1, num_points), or float
        """
        return np.where(a <= b, 1., 0.).T

    # === posterior covariances start here ===
    def V(self, t: float):
        """Posterior variance of f at ``t``."""
        # Compute the needed k vector
        assert self.ready
        T = np.array(self._T)
        kvec = np.concatenate([self.k(t, T), self.kd(t, T)])
        ktt = self.k(t, t)

        return ktt - np.dot(kvec, self.solve_G(kvec))

    def Vd(self, t: float):
        """Posterior co-variance of f and df at ``t``."""
        assert self.ready
        T = np.array(self._T)
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
        assert self.ready
        T = np.array(self._T)
        dkdtt = self.dkd(t, t)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kvec = np.concatenate([dktT, dkdtT])

        return dkdtt - np.dot(kvec, self.solve_G(kvec))

    def Cov_0(self, t: float) -> float:
        """
        :return: co-variance cov(f(0), f(t)).
        """
        assert self.ready
        T = np.array(self._T)
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
        assert self.ready
        T = np.array(self._T)
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
        assert self.ready
        T = np.array(self._T)
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
        assert self.ready
        T = np.array(self._T)
        dkd0t = self.dkd(0., t)
        dk0T = self.kd(T, 0.)
        dkd0T = self.dkd(0., T)
        dktT = self.kd(T, t)
        dkdtT = self.dkd(t, T)
        kvec_a = np.concatenate([dk0T, dkd0T])
        kvec_b = np.concatenate([dktT, dkdtT])

        return dkd0t - np.dot(kvec_a, self.solve_G(kvec_b))
