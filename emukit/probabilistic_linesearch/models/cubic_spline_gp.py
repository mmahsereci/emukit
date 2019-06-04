# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lu_solve, lu_factor
from typing import Tuple, Union

from ..interfaces.models import INoisyModelWithGradients


class CubicSplineGP(INoisyModelWithGradients):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: float,
                 vardY: float, offset: float = 10.0):
        """
        All values are in the scaled space.

        :param X: 1D locations of datapoints (num_dat, 1)
        :param Y: noisy function value at starting point of line search (num_dat, 1)
        :param dY: noisy gradient at starting point of line search (num_dat, 1)
        :param varY: The variance of the noisy function values (num_dat, 1) or positive scalar
        :param vardY: The variance of the noisy projected gradients, (num_dat, 1) or positive scalar
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
        self._w = None
        self._LU = None
        self._LU_piv = None

        # Observation counter and arrays to store observations
        self.N = X.shape[0]
        self._T = X
        self._Y = Y
        self._dY = dY

        # Note: this is sigmaf and sigmadf in the paper
        if isinstance(varY, float):
            self._varY = np.array([varY])
        elif isinstance(varY, np.ndarray):
            self._varY = varY
        else:
            raise TypeError

        if isinstance(vardY, float):
            self._vardY = np.array([vardY])
        elif isinstance(vardY, np.ndarray):
            self._vardY = vardY
        else:
            raise TypeError

        # Switch that remembers whether we are ready for inference (calls to mu,
        # V, etc...). It is set to False when the GP is manipulated (points added,
        # noise level adjusted, reset). After such manipulations, gp.update() has
        # to be called. Remember current best observation of exp. improvement
        self.ready = False

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
        """The line search does not optimize its hyper, but sets them once at the beginning."""
        pass

    def set_data(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: Union[np.ndarray, float],
                 vardY: Union[np.ndarray, float]) -> None:
        """
        Sets training data in model

        :param X: new points (num_points, 1)
        :param Y: noisy function values at new points X, (num_points, 1)
        :param dY: noisy gradients  at new points X, (num_points, 1)
        :param varY: variances of Y, array (num_points, 1), or positive scalar
        :param vardY: variances of dY, array (num_points, 1) or positive scalar
        """

        self._T = X
        self._Y = Y
        self._dY = dY
        # Note: this is sigma
        if isinstance(varY, float):
            self._varY = np.array([varY])
        elif isinstance(varY, np.ndarray):
            self._varY = varY
        else:
            raise TypeError

        if isinstance(vardY, float):
            self._vardY = np.array([vardY])
        elif isinstance(vardY, np.ndarray):
            self._vardY = vardY
        else:
            raise TypeError

        self.N = X.shape[0]

        # Todo: check later of we need those
        self.ready = False
        self._update()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance if function values for given points

        :param X: array of shape (n_points x 1) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x 1)
        """
        X = X.squeeze()
        mean = np.zeros((X.shape[0], 1))
        variance = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            mean[i, :], variance[i, :] = self.m(x), self.V(x)

        return mean, variance

    def _update_kernel_gram(self) -> None:
        """Computes the kernel matrix jointly for the 1d function values and 1d gradients"""
        # Todo: since hypers do not change we can append.
        # Set up the kernel matrices.
        # Also loop need to go over j<=i only

        N = self.N

        self._K = np.zeros((N, N))
        self._Kd = np.zeros((N, N))
        self._dKd = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self._K[i, j] = self.k(self._T[i], self._T[j])
                self._Kd[i, j] = self.kd(self._T[i], self._T[j])
                self._dKd[i, j] = self.dkd(self._T[i], self._T[j])

        # Put together the Gram matrix
        self._G = np.zeros((int(2 * N), int(2 * N)))
        self._G[:N, :N] = self._K + np.diag(self._varY)
        self._G[N:, N:] = self._dKd + np.diag(self._vardY)
        self._G[:N, N:] = self._Kd
        self._G[N:, :N] = self._Kd.T

    def _update(self):
        """Set up the Gram matrix and compute its LU decomposition to make the GP
        ready for inference (calls to ``.gp.mu(t)``, ``gp.V(t)``, etc...).

        Call this method after you have manipulated the GP by
           - ``gp.reset()`` ing,
           - adding observations with ``gp.add(t, f, df)``, or
           - adjusting the sigmas via ``gp.update_sigmas()``.
        and want to perform inference next."""

        if self.ready:
            return

        self._update_kernel_gram()  # update self._K and self._G

        # Compute the LU decomposition of G and store it
        self._LU, self._LU_piv = lu_factor(self._G, check_finite=True)

        # Todo: remove this when not needed anymore
        # Set ready switch to True
        self.ready = True

        # Pre-compute the regression weights used in mu
        self._w = self.solve_G(np.array(self._Y + self._dY))

    def solve_G(self, b):
        """
        Solve Gx=b where G is the Gram matrix of the GP.

        """
        assert self.ready
        return lu_solve((self._LU, self._LU_piv), b, check_finite=True)

    # === means start here ===
    # TODO: check the return types
    def m(self, t: float) -> float:
        """Posterior mean of f at ``t``."""
        # Compute kernel vector (k and kd) of the query t and the observations T
        # Then perform inner product with the pre-computed GP weights
        assert self.ready
        T = np.array(self._T)
        kvec = np.concatenate([self.k(t, T), self.kd(t, T)])

        return np.dot(self._w, kvec)[0]

    def dm(self, t: float) -> float:
        """Evaluate first derivative of the posterior mean of df at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        assert self.ready
        T = np.array(self._T)
        kvec = np.concatenate([self.kd(T, t), self.dkd(t, T)])

        return np.dot(self._w, kvec)[0]

    def d2m(self, t: float) -> float:
        """Evaluate 2nd derivative of the posterior mean of f at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        assert self.ready
        T = np.array(self._T)
        kvec = np.concatenate([self.ddk(t, T), self.ddkd(t, T)])

        return np.dot(self._w, kvec)[0]

    def d3m(self, t: float) -> float:
        """Evaluate 3rd derivative of the posterior mean of f at ``t``."""
        # Same is in mu, with the respective "derivative kernel vectors"
        assert self.ready
        T = np.array(self._T)
        kvec = np.concatenate([self.dddk(t, T), np.zeros(self.N)])

        return np.dot(self._w, kvec)[0]

    # === covariances start here ===
    def k(self, x, y):
        """
        Kernel function.
        """
        mi = self._offset + np.minimum(x, y)
        return mi ** 3 / 3.0 + 0.5 * np.abs(x - y) * mi ** 2

    def kd(self, x, y):
        """
        Derivative of kernel function, 1st derivative w.r.t. right argument.
        """
        xx = x + self._offset
        yy = y + self._offset
        return np.where(x < y, 0.5 * xx ** 2, xx * yy - 0.5 * yy ** 2)

    def dkd(self, x, y):
        """
        Derivative of kernel function,  1st derivative w.r.t. both arguments.
        """
        xx = x + self._offset
        yy = y + self._offset
        return np.minimum(xx, yy)

    def ddk(self, x, y):
        """
        Derivative of kernel function,  2nd derivative w.r.t. left argument.
        """
        return np.where(x < y, y - x, 0.)

    def dddk(self, x, y):
        """
        Derivative of kernel function,  3rd derivative w.r.t. left argument.
        """
        return np.where(x < y, -1., 0.)

    def ddkd(self, x, y):
        """Derivative of kernel function,  2nd derivative w.r.t. left argument,
        1st derivative w.r.t. right argument."""
        return np.where(x < y, 1., 0.)

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
