# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List
from scipy.linalg import lu_solve, lu_factor
from scipy.special import erf

from ...core.interfaces import IModel


class INoisyModelWithGradients:
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

    def predict_with_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance of function values at X, both shapes (n_points, 1), as well
        mean of gradients and their variances at X, both shapes (n_points x n_dim)
        """
        raise NotImplementedError

    def set_data(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: np.ndarray, vardY: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: noisy function values at new points X
        :param dY: noisy gradients  at new points X
        :param varY: variances of Y
        :param vardY: variances of dY
        """
        raise NotImplementedError

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError

    @property
    def dY(self):
        raise NotImplementedError

    @property
    def varY(self):
        raise NotImplementedError

    @property
    def vardY(self):
        raise NotImplementedError


class CubicSplineGP_old(IModel):
    """A once integrated Wiener process (cubic spline Gaussian process)."""


    # ==================================
    # Note: cubic min starts here
    # Todo: this need to move to acqusition
    def expected_improvement(self, t):
        """Computes the expected improvement at position ``t`` under the current
        GP model.

        Reference "current best" is the observed ``t`` with minimal posterior
        mean."""

        assert isinstance(t, (float, np.float32, np.float64))

        # Find the observation with minimal posterior mean, if it has not yet been
        # computed by a previous call to this method
        if self.min_obs is None:
            self.min_obs = min(self.mu(tt) for tt in self.ts)

        # Compute posterior mean and variance at t
        m, v = self.mu(t), self.V(t)

        # Compute the two terms in the formula for EI and return the sum
        t1 = 0.5 * (self.min_obs - m) * (1 + erf((self.min_obs - m) / np.sqrt(2. * v)))
        t2 = np.sqrt(0.5 * v / np.pi) * np.exp(-0.5 * (self.min_obs - m) ** 2 / v)

        return t1 + t2


class CubicSplineGP(IModel):
    """A once integrated Wiener process (cubic spline Gaussian process)."""

    def __init__(self, T: np.ndarray, Y: np.ndarray, dY: np.ndarray, sigma2f: float,
                 sigma2df: float, offset: float = 10.0):
        """
        All values are in the scaled space.

        :param T: 1D locations of datapoints (num_dat, 1)
        :param Y: noisy function value at starting point of line search
        :param dY: noisy gradient at starting point of line search (n_dim, 1)
        :param sigma2f: The variance of the noisy function values
        :param sigma2df: The variance of the noisy projected gradients
        :param offset: offset of the kernel to the left in inputs space
        """

        # Todo: we might need an IModel extension that can handle noise
        # kernel parameters
        self._offset = offset

        # value and gradient noise
        self._sigma2f = sigma2f
        self._sigma2df = sigma2df

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
        # Todo: why is N zero and not 1?
        self.N = 0
        self._T = []
        self._Y = []
        self._dY = []
        # Todo: rename those
        self._varY = []
        self._vardY = []

        # Switch that remembers whether we are ready for inference (calls to mu,
        # V, etc...). It is set to False when the GP is manipulated (points added,
        # noise level adjusted, reset). After such manipulations, gp.update() has
        # to be called. Remember current best observation of exp. improvement
        self.ready = False
        self.min_obs = None

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

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        # Todo: how does loop use predict? do we need it at all?
        raise NotImplementedError

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

        # Set up the kernel matrices.
        # Todo: since hypers do not change we can append.
        # Also loop need to go over j<=i only
        self._K = np.zeros([self.N, self.N])
        self._Kd = np.zeros([self.N, self.N])
        self._dKd = np.zeros([self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                self._K[i, j] = self.k(self._T[i], self._T[j])
                self._Kd[i, j] = self.kd(self._T[i], self._T[j])
                self._dKd[i, j] = self.dkd(self._T[i], self._T[j])

        # Put together the Gram matrix
        self._G = np.zeros((int(2 * self.N), int(2 * self.N)))
        self._G[:self.N, :self.N] = self._K + np.diag(self._varY)
        self._G[self.N:, self.N:] = self._dKd + np.diag(self._vardY)
        self._G[:self.N, self.N:] = self._Kd
        self._G[self.N:, :self.N] = self._Kd.T

        # Todo: remove this when not needed anymore
        # S_y = np.diag(self._varY)
        # S_dy = np.diag(self._vardY)
        # self._G = np.bmat([[self._K + S_y, self._Kd],
        #                    [self._Kd.T, self._dKd + S_dy]])

        # Compute the LU decomposition of G and store it
        self._LU, self._LU_piv = lu_factor(self._G, check_finite=True)

        # Todo: remove this when not needed anymore
        # Set ready switch to True
        self.ready = True

        # Pre-compute the regression weights used in mu
        self._w = self.solve_G(np.array(self._Y + self._dY))

    def add_single_datapoint(self, t: float, f: float, df: float, vary=0.0, vardy=0.0) -> None:
        """Add a new observation to the GP.

        This stores the observation internally, but does NOT yet set up and invert
        the Gram matrix. Add observations with repeated calls to this method, then
        call ``gp.update()`` to set up and invert the Gram matrix. Only then you
        can perform inference (calls to ``gp.mu(t)``, ``gp.V(t)``, etc...)."""

        self.ready = False
        self.min_obs = None

        # Todo: why is vary, vardy default 0?
        self.N += 1
        self._T.append(t)
        self._Y.append(f)
        self._dY.append(df)
        self._varY.append(vary)
        self._vardY.append(vardy)

        # Todo: can we juts update the GP here?
        # Note: I added that
        self._update()

    # =============================================
    # Note: means start here
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

    # =============================================
    # Note: covariances start here
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

    # =============================================
    # Note: inference helpers start here
    def solve_G(self, b):
        """
        Solve ``Gx=b`` where ``G`` is the Gram matrix of the GP.

        Uses the internally-stored LU decomposition of ``G`` computed in
        ``gp.update()``.
        """
        assert self.ready
        return lu_solve((self._LU, self._LU_piv), b, check_finite=True)
