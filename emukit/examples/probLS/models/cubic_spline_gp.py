# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import solve_triangular, lu_factor, lu_solve, solve
from typing import Tuple, Union

from ..interfaces.models import IModelWithObservedGradientsAndNoise


class CubicSplineGP(IModelWithObservedGradientsAndNoise):
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

    def set_data(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: float = None, vardY: float = None) -> None:
        """
        Sets training data in model

        :param X: new points (num_points, 1)
        :param Y: noisy function values at new points X, (num_points, 1)
        :param dY: noisy gradients  at new points X, (num_points, 1)
        :param varY: The variance of the noisy function values. This is sigmaf^2 in the original paper.
        :param vardY: The variance of the noisy projected gradients. This is sigmadf^2 in the original paper.
        """

        self._T = X
        self._Y = Y
        self._dY = dY
        self.N = X.shape[0]

        if varY is not None:
            if not isinstance(varY, float):
                raise TypeError('varY must be float. ', type(varY), ' given.')
            self._varY = np.array(self.N * [varY])
        if vardY is not None:
            if not isinstance(vardY, float):
                raise TypeError('vardY must be float. ', type(vardY), ' given.')
            self._vardY = np.array(self.N * [vardY])

        if self._varY is None:
            raise ValueError("varY is not provided.")

        if self._vardY is None:
            raise ValueError("vardY is not provided.")

        self._update()

    def get_index_of_lowest_observed_mean(self) -> int:
        """
        Computes the index of the datapoint which has the lowest GP posterior mean
        :return: Index corresponding to tthe lowest observed GP mean. The initial evaluation is excluded.
        """
        M = np.array([self.m(t) for t in self.X[1:, 0]])
        idx_min = np.argmin(M) + 1
        return int(idx_min)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance if function values for given points

        :param X: array of shape (n_points x 1) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x 1)
        """
        means = np.zeros(X.shape)
        variances = np.zeros(X.shape)
        for i in range(X.shape[0]):
            means[i, 0], variances[i, :] = self.m(X[i, 0]), self.V(X[i, 0])

        return means, variances

    def predict_with_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance of function values at X, both shapes (n_points, 1), as well
        mean of gradients and their variances at X, both shapes (n_points x n_dim)
        """
        raise NotImplementedError

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
            self._K[i, :] = self.k(self._T[i, 0], self._T)
            self._Kd[i, :] = self.kd(self._T[i, 0], self._T)
            self._dKd[i, :] = self.dkd(self._T[i, 0], self._T)
        self._K = 0.5 * (self._K + self._K.T)
        self._dKd = 0.5 * (self._dKd + self._dKd.T)

        # Gram matrix
        self._G = np.zeros((int(2 * N), int(2 * N)))
        self._G[:N, :N] = self._K + np.diag(self._varY)
        self._G[N:, N:] = self._dKd + np.diag(self._vardY)
        self._G[:N, N:] = self._Kd
        self._G[N:, :N] = self._Kd.T
        self._G = 0.5 * (self._G + self._G.T)

        # Gram decomposition and weights
        #self._LU, self._LU_piv = lu_factor(self._G, check_finite=True)
        self._A = self._solve_gram(np.vstack([self._Y, self._dY]).squeeze())  # shape (2 N,)

    def _solve_gram(self, b: np.ndarray) -> np.ndarray:
        """
        Solves linear system G x = b by using the precomputed Cholesky of G
        :param b: right side of linear equation (vector)
        :return: solution x of linear system
        """
        return solve(self._G, b)
        #return lu_solve((self._LU, self._LU_piv), b, check_finite=True)

    # === posterior means start here ===
    def m(self, t: float) -> float:
        """
        The marginal posterior mean of function values
        :param t: location where mean is evaluated
        :return: the posterior mean at t
        """
        kvec = np.hstack([self.k(t, self._T), self.kd(t, self._T)]).squeeze()
        return (kvec * self._A).sum()

    def d1m(self, t: float) -> float:
        """
        First derivative of posterior mean
        :param t: location where mean is evaluated
        :return: First derivative of posterior mean at t
        """
        kvec = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        return (kvec * self._A).sum()

    def d2m(self, t: float) -> float:
        """
        Second derivative of posterior mean
        :param t: location where mean is evaluated
        :return: Second derivative of posterior mean at t
        """
        kvec = np.hstack([self.ddk(t, self._T), self.ddkd(t, self._T)]).squeeze()
        return (kvec * self._A).sum()

    def d3m(self, t: float) -> float:
        """
        Third derivative of posterior mean
        :param t: location where mean is evaluated
        :return: Third derivative of posterior mean at t
        """
        kvec = np.hstack([self.dddk(t, self._T), np.zeros((1, self.N))]).squeeze()
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
        result = ab_min ** 3 / 3.0 + 0.5 * np.absolute(a - b) * ab_min ** 2
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
        result = np.where(b > a, 0.5 * aa ** 2, aa * bb - 0.5 * bb ** 2)
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
    def V(self, t: float) -> float:
        """
        posterior variance of function value at t
        :param t: location where variance is computed
        :return: posterior variance
        """
        kvec = np.hstack([self.k(t, self._T), self.kd(t, self._T)]).squeeze()
        var_reduction = (kvec * self._solve_gram(kvec)).sum()
        return self.k(t, t) - var_reduction

    def Vd(self, t: float) -> float:
        """
        posterior covariance between function value and gradient at t
        :param t: location where covariance is computed
        :return: posterior covariance
        """
        kvec_left = np.hstack([self.k(t, self._T), self.kd(t, self._T)]).squeeze()
        kvec_right = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        cov_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.kd(t, t) - cov_reduction

    def dVd(self, t: float) -> float:
        """
        posterior variance of gradient at t
        :param t: location where variance is computed
        :return: posterior variance
        """
        kvec_left = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        kvec_right = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        var_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.dkd(t, t) - var_reduction

    def V0f(self, t: float) -> float:
        """
        posterior covariance of function value at t=0 and t
        :param t: location where covariance is computed
        :return: posterior covariance
        """
        kvec_left = np.hstack([self.k(0., self._T), self.kd(0., self._T)]).squeeze()
        kvec_right = np.hstack([self.k(t, self._T), self.kd(t, self._T)]).squeeze()
        cov_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.k(0, t) - cov_reduction

    def Vd0f(self, t: float) -> float:
        """
        posterior covariance of gradient at t=0 and function value t
        :param t: location where covariance is computed
        :return: posterior covariance
        """
        kvec_left = np.hstack([self.dk(0., self._T), self.dkd(0., self._T)]).squeeze()
        kvec_right = np.hstack([self.k(t, self._T), self.kd(t, self._T)]).squeeze()
        cov_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.dk(0, t) - cov_reduction

    def V0df(self, t: float) -> float:
        """
        posterior covariance of function value at t=0 and gradient t
        :param t: location where covariance is computed
        :return: posterior covariance
        """
        kvec_left = np.hstack([self.k(0., self._T), self.kd(0., self._T)]).squeeze()
        kvec_right = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        cov_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.kd(0, t) - cov_reduction

    def Vd0df(self, t: float) -> float:
        """
        posterior covariance gradients at t=0 and gradient t
        :param t: location where covariance is computed
        :return: posterior covariance
        """
        kvec_left = np.hstack([self.dk(0., self._T), self.dkd(0., self._T)]).squeeze()
        kvec_right = np.hstack([self.dk(t, self._T), self.dkd(t, self._T)]).squeeze()
        cov_reduction = (kvec_left * self._solve_gram(kvec_right)).sum()
        return self.dkd(0, t) - cov_reduction
