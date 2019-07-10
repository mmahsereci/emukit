# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple
from scipy.linalg import lapack

from emukit.quadrature.methods.warped_bq_model import WarpedBayesianQuadratureModel
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBF


class Masha(WarpedBayesianQuadratureModel):
    """
    Base class for WSABI, Gunter et al. 2014

    WSABI must be used with the RBF kernel.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, f_star: float):
        """
        :param base_gp: a model derived from BaseGaussianProcess with QuadratureRBF quadrature kernel
        """
        if not isinstance(base_gp.kern, QuadratureRBF):
            raise ValueError("WSABI can only be used with quadrature kernel which are instances of  QuadratureRBF, ",
                             base_gp.kern.__class__.__name__, " given instead.")

        self.f_star = f_star

        super(Masha, self).__init__(base_gp)

    def transform(self, Y):
        """ Transform from base-GP to integrand """
        return self.f_star - 0.5 * (Y ** 2)

    def inverse_transform(self, Y):
        """ Transform from integrand to base-GP """
        return np.sqrt(np.absolute(2. * (self.f_star - Y)))

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """
        :param A: a square matrix, shape (N, N)
        :return: the symmetrized matrix 0.5 (A + A')
        """
        return 0.5 * (A + A.T)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.f_star - 0.5 * (mean_base**2)
        var_approx = (mean_base * mean_base) * var_base

        return mean_approx, var_approx, mean_base, var_base

    def predict_base_with_full_covariance(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray]:
        """
        Computes predictive means and covariance of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict, shape (n_points, input_dim)
        :returns: predictive mean and covariance of warped GP, predictive mean and covariance of base-GP in that order.
        mean shapes both (n_points, 1) and covariance shapes both (n_points, n_points)
        """
        mean_base, cov_base = self.base_gp.predict_with_full_covariance(X_pred)

        mean_approx = self.f_star - 0.5 * (mean_base**2)
        cov_approx = np.outer(mean_base, mean_base) * cov_base
        cov_approx = self._symmetrize(cov_approx)  # for numerical stability

        return mean_approx, cov_approx, mean_base, cov_base

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :returns: estimator of integral and its variance
        """
        N, D = self.X.shape

        # weights and kernel
        X = self.X / np.sqrt(2)
        K = self.base_gp.kern.K(X, X)
        weights = self.base_gp.graminv_residual()

        # integral of scaled kernel
        X_sums = 0.5 * (self.X.T[:, :, None] + self.X.T[:, None, :])
        X_sums_vec = X_sums.reshape(D, -1).T

        lengthscale_factor = 1./np.sqrt(2)
        qK_vec = self.base_gp.kern.qK(X_sums_vec, lengthscale_factor=lengthscale_factor)
        qK = qK_vec.reshape(N, N)

        # integral mean
        integral_mean = self.f_star - 0.5 * np.sum(np.outer(weights, weights) * qK * K)

        # integral variance
        qK_weights = np.dot(qK, weights)  # 1 x N
        lower_chol = self.base_gp.gram_chol()
        gram_inv_qK_weights = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, qK_weights.T, lower=1)[0]),
                                            lower=0)[0]

        second_term = np.dot(qK_weights, gram_inv_qK_weights)

        integral_variance = second_term[0, 0]

        return float(integral_mean), integral_variance
