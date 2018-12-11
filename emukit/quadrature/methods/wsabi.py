# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple
from scipy.linalg import lapack

from emukit.quadrature.methods.warped_bq_model import WarpedBayesianQuadratureModel
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBF


class WSABI(WarpedBayesianQuadratureModel):
    """
    Base class for WSABI, Gunter et al. 2014

    WSABI must be used with the RBF kernel.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, adapt_offset: bool = False):
        """
        :param base_gp: a model derived from BaseGaussianProcess with QuadratureRBF quadrature kernel
        :param adapt_offset: if True the offset the offset will be updated after new datapoints have been collected,
        if False, the offset will be constant at 0. defaults to False. Offset refers to 'alpha' in Gunter et al.
        """
        if not isinstance(base_gp.kern, QuadratureRBF):
            raise ValueError("WSABI can only be used with quadrature kernel which are instances of  QuadratureRBF, ",
                             base_gp.kern.__class__.__name__, " given instead.")

        self.adapt_offset = adapt_offset
        if adapt_offset:  # TODO: must be zero if not integrated over prob measure. otherwise integral is infty
            self._compute_and_set_offset()
        else:
            self.offset = 0.

        super(WSABI, self).__init__(base_gp)

    # TODO: check how it can be called after function evaluations (currently it is not)
    # TODO: of this is called also the Y values need to be transformed again.
    def _compute_and_set_offset(self):
        """if adapted, it uses the value given in Gunter et al. 2014"""
        if self.adapt_offset:
            # get data before offset is changed
            Y = self.Y.copy()

            # compute and set the new offset. this will change the transformation
            minL = min(self.base_gp.Y)[0]  # TODO: check if this returns the correct thing
            self.offset = 0.8 * minL

            # need to reset data because the transformation changed with the  offset
            self.set_data(self.X, Y)

    def transform(self, Y):
        """ Transform from base-GP to integrand """
        return 0.5*(Y*Y) + self.offset

    def inverse_transform(self, Y):
        """ Transform from integrand to base-GP """
        return np.sqrt(np.absolute(2.*(Y - self.offset)))

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        """
        :param A: a square matrix, shape (N, N)
        :return: the symmetrized matrix 0.5 (A + A')
        """
        return 0.5 * (A + A.T)


class WSABIL(WSABI):
    """
    The WSABI-L Bayesian quadrature model (Gunter et al. 2014)
    WSABI-L must be used with the RBF kernel.

    WSABI-L approximates the integrand as follows:
    - squared transformation of gp-base-model (chi-squared).
    - linear expansion around the mean of the base-gp for fixed input location x.
    - Gaussian is defined by first and second moment of linear expansion.
    """
    def __init__(self, base_gp: IBaseGaussianProcess, adapt_offset: bool = False):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        :param adapt_offset: if True the offset the offset will be updated after new datapoints have been collected,
        if False, the offset will be constant at 0. defaults to False. Offset refers to 'alpha' in Gunter et al.
        """
        super(WSABIL, self).__init__(base_gp, adapt_offset)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.offset + 0.5 * (mean_base**2)
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

        mean_approx = self.offset + 0.5 * (mean_base**2)
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
        # TODO: need to multiply offset with integration domain of not a prop measure, otherwise 1.
        integral_mean = self.offset + 0.5 * np.sum(np.outer(weights, weights) * qK * K)

        # integral variance

        return float(integral_mean), 1.


class WSABIM(WSABI):
    """
    The WSABI-M Bayesian quadrature model (Gunter et al. 2014)
    WSABI-M must be used with the RBF kernel.

    WSABI-M approximates the integrand as follows:
    - squared transformation of gp-base-model (chi-squared).
    - Gaussian is defined by first and second moment of chi-square-distribution.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, adapt_offset: bool = False):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        :param adapt_offset: if True the offset the offset will be updated after new datapoints have been collected,
        if False, the offset will be constant at 0. defaults to False. Offset refers to 'alpha' in Gunter et al.
        """
        super(WSABIM, self).__init__(base_gp, adapt_offset)

    def predict_base(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes predictive means and variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :returns: predictive mean and variances of warped GP, and predictive mean and variances of base-GP in that order
        all shapes (n_points, 1).
        """
        mean_base, var_base = self.base_gp.predict(X_pred)

        mean_approx = self.offset + 0.5 * (mean_base**2 + var_base)
        var_approx = 0.5 * var_base**2. + (mean_base**2 * var_base)

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
        var_base = np.diag(cov_base)[:, np.newaxis]

        mean_approx = self.offset + 0.5 * (mean_base**2 + var_base)
        cov_approx = 0.5 * cov_base**2. + np.outer(mean_base, mean_base) * cov_base
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
        first_term = 0.5 * np.sum(np.outer(weights, weights) * qK * K)

        init_qvar = (self.base_gp.kern.variance * self.base_gp.kern.integral_bounds.get_area_of_integration_domain())[0]
        lower_chol = self.base_gp.gram_chol()
        gram_inv = lapack.dtrtrs(lower_chol.T, (lapack.dtrtrs(lower_chol, np.eye(lower_chol.shape[0]), lower=1)[0]),
                                 lower=0)[0]
        second_term = 0.5 * (init_qvar - np.sum(gram_inv * qK))

        # integral mean
        integral_mean = self.offset + first_term + second_term

        # integral variance

        return integral_mean, 1.
