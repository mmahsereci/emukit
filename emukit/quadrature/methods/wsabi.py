# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple
from scipy.linalg import lapack

from emukit.quadrature.methods.warped_bq_model import WarpedBayesianQuadratureModel
from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess


class WSABI(WarpedBayesianQuadratureModel):
    """Base class for WSABI, Gunter et al. 2014"""
    def __init__(self, base_gp: IBaseGaussianProcess, adapt_offset: bool = False):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        :param adapt_offset: if True the offset the offset will be updated after new datapoints have been collected,
        if False, the offset will be constant at 0. defaults to False. Offset refers to 'alpha' in Gunter et al.
        """
        self.adapt_offset = adapt_offset
        if adapt_offset:
            self._compute_and_set_offset()
        else:
            self.offset = 0.

        super(WSABI, self).__init__(base_gp)

    def transform(self, Y):
        """ Transform from base-GP to integrand """
        return 0.5*(Y*Y) + self.offset

    def inverse_transform(self, Y):
        """ Transform from integrand to base-GP """
        return np.sqrt(np.absolute(2.*(Y - self.offset)))

    # TODO: check how it can be called after function evaluations (currently it is not)
    def _update_offset(self):
        if self.offset:
            self._compute_and_set_offset()

    def _compute_and_set_offset(self):
        """if adapted, it uses the value given in Gunter et al. 2014"""
        if self.adapt_offset:
            minL = min(self.base_gp.Y)[0]  # TODO: check if this returns the correct thing
            self.offset = 0.8 * minL

    def _symmetrize(self, A: np.ndarray) -> np.ndarray:
        """
        :param A: a square matrix, shape (N, N)
        :return: the symmetrized matrix 0.5 (A + A')
        """
        return 0.5 * (A + A.T)


class WSABIL(WSABI):
    """
    The WSABI-L Bayesian quadrature model (Gunter et al. 2014)

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

    # TODO: this hold for offset = 0 only?
    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance.

        :returns: estimator of integral and its variance
        """
        # integral mean
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]

        # integral variance
        qKq = self.base_gp.kern.qKq()
        gram_chol = self.base_gp.gram_chol()
        integral_var = qKq - np.square(lapack.dtrtrs(gram_chol, kernel_mean_X.T, lower=1)[0]).sum(axis=0,
                                                                                                  keepdims=True).T[0, 0]
        return integral_mean, integral_var


class WSABIM(WSABI):
    """
    The WSABI-M Bayesian quadrature model (Gunter et al. 2014)

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
        raise NotImplementedError
