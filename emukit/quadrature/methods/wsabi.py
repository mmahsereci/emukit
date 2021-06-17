"""WSABI model as in Gunter et al. 2014"""


import numpy as np

import emukit.quadrature.methods.bounded_bq_model
from ..interfaces.base_gp import IBaseGaussianProcess
from .bounded_bq_model import BoundedBQModel


class WSABIL(BoundedBQModel):
    """WSABI-L Warped Sequential Active Bayesian Integration with linear approximation [1].

    WSABI-L must be used with the RBF kernel and the Gaussian integration measure. This means that the kernel of
    :attr:`base_gp` must be of type :class:`emukit.quadrature.kernels.QuadratureRBFIsoGaussMeasure`.

    - The linear approximation is described in [1] in section 3.1, Eq (9) and (10).

    - The offset :math:`\alpha` will either be set to a small value if ``adapt_alpha`` is ``False``.
      Else it will be adapted according to :math:`0.8 \min(Y)` as in Gunter et al. 2014, page 3, footnote,
      where :math:`Y` are the collected integrand evaluations so far.

    - WSABI-L uses uncertainty sampling as acquisition strategy.

    .. [1]  Gunter et al. 2014 *Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature*,
            Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    .. seealso:: :class:`emukit.quadrature.methods.bounded_bq_model.BoundedBQModel`.

    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_alpha: bool=True):
        """
        :param base_gp: A model derived from :class:`emukit.quadrature.interfaces.IBaseGaussianProcess`.
                        Must use :class:`emukit.quadrature.kernels.QuadratureRBFIsoGaussMeasure` as kernel.
        :param X: The initial locations of integrand evaluations, shape (num_points, input_dim).
        :param Y: The values of the integrand at X, shape (num_points, 1).
        :param adapt_alpha: If ``True``, the offset :math:`\alpha` will be adapted. If ``False`` :math:`\alpha` will be
               fixed to a small value for numerical stability. Default is ``True``.
        """
        self.adapt_alpha = adapt_alpha
        self._small_alpha = 1e-8  # only used if alpha is not adapted
        alpha = self._compute_alpha(X, Y)

        super(WSABIL, self).__init__(base_gp=base_gp, X=X, Y=Y, bound=alpha, is_lower_bounded=True)

    def _compute_alpha(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the offset :math:`\alpha`.

        :param X: Observation locations, shape (num_points, input_dim)
        :param Y: Values of observations, shape (num_points, 1)
        :return: The offset :math:`\alpha`.
        """
        if self.adapt_alpha:
            return 0.8 * min(Y)[0]
        return self._small_alpha

    def update_parameters(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Compute and set the offset :math:`\alpha` in the warping.

        :param X: observation locations, shape (num_points, input_dim)
        :param Y: values of observations, shape (num_points, 1)
        """
        self._warping.update_parameters(offset=self._compute_alpha(X, Y))