# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.loop import UserFunctionResult


class NoisyUserFunctionWithGradientsResult(UserFunctionResult):
    """
    A class that records the inputs, outputs and meta-data of an evaluation of the noisy user function.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: np.ndarray, vardY: np.ndarray) -> None:
        """
        :param X: Function input, (num_dim, )
        :param Y: Function output(s), (1, )
        :param dY: Gradient(s) corresponding to Y (num_dim, )
        :param varY: (Estimated) variance estimates corresponding to Y (1, )
        :param vardY: (Estimated) variances corresponding to dY (num_dim, )
        """
        super().__init__(X, Y, cost=None)
        if dY.ndim != 1:
            raise ValueError("dY is expected to be 1-dimensional, actual dimensionality is {}".format(dY.ndim))

        if varY.ndim != 1:
            raise ValueError("varY is expected to be 1-dimensional, actual dimensionality is {}".format(varY.ndim))

        if vardY.ndim != 1:
            raise ValueError("vardY is expected to be 1-dimensional, actual dimensionality is {}".format(vardY.ndim))

        self.dY = dY
        self.varY = varY
        self.vardY = vardY
