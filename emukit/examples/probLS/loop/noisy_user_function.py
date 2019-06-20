# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Callable, List
import numpy as np

from emukit.core.loop import UserFunction
from .probls_user_function_result import NoisyUserFunctionWithGradientsResult


import logging
_log = logging.getLogger(__name__)


class NoisyUserFunctionWithGradientsWrapper(UserFunction):
    """ Wraps a user-provided python function. """
    def __init__(self, f: Callable):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a tuple of 2d numpy arrays
                  containing noisy function values y, noisy gradients dy, variances of y, variances of dy.
        """
        self.f = f

    def evaluate(self, inputs: np.ndarray) -> List[NoisyUserFunctionWithGradientsResult]:
        """
        Evaluates python function by providing it with numpy types and converts the output
        to a List of NoisyUserFunctionWithGradientsResult

        :param inputs: function inputs at which to evaluate function
        :return: function results
        """
        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, "
                             "actual input dimensionality is {}".format(inputs.ndim))

        _log.info("Evaluating user function for {} point(s)".format(inputs.shape[0]))
        outputs = self.f(inputs)

        if isinstance(outputs, tuple):
            y_out, dy_out, vary_out, vardy_out = outputs
        else:
            raise ValueError("User provided function should return a tuple of ndarrays, "
                             "{} received".format(type(outputs)))

        if y_out.ndim != 2:
            raise ValueError("First output of user function should be 2d array, actual output dimensionality is "
                             "{}".format(y_out.ndim))

        if dy_out.ndim != 2:
            raise ValueError("Second output of user function should be 2d array, actual output dimensionality is "
                             "{}".format(dy_out.ndim))

        if vary_out.ndim != 2:
            raise ValueError("Third output of user function should be 2d array, actual output dimensionality is "
                             "{}".format(vary_out.ndim))

        if vardy_out.ndim != 2:
            raise ValueError("Fourth output of user function should be 2d array, actual output dimensionality is "
                             "{}".format(vardy_out.ndim))

        results = []
        for x, y, dy, vary, vardy in zip(inputs, y_out, dy_out, vary_out, vardy_out):
            results.append(NoisyUserFunctionWithGradientsResult(x, y.flatten(), dy, vary, vardy))
        return results


