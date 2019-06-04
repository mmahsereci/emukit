# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Union


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

    def set_data(self, X: np.ndarray, Y: np.ndarray, dY: np.ndarray, varY: Union[np.ndarray, float],
                 vardY: Union[np.ndarray, float]) -> None:
        """
        Sets training data in model

        :param X: new points (num_points, num_dim)
        :param Y: noisy function values at new points X, (num_points, 1)
        :param dY: noisy gradients  at new points X, (num_points, num_dim)
        :param varY: variances of Y, array (num_points, 1), or positive scalar
        :param vardY: variances of dY, array (num_points, num_dim) or positive scalar
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
