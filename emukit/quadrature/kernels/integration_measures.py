# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List


class IntegrationMeasure:
    """An integration measure"""

    def __init__(self, name: str):
        """
        :param name: Name of the integration measure
        """
        self.name = name

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computes, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        raise NotImplementedError

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        raise NotImplementedError


class UniformMeasure(IntegrationMeasure):
    """The Uniform measure"""

    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        :param bounds: List of D tuples, where D is the dimensionality of the domain and the tuples contain the lower
        and upper bounds of the box defining the uniform measure i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        super().__init__('UniformMeasure')

        # checks if lower bounds are smaller than upper bounds.
        for bounds_d in bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError("Upper bound of uniform measure must be larger than lower bound. Found a pair "
                                 "containing (" + str(lb_d) + ", " + str(ub_d) + ").")

        self.bounds = bounds
        # uniform measure has constant density which is computed here.
        self.density = self._compute_density()

    def _compute_density(self) -> float:
        differences = np.array([x[1] - x[0] for x in self.bounds])
        volume = np.prod(differences)

        if not volume > 0:
            raise NumericalPrecisionError("Domain volume of uniform measure is not positive. Its value is "
                                          + str(volume) + ". It might be numerical problems...")
        return np.float(1. / volume)

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computes, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        # check if points are inside the box
        bounds_lower = np.array([b[0] for b in self.bounds])
        bounds_upper = np.array([b[1] for b in self.bounds])
        inside_lower = 1 - (x < bounds_lower)
        inside_upper = 1 - (x > bounds_upper)
        inside_upper_lower = (inside_lower * inside_upper).sum(axis=1) == x.shape[1]
        return inside_upper_lower * self.density

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        return self.bounds


class IsotropicGaussianMeasure(IntegrationMeasure):
    """The isotropic Gaussian measure"""

    def __init__(self, mean: np.ndarray, variance: float):
        """
        :param mean: the mean of the Gaussian, shape (dim, )
        :param variance: the scalar variance of the isotropic covariance matrix of the Gaussian.
        """
        super().__init__('GaussianMeasure')
        # check mean
        if not isinstance(mean, np.ndarray):
            raise TypeError('Mean must be of type numpy.ndarray, ' + str(type(mean)) + ' given.')

        if mean.ndim != 1:
            raise ValueError('Dimension of mean must be 1, dimension ' + str(mean.ndim) + ' given.')

        # check covariance
        if not isinstance(variance, float):
            raise TypeError('Variance must be of type float, ' + str(type(variance)) + ' given.')

        if not variance > 0:
            raise ValueError('Variance must be positive, current value is ', variance, '.')

        self.mean = mean
        self.variance = variance
        self.dim = mean.shape[0]

    @property
    def full_covariance_matrix(self):
        return self.variance * np.eye(self.dim)

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computes, shape (num_points, dim)
        :return: the density at x, shape (num_points, )
        """
        factor = (2 * np.pi * self.variance) ** (self.dim / 2)
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(- np.sum(scaled_diff ** 2, axis=1)) / factor

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        # Note: the factor 10 is pretty arbitrary here. Investigate if we can find a formula that adapts to
        # higher dimensions where most of the volume is located in a sphere rather than around the mean.
        factor = 10
        lower = self.mean - factor * np.sqrt(self.variance)
        upper = self.mean + factor * np.sqrt(self.variance)
        return [(lb, up) for lb, up in zip(lower, upper)]


class NumericalPrecisionError(Exception):
    pass
