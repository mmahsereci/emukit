# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.loop import CandidatePointCalculator
from . import ProbLSLoopState
from ...probabilistic_linesearch.acquisitions import WolfeProbability


class ProbLSCandidatePointCalculator(CandidatePointCalculator):
    """ Computes the next point(s) for function evaluation """

    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: WolfeProbability) -> None:
        """
        :param acquisition: The acquisition function
        """
        # Todo: replace acq in init with union PW, PW+EI etc
        self.acquisition = acquisition

    def compute_next_points(self, loop_state: ProbLSLoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: will be ignored
        :return: (1 x n_dims) array of next inputs to evaluate the function at
        """
        # Todo: compute mins of mean and one extrapolation point, choose among those the one with best PW+EI
        # Todo: decide of to return x or alpha. Probably x is better for user function
        x = np.array([0.])
        return x