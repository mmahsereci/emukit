# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Union

from emukit.core.loop import StoppingCondition
from .probls_wolfe_conditions import WolfeConditions


class WolfeThresholdStoppingCondition(StoppingCondition):
    """ Stops when point is found that fulfills the probabilistic Wolfe conditions """
    def __init__(self, wolfe_conditions: WolfeConditions) -> None:
        """
        :param wolfe_conditions: The probabilistic Wolfe conditions used in the loop, i,e., the same as used in the
        acquisition function.
        """
        self.wolfe_conditions = wolfe_conditions

    def should_stop(self, wolfe_probabilities: np.ndarray) -> Tuple[bool, Union[int, None]]:
        """
        :param wolfe_probabilities: The wolfe probabilities in the order of evaluation
        :return: True if point point that has Wolfe probability larger than threshold cw; index of which evaluation has
        been accepted (None if no point is accepted).
        """
        num_evals = wolfe_probabilities.shape[0]
        wolfe_idx = np.where(wolfe_probabilities > self.wolfe_conditions.cw)[0]
        wolfe_set = wolfe_probabilities[wolfe_idx]

        # empty wolfe set
        if len(wolfe_set) == 0:
            return False, None

        # Wolfe set has one entry
        elif len(wolfe_set) == 1:
            accept = wolfe_probabilities[0] > self.wolfe_conditions.cw
            idx = wolfe_idx[0]
            return accept, int(idx)

        # Wolfe set has more than one entry
        else:
            # check last eval first if it exists
            if wolfe_idx[-1] == num_evals - 1:
                accept = wolfe_probabilities[-1] > self.wolfe_conditions.cw
                # return of last point is acceptable
                if accept:
                    idx = wolfe_idx[-1]
                    return accept, int(idx)
                # check the other points and choose the one with highest Wolfe probability
                idx_wolfe_set = wolfe_set.armax()
                idx = wolfe_idx[idx_wolfe_set]
                return True, int(idx)
