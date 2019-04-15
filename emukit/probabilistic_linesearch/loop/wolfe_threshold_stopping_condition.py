# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...core.loop import StoppingCondition
from . import WolfeConditions, ProbLSLoopState


class WolfeThresholdStoppingCondition(StoppingCondition):
    """ Stops when point is found that fulfills the probabilistic Wolfe conditions """
    def __init__(self, wolfe_conditions: WolfeConditions) -> None:
        """
        :param wolfe_conditions: The probabilistic Wolfe conditions used in the loop
        """
        self.wolfe_conditions = wolfe_conditions

    def should_stop(self, loop_state: ProbLSLoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: True if point point that has Wolfe probability larger than threshold cw
        """
        # Todo: this might need a new loop state
        # Todo: implement this
        return False
