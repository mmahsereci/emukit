# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Callable

from ...core.loop import LoopState, ModelUpdater
from ...core.interfaces import IModel


class NoisyModelWithGradientsDataOnlyUpdater(ModelUpdater):
    """ Updates hyper-parameters every nth iteration, where n is defined by the user """
    def __init__(self, model: IModel, y_extractor_fcn: Callable=None) -> None:
        """
        :param model: Emukit emulator model
        :param y_extractor_fcn: A function that takes in loop state and returns the training targets.
                                      Defaults to a function returning loop_state.Y
        """
        self.model = model

        # Todo: change this so it can extract the noise stuff, too
        if y_extractor_fcn is None:
            self.y_extractor_fcn = lambda loop_state: loop_state.Y
        else:
            self.y_extractor_fcn = y_extractor_fcn

    def update(self, loop_state: LoopState) -> None:
        """
        :param loop_state: Object that contains current state of the loop
        """
        # Todo: I can set data or just add one datapoint
        targets = self.y_extractor_fcn(loop_state)
        self.model.set_data(loop_state.X, targets)
