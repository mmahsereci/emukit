# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from emukit.core.loop import ModelUpdater
from emukit.examples.probLS.models.cubic_spline_gp import CubicSplineGP
from emukit.examples.probLS.loop.probls_loop_state import ProbLSLoopState


class ProbLSModelUpdater(ModelUpdater):
    """ Updates hyper-parameters every nth iteration, where n is defined by the user """
    def __init__(self, model: CubicSplineGP) -> None:
        """
        :param model: Emukit emulator model
        """
        self.model = model

    def update(self, loop_state: ProbLSLoopState) -> None:
        """
        Updates the training data of the model.
        :param loop_state: Object that contains current state of the loop
        """
        self.model.set_data(loop_state.X_transformed,
                            loop_state.Y_transformed,
                            loop_state.dY_transformed)
