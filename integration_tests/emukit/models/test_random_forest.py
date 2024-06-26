# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.examples.models.random_forest import RandomForest


@pytest.fixture
def model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = RandomForest(x_init, y_init)
    return model


def test_predict_shape(model):
    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    m, v = model.predict(x_test)

    assert m.shape == (10, 1)
    assert v.shape == (10, 1)


def test_update_data(model):
    rng = np.random.RandomState(43)
    x_new = rng.rand(5, 2)
    y_new = rng.rand(5, 1)
    model.set_data(x_new, y_new)

    assert model.X.shape == x_new.shape
    assert model.Y.shape == y_new.shape
