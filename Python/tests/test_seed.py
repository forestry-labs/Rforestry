import platform

import numpy as np
import pytest
from helpers import get_data

from random_forestry import RandomForest


@pytest.mark.skipif(platform.system() == "Darwin", reason="This is expected to fail on MacOS")
def test_different_predictions():

    X, y = get_data()

    forest1 = RandomForest(ntree=10, max_depth=2, seed=1, scale=False)
    forest1.fit(X, y)
    predictions_1 = forest1.predict(X)

    forest2 = RandomForest(ntree=10, max_depth=2, seed=1, scale=False)
    forest2.fit(X, y)
    predictions_2 = forest2.predict(X)

    assert np.array_equal(predictions_1, predictions_2)

    X, y = get_data()

    forest3 = RandomForest(ntree=10, max_depth=2, seed=1, scale=False)
    forest3.fit(X, y)
    predictions_3 = forest3.predict(X)

    assert np.array_equal(predictions_1, predictions_3)
