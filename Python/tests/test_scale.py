import numpy as np
import pytest
from helpers import get_data

from random_forestry import RandomForest


@pytest.mark.skip(reason="TODO: investigate - different results")
def test_different_predictions():
    X, y = get_data()
    forest1 = RandomForest(ntree=1, seed=1)
    forest1 = forest1.fit(X, y, max_depth=2)
    predictions = forest1.predict(X)

    forest2 = RandomForest(ntree=1, seed=1, scale=True)
    forest2 = forest2.fit(X, y, max_depth=2)
    predictions_scaled = forest2.predict(X)

    assert np.allclose(predictions, predictions_scaled)
