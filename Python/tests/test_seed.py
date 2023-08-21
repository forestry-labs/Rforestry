import numpy as np
from helpers import get_data

from random_forestry import RandomForest


def test_different_predictions():
    X, y = get_data()

    forest1 = RandomForest(ntree=10, seed=1, scale=False)
    forest1.fit(X, y, max_depth=2)
    predictions_1 = forest1.predict(X)

    forest2 = RandomForest(ntree=10, seed=1, scale=False)
    forest2.fit(X, y, max_depth=2)
    predictions_2 = forest2.predict(X)

    assert np.array_equal(predictions_1, predictions_2)

    forest3 = RandomForest(ntree=10, seed=1, scale=False)
    forest3.fit(X, y, max_depth=2)
    predictions_3 = forest3.predict(X)

    assert np.array_equal(predictions_1, predictions_3)
