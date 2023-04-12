# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from helpers import get_data

from random_forestry import RandomForest


@pytest.fixture
def forest():
    forest = RandomForest(seed=432432)

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_predict_settings(forest):
    X, _ = get_data()

    predictions_1 = forest.predict(newdata=X)
    predictions_2 = forest.predict(newdata=X, trees=np.arange(500))
    assert np.array_equal(predictions_1, predictions_2)


def test_linearity(forest):
    X, _ = get_data()

    predictions_1 = forest.predict(newdata=X, trees=np.array([1]))
    predictions_2 = forest.predict(newdata=X, trees=[2])
    predictions_3 = forest.predict(newdata=X, trees=[3])
    predictions_4 = forest.predict(newdata=X, trees=[4])

    predictions_all = forest.predict(newdata=X, trees=[1, 2, 3, 4])
    predictions_agg = 0.25 * (predictions_1 + predictions_2 + predictions_3 + predictions_4)

    assert np.array_equal(predictions_all, predictions_agg)

    predictions_all = forest.predict(newdata=X, trees=[1, 1, 1, 2, 2])
    predictions_agg = (predictions_1 * 3 + predictions_2 * 2) / 5

    assert np.array_equal(predictions_all, predictions_agg)
