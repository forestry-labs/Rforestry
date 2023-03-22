# pylint: disable=redefined-outer-name

import platform
from typing import Any

import numpy as np
import pytest
from helpers import get_data
from random_forestry import RandomForest


@pytest.fixture
def forest():
    forest = RandomForest(
        ntree=500,
        replace=True,
        sample_fraction=0.8,
        mtry=3,
        nodesize_strict_spl=5,
        nthread=1,
        splitrule="variance",
        splitratio=1,
        nodesize_strict_avg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest


@pytest.fixture
def predictions(forest: RandomForest):
    X, _ = get_data()
    return forest.predict(X)


def test_newdata_wrong_dim(forest: RandomForest):
    X, _ = get_data()
    ncol = X.shape[1]
    X = X.iloc[:, 0 : ncol - 1]

    with pytest.raises(ValueError):
        assert forest.predict(X)


def test_newdata_shuffled_warning(forest: RandomForest):
    X, _ = get_data()
    with pytest.warns(UserWarning):
        forest.predict(X.iloc[:, ::-1])


def test_equal_predictions(forest: RandomForest):
    X, _ = get_data()
    predictions_1 = forest.predict(X)
    predictions_2 = forest.predict(X.iloc[:, ::-1])

    assert np.array_equal(predictions_1, predictions_2)


@pytest.mark.skipif(platform.system() == "Darwin", reason="This is expected to fail on MacOS")
def test_error(predictions: Any):
    _, y = get_data()
    print(np.mean((predictions - y) ** 2))
    assert np.mean((predictions - y) ** 2) < 1
