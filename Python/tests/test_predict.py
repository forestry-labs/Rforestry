import re

import numpy as np
import pytest
from helpers import get_data

from random_forestry import RandomForest


def test_predict():
    forest = RandomForest()

    X, y = get_data()
    forest.fit(X, y)

    with pytest.raises(
        ValueError, match="When using an aggregation that is not oob or doubleOOB, one must supply newdata"
    ):
        forest.predict()

    with pytest.raises(TypeError, match=re.escape("predict() takes from 1 to 2 positional arguments but 3 were given")):
        forest.predict(X, 4)

    with pytest.raises(ValueError):
        # match=re.escape(
        #    "Expected 2D array, got scalar array instead:\narray=5\nReshape your data either using "
        #    "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) "
        #    "if it contains a single sample.",
        # )
        # ):
        forest.predict(5)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected 2D array, got 1D array instead:\narray=[2 3 4 8].\nReshape your data either using "
            "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample.",
        ),
    ):
        forest.predict([2, 3, 4, 8])

    with pytest.raises(
        ValueError, match="When using tree indices, we must have exact = True and aggregation = 'average'"
    ):
        forest.predict(newdata=X, trees=[1, 2, 3, 4, 4, 499], aggregation="average", exact=False)
    with pytest.raises(
        ValueError, match="When using tree indices, we must have exact = True and aggregation = 'average'"
    ):
        forest.predict(newdata=X, trees=[1, 2, 3, 4, 4, 499], aggregation="oob")

    with pytest.raises(ValueError, match="trees must contain indices which are integers between -ntree and ntree-1"):
        forest.predict(newdata=X, trees=np.array([1, 2, 3, 4, 4, 500]))

    try:
        forest.predict(newdata=X, trees=np.array([-500, 2, 3, 4, 4, 499]))
    except ValueError:
        assert False

    with pytest.raises(ValueError, match=("newdata has different columns then the ones the forest was trained with.")):
        X.columns = ["a", "b", "c", "d"]
        forest.predict(X)
