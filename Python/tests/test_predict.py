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

    with pytest.raises(
        AttributeError, match="newdata must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list"
    ):
        forest.predict(5)

    with pytest.raises(ValueError, match="newdata has 1, but the forest was trained with 4 columns."):
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
