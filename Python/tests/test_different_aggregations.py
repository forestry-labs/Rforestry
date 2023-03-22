import pytest
from helpers import get_data
from random_forestry import RandomForest


def test_predict_error():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)

    with pytest.raises(ValueError):
        forest.predict(aggregation="average")


def test_prediction_types():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)

    p = forest.predict(X, aggregation="average")
    print(p)
