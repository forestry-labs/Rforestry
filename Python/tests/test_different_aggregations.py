import pytest
from helpers import get_data
from random_forestry import RandomForest


def test_predict_error():
    X, y = get_data()

    forest = RandomForest()
    forest.fit(X, y)

    with pytest.raises(ValueError):
        forest.predict(aggregation="average")


def test_predict_average():
    X, y = get_data()

    forest = RandomForest()
    forest.fit(X, y)
    prediction = forest.predict(X, aggregation="average")
    assert len(prediction) == len(X)


def test_predict_oob():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)

    prediction = forest.predict(X, aggregation="oob")
    assert len(prediction) == len(X)


def test_predict_double_oob():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)

    prediction = forest.predict(X, aggregation="doubleOOB")
    assert len(prediction) == len(X)