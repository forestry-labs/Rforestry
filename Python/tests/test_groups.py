from helpers import get_data
from pandas import Series

from random_forestry import RandomForest


def test_groups():
    X, y = get_data()

    forest = RandomForest()
    groups = Series(range(len(X)), dtype="category")
    forest.fit(X, y, groups=groups)
    pred = forest.predict(X, aggregation="average")
    assert len(pred) == len(X)


def test_groups_oob():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    groups = Series([i for i in range(len(X) // 2) for _ in range(2)], dtype="category")
    forest.fit(X, y, groups=groups)
    pred = forest.predict(X, aggregation="oob", return_weight_matrix=True)
    assert len(pred["predictions"]) == len(X)
