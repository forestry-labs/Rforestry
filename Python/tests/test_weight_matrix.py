from helpers import get_data
from numpy.testing import assert_array_equal
from random_forestry import RandomForest

def test_average():
    X, y = get_data()

    forest = RandomForest()
    forest.fit(X, y)

    pred = forest.predict(X, aggregation="average")
    pred_weight_matrix = forest.predict(X, aggregation="average", return_weight_matrix=True)

    assert_array_equal(pred, pred_weight_matrix['predictions'])
    assert pred_weight_matrix['weightMatrix'].shape == (len(X.index), len(X.index))

def test_predict_weight_matrix():
    X, y = get_data()

    forest = RandomForest()
    forest.fit(X, y)

    pred = forest.predict(X, aggregation="oob")
    pred_weight_matrix = forest.predict(X, aggregation="oob", return_weight_matrix=True)

    assert_array_equal(pred, pred_weight_matrix['predictions'])
    assert pred_weight_matrix['weightMatrix'].shape == (len(X.index), len(X.index))
