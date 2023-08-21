import pytest
from helpers import get_data


def test_predict_error(forest):
    with pytest.raises(ValueError):
        forest.predict(aggregation="average")


def test_predict_average(forest):
    X, _ = get_data()
    prediction = forest.predict(X, aggregation="average")
    assert len(prediction) == len(X)


@pytest.mark.forest_parameters(oob_honest=True)
def test_predict_oob(forest):
    X, _ = get_data()
    prediction = forest.predict(X, aggregation="oob")
    assert len(prediction) == len(X)


@pytest.mark.forest_parameters(oob_honest=True)
def test_predict_double_oob(forest):
    X, _ = get_data()
    prediction = forest.predict(X, aggregation="doubleOOB")
    assert len(prediction) == len(X)
