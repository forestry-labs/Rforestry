import pytest
from helpers import get_data
from numpy.testing import assert_array_equal


def _test_predictions(forest, aggregation):
    X, _ = get_data()

    pred = forest.predict(X, aggregation=aggregation)
    pred_weight_matrix = forest.predict(X, aggregation=aggregation, return_weight_matrix=True)

    assert_array_equal(pred, pred_weight_matrix["predictions"])
    assert pred_weight_matrix["weightMatrix"].shape == (len(X.index), len(X.index))


@pytest.mark.forest_parameters(oob_honest=True)
@pytest.mark.parametrize("aggregation", ["average", "oob", "doubleOOB"])
def test_predictions_oob_honest_true(forest, aggregation):
    _test_predictions(forest, aggregation)


@pytest.mark.parametrize("aggregation", ["average", "oob", pytest.param("doubleOOB", marks=pytest.mark.xfail)])
def test_predictions_oob_honest_default(forest, aggregation):
    _test_predictions(forest, aggregation)
