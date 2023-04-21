import re

import numpy as np
import pytest
from helpers import get_data
from numpy.testing import assert_array_equal

from random_forestry import RandomForest


def test_fit_validator():
    forest = RandomForest()

    X, y = get_data()

    with pytest.raises(
        ValueError, match=re.escape("Found input variables with inconsistent numbers of samples: [150, 149]")
    ):
        forest.fit(X, y.drop(0))

    with pytest.raises(ValueError, match="Input y contains NaN."):
        forest.fit(X, y.replace(0, np.NaN))

    with pytest.raises(ValueError, match="Training data column cannot be all missing values."):
        X_nan = X.copy()
        X_nan["nan_col"] = np.NaN
        forest.fit(X_nan, y)

    with pytest.raises(
        TypeError,
        match=re.escape(
            (
                "fit() takes 3 positional arguments but 4 positional arguments "
                "(and 5 keyword-only arguments) were given"
            )
        ),
    ):
        forest.fit(X, y, 23)

    with pytest.raises(ValueError, match="monotonic_constraints must have the size of x"):
        forest.fit(X, y, monotonic_constraints=[0, 2, 3])

    monotonic_constraints = np.array([0] * X.shape[1])
    monotonic_constraints[0] = 2
    with pytest.raises(ValueError, match="monotonic_constraints must be either 1, 0, or -1"):
        forest.fit(X, y, monotonic_constraints=monotonic_constraints)

    forest.set_params(linear=True)
    monotonic_constraints = np.array([0] * X.shape[1])
    monotonic_constraints[0] = 1
    with pytest.raises(ValueError, match="Cannot use linear splitting with monotonic_constraints"):
        forest.fit(X, y, monotonic_constraints=monotonic_constraints)

    observation_weights = np.array([0] * X.shape[0])
    forest.set_params(replace=True)
    with pytest.raises(ValueError, match="There must be at least one non-zero weight in observation_weights"):
        forest.fit(X, y, observation_weights=observation_weights)

    observation_weights = np.array([1] * X.shape[0])
    with pytest.raises(ValueError, match=re.escape("observation_weights must have length len(x)")):
        forest.fit(X, y, observation_weights=np.delete(observation_weights, 0))

    observation_weights[0] = -2
    with pytest.raises(ValueError, match="The entries in observation_weights must be non negative"):
        forest.fit(X, y, observation_weights=observation_weights)

    lin_feats = np.array([0] * X.shape[1])
    lin_feats[0] = -1
    with pytest.raises(
        ValueError, match=re.escape("lin_feats must contain positive integers less than len(x.columns).")
    ):
        forest.fit(X, y, lin_feats=lin_feats)

    lin_feats = np.array([0] * X.shape[1])
    lin_feats[0] = X.shape[1] + 1
    with pytest.raises(
        ValueError, match=re.escape("lin_feats must contain positive integers less than len(x.columns).")
    ):
        forest.fit(X, y, lin_feats=lin_feats)


def test_observation_weights():
    X, y = get_data()
    forest = RandomForest()
    observation_weights = np.array([1] * X.shape[0])
    forest.fit(X, y, observation_weights=observation_weights)
    pred_avg = forest.predict(X, aggregation="average")

    assert len(pred_avg) == len(X)

    forest = RandomForest()
    observation_weights = np.array([0] * X.shape[0])
    n_weighted_obs = 10
    observation_weights[0:n_weighted_obs] = 1
    forest.fit(X, y, observation_weights=observation_weights)
    pred = forest.predict(X, aggregation="average")
    pred_weight_matrix = forest.predict(X, aggregation="average", return_weight_matrix=True)

    assert_array_equal(pred, pred_weight_matrix["predictions"])
    assert not np.any(pred_weight_matrix["weightMatrix"][:, n_weighted_obs : (X.shape[0])])
