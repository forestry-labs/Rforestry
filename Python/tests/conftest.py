from typing import Type

import pytest
from helpers import get_data

from random_forestry import RandomForest


@pytest.fixture
def forest(request: Type[pytest.FixtureRequest]):
    X, y = get_data()

    forest_parameters = request.node.get_closest_marker("forest_parameters")
    if hasattr(forest_parameters, "kwargs"):
        return RandomForest(**forest_parameters.kwargs).fit(X, y)
    else:
        return RandomForest().fit(X, y)
