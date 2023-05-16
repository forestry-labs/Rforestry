# import platform

import time

import pytest
from helpers import get_data

from random_forestry import RandomForest

X, y = get_data()


class TestAfterInit:
    @pytest.mark.skip
    def test_when_default_seed(self):
        forest_1 = RandomForest()
        time.sleep(1)
        forest_2 = RandomForest()
        assert forest_1 != forest_2

    def test_when_equal_seed(self):
        forest_1 = RandomForest(seed=123)
        forest_2 = RandomForest(seed=123)
        assert forest_1 == forest_2

    def test_when_different_params(self):
        forest_1 = RandomForest(seed=56, ntree=34)
        forest_2 = RandomForest(seed=56, nthread=6)
        assert forest_1 != forest_2


class TestAfterFit:
    def test_it_is_different(self):
        forest_1 = RandomForest(seed=123)
        forest_2 = RandomForest(seed=123).fit(X, y)
        assert forest_1 != forest_2

    def test_no_randomness_added(self):
        forest_1 = RandomForest(seed=123).fit(X, y)
        forest_2 = RandomForest(seed=123).fit(X, y)
        assert forest_1 == forest_2

    def test_idempotency(self):
        forest_1 = RandomForest(seed=123).fit(X, y)
        forest_2 = RandomForest(seed=123).fit(X, y).fit(X, y)
        assert forest_1 == forest_2

    @pytest.mark.skip
    def test_different_params(self):
        forest_1 = RandomForest(seed=123).fit(X, y, double_bootstrap=True)
        forest_2 = RandomForest(seed=123).fit(X, y, double_bootstrap=False)
        assert forest_1 != forest_2

        forest_1 = RandomForest(seed=123).fit(X, y, max_obs=4)
        forest_2 = RandomForest(seed=123).fit(X, y, max_obs=5)
        assert forest_1 != forest_2
