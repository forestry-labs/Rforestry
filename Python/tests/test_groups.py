import numpy as np
from helpers import get_data
from pandas import Series

from random_forestry import RandomForest


class TestGroups:
    def _predict(self, **kwargs):
        X, y = get_data()

        forest = RandomForest(**kwargs).fit(X, y)
        self.pred_avg = forest.predict(X, aggregation="average")
        self.pred_oob = forest.predict(X, aggregation="oob")

        groups = Series([i for i in range(len(X) // 10) for _ in range(10)])
        forest = RandomForest(**kwargs).fit(X, y, groups=groups)
        self.pred_avg_groups = forest.predict(X, aggregation="average")
        self.pred_oob_groups = forest.predict(X, aggregation="oob")

    def test_groups_honest_default(self):
        self._predict()
        assert np.array_equal(self.pred_avg, self.pred_avg_groups)
        assert not np.array_equal(self.pred_oob, self.pred_oob_groups)

    def test_groups_honest_true(self):
        self._predict(oob_honest=True)
        assert not np.array_equal(self.pred_avg, self.pred_avg_groups)
        assert not np.array_equal(self.pred_oob, self.pred_oob_groups)
