import pytest
from helpers import get_data

from random_forestry import RandomForest


def test_properties():
    X, y = get_data()
    assert RandomForest(oob_honest=True).fit(X, y, splitratio=0.3).splitratio_ == 1

    assert RandomForest(oob_honest=True).fit(X, y, replace=False).replace_

    # assert RandomForest().fit(X, y, splitratio=0, double_tree=True).double_tree_ is False
    assert RandomForest().fit(X, y, splitratio=0.3, double_tree=True).double_tree_
    assert RandomForest().fit(X, y, splitratio=1, double_tree=True).double_tree_ is False

    assert RandomForest().fit(X, y, interaction_depth=23, max_depth=4).interaction_depth_ == 4

    # with pytest.raises(ValidationError):
    #    RandomForest(ntree=False)

    # with pytest.raises(ValidationError):
    #    RandomForest(verbose=12)

    # with pytest.raises(ValueError):
    #    RandomForest(ntree=0)

    with pytest.raises(ValueError):
        RandomForest().fit(X, y, splitratio=1.4)

    # with pytest.raises(ValueError):
    #    RandomForest(min_split_gain=0.2, linear=False)
