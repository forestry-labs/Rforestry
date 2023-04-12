import pytest
from pydantic import ValidationError

from random_forestry import RandomForest


def test_properties():

    assert RandomForest(oob_honest=True, splitratio=0.3).splitratio == 1

    assert RandomForest(oob_honest=True, replace=False).replace

    assert RandomForest(double_tree=True, splitratio=0).double_tree is False
    assert RandomForest(double_tree=True, splitratio=0.3).double_tree
    assert RandomForest(double_tree=True, splitratio=1).double_tree is False

    assert RandomForest(interaction_depth=23, max_depth=4).interaction_depth == 4

    with pytest.raises(ValidationError):
        RandomForest(ntree=False)

    with pytest.raises(ValidationError):
        RandomForest(verbose=12)

    with pytest.raises(ValueError):
        RandomForest(ntree=0)

    with pytest.raises(ValueError):
        RandomForest(splitratio=1.4)

    with pytest.raises(ValueError):
        RandomForest(min_split_gain=0.2, linear=False)
