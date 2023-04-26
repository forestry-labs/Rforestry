# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from helpers import get_data

from random_forestry import RandomForest


@pytest.fixture
def forest():
    forest = RandomForest(
        ntree=500,
        sample_fraction=0.8,
        nodesize_strict_spl=5,
        splitrule="variance",
        nodesize_strict_avg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y, mtry=3, splitratio=1, replace=True)
    return forest


def test_translate_single_tree(forest: RandomForest):
    forest.translate_tree(0)

    assert len(forest.saved_forest_) == forest.ntree
    assert forest.saved_forest_[0]  # saved_forest[0] will be filled after translation
    assert all(forest.saved_forest_[i] == {} for i in range(1, forest.ntree))

    # numNodes = forest.saved_forest[0]["children_right"].size
    # assert not any(forest.saved_forest[0][key].size != numNodes for key in fr.saved_forest[0].keys() )


def test_all_trees(forest: RandomForest):
    X, _ = get_data()

    forest.translate_tree(0)
    assert forest.saved_forest_[0]
    assert len(forest.saved_forest_) == forest.ntree

    # Translating more trees
    forest.translate_tree([0, 1, 2])
    assert forest.saved_forest_[0]
    assert forest.saved_forest_[1]
    assert forest.saved_forest_[2]

    forest.translate_tree()

    for i in range(forest.ntree):
        assert forest.saved_forest_[i]

        num_nodes = forest.saved_forest_[i]["threshold"].size
        num_leaf_nodes = forest.saved_forest_[i]["values"].size
        # assert not any(forest.saved_forest_[i][key].size != numNodes for key in forest.saved_forest[i].keys())
        assert len(forest.saved_forest_[i]["feature"]) == num_nodes + num_leaf_nodes
        assert len(forest.saved_forest_[i]["na_left_count"]) == num_nodes
        assert len(forest.saved_forest_[i]["na_right_count"]) == num_nodes
        assert len(forest.saved_forest_[i]["na_default_direction"]) == num_nodes

        assert np.amax(forest.saved_forest_[i]["splitting_sample_idx"]) <= X.shape[0]
        assert np.amax(forest.saved_forest_[i]["averaging_sample_idx"]) <= X.shape[0]

        assert np.amax(forest.saved_forest_[i]["feature"]) <= X.shape[1]
