# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from helpers import get_data
from random_forestry import RandomForest


@pytest.fixture
def forest():
    forest = RandomForest(
        ntree=500,
        replace=True,
        sample_fraction=0.8,
        mtry=3,
        nodesize_strict_spl=5,
        splitrule="variance",
        splitratio=1,
        nodesize_strict_avg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_translate_single_tree(forest: RandomForest):
    assert not forest.saved_forest

    forest.translate_tree(0)
    print("Length of pyforest")
    print(len(forest.saved_forest))
    print(forest.saved_forest[0])

    assert len(forest.saved_forest) == forest.ntree
    assert forest.saved_forest[0]  # saved_forest[0] will be filled after translation
    assert all(forest.saved_forest[i] == {} for i in range(1, forest.ntree))

    # numNodes = fr.saved_forest[0]['children_right'].size
    # assert not any(fr.saved_forest[0][key].size != numNodes for key in fr.saved_forest[0].keys() )


def test_all_trees(forest: RandomForest):
    X, _ = get_data()

    forest.translate_tree(0)
    assert forest.saved_forest[0]
    assert len(forest.saved_forest) == forest.ntree

    # Translating more trees
    forest.translate_tree([0, 1, 2])
    assert forest.saved_forest[0]
    assert forest.saved_forest[1]
    assert forest.saved_forest[2]

    forest.translate_tree()

    for i in range(forest.ntree):
        assert forest.saved_forest[i]

        num_nodes = forest.saved_forest[i]["threshold"].size
        num_leaf_nodes = forest.saved_forest[i]["values"].size
        # assert not any(forest.saved_forest[i][key].size != numNodes for key in forest.saved_forest[i].keys())
        assert len(forest.saved_forest[i]["feature"]) == num_nodes+num_leaf_nodes
        assert len(forest.saved_forest[i]["na_left_count"]) == num_nodes
        assert len(forest.saved_forest[i]["na_right_count"]) == num_nodes
        assert len(forest.saved_forest[i]["na_default_direction"]) == num_nodes

        assert np.amax(forest.saved_forest[i]["splitting_sample_idx"]) <= X.shape[0]
        assert np.amax(forest.saved_forest[i]["averaging_sample_idx"]) <= X.shape[0]

        assert np.amax(forest.saved_forest[i]["feature"]) <= X.shape[1]
