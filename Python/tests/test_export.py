import json

from helpers import get_data

from random_forestry import RandomForest


def test_export_json():
    X, y = get_data()
    forest = RandomForest()
    forest.fit(X, y)

    json_str = forest.export_json()

    # Check JSON structure and hardcoded values which shouldn't change
    json_obj = json.loads(json_str)

    def node_exists_with_nonzero(node_key_name):
        return any(any(abs(n.get(node_key_name, 0)) > 0.001 for n in t["nodes"]) for t in json_obj["trees"])

    assert node_exists_with_nonzero("threshold")
    assert node_exists_with_nonzero("leaf_value")

    assert type(json_obj) is dict
    assert json_obj["num_feature"] == 4
    assert json_obj["task_type"] == "kBinaryClfRegr"
    assert json_obj["average_tree_output"]
    assert json_obj["task_param"] == {
        "grove_per_class": False,
        "leaf_vector_size": 1,
        "num_class": 1,
        "output_type": "float",
    }
    assert json_obj["model_param"] == {"global_bias": 0.0, "pred_transform": "identity"}
    assert type(json_obj["trees"]) is list

    for t_info in json_obj["trees"]:
        assert type(t_info) is dict
        assert type(t_info["root_id"]) is int
        assert type(t_info["nodes"]) is list

        for n_info in t_info["nodes"]:
            assert type(n_info) is dict
            n_info: dict
            assert type(n_info["node_id"]) is int

            leaf_value = n_info.get("leaf_value")
            if leaf_value is not None:
                assert type(leaf_value) is float
                continue

            # non-leaf node
            assert type(n_info["split_feature_id"]) is int
            assert type(n_info["default_left"]) is bool

            assert type(n_info["left_child"]) is int
            assert type(n_info["right_child"]) is int

            split_type = n_info["split_type"]
            assert split_type in {"numerical", "categorical"}
            if split_type == "numerical":
                assert n_info["comparison_op"] == "<"
                assert type(n_info["threshold"]) is float
            elif split_type == "categorical":
                assert type(n_info["categories_list"]) is list
                assert len(n_info["categories_list"]) > 0
                assert all(type(x) is int for x in n_info["categories_list"])
