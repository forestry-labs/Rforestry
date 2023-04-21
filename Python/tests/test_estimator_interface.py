from sklearn.utils.estimator_checks import check_estimator

from random_forestry import RandomForest


def test_all_estimators():
    return check_estimator(RandomForest())


print(RandomForest._get_param_names())
