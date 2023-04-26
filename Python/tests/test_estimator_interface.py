from sklearn.utils.estimator_checks import parametrize_with_checks

from random_forestry import RandomForest


@parametrize_with_checks([RandomForest()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
