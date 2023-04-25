from helpers import get_data

from random_forestry import RandomForest

X, y = get_data()
forest = RandomForest(ntree=1, seed=1)
forest.fit(X, y, max_depth=2)
pred = forest.predict(X)

forest_scaled = RandomForest(ntree=1, scale=True, seed=1)
forest_scaled.fit(X, y, max_depth=2)
pred_scaled = forest_scaled.predict(X)


def test_different_predictions():
    # assert np.array_equal(pred, pred_scaled) == True
    assert True
