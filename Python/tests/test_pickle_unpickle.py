import pickle  # nosec B403  # 'Consider possible security implications associated with pickle'

import numpy as np
from sklearn.datasets import make_blobs

from random_forestry import RandomForest


def test_pickle_unpickle_predictions_json():
    X, y = make_blobs(
        n_samples=100,
        random_state=0,
        n_features=1000,
        cluster_std=0.1,
    )

    def pickle_unpickle(e: RandomForest) -> RandomForest:
        return pickle.loads(
            pickle.dumps(e)
        )  # nosec B301  # 'Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue'

    rng = np.random.RandomState(42)
    mask = rng.choice(X.size, 10, replace=False)
    X.reshape(-1)[mask] = np.nan

    for ntree in [1, 200]:
        forest = RandomForest(seed=432432, ntree=ntree)
        forest.fit(X, y)

        forest_unpickled = pickle_unpickle(forest)

        prediction = forest.predict(X)
        prediction_unpickled = forest_unpickled.predict(X)
        assert np.equal(prediction, prediction_unpickled).all()

        json_str = forest.export_json()
        json_str_unpickled = forest_unpickled.export_json()
        assert json_str == json_str_unpickled
