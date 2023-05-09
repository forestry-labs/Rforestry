import os
import sys
from contextlib import contextmanager

from numpy import ndarray
from pandas import Series
from random_forestry import RandomForest
from sklearn import datasets


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout, sys.stdout = sys.stdout, devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def test_stdout(test_name: str, results: ndarray) -> None:
    print("=== " + test_name + " ===")
    for res in results:
        print(f"{res:.6f}")


iris = datasets.load_iris()
X = iris.data[:, 1:4]
y = iris.data[:, 0]

# Test default parameters
with suppress_stdout():
    forest = RandomForest(seed=2)
    forest.fit(X, y)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

test_stdout("Default parameters: average", pred_avg)
test_stdout("Default parameters: oob", pred_oob)

# Test oob honest
with suppress_stdout():
    forest = RandomForest(seed=2, oob_honest=True)
    forest.fit(X, y)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

test_stdout("OOB honest: average", pred_avg)
test_stdout("OOB honest: oob", pred_oob)

# Test groups
groups = Series([i for i in range(len(X) // 10) for _ in range(10)])
with suppress_stdout():
    forest = RandomForest(seed=2)
    forest.fit(X, y, groups=groups)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

test_stdout("Groups: average", pred_avg)
test_stdout("Groups: oob", pred_oob)

# Test oob honest and groups
with suppress_stdout():
    forest = RandomForest(seed=2, oob_honest=True)
    groups = Series([i for i in range(len(X) // 10) for _ in range(10)])
    forest.fit(X, y, groups=groups)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

test_stdout("OOB honest and groups: average", pred_avg)
test_stdout("OOB honest and groups: oob", pred_oob)
