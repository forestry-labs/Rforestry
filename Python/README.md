## Rforestry: Random Forests, Linear Trees, and Gradient Boosting for Inference and Interpretability

Sören Künzel, Theo Saarinen, Simon Walter, Sam Antonyan, Edward Liu, Boban Petrovic, Allen Tang, Jasjeet Sekhon


## Introduction

random_forestry is a Python library that contains a fast implementation of Honest Random Forests, decision trees, and Linear Random Forests, with an emphasis on inference and interpretability. Training and predicting with a model matches the scikit-learn API in order to allow easy integration with existing packages.

Several features are specific to this library and allow for the training of tree based models with useful properties, including:

 - Honesty: the practice of splitting the training data set into two disjoint sets and using one set to determine the tree structure of a decision tree, and the other to determine the node values used for predictions.
 - Monotonic Constraints: the ability to specify a constraint on the decision tree or random forest model that forces the prediction of the model to be monotonic
 - Custom Bootstrap sampling methods: several different methods to determine how the samples for each tree in a forest are selected. This can be useful when the observations are not i.i.d. and the sampling scheme can exploit patterns in the data. This also interacts with honesty.
 - Linear aggregation: rather than using the average training outcome to predict for new observations, one can run a Ridge regression within each terminal node and use this for the predictions of that tree. Tree construction is modified according to an algorithm proposed in Linear Aggregation in Tree-based Estimators (https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2026780) to select recursive splitting points that are optimal for downstream Ridge regression aggregation.

These are some of the current interesting features, and more will be added in the future.

For full documentation, see the documentation site (https://random-forestry.readthedocs.io/en/latest/) and for the source code, see the github repo (https://github.com/forestry-labs/Rforestry).


## Developer Installation

1. Install [CMake](https://cmake.org/download/). MacOS users: install it via [Homebrew](https://brew.sh): `brew install cmake`
2. Install [Anaconda](https://docs.continuum.io/free/anaconda/install/mac-os/)
3. Open console (or Anaconda command prompt under Windows), clone this git repository **with submodules**, create Anaconda environment, build, install and test the package:
```bash
git clone --recursive https://github.com/forestry-labs/Rforestry.git
cd Rforestry/Python

conda create -n rforestry python pandas build pytest pytest-xdist pytest-sugar pytest-cov mypy
conda activate rforestry

python -m build --sdist
pip install dist/random-forestry-*.tar.gz
pytest tests/
```
4. To be able to run and debug via IDE without full package installation, run this script from the same `Rforestry/Python` folder, it will generate binary of the C++ extension and Python stubs for it:
```bash
mkdir build

pushd build
cmake -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../random_forestry ../extension
cmake --build .
popd

PYTHONPATH=random_forestry stubgen -m extension -o .
```


### Python Package Usage

Then the python code can be called:

```python
import numpy as np
import pandas as pd
from random import randrange
from random_forestry import RandomForest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'sepal length (cm)']
y = df['sepal length (cm)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fr = RandomForest(ntree = 500)

print("Fitting the forest")
fr.fit(X_train, y_train)


print("Predicting with the forest")
forest_preds = fr.predict(X_test)

```

### Plotting the forest

For visualizing the trees, make sure to install the [dtreeviz](https://github.com/parrt/dtreeviz#readme) python library.

```python
from dtreeviz.trees import *
from forestry_shadow import ShadowForestryTree


shadow_forestry = ShadowForestryTree(fr, X, y, X.columns.values, 'sepal length (cm)', tree_id=0)

viz = dtreeviz(shadow_forestry,
                scale=3.0,
                target_name='sepal length (cm)',
                feature_names=X.columns.values)

viz.view()

```

