Usage
======

Here are some examples of how to use the forestry package and some of its features. For a comprehensive
overview of all the classes and functions, check out the :doc:`API Reference <api>`.

.. contents:: Contents
    :depth: 2
    :local:


Installation
------------

To use **random_forestry**, first install it using pip:

.. code-block:: console

   (.venv) $ pip install random_forestry

.. _set_get:

Setting the Parameters
----------------------

Here is an example of how to use :meth:`get_params() <random_forestry.RandomForest.get_params>`
and :meth:`set_params() <forestry.RandomForest.set_params>` to get and set the parameters
of the random_forestry.

.. code-block:: Python

    from random_forestry import RandomForest

    # Create a RandomForest object
    fr = RandomForest(ntree=100, mtry=3, oob_honest=True)

    # Check out the list of parameters
    print(fr.get_parameters())

    # Modify some parameters
    newparams = {'max_depth': 10, 'oob_honest': False}
    fr.set_parameters(**newparams)
    fr.set_parameters(seed=1729)

    # Check out the new parameters
    print(fr.get_parameters())


.. _train_test:

Training and Prediction
-----------------------

Here is an example of how to train a RandomForest estimator and use it to make
predictions.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Getting the dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Splitting the data into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create a RandomForest object
    fr = RandomForest(scale=False)

    print('Traingng the forest')
    fr.fit(X_train, y_train)

    print('Making predictions')
    preds = fr.predict(X_test)

    print('The coefficient of determination is ' +
            str(fr.score(X_test, y_test)))


.. _custom:

Custom Sampling Methods
-----------------------
The three parameters `groups`, `minTreesPerFold`, and `foldSize` are used for customizing the sampling scheme in a
random forest model.

The `groups` parameter specifies the group membership of each training observation, which is used in the
aggregation when doing out of bag predictions. This allows the user to specify custom subgroups which will
be used to create predictions that do not use any data from a common group to make predictions for any observation
in the group.

The `minTreesPerFold` parameter specifies the minimum number of trees that we make sure have been created
leaving out each fold of groups. If this parameter is set to a positive integer, the bootstrap sampling
scheme is modified to ensure that exactly that many trees have each group left out. This is achieved by,
for each fold, creating `minTreesPerFold` trees which are built on observations sampled from the set of training
observations which are not in a group in the current fold.

The `foldSize` parameter specifies the number of groups that are selected randomly for each fold to be left
out when using `minTreesPerFold`. When `minTreesPerFold` is set and `foldSize` is set, all possible groups will be
partitioned into folds, each containing `foldSize` unique groups. Then `minTreesPerFold` trees are grown with each
entire fold of groups left out. If `ntree` is greater than the product of the number of folds and `minTreesPerFold`,
we create at least `max(# folds * minTreesPerFold, ntree)` total trees, in which at least `minTreesPerFold` are created
leaving out each fold.

In summary, these parameters allow the user to create custom resampling schemes and provide predictions consistent
with the out-of-group set. They provide more control over the sampling process in the random forest model and
can be useful in situations where the default sampling scheme is not appropriate.

.. _missingness:

Treatment of Missing Data
-------------------------

For the handling of missing data, we now test any potential split by putting all NA's to the right, and
all NA's to the left, and taking  the choice which gives the best MSE for the split. Under this version of handling
the potential splits, we will still respect monotonic constraints. So if we put all NA's to either side, and the resulting leaf nodes have means which violate
the monotone constraints, the split will be rejected.


Monotonic Constraints
---------------------

This example shows how to set the monotonic constraints. They must be specified using an array of size *ncol* specifying monotonic
relationships between the continuous features and the outcome. Its entries are in -1, 0, 1, in which
1 indicates an increasing monotonic relationship, -1 indicates a decreasing monotonic relationship, and 0 indicates no constraint.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Gives a positive monotonic relationship between the first feature and the outcome
    # a negative monotonic relationship between the second feature, and no
    # constraints for the other features.
    constraints = np.array([1, -1, 0, 0])
    # Create a RandomForest object
    fr = RandomForest(oob_honest=True, scale=False, monotonic_constraints=constraints)

    print('Traingng the forest')
    fr.fit(X, y)

    print('Making out-of-bag predictions')
    preds = fr.predict(aggregation='oob')
    print('OOB ERROR: ' + str(fr.get_oob()))


.. _categorical:

Handling Categorical Data
-------------------------

Splits are made differently for categorical features. In order for the program to recognize that a given
feature is categorical rather than continuous, the user must convert it into a
`Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_.

.. note::

    If a feature data is not numeric, the program will automatically consider it as a `Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_.

Here is an example of how to use categorical features.

.. code-block:: Python

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from random_forestry import RandomForest

    # Getting the dataset
    data = load_diabetes(as_frame=True, scaled=False).frame
    X = data.iloc[:, :-1]
    y = data['target']

    # Making 'sex' categorical
    X['sex'] = X['sex'].astype('category')

    # Splitting the data into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Initialize a train
    fr = RandomForest()
    print('training the model')
    fr.fit(X_train, y_train)

    # Make predictions
    print('making predictions')
    preds = fr.predict(X_test)

    print('The coefficient of determination is ' +
                str(fr.score(X_test, y_test)))


.. _oob:

Prediction with Out-of-Bag Aggregation
--------------------------------------

This is an example of using out-of-bag aggregation. Check out :meth:`predict(..., aggregation='oob') <forestry.RandomForest.predict>`
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object
    fr = RandomForest(oob_honest=True, scale=False)

    print('Traingng the forest')
    fr.fit(X, y)

    print('Making out-of-bag predictions')
    preds = fr.predict(aggregation='oob')
    print('OOB ERROR: ' + str(fr.get_oob()))


.. _doubleOOB:

Prediction with Double Out-of-Bag Aggregation
---------------------------------------------

This is an example of using double OOB aggregation. Check out :meth:`predict(..., aggregation='doubleOOB') <forestry.RandomForest.predict>`
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object
    fr = RandomForest(oob_honest=True, double_bootstrap=True, scale=False)

    print('Training the forest')
    fr.fit(X, y)

    print('Making doubleOOB predictions')
    preds = fr.predict(aggregation='doubleOOB')
    print(preds)


.. _vi:

Variable Importance
-------------------

This is an example how to get the variable importance. Check out the :meth:`API <forestry.RandomForest.get_vi>`
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_breast_cancer
    import numpy as np

    # Getting the dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Create a RandomForest object and train
    fr = RandomForest(scale=False, max_depth=50)
    fr.fit(X, y)

    var_importance = fr.get_vi()
    print(var_importance)


.. _tree_struc:

Retrieve the Tree Structure
---------------------------

This is an example of how to retrieve the underlying tree structure in the forest. To do that,
we need to use the :meth:`translate_tree() <forestry.RandomForest.translate_tree>` function,
which fills the :ref:`saved_forest <translate-label>` attribute for the corresponding tree.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object and train
    fr = RandomForest(scale=False, max_depth=50)
    fr.fit(X, y)

    # Translate the first tree in the forest
    fr.translate_tree(0)
    print(fr.saved_forest[0])

