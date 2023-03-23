.. random_forestry documentation master file, created by
   sphinx-quickstart on Thu Mar 23 13:22:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to random_forestry's documentation!
===========================================

.. note::

   This project is under active development.

**random_forestry** is a Python library that contains a fast implementation
of Honest Random Forests, decision trees, and Linear Random Forests, with an emphasis on inference and interpretability.
Training and predicting with a model matches the scikit-learn API in order to
allow easy integration with existing packages.

Several features are specific to this library and allow for the training of
tree based models with useful properties, including:
 - **Honesty**: the practice of splitting the training data set into two disjoint sets
   and using one set to determine the tree structure of a decision tree, and the
   other to determine the node values used for predictions.
 - **Monotonic Constraints**: the ability to specify a constraint on the decision tree
   or random forest model that forces the prediction of the model to be monotonic
 - **Custom Bootstrap sampling methods**: several different methods to determine how the samples for
   each tree in a forest are selected. This can be useful when the observations
   are not i.i.d. and the sampling scheme can exploit patterns in the data. This also
   interacts with honesty.
 - **Linear aggregation**: rather than using the average training outcome to predict for new observations,
   one can run a Ridge regression within each terminal node and use this for the predictions
   of that tree. Tree construction is modified according to an algorithm proposed
   in `Linear Aggregation in Tree-based Estimators` (https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2026780)
   to select recursive splitting points that are optimal for downstream Ridge regression aggregation.

These are some of the current interesting features, and more will be added in the future.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::

   usage
   api

