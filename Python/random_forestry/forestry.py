import dataclasses
import math
import os
import pickle  # nosec B403 - 'Consider possible security implications associated with pickle'
import sys
import warnings
from pathlib import Path
from random import randrange
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (  # pylint: disable=no-name-in-module
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    confloat,
    conint,
)
from pydantic.dataclasses import dataclass

from . import extension, preprocessing  # type: ignore
from .processed_dta import ProcessedDta
from .validators.fit_validator import FitValidator
from .validators.predict_validator import PredictValidator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, smart_union=True, validate_all=True))
class RandomForest:
    """
    The Random Forest Regressor class.

    :param ntree: The number of trees to grow in the forest.
    :type ntree: *int, optional, default=500*
    :param replace: An indicator of whether sampling of the training data is done with replacement.
    :type replace: *bool, optional, default=True*
    :param sampsize: The size of total samples to draw for the training data. If sampling with replacement, the default
     value is the length of the training data. If sampling without replacement, the default value is two-thirds of the
     length of the training data.
    :type sampsize: *int, optional*
    :param sample_fraction: If this is given, then sampsize is ignored and set to
     be ``round(len(y) * sample_fraction)`` . It must be a real number between 0 and 1.
    :type sample_fraction: *float, optional*
    :param mtry: The number of variables randomly selected at each split point. The default value is set to be
     one-third of the total number of features of the training data.
    :type mtry: *int, optional*
    :param nodesize_spl: Minimum observations contained in terminal nodes.
    :type nodesize_spl: *int, optional, default=5*
    :param nodesize_avg: Minimum size of terminal nodes for averaging dataset.
    :type nodesize_avg: *int, optional, default=5*
    :param nodesize_strict_spl: Minimum observations to follow strictly in terminal nodes.
    :type nodesize_strict_spl: *int, optional, default=1*
    :param nodesize_strict_avg: The minimum size of terminal nodes for averaging data set to follow when predicting.
     No splits are allowed that result in nodes with observations less than this parameter.
     This parameter enforces overlap of the averaging data set with the splitting set when training.
     When using honesty, splits that leave less than nodesizeStrictAvg averaging
     observations in either child node will be rejected, ensuring every leaf node
     also has at least nodesizeStrictAvg averaging observations.
    :type nodesize_strict_avg: *int, optional, default=1*
    :param min_split_gain: Minimum loss reduction to split a node further in a tree.
    :type min_split_gain: *float, optional, default=0*
    :param max_depth: Maximum depth of a tree.
    :type max_depth: *int, optional, default=99*
    :param interaction_depth: All splits at or above interaction depth must be on variables
     that are not weighting variables (as provided by the interactionVariables argument in fit).
    :type interaction_depth: *int, optional, default=maxDepth*
    :param splitratio: Proportion of the training data used as the splitting dataset.
     It is a ratio between 0 and 1. If the ratio is 1 (the default), then the splitting
     set uses the entire data, as does the averaging set---i.e., the standard Breiman RF setup.
     If the ratio is 0, then the splitting data set is empty, and the entire dataset is used
     for the averaging set (This is not a good usage, however, since there will be no data available for splitting).
    :type splitratio: *double, optional, default=1*
    :param oob_honest: In this version of honesty, the out-of-bag observations for each tree
     are used as the honest (averaging) set. This setting also changes how predictions
     are constructed. When predicting for observations that are out-of-sample
     ``(predict(..., aggregation = "average"))`` , all the trees in the forest
     are used to construct predictions. When predicting for an observation that was in-sample
     ``(predict(..., aggregation = "oob"))`` , only the trees for which that observation
     was not in the averaging set are used to construct the prediction for that observation.
     *aggregation="oob"* (out-of-bag) ensures that the outcome value for an observation
     is never used to construct predictions for a given observation even when it is in sample.
     This property does not hold in standard honesty, which relies on an asymptotic
     subsampling argument. By default, when *oob_honest=True*, the out-of-bag observations
     for each tree are resamples with replacement to be used for the honest (averaging)
     set. This results in a third set of observations that are left out of both
     the splitting and averaging set, we call these the double out-of-bag (doubleOOB)
     observations. In order to get the predictions of only the trees in which each
     observation fell into this doubleOOB set, one can run ``predict(... , aggregation = "doubleOOB")`` .
     In order to not do this second bootstrap sample, the doubleBootstrap flag can
     be set to *False*.
    :type oob_honest: *bool, optional, default=False*
    :param double_bootstrap: The doubleBootstrap flag provides the option to resample
     with replacement from the out-of-bag observations set for each tree to construct
     the averaging set when using OOBhonest. If this is *False*, the out-of-bag observations
     are used as the averaging set. By default this option is *True* when running *oob_honest=True*.
     This option increases diversity across trees.
    :type double_bootstrap: *bool, optional, default=oob_honest*
    :param seed: Random number generator seed. The default value is a random integer.
    :type seed: *int, optional*
    :param verbose: Indicator to train the forest in verbose mode.
    :type verbose: *bool, optional, default=False*
    :param nthread: Number of threads to train and predict the forest. The default
     number is 0 which represents using all cores.
    :type nthread: *int, optional, default=0*
    :param splitrule: Only variance is implemented at this point and, it
     specifies the loss function according to which the splits of random forest
     should be made.
    :type splitrule: *str, optional, default='variance'*
    :param middle_split: Indicator of whether the split value is takes the average of two feature
     values. If *False*, it will take a point based on a uniform distribution
     between two feature values.
    :type middle_split: *bool, optional, default=False*
    :param max_obs: The max number of observations to split on. The default is the number of observations.
    :type max_obs: *int, optional*
    :param linear: Indicator that enables Ridge penalized splits and linear aggregation
     functions in the leaf nodes. This is recommended for data with linear outcomes.
     For implementation details, see: https://arxiv.org/abs/1906.06463.
    :type linear: *bool, optional, default=False*
    :param min_trees_per_fold: The number of trees which we make sure have been created leaving
     out each fold (each fold is a set of randomly selected groups).
     This is 0 by default, so we will not give any special treatment to
     the groups when sampling observations, however if this is set to a positive integer, we
     modify the bootstrap sampling scheme to ensure that exactly that many trees
     have each group left out. We do this by, for each fold, creating min_trees_per_fold
     trees which are built on observations sampled from the set of training observations
     which are not in a group in the current fold. The folds form a random partition of
     all of the possible groups, each of size foldSize. This means we create at
     least # folds * min_trees_per_fold trees for the forest.
     If ntree > # folds * min_trees_per_fold, we create
     max(# folds * min_trees_per_fold, ntree) total trees, in which at least min_trees_per_fold
     are created leaving out each fold.
    :type min_trees_per_fold: *int, optional, default=0*
    :param fold_size: The number of groups that are selected randomly for each fold to be
     left out when using minTreesPerFold. When minTreesPerFold is set and foldSize is
     set, all possible groups will be partitioned into folds, each containing foldSize unique groups
     (if foldSize doesn't evenly divide the number of groups, a single fold will be smaller,
     as it will contain the remaining groups). Then minTreesPerFold are grown with each
     entire fold of groups left out.
    :type fold_size: *int, optional, default=1*
    :param monotone_avg: This is a flag that indicates whether or not monotonic
     constraints should be enforced on the averaging set in addition to the splitting set.
     This flag is meaningless unless both honesty and monotonic constraints are in use.
    :type monotone_avg: *bool, optional, default=False*
    :param overfit_penalty: Value to determine how much to penalize the magnitude
     of coefficients in ridge regression when using linear splits.
    :type overfit_penalty: *float, optional, default=1*
    :param scale: A parameter which indicates whether or not we want to scale and center
     the covariates and outcome before doing the regression. This can help with
     stability, so the default is *True*.
    :param na_direction: Sets a default direction for missing values in each split
     node during training. It test placing all missing values to the left and
     right, then selects the direction that minimizes loss. If no missing values
     exist, then a default direction is randomly selected in proportion to the
     distribution of observations on the left and right. (Default = FALSE)
    :type na_direction: *bool, optional, default=False*
    :type scale: *bool, optional, default=True*
    :param double_tree: Indicator of whether the number of trees is doubled as averaging and splitting
     data can be exchanged to create decorrelated trees.
    :type double_tree: *bool, optional, default=False*

    :ivar processed_dta: A data structure containing information about the data after it has been preprocessed.
     *processed_dta* has the following entries:

     * processed_x (*pandas.DataFrame*) - The processed feature matrix.

     * y (*numpy.array of shape[nrows,]*) - The processed target values.

     * categorical_feature_cols (*numpy.array*) - An array of the indices of the categorical features
       in the feature matrix.

      .. note::
        In order for the program to recognize a feature as categorical, it **must** be converted into a
        `Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_. The
        simplest way to do it is to use::

            df['categorical'] = df['categorical'].astype('category')

        Check out the :ref:`Handling Categorical Data <categorical>` section for an example
        of how to use categorical features.

     * categorical_feature_mapping (*list[dict]*) - For each categorical feature, the data is encoded into
       numeric represetation. Those encodings are saved in *categoricalFeatureMapping*. *categoricalFeatureMapping[i]*
       and has the following entries:

        * categorical_feature_col (*int*) - The index of the current categorical feature column.

        * unique_feature_values (*list*) - The categories of the current categorical feature.

        * numeric_feature_values (*numpy.array*) - The categories of the current categorical feature encoded
          into numeric represetation.

     * feature_weights (*numpy.array of shape[ncols]*) - an array of sampling probabilities/weights for each feature
       used when subsampling *mtry* features at each node.
       Check out :meth:`fit() <random_forestry.RandomForest.fit>` fot more details.

     * feature_weights_variables (*numpy.array*) - Indices of the features which weight
       more than ``max(feature_weights)*0.001`` .

     * deep_feature_weights (*numpy.array of shape[ncols]*) - Used in place of *feature_weights* for splits
       below *interaction_depth*. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * deep_feature_weights_variables (*numpy.array*) - Indices of the features which weight more than
       ``max(deep_feature_weights)*0.001`` .

     * observation_weights (*numpy.array of shape[nrows]*) - Denotes the weights for each training observation that
       determine how likely the observation is to be selected in each bootstrap sample.
       Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * monotonic_constraints (*numpy.array of shape[ncols]*) - An array of size *ncol* specifying monotonic
       relationships between the continuous features and the outcome. Its entries are in -1, 0, 1, in which
       1 indicates an increasing monotonic relationship, -1 indicates a decreasing monotonic relationship, and
       0 indicates no constraint. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * group_memberships(*numpy.array of shape[nrows]*) - Factorized group membership of each training observation.

     * linear_feature_cols (*numpy.array*) - An array containing the indices of which features to split linearly on
       when using linear penalized splits. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * col_means (*numpy.array of shape[ncols]*) - The mean value of each column.

     * col_sd (*numpy.array of shape[ncols]*) - The standard deviation of each column.

     * has_nas (*bool*) - Specifies whether the feature matrix contains missing observations or not.

     * na_direction (*bool*) - Sets a default direction for missing values in each split node during training

     * n_observations (*int*) - The number of observations in the training data.

     * num_columns (*int*) - The number of features in the training data.

     * feat_names (*numpy.array of shape[ncols]*) - The names of the features used for training.

     Note that **all** of the entries in processed_dta are set to ``None`` or empty containers during initialization.
     They are only assigned a value after :meth:`fit() <forestry.RandomForest.fit>` is called.
    :vartype processed_dta: ProcessedDta

     .. _translate-label:

    :ivar saved_forest: For any tree *i* in the forest, *saved_forest[i]* is a dictionary which gives access to the
     underlying structrure of that tree. *saved_forest[i]* has the following entries:

     * children_right (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*,
       *children_right[id]* gives the id of the right child of that node. If leaf node, *children_right[id]* is *-1*.

     * children_left (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*,
       *children_left[id]* gives the id of the left child of that node. If leaf node, *children_left[id]* is *-1*.

     * feature (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *feature[id]*
       gives the index of the splitting feature in that node. If leaf node, *feature[id]* is the negative number of
       observations in the averaging set of that node.

     * n_node_samples (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*,
       *feature[id]* gives the number of observations in the averaging set of that node.

     * threshold (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *threshold[id]*
       gives the splitting point (threshold) of the split in that node. If leaf node, *threshold[id]* is *0.0*.

     * values (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, if that node is
       a leaf node, *values[id]* gives the prediction made by that node. Otherwise, *values[id]* is *0.0*.

     .. note::
        When a *RandomForest* is initialized, *saved_forest* is set to a list of *ntree* empty dictionaries. In order to
        populate those dictionaries, the :meth:`translate_tree() <random_forestry.RandomForest.translate_tree>` method
        must be used.

    :vartype saved_forest: list[dict]
    :ivar forest: A ctypes pointer to the *forestry* object in C++. It is initially set to *None* and updated only
     after :meth:`fit() <forestry.RandomForest.fit>` is called.
    :vartype forest: ctypes.c_void_p
    :ivar dataframe: A ctypes pointer to the *DataFrame* object in C++. It is initially set to *None* and updated only
     after :meth:`fit() <forestry.RandomForest.fit>` is called.
    :vartype dataframe: ctypes.c_void_p

    """

    ntree: conint(gt=0, strict=True) = 500
    replace: StrictBool = True
    sampsize: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    sample_fraction: Optional[Union[conint(gt=0, strict=True), confloat(gt=0, strict=True)]] = None
    mtry: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    nodesize_spl: conint(gt=0, strict=True) = 5
    nodesize_avg: conint(gt=0, strict=True) = 5
    nodesize_strict_spl: conint(gt=0, strict=True) = 1
    nodesize_strict_avg: conint(gt=0, strict=True) = 1
    min_split_gain: confloat(ge=0) = 0
    max_depth: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    interaction_depth: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    splitratio: confloat(ge=0, le=1) = 1.0
    oob_honest: StrictBool = False
    double_bootstrap: Optional[StrictBool] = None  # Add a default value.
    seed: conint(ge=0, strict=True) = randrange(1001)  # nosec B311
    verbose: StrictBool = False
    nthread: conint(ge=0, strict=True) = 0
    splitrule: str = "variance"
    middle_split: StrictBool = False
    max_obs: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    linear: StrictBool = False
    min_trees_per_fold: conint(ge=0, strict=True) = 0
    fold_size: conint(gt=0, strict=True) = 1
    monotone_avg: StrictBool = False
    overfit_penalty: Union[StrictInt, StrictFloat] = 1
    scale: StrictBool = False
    double_tree: StrictBool = False
    na_direction: StrictBool = False

    forest: Optional[pd.DataFrame] = dataclasses.field(default=None, init=False)
    dataframe: Optional[pd.DataFrame] = dataclasses.field(default=None, init=False)
    processed_dta: Optional[ProcessedDta] = dataclasses.field(default=None, init=False)
    saved_forest: List[Dict] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.nthread > os.cpu_count():
            raise ValueError("nthread cannot exceed total cores in the computer: " + str(os.cpu_count()))

        if self.min_split_gain > 0 and not self.linear:
            raise ValueError("min_split_gain cannot be set without setting linear to be true.")

        if self.double_bootstrap is None:
            self.double_bootstrap = self.oob_honest

        if self.oob_honest and (self.splitratio != 1):
            warnings.warn("oob_honest is set to true, so we will run OOBhonesty rather than standard honesty.")
            self.splitratio = 1

        if self.oob_honest and not self.replace:
            warnings.warn("replace must be set to TRUE to use OOBhonesty, setting this to True now")
            self.replace = True

        if self.double_tree and self.splitratio in (0, 1):
            warnings.warn("Trees cannot be doubled if splitratio is 1. We have set double_tree to False.")
            self.double_tree = False

        if (
            self.interaction_depth is not None
            and self.max_depth is not None
            and self.interaction_depth > self.max_depth
        ):
            warnings.warn(
                "interaction_depth cannot be greater than max_depth. We have set interaction_depth to max_depth."
            )
            self.interaction_depth = self.max_depth

    def _get_seed(self, seed: Optional[int]) -> int:
        if seed is None:
            return self.seed
        if (not isinstance(seed, int)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        return seed

    def _set_nodesize_strict(self) -> None:
        # if the splitratio is 1, then we use adaptive rf and avgSampleSize is
        # equal to the total sampsize

        if self.splitratio in (0, 1):
            split_sample_size = self.sampsize
            avg_sample_size = self.sampsize
        else:
            split_sample_size = self.splitratio * self.sampsize
            avg_sample_size = math.floor(self.sampsize - split_sample_size)
            split_sample_size = math.floor(split_sample_size)

        if self.nodesize_strict_spl > split_sample_size:
            warnings.warn(
                "nodesizeStrictSpl cannot exceed splitting sample size. ",
                "We have set nodesizeStrictSpl to be the maximum.",
            )
            self.nodesize_strict_spl = split_sample_size

        if self.nodesize_strict_avg > avg_sample_size:
            warnings.warn(
                "nodesizeStrictAvg cannot exceed averaging sample size. ",
                "We have set nodesizeStrictAvg to be the maximum.",
            )
            self.nodesize_strict_avg = avg_sample_size

        if self.double_tree:
            if self.nodesize_strict_avg > split_sample_size:
                warnings.warn(
                    "nodesizeStrictAvg cannot exceed splitting sample size. ",
                    "We have set nodesizeStrictAvg to be the maximum.",
                )
                self.nodesize_strict_avg = split_sample_size
            if self.nodesize_strict_spl > avg_sample_size:
                warnings.warn(
                    "nodesize_strict_spl cannot exceed averaging sample size. ",
                    "We have set nodesize_strict_spl to be the maximum.",
                )
                self.nodesize_strict_spl = avg_sample_size

    def _get_weights_variables(self, weights: np.ndarray) -> np.ndarray:
        weights_variables = [i for i in range(weights.size) if weights[i] > max(weights) * 0.001]
        if len(weights_variables) < self.mtry:
            raise ValueError("mtry is too large. Given the feature weights, can't select that many features.")

        weights_variables = np.array(weights_variables, dtype=np.ulonglong)
        return weights_variables

    def _get_group_memberships(self, nrow: int, groups: Optional[pd.Series]) -> np.ndarray:
        if groups is None:
            return np.zeros(nrow, dtype=np.ulonglong)
        codes, levels = pd.factorize(groups)
        # Increment array to avoid having 0s
        codes += 1

        # Print warning if the group number and minTreesPerFold results in a large forest
        if self.min_trees_per_fold > 0 and (-(len(levels) // -self.fold_size)) * self.min_trees_per_fold > 2000:
            warnings.warn(
                "Using "
                + str(len(levels))
                + " groups with fold size "
                + str(self.fold_size)
                + " and "
                + str(self.min_trees_per_fold)
                + " trees per fold will train "
                + str(len(levels) * self.min_trees_per_fold)
                + " trees in the forest"
            )
        return codes

    @FitValidator
    def fit(
        self,
        x: Union[pd.DataFrame, pd.Series, List],
        y: np.ndarray,
        *,
        interaction_variables: Optional[List] = None,  # pylint: disable=unused-argument
        feature_weights: Optional[np.ndarray] = None,
        deep_feature_weights: Optional[np.ndarray] = None,
        observation_weights: Optional[np.ndarray] = None,
        lin_feats: Optional[Union[np.ndarray, List]] = None,  # Add a default value.
        monotonic_constraints: Optional[np.ndarray] = None,  # Add a default value.
        groups: Optional[pd.Series] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Trains all the trees in the forest.

        :param x: The feature matrix.
        :type x: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nrows, ncols]*
        :param y: The target values.
        :type y: *array_like of shape [nrows,]*
        :param interactionVariables: Indices of weighting variables.
        :type interactionVariables: *array_like, optional, default=[]*
        :param featureWeights: a list of sampling probabilities/weights for each
         feature used when subsampling *mtry* features at each node above or at *interactionDepth*.
         The default is to use uniform probabilities.
        :type featureWeights: *array_like of shape [ncols,], optional*
        :param deepFeatureWeights: Used in place of *featureWeights* for splits below *interactionDepth*.
         The default is to use uniform probabilities.
        :type deepFeatureWeights: *array_like of shape [ncols,], optional*
        :param observationWeights: Denotes the weights for each training observation
         that determine how likely the observation is to be selected in each bootstrap sample.
         The default is to use uniform probabilities. This option is not allowed when sampling is
         done without replacement.
        :type observationWeights: *array_like of shape [nrows,], optional*
        :param linFeats: A list containing the indices of which features to split
         linearly on when using linear penalized splits (defaults to use all numerical features).
        :type linFeats: *array_like, optional*
        :param monotonicConstraints: Specifies monotonic relationships between the continuous
         features and the outcome. Supplied as a list of length *ncol* with entries in
         1, 0, -1, with 1 indicating an increasing monotonic relationship, -1 indicating
         a decreasing monotonic relationship, and 0 indicating no constraint.
         Constraints supplied for categorical variable will be ignored. Defaults to all 0-s (no constraints).
        :type monotonicConstraints: *array_like of shape [ncols,], optional*
        :param groups: A pandas series specifying the group membership of each training observation.
         These groups are used in the aggregation when doing out of bag predictions in
         order to predict with only trees where the entire group was not used for aggregation.
         This allows the user to specify custom subgroups which will be used to create
         predictions which do not use any data from a common group to make predictions for
         any observation in the group. This can be used to create general custom
         resampling schemes, and provide predictions consistent with the Out-of-Group set.
        :type groups: *pandas.Series, optional*,
         or other pandas categorical dtypes, optional, default=None*
        :param seed: Random number generator seed. The default value is the *RandomForest* seed.
        :type seed: *int, optional*
        :rtype: None
        """

        # Make sure that all the parameters exist when passed to RandomForest

        feat_names = preprocessing.get_feat_names(x)

        x = (pd.DataFrame(x)).copy()
        y = (np.array(y, dtype=np.double)).copy()

        nrow, ncol = x.shape

        if self.max_depth is None:
            self.max_depth = round(nrow / 2) + 1

        if self.interaction_depth is None:
            self.interaction_depth = self.max_depth

        if self.max_obs is None:
            self.max_obs = y.size

        self.sampsize = preprocessing.get_sampsize(self, x)
        self.mtry = preprocessing.get_mtry(self, x)

        self._set_nodesize_strict()

        feature_weights_variables = self._get_weights_variables(feature_weights)
        deep_feature_weights_variables = self._get_weights_variables(deep_feature_weights)

        feature_weights /= np.sum(feature_weights)
        deep_feature_weights /= np.sum(deep_feature_weights)
        if self.replace:
            observation_weights /= np.sum(observation_weights)

        group_memberships = self._get_group_memberships(nrow, groups)

        (
            processed_x,
            categorical_feature_cols,
            categorical_feature_mapping,
        ) = preprocessing.preprocess_training(x, y)

        if categorical_feature_cols.size != 0:
            monotonic_constraints[categorical_feature_cols] = 0

        col_means = col_sd = np.repeat(0.0, ncol + 1)
        if self.scale:
            processed_x, y, col_means, col_sd = preprocessing.scale(x, y, processed_x, categorical_feature_cols)

        # cpp linking
        processed_x.reset_index(drop=True, inplace=True)

        self.dataframe: pd.DataFrame = extension.get_data(
            np.ascontiguousarray(pd.concat([processed_x, pd.Series(y)], axis=1).values[:, :], np.double).ravel(),
            categorical_feature_cols,
            categorical_feature_cols.size,
            lin_feats,
            lin_feats.size,
            feature_weights,
            feature_weights_variables,
            feature_weights_variables.size,
            deep_feature_weights,
            deep_feature_weights_variables,
            deep_feature_weights_variables.size,
            observation_weights,
            monotonic_constraints,
            group_memberships,
            self.monotone_avg,
            nrow,
            ncol + 1,
            self._get_seed(seed),
        )

        self.forest: pd.DataFrame = extension.train_forest(
            self.dataframe,
            self.ntree,
            self.replace,
            self.sampsize,
            self.splitratio,
            self.oob_honest,
            self.double_bootstrap,
            self.mtry,
            self.nodesize_spl,
            self.nodesize_avg,
            self.nodesize_strict_spl,
            self.nodesize_strict_avg,
            self.min_split_gain,
            self.max_depth,
            self.interaction_depth,
            self._get_seed(seed),
            self.nthread,
            self.verbose,
            self.middle_split,
            self.max_obs,
            self.min_trees_per_fold,
            self.fold_size,
            x.isnull().values.any(),
            self.na_direction,
            self.linear,
            self.overfit_penalty,
            self.double_tree,
        )

        # Update the fields
        self.processed_dta = ProcessedDta(
            processed_x=processed_x,
            y=y,
            categorical_feature_cols=categorical_feature_cols,
            categorical_feature_mapping=categorical_feature_mapping,
            feature_weights=feature_weights,
            feature_weights_variables=feature_weights_variables,
            deep_feature_weights=deep_feature_weights,
            deep_feature_weights_variables=deep_feature_weights_variables,
            observation_weights=observation_weights,
            monotonic_constraints=monotonic_constraints,
            linear_feature_cols=lin_feats,
            groups=group_memberships,
            col_means=col_means,
            col_sd=col_sd,
            has_nas=x.isnull().values.any(),
            n_observations=nrow,
            num_columns=ncol,
            feat_names=feat_names,
        )

    def _get_test_data(self, newdata: Optional[pd.DataFrame]) -> np.ndarray:
        if newdata is None:
            return np.ascontiguousarray(self.processed_dta.processed_x.values[:, :], np.double).ravel()

        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        return np.ascontiguousarray(processed_x.values[:, :], np.double).ravel()

    def _get_n_preds(self, newdata: Optional[pd.DataFrame]) -> int:
        if newdata is None:
            return self.processed_dta.n_observations

        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        return len(processed_x.index)

    def _scale_ret_values(
        self, ret_values: Optional[Tuple[np.ndarray, np.ndarray]], include_coefficients: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if include_coefficients:
            return (
                ret_values[0] * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1],
                ret_values[1],
                ret_values[2],
            )
        return (ret_values[0] * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1], ret_values[1])

    def _aggregation_oob(
        self,
        newdata: Optional[pd.DataFrame],
        exact: bool,
        return_weight_matrix: bool,
        training_idx: Optional[np.ndarray],
        hier_shrinkage_lambda: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if newdata is not None:
            processed_x = preprocessing.preprocess_testing(
                newdata,
                self.processed_dta.categorical_feature_cols,
                self.processed_dta.categorical_feature_mapping,
            )
            if len(processed_x.index) != self.processed_dta.n_observations and training_idx is None:
                raise ValueError("Attempting to do OOB predictions on a dataset which doesn't match the training data!")
            if training_idx and len(training_idx) != len(newdata.index):
                raise ValueError("Training Indices must be of the same length as newdata")
            if self.scale:
                processed_x = preprocessing.scale_center(
                    processed_x,
                    self.processed_dta.categorical_feature_cols,
                    self.processed_dta.col_means,
                    self.processed_dta.col_sd,
                )

        if training_idx and (
            not np.issubdtype(training_idx.dtype, np.integer)
            or np.any((training_idx < 0) | (training_idx >= self.processed_dta.n_observations))
        ):
            raise ValueError(
                "Training Indices must contain integers between 0 and the number of training observations - 1"
            )

        if hier_shrinkage_lambda is not None:
            if hier_shrinkage_lambda < 0:
                raise ValueError("Hierarchical shrinkage parameter must be positive!")
            else:
                hier_shrinkage = True
                lambda_shrinkage = hier_shrinkage_lambda
        else:
            hier_shrinkage = False
            lambda_shrinkage = 0

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0

        ret_values = extension.predict_oob_forest(
            self.forest,
            self.dataframe,
            self._get_test_data(processed_x),
            False,
            exact,
            return_weight_matrix,
            self.verbose,
            training_idx is not None,
            n_preds,
            n_weight_matrix,
            training_idx if training_idx else [],
            hier_shrinkage,
            lambda_shrinkage,
        )
        # If the forest was trained with scaled values we need to rescale + re center the predictions
        if self.scale:
            return self._scale_ret_values(ret_values)
        else:
            return ret_values

    def _aggregation_double_oob(
        self,
        newdata: Optional[pd.DataFrame],
        exact: bool,
        return_weight_matrix: bool,
        training_idx: Optional[np.ndarray],
        hier_shrinkage_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if newdata is not None:
            processed_x = preprocessing.preprocess_testing(
                newdata,
                self.processed_dta.categorical_feature_cols,
                self.processed_dta.categorical_feature_mapping,
            )
            if len(processed_x.index) != self.processed_dta.n_observations and training_idx is None:
                raise ValueError("Attempting to do OOB predictions on a dataset which doesn't match the training data!")
            if training_idx and len(training_idx) != len(newdata.index):
                raise ValueError("Training Indices must be of the same length as newdata")
            if self.scale:
                processed_x = preprocessing.scale_center(
                    processed_x,
                    self.processed_dta.categorical_feature_cols,
                    self.processed_dta.col_means,
                    self.processed_dta.col_sd,
                )

        if training_idx and (
            not np.issubdtype(training_idx.dtype, np.integer)
            or np.any((training_idx < 0) | (training_idx >= self.processed_dta.n_observations))
        ):
            raise ValueError(
                "Training Indices must contain integers between 0 and the number of training observations - 1"
            )

        if hier_shrinkage_lambda is not None:
            if hier_shrinkage_lambda < 0:
                raise ValueError("Hierarchical shrinkage parameter must be positive!")
            hier_shrinkage = True
            lambda_shrinkage = hier_shrinkage_lambda
        else:
            hier_shrinkage = False
            lambda_shrinkage = 0

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0

        ret_values = extension.predict_oob_forest(
            self.forest,
            self.dataframe,
            self._get_test_data(processed_x),
            True,
            exact,
            return_weight_matrix,
            self.verbose,
            training_idx is not None,
            n_preds,
            n_weight_matrix,
            training_idx if training_idx else [],
            hier_shrinkage,
            lambda_shrinkage,
        )

        # If the forest was trained with scaled values we need to rescale + re center the predictions
        if self.scale:
            return self._scale_ret_values(ret_values)
        else:
            return ret_values

    def _aggregation_coefs(
        self, newdata: pd.DataFrame, exact: bool, seed: int, nthread: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )

        if self.scale:
            processed_x = preprocessing.scale_center(
                processed_x,
                self.processed_dta.categorical_feature_cols,
                self.processed_dta.col_means,
                self.processed_dta.col_sd,
            )

        ret_values = extension.predict_forest(
            self.forest,
            self.dataframe,
            np.ascontiguousarray(processed_x.values[:, :], np.double).ravel(),
            seed,
            nthread,
            exact,
            False,
            True,
            False,
            np.zeros(self.ntree, dtype=np.ulonglong),
            len(processed_x.index),
            self._get_n_preds(newdata),
            0,
            self.processed_dta.n_observations * (self.processed_dta.linear_feature_cols.size + 1),
        )

        # If the forest was trained with scaled values we need to rescale + re center the predictions
        if self.scale:
            return self._scale_ret_values(ret_values, include_coefficients=True)
        else:
            return ret_values

    def _aggregation_fallback(
        self,
        newdata: pd.DataFrame,
        exact: bool,
        seed: int,
        nthread: int,
        return_weight_matrix: bool,
        trees: Optional[np.ndarray],
        hier_shrinkage_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        if self.scale:
            processed_x = preprocessing.scale_center(
                processed_x,
                self.processed_dta.categorical_feature_cols,
                self.processed_dta.col_means,
                self.processed_dta.col_sd,
            )

        tree_weights = np.zeros(self.ntree, dtype=np.ulonglong)
        if trees is not None:
            # If trees are being used, we need to convert them into a weight vector
            for tree in trees:
                tree_weights[tree] += 1
            use_weights = True
        else:
            use_weights = False

        if hier_shrinkage_lambda is not None:
            if hier_shrinkage_lambda < 0:
                raise ValueError("Hierarchical shrinkage parameter must be positive!")
            hier_shrinkage = True
            lambda_shrinkage = hier_shrinkage_lambda
        else:
            hier_shrinkage = False
            lambda_shrinkage = 0

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0

        ret_values = extension.predict_forest(
            self.forest,
            self.dataframe,
            np.ascontiguousarray(processed_x.values[:, :], np.double).ravel(),
            seed,
            nthread,
            exact,
            return_weight_matrix,
            False,
            use_weights,
            tree_weights,
            len(processed_x.index),
            n_preds,
            n_weight_matrix,
            0,
            hier_shrinkage,
            lambda_shrinkage,
        )

        # If the forest was trained with scaled values we need to rescale + re center the predictions
        if self.scale:
            return self._scale_ret_values(ret_values, include_coefficients=True)
        else:
            return ret_values

    @PredictValidator
    def predict(
        self,
        newdata: Optional[Union[pd.DataFrame, pd.Series, List]] = PredictValidator.DEFAULT_NEWDATA,
        *,
        aggregation: str = PredictValidator.DEFAULT_AGGREGATION,
        seed: Optional[int] = None,
        nthread: Optional[int] = None,
        exact: bool = True,
        trees: Optional[np.ndarray] = None,
        training_idx: Optional[np.ndarray] = None,
        return_weight_matrix: bool = False,
        hier_shrinkage_lambda: float = None,
    ) -> Union[np.ndarray, dict]:
        """
        Return the prediction from the forest.

        :param newdata: Testing predictors.
        :type newdata: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols],
         default=None*
        :param aggregation: How the individual tree predictions are aggregated:
         'average' returns the mean of all trees in the forest; 'terminalNodes' also returns
         the weightMatrix, as well as "terminalNodes" - a matrix where
         the i-th entry of the j-th column is the index of the leaf node to which the
         i-th observation is assigned in the j-th tree; and "sparse" - a matrix
         where the ioth entry in the j-th column is 1 if the ith observation in
         newdata is assigned to the j-th leaf and 0 otherwise. In each tree the
         leaves are indexed using a depth first ordering, and, in the "sparse"
         representation, the first leaf in the second tree has column index one more than
         the number of leaves in the first tree and so on. So, for example, if the
         first tree has 5 leaves, the sixth column of the "sparse" matrix corresponds
         to the first leaf in the second tree.
         'oob' returns the out-of-bag predictions for the forest. We assume
         that the ordering of the observations in newdata have not changed from
         training. If the ordering has changed, we will get the wrong OOB indices.
         'doubleOOB' is an experimental flag, which can only be used when *OOBhonest=True*
         and *doubleBootstrap=True*. When both of these settings are on, the
         splitting set is selected as a bootstrap sample of observations and the
         averaging set is selected as a bootstrap sample of the observations which
         were left out of bag during the splitting set selection. This leaves a third
         set which is the observations which were not selected in either bootstrap sample.
         For each observation, this predict flag gives the predictions using only the trees
         in which the observation fell into this third set (so was neither a splitting
         nor averaging example).
         'coefs' is an aggregation option which works only when linear aggregation
         functions have been used. This returns the linear coefficients for each
         linear feature which were used in the leaf node regression of each predicted point.
        :type aggregation: *str, optional, default='average'*
        :param seed: Random number generator seed. The default value is the *RandomForest* seed.
        :type seed: *int, optional*
        :param nthread: The number of threads with which to run the predictions with.
         This will default to the number of threads with which the forest was trained
         with.
        :type nthread: *int, optional*
        :param exact: This specifies whether the forest predictions should be aggregated
         in a reproducible ordering. Due to the non-associativity of floating point
         addition, when we predict in parallel, predictions will be aggregated in
         varied orders as different threads finish at different times.
         By default, exact is *True* unless ``N>100,000`` or a custom aggregation
         function is used.
        :type exact: *bool, optional*
        :param trees: A list of indices in the range *[0, ntree)*, which tells
         predict which trees in the forest to use for the prediction. Predict will by
         default take the average of all trees in the forest, although this flag
         can be used to get single tree predictions, or averages of different trees
         with different weightings.
             .. note::
                 Duplicate entries are allowed, so if ``trees = [0,1,1]``
                 this will predict the weighted average prediction of only trees 0 and 1 weighted by:
                 ``predict(..., trees = [0,1,1]) = (predict(..., trees = [0]) + 2*predict(..., trees = [1])) / 3``
                 we must have ``exact = True`` , and ``aggregation = "average"`` to use tree indices.
         Defaults to using all trees equally weighted.
        :type trees: *array_like, optional*
        :param training_idx: When doing OOB predictions with a data set that is of a different size than the
         training data, training_idx holds the indices of the training observations that should be used for
         determining the out-of-bag set for each observation in newdata. Entries must be between 1 and the number
         of training observations, and the length must be equal to the number of observations in newdata.
        :type training_idx: *array_like, optional*
        :param weightMatrix: An indicator of whether or not we should also return a
         matrix of the weights given to each training observation when making each
         prediction. When getting the weight matrix, aggregation must be one of
         'average', 'oob', and 'doubleOOB'. his is a normal text paragraph.
        :type weightMatrix: *bool, optional, default=False*
        :param hier_shrinkage_lambda The shrinkage parameter to use for hierarchical shrinkage.
         By default, this is set to 0 (equal to zero shrinkage).
        :type hier_shrinkage_lambda: *float, optional, default=None*
        :return: An array of predicted responses.
        :rtype: numpy.array
        """

        if aggregation == "oob":
            predictions, weight_matrix = self._aggregation_oob(
                newdata,
                exact,
                return_weight_matrix,
                training_idx,
                hier_shrinkage_lambda,
            )

        elif aggregation == "doubleOOB":
            predictions, weight_matrix = self._aggregation_double_oob(
                newdata,
                exact,
                return_weight_matrix,
                training_idx,
                hier_shrinkage_lambda,
            )

        elif aggregation == "coefs":
            predictions, weight_matrix, coefficients = self._aggregation_coefs(
                newdata,
                exact,
                self._get_seed(seed),
                nthread,
            )
            return {
                "predictions": predictions,
                "coef": np.lib.stride_tricks.as_strided(
                    coefficients,
                    shape=(
                        self.processed_dta.n_observations,
                        self.processed_dta.linear_feature_cols.size + 1,
                    ),
                    strides=(
                        coefficients.itemsize * (self.processed_dta.linear_feature_cols.size + 1),
                        coefficients.itemsize,
                    ),
                ),
            }

        else:
            predictions, weight_matrix, _ = self._aggregation_fallback(
                newdata,
                exact,
                self._get_seed(seed),
                nthread,
                return_weight_matrix,
                trees,
                hier_shrinkage_lambda,
            )

        if return_weight_matrix:
            return {
                "predictions": predictions,
                "weightMatrix": np.lib.stride_tricks.as_strided(
                    weight_matrix,
                    shape=(self._get_n_preds(newdata), self.processed_dta.n_observations),
                    strides=(
                        weight_matrix.itemsize * self.processed_dta.n_observations,
                        weight_matrix.itemsize,
                    ),
                ),
            }

        return predictions

    def get_oob(self, no_warning: bool = False, hier_shrinkage_lambda: float = None) -> Optional[float]:
        """
        Calculate the out-of-bag error of a given forest. This is done
        by using the out-of-bag predictions for each observation, and calculating the
        MSE over the entire forest.

        :param noWarning: A flag to not display warnings.
        :type noWarning: *bool, optional, default=False*
        :return: The OOB error of the forest.
        :rtype: float

        """

        preprocessing.forest_checker(self)

        if (not self.replace) and (self.ntree * (self.processed_dta.n_observations - self.sampsize)) < 10:
            if not no_warning:
                warnings.warn("Samples are drawn without replacement and sample size is too big!")
            return None

        preds = self.predict(
            newdata=None,
            aggregation="oob",
            exact=True,
            hier_shrinkage_lambda=hier_shrinkage_lambda,
        )
        preds = preds[~np.isnan(preds)]

        # Only calc mse on non missing predictions
        y_true = self.processed_dta["y"]
        y_true = y_true[~np.isnan(y_true)]

        if self.scale:
            y_true = y_true * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1]

        return np.mean((y_true - preds) ** 2)

    def get_vi(self, no_warning: bool = False) -> Optional[np.ndarray]:
        """
        Calculate the percentage increase in OOB error of the forest
        when each feature is shuffled.

        :param noWarning: A flag to not display warnings.
        :return: The variable importance of the forest.

        """

        preprocessing.forest_checker(self)

        if (not self.replace) and (self.ntree * (self.processed_dta.n_observations - self.sampsize)) < 10:
            if not no_warning:
                warnings.warn("Samples are drawn without replacement and sample size is too big!")
            return None

        cpp_vi = extension.get_vi(self.forest)

        result = np.empty(self.processed_dta.num_columns)
        for i in range(self.processed_dta.num_columns):
            result[i] = extension.vector_get(cpp_vi, i)

        return result

    def score(
        self, X: Union[pd.DataFrame, pd.Series, List], y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Gets the coefficient of determination (R :sup:`2`).

        :param X: Testing samples.
        :type X: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols]*
        :param y: True outcome values of X.
        :type y: *array_like of shape [nsamples,]*
        :param sample_weight: Sample weights. Uses equal weights by default.
        :type sample_weight: *array_like of shape [nsamples,], optional, default=None*
        :return: The value of R :sup:`2`.
        :rtype: float

        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(newdata=X, aggregation="average"), sample_weight=sample_weight)

    def translate_tree(self, tree_ids: Optional[Union[int, np.ndarray]] = None) -> None:
        """
        Given a trained forest, translates the selected trees by allowing access to its underlying structure. After
        translating tree *i*, its structure will be stored as a dictionary in :ref:`saved_forest <translate-label>`
        and can be accessed by ``[RandomForest object].saved_forest[i]``. Check out the
        :ref:`saved_forest <translate-label>` attribute for more details about its structure.

        :param tree_ids: The indices of the trees to be translated. By default, all the trees in the forest
         are translated.
        :type tree_ids: *int/array_like, optional*
        :rtype: None
        """

        if len(self.saved_forest) == 0:
            self.saved_forest = [{} for _ in range(self.ntree)]

        if tree_ids is None:
            idx = np.arange(self.ntree)
        else:
            if isinstance(tree_ids, (int, np.integer)):
                idx = np.array([tree_ids])
            else:
                idx = np.array(tree_ids)

        for cur_id in idx:
            if self.saved_forest[cur_id]:
                continue

            num_nodes = extension.get_tree_node_count(self.forest, cur_id)

            # Initialize arrays to pass to C
            split_info = np.empty(self.sampsize + 1, dtype=np.intc)
            averaging_info = np.empty(self.sampsize + 1, dtype=np.intc)

            tree_info = np.empty(num_nodes * 8 + 1, dtype=np.double)

            extension.fill_tree_info(self.forest, cur_id, tree_info, split_info, averaging_info)
            self.saved_forest[cur_id]["feature"] = np.empty(num_nodes, dtype=np.intc)
            self.saved_forest[cur_id]["threshold"] = np.empty(num_nodes, dtype=np.double)
            self.saved_forest[cur_id]["values"] = np.empty(num_nodes, dtype=np.double)
            self.saved_forest[cur_id]["na_left_count"] = np.empty(num_nodes, dtype=np.intc)
            self.saved_forest[cur_id]["na_right_count"] = np.empty(num_nodes, dtype=np.intc)
            self.saved_forest[cur_id]["na_default_direction"] = np.empty(num_nodes, dtype=np.intc)
            self.saved_forest[cur_id]["average_count"] = np.empty(num_nodes, dtype=np.intc)
            self.saved_forest[cur_id]["split_count"] = np.empty(num_nodes, dtype=np.intc)

            for i in range(num_nodes):
                self.saved_forest[cur_id]["feature"][i] = int(tree_info[i])
            for i in range(num_nodes):
                self.saved_forest[cur_id]["values"][i] = tree_info[num_nodes + i]
            for i in range(num_nodes):
                self.saved_forest[cur_id]["threshold"][i] = tree_info[num_nodes * 2 + i]
                self.saved_forest[cur_id]["na_left_count"][i] = int(tree_info[num_nodes * 3 + i])
                self.saved_forest[cur_id]["na_right_count"][i] = int(tree_info[num_nodes * 4 + i])
                self.saved_forest[cur_id]["na_default_direction"][i] = int(tree_info[num_nodes * 5 + i])
                self.saved_forest[cur_id]["average_count"][i] = int(tree_info[num_nodes * 6 + i])
                self.saved_forest[cur_id]["split_count"][i] = int(tree_info[num_nodes * 7 + i])

            num_split_idx = int(split_info[0])
            self.saved_forest[cur_id]["splitting_sample_idx"] = np.empty(num_split_idx, dtype=np.intc)
            for i in range(num_split_idx):
                self.saved_forest[cur_id]["splitting_sample_idx"][i] = int(split_info[i + 1])

            num_av_idx = int(averaging_info[0])
            self.saved_forest[cur_id]["averaging_sample_idx"] = np.empty(num_av_idx, dtype=np.intc)
            for i in range(num_av_idx):
                self.saved_forest[cur_id]["averaging_sample_idx"][i] = int(averaging_info[i + 1])

            self.saved_forest[cur_id]["seed"] = int(tree_info[num_nodes * 8])

    def get_parameters(self) -> dict:
        """
        Get the parameters of `RandomForest`.

        :return: A dictionary mapping parameter names of the `RandomForest` to their values.
        :rtype: dict
        """

        return {
            parameter: value
            for parameter, value in self.__dict__.items()
            if parameter not in ["forest", "dataframe", "processed_dta", "saved_forest", "__pydantic_initialised__"]
        }

    def set_parameters(self, **new_parameters: dict) -> Self:
        """
        Set the parameters of the *RandomForest*.

        :param **params: Forestry parameters.
        :type **params: *dict*
        :return: A new *RandomForest* object with the given parameters.
         Note: this reinitializes the *RandomForest* object,
         so fit must be called on the new estimator.
        :rtype: *RandomForest*
        """

        if not new_parameters:
            return self

        current_parameters = self.get_parameters()
        for parameter in new_parameters:
            if parameter not in current_parameters.keys():
                raise ValueError(
                    f"Invalid parameter {parameter} for RandomForest. Check the list of available parameters "
                    "with `estimator.get_parameters().keys()`."
                )

        self.__init__(**{**current_parameters, **new_parameters})  # pylint: disable=unnecessary-dunder-call
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["dataframe"]
        del state["forest"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        state["dataframe"] = extension.get_data(
            np.ascontiguousarray(
                pd.concat(
                    [
                        state["processed_dta"].processed_x,
                        pd.Series(state["processed_dta"].y),
                    ],
                    axis=1,
                ).values[:, :],
                np.double,
            ).ravel(),
            state["processed_dta"].categorical_feature_cols,
            state["processed_dta"].categorical_feature_cols.size,
            state["processed_dta"].linear_feature_cols,
            state["processed_dta"].linear_feature_cols.size,
            state["processed_dta"].feature_weights,
            state["processed_dta"].feature_weights_variables,
            state["processed_dta"].feature_weights_variables.size,
            state["processed_dta"].deep_feature_weights,
            state["processed_dta"].deep_feature_weights_variables,
            state["processed_dta"].deep_feature_weights_variables.size,
            state["processed_dta"].observation_weights,
            state["processed_dta"].monotonic_constraints,
            state["processed_dta"].group_memberships,
            state["monotone_avg"],
            state["processed_dta"].n_observations,
            state["processed_dta"].num_columns + 1,
            state["seed"],
        )

        tree_counts = np.empty(state["ntree"] * 4, dtype=np.intc)
        total_nodes, total_leaf_nodes, total_split_idx, total_av_idx = 0, 0, 0, 0
        for i in range(state["ntree"]):
            tree_counts[4 * i] = state["saved_forest"][i]["threshold"].size
            total_nodes += tree_counts[4 * i]

            tree_counts[4 * i + 1] = state["saved_forest"][i]["splitting_sample_idx"].size
            total_split_idx += tree_counts[4 * i + 1]

            tree_counts[4 * i + 2] = state["saved_forest"][i]["averaging_sample_idx"].size
            total_av_idx += tree_counts[4 * i + 2]

            tree_counts[4 * i + 3] = state["saved_forest"][i]["values"].size
            total_leaf_nodes += tree_counts[4 * i + 3]

        features = np.empty(total_nodes + total_leaf_nodes, dtype=np.intc)

        thresholds = np.empty(total_nodes, dtype=np.double)
        na_left_counts = np.empty(total_nodes, dtype=np.intc)
        na_right_counts = np.empty(total_nodes, dtype=np.intc)
        na_default_direction = np.empty(total_nodes, dtype=np.intc)

        sample_split_idx = np.empty(total_split_idx, dtype=np.intc)
        sample_av_idx = np.empty(total_av_idx, dtype=np.intc)

        predict_weights = np.empty(total_leaf_nodes, dtype=np.double)
        tree_seeds = np.empty(state["ntree"], dtype=np.uintc)

        ind, ind_s, ind_a, ind_val, ft_val = 0, 0, 0, 0, 0
        for i in range(state["ntree"]):
            for j in range(tree_counts[4 * i]):
                thresholds[ind] = state["saved_forest"][i]["threshold"][j]
                na_left_counts[ind] = state["saved_forest"][i]["na_left_count"][j]
                na_right_counts[ind] = state["saved_forest"][i]["na_right_count"][j]
                na_default_direction[ind] = state["saved_forest"][i]["na_default_direction"][j]
                ind += 1

            for j in range(tree_counts[4 * i + 1]):
                sample_split_idx[ind_s] = state["saved_forest"][i]["splitting_sample_idx"][j]
                ind_s += 1

            for j in range(tree_counts[4 * i + 2]):
                sample_av_idx[ind_a] = state["saved_forest"][i]["averaging_sample_idx"][j]
                ind_a += 1

            for j in range(tree_counts[4 * i + 3]):
                predict_weights[ind_val] = state["saved_forest"][i]["values"][j]
                ind_val += 1

            for j in range((tree_counts[4 * i] + tree_counts[4 * i + 3])):
                features[ft_val] = state["saved_forest"][i]["feature"][j]
                ft_val += 1

            tree_seeds[i] = state["saved_forest"][i]["seed"]

        state["forest"] = extension.reconstruct_tree(
            state["dataframe"],
            state["ntree"],
            state["replace"],
            state["sampsize"],
            state["splitratio"],
            state["oob_honest"],
            state["double_bootstrap"],
            state["mtry"],
            state["nodesize_spl"],
            state["nodesize_avg"],
            state["nodesize_strict_spl"],
            state["nodesize_strict_avg"],
            state["min_split_gain"],
            state["max_depth"],
            state["interaction_depth"],
            state["seed"],
            state["nthread"],
            state["verbose"],
            state["middle_split"],
            state["max_obs"],
            state["min_trees_per_fold"],
            state["fold_size"],
            state["processed_dta"].has_nas,
            state["na_direction"],
            state["linear"],
            state["overfit_penalty"],
            state["double_tree"],
            tree_counts,
            thresholds,
            features,
            na_left_counts,
            na_right_counts,
            na_default_direction,
            sample_split_idx,
            sample_av_idx,
            predict_weights,
            tree_seeds,
        )

    # Saving and loading
    def save_forestry(self, filename: Path) -> None:
        """
        Given a trained forest, saves the forest using pickle in the file given by *filename*. This can be used
        to save a model for future analysis or share a model after training.

        :param filename: The name of the file to save the forest model to
        :type Path: *char/array_like, optional*
        :rtype: None
        """
        self.translate_tree()

        with open(filename, "wb") as output_file:  # Overwrites any existing file.
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_forestry(filename: Path) -> Self:
        """
        Loads a forest that has been saved using *save_forestry*. Since the forest contains a pointer to the
        C++ object, it is necessary to rebuild this object and relink the pointer before the forest can
        be used to make predictions etc.

        :param filename: The name of the file to save the
        :type Path: *char/array_like, optional*
        :rtype: None
        """
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)  # nosec B301

    def export_json(self):
        """
        Export forest to Treelite JSON string
        """
        return extension.export_json(self.forest, self.processed_dta.col_sd, self.processed_dta.col_means)

    def __del__(self):
        # Free the pointers to foretsry and dataframe
        extension.delete_forestry(self.forest, self.dataframe)
