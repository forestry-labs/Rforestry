import os
import warnings
from math import ceil
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import check_X_y

from . import is_negative, is_positive
from .base_validator import BaseValidator


class FitValidator(BaseValidator):
    DEFAULT_MTRY: Final = None
    DEFAULT_SAMPSIZE: Final = None
    DEFAULT_MAX_OBS: Final = None
    DEFAULT_MAX_DEPTH: Final = None
    DEFAULT_INTERACTION_DEPTH: Final = None
    DEFAULT_DOUBLE_BOOTSTRAP: Final = None
    DEFAULT_DOUBLE_TREE: Final = False
    DEFAULT_SPLITRATIO: Final = 1.0
    DEFAULT_REPLACE: Final = True

    @staticmethod
    def validate_monotonic_constraints(forest, X, **kwargs):
        _, ncols = X.shape

        if "monotonic_constraints" not in kwargs:
            monotonic_constraints = np.zeros(ncols, dtype=np.intc)
        else:
            monotonic_constraints = np.array(kwargs["monotonic_constraints"], dtype=np.intc)

        if monotonic_constraints.size != ncols:
            raise ValueError("monotonic_constraints must have the size of x")
        if any(i not in (0, 1, -1) for i in monotonic_constraints):
            raise ValueError("monotonic_constraints must be either 1, 0, or -1")
        if any(i != 0 for i in monotonic_constraints) and forest.linear:
            raise ValueError("Cannot use linear splitting with monotonic_constraints")

        return monotonic_constraints

    @staticmethod
    def validate_observation_weights(forest, X, **kwargs):
        nrows, _ = X.shape

        if not FitValidator.validate_replace(forest, **kwargs):
            observation_weights = np.zeros(nrows, dtype=np.double)
        elif "observation_weights" not in kwargs:
            observation_weights = np.repeat(1.0, nrows)
        else:
            observation_weights = np.array(kwargs["observation_weights"], dtype=np.double)

        if observation_weights.size != nrows:
            raise ValueError("observation_weights must have length len(x)")
        if any(i < 0 for i in observation_weights):
            raise ValueError("The entries in observation_weights must be non negative")
        if FitValidator.validate_replace(forest, **kwargs) and np.sum(observation_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in observation_weights")

        return observation_weights

    @staticmethod
    def validate_lin_feats(X, **kwargs):
        _, ncols = X.shape

        if "lin_feats" not in kwargs:
            lin_feats = np.arange(ncols, dtype=np.ulonglong)
        else:
            lin_feats = pd.unique(np.array(kwargs["lin_feats"], dtype=np.ulonglong))

        if any(i < 0 or i >= ncols for i in lin_feats):
            raise ValueError("lin_feats must contain positive integers less than len(x.columns).")

        return lin_feats

    @staticmethod
    def validate_feature_weights(X, **kwargs):
        _, ncols = X.shape

        if "feature_weights" not in kwargs:
            feature_weights = np.repeat(1.0, ncols)
            interaction_variables = [] if "interaction_variables" not in kwargs else kwargs["interaction_variables"]
            feature_weights[interaction_variables] = 0.0
        else:
            feature_weights = np.array(kwargs["feature_weights"], dtype=np.double)

        if feature_weights.size != ncols:
            raise ValueError("feature_weights must have length len(x.columns)")

        if any(i < 0 for i in feature_weights):
            raise ValueError("The entries in feature_weights must be non negative")

        if np.sum(feature_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in feature_weights")

        return feature_weights

    @staticmethod
    def validate_deep_feature_weights(X, **kwargs):
        _, ncols = X.shape

        if "deep_feature_weights" not in kwargs:
            deep_feature_weights = np.repeat(1.0, ncols)
        else:
            deep_feature_weights = np.array(kwargs["deep_feature_weights"], dtype=np.double)

        if deep_feature_weights.size != ncols:
            raise ValueError("deep_feature_weights must have length len(x.columns)")

        if any(i < 0 for i in deep_feature_weights):
            raise ValueError("The entries in deep_feature_weights must be non negative")

        if np.sum(deep_feature_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in deep_feature_weights")

        return deep_feature_weights

    @staticmethod
    def validate_groups(**kwargs):
        if "groups" not in kwargs:
            return None
        groups = kwargs["groups"]
        if len(groups.unique()) == 1:
            raise ValueError("groups must have more than 1 level to be left out from sampling.")
        return groups

    @staticmethod
    def validate_mtry(X, **kwargs) -> int:
        _, ncols = X.shape

        mtry = kwargs.get("mtry", FitValidator.DEFAULT_MTRY)
        if mtry is None:
            mtry = max((ncols // 3), 1)

        if mtry > ncols:
            raise ValueError("mtry cannot exceed total amount of features in x.")

        if not is_positive(int, mtry):
            raise ValueError("mtry must be positive_integer")
        return mtry

    @staticmethod
    def validate_sampsize(forest, X, **kwargs) -> int:
        nrows, _ = X.shape

        sampsize = kwargs.get("sampsize", FitValidator.DEFAULT_SAMPSIZE)
        if sampsize is None:
            sampsize = nrows if FitValidator.validate_replace(forest, **kwargs) else ceil(0.632 * nrows)

        # only if sample.fraction is given, update sampsize
        if forest.sample_fraction is not None:
            sampsize = ceil(forest.sample_fraction * nrows)

        if not FitValidator.validate_replace(forest, **kwargs) and sampsize > nrows:
            raise ValueError("You cannot sample without replacement with size more than total number of observations.")

        if not is_positive(int, sampsize):
            raise ValueError("sampsize must be positive integer")
        return sampsize

    @staticmethod
    def validate_double_bootstrap(forest, **kwargs) -> bool:
        double_bootstrap = kwargs.get("double_bootstrap", FitValidator.DEFAULT_DOUBLE_BOOTSTRAP)
        if double_bootstrap is not None:
            return double_bootstrap
        return forest.oob_honest

    @staticmethod
    def validate_replace(forest, **kwargs) -> bool:
        replace = kwargs.get("replace", FitValidator.DEFAULT_REPLACE)

        if forest.oob_honest and not replace:
            warnings.warn("replace must be set to TRUE to use OOBhonesty, setting this to True now")
            return True

        return replace

    @staticmethod
    def validate_double_tree(forest, **kwargs) -> bool:
        double_tree = kwargs.get("double_tree", FitValidator.DEFAULT_DOUBLE_TREE)

        if double_tree and FitValidator.validate_splitratio(forest, **kwargs) in (0, 1):
            warnings.warn("Trees cannot be doubled if splitratio is 0 or 1. We have set double_tree to False.")
            return False

        return double_tree

    @staticmethod
    def validate_splitratio(forest, **kwargs) -> float:
        splitratio = kwargs.get("splitratio", FitValidator.DEFAULT_SPLITRATIO)

        if forest.oob_honest and splitratio != 1:
            warnings.warn("oob_honest is set to true, so we will run OOBhonesty rather than standard honesty.")
            return 1

        if splitratio < 0 or splitratio > 1:
            raise ValueError("SplitRatio needs to be in range (0,1)")

        return splitratio

    @staticmethod
    def validate_max_obs(y, **kwargs):
        max_obs = kwargs.get("max_obs", FitValidator.DEFAULT_MAX_OBS)
        if max_obs is None:
            max_obs = y.size

        if not is_positive(int, max_obs):
            raise ValueError("max_obs must be positive_integer")
        return max_obs

    @staticmethod
    def validate_max_depth(X, **kwargs):
        nrows, _ = X.shape

        max_depth = kwargs.get("max_depth", FitValidator.DEFAULT_MAX_DEPTH)
        if max_depth is None:
            max_depth = round(nrows / 2) + 1

        if not is_positive(int, max_depth):
            raise ValueError("max_depth must be positive_integer")
        return max_depth

    @staticmethod
    def validate_interaction_depth(**kwargs):
        interaction_depth = kwargs.get("interaction_depth", FitValidator.DEFAULT_INTERACTION_DEPTH)
        if interaction_depth is None:
            interaction_depth = kwargs["max_depth"]

        if interaction_depth > kwargs["max_depth"]:
            warnings.warn(
                "interaction_depth cannot be greater than max_depth. We have set interaction_depth to max_depth."
            )
            interaction_depth = kwargs["max_depth"]

        if not is_positive(int, interaction_depth):
            raise ValueError("interaction_depth must be positive_integer")
        return interaction_depth

    @staticmethod
    def prevalidate(forest) -> None:
        for parameter in [
            "ntree",
            "nodesize_spl",
            "nodesize_avg",
            "nodesize_strict_spl",
            "nodesize_strict_avg",
            "fold_size",
        ]:
            if not is_positive(int, getattr(forest, parameter)):
                raise ValueError(f"{parameter} must be positive integer")

        for parameter in ["seed", "nthread", "min_trees_per_fold"]:
            if is_negative(int, getattr(forest, parameter)):
                raise ValueError(f"{parameter} must be non negative integer")

        if forest.sample_fraction:
            if not is_positive([int, float], forest.sample_fraction):
                raise ValueError("sample_fraction must be positive or None")

        if is_negative(float, forest.min_split_gain):
            raise ValueError("min_split_gain must be non negative float")

        if forest.nthread > os.cpu_count():
            raise ValueError("nthread cannot exceed total cores in the computer: " + str(os.cpu_count()))

        if forest.min_split_gain > 0 and not forest.linear:
            raise ValueError("min_split_gain cannot be set without setting linear to be true.")

    def __call__(self, forest, X, y, *args, **kwargs):
        _, y = check_X_y(X, y, accept_sparse=False, force_all_finite="allow-nan")
        X = pd.DataFrame(X).copy()

        FitValidator.prevalidate(forest)

        if len(args) > 0:
            raise ValueError("There can be only 2 non-keyword arguments: X, y")

        # Check if the input dimension of x matches y
        nrows, _ = X.shape
        if nrows != y.size:
            raise ValueError("The dimension of input dataset x doesn't match the output y.")

        try:
            if np.isnan(y).any():
                raise ValueError("y contains missing data")
        except TypeError:
            raise ValueError("Unknown label type")

        if len(X.columns[X.isnull().all()]) > 0:
            raise ValueError("Training data column cannot be all missing values.")

        if forest.linear and X.isnull().values.any():
            raise ValueError("Cannot do imputation splitting with linear.")

        kwargs["mtry"] = FitValidator.validate_mtry(X, **kwargs)
        kwargs["splitratio"] = FitValidator.validate_splitratio(forest, **kwargs)
        kwargs["replace"] = FitValidator.validate_replace(forest, **kwargs)
        kwargs["double_tree"] = FitValidator.validate_double_tree(forest, **kwargs)
        kwargs["sampsize"] = FitValidator.validate_sampsize(forest, X, **kwargs)
        kwargs["double_bootstrap"] = FitValidator.validate_double_bootstrap(forest, **kwargs)
        kwargs["max_obs"] = FitValidator.validate_max_obs(y, **kwargs)
        kwargs["max_depth"] = FitValidator.validate_max_depth(X, **kwargs)
        kwargs["interaction_depth"] = FitValidator.validate_interaction_depth(**kwargs)
        kwargs["monotonic_constraints"] = FitValidator.validate_monotonic_constraints(forest, X, **kwargs)
        kwargs["lin_feats"] = FitValidator.validate_lin_feats(X, **kwargs)
        kwargs["feature_weights"] = FitValidator.validate_feature_weights(X, **kwargs)
        kwargs["deep_feature_weights"] = FitValidator.validate_deep_feature_weights(X, **kwargs)
        kwargs["observation_weights"] = FitValidator.validate_observation_weights(forest, X, **kwargs)
        kwargs["groups"] = FitValidator.validate_groups(**kwargs)

        return self.function(forest, X, y, *args, **kwargs)
