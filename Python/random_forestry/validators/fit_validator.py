import numpy as np
import pandas as pd
from sklearn.base import check_X_y

from .. import preprocessing
from .base_validator import BaseValidator


class FitValidator(BaseValidator):
    def validate_monotonic_constraints(self, _self, x, **kwargs):
        x = pd.DataFrame(x).copy()
        _, ncols = x.shape

        if "monotonic_constraints" not in kwargs:
            monotonic_constraints = np.zeros(ncols, dtype=np.intc)
        else:
            monotonic_constraints = np.array(kwargs["monotonic_constraints"], dtype=np.intc)

        if monotonic_constraints.size != ncols:
            raise ValueError("monotonic_constraints must have the size of x")
        if any(i not in (0, 1, -1) for i in monotonic_constraints):
            raise ValueError("monotonic_constraints must be either 1, 0, or -1")
        if any(i != 0 for i in monotonic_constraints) and _self.linear:
            raise ValueError("Cannot use linear splitting with monotonic_constraints")

        return monotonic_constraints

    def validate_observation_weights(self, _self, x, **kwargs):
        x = pd.DataFrame(x).copy()
        nrows, _ = x.shape

        if not _self.replace:
            observation_weights = np.zeros(nrows, dtype=np.double)
        elif "observation_weights" not in kwargs:
            observation_weights = np.repeat(1.0, nrows)
        else:
            observation_weights = np.array(kwargs["observation_weights"], dtype=np.double)

        if observation_weights.size != nrows:
            raise ValueError("observation_weights must have length len(x)")
        if any(i < 0 for i in observation_weights):
            raise ValueError("The entries in observation_weights must be non negative")
        if _self.replace and np.sum(observation_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in observation_weights")

        return observation_weights

    def validate_lin_feats(self, x, **kwargs):
        x = pd.DataFrame(x).copy()
        _, ncols = x.shape

        if "lin_feats" not in kwargs:
            lin_feats = np.arange(ncols, dtype=np.ulonglong)
        else:
            lin_feats = pd.unique(np.array(kwargs["lin_feats"], dtype=np.ulonglong))

        if any(i < 0 or i >= ncols for i in lin_feats):
            raise ValueError("lin_feats must contain positive integers less than len(x.columns).")

        return lin_feats

    def validate_feature_weights(self, x, **kwargs):
        x = pd.DataFrame(x).copy()
        _, ncols = x.shape

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

    def validate_deep_feature_weights(self, x, **kwargs):
        x = pd.DataFrame(x).copy()
        _, ncols = x.shape

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

    def validate_groups(self, **kwargs):
        if "groups" in kwargs:
            groups = kwargs["groups"]
            if len(groups.unique()) == 1:
                raise ValueError("groups must have more than 1 level to be left out from sampling.")
            return groups
        return None

    def __call__(self, _self, x, y, *args, **kwargs):
        _, y = check_X_y(x, y, accept_sparse=True)
        x = pd.DataFrame(x).copy()
        # y = (np.array(y, dtype=np.double)).copy()

        nrows, ncols = x.shape

        # Check if the input dimension of x matches y
        if nrows != y.size:
            raise ValueError("The dimension of input dataset x doesn't match the output y.")

        if np.isnan(y).any():
            raise ValueError("y contains missing data.")

        if len(x.columns[x.isnull().all()]) > 0:
            raise ValueError("Training data column cannot be all missing values.")

        if _self.linear and x.isnull().values.any():
            raise ValueError("Cannot do imputation splitting with linear.")

        if not _self.replace and preprocessing.get_sampsize(_self, x) > nrows:
            raise ValueError("You cannot sample without replacement with size more than total number of observations.")
        if preprocessing.get_mtry(_self, x) > ncols:
            raise ValueError("mtry cannot exceed total amount of features in x.")

        kwargs["monotonic_constraints"] = self.validate_monotonic_constraints(_self, x, **kwargs)

        kwargs["lin_feats"] = self.validate_lin_feats(x, **kwargs)

        kwargs["feature_weights"] = self.validate_feature_weights(x, **kwargs)

        kwargs["deep_feature_weights"] = self.validate_deep_feature_weights(x, **kwargs)

        kwargs["observation_weights"] = self.validate_observation_weights(_self, x, **kwargs)

        if "groups" in kwargs:
            kwargs["groups"] = self.validate_groups(**kwargs)

        return self.function(_self, x, y, **kwargs)
