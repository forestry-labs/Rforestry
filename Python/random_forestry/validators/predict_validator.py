import warnings
from typing import Final, List, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted

from .base_validator import BaseValidator


class PredictValidator(BaseValidator):
    DEFAULT_X: Final = None
    DEFAULT_AGGREGATION: Final[str] = "average"

    @staticmethod
    def get_X(*args, **kwargs) -> Union[pd.DataFrame, pd.Series, List, None]:
        if len(args) > 1:
            raise TypeError(f"predict() takes 0 or 1 positional argument but {len(args)} were given")

        if "X" in kwargs:
            if len(args) == 1:
                raise AttributeError("X specified both in args and kwargs")
            else:
                return kwargs["X"]
        else:
            if len(args) == 1:
                return args[0]
            else:
                return PredictValidator.DEFAULT_X

    @staticmethod
    def validate_X(forest, *args, **kwargs) -> pd.DataFrame:
        X = PredictValidator.get_X(*args, **kwargs)

        if X is not None:
            check_array(X, accept_sparse=False, force_all_finite="allow-nan")
            if not (isinstance(X, (pd.DataFrame, pd.Series, list)) or type(X).__module__ == np.__name__):
                raise AttributeError("X must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list")

            X = (pd.DataFrame(X)).copy()
            X.reset_index(drop=True, inplace=True)

            if len(X.columns) != forest.processed_dta_.num_columns:
                raise ValueError(
                    f"X has {len(X.columns)}, "
                    f"but the forest was trained with {forest.processed_dta_.num_columns} columns."
                )

            if forest.processed_dta_.feat_names is not None:
                if not set(X.columns) == set(forest.processed_dta_.feat_names):
                    raise ValueError("X has different columns then the ones the forest was trained with.")

                # If linear is true we can't predict observations with some features missing.
                if forest.linear and X.isnull().values.any():
                    raise ValueError("linear does not support missing data")

                if not (X.columns == forest.processed_dta_.feat_names).all():
                    warnings.warn("X columns have been reordered so that they match the training feature matrix")
                    X = X[forest.processed_dta_.feat_names]

        return X

    @staticmethod
    def validate_exact(**kwargs) -> bool:
        if "exact" in kwargs:
            return kwargs["exact"]
        if kwargs.get("X") is None:
            return True
        if len(kwargs["X"].index) > 1e5:
            return False
        return True

    @staticmethod
    def validate_trees(forest, *args, **kwargs) -> None:
        if "trees" in kwargs:
            if not kwargs["exact"] or kwargs["aggregation"] != "average":
                raise ValueError("When using tree indices, we must have exact = True and aggregation = 'average'")

            if any(
                (not isinstance(i, (int, np.integer))) or (i < -forest.ntree) or (i >= forest.ntree)
                for i in kwargs["trees"]
            ):
                raise ValueError("trees must contain indices which are integers between -ntree and ntree-1")

    @staticmethod
    def validate_aggregation(forest, *args, **kwargs) -> str:
        aggregation = kwargs.get("aggregation", PredictValidator.DEFAULT_AGGREGATION)

        if aggregation == "oob":
            pass
        elif aggregation == "doubleOOB":
            if not forest.double_bootstrap_:
                raise ValueError(
                    "Attempting to do double OOB predictions "
                    "with a forest that was not trained with doubleBootstrap = True"
                )
        elif aggregation == "coefs":
            if not forest.linear:
                raise ValueError("Aggregation can only be linear with setting the parameter linear = True")
            if kwargs.get("X") is None:
                raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply X")
        else:
            if kwargs.get("X") is None:
                raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply X")

        return aggregation

    def __call__(self, forest, *args, **kwargs):
        check_is_fitted(forest)

        kwargs["X"] = PredictValidator.validate_X(forest, *args, **kwargs)
        kwargs["exact"] = PredictValidator.validate_exact(**kwargs)
        kwargs["aggregation"] = PredictValidator.validate_aggregation(forest, *args, **kwargs)
        kwargs["nthread"] = kwargs.get("nthread", forest.nthread)

        PredictValidator.validate_trees(forest, *args, **kwargs)

        return self.function(forest, **kwargs)
