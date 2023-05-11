import warnings
from typing import Final, List, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted

from .base_validator import BaseValidator


class PredictValidator(BaseValidator):
    DEFAULT_NEWDATA: Final = None
    DEFAULT_AGGREGATION: Final[str] = "average"

    def get_newdata(self, *args, **kwargs) -> Union[pd.DataFrame, pd.Series, List, None]:
        if len(args) > 1:
            raise TypeError(f"predict() takes from 1 to 2 positional arguments but {len(args)+1} were given")

        if "newdata" in kwargs:
            if len(args) == 1:
                raise AttributeError("newdata specified both in args and kwargs")
            else:
                return kwargs["newdata"]
        else:
            if len(args) == 1:
                return args[0]
            else:
                return __class__.DEFAULT_NEWDATA

    def validate_newdata(self, forest, *args, **kwargs) -> pd.DataFrame:
        newdata = self.get_newdata(*args, **kwargs)

        if newdata is not None:
            check_array(newdata)  # we can't run check_array() on None
            if not (isinstance(newdata, (pd.DataFrame, pd.Series, list)) or type(newdata).__module__ == np.__name__):
                raise AttributeError(
                    "newdata must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list"
                )

            newdata = (pd.DataFrame(newdata)).copy()
            newdata.reset_index(drop=True, inplace=True)

            if len(newdata.columns) != forest.processed_dta_.num_columns:
                raise ValueError(
                    f"newdata has {len(newdata.columns)}, "
                    f"but the forest was trained with {forest.processed_dta_.num_columns} columns."
                )

            if forest.processed_dta_.feat_names is not None:
                if not set(newdata.columns) == set(forest.processed_dta_.feat_names):
                    raise ValueError("newdata has different columns then the ones the forest was trained with.")

                # If linear is true we can't predict observations with some features missing.
                if forest.linear and newdata.isnull().values.any():
                    raise ValueError("linear does not support missing data")

                if not all(newdata.columns == forest.processed_dta_.feat_names):
                    warnings.warn("newdata columns have been reordered so that they match the training feature matrix")
                    newdata = newdata[forest.processed_dta_.feat_names]

        return newdata

    def validate_exact(self, **kwargs) -> bool:
        if "exact" in kwargs:
            return kwargs["exact"]
        if kwargs["newdata"] is None:
            return True
        if len(kwargs["newdata"].index) > 1e5:
            return False
        return True

    def validate_trees(self, forest, *args, **kwargs) -> None:
        if "trees" in kwargs:
            if not kwargs["exact"] or kwargs["aggregation"] != "average":
                raise ValueError("When using tree indices, we must have exact = True and aggregation = 'average' ")

            if any(
                (not isinstance(i, (int, np.integer))) or (i < -forest.ntree) or (i >= forest.ntree)
                for i in kwargs["trees"]
            ):
                raise ValueError("trees must contain indices which are integers between -ntree and ntree-1")

    def validate_aggregation(self, forest, *args, **kwargs) -> str:
        aggregation = kwargs.get("aggregation", __class__.DEFAULT_AGGREGATION)

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
                raise ValueError("Aggregation can only be linear with setting the parameter linear = True.")
            if kwargs["newdata"] is None:
                raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply newdata")
        else:
            if kwargs["newdata"] is None:
                raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply newdata")

        return aggregation

    def __call__(self, forest, *args, **kwargs):
        check_is_fitted(forest)

        kwargs["newdata"] = self.validate_newdata(forest, *args, **kwargs)
        kwargs["exact"] = self.validate_exact(**kwargs)
        kwargs["aggregation"] = self.validate_aggregation(forest, *args, **kwargs)
        kwargs["nthread"] = kwargs.get("nthread", forest.nthread)

        self.validate_trees(forest, *args, **kwargs)

        return self.function(forest, **kwargs)
