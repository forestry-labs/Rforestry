import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut


def get_feat_names(x: Union[pd.DataFrame, pd.Series, List]) -> Optional[np.ndarray]:
    if isinstance(x, pd.DataFrame):
        return x.columns.values
    if type(x).__module__ == np.__name__ or isinstance(x, (list, pd.Series)):
        print(
            "x does not have column names. ",
            "The check that columns are provided in the same order when training and predicting will be skipped",
            file=sys.stderr,
        )
        return None

    raise AttributeError("x must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list")


def find_match(arr_a: Union[np.ndarray, List], arr_b: Union[np.ndarray, List]) -> np.ndarray:
    """
    --------------------------------------

    Helper Function
    @return a nunpy array indicating the indices of first occurunces of
      the elements of arr_a in arr_b
    """

    temp_dict = {}

    for index, element in enumerate(arr_b):
        if isinstance(element, int):
            element = float(element)
        if str(element) not in temp_dict:
            temp_dict[str(arr_b[index])] = index

    return np.array([temp_dict[str(float(val)) if isinstance(val, int) else str(val)] for val in arr_a])


# Given a dataframe with Y and Y.hat at least, fits an OLS and gives the LOO
# predictions on the sample
def loo_pred_helper(data_frame: pd.DataFrame) -> dict:
    Y = data_frame["Y"]
    X = data_frame.loc[:, data_frame.columns != "Y"]
    X = sm.add_constant(X)

    adjust_lm = sm.OLS(Y, X).fit()

    cv = LeaveOneOut()
    cv_pred = np.empty(Y.size)

    for i, (train, test) in enumerate(cv.split(X)):
        # split data
        X_train, X_test = X.iloc[train, :], X.iloc[test, :]
        y_train, _ = Y[train], Y[test]

        # fit model
        model = sm.OLS(y_train, X_train).fit()
        cv_pred[i] = model.predict(X_test)

    return {"insample_preds": cv_pred, "adjustment_model": adjust_lm}


def preprocess_training(x: pd.DataFrame, y) -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    """
    -- Methods for Preprocessing Data --------------------------------------------
    @title preprocess_training
    @description Perform preprocessing for the training data, including
      converting data to dataframe, and encoding categorical data into numerical
      representation.
    @inheritParams RandomForest
    @return A list of two datasets along with necessary information that encodes
      the preprocessing.
    """

    # Check if the input dimension of x matches y
    if len(x.index) != y.size:
        raise ValueError("The dimension of input dataset x doesn't match the output vector y.")

    # Track the order of all features
    feature_names = x.columns.values
    if feature_names.size == 0:
        warnings.warn("No names are given for each column.")

    # Track all categorical features (both factors and characters)
    categorical_feature_cols = np.array((x.select_dtypes("category")).columns)
    feature_character_cols = np.array((x.select_dtypes("object")).columns)

    if feature_character_cols.size != 0:  # convert to a factor
        warnings.warn("Character value features will be cast to categorical data.")
        categorical_feature_cols = np.concatenate((categorical_feature_cols, feature_character_cols), axis=0)

    categorical_feature_cols = x.columns.get_indexer(categorical_feature_cols)

    # For each categorical feature, encode x into numeric representation and
    # save the encoding mapping
    categorical_feature_mapping: List[Dict] = []
    for categorical_feature_col in categorical_feature_cols:
        x.iloc[:, categorical_feature_col] = pd.Series(
            x.iloc[:, categorical_feature_col], dtype="category"
        ).cat.remove_unused_categories()

        categorical_feature_mapping.append(
            {
                "categoricalFeatureCol": categorical_feature_col,
                "uniqueFeatureValues": list(x.iloc[:, categorical_feature_col].cat.categories),
                "numericFeatureValues": np.arange(len(x.iloc[:, categorical_feature_col].cat.categories)),
            }
        )

        x.iloc[:, categorical_feature_col] = pd.Series(x.iloc[:, categorical_feature_col].cat.codes, dtype="category")

    return x, categorical_feature_cols, categorical_feature_mapping


def preprocess_testing(x, categorical_feature_cols: np.ndarray, categorical_feature_mapping: List[Dict]) -> Any:
    """
    @title preprocess_testing
    @description Perform preprocessing for the testing data, including converting
      data to dataframe, and testing if the columns are consistent with the
      training data and encoding categorical data into numerical representation
      in the same way as training data.
    @inheritParams RandomForest
    @param categorical_feature_cols A list of index for all categorical data. Used
      for trees to detect categorical columns.
    @param categorical_feature_mapping A list of encoding details for each
      categorical column, including all unique factor values and their
      corresponding numeric representation.
    @return A preprocessed training dataaset x
    """

    # Track the order of all features
    testing_feature_names = x.columns.values

    if testing_feature_names.size == 0:
        warnings.warn("No names are given for each column.")

    # Track all categorical features (both factors and characters)
    feature_factor_cols = np.array((x.select_dtypes("category")).columns)
    feature_character_cols = np.array((x.select_dtypes("object")).columns)

    testing_categorical_feature_cols = np.concatenate((feature_factor_cols, feature_character_cols), axis=0)
    testing_categorical_feature_cols = x.columns.get_indexer(testing_categorical_feature_cols)

    if (set(categorical_feature_cols) - set(testing_categorical_feature_cols)) or (
        set(testing_categorical_feature_cols) - set(categorical_feature_cols)
    ):
        raise ValueError("Categorical columns are different between testing and training data.")

    # For each categorical feature, encode x into numeric representation
    for categorical_feature_mapping_ in categorical_feature_mapping:
        categorical_feature_col = categorical_feature_mapping_["categoricalFeatureCol"]
        # Get all unique feature values
        testing_unique_feature_values = x.iloc[:, categorical_feature_col].unique()
        unique_feature_values = categorical_feature_mapping_["uniqueFeatureValues"]
        numeric_feature_values = categorical_feature_mapping_["numericFeatureValues"]

        # If testing dataset contains more, adding new factors to the mapping list
        diff_unique_feature_values = set(testing_unique_feature_values) - set(unique_feature_values)
        if diff_unique_feature_values:
            unique_feature_values = np.concatenate(
                (list(unique_feature_values), list(diff_unique_feature_values)), axis=0
            )
            numeric_feature_values = np.arange(unique_feature_values.size)

            # update
            categorical_feature_mapping_["uniqueFeatureValues"] = unique_feature_values
            categorical_feature_mapping_["numericFeatureValues"] = numeric_feature_values

        x.iloc[:, categorical_feature_col] = pd.Series(
            find_match(x.iloc[:, categorical_feature_col], unique_feature_values),
            dtype="category",
        )

    # Return transformed data and encoding information
    return x


def scale_center(
    x: pd.DataFrame, categorical_feature_cols: np.ndarray, col_means: np.ndarray, col_sd: np.ndarray
) -> pd.DataFrame:
    """
    @title scale_center
    @description Given a dataframe, scale and center the continous features
    @param x A dataframe in order to be processed.
    @param categoricalFeatureCols A vector of the categorical features, we
      don't want to scale/center these.
    @param colMeans A vector of the means to center each column.
    @param colSd A vector of the standard deviations to scale each column with.
    @return A scaled and centered  dataset x
    """

    for col_idx in range(len(x.columns)):
        if col_idx not in categorical_feature_cols:
            if col_sd[col_idx] != 0:
                x.iloc[:, col_idx] = (x.iloc[:, col_idx] - col_means[col_idx]) / col_sd[col_idx]
            else:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] - col_means[col_idx]

    return x


def unscale_uncenter(x: pd.DataFrame, categorical_feature_cols: list, col_means: list, col_sd: list) -> pd.DataFrame:
    """
    @title unscale_uncenter
    @description Given a dataframe, un scale and un center the continous features
    @param x A dataframe in order to be processed.
    @param categoricalFeatureCols A vector of the categorical features, we
      don't want to scale/center these. Should be 1-indexed.
    @param colMeans A vector of the means to add to each column.
    @param colSd A vector of the standard deviations to rescale each column with.
    @return A dataset x in it's original scaling
    """

    for index, column in x.columns:
        if column not in categorical_feature_cols:
            if col_sd[index] != 0:
                x.iloc[:, index] = x.iloc[:, index] * col_sd[index] + col_means[index]
            else:
                x.iloc[:, index] = x.iloc[:, index] + col_means[index]

    return x


def scale(x, y, processed_x, categorical_feature_cols):
    _, ncol = x.shape
    col_means = np.repeat(0.0, ncol + 1)
    col_sd = np.repeat(0.0, ncol + 1)

    for col_idx in range(ncol):
        if col_idx not in categorical_feature_cols:
            col_means[col_idx] = np.nanmean(processed_x.iloc[:, col_idx])
            col_sd[col_idx] = np.nanstd(processed_x.iloc[:, col_idx])

    # Scale columns of X
    processed_x = scale_center(processed_x, categorical_feature_cols, col_means, col_sd)

    # Center and scale Y
    col_means[ncol] = np.nanmean(y)
    col_sd[ncol] = np.nanstd(y)
    if col_sd[ncol] != 0:
        y = (y - col_means[ncol]) / col_sd[ncol]
    else:
        y = y - col_means[ncol]

    return processed_x, y, col_means, col_sd
