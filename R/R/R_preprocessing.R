# --header ---------------------------------------------------------------------
# globally import the ability to create classes methods etc
#' @import methods
NULL
# This is just here, because methods will be used in many places for creating
# classes, methods etc etc.


# -- Methods for Preprocessing Data --------------------------------------------
#' @title preprocess_training
#' @description Perform preprocessing for the training data, including
#'   converting data to dataframe, and encoding categorical data into numerical
#'   representation.
#' @inheritParams forestry
#' @return A list of two datasets along with necessary information that encodes
#'   the preprocessing.
preprocess_training <- function(x, y) {
  x <- as.data.frame(x)
  # Check if the input dimension of x matches y
  if (nrow(x) != length(y)) {
    stop("The dimension of input dataset x doesn't match the output vector y.")
  }

  # Track the order of all features
  featureNames <- colnames(x)
  if (is.null(featureNames)) {
    warning("No names are given for each column.")
  }

  # Track all categorical features (both factors and characters)
  featureFactorCols <- which(sapply(x, is.factor) == TRUE)
  featureCharacterCols <- which(sapply(x, is.character) == TRUE)
  if (length(featureCharacterCols) != 0) {
    stop("Character value features must be cast to factors.")
  }
  categoricalFeatureCols <-
    c(featureFactorCols, featureCharacterCols)
  if (length(categoricalFeatureCols) == 0) {
    categoricalFeatureCols <- list()
  } else {
    categoricalFeatureCols <- list(categoricalFeatureCols)
  }

  # For each categorical feature, encode x into numeric representation and
  # save the encoding mapping
  categoricalFeatureMapping <- list()
  dummyIndex <- 1
  for (categoricalFeatureCol in unlist(categoricalFeatureCols)) {
    # drop unused levels
    x[, categoricalFeatureCol] <- factor(x[, categoricalFeatureCol])
    categoricalFeatureMapping[[dummyIndex]] <- list(
      "categoricalFeatureCol" = categoricalFeatureCol,
      "uniqueFeatureValues" = levels(x[, categoricalFeatureCol]),
      "numericFeatureValues" = 1:length(levels(x[, categoricalFeatureCol]))
    )
    x[, categoricalFeatureCol] <- as.integer(x[, categoricalFeatureCol])
    dummyIndex <- dummyIndex + 1
  }

  # Return transformed data and encoding information
  return(
    list(
      "x" = x,
      "categoricalFeatureCols" = categoricalFeatureCols,
      "categoricalFeatureMapping" = categoricalFeatureMapping
    )
  )
}

#' @title preprocess_testing
#' @description Perform preprocessing for the testing data, including converting
#'   data to dataframe, and testing if the columns are consistent with the
#'   training data and encoding categorical data into numerical representation
#'   in the same way as training data.
#' @inheritParams forestry
#' @param categoricalFeatureCols A list of index for all categorical data. Used
#'   for trees to detect categorical columns.
#' @param categoricalFeatureMapping A list of encoding details for each
#'   categorical column, including all unique factor values and their
#'   corresponding numeric representation.
#' @return A preprocessed training dataaset x
preprocess_testing <- function(x,
                               categoricalFeatureCols,
                               categoricalFeatureMapping) {
  x <- as.data.frame(x)

  # Track the order of all features
  testingFeatureNames <- colnames(x)
  if (is.null(testingFeatureNames)) {
    warning("No names are given for each column.")
  }

  # Track all categorical features (both factors and characters)
  featureFactorCols <- which(sapply(x, is.factor) == TRUE)
  featureCharacterCols <- which(sapply(x, is.character) == TRUE)
  testingCategoricalFeatureCols <-
    c(featureFactorCols, featureCharacterCols)
  if (length(testingCategoricalFeatureCols) == 0) {
    testingCategoricalFeatureCols <- list()
  } else {
    testingCategoricalFeatureCols <- list(testingCategoricalFeatureCols)
  }

  if (length(setdiff(categoricalFeatureCols,
                     testingCategoricalFeatureCols)) != 0) {
    stop("Categorical columns are different between testing and training data.")
  }

  # For each categorical feature, encode x into numeric representation
  for (categoricalFeatureMapping_ in categoricalFeatureMapping) {
    categoricalFeatureCol <-
      categoricalFeatureMapping_$categoricalFeatureCol
    # Get all unique feature values
    testingUniqueFeatureValues <- unique(x[, categoricalFeatureCol])
    uniqueFeatureValues <-
      categoricalFeatureMapping_$uniqueFeatureValues
    numericFeatureValues <-
      categoricalFeatureMapping_$numericFeatureValues

    # If testing dataset contains more, adding new factors to the mapping list
    diffUniqueFeatureValues <- setdiff(testingUniqueFeatureValues,
                                       uniqueFeatureValues)
    if (length(diffUniqueFeatureValues) != 0) {
      uniqueFeatureValues <-
        c(uniqueFeatureValues, diffUniqueFeatureValues)
      numericFeatureValues <- 1:length(uniqueFeatureValues)
    }

    x[, categoricalFeatureCol] <- match(
      as.character(x[, categoricalFeatureCol]),
      uniqueFeatureValues
    )
  }

  # Return transformed data and encoding information
  return(x)
}


#' @title scale_center
#' @description Given a dataframe, scale and center the continous features
#' @param x A dataframe in order to be processed.
#' @param categoricalFeatureCols A vector of the categorical features, we
#'   don't want to scale/center these. Should be 1-indexed.
#' @param colMeans A vector of the means to center each column.
#' @param colSd A vector of the standard deviations to scale each column with.
#' @return A scaled and centered  dataset x
scale_center <- function(
  x,
  categoricalFeatureCols,
  colMeans,
  colSd
) {
  # Center and scale continous features and outcome
  for (col_idx in 1:ncol(x)) {
    if (col_idx %in% categoricalFeatureCols) {
      next
    } else {
      if (colSd[col_idx] != 0) {
        x[,col_idx] <- ( x[,col_idx] - colMeans[col_idx]) / colSd[col_idx]
      } else {
        x[,col_idx] <- ( x[,col_idx] - colMeans[col_idx])
      }
    }
  }
  return(x)
}

#' @title unscale_uncenter
#' @description Given a dataframe, un scale and un center the continous features
#' @param x A dataframe in order to be processed.
#' @param categoricalFeatureCols A vector of the categorical features, we
#'   don't want to scale/center these. Should be 1-indexed.
#' @param colMeans A vector of the means to add to each column.
#' @param colSd A vector of the standard deviations to rescale each column with.
#' @return A dataset x in it's original scaling
unscale_uncenter <- function(
  x,
  categoricalFeatureCols,
  colMeans,
  colSd
) {
  # Center and scale continous features and outcome
  for (col_idx in 1:ncol(x)) {
    if (col_idx %in% categoricalFeatureCols) {
      next
    } else {
      if (colSd[col_idx] != 0) {
        x[,col_idx] <- ( x[,col_idx]* colSd[col_idx] + colMeans[col_idx])
      } else {
        x[,col_idx] <- ( x[,col_idx] + colMeans[col_idx])
      }
    }
  }
  return(x)
}
