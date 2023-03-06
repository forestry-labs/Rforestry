#' @include forestry.R
#' @importFrom stats runif sd predict
NULL

# ---Computing lp distances-----------------------------------------------------
#' @name compute_lp-forestry
#' @title compute lp distances
#' @rdname compute_lp-forestry
#' @description Return the L_p norm distances of selected test observations
#'   relative to the training observations which the forest was trained on.
#' @param object A `forestry` object.
#' @param newdata A data frame of test predictors.
#' @param feature A string denoting the dimension for computing lp distances.
#' @param p A positive real number determining the norm p-norm used.
#' @param scale A boolean indicating whether or not we want to center + scale
#'   the features (based on the mean and sd of the training data) before calculating
#'   the L_p norm. This is useful for computing the detachment index, but can be
#'   less useful when we need to interpret the L_p distances.
#' @param aggregation The aggregation used when the weightMatrix is calculated.
#'   This can be useful for calculating the lp distances on observations in
#'   the training data. This must be one of `average`, `oob`, or `doubleOOB`.
#'   When newdata has fewer rows than the training data, one must also pass
#'   the vector of training indices corresponding to the indices of the observations
#'   in the original data set. Default is `average`.
#' @param trainingIdx This is an optional parameter that must be set when
#'   aggregation is set to `oob` or `doubleOOB` and the newdata is not the same
#'   size as the training data.
#' @return A vector of the lp distances.
#' @examples
#'
#' # Set seed for reproductivity
#' set.seed(292313)
#'
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 11)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train,nthread = 2)
#' predict(rf, x_test)
#'
#' # Compute the l2 distances in the "Petal.Length" dimension
#' distances_2 <- compute_lp(object = rf,
#'                           newdata = x_test,
#'                           feature = "Petal.Length",
#'                           p = 2)
#'@export
compute_lp <- function(object,
                       newdata,
                       feature,
                       p,
                       scale=FALSE,
                       aggregation="average",
                       trainingIdx=NULL){

  # Checks and parsing:
  if (!inherits(object, "forestry")) {
    stop("The object submitted is not a forestry random forest")
  }
  newdata <- as.data.frame(newdata)
  train_set <- slot(object, "processed_dta")$processed_x

  if (!(feature %in% colnames(train_set))) {
    stop("The submitted feature is not in the set of possible features")
  }

  if (!(aggregation %in% c("average", "oob","doubleOOB"))) {
    stop("Aggregation must be average, oob, or doubleOOB")
  }

  # Compute distances

  # Calculate the weight matrix with the correct aggregation
  args.predict <- list("aggregation" = aggregation,
                       "object" = object,
                       "newdata" = newdata,
                       "weightMatrix" = TRUE)
  if (!is.null(trainingIdx)) {
    args.predict$trainingIdx <- trainingIdx
  }
  y_weights <- do.call(predict, args.predict)$weightMatrix

  if (is.factor(newdata[1, feature])) {
    # Get categorical feature mapping
    mapping <- slot(object, "categoricalFeatureMapping")

    # Change factor values to corresponding integer levels
    #TODO(Rowan): This is specific for the iris data sets.
    # * Run it on data set with several factor valued covariates
    # * Implement a test with a data set with two factor-valued column
    # * Try to use:
    # processed_x <- preprocess_testing(newdata,
    #                                   object@categoricalFeatureCols,
    #                                   object@categoricalFeatureMapping)
    feat_ind <- which(sapply(mapping, "[[", 1) == 
                        which(names(train_set) == feature))
    factor_vals <- mapping[[feat_ind]][2][[1]]
    map <- function(x) {
      return(which(factor_vals == x)[1])
    }
    newdata[ ,feature] <- unlist(lapply(newdata[,feature], map))

    diff_mat <- matrix(newdata[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = FALSE) !=
                matrix(train_set[,feature],
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = TRUE)
    diff_mat[diff_mat] <- 1
  } else {

    # If we scale the features, use the mean and SD from the training data
    if (scale) {
      feat_mean <- mean(train_set[,feature], na.rm = TRUE)
      feat_sd <- sd(train_set[,feature], na.rm = TRUE)
      feature_in_newdata <- scale(newdata[,feature], center = feat_mean, scale = feat_sd)[,]
      feature_in_traindata <- scale(train_set[,feature], center = feat_mean, scale = feat_sd)[,]
    } else {
      feature_in_newdata <- newdata[,feature]
      feature_in_traindata <- train_set[,feature]
    }


    diff_mat <- matrix(feature_in_newdata,
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = FALSE) -
                matrix(feature_in_traindata,
                       nrow = nrow(newdata),
                       ncol = nrow(train_set),
                       byrow = TRUE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat) ^ p

  # Compute final Lp distances
  distances <- apply(y_weights * diff_mat, 1, sum) ^ (1 / p)

  # Ensure that the Lp distances for a factor are between 0 and 1
  if (is.factor(newdata[1, feature])) {
    distances[distances < 0] <- 0
    distances[distances > 1] <- 1
  }

  return(distances)
}



