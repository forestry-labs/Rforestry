#' @useDynLib Rforestry, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' @include R_preprocessing.R
#-- Sanity Checker -------------------------------------------------------------
#' @name training_data_checker
#' @title Training data check
#' @rdname training_data_checker-forestry
#' @description Check the input to forestry constructor
#' @inheritParams forestry
#' @param featureWeights weights used when subsampling features for nodes above or at interactionDepth.
#' @param deepFeatureWeights weights used when subsampling features for nodes below interactionDepth.
#' @param hasNas indicates if there is any missingness in x.
#' @return A list of parameters after checking the selected parameters are valid.
training_data_checker <- function(x,
                                  y,
                                  ntree,
                                  replace,
                                  sampsize,
                                  mtry,
                                  nodesizeSpl,
                                  nodesizeAvg,
                                  nodesizeStrictSpl,
                                  nodesizeStrictAvg,
                                  minSplitGain,
                                  maxDepth,
                                  interactionDepth,
                                  splitratio,
                                  OOBhonest,
                                  nthread,
                                  middleSplit,
                                  doubleTree,
                                  linFeats,
                                  monotonicConstraints,
                                  groups,
                                  featureWeights,
                                  deepFeatureWeights,
                                  observationWeights,
                                  linear,
                                  hasNas
                                  ) {
  x <- as.data.frame(x)
  nfeatures <- ncol(x)

  # Check if the input dimension of x matches y
  if (nrow(x) != length(y)) {
    stop("The dimension of input dataset x doesn't match the output vector y.")
  }

  if (linear && hasNas) {
    stop("Cannot do imputation splitting with linear")
  }

  if (any(is.na(y))) {
    stop("y contains missing data.")
  }

  if (!is.logical(replace)) {
    stop("replace must be TRUE or FALSE.")
  }

  if (ntree <= 0 || ntree %% 1 != 0) {
    stop("ntree must be a positive integer.")
  }

  if (sampsize <= 0 || sampsize %% 1 != 0) {
    stop("sampsize must be a positive integer.")
  }

  if (max(linFeats) >= nfeatures || any(linFeats < 0)) {
    stop("linFeats must be a positive integer less than ncol(x).")
  }

  if (!replace && sampsize > nrow(x)) {
    stop(
      paste(
        "You cannot sample without replacement with size more than",
        "total number of observations."
      )
    )
  }
  if (mtry <= 0 || mtry %% 1 != 0) {
    stop("mtry must be a positive integer.")
  }
  if (mtry > nfeatures) {
    stop("mtry cannot exceed total amount of features in x.")
  }

  if (nodesizeSpl <= 0 || nodesizeSpl %% 1 != 0) {
    stop("nodesizeSpl must be a positive integer.")
  }
  if (nodesizeAvg < 0 || nodesizeAvg %% 1 != 0) {
    stop("nodesizeAvg must be a positive integer.")
  }

  if (nodesizeStrictSpl <= 0 || nodesizeStrictSpl %% 1 != 0) {
    stop("nodesizeStrictSpl must be a positive integer.")
  }
  if (nodesizeStrictAvg < 0 || nodesizeStrictAvg %% 1 != 0) {
    stop("nodesizeStrictAvg must be a positive integer.")
  }
  if (minSplitGain < 0) {
    stop("minSplitGain must be greater than or equal to 0.")
  }
  if (minSplitGain > 0 && !linear) {
    stop("minSplitGain cannot be set without setting linear to be true.")
  }
  if (maxDepth <= 0 || maxDepth %% 1 != 0) {
    stop("maxDepth must be a positive integer.")
  }
  if (interactionDepth <= 0 || interactionDepth %% 1 != 0) {
    stop("interactionDepth must be a positive integer.")
  }

  if (length(monotonicConstraints) != ncol(x)) {
    stop("monotoneConstraints must be the size of x")
  }

  if (any((monotonicConstraints != 1 ) & (monotonicConstraints != -1 ) & (monotonicConstraints != 0 ))) {
    stop("monotonicConstraints must be either 1, 0, or -1")
  }

  if (any(monotonicConstraints != 0) && linear) {
    stop("Cannot use linear splitting with monotoneConstraints")
  }

  if (replace == FALSE) {
    observationWeights = rep(1, nrow(x))
  }

  if(length(observationWeights) != nrow(x)) {
    stop("observationWeights must have length nrow(x)")
  }
  if(any(observationWeights < 0)) {
    stop("The entries in observationWeights must be non negative")
  }
  if(sum(observationWeights) == 0) {
    stop("There must be at least one non-zero weight in observationWeights")
  }

  observationWeights <- observationWeights/sum(observationWeights)

  # if the splitratio is 1, then we use adaptive rf and avgSampleSize is the
  # equal to the total sampsize
  if (splitratio == 0 || splitratio == 1) {
    splitSampleSize <- sampsize
    avgSampleSize <- sampsize
  } else {
    splitSampleSize <- splitratio * sampsize
    avgSampleSize <- floor(sampsize - splitSampleSize)
    splitSampleSize <- floor(splitSampleSize)
  }

  if (nodesizeStrictSpl > splitSampleSize) {
    warning(
      paste(
        "nodesizeStrictSpl cannot exceed splitting sample size.",
        "We have set nodesizeStrictSpl to be the maximum"
      )
    )
    nodesizeStrictSpl <- splitSampleSize
  }
  if (nodesizeStrictAvg > avgSampleSize) {
    warning(
      paste(
        "nodesizeStrictAvg cannot exceed averaging sample size.",
        "We have set nodesizeStrictAvg to be the maximum"
      )
    )
    nodesizeStrictAvg <- avgSampleSize
  }
  if (doubleTree) {
    if (splitratio == 0 || splitratio == 1) {
      warning("Trees cannot be doubled if splitratio is 1. We have set
              doubleTree to FALSE")
      doubleTree <- FALSE
    } else {
      if (nodesizeStrictAvg > splitSampleSize) {
        warning(
          paste(
            "nodesizeStrictAvg cannot exceed splitting sample size.",
            "We have set nodesizeStrictAvg to be the maximum"
          )
        )
        nodesizeStrictAvg <- splitSampleSize
      }
      if (nodesizeStrictSpl > avgSampleSize) {
        warning(
          paste(
            "nodesizeStrictSpl cannot exceed averaging sample size.",
            "We have set nodesizeStrictSpl to be the maximum"
          )
        )
        nodesizeStrictSpl <- avgSampleSize
      }
    }
  }

  if (splitratio < 0 || splitratio > 1) {
    stop("splitratio must in between 0 and 1.")
  }

  if (!is.null(groups)) {
    if (!is.factor(groups)) {
      stop("groups must be supplied as a vector of factors")
    }
  }

  if (OOBhonest && (splitratio != 1)) {
    warning("OOBhonest is set to true, so we will run OOBhonesty rather
            than standard honesty")
    splitratio <- 1
  }

  if (OOBhonest && replace == FALSE) {
    warning("replace must be set to TRUE to use OOBhonesty, setting this to
            TRUE now")
    replace <- TRUE
  }

  if (nthread < 0 || nthread %% 1 != 0) {
    stop("nthread must be a nonegative integer.")
  }

  if (nthread > 0) {
    #' @import parallel
    if (tryCatch(
      nthread > parallel::detectCores(),
      error = function(x) {
        FALSE
      }
    )) {
      stop(paste0(
        "nthread cannot exceed total cores in the computer: ",
        detectCores()
      ))
    }
  }

  if (!is.logical(middleSplit)) {
    stop("middleSplit must be TRUE or FALSE.")
  }
  return(list("x" = x,
              "y" = y,
              "ntree" = ntree,
              "replace" = replace,
              "sampsize" = sampsize,
              "mtry" = mtry,
              "nodesizeSpl" = nodesizeSpl,
              "nodesizeAvg" = nodesizeAvg,
              "nodesizeStrictSpl" = nodesizeStrictSpl,
              "nodesizeStrictAvg" = nodesizeStrictAvg,
              "minSplitGain" = minSplitGain,
              "maxDepth" = maxDepth,
              "interactionDepth" = interactionDepth,
              "splitratio" = splitratio,
              "OOBhonest" = OOBhonest,
              "nthread" = nthread,
              "middleSplit" = middleSplit,
              "doubleTree" = doubleTree,
              "linFeats" = linFeats,
              "monotonicConstraints" = monotonicConstraints,
              "featureWeights" = featureWeights,
              "deepFeatureWeights" = deepFeatureWeights,
              "observationWeights" = observationWeights,
              "hasNas" = hasNas))
}

#' @title Test data check
#' @name testing_data_checker-forestry
#' @description Check the testing data to do prediction
#' @param object A forestry object.
#' @param newdata A data frame of testing predictors.
#' @param hasNas TRUE if the there were NAs in the training data FALSE otherwise.
#' @return A feature dataframe if it can be used for new predictions.
testing_data_checker <- function(object, newdata, hasNas) {
  if(ncol(newdata) != object@processed_dta$numColumns) {
    stop(paste0("newdata has ", ncol(newdata), " but the forest was trained with ",
                object@processed_dta$numColumns, " columns.")
    )
  }
  if(!is.null(object@processed_dta$featNames)) {
    if(!all(names(newdata) == object@processed_dta$featNames)) {
      warning("newdata columns have been reordered so that they match the training feature matrix")
      matchingPositions <- match(object@processed_dta$featNames, names(newdata))
      newdata <- newdata[, matchingPositions]
    }
  }

  # If linear is true we can't predict observations with some features missing.
  if(object@linear && any(is.na(newdata))) {
      stop("linear does not support missing data")
  }
  return(newdata)
}

sample_weights_checker <- function(featureWeights, mtry, ncol) {
    if(length(featureWeights) != ncol) {
      stop("featureWeights and deepFeatureWeights must have length ncol(x)")
    }
    if(any(featureWeights < 0)) {
      stop("The entries in featureWeights and deepFeatureWeights must be non negative")
    }
    if(sum(featureWeights) == 0) {
      stop("There must be at least one non-zero weight in featureWeights and deepFeatureWeights")
    }

  # "-1" needed when using zero-indexing in C++ code.
  featureWeightsVariables <- which(featureWeights > max(featureWeights)*0.001) - 1
  useAll <- length(featureWeightsVariables) < mtry
  featureWeights <- if (useAll)  numeric(0) else featureWeights
  return(list(featureWeightsVariables = featureWeightsVariables, featureWeights = featureWeights))
}

nullptr <- new("externalptr")
forest_checker <- function(object) {
  #' Checks if forestry object has valid pointer for C++ object.
  #' @param object a forestry object
  #' @return A message if the forest does not have a valid C++ pointer.
  if(identical(object@forest, nullptr)) {
    stop("Forest pointer is null. ",
         "Was the forest saved and loaded incorrectly? ",
         "To save and reload use make_savable and relinkCPP_prt.")
  }
}

# -- Random Forest Constructor -------------------------------------------------
setClass(
  Class = "forestry",
  slots = list(
    forest = "externalptr",
    dataframe = "externalptr",
    processed_dta = "list",
    R_forest = "list",
    categoricalFeatureCols = "list",
    categoricalFeatureMapping = "list",
    ntree = "numeric",
    replace = "logical",
    sampsize = "numeric",
    mtry = "numeric",
    nodesizeSpl = "numeric",
    nodesizeAvg = "numeric",
    nodesizeStrictSpl = "numeric",
    nodesizeStrictAvg = "numeric",
    minSplitGain = "numeric",
    maxDepth = "numeric",
    interactionDepth = "numeric",
    splitratio = "numeric",
    OOBhonest = "logical",
    doubleBootstrap = "logical",
    middleSplit = "logical",
    y = "vector",
    maxObs = "numeric",
    hasNas = "logical",
    linear = "logical",
    linFeats = "numeric",
    monotonicConstraints = "numeric",
    monotoneAvg = "logical",
    featureWeights = "numeric",
    featureWeightsVariables = "numeric",
    deepFeatureWeights = "numeric",
    deepFeatureWeightsVariables = "numeric",
    observationWeights = "numeric",
    overfitPenalty = "numeric",
    doubleTree = "logical",
    groupsMapping = "list",
    groups = "numeric"
  )
)

setClass(
  Class = "multilayerForestry",
  slots = list(
    forest = "externalptr",
    dataframe = "externalptr",
    processed_dta = "list",
    R_forests = "list",
    R_residuals = "list",
    categoricalFeatureCols = "list",
    categoricalFeatureMapping = "list",
    ntree = "numeric",
    nrounds = "numeric",
    eta = "numeric",
    replace = "logical",
    sampsize = "numeric",
    mtry = "numeric",
    nodesizeSpl = "numeric",
    nodesizeAvg = "numeric",
    nodesizeStrictSpl = "numeric",
    nodesizeStrictAvg = "numeric",
    minSplitGain = "numeric",
    maxDepth = "numeric",
    interactionDepth = "numeric",
    splitratio = "numeric",
    OOBhonest = "logical",
    doubleBootstrap = "logical",
    middleSplit = "logical",
    y = "vector",
    maxObs = "numeric",
    linear = "logical",
    linFeats = "numeric",
    monotonicConstraints = "numeric",
    monotoneAvg = "logical",
    featureWeights = "numeric",
    featureWeightsVariables = "numeric",
    deepFeatureWeights = "numeric",
    deepFeatureWeightsVariables = "numeric",
    observationWeights = "numeric",
    overfitPenalty = "numeric",
    gammas = "numeric",
    doubleTree = "logical",
    groupsMapping = "list",
    groups = "numeric"
  )
)


#' @title forestry
#' @rdname forestry
#' @param x A data frame of all training predictors.
#' @param y A vector of all training responses.
#' @param ntree The number of trees to grow in the forest. The default value is
#'   500.
#' @param replace An indicator of whether sampling of training data is with
#'   replacement. The default value is TRUE.
#' @param sampsize The size of total samples to draw for the training data. If
#'   sampling with replacement, the default value is the length of the training
#'   data. If sampling without replacement, the default value is two-thirds of
#'   the length of the training data.
#' @param sample.fraction If this is given, then sampsize is ignored and set to
#'   be round(length(y) * sample.fraction). It must be a real number between 0 and 1
#' @param mtry The number of variables randomly selected at each split point.
#'   The default value is set to be one-third of the total number of features of the training data.
#' @param nodesizeSpl Minimum observations contained in terminal nodes.
#'   The default value is 5.
#' @param nodesizeAvg Minimum size of terminal nodes for averaging dataset.
#'   The default value is 5.
#' @param nodesizeStrictSpl Minimum observations to follow strictly in terminal nodes.
#'   The default value is 1.
#' @param nodesizeStrictAvg The minimum size of terminal nodes for averaging data set to follow when predicting.
#'   No splits are allowed that result in nodes with observations less than this parameter.
#'   This parameter enforces overlap of the averaging data set with the splitting set when training.
#'   When using honesty, splits that leave less than nodesizeStrictAvg averaging
#'   observations in either child node will be rejected, ensuring every leaf node
#'   also has at least nodesizeStrictAvg averaging observations. The default value is 1.
#' @param minSplitGain Minimum loss reduction to split a node further in a tree.
#' @param maxDepth Maximum depth of a tree. The default value is 99.
#' @param interactionDepth All splits at or above interaction depth must be on
#'   variables that are not weighting variables (as provided by the interactionVariables argument).
#' @param interactionVariables Indices of weighting variables.
#' @param featureWeights (optional) vector of sampling probabilities/weights for each
#'   feature used when subsampling mtry features at each node above or at interactionDepth.
#'   The default is to use uniform probabilities.
#' @param deepFeatureWeights Used in place of featureWeights for splits below interactionDepth.
#' @param observationWeights Denotes the weights for each training observation
#'   that determine how likely the observation is to be selected in each bootstrap sample.
#'   This option is not allowed when sampling is done without replacement.
#' @param splitratio Proportion of the training data used as the splitting dataset.
#'   It is a ratio between 0 and 1. If the ratio is 1 (the default), then the splitting
#'   set uses the entire data, as does the averaging set---i.e., the standard Breiman RF setup.
#'   If the ratio is 0, then the splitting data set is empty, and the entire dataset is used
#'   for the averaging set (This is not a good usage, however, since there will be no data available for splitting).
#' @param OOBhonest In this version of honesty, the out-of-bag observations for each tree
#'   are used as the honest (averaging) set. This setting also changes how predictions
#'   are constructed. When predicting for observations that are out-of-sample
#'   (using Predict(..., aggregation = "average")), all the trees in the forest
#'   are used to construct predictions. When predicting for an observation that was in-sample (using
#'   predict(..., aggregation = "oob")), only the trees for which that observation
#'   was not in the averaging set are used to construct the prediction for that observation.
#'   aggregation="oob" (out-of-bag) ensures that the outcome value for an observation
#'   is never used to construct predictions for a given observation even when it is in sample.
#'   This property does not hold in standard honesty, which relies on an asymptotic subsampling argument.
#' @param doubleBootstrap The doubleBootstrap flag provides the option to resample
#'   with replacement from the out-of-bag observations set for each tree to construct
#'   the averaging set when using OOBhonest. If this is FALSE, the out-of-bag observations
#'   are used as the averaging set. By default this option is TRUE when running OOBhonest = TRUE.
#'   This option increases diversity across trees.
#' @param seed random seed
#' @param verbose Indicator to train the forest in verbose mode
#' @param nthread Number of threads to train and predict the forest. The default
#'   number is 0 which represents using all cores.
#' @param splitrule Only variance is implemented at this point and it
#'   specifies the loss function according to which the splits of random forest
#'   should be made.
#' @param middleSplit Indicator of whether the split value is takes the average of two feature
#'   values. If FALSE, it will take a point based on a uniform distribution
#'   between two feature values. (Default = FALSE)
#' @param doubleTree if the number of tree is doubled as averaging and splitting
#'   data can be exchanged to create decorrelated trees. (Default = FALSE)
#' @param reuseforestry Pass in an `forestry` object which will recycle the
#'   dataframe the old object created. It will save some space working on the
#'   same data set.
#' @param maxObs The max number of observations to split on.
#' @param savable If TRUE, then RF is created in such a way that it can be
#'   saved and loaded using save(...) and load(...). However, setting it to TRUE
#'   (default) will take longer and use more memory. When
#'   training many RF, it makes sense to set this to FALSE to save time and memory.
#' @param saveable deprecated. Do not use.
#' @param linear Indicator that enables Ridge penalized splits and linear aggregation
#'   functions in the leaf nodes. This is recommended for data with linear outcomes.
#'   For implementation details, see: https://arxiv.org/abs/1906.06463. Default is FALSE.
#' @param linFeats A vector containing the indices of which features to split
#'   linearly on when using linear penalized splits (defaults to use all numerical features).
#' @param monotonicConstraints Specifies monotonic relationships between the continuous
#'   features and the outcome. Supplied as a vector of length p with entries in
#'   1,0,-1 which 1 indicating an increasing monotonic relationship, -1 indicating
#'   a decreasing monotonic relationship, and 0 indicating no constraint.
#'   Constraints supplied for categorical variable will be ignored.
#' @param groups A vector of factors specifying the group membership of each training observation.
#'   these groups are used in the aggregation when doing out of bag predictions in
#'   order to predict with only trees where the entire group was not used for aggregation.
#'   This allows the user to specify custom subgroups which will be used to create
#'   predictions which do not use any data from a common group to make predictions for
#'   any observation in the group. This can be used to create general custom
#'   resampling schemes, and provide predictions consistent with the Out-of-Group set.
#' @param monotoneAvg This is a boolean flag that indicates whether or not monotonic
#'   constraints should be enforced on the averaging set in addition to the splitting set.
#'   This flag is meaningless unless both honesty and monotonic constraints are in use.
#'   The default is FALSE.
#' @param overfitPenalty Value to determine how much to penalize the magnitude
#'   of coefficients in ridge regression when using linear splits.
#' @return A `forestry` object.
#' @examples
#' set.seed(292315)
#' library(Rforestry)
#' test_idx <- sample(nrow(iris), 3)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train, nthread = 2)
#' predict(rf, x_test)
#'
#' set.seed(49)
#' library(Rforestry)
#'
#' n <- c(100)
#' a <- rnorm(n)
#' b <- rnorm(n)
#' c <- rnorm(n)
#' y <- 4*a + 5.5*b - .78*c
#' x <- data.frame(a,b,c)
#'
#' forest <- forestry(
#'           x,
#'           y,
#'           ntree = 10,
#'           replace = TRUE,
#'           nodesizeStrictSpl = 5,
#'           nodesizeStrictAvg = 5,
#'           nthread = 2,
#'           linear = TRUE
#'           )
#'
#' predict(forest, x)
#' @note
#' Treatment of Missing Data
#'
#' In version 0.9.0.34, we have modified the handling of missing data. Instead of
#' the greedy approach used in previous iterations, we now test any potential
#' split by putting all NA's to the right, and all NA's to the left, and taking
#' the choice which gives the best MSE for the split. Under this version of handling
#' the potential splits, we will still respect monotonic constraints. So if we put all
#' NA's to either side, and the resulting leaf nodes have means which violate
#' the monotone constraints, the split will be rejected.
#' @export
forestry <- function(x,
                     y,
                     ntree = 500,
                     replace = TRUE,
                     sampsize = if (replace)
                       nrow(x)
                     else
                       ceiling(.632 * nrow(x)),
                     sample.fraction = NULL,
                     mtry = max(floor(ncol(x) / 3), 1),
                     nodesizeSpl = 5,
                     nodesizeAvg = 5,
                     nodesizeStrictSpl = 1,
                     nodesizeStrictAvg = 1,
                     minSplitGain = 0,
                     maxDepth = round(nrow(x) / 2) + 1,
                     interactionDepth = maxDepth,
                     interactionVariables = numeric(0),
                     featureWeights = NULL,
                     deepFeatureWeights = NULL,
                     observationWeights = NULL,
                     splitratio = 1,
                     OOBhonest = FALSE,
                     doubleBootstrap = if (OOBhonest)
                       TRUE
                     else
                       FALSE,
                     seed = as.integer(runif(1) * 1000),
                     verbose = FALSE,
                     nthread = 0,
                     splitrule = "variance",
                     middleSplit = FALSE,
                     maxObs = length(y),
                     linear = FALSE,
                     linFeats = 0:(ncol(x)-1),
                     monotonicConstraints = rep(0, ncol(x)),
                     groups = NULL,
                     monotoneAvg = FALSE,
                     overfitPenalty = 1,
                     doubleTree = FALSE,
                     reuseforestry = NULL,
                     savable = TRUE,
                     saveable = TRUE) {
  if(is.matrix(x) && is.null(colnames(x))) {
    message("x does not have column names. The check that columns are provided in the same order
            when training and predicting will be skipped")
  }
  featNames <-
    if(is.matrix(x)) {
      colnames(x)
    } else if(is.data.frame(x)) {
      names(x)
    } else
      stop("x must be a matrix or data.frame")

  x <- as.data.frame(x)
  # only if sample.fraction is given, update sampsize
  if (!is.null(sample.fraction))
    sampsize <- ceiling(sample.fraction * nrow(x))
  linFeats <- unique(linFeats)

  x <- as.data.frame(x)
  # Preprocess the data
  hasNas <- any(is.na(x))

  #Translating interactionVariables to featureWeights syntax
  if(is.null(featureWeights)) {
    featureWeights <- rep(1, ncol(x))
    featureWeights[interactionVariables + 1] <- 0
  }
  if(is.null(deepFeatureWeights)) {
    deepFeatureWeights <- rep(1, ncol(x))
  }
  if(is.null(observationWeights)) {
    observationWeights <- rep(1, nrow(x))
  }
  updated_variables <-
    training_data_checker(
      x = x,
      y = y,
      ntree = ntree,
      replace = replace,
      sampsize = sampsize,
      mtry = mtry,
      nodesizeSpl = nodesizeSpl,
      nodesizeAvg = nodesizeAvg,
      nodesizeStrictSpl = nodesizeStrictSpl,
      nodesizeStrictAvg = nodesizeStrictAvg,
      minSplitGain = minSplitGain,
      maxDepth = maxDepth,
      interactionDepth = interactionDepth,
      splitratio = splitratio,
      OOBhonest = OOBhonest,
      nthread = nthread,
      middleSplit = middleSplit,
      doubleTree = doubleTree,
      linFeats = linFeats,
      monotonicConstraints = monotonicConstraints,
      groups = groups,
      featureWeights = featureWeights,
      deepFeatureWeights = deepFeatureWeights,
      observationWeights = observationWeights,
      linear = linear,
      hasNas = hasNas)

  for (variable in names(updated_variables)) {
    assign(x = variable, value = updated_variables[[variable]],
           envir = environment())
  }

  sample_weights_check <- sample_weights_checker(featureWeights, mtry, ncol = ncol(x))
  deep_sample_weights_check <- sample_weights_checker(deepFeatureWeights, mtry, ncol = ncol(x))
  featureWeights <- sample_weights_check$featureWeights
  featureWeightsVariables <- sample_weights_check$featureWeightsVariables
  deepFeatureWeights <- deep_sample_weights_check$featureWeights
  deepFeatureWeightsVariables <- deep_sample_weights_check$featureWeightsVariables

  # Total number of obervations
  nObservations <- length(y)
  numColumns <- ncol(x)

  groupsMapping <- list()
  if (!is.null(groups)) {
    groupsMapping <- list("groupValue" = levels(groups),
                          "groupNumericValue" = 1:length(levels(groups)))

  }

  if (!is.null(groups)) {
    groupVector <- as.integer(groups)
  } else {
    groupVector <- rep(0, nrow(x))
  }

  if (is.null(reuseforestry)) {
    preprocessedData <- preprocess_training(x, y)
    processed_x <- preprocessedData$x
    categoricalFeatureCols <-
      preprocessedData$categoricalFeatureCols
    categoricalFeatureMapping <-
      preprocessedData$categoricalFeatureMapping

    categoricalFeatureCols_cpp <- unlist(categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      # If we have monotonic constraints on any categorical features we need to
      # zero these out as we cannot do monotonicity with categorical features
      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }
    # Create rcpp object
    # Create a forest object
    forest <- tryCatch({
      rcppDataFrame <- rcpp_cppDataFrameInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        featureWeights = featureWeights,
        featureWeightsVariables = featureWeightsVariables,
        deepFeatureWeights =  deepFeatureWeights,
        deepFeatureWeightsVariables = deepFeatureWeightsVariables,
        observationWeights = observationWeights,
        monotonicConstraints = monotonicConstraints,
        groupMemberships = groupVector,
        monotoneAvg = monotoneAvg
      )

      rcppForest <- rcpp_cppBuildInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        replace,
        sampsize,
        mtry,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        interactionDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        featureWeights,
        featureWeightsVariables,
        deepFeatureWeights,
        deepFeatureWeightsVariables,
        observationWeights,
        monotonicConstraints,
        groupVector,
        monotoneAvg,
        hasNas,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )
      processed_dta <- list(
        "processed_x" = processed_x,
        "y" = y,
        "categoricalFeatureCols_cpp" = categoricalFeatureCols_cpp,
        "linearFeatureCols_cpp" = linFeats,
        "nObservations" = nObservations,
        "numColumns" = numColumns,
        "featNames" = featNames
      )
      R_forest <- list()

      return(
        new(
          "forestry",
          forest = rcppForest,
          dataframe = rcppDataFrame,
          processed_dta = processed_dta,
          R_forest = R_forest,
          categoricalFeatureCols = categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          interactionDepth = interactionDepth,
          splitratio = splitratio,
          OOBhonest = OOBhonest,
          doubleBootstrap = doubleBootstrap,
          middleSplit = middleSplit,
          maxObs = maxObs,
          featureWeights = featureWeights,
          featureWeightsVariables = featureWeightsVariables,
          deepFeatureWeights =  deepFeatureWeights,
          deepFeatureWeightsVariables = deepFeatureWeightsVariables,
          observationWeights = observationWeights,
          hasNas = hasNas,
          linear = linear,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = groupsMapping,
          groups = groupVector
        )
      )
    },
    error = function(err) {
      print(err)
      return(NULL)
    })

  } else {
    categoricalFeatureCols_cpp <-
      unlist(reuseforestry@categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      # If we have monotonic constraints on any categorical features we need to
      # zero these out as we cannot do monotonicity with categorical features
      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    categoricalFeatureMapping <-
      reuseforestry@categoricalFeatureMapping

    # Create rcpp object
    # Create a forest object
    forest <- tryCatch({
      rcppForest <- rcpp_cppBuildInterface(
        x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        replace,
        sampsize,
        mtry,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        interactionDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        featureWeights = featureWeights,
        featureWeightsVariables = featureWeightsVariables,
        deepFeatureWeights =  deepFeatureWeights,
        deepFeatureWeightsVariables = deepFeatureWeightsVariables,
        observationWeights,
        monotonicConstraints,
        groupVector,
        monotoneAvg,
        hasNas,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        reuseforestry@dataframe
      )

      return(
        new(
          "forestry",
          forest = rcppForest,
          dataframe = reuseforestry@dataframe,
          processed_dta = reuseforestry@processed_dta,
          R_forest = reuseforestry@R_forest,
          categoricalFeatureCols = reuseforestry@categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          interactionDepth = interactionDepth,
          splitratio = splitratio,
          OOBhonest = OOBhonest,
          doubleBootstrap = doubleBootstrap,
          middleSplit = middleSplit,
          maxObs = maxObs,
          featureWeights = featureWeights,
          deepFeatureWeights = deepFeatureWeights,
          observationWeights = observationWeights,
          hasNas = hasNas,
          linear = linear,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = groupsMapping,
          groups = groupVector
        )
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })
  }
  if(savable)
    forest <- make_savable(forest)
  return(forest)
}

# -- Multilayer Random Forest Constructor --------------------------------------
#' @name multilayer-forestry
#' @title Multilayer forestry
#' @rdname multilayer-forestry
#' @description Construct a gradient boosted ensemble with random forest base learners.
#' @inheritParams forestry
#' @param featureWeights weights used when subsampling features for nodes above or at interactionDepth.
#' @param deepFeatureWeights weights used when subsampling features for nodes below interactionDepth.
#' @param nrounds Number of iterations used for gradient boosting.
#' @param eta Step size shrinkage used in gradient boosting update.
#' @return A `multilayerForestry` object.
#' @export
multilayerForestry <- function(x,
                     y,
                     ntree = 500,
                     nrounds = 1,
                     eta = 0.3,
                     replace = FALSE,
                     sampsize = nrow(x),
                     sample.fraction = NULL,
                     mtry = ncol(x),
                     nodesizeSpl = 3,
                     nodesizeAvg = 3,
                     nodesizeStrictSpl = max(round(nrow(x)/128), 1),
                     nodesizeStrictAvg = max(round(nrow(x)/128), 1),
                     minSplitGain = 0,
                     maxDepth = 99,
                     splitratio = 1,
                     OOBhonest = FALSE,
                     doubleBootstrap = if (OOBhonest)
                       TRUE
                     else
                       FALSE,
                     seed = as.integer(runif(1) * 1000),
                     verbose = FALSE,
                     nthread = 0,
                     splitrule = "variance",
                     middleSplit = TRUE,
                     maxObs = length(y),
                     linear = FALSE,
                     linFeats = 0:(ncol(x)-1),
                     monotonicConstraints = rep(0, ncol(x)),
                     groups = NULL,
                     monotoneAvg = FALSE,
                     featureWeights = rep(1, ncol(x)),
                     deepFeatureWeights = featureWeights,
                     observationWeights = NULL,
                     overfitPenalty = 1,
                     doubleTree = FALSE,
                     reuseforestry = NULL,
                     savable = TRUE,
                     saveable = saveable
) {
  # Check for named columns
  if(is.matrix(x) && is.null(colnames(x))) {
    message("x does not have column names. The check that columns are provided in the same order
            when training and predicting will be skipped")
  }
  # only if sample.fraction is given, update sampsize
  if (!is.null(sample.fraction))
    sampsize <- ceiling(sample.fraction * nrow(x))
  linFeats <- unique(linFeats)

  x <- as.data.frame(x)
  hasNas <- any(is.na(x))
  if (hasNas) {
    stop("x has missing data.")
  }

  #Translating interactionVariables to featureWeights syntax
  if(is.null(featureWeights)) {
    interactionVariables <- numeric(0)
    featureWeights <- rep(1, ncol(x))
    featureWeights[interactionVariables + 1] <- 0
  }
  if(is.null(deepFeatureWeights)) {
    deepFeatureWeights <- rep(1, ncol(x))
  }
  if(is.null(observationWeights)) {
    observationWeights <- rep(1, nrow(x))
  }

  # Preprocess the data
  updated_variables <-
    training_data_checker(
      x = x,
      y = y,
      ntree = ntree,
      replace = replace,
      sampsize = sampsize,
      mtry = mtry,
      nodesizeSpl = nodesizeSpl,
      nodesizeAvg = nodesizeAvg,
      nodesizeStrictSpl = nodesizeStrictSpl,
      nodesizeStrictAvg = nodesizeStrictAvg,
      minSplitGain = minSplitGain,
      maxDepth = maxDepth,
      interactionDepth = maxDepth, # Make maxdepth for multilayer
      splitratio = splitratio,
      OOBhonest = OOBhonest,
      nthread = nthread,
      middleSplit = middleSplit,
      doubleTree = doubleTree,
      linFeats = linFeats,
      monotonicConstraints = monotonicConstraints,
      featureWeights = featureWeights,
      deepFeatureWeights = deepFeatureWeights,
      observationWeights = observationWeights,
      groups = groups,
      linear = linear,
      hasNas = hasNas)

  for (variable in names(updated_variables)) {
    assign(x = variable, value = updated_variables[[variable]],
           envir = environment())
  }

  sample_weights_check <- sample_weights_checker(featureWeights, mtry, ncol = ncol(x))
  deep_sample_weights_check <- sample_weights_checker(deepFeatureWeights, mtry, ncol = ncol(x))
  featureWeights <- sample_weights_check$featureWeights
  featureWeightsVariables <- sample_weights_check$featureWeightsVariables
  deepFeatureWeights <- deep_sample_weights_check$featureWeights
  deepFeatureWeightsVariables <- deep_sample_weights_check$featureWeightsVariables

  # Total number of obervations
  nObservations <- length(y)
  numColumns <- ncol(x)

  groupsMapping <- list()
  if (!is.null(groups)) {
    groupsMapping <- list("groupValue" = levels(groups),
                          "groupNumericValue" = 1:length(levels(groups)))

  }
  if (!is.null(groups)) {
    groupVector <- as.integer(groups)
  } else {
    groupVector <- rep(0, nrow(x))
  }

  if (is.null(reuseforestry)) {
    preprocessedData <- preprocess_training(x, y)
    processed_x <- preprocessedData$x
    categoricalFeatureCols <-
      preprocessedData$categoricalFeatureCols
    categoricalFeatureMapping <-
      preprocessedData$categoricalFeatureMapping

    categoricalFeatureCols_cpp <- unlist(categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    # Create rcpp object
    # Create a forest object
    multilayerForestry <- tryCatch({
      rcppDataFrame <- rcpp_cppDataFrameInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        featureWeights = featureWeights,
        featureWeightsVariables = featureWeightsVariables,
        deepFeatureWeights =  deepFeatureWeights,
        deepFeatureWeightsVariables = deepFeatureWeightsVariables,
        observationWeights = observationWeights,
        monotonicConstraints = monotonicConstraints,
        groupMemberships = groupVector,
        monotoneAvg = monotoneAvg
      )

      rcppForest <- rcpp_cppMultilayerBuildInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        nrounds,
        eta,
        replace,
        sampsize,
        mtry,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        featureWeights,
        featureWeightsVariables,
        deepFeatureWeights,
        deepFeatureWeightsVariables,
        observationWeights,
        monotonicConstraints,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )
      processed_dta <- list(
        "processed_x" = processed_x,
        "y" = y,
        "categoricalFeatureCols_cpp" = categoricalFeatureCols_cpp,
        "linearFeatureCols_cpp" = linFeats,
        "nObservations" = nObservations,
        "numColumns" = numColumns
      )
      R_forests <- list()
      gammas <- rep(0,nrounds)
      R_residuals <- list()

      return(
        new(
          "multilayerForestry",
          forest = rcppForest,
          dataframe = rcppDataFrame,
          processed_dta = processed_dta,
          R_forests = R_forests,
          R_residuals = R_residuals,
          categoricalFeatureCols = categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          nrounds = nrounds,
          eta = eta,
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          OOBhonest = OOBhonest,
          doubleBootstrap = doubleBootstrap,
          middleSplit = middleSplit,
          maxObs = maxObs,
          featureWeights = featureWeights,
          featureWeightsVariables = featureWeightsVariables,
          deepFeatureWeights =  deepFeatureWeights,
          deepFeatureWeightsVariables = deepFeatureWeightsVariables,
          observationWeights = observationWeights,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          linear = linear,
          linFeats = linFeats,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = groupsMapping,
          gammas = gammas,
          groups = groupVector
        )
      )
    },
    error = function(err) {
      print(err)
      return(NULL)
    })

  } else {
    categoricalFeatureCols_cpp <-
      unlist(reuseforestry@categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    categoricalFeatureMapping <-
      reuseforestry@categoricalFeatureMapping

    # Create rcpp object
    # Create a forest object
    multilayerForestry <- tryCatch({
      rcppForest <- rcpp_cppMultilayerBuildInterface(
        x,
        y,
        categoricalFeatureCols_cpp,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        nrounds,
        eta,
        replace,
        sampsize,
        mtry,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        featureWeights,
        monotonicConstraints,
        observationWeights,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        reuseforestry@dataframe
      )

      return(
        new(
          "multilayerForestry",
          forest = rcppForest,
          dataframe = reuseforestry@dataframe,
          processed_dta = reuseforestry@processed_dta,
          R_forests = reuseforestry@R_forests,
          R_residuals = reuseforestry@R_residuals,
          categoricalFeatureCols = reuseforestry@categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          OOBhonest = OOBhonest,
          doubleBootstrap = doubleBootstrap,
          middleSplit = middleSplit,
          maxObs = maxObs,
          featureWeights = featureWeights,
          observationWeights = observationWeights,
          linear = linear,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = reuseforestry@groupsMapping,
          gammas = reuseforestry@gammas,
          groups = groupVector
        )
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })
  }
  if(savable || saveable)
    multilayerForestry <- make_savable(multilayerForestry)
  return(multilayerForestry)
}

# -- Predict Method ------------------------------------------------------------
#' predict-forestry
#' @name predict-forestry
#' @rdname predict-forestry
#' @description Return the prediction from the forest.
#' @param object A `forestry` object.
#' @param newdata A data frame of testing predictors.
#' @param aggregation How the individual tree predictions are aggregated:
#'   `average` returns the mean of all trees in the forest; `weightMatrix`
#'   returns a list consisting of "weightMatrix", the adaptive nearest neighbor
#'   weights used to construct the predictions; `terminalNodes` also returns
#'   the weightMatrix, as well as "terminalNodes", a matrix where
#'   the ith entry of the jth column is the index of the leaf node to which the
#'   ith observation is assigned in the jth tree; and "sparse", a matrix
#'   where the ith entry in the jth column is 1 if the ith observation in
#'   newdata is assigned to the jth leaf and 0 otherwise. In each tree the
#'   leaves are indexed using a depth first ordering, and, in the "sparse"
#'   representation, the first leaf in the second tree has column index one more than
#'   the number of leaves in the first tree and so on. So, for example, if the
#'   first tree has 5 leaves, the sixth column of the "sparse" matrix corresponds
#'   to the first leaf in the second tree.
#'   `oob` returns the out-of-bag predictions for the forest. We assume
#'   that the ordering of the observations in newdata have not changed from
#'   training. If the ordering has changed, we will get the wrong OOB indices.
#'   `doubleOOB` is an experimental flag, which can only be used when OOBhonest = TRUE
#'   and doubleBootstrap = TRUE. When both of these settings are on, the
#'   splitting set is selected as a bootstrap sample of observations and the
#'   averaging set is selected as a bootstrap sample of the observations which
#'   were left out of bag during the splitting set selection. This leaves a third
#'   set which is the observations which were not selected in either bootstrap sample.
#'   This predict flag gives the predictions using- for each observation- only the trees
#'   in which the observation fell into this third set (so was neither a splitting
#'   nor averaging example).
#'   `coefs` is an aggregation option which works only when linear aggregation
#'   functions have been used. This returns the linear coefficients for each
#'   linear feature which were used in the leaf node regression of each predicted
#'   point.
#' @param seed random seed
#' @param nthread The number of threads with which to run the predictions with.
#'   This will default to the number of threads with which the forest was trained
#'   with.
#' @param exact This specifies whether the forest predictions should be aggregated
#'   in a reproducible ordering. Due to the non-associativity of floating point
#'   addition, when we predict in parallel, predictions will be aggregated in
#'   varied orders as different threads finish at different times.
#'   By default, exact is TRUE unless N > 100,000 or a custom aggregation
#'   function is used.
#' @param trees A vector (1-indexed) of indices in the range 1:ntree which tells
#'   predict which trees in the forest to use for the prediction. Predict will by
#'   default take the average of all trees in the forest, although this flag
#'   can be used to get single tree predictions, or averages of diffferent trees
#'   with different weightings. Duplicate entries are allowed, so if trees = c(1,2,2)
#'   this will predict the weighted average prediction of only trees 1 and 2 weighted by:
#'   predict(..., trees = c(1,2,2)) = (predict(..., trees = c(1)) +
#'                                      2*predict(..., trees = c(2))) / 3.
#'   note we must have exact = TRUE, and aggregation = "average" to use tree indices.
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @export
predict.forestry <- function(object,
                             newdata = NULL,
                             aggregation = "average",
                             seed = as.integer(runif(1) * 10000),
                             nthread = 0,
                             exact = NULL,
                             trees = NULL,
                             ...) {

  if (is.null(newdata) && !(aggregation == "oob" || aggregation == "doubleOOB")) {
    stop("When using an aggregation that is not oob or doubleOOB, one must supply newdata")
  }

  if ((!(object@linear)) && (aggregation == "coefs")) {
    stop("Aggregation can only be linear with setting the parameter linear = TRUE.")
  }

  # Preprocess the data. We only run the data checker if ridge is turned on,
  # because even in the case where there were no NAs in train, we still want to predict.
  if (!is.null(newdata)) {
    forest_checker(object)
    newdata <- testing_data_checker(object, newdata, object@hasNas)
    newdata <- as.data.frame(newdata)

    processed_x <- preprocess_testing(newdata,
                                      object@categoricalFeatureCols,
                                      object@categoricalFeatureMapping)

  }

  # Set exact aggregation method if nobs < 100,000 and average aggregation
  if (is.null(exact)) {
    if (nrow(newdata) > 1e5 || aggregation != "average") {
      exact = FALSE
    } else {
      exact = TRUE
    }
  }

  # We can only use tree aggregations if exact = TRUE and aggregation = "average"
  if (!is.null(trees) && ((exact != TRUE) || (aggregation != "average"))) {
    stop("When using tree indices, we must have exact = TRUE and aggregation = \"average\" ")
  }

  if (any(trees < 1) || any(trees > object@ntree) || any(trees %% 1 != 0)) {
    stop("trees must contain indices which are integers between 1 and ntree")
  }

  # If trees are being used, we need to convert them into a weight vector
  if (!is.null(trees)) {
    tree_weights <- rep(0, object@ntree)
    for (i in 1:length(trees)) {
      tree_weights[trees[i]] = tree_weights[trees[i]] + 1
    }
    use_weights <- TRUE
  } else {
    tree_weights <- rep(0, object@ntree)
    use_weights <- FALSE
  }


  # If option set to terminalNodes, we need to make matrix of ID's
  if (aggregation == "oob") {

    if (!is.null(newdata) && (object@processed_dta$nObservations != nrow(newdata))) {
      warning(paste(
        "Attempting to do OOB predictions on a dataset which doesn't match the",
        "training data!"
      ))
      return(NA)
    }

    if (is.null(newdata)) {
      rcppPrediction <- tryCatch({
        rcpp_OBBPredictionsInterface(object@forest,
                                     NULL,  # Give null for the dataframe
                                     FALSE, # Tell predict we don't have an existing dataframe
                                     FALSE
        )
      }, error = function(err) {
        print(err)
        return(NULL)
      })
    } else {
      rcppPrediction <- tryCatch({
        rcpp_OBBPredictionsInterface(object@forest,
                                     processed_x,
                                     TRUE, # Give dataframe flag
                                     FALSE
        )
      }, error = function(err) {
        print(err)
        return(NULL)
      })
    }

  } else if (aggregation == "doubleOOB") {

    if (!is.null(newdata) && (object@sampsize != nrow(newdata))) {
      warning(paste(
        "Attempting to do OOB predictions on a dataset which doesn't match the",
        "training data!"
      ))
      return(NA)
    }

    if (!object@doubleBootstrap) {
      warning(paste(
        "Attempting to do double OOB predictions with a forest that was not trained
        with doubleBootstrap = TRUE"
      ))
      return(NA)
    }

    if (is.null(newdata)) {
      rcppPrediction <- tryCatch({
        rcpp_OBBPredictionsInterface(object@forest,
                                     NULL,  # Give null for the dataframe
                                     FALSE, # Tell predict we don't have an existing dataframe
                                     TRUE
        )
      }, error = function(err) {
        print(err)
        return(NULL)
      })
    } else {
      rcppPrediction <- tryCatch({
        rcpp_OBBPredictionsInterface(object@forest,
                                     processed_x,
                                     TRUE, # Give dataframe flag
                                     TRUE
        )
      }, error = function(err) {
        print(err)
        return(NULL)
      })
    }

  } else {
    rcppPrediction <- tryCatch({
      rcpp_cppPredictInterface(object@forest,
                               processed_x,
                               aggregation,
                               seed = seed,
                               nthread = nthread,
                               exact = exact,
                               use_weights = use_weights,
                               tree_weights = tree_weights)
    }, error = function(err) {
      print(err)
      return(NULL)
    })
  }

  # In the case aggregation is set to "linear"
  # rccpPrediction is a list with an entry $coef
  # which gives pointwise regression coeffficients averaged across the forest
  if (aggregation == "coefs") {
    if(length(object@linFeats) == 1) {
      newdata <- data.frame(newdata)
    }
    coef_names <- colnames(newdata)[object@linFeats + 1]
    coef_names <- c(coef_names, "Intercept")
    colnames(rcppPrediction$coef) <- coef_names
  }

  if (aggregation == "average") {
    return(rcppPrediction$prediction)
  } else if (aggregation == "oob") {
    return(rcppPrediction)
  } else if (aggregation == "doubleOOB") {
    return(rcppPrediction)
  } else if (aggregation == "weightMatrix") {
    return(rcppPrediction)
  } else if (aggregation == "coefs") {
    return(rcppPrediction)
  } else if (aggregation == "terminalNodes") {
    terminalNodes <- rcppPrediction$terminalNodes
    nobs <- nrow(newdata)
    ntree <- object@ntree
    sparse_rep <- matrix(nrow = nobs, ncol = 0)
    for (i in 1:ntree) {
      sparse_rep_single_tree <- matrix(data = rep(0, terminalNodes[nobs+1,i]),
                                       nrow = nobs,
                                       ncol = terminalNodes[nobs+1,i])
      for (j in 1:nobs) {
        sparse_rep_single_tree[j,terminalNodes[j,i]] <- 1
      }
      sparse_rep <- cbind(sparse_rep, sparse_rep_single_tree)
    }
    rcppPrediction[["sparse"]] <- sparse_rep
    return(rcppPrediction)
  }
}



# -- Multilayer Predict Method -------------------------------------------------------
#' predict-multilayer-forestry
#' @name predict-multilayer-forestry
#' @rdname predict-multilayer-forestry
#' @description Return the prediction from the forest.
#' @param object A `multilayerForestry` object.
#' @param newdata A data frame of testing predictors.
#' @param aggregation How shall the leaf be aggregated. The default is to return
#'   the mean of the leave `average`. Other options are `weightMatrix` which
#'   returns the adaptive nearest neighbor weights used to construct the predictions.
#' @param seed random seed
#' @param nthread The number of threads with which to run the predictions with.
#'   This will default to the number of threads with which the forest was trained
#'   with.
#' @param exact This specifies whether the forest predictions should be aggregated
#'   in a reproducible ordering. Due to the non-associativity of floating point
#'   addition, when we predict in parallel, predictions will be aggregated in
#'   varied orders as different threads finish at different times.
#'   By default, exact is TRUE unless N > 100,000 or a custom aggregation
#'   function is used.
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @export
predict.multilayerForestry <- function(object,
                                       newdata,
                                       aggregation = "average",
                                       seed = as.integer(runif(1) * 10000),
                                       nthread = 0,
                                       exact = NULL,
                                       ...) {
    forest_checker(object)

    if (is.null(exact)) {
      if (nrow(newdata) > 1e5 || aggregation != "average") {
        exact = FALSE
      } else {
        exact = TRUE
      }
    }

   # Preprocess the data. We only run the data checker if ridge is turned on,
   # because even in the case where there were no NAs in train, we still want to predict.
    if(object@linear) {
      testing_data_checker(newdata, FALSE)
    }

    processed_x <- preprocess_testing(newdata,
                                      object@categoricalFeatureCols,
                                      object@categoricalFeatureMapping)

    rcppPrediction <- tryCatch({
      rcpp_cppMultilayerPredictInterface(object@forest, processed_x,
                                         aggregation, seed, nthread, exact)
    }, error = function(err) {
      print(err)
      return(NULL)
    })

    if (aggregation == "average") {
      return(rcppPrediction$prediction)
    } else if (aggregation == "weightMatrix") {
      return(rcppPrediction)
    }
  }



# -- Calculate OOB Error -------------------------------------------------------
#' getOOB-forestry
#' @name getOOB-forestry
#' @rdname getOOB-forestry
#' @description Calculate the out-of-bag error of a given forest. This is done
#' by using the out-of-bag predictions for each observation, and calculating the
#' MSE over the entire forest.
#' @param object A `forestry` object.
#' @param noWarning flag to not display warnings
#' @aliases getOOB,forestry-method
#' @return The OOB error of the forest.
#' @export
getOOB <- function(object,
                   noWarning) {
    # TODO (all): find a better threshold for throwing such warning. 25 is
    # currently set up arbitrarily.
  forest_checker(object)
    if (!object@replace &&
        object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                        object@sampsize) < 10) {
      if (!noWarning) {
        warning(paste(
          "Samples are drawn without replacement and sample size",
          "is too big!"
        ))
      }
      return(NA)
    }

    rcppOOB <- tryCatch({
      return(rcpp_OBBPredictInterface(object@forest))
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppOOB)
}

# -- Calculate Splitting Proportions -------------------------------------------
#' getSplitProps-forestry
#' @name getSplitProps-forestry
#' @rdname getSplitProps-forestry
#' @description Retrieves the proportion of splits for each feature in the given
#'  forestry object. These proportions are calculated as the number of splits
#'  on feature i in the entire forest over total the number of splits in the
#'  forest.
#' @param object A trained model object of class "forestry".
#' @return A vector of length equal to the number of columns
#' @seealso \code{\link{forestry}}
#' @export
getSplitProps <- function(object) {

  # Make forest saveable so we can access the tree data in R
  object <- make_savable(object)

  # Dataframe to hold the splitting counts for each tree
  data <- data.frame(matrix(rep(0,
                                object@ntree*length(object@processed_dta$featNames)),
                            nrow = object@ntree))
  # Store feat names
  names(data) <- object@processed_dta$featNames

  # Cycle through all trees and splits in each tree and tally the count for
  # the respective feature split count
  for (i in 1:nrow(data)) {
    tree_vars <- object@R_forest[[i]]$var_id
    for (j in 1:length(tree_vars)) {
      if (tree_vars[j] > 0) {
        data[i,tree_vars[j]] = data[i,tree_vars[j]]+1
      }
    }
  }

  # Aggregate by all trees inn forest
  count_totals <- apply(data, 2, sum)

  # Return normalized counts (i.e. the split proportions)
  return(count_totals / sum(count_totals))

}


# -- Calculate OOB Predictions -------------------------------------------------
#' getOOBpreds-forestry
#' @name getOOBpreds-forestry
#' @rdname getOOBpreds-forestry
#' @description Calculate the out-of-bag predictions of a given forest.
#' @param object A trained model object of class "forestry".
#' @param newdata A possible new data frame on which to run out of bag
#'  predictions. If this is not NULL, we assume that the indices of
#'  newdata are the same as the indices of the training set, and will use
#'  these to find which trees the observation is considered in/out of bag for.
#' @param doubleOOB A flag specifying whether or not we should use the double OOB
#'  set for the OOB predictions. This is the set of observations for each tree which
#'  were in neither the averaging set nor the splitting set. Note that the forest
#'  must have been trained with doubleBootstrap = TRUE for this to be used. Default
#'  is FALSE.
#' @param noWarning Flag to not display warnings.
#' @return The vector of all training observations, with their out of bag
#'  predictions. Note each observation is out of bag for different trees, and so
#'  the predictions will be more or less stable based on the observation. Some
#'  observations may not be out of bag for any trees, and here the predictions
#'  are returned as NA.
#' @seealso \code{\link{forestry}}
#' @export
getOOBpreds <- function(object,
                        newdata = NULL,
                        doubleOOB = FALSE,
                        noWarning = FALSE
                        ) {

  if (!object@replace &&
      object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                      object@sampsize) < 10) {
    if (!noWarning) {
      warning(paste(
        "Samples are drawn without replacement and sample size",
        "is too big!"
      ))
    }
    return(NA)
  }


  if (!is.null(newdata) && (object@sampsize != nrow(newdata))) {
    warning(paste(
      "Attempting to do OOB predictions on a dataset which doesn't match the",
      "training data!"
    ))
    return(NA)
  }

  if (!noWarning) {
    warning(paste("OOB predictions have been moved to the predict() function.
                  Run OOB predictions by calling predict(..., aggregation = \"oob\")"))
  }

  if (doubleOOB && !object@doubleBootstrap) {
    doubleOOB <- FALSE
    warning(paste("Cannot do doubleOOB preds if the forest was trained with doubleBootstrap = FALSE",
                  "Setting doubleOOB = FALSE"))
  }

  if (!is.null(newdata)) {
    newdata <- testing_data_checker(object, newdata, object@hasNas)
    newdata <- as.data.frame(newdata)

    processed_x <- preprocess_testing(newdata,
                                      object@categoricalFeatureCols,
                                      object@categoricalFeatureMapping)

    existing_df = TRUE
  } else {
    existing_df = FALSE
    processed_x = NULL
  }

  rcppOOBpreds <- tryCatch({
    return(rcpp_OBBPredictionsInterface(object@forest,
                                        processed_x,
                                        existing_df,
                                        doubleOOB))
  }, error = function(err) {
    print(err)
    return(NA)
  })
  rcppOOBpreds[is.nan(rcppOOBpreds)] <- NA
  return(rcppOOBpreds)
}

# -- Calculate Variable Importance ---------------------------------------------
#' getVI-forestry
#' @rdname getVI-forestry
#' @description Calculate the percentage increase in OOB error of the forest
#'  when each feature is shuffled.
#' @param object A `forestry` object.
#' @param noWarning flag to not display warnings
#' @note No seed is passed to this function so it is
#'   not possible in the current implementation to replicate the vector
#'   permutations used when measuring feature importance.
#' @return The variable importance of the forest.
#' @export
getVI <- function(object,
                           noWarning) {
  forest_checker(object)
    # Keep warning for small sample size
    if (!object@replace &&
        object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                        object@sampsize) < 10) {
      if (!noWarning) {
        warning(paste(
          "Samples are drawn without replacement and sample size",
          "is too big!"
        ))
      }
      return(NA)
    }

    rcppVI <- tryCatch({
      return(rcpp_VariableImportanceInterface(object@forest))
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppVI)
  }

# -- Calculate Confidence Interval estimates for a new feature -----------------
#' getCI-forestry
#' @rdname getCI-forestry
#' @description For a new set of features, calculate the confidence intervals
#'  for each new observation.
#' @param object A `forestry` object.
#' @param newdata A set of new observations for which we want to predict the
#'  outcomes and use confidence intervals.
#' @param level The confidence level at which we want to make our intervals. Default
#'  is to use .95 which corresponds to 95 percentile confidence intervals.
#' @param B Number of bootstrap draws to use when using method = "OOB-bootstrap"
#' @param method A flag for the different ways to create the confidence intervals.
#'  Right now we have two ways of doing this. One is the `OOB-bootstrap` flag which
#'  uses many bootstrap pulls from the set of OOB trees then with these different
#'  pulls, we use the set of trees to predict for the new feature and give the
#'  confidence set over the many bootstrap draws. The other method- `OOB-conformal`-
#'  creates intervals by taking the set of doubleOOB trees for each observation, and
#'  using the predictions of these trees to give conformal intervals. So for an
#'  observation obs_i, let S_i be the set of trees for which obs_i was in neither
#'  the splitting set nor the averaging set (or the set of trees for which obs_i
#'  was "doubleOOB"), we then predict for obs_i with only the trees in S_i.
#'  doubleOOB_tree_preds <- predict(S_i, obs_i):
#'  Then CI(obs_i, alpha = .95) = quantile(doubleOOB_tree_preds - y_i, probs = .95).
#'  The `local-conformal` option takes the residuals of each training point (using)
#'  OOB predictions, and then uses the weights of the random forest to determine
#'  the quantiles of the residuals in the local neighborhood of the predicted point.
#'  Default is `OOB-conformal`.
#' @param noWarning flag to not display warnings
#' @return The confidence intervals for each observation in newdata.
#' @export
getCI <- function(object,
                  newdata,
                  level = .95,
                  B = 100,
                  method = "OOB-conformal",
                  noWarning = FALSE) {

  if ((method == "OOB-conformal") && !(object@OOBhonest && object@doubleBootstrap)) {
    stop("We cannot do OOB-conformal intervals unless both OOBhonest and doubleBootstrap are TRUE")
  }

  if ((method == "OOB-bootstrap") && !(object@OOBhonest)) {
    stop("We cannot do OOB-bootstrap intervals unless OOBhonest is TRUE")
  }

  if ((method == "local-conformal") && !(object@OOBhonest)) {
    stop("We cannot do local-conformal intervals unless OOBhonest is TRUE")
  }

  # Check the Rforestry object
  forest_checker(object)

  # Check the newdata
  newdata <- testing_data_checker(object, newdata, object@hasNas)
  newdata <- as.data.frame(newdata)
  processed_x <- preprocess_testing(newdata,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  if (method == "OOB-bootstrap") {
    #warning("OOB-bootstrap not implemented")

    # Now we do B bootstrap pulls of the trees in order to do prediction
    # intervals for newdata
    prediction_array <- data.frame(matrix(nrow = nrow(newdata), ncol = B))

    for (i in 1:B) {
      bootstrap_i <- sample(x = (1:object@ntree),
                            size = object@ntree,
                            replace = TRUE)
      pred_i <- predict(object, newdata = newdata, trees = bootstrap_i)
      prediction_array[,i] <- pred_i
    }
    quantiles <- apply(prediction_array,
                       MARGIN = 1,
                       FUN = quantile,
                       probs = c((1-level) / 2, 1 - (1-level) / 2))

    predictions <- list("Predictions" = predict(object, newdata = newdata),
                        "CI.upper" = quantiles[2,],
                        "CI.lower" = quantiles[1,],
                        "Level" = level)

    return(predictions)
  } else if (method == "OOB-conformal") {
    # Get double OOB predictions and the residuals
    y_pred <- predict(object, aggregation = "doubleOOB")
    res <- y_pred - object@processed_dta$y

    # Get (1-level) / 2 and 1 - (1-level) / 2 quantiles of the residuals
    quantiles <- quantile(res, probs = c((1-level) / 2, 1 - (1-level) / 2))

    # Get predictions on newdata
    predictions_new <- predict(object, newdata)

    predictions <- list("Predictions" = predictions_new,
                        "CI.upper" = predictions_new + unname(quantiles[2]),
                        "CI.lower" = predictions_new + unname(quantiles[1]),
                        "Level" = level)
    return(predictions)
  } else if (method == "local-conformal") {
    OOB_preds <- predict(object, aggregation = "oob")
    OOB_res <- object@processed_dta$y - OOB_preds

    preds <- predict(object, newdata = newdata, aggregation = "weightMatrix")
    weights <- preds$weightMatrix

    CI_local <- data.frame(lower = NA, upper = NA)

    for (i in 1:nrow(newdata)) {
      cur_weights = weights[i,]
      data.frame(OOB_res, cur_weights) %>%
        dplyr::filter(cur_weights > 0) %>%
        dplyr::arrange(OOB_res) %>%
        dplyr::mutate(cur_weights = cumsum(cur_weights)) %>%
        dplyr::filter(cur_weights >= .025, cur_weights <= .975) %>%
        dplyr::summarise(
          lower = dplyr::first(OOB_res),
          upper = dplyr::last(OOB_res)
        ) -> cur_bound

      CI_local <- rbind(CI_local, cur_bound)
    }

    CI_local <- CI_local[-1,]

    # Get predictions on newdata
    predictions_new <- predict(object, newdata)

    predictions <- list("Predictions" = predictions_new,
                        "CI.lower" = predictions_new + CI_local$lower,
                        "CI.upper" = predictions_new + CI_local$upper,
                        "Level" = level)
    return(predictions)
  } else {
    return(NA)
  }
}



# -- Add More Trees ------------------------------------------------------------
#' addTrees-forestry
#' @rdname addTrees-forestry
#' @description Add more trees to the existing forest.
#' @param object A `forestry` object.
#' @param ntree Number of new trees to add
#' @return A `forestry` object
#' @export
addTrees <- function(object,
                     ntree) {
    forest_checker(object)
    if (ntree <= 0 || ntree %% 1 != 0) {
      stop("ntree must be a positive integer.")
    }

    tryCatch({
      rcpp_AddTreeInterface(object@forest, ntree)
      object@ntree = object@ntree + ntree
      return(object)
    }, error = function(err) {
      print(err)
      return(NA)
    })

  }



# -- Auto-Tune -----------------------------------------------------------------
#' autoforestry-forestry
#' @rdname autoforestry-forestry
#' @inheritParams forestry
#' @param sampsize The size of total samples to draw for the training data.
#' @param num_iter Maximum iterations/epochs per configuration. Default is 1024.
#' @param eta Downsampling rate. Default value is 2.
#' @param verbose if tuning process in verbose mode
#' @param seed random seed
#' @param nthread Number of threads to train and predict the forest. The default
#'   number is 0 which represents using all cores.
#' @return A `forestry` object
#' @import stats
#' @export
autoforestry <- function(x,
                         y,
                         sampsize = as.integer(nrow(x) * 0.75),
                         num_iter = 1024,
                         eta = 2,
                         verbose = FALSE,
                         seed = 24750371,
                         nthread = 0) {
  if (verbose) {
    print("Start auto-tuning.")
  }

  # Creat a dummy tree just to reuse its data.
  dummy_tree <-
    forestry(
      x,
      y,
      ntree = 1,
      nodesizeSpl = nrow(x),
      nodesizeAvg = nrow(x)
    )

  # Number of unique executions of Successive Halving (minus one)
  s_max <- as.integer(log(num_iter) / log(eta))

  # Total number of iterations (without reuse) per execution of
  # successive halving (n,r)
  B <- (s_max + 1) * num_iter

  if (verbose) {
    print(
      paste(
        "Hyperband will run successive halving in",
        s_max,
        "times, with",
        B,
        "iterations per execution."
      )
    )
  }

  # Begin finite horizon hyperband outlerloop
  models <- vector("list", s_max + 1)
  models_OOB <- vector("list", s_max + 1)

  set.seed(seed)

  for (s in s_max:0) {
    if (verbose) {
      print(paste("Hyperband successive halving round", s_max + 1 - s))
    }

    # Initial number of configurations
    n <- as.integer(ceiling(B / num_iter / (s + 1) * eta ^ s))

    # Initial number of iterations to run configurations for
    r <- num_iter * eta ^ (-s)

    if (verbose) {
      print(paste(">>> Total number of configurations:", n))
      print(paste(
        ">>> Number of iterations per configuration:",
        as.integer(r)
      ))
    }

    # Begin finite horizon successive halving with (n,r)
    # Generate parameters:
    allConfigs <- data.frame(
      mtry = sample(1:ncol(x), n, replace = TRUE),
      min_node_size_spl = NA, #sample(1:min(30, nrow(x)), n, replace = TRUE),
      min_node_size_ave = NA, #sample(1:min(30, nrow(x)), n, replace = TRUE),
      splitratio = runif(n, min = 0.1, max = 1),
      replace = sample(c(TRUE, FALSE), n, replace = TRUE),
      middleSplit = sample(c(TRUE, FALSE), n, replace = TRUE)
    )

    min_node_size_spl_raw <- floor(allConfigs$splitratio * sampsize *
                                     rbeta(n, 1, 3))
    allConfigs$min_node_size_spl <- ifelse(min_node_size_spl_raw == 0, 1,
                                           min_node_size_spl_raw)
    min_node_size_ave <- floor((1 - allConfigs$splitratio) * sampsize *
                                 rbeta(n, 1, 3))
    allConfigs$min_node_size_ave <- ifelse(min_node_size_ave == 0, 1,
                                           min_node_size_ave)

    if (verbose) {
      print(paste(">>>", n, " configurations have been generated."))
    }

    val_models <- vector("list", nrow(allConfigs))
    r_old <- 1
    for (j in 1:nrow(allConfigs)) {
      tryCatch({
        val_models[[j]] <- forestry(
          x = x,
          y = y,
          ntree = r_old,
          mtry = allConfigs$mtry[j],
          nodesizeSpl = allConfigs$min_node_size_spl[j],
          nodesizeAvg = allConfigs$min_node_size_ave[j],
          splitratio = allConfigs$splitratio[j],
          replace = allConfigs$replace[j],
          sampsize = sampsize,
          nthread = nthread,
          middleSplit = allConfigs$middleSplit[j],
          reuseforestry = dummy_tree
        )
      }, error = function(err) {
        val_models[[j]] <- NULL
      })
    }

    if (s != 0) {
      for (i in 0:(s - 1)) {
        # Run each of the n_i configs for r_i iterations and keep best
        # n_i/eta
        n_i <- as.integer(n * eta ^ (-i))
        r_i <- as.integer(r * eta ^ i)
        r_new <- r_i - r_old

        # if (verbose) {
        #   print(paste("Iterations", i))
        #   print(paste("Total number of configurations:", n_i))
        #   print(paste("Number of iterations per configuration:", r_i))
        # }

        val_losses <- vector("list", nrow(allConfigs))

        # Iterate to evaluate each parameter combination and cut the
        # parameter pools in half every iteration based on its score
        for (j in 1:nrow(allConfigs)) {
          if (r_new > 0 && !is.null(val_models[[j]])) {
            val_models[[j]] <- addTrees(val_models[[j]], r_new)
          }
          if (!is.null(val_models[[j]])) {
            val_losses[[j]] <- getOOB(val_models[[j]], noWarning = TRUE)
            if (is.na(val_losses[[j]])) {
              val_losses[[j]] <- Inf
            }
          } else {
            val_losses[[j]] <- Inf
          }
        }

        r_old <- r_i

        val_losses_idx <-
          sort(unlist(val_losses), index.return = TRUE)
        val_top_idx <- val_losses_idx$ix[0:as.integer(n_i / eta)]
        allConfigs <- allConfigs[val_top_idx,]
        val_models <- val_models[val_top_idx]
        gc()
        rownames(allConfigs) <- 1:nrow(allConfigs)

        # if (verbose) {
        #   print(paste(length(val_losses_idx$ix) - nrow(allConfigs),
        #               "configurations have been eliminated."))
        # }
      }
    }
    # End finite horizon successive halving with (n,r)
    if (!is.null(val_models[[1]])) {
      best_OOB <- getOOB(val_models[[1]], noWarning = TRUE)
      if (is.na(best_OOB)) {
        stop()
        best_OOB <- Inf
      }
    } else {
      stop()
      best_OOB <- Inf
    }
    if (verbose) {
      print(paste(">>> Successive halving ends and the best model is saved."))
      print(paste(">>> OOB:", best_OOB))
    }

    if (!is.null(val_models[[1]]))
      models[[s + 1]] <- val_models[[1]]
    models_OOB[[s + 1]] <- best_OOB

  }

  # End finite horizon hyperband outlerloop and sort by performance
  model_losses_idx <- sort(unlist(models_OOB), index.return = TRUE)

  if (verbose) {
    print(
      paste(
        "Best model is selected from best-performed model in",
        s_max,
        "successive halving, with OOB",
        models_OOB[model_losses_idx$ix[1]]
      )
    )
  }

  return(models[[model_losses_idx$ix[1]]])
}

# -- Save RF -----------------------------------------------------
#' save RF
#' @rdname saveForestry-forestry
#' @description This wrapper function checks the forestry object, makes it
#'  saveable if needed, and then saves it.
#' @param object an object of class `forestry`
#' @param filename a filename in which to store the `forestry` object
#' @param ... additional arguments useful for specifying compression type and level
#' @return Saves the forest into filename.
#' @export
saveForestry <- function(object, filename, ...){
  # First we need to make sure the object is saveable
  object <- make_savable(object)
  base::save(object, file = filename, ...)
}

# -- Load RF -----------------------------------------------------
#' load RF
#' @rdname loadForestry-forestry
#' @description This wrapper function checks the forestry object, makes it
#'  saveable if needed, and then saves it.
#' @param filename a filename in which to store the `forestry` object
#' @return The loaded forest from filename.
#' @export
loadForestry <- function(filename){
  # First we need to make sure the object is saveable
  name <- base::load(file = filename, envir = environment())
  rf <- get(name)
  rf <- relinkCPP_prt(rf)
  return(rf)
}

# -- Translate C++ to R --------------------------------------------------------
#' @title Cpp to R translator
#' @description Add more trees to the existing forest.
#' @param object external CPP pointer that should be translated from Cpp to an R
#'   object
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @export
CppToR_translator <- function(object) {
    tryCatch({
      return(rcpp_CppToR_translator(object))
    }, error = function(err) {
      print(err)
      return(NA)
    })
  }


# -- relink forest CPP ptr -----------------------------------------------------
#' relink CPP ptr
#' @rdname relinkCPP
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @param object an object of class `forestry` or class `multilayerForestry`
#' @return Relinks the pointer to the correct C++ object.
#' @export
relinkCPP_prt <- function(object) {
    # 1.) reconnect the data.frame to a cpp data.frame
    # 2.) reconnect the forest.


    tryCatch({
      # Now we have to decide whether we use reconstruct tree or reconstructforests
      if (class(object)[1] == "forestry") {
        if(!length(object@R_forest))
          stop("Forest was saved without first calling `forest <- make_savable(forest)`. ",
               "This forest cannot be reconstructed.")

        # In this case we use the tree constructor
        forest_and_df_ptr <- rcpp_reconstructree(
          x = object@processed_dta$processed_x,
          y = object@processed_dta$y,
          catCols = object@processed_dta$categoricalFeatureCols_cpp,
          linCols = object@processed_dta$linearFeatureCols_cpp,
          numRows = object@processed_dta$nObservations,
          numColumns = object@processed_dta$numColumns,
          R_forest = object@R_forest,
          replace = object@replace,
          sampsize = object@sampsize,
          splitratio = object@splitratio,
          OOBhonest = object@OOBhonest,
          doubleBootstrap = object@doubleBootstrap,
          mtry = object@mtry,
          nodesizeSpl = object@nodesizeSpl,
          nodesizeAvg = object@nodesizeAvg,
          nodesizeStrictSpl = object@nodesizeStrictSpl,
          nodesizeStrictAvg = object@nodesizeStrictAvg,
          minSplitGain = object@minSplitGain,
          maxDepth = object@maxDepth,
          interactionDepth = object@interactionDepth,
          seed = sample(.Machine$integer.max, 1),
          nthread = 0, # will use all threads available.
          verbose = FALSE,
          middleSplit = object@middleSplit,
          hasNas = object@hasNas,
          maxObs = object@maxObs,
          featureWeights = object@featureWeights,
          featureWeightsVariables = object@featureWeightsVariables,
          deepFeatureWeights = object@deepFeatureWeights,
          deepFeatureWeightsVariables = object@deepFeatureWeightsVariables,
          observationWeights = object@observationWeights,
          monotonicConstraints = object@monotonicConstraints,
          groupMemberships = as.integer(object@groups),
          monotoneAvg = object@monotoneAvg,
          linear = object@linear,
          overfitPenalty = object@overfitPenalty,
          doubleTree = object@doubleTree)

        object@forest <- forest_and_df_ptr$forest_ptr
        object@dataframe <- forest_and_df_ptr$data_frame_ptr
      } else if (class(object)[1] == "multilayerForestry") {
        #print("Got past filtering")

        if(!length(object@R_forests))
          stop("Forest was saved without first calling `forest <- make_savable(forest)`. ",
               "This forest cannot be reconstructed.")
        # Now we use the forest constructor
        forest_and_df_ptr <- rcpp_reconstruct_forests(
          x = object@processed_dta$processed_x,
          y = object@processed_dta$y,
          catCols = object@processed_dta$categoricalFeatureCols_cpp,
          linCols = object@processed_dta$linearFeatureCols_cpp,
          numRows = object@processed_dta$nObservations,
          numColumns = object@processed_dta$numColumns,
          R_forests = object@R_forests,   # Now we pass R_forests
          R_residuals = object@R_residuals,
          nrounds = object@nrounds,
          eta = object@eta,
          replace = object@replace,
          sampsize = object@sampsize,
          splitratio = object@splitratio,
          OOBhonest = object@OOBhonest,
          doubleBootstrap = object@doubleBootstrap,
          mtry = object@mtry,
          nodesizeSpl = object@nodesizeSpl,
          nodesizeAvg = object@nodesizeAvg,
          nodesizeStrictSpl = object@nodesizeStrictSpl,
          nodesizeStrictAvg = object@nodesizeStrictAvg,
          minSplitGain = object@minSplitGain,
          maxDepth = object@maxDepth,
          seed = sample(.Machine$integer.max, 1),  # We might want to save this later
          nthread = 0, # will use all threads available.
          verbose = FALSE,
          middleSplit = object@middleSplit,
          maxObs = object@maxObs,
          featureWeights = object@featureWeights,
          featureWeightsVariables = object@featureWeightsVariables,
          deepFeatureWeights = object@deepFeatureWeights,
          deepFeatureWeightsVariables = object@deepFeatureWeightsVariables,
          observationWeights = object@observationWeights,
          monotonicConstraints = object@monotonicConstraints,
          groupMemberships = as.integer(object@groups),
          monotoneAvg = object@monotoneAvg,
          gammas = object@gammas,
          linear = object@linear,
          overfitPenalty = object@overfitPenalty,
          doubleTree = object@doubleTree)

        object@forest <- forest_and_df_ptr$forest_ptr
        object@dataframe <- forest_and_df_ptr$data_frame_ptr
      } else {
        stop("We can only load an object of class forestry or multilayerForestry")
      }


    }, error = function(err) {
      print('Problem when trying to create the forest object in Cpp')
      print(err)
      return(NA)
    })

    return(object)
}




# -- make savable --------------------------------------------------------------
#' make_savable
#' @name make_savable
#' @rdname make_savable
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @param object an object of class `forestry`
#' @note  `make_savable` does not translate all of the private member variables
#'   of the C++ forestry object so when the forest is reconstructed with
#'   `relinkCPP_prt` some attributes are lost. For example, `nthreads` will be
#'   reset to zero. This makes it impossible to disable threading when
#'   predicting for forests loaded from disk.
#' @examples
#' set.seed(323652639)
#' x <- iris[, -1]
#' y <- iris[, 1]
#' forest <- forestry(x, y, ntree = 3, nthread = 2)
#' y_pred_before <- predict(forest, x)
#'
#' forest <- make_savable(forest)
#'
#' wd <- tempdir()
#
#' saveForestry(forest, filename = file.path(wd, "forest.Rda"))
#' rm(forest)
#'
#' forest <- loadForestry(file.path(wd, "forest.Rda"))
#'
#' y_pred_after <- predict(forest, x)
#'
#' file.remove(file.path(wd, "forest.Rda"))
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @aliases make_savable,forestry-method
#' @export
make_savable <- function(object) {
    # We check if it is either a forestry object, or a multilayer forestry object
    # and save it accordingly
    if (class(object)[1] == "forestry") {
      object@R_forest <- CppToR_translator(object@forest)
    } else if (class(object)[1] == "multilayerForestry") {
      object@R_forests <- rcpp_multilayer_CppToR_translator(object@forest)
      object@gammas <- rcpp_gammas_translator(object@forest)
      object@R_residuals <- rcpp_residuals_translator(object@forest)
    }

    return(object)
}

# Add .onAttach file to give citation information
.onAttach <- function( ... )
{
  Lib <- dirname(system.file(package = "Rforestry"))
  version <- utils::packageDescription("Rforestry", lib.loc = Lib)$Version
  BuildDate <- utils::packageDescription("Rforestry", lib.loc = Lib)$Built

  message <- paste("## \n##  Rforestry (Version ", version, ", Build Date: ", BuildDate, ")\n",
                   "##  See https://github.com/forestry-labs for additional documentation.\n",
                   "##  Please cite software as:\n",
                   "##    Soren R. Kunzel, Theo F. Saarinen, Edward W. Liu, Jasjeet S. Sekhon. 2019.\n",
                   "##    \''Linear Aggregation in Tree-based Estimators.\'' arXiv preprint \n",
                   "##    arXiv:1906.06463. https://arxiv.org/abs/1906.06463 \n",
                   "##",
                   sep = "")
  packageStartupMessage(message)
}

