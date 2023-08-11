#' @useDynLib Rforestry, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats predict
#' @importFrom pROC roc
#' @importFrom utils tail
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
                                  doubleBootstrap,
                                  nthread,
                                  middleSplit,
                                  doubleTree,
                                  linFeats,
                                  monotonicConstraints,
                                  groups,
                                  featureWeights,
                                  deepFeatureWeights,
                                  observationWeights,
                                  customSplitSample,
                                  customAvgSample,
                                  customExcludeSample,
                                  linear,
                                  scale,
                                  hasNas,
                                  naDirection
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

  if (any(sapply(x, function(x) all(is.na(x))))) {
    stop("training data column cannot be all NAs")
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

  if (length(customSplitSample) != 0 || length(customAvgSample) != 0) {
    message("When customSplitSample is set, other sampling parameters are ignored")

    if (splitratio != 1 || OOBhonest) {
      warning("When customSplitSample is set, other honesty implementations are ignored")
    }

    # Check that we provide splitting samples as well as averaging samples
    if (length(customSplitSample) != ntree || length(customAvgSample) != ntree) {
      stop("Custom splitting and averaging samples must be provided for every tree")
    }

    # Check that averaging sample and splitting samples are disjoint
    for (i in 1:ntree) {
      if (any(customAvgSample[[i]] %in% customSplitSample[[i]])) {
        stop("Splitting and averaging samples must be disjoint")
      }

      # Check that provided samples are integers in the correct range
      if (length(customSplitSample[[i]]) == 0 ||
          any(customSplitSample[[i]] <= 0) ||
          any(customSplitSample[[i]] %% 1 != 0) ||
          any(customSplitSample[[i]] > nrow(x))) {
        stop(
          "customSplitSample must contain positive integers up to the number of observations in x"
        )
      }
      if (length(customAvgSample[[i]]) == 0 ||
          any(customAvgSample[[i]] <= 0) ||
          any(customAvgSample[[i]] %% 1 != 0) ||
          any(customAvgSample[[i]] > nrow(x))) {
        stop(
          "customAvgSample must contain positive integers up to the number of observations in x"
        )
      }
    }
    # Check excluded sample is disjoint from both splitting and averaging set
    if (length(customExcludeSample) != 0) {
      if (length(customExcludeSample) != ntree) {
        stop("customExcludeSample must be equal in length to ntree")
      }

      for (i in 1:ntree) {
        if (any(customExcludeSample[[i]] %in% customAvgSample[[i]])) {
          stop("Excluded samples must be disjoint from averaging samples")
        }
        # Check that included samples are integers in the correct range
        if (any(customExcludeSample[[i]] <= 0) || any(customExcludeSample[[i]] %% 1 != 0) ||
            any(customExcludeSample[[i]] > nrow(x))) {
          stop("customExcludeSample must contain positive integers up to the number of observations in x")
        }
      }
    }

    # Set OOB honest flag to be TRUE so we have proper prediction handling
    OOBhonest=TRUE
    doubleBootstrap = TRUE

    # Now since we will pass to C++ the indices need to be 0-indexed, so convert
    # from R 1-indexed indices to 0 indexed indices
    for (i in 1:ntree) {
      customAvgSample[[i]] = customAvgSample[[i]]-1
      customSplitSample[[i]] = customSplitSample[[i]]-1
    }
    if (length(customExcludeSample) != 0) {
      for (i in 1:length(customExcludeSample)) {
        customExcludeSample[[i]] = customExcludeSample[[i]]-1
      }
    }
  }

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
    if (length(levels(groups)) == 1) {
      stop("groups must have more than 1 level to be left out from sampling")
    }
    if (length(groups) != nrow(x)) {
      stop("Length of groups must equal the number of observations")
    }

    # Check that the custom samples come from disjoint groups
    if (length(customSplitSample) != 0) {
      # Check splitting and averaging have disjoint groups
      for (i in 1:ntree) {
        if (any(groups[customAvgSample[[i]]+1] %in% groups[customSplitSample[[i]]+1])) {
          stop("Splitting and averaging samples must contain disjoint groups")
        }
      }
      # Check customExcludeSample has disjoint groups
      if (length(customExcludeSample) != 0) {
        for (i in 1:ntree) {
          if (any(groups[customExcludeSample[[i]] + 1]
                  %in% groups[customAvgSample[[i]] + 1])) {
            stop("Excluded samples must contain groups disjoint from those in the averaging samples")
          }
        }
      }
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
              "doubleBootstrap" = doubleBootstrap,
              "nthread" = nthread,
              "groups" = groups,
              "middleSplit" = middleSplit,
              "doubleTree" = doubleTree,
              "linFeats" = linFeats,
              "monotonicConstraints" = monotonicConstraints,
              "featureWeights" = featureWeights,
              "scale" = scale,
              "deepFeatureWeights" = deepFeatureWeights,
              "observationWeights" = observationWeights,
              "customSplitSample" = customSplitSample,
              "customAvgSample" = customAvgSample,
              "customExcludeSample" = customExcludeSample,
              "hasNas" = hasNas,
              "naDirection" = naDirection
        ))
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
    naDirection = "logical",
    linear = "logical",
    linFeats = "numeric",
    monotonicConstraints = "numeric",
    monotoneAvg = "logical",
    featureWeights = "numeric",
    featureWeightsVariables = "numeric",
    deepFeatureWeights = "numeric",
    deepFeatureWeightsVariables = "numeric",
    observationWeights = "numeric",
    customSplitSample = "list",
    customAvgSample = "list",
    customExcludeSample = "list",
    overfitPenalty = "numeric",
    doubleTree = "logical",
    groupsMapping = "list",
    groups = "numeric",
    scale = "logical",
    colMeans = "numeric",
    colSd = "numeric",
    minTreesPerFold = "numeric",
    foldSize = "numeric"
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
#' @param customSplitSample List of vectors for user-defined splitting observations per tree. The vector at
#'   index i contains the indices of the sampled splitting observations, with replacement allowed, for tree i.
#'   This feature overrides other sampling parameters and must be set in conjunction with customAvgSample.
#' @param customAvgSample List of vectors for user-defined averaging observations per tree. The vector at
#'   index i contains the indices of the sampled splitting observations, with replacement allowed, for tree i.
#'   This feature overrides other sampling parameters and must be set in conjunction with customSplitSample.
#' @param customExcludeSample An optional list of vectors for user-defined excluded observations per tree. The vector at
#'   index i contains the indices of the excluded observations for tree i. An observation is considered excluded if it does
#'   not appear in the splitting or averaging set and has been explicitly withheld from being sampled for a tree.
#'   Excluded observations are not considered out-of-bag, so when we call predict with aggregation = "oob",
#'   when we predict for an observation, we will only use the predictions of trees in which the
#'   observation was in the customSplitSample (and neither in the customAvgSample nor the customExcludeSample).
#'   This parameter is optional even when customSplitSample and customAvgSample are set.
#'   It is also optional at the tree level, so can have fewer than ntree entries. When given fewer than
#'   ntree entries, for example K, the entries will be applied to the first K trees in the forest and
#'   the remaining trees will have no excludedSamples.
#' @param splitratio Proportion of the training data used as the splitting dataset.
#'   It is a ratio between 0 and 1. If the ratio is 1 (the default), then the splitting
#'   set uses the entire data, as does the averaging set---i.e., the standard Breiman RF setup.
#'   If the ratio is 0, then the splitting data set is empty, and the entire dataset is used
#'   for the averaging set (This is not a good usage, however, since there will be no data available for splitting).
#' @param OOBhonest In this version of honesty, the out-of-bag observations for each tree
#'   are used as the honest (averaging) set. This setting also changes how predictions
#'   are constructed. When predicting for observations that are out-of-sample
#'   (using predict(..., aggregation = "average")), all the trees in the forest
#'   are used to construct predictions. When predicting for an observation that was in-sample (using
#'   predict(..., aggregation = "oob")), only the trees for which that observation
#'   was not in the averaging set are used to construct the prediction for that observation.
#'   aggregation="oob" (out-of-bag) ensures that the outcome value for an observation
#'   is never used to construct predictions for a given observation even when it is in sample.
#'   This property does not hold in standard honesty, which relies on an asymptotic
#'   subsampling argument. By default, when OOBhonest = TRUE, the out-of-bag observations
#'   for each tree are resamples with replacement to be used for the honest (averaging)
#'   set. This results in a third set of observations that are left out of both
#'   the splitting and averaging set, we call these the double out-of-bag (doubleOOB)
#'   observations. In order to get the predictions of only the trees in which each
#'   observation fell into this doubleOOB set, one can run predict(... , aggregation = "doubleOOB").
#'   In order to not do this second bootstrap sample, the doubleBootstrap flag can
#'   be set to FALSE.
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
#' @param naDirection Sets a default direction for missing values in each split
#'   node during training. It test placing all missing values to the left and
#'   right, then selects the direction that minimizes loss. If no missing values
#'   exist, then a default direction is randomly selected in proportion to the
#'   distribution of observations on the left and right. (Default = FALSE)
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
#' @param minTreesPerFold The number of trees which we make sure have been created leaving
#'   out each fold (each fold is a set of randomly selected groups).
#'    This is 0 by default, so we will not give any special treatment to
#'   the groups when sampling observations, however if this is set to a positive integer, we
#'   modify the bootstrap sampling scheme to ensure that exactly that many trees
#'   have each group left out. We do this by, for each fold, creating minTreesPerFold
#'   trees which are built on observations sampled from the set of training observations
#'   which are not in a group in the current fold. The folds form a random partition of
#'   all of the possible groups, each of size foldSize. This means we create at
#'   least # folds * minTreesPerFold trees for the forest.
#'   If ntree > # folds * minTreesPerFold, we create
#'   max(# folds * minTreesPerFold, ntree) total trees, in which at least minTreesPerFold
#'   are created leaving out each fold.
#' @param foldSize The number of groups that are selected randomly for each fold to be
#'   left out when using minTreesPerFold. When minTreesPerFold is set and foldSize is
#'   set, all possible groups will be partitioned into folds, each containing foldSize unique groups
#'   (if foldSize doesn't evenly divide the number of groups, a single fold will be smaller,
#'   as it will contain the remaining groups). Then minTreesPerFold are grown with each
#'   entire fold of groups left out.
#' @param monotoneAvg This is a boolean flag that indicates whether or not monotonic
#'   constraints should be enforced on the averaging set in addition to the splitting set.
#'   This flag is meaningless unless both honesty and monotonic constraints are in use.
#'   The default is FALSE.
#' @param overfitPenalty Value to determine how much to penalize the magnitude
#'   of coefficients in ridge regression when using linear splits.
#' @param scale A parameter which indicates whether or not we want to scale and center
#'   the covariates and outcome before doing the regression. This can help with
#'   stability, so by default is TRUE.
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
                     customSplitSample = NULL,
                     customAvgSample = NULL,
                     customExcludeSample = NULL,
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
                     minTreesPerFold = 0,
                     foldSize = 1,
                     monotoneAvg = FALSE,
                     overfitPenalty = 1,
                     scale = TRUE,
                     doubleTree = FALSE,
                     naDirection = FALSE,
                     reuseforestry = NULL,
                     savable = TRUE,
                     saveable = TRUE
) {
  # Make sure that all the parameters exist when passed to forestry
  tryCatch({
    check_args <- c(as.list(environment()))
    rm(check_args)
  },  error = function(err) {
    err <- as.character(err)
    err <- gsub("Error in as.list.environment(environment()): ","", err, fixed = TRUE)
    stop(paste0("A parameter passed is not assigned: ", err))
  })

  # Should not scale if we have only one value of Y
  if (sd(y) == 0) {
    scale = FALSE
  }

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

  # Create vectors with the column means and SD's for scaling
  colMeans <- rep(0, ncol(x)+1)
  colSd <- rep(0, ncol(x)+1)

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
  if(is.null(customSplitSample)) {
    customSplitSample <- list()
  }
  if(is.null(customAvgSample)) {
    customAvgSample <- list()
  }
  if(is.null(customExcludeSample)) {
    customExcludeSample <- list()
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
      doubleBootstrap = doubleBootstrap,
      nthread = nthread,
      middleSplit = middleSplit,
      doubleTree = doubleTree,
      linFeats = linFeats,
      monotonicConstraints = monotonicConstraints,
      groups = groups,
      featureWeights = featureWeights,
      deepFeatureWeights = deepFeatureWeights,
      observationWeights = observationWeights,
      customSplitSample = customSplitSample,
      customAvgSample = customAvgSample,
      customExcludeSample = customExcludeSample,
      linear = linear,
      scale = scale,
      hasNas = hasNas,
      naDirection = naDirection)

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
    if ((foldSize %% 1 != 0) || (foldSize < 1) || (foldSize > length(levels(groups)))) {
      stop("foldSize must be an integer between 1 and the # of groups")
    }

    groupVector <- as.integer(groups)

    # Print warning if the group number and minTreesPerFold results in a large
    # forest
    if (minTreesPerFold>0 && (length(levels(groups)) / foldSize)*minTreesPerFold > 2000) {
      print(paste0("Using ",(length(levels(groups)) / foldSize)," folds with ",
                     minTreesPerFold," trees per group will train ",
                   ceiling(length(levels(groups)) / foldSize)*minTreesPerFold," trees in the forest."))
    }
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

    if (scale) {
      # Get colMeans and ColSd's
      for (col_idx in 1:ncol(processed_x)) {
        if ((col_idx-1) %in% categoricalFeatureCols_cpp) {
          next
        } else {
          colMeans[col_idx] <- mean(processed_x[,col_idx], na.rm = TRUE)
          colSd[col_idx] <- sd(processed_x[,col_idx], na.rm = TRUE)
        }
      }

      # Scale columns of X
      processed_x <- scale_center(processed_x,
                                  (categoricalFeatureCols_cpp+1),
                                  colMeans,
                                  colSd)

      # Center and scale Y
      colMeans[ncol(processed_x)+1] <- mean(y, na.rm = TRUE)
      colSd[ncol(processed_x)+1] <- sd(y, na.rm = TRUE)
      y <- (y-colMeans[ncol(processed_x)+1]) / colSd[ncol(processed_x)+1]
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
        customSplitSample = customSplitSample,
        customAvgSample = customAvgSample,
        customExcludeSample = customExcludeSample,
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
        customSplitSample,
        customAvgSample,
        customExcludeSample,
        monotonicConstraints,
        groupVector,
        minTreesPerFold,
        foldSize,
        monotoneAvg,
        hasNas,
        naDirection,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )

      # We don't want to save the scaled df, so unceneter and unscale for the R
      # object
      if (scale) {
        processed_x <- unscale_uncenter(processed_x,
                                        categoricalFeatureCols,
                                        colMeans,
                                        colSd)
        y <-  y * colSd[ncol(processed_x)+1] + colMeans[ncol(processed_x)+1]
      }

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
          ntree = ifelse(minTreesPerFold == 0,
                         ntree * (doubleTree + 1), max(ntree * (doubleTree + 1),
                         ceiling(length(levels(groups)) / foldSize)*minTreesPerFold)),
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
          customSplitSample = customSplitSample,
          customAvgSample = customAvgSample,
          customExcludeSample = customExcludeSample,
          hasNas = hasNas,
          naDirection = naDirection,
          linear = linear,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = groupsMapping,
          groups = groupVector,
          colMeans = colMeans,
          colSd = colSd,
          scale = scale,
          minTreesPerFold = minTreesPerFold,
          foldSize = foldSize
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

    if (scale) {
      # Center and scale continous features and outcome
      for (col_idx in  1:ncol(processed_x)) {
        if ((col_idx-1) %in% categoricalFeatureCols_cpp) {
          next
        } else {
          colMeans[col_idx] <- mean(processed_x[,col_idx], na.rm = TRUE)
          colSd[col_idx] <- sd(processed_x[,col_idx], na.rm = TRUE)
        }
      }

      # Scale columns of X
      processed_x <- scale_center(processed_x,
                                  (categoricalFeatureCols_cpp+1),
                                  colMeans,
                                  colSd)

      # Center and scale Y
      colMeans[ncol(processed_x)+1] <- mean(y, na.rm = TRUE)
      colSd[ncol(processed_x)+1] <- sd(y, na.rm = TRUE)
      y <- (y-colMeans[ncol(processed_x)+1]) / colSd[ncol(processed_x)+1]
    }

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
        customSplitSample,
        customAvgSample,
        customExcludeSample,
        monotonicConstraints,
        groupVector,
        minTreesPerFold,
        foldSize,
        monotoneAvg,
        hasNas,
        naDirection,
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
          ntree = ifelse(minTreesPerFold == 0,
                         ntree * (doubleTree + 1), max(ntree * (doubleTree + 1),
                         length(levels(groups))*minTreesPerFold)),
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
          customSplitSample = customSplitSample,
          customAvgSample = customAvgSample,
          customExcludeSample = customExcludeSample,
          hasNas = hasNas,
          naDirection = naDirection,
          linear = linear,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          monotoneAvg = monotoneAvg,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree,
          groupsMapping = groupsMapping,
          groups = groupVector,
          colMeans = colMeans,
          colSd = colSd,
          scale = scale,
          minTreesPerFold = minTreesPerFold,
          foldSize = foldSize
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

# -- Export to Treelite JSON Function --------------------------------------------
#' export_json
#' @name export_json
#' @rdname export_json
#' @description Returns a JSON string representation of the forest. The JSON format aligns with the spec defined for import by the Treelite library.
#' @param object A `forestry` object.
#' @return A string containing Treelite JSON of the forest
#' @export
export_json <- function(object) {
  tryCatch({
    return(rcpp_ExportJson(object))
  }, error = function(err) {
    print(err)
    return(NULL)
  })
}


# -- Predict Method ------------------------------------------------------------
#' predict-forestry
#' @name predict-forestry
#' @rdname predict-forestry
#' @description Return the prediction from the forest.
#' @param object A `forestry` object.
#' @param newdata A data frame of testing predictors.
#' @param aggregation How the individual tree predictions are aggregated:
#'   `average` returns the mean of all trees in the forest; `terminalNodes` also returns
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
#' @param holdOutIdx This is an optional argument, containing a vector of indices
#'   from the training data set that should be not be allowed to influence the
#'   predictions of the forest. When a vector of indices of training observations are
#'   given, the predictions will be made only with trees in the forest that
#'   do not contain any of these indices in either the splitting or averaging sets.
#'   This cannot be used at the same time as any other aggregation options.
#'   If `weightMatrix = TRUE`, this will return the
#'   weightMatrix corresponding to the predictions made with trees respecting
#'   holdOutIdx. If there are no trees that have held out all of the indices
#'   in holdOutIdx, then the predictions will return NaN.
#' @param trainingIdx This is an optional parameter to give the indices of the observations
#'   in `newdata` from the training data set. This is used when we want to run predict on
#'   only a subset of observations from the training data set and use `aggregation = "oob"` or
#'   `aggregation = "doubleOOB"`. For example, at the tree level, a tree make out of
#'   bag (`aggregation = "oob"`) predictions for the indices in the set
#'   setdiff(trainingIdx,tree$averagingIndices) and will make double out-of-bag
#'   predictions for the indices in the set
#'   setdiff(trainingIdx,union(tree$averagingIndices,tree$splittingIndices).
#'   Note, this parameter must be set when predict is called with an out-of-bag
#'   aggregation option on a data set not matching the original training data size.
#'   The order of indices in `trainingIdx` also needs to match the order of observations
#'   in newdata. So for an arbitrary index set `trainingIdx` and dataframe `newdata`,
#'    of the same size as the training set, the predictions from `predict(rf, newdata[trainingIdx,],`
#'   `aggregation = "oob", trainingIdx = trainingIdx)` should match the
#'   predictions of to `predict(rf, newdata, aggregation = "oob")[trainingIdx]`.
#'   This option also works with the `weightMatrix` option and will return the
#'   (smaller) weightMatrix for the observations in the passed data frame.
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
#' @param weightMatrix An indicator of whether or not we should also return a
#'   matrix of the weights given to each training observation when making each
#'   prediction. When getting the weight matrix, aggregation must be one of
#'   `average`, `oob`, and `doubleOOB`.
#' @param hierShrinkageLambda The shrinkage parameter to use for hierarchical shrinkage.
#'   By default, this is set to 0 (equal to zero shrinkage). 
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @export
predict.forestry <- function(object,
                             newdata = NULL,
                             aggregation = "average",
                             holdOutIdx = NULL,
                             trainingIdx = NULL,
                             seed = as.integer(runif(1) * 10000),
                             nthread = 0,
                             exact = NULL,
                             trees = NULL,
                             weightMatrix = FALSE,
                             hierShrinkageLambda = NULL,
                             ...) {

  if (is.null(newdata) && !(aggregation == "oob" || aggregation == "doubleOOB")) {
    stop("When using an aggregation that is not oob or doubleOOB, one must supply newdata")
  }

  if ((!(object@linear)) && (aggregation == "coefs")) {
    stop("Aggregation can only be linear with setting the parameter linear = TRUE.")
  }

  if (!is.null(holdOutIdx) && !is.null(trees)) {
    stop("Only one of holdOutIdx and trees must be set at one time")
  }

  if (!is.null(holdOutIdx) && (aggregation != "average")) {
    stop("holdOutIdx can only be used when aggregation is average")
  }

  if (aggregation %in% c("oob", "doubleOOB") && (!is.null(newdata)) && is.null(trainingIdx) && (nrow(newdata) != (object@processed_dta$nObservations))) {
    stop(paste0("trainingIdx must be set when doing out of bag predictions with a data set ",
                "not equal in size to the training data set"))
  }

  # Check that holdOutIdx entries are valid
  if (!is.null(holdOutIdx)) {
    # Check that indices are integers within the range of the training set indices
    if (any(holdOutIdx %% 1 != 0) ||
        (max(holdOutIdx) > nrow(object@processed_dta$processed_x)) ||
        (min(holdOutIdx) < 1) ) {
      stop("holdOutIdx must contain only integers in the range of the training set indices")
    }
  }

  # Check that trainingIdx entries are valid
  if (!is.null(trainingIdx)) {
    if (nrow(newdata) != length(trainingIdx)) {
      stop(paste0("The length of trainingIdx must be the same as the number of ",
                  "observations in the training data"))
    }

    # Check that indices are integers within the range of the training set indices
    if (any(trainingIdx %% 1 != 0) ||
        (max(trainingIdx) > nrow(object@processed_dta$processed_x)) ||
        (min(trainingIdx) < 1) ) {
      stop("trainingIdx must contain only integers in the range of the training set indices")
    }

    # Check that correct aggregation is used
    if (!(aggregation %in% c("oob", "doubleOOB"))) {
      warning(paste0("trainingIdx are only used when aggregation is oob or doubleOOB.",
                     " The current aggregation doesn't match either so trainingIdx will be ignored"))
      trainingIdx <- NULL
    }
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

    if (object@scale) {
      # Cycle through all continuous features and center / scale
      processed_x <- scale_center(processed_x,
                                  (unname(object@processed_dta$categoricalFeatureCols_cpp)+1),
                                  object@colMeans,
                                  object@colSd)
    }
  }

  # Set exact aggregation method if nobs < 100,000 and average aggregation
  if (is.null(exact)) {
    if (!is.null(newdata) && nrow(newdata) > 1e5) {
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

  # hierarchical shrinkage is not set so we set our two internal parameters for Hs
  if (!is.null(hierShrinkageLambda)){
    # hierarchical shrinkage factor must be positive
    if (hierShrinkageLambda < 0){
      stop("The value of the hierarchical shrinkage parameter must be positive")
    }
    hierShrinkage = TRUE
    lambdaShrinkage = hierShrinkageLambda
  } else {
    hierShrinkage=FALSE
    lambdaShrinkage=0
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

  if (!is.null(trainingIdx)) {
    useTrainingIndices <- TRUE
    trainingIndices <- trainingIdx-1
  } else {
    useTrainingIndices <- FALSE
    trainingIndices <- c(-1)
  }


  # If option set to terminalNodes, we need to make matrix of ID's
  if (!is.null(holdOutIdx)) {
    if (is.null(newdata)) {
      if (object@scale) {
        processed_x <- scale_center(object@processed_dta$processed_x,
                                    (unname(object@processed_dta$categoricalFeatureCols_cpp)+1),
                                    object@colMeans,
                                    object@colSd)
      } else {
        processed_x <- object@processed_dta$processed_x
      }
    }

    rcppPrediction <- tryCatch({
      rcpp_cppPredictInterface(object@forest,
                               processed_x,
                               aggregation,
                               seed = seed,
                               nthread = nthread,
                               exact = exact,
                               returnWeightMatrix = weightMatrix,
                               use_weights = use_weights,
                               use_hold_out_idx = TRUE,
                               tree_weights = tree_weights,
                               hold_out_idx = (holdOutIdx-1), # Change to 0 indexed for C++
                               hierShrinkage = hierShrinkage,
                               lambdaShrinkage = lambdaShrinkage)
    }, error = function(err) {
      print(err)
      return(NULL)
    })

  } else if (aggregation == "oob") {

    if (is.null(newdata)) {
      if (object@scale) {
        processed_x <- scale_center(object@processed_dta$processed_x,
                                    (unname(object@processed_dta$categoricalFeatureCols_cpp)+1),
                                    object@colMeans,
                                    object@colSd)
      } else {
        processed_x <- object@processed_dta$processed_x
      }
    }

    rcppPrediction <- tryCatch({
      rcpp_OBBPredictionsInterface(object@forest,
                                   processed_x,  # If we don't provide a dataframe, provide the forest DF
                                   TRUE, # Tell predict we don't have an existing dataframe
                                   FALSE,
                                   weightMatrix,
                                   exact,
                                   useTrainingIndices,
                                   trainingIndices,
                                   hierShrinkage,
                                   lambdaShrinkage
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })

  } else if (aggregation == "doubleOOB") {

    if (!object@doubleBootstrap) {
      stop(paste(
        "Attempting to do double OOB predictions with a forest that was not trained
        with doubleBootstrap = TRUE"
      ))
      return(NA)
    }

    if (is.null(newdata)) {

      if (object@scale) {
        processed_x <- scale_center(object@processed_dta$processed_x,
                                    (unname(object@processed_dta$categoricalFeatureCols_cpp)+1),
                                    object@colMeans,
                                    object@colSd)
      } else {
        processed_x <- object@processed_dta$processed_x
      }
    }

    rcppPrediction <- tryCatch({
      rcpp_OBBPredictionsInterface(object@forest,
                                   processed_x,  # Give null for the dataframe
                                   TRUE, # Tell predict we don't have an existing dataframe
                                   TRUE,
                                   weightMatrix,
                                   exact,
                                   useTrainingIndices,
                                   trainingIndices,
                                   hierShrinkage,
                                   lambdaShrinkage
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })


  } else {
    rcppPrediction <- tryCatch({
      rcpp_cppPredictInterface(object@forest,
                               processed_x,
                               aggregation,
                               seed = seed,
                               nthread = nthread,
                               exact = exact,
                               returnWeightMatrix = weightMatrix,
                               use_weights = use_weights,
                               use_hold_out_idx = FALSE,
                               tree_weights = tree_weights,
                               hold_out_idx = c(-1),
                               hierShrinkage = hierShrinkage,
                               lambdaShrinkage = lambdaShrinkage)
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

  # If we have scaled the observations, we want to rescale the predictions
  if (object@scale) {
    rcppPrediction$predictions <- rcppPrediction$predictions*object@colSd[length(object@colSd)] +
      object@colMeans[length(object@colMeans)]
  }

  # If we have a weightMatrix for the training Idx set, pass that back only
  #if (!is.null(trainingIdx)) {
  #  rcppPrediction$weightMatrix <- rcppPrediction$weightMatrix[trainingIdx,]
  #}

  if (aggregation == "average" && weightMatrix) {
    return(rcppPrediction[c(1,2)])
  } else if (aggregation == "oob" && weightMatrix) {
    return(rcppPrediction)
  } else if (aggregation == "doubleOOB" && weightMatrix) {
    return(rcppPrediction)
  } else if (aggregation == "average") {
    return(rcppPrediction$predictions)
  } else if (aggregation == "oob") {
    return(rcppPrediction$predictions)
  } else if (aggregation == "doubleOOB") {
    return(rcppPrediction$predictions)
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
        sparse_rep_single_tree[j,terminalNodes[j,i]+1] <- 1
      }
      sparse_rep <- cbind(sparse_rep, sparse_rep_single_tree)
    }
    rcppPrediction[["sparse"]] <- sparse_rep
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
#' @param hierShrinkageLambda The shrinkage parameter to use for hierarchical shrinkage.
#'  By default, this is set to 0 (equal to zero shrinkage).
#' @aliases getOOB,forestry-method
#' @return The OOB error of the forest.
#' @export
getOOB <- function(object,
                   noWarning,
                   hierShrinkageLambda = NULL) {
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
      preds <- predict(object, aggregation = "oob", hierShrinkageLambda = hierShrinkageLambda)
      # Only calc mse on non missing predictions
      y_true <- object@processed_dta$y[which(!is.nan(preds))]


      mse <- mean((preds[which(!is.nan(preds))] -
                     y_true)^2)
      return(mse)
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppOOB)
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
#' @param hierShrinkageLambda The shrinkage parameter to use for hierarchical shrinkage.
#'  By default, this is set to 0 (equal to zero shrinkage).
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
                        noWarning = FALSE,
                        hierShrinkageLambda = NULL
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


  } else {
    # Else we take the data the forest was trained with
    processed_x <- object@processed_dta$processed_x
  }
  
  # hierarchical shrinkage is not set so we set our two internal parameters for Hs
  if (!is.null(hierShrinkageLambda)){
    # hierarchical shrinkage factor must be positive
    if (hierShrinkageLambda < 0){
      stop("The value of the hierarchical shrinkage parameter must be positive")
    }
    hierShrinkage = TRUE
    lambdaShrinkage = hierShrinkageLambda
  } else {
    hierShrinkage = FALSE
    lambdaShrinkage = 0
  }

  if (object@scale) {
    # Cycle through all continuous features and center / scale
    processed_x <- scale_center(processed_x,
                                (unname(object@processed_dta$categoricalFeatureCols_cpp)+1),
                                object@colMeans,
                                object@colSd)
  }

  rcppOOBpreds <- tryCatch({
    rcppPrediction <- rcpp_OBBPredictionsInterface(object@forest,
                                                   processed_x,
                                                   TRUE,
                                                   doubleOOB,
                                                   FALSE,
                                                   TRUE,
                                                   FALSE,
                                                   c(-1),
                                                   hierShrinkage,
                                                   lambdaShrinkage)

    # If we have scaled the observations, we want to rescale the predictions
    if (object@scale) {
      rcppPrediction$predictions <- rcppPrediction$predictions*object@colSd[length(object@colSd)] +
        object@colMeans[length(object@colMeans)]
    }

    return(rcppPrediction$predictions)
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
#' @param metric A parameter to determine how the predictions of the forest with
#'   a permuted variable are compared to the predictions of the standard forest.
#'   Must be one of c("mse","auc","tnr"), "mse" gives the percentage increase in
#'   mse when the feature is permuted, "auc" gives the percentage decrease in AUC
#'   when the feature is permuted, and "tnr" gives the percentage decrease in
#'   TNR when the TPR is 99\% when the feature is permuted.
#' @param seed A parameter to seed the random number generator for shuffling
#'   the features of X.
#' @note Pass a seed to this function so it is
#'   possible to replicate the vector permutations used when measuring feature importance.
#' @return The variable importance of the forest.
#' @export
getVI <- function(object,
                  noWarning,
                  metric = "mse",
                  seed = 1) {
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

    # Set seed for permutations
    set.seed(seed)

    if (!(metric %in% c("mse","auc","tnr"))) {
      stop("metric must be one of mse, auc, or tnr")
    }

    if (metric %in% c("auc","tnr")) {
      if (length(unique(object@processed_dta$y)) != 2) {
        stop("forest must be trained for binary classification if the metric is set to be auc or tnr")
      }
    }


    # In order to call predict, need categorical variables changed back to strings
    x = object@processed_dta$processed_x
    dummyIndex <- 1
    for (categoricalFeatureCol in unlist(object@categoricalFeatureCols)) {
      # levels
      map = data.frame(strings = object@categoricalFeatureMapping[[dummyIndex]]$uniqueFeatureValues)
      x[, categoricalFeatureCol] <- sapply(x[, categoricalFeatureCol], function(i){return(map$strings[i])})
      dummyIndex <- dummyIndex + 1
    }

    vi <- rep(NA, object@processed_dta$numColumns)


    if (metric == "mse") {
      oob_mse <- mean((predict(object, aggregation = "oob") - object@processed_dta$y)**2)
      for (feat_i in 1:object@processed_dta$numColumns) {
        x_mod = x
        shuffled = x[sample(1:object@processed_dta$nObservations),feat_i]
        x_mod[,feat_i] = shuffled
        mse_i = mean((predict(object, newdata = x_mod, aggregation = "oob", seed = seed)
                      - object@processed_dta$y)**2)
        vi[feat_i] = sqrt(mse_i)/sqrt(oob_mse) - 1
      }
    } else if (metric %in% c("auc","tnr")) {
      evalAUC <- function(truth, pred){
        roc_model <- roc(response = truth, predictor = as.numeric(pred),quiet=TRUE)
        idx <- tail(which(roc_model$sensitivities >= 0.99), 1)
        tnr_model <- roc_model$specificities[idx]
        return(round(c(roc_model$auc, tnr_model), 7))
      }

      # Pull oob predictions and then cycle through and get the increase in different metrics
      oob_predictions <- predict(object, aggregation = "oob")
      metrics_oob <- evalAUC(truth = object@processed_dta$y, pred = oob_predictions)
      for (feat_i in 1:object@processed_dta$numColumns) {
        x_mod = x
        shuffled = x[sample(1:object@processed_dta$nObservations),feat_i]
        x_mod[,feat_i] = shuffled
        predict_i = predict(object, newdata = x_mod, aggregation = "oob", seed = seed)

        metrics_i = evalAUC(truth = object@processed_dta$y, pred = predict_i)

        if (metric == "auc") {
          vi[feat_i] = metrics_oob[1]/metrics_i[1] - 1
        } else if (metric == "tnr") {
          vi[feat_i] = metrics_oob[2]/metrics_i[2] - 1
        }
      }
    }
    return(vi)
}


# -- Get the observations used for prediction ----------------------------------
#' predictInfo-forestry
#' @rdname predictInfo-forestry
#' @description Get the observations which are used to predict for a set of new
#'  observations using either all trees (for out of sample observations), or
#'  tree for which the observation is out of averaging set or out of sample entirely.
#' @param object A `forestry` object.
#' @param newdata Data on which we want to do predictions. Must be the same length
#'  as the training set if we are doing `oob` or `doubleOOB` aggregation.
#' @param aggregation Specifies which aggregation version is used to predict for the
#' observation, must be one of `average`,`oob`, and `doubleOOB`.
#' @return A list with four entries. `weightMatrix` is a matrix specifying the
#'  weight given to training observation i when prediction on observation j.
#'  `avgIndices` gives the indices which are in the averaging set for each new
#'  observation. `avgWeights` gives the weights corresponding to each averaging
#'  observation returned in `avgIndices`. `obsInfo` gives the full observation vectors
#'  which were used to predict for an observation, as well as the weight given
#'  each observation.
#' @export
predictInfo <- function(object,
                        newdata,
                        aggregation = "average")
{

  if (!(aggregation %in% c("oob", "doubleOOB","average"))) {
    stop("We can only use aggregation as oob or doubleOOB")
  }
  p <- predict(object, newdata = newdata,
               exact=TRUE,aggregation = aggregation,weightMatrix = TRUE)

  # Get the averaging indices used for each observation
  acive_indices <- apply(p$weightMatrix, MARGIN = 1, function(x){return(which(x != 0))})
  # Get the relative weight given to each averaging observation outcome
  weights <- apply(p$weightMatrix, MARGIN = 1, function(x){return(x[which(x != 0)])})

  # Get the observations which correspond to the averaging indices used to predict each outcome
  # Want observations by descending weight
  observations <- apply(p$weightMatrix, MARGIN = 1, function(y){df <- data.frame(newdata[which(y != 0),],
                                                                                 "Weight" = y[which(y != 0)]);
                                                                colnames(df) <- c(colnames(newdata), "Weight");
                                                                      return(df)})

  obs_sorted <- lapply(observations, function(x){return(x[order(x$Weight,decreasing=TRUE),])})



  return(list("weightMatrix" = p$weightMatrix,
              "avgIndices" = acive_indices,
              "avgWeights" = weights,
              "obsInfo" = observations))
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
#' @param object an object of class `forestry`
#' @return Relinks the pointer to the correct C++ object.
#' @export
relinkCPP_prt <- function(object) {
    # 1.) reconnect the data.frame to a cpp data.frame
    # 2.) reconnect the forest.


  tryCatch({
    # Now we have to decide whether we use reconstruct tree or reconstructforests
    if (!length(object@R_forest))
      stop(
        "Forest was saved without first calling `forest <- make_savable(forest)`. ",
        "This forest cannot be reconstructed."
      )

    # If tree has scaling, we need to scale + center the X and Y before
    # giving to C++
    if (object@scale) {
      processed_x <- scale_center(object@processed_dta$processed_x,
                                  (
                                    unname(object@processed_dta$categoricalFeatureCols_cpp) + 1
                                  ),
                                  object@colMeans,
                                  object@colSd)

      processed_y <-
        (object@processed_dta$y - object@colMeans[ncol(processed_x) + 1]) /
        object@colSd[ncol(processed_x) + 1]
    } else {
      processed_x <- object@processed_dta$processed_x
      processed_y <- object@processed_dta$y
    }


    # In this case we use the tree constructor
    forest_and_df_ptr <- rcpp_reconstructree(
      x = processed_x,
      y = processed_y,
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
      nthread = 0,
      # will use all threads available.
      verbose = FALSE,
      middleSplit = object@middleSplit,
      hasNas = object@hasNas,
      naDirection = object@naDirection,
      maxObs = object@maxObs,
      minTreesPerFold = object@minTreesPerFold,
      featureWeights = object@featureWeights,
      featureWeightsVariables = object@featureWeightsVariables,
      deepFeatureWeights = object@deepFeatureWeights,
      deepFeatureWeightsVariables = object@deepFeatureWeightsVariables,
      observationWeights = object@observationWeights,
      customSplitSample = object@customSplitSample,
      customAvgSample = object@customAvgSample,
      customExcludeSample = object@customExcludeSample,
      monotonicConstraints = object@monotonicConstraints,
      groupMemberships = as.integer(object@groups),
      monotoneAvg = object@monotoneAvg,
      linear = object@linear,
      overfitPenalty = object@overfitPenalty,
      doubleTree = object@doubleTree
    )
    object@forest <- forest_and_df_ptr$forest_ptr
    object@dataframe <- forest_and_df_ptr$data_frame_ptr


  },
  error = function(err) {
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
    object@R_forest <- CppToR_translator(object@forest)
    return(object)
}

# Add .onAttach file to give citation information
.onAttach <- function( ... )
{
  Lib <- dirname(system.file(package = "Rforestry"))
  version <- utils::packageDescription("Rforestry")$Version
  BuildDate <- utils::packageDescription("Rforestry")$Built

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

