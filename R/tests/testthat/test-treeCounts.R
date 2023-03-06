test_that("Tests pulling the number of trees used for OOB/ doubleOOB predictions", {
  x <- iris[, -1]
  y <- iris[, 1]

  # Given the random forest model, rf,
  # the data you want to predict on, newdata,
  # a flag to specify if one should use doubleOOB or oob predictions
  # (double = FALSE runs oob predictions, double = TRUE runs doubleOOB predictions)
  # and a flag to specify whether one should return just the predictions
  # or the number of trees used to make each prediction as well
  doob_by_hand <- function(rf,
                           newdata,
                           double = TRUE,
                           tree_counts = FALSE) {
    rf <- make_savable(rf)
    preds <- rep(0,nrow(newdata))
    counts <- rep(0,nrow(newdata))
    for (obsIdx in 1:nrow(newdata)) {
      treeIdx <- c()
      for (idx in 1:rf@ntree) {
        splIdx <- sort(unique(rf@R_forest[[idx]]$splittingSampleIndex))
        avgIdx <- sort(unique(rf@R_forest[[idx]]$averagingSampleIndex))

        if (double) {
          treeObs <- sort(unique(c(splIdx,avgIdx)))
        } else {
          treeObs <- sort(unique(c(avgIdx)))
        }

        if (!(obsIdx %in% treeObs)) {
          treeIdx <- c(treeIdx, idx)
        }
      }
      preds[obsIdx] <- predict(rf, newdata = newdata[obsIdx,], trees = treeIdx)
      counts[obsIdx] <- length(treeIdx)
    }
    if (tree_counts) {
      return(list("preds" = preds, "counts" = counts))
    } else {
      return(preds)
    }
  }

  # Set seed for reproducibility
  set.seed(24750371)

  forest <- forestry(
    x,
    y,
    ntree = 100,
    replace = TRUE,
    sampsize = nrow(x),
    maxDepth = 2,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    OOBhonest = TRUE
  )

  context("Check tree counts for oob predictions")
  # Get OOB predictions + weightMatrix
  preds_oobw <- predict(forest, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(names(preds_oobw), c("predictions","weightMatrix","treeCounts"))

  # Make sure we are getting the same predictions
  expect_equal( all.equal(
    (preds_oobw$weightMatrix %*% as.matrix(y))[,1],
    preds_oobw$predictions,
    tolerance = 1e-3
  ) , TRUE)

  oob_counts_cpp <- preds_oobw$treeCounts

  forest <- make_savable(forest)
  oob_preds_hand <- doob_by_hand(forest, x, double = FALSE, tree_counts = TRUE)

  expect_equal(all.equal(oob_preds_hand$counts, oob_counts_cpp), TRUE)

  context("Check tree counts for doubleOOB predictions")
  # Just get preds
  preds_double <- predict(forest, newdata = x, aggregation = "doubleOOB")
  expect_equal(length(preds_double),150)

  # Get preds + weightMatrix
  preds_doublew <- predict(forest, newdata = x, weightMatrix = TRUE, aggregation = "doubleOOB")
  expect_equal(names(preds_doublew), c("predictions","weightMatrix","treeCounts"))

  # Make sure we are getting the same predictions
  expect_equal( all.equal(
    (preds_doublew$weightMatrix %*% as.matrix(y))[,1],
    preds_doublew$predictions,
    tolerance = 1e-3
  ) , TRUE)

  doob_preds_hand <- doob_by_hand(forest, x, double = TRUE, tree_counts = TRUE)
  doob_counts_cpp <- preds_doublew$treeCounts

  expect_equal(all.equal(doob_preds_hand$counts, doob_counts_cpp), TRUE)

  context("High precision test for tree counts")
  skip_if_not_mac()
  expect_equal(all.equal(doob_counts_cpp[1:10], c(16, 12, 10, 17, 14, 14,  8, 14, 16, 10)), TRUE)
  skip_if_not_mac()
  expect_equal(all.equal(oob_counts_cpp[1:10], c(78, 78, 72, 80, 74, 81, 73, 81, 80, 81)), TRUE)
  skip_if_not_mac()
  # Check the group sizes are fairly close to what we expect

  # Expect double OOB to be 1/e^2 and oob to be 1-1/e + 1/e^2
  expect_gt(mean(oob_counts_cpp) / 100, .66)

  expect_gt(mean(doob_counts_cpp) / 100, .1)

})
