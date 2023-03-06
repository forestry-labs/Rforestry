test_that("Tests the weightMatrix flag for various aggregation types", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('weightMatrix + averaging aggregation predictions')
  # Set seed for reproducibility
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    OOBhonest = TRUE
  )

  # Just get preds
  preds_normal <- predict(forest, newdata = x)
  expect_equal(length(preds_normal),150)


  # Get preds + weightMatrix
  preds_weight <- predict(forest, newdata = x, weightMatrix = TRUE)

  # Make sure we are getting the same predictions
  expect_equal( all.equal(
    (preds_weight$weightMatrix %*% as.matrix(y))[,1],
    preds_weight$predictions,
    tolerance = 1e-3
  ) , TRUE)


  # Make sure we get right entries
  expect_equal(names(preds_weight), c("predictions","weightMatrix"))

  # Get OOB predictions
  preds_oob <- predict(forest, newdata = x, aggregation = "oob")
  expect_equal(length(preds_oob),150)

  # Get OOB predictions + weightMatrix
  preds_oobw <- predict(forest, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(names(preds_oobw), c("predictions","weightMatrix","treeCounts"))

  # Make sure we are getting the same predictions
  expect_equal( all.equal(
    (preds_oobw$weightMatrix %*% as.matrix(y))[,1],
    preds_oobw$predictions,
    tolerance = 1e-3
  ) , TRUE)

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

})
