test_that("Tests if OOB predictions are working correctly (normal setting)", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Predictions')
  # Set seed for reproductivity
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
    nodesizeStrictAvg = 5
  )

  # Test OOB predictions
  expect_equal(sum((getOOBpreds(forest) -  iris[,1])^2), getOOB(forest), tolerance = 1e-5)

  skip_if_not_mac()

  expect_equal(all.equal(getOOBpreds(forest)[1:10], c(5.092647817, 4.664031165,
                                                      4.650426049, 4.870883947,
                                                      5.084049999, 5.344246144,
                                                      5.069991851, 5.060238528,
                                                      4.766551234, 4.790776227)), TRUE)
})


test_that("Tests if OOB predictions are working correctly (extreme setting)", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Predictions')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test a very extreme setting
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = FALSE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  expect_warning(
    testOOBpreds <- getOOBpreds(forest, FALSE),
    "Samples are drawn without replacement and sample size is too big!"
  )

  expect_equal(testOOBpreds, NA, tolerance = 1e-4)
})
