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

  expect_equal(all.equal(getOOBpreds(forest)[1:10], c(5.09195014288, 4.66466649643,
                                                      4.65042604918, 4.87281687100,
                                                      5.08349279822, 5.34483093904,
                                                      5.06971226922, 5.06069487707,
                                                      4.76761805874, 4.79213639568)), TRUE)
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
