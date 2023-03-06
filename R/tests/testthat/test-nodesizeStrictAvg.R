test_that("Test if nodesizeStrictSpl is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('nodesizeStrictSpl when using honesty')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 1,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    OOBhonest = TRUE,
    seed = 8921,
    nodesizeStrictAvg = 5
  )

  forest <- make_savable(forest)
  # Test OOB
  p <- predict(forest, newdata = x[1,], weightMatrix = TRUE)

  # We expect at least 5 observations were used for the prediction as we have set
  # nodesizeStrictAvg = 5
  skip_if_not_mac()
  expect_gt(length(which(p$weightMatrix != 0 )), 4)

  skip_if_not_mac()
  expect_equal(length(which(p$weightMatrix != 0 )), 7)


  context("Test a greater number of required averaging observations")
  forest <- forestry(
    x,
    y,
    ntree = 1,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    OOBhonest = TRUE,
    seed = 8921,
    nodesizeStrictAvg = 10
  )

  forest <- make_savable(forest)
  # Test OOB
  p <- predict(forest, newdata = x[1,], weightMatrix = TRUE)

  # We expect at least 10 observations were used for the prediction as we have set
  # nodesizeStrictAvg = 10
  expect_gt(length(which(p$weightMatrix != 0 )), 10)

  skip_if_not_mac()
  expect_equal(length(which(p$weightMatrix != 0 )), 15)

  context("Test the same without OOB honesty")
  forest <- forestry(
    x,
    y,
    ntree = 1,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    OOBhonest = FALSE,
    splitratio = .5,
    seed = 8921,
    nodesizeStrictAvg = 10
  )

  forest <- make_savable(forest)
  # Test OOB
  p <- predict(forest, newdata = x[1,], weightMatrix = TRUE)

  # We expect at least 10 observations were used for the prediction as we have set
  # nodesizeStrictAvg = 10
  expect_gt(length(which(p$weightMatrix != 0 )), 8)

  skip_if_not_mac()
  expect_equal(length(which(p$weightMatrix != 0 )), 9)

})
