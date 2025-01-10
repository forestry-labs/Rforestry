test_that("Tests that random forest is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Forestry base function')
  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 1,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    seed = 2
  )
  plot(forest)
  # Test predict
  y_pred <- predict(forest, x, seed = 2)

  # Test feature.new validation
  expect_error(predict(forest, x[, -1], ))
  expect_warning(y_pred_shuffled <- predict(forest, x[, ncol(x):1], seed = 2))
  expect_equal(y_pred, y_pred_shuffled, tolerance = 1e-12)

  # Mean Square Error
  skip_if_not_mac()

  mean((y_pred - y) ^ 2)
  expect_equal(mean((y_pred - y) ^ 2), 0.064760523023031, tolerance = 1e-12)

  # Test factors with missing obs and unused levels are correctly handled
  x$Species[1:70] <- NA
  forest <- forestry(
    x,
    y, seed = 2)
  y_pred <- predict(forest, x, seed = 2)
  expect_equal(mean((y_pred - y) ^ 2), 0.10277113497796, tolerance = 1e-12)
})
