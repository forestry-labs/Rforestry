test_that("Tests that multilayerForestry is working correctly", {
  #skip("Multilayer forestry has become non-deterministic.")
  x <- iris[, -2]
  y <- iris[, 2]
  context('MultilayerForestry base function')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- multilayerForestry(
    x,
    y,
    ntree = 500,
    nrounds = 2,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    maxDepth = 4,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    seed = 2
  )
  # Test predict
  y_pred <- predict(forest, x, seed = 2)

  # Mean Square Error
  sum((y_pred - y) ^ 2)

  skip_if_not_mac()

  # Multilayer forestry is non deterministic, this needs to be fixed, but for
  # now test that it at least runs without crashing
  expect_equal(sum((y_pred - y) ^ 2), 13.849777575910220, tolerance = 1e-8)
})
