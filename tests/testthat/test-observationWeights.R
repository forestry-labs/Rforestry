test_that("Tests that observationWeights for the bootstrap is working correctly", {
  #skip("Multilayer forestry has become non-deterministic.")
  x <- iris[, -2]
  y <- iris[, 2]
  context('Test observationWeights')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    observationWeights = c(rep(1, 50), rep(2, 50), rep(3,50)),
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

  # Check the predictions from a weighted forest
  expect_equal(sum((y_pred - y) ^ 2), 8.658285584297157, tolerance = 1e-8)

  forest <- make_savable(forest)

  # Check we have gotten the same bootstrap pulls
  expect_equal(all.equal(forest@R_forest[[1]]$splittingSampleIndex[1:50],
                         c( 43, 149, 133,  91,  82,  32, 119,  63, 141,  29, 121,
                            142,  69, 104,  20, 101, 105, 143,  24, 143,   7,  59,
                            120, 118,  66,  44, 135, 146,  63, 143, 140,  64, 110,
                            119,144,  95,  80,  79, 133,  13,  29,  93, 120,  68,
                            122,  58, 105,  47, 135,   6)), TRUE)
})
