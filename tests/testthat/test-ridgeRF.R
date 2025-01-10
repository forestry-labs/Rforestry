test_that("Tests if ridgeRF works", {
  context('Tests RidgeRF')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  x
  y
  iris


  set.seed(275)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 200,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    linear = FALSE,
    overfitPenalty = 50
  )

  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  mean((y_pred - y) ^ 2)

  expect_equal(mean((y_pred - y) ^ 2), 0.0199184561243013, tolerance = 1e-3)

  for (seed in 270:275) {
    set.seed(seed)

    # Test forestry (mimic RF)
    forest <- forestry(
      x,
      y,
      ntree = 1,
      replace = TRUE,
      sample.fraction = .8,
      mtry = 3,
      nodesizeStrictSpl = 5,
      nthread = 2,
      splitrule = "variance",
      splitratio = 1,
      nodesizeStrictAvg = 5,
      linear = TRUE,
      overfitPenalty = 1000
    )
  }
  set.seed(231428176)
  forest <- forestry(
    x,
    y,
    ntree = 200,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    linear = TRUE,
    overfitPenalty = 1000
  )
  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  mean((y_pred - y) ^ 2)

  expect_equal(mean((y_pred - y) ^ 2), 0.0200273762841208, tolerance = 1e-3)
})
