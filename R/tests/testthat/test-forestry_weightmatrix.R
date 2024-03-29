test_that("Tests that random forest is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Weight matrix')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    scale=TRUE
  )

  # Test predict
  full_predictions <- predict(forest, x[1:10,], weightMatrix = TRUE)
  y_pred <- predict(forest, x[1:10,])
  expect_equal(full_predictions$predictions, y_pred, tolerance = 1e-12)

  expect_equal(full_predictions$weightMatrix %*% as.matrix(y),
               as.matrix(y_pred), tolerance = 1e-12)

  full_predictions <- predict(forest, x, weightMatrix = TRUE)
  y_pred <- predict(forest, x)
  expect_equal(full_predictions$predictions, y_pred, tolerance = 1e-12)

  expect_equal(full_predictions$weightMatrix %*% as.matrix(y),
               as.matrix(y_pred), tolerance = 1e-12)



})
