test_that("Tests that terminal nodes are correct", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Terminal nodes')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    ntree = 1,
    x,
    y,
    replace = TRUE,
    maxDepth = 5
  )

  skip_if_not_mac()

  # Test predict
  full_predictions <- predict(forest, x[c(5, 100, 104),], aggregation = 'weightMatrix')$terminalNodes
  expect_equal(full_predictions, matrix(c(5,14,14,19), ncol = 1))
})
