test_that("Tests that terminal nodes are correct", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Terminal nodes')
  # Set seed for reproducibility
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    ntree = 1,
    x,
    y,
    replace = TRUE,
    nthread = 2,
    maxDepth = 5,
    seed = 2323
  )

  skip_if_not_mac()

  # Test predict
  full_predictions <- predict(forest, x[c(5, 100, 104),], aggregation = 'terminalNodes')$terminalNodes
  expect_equal(full_predictions, matrix(c(1, 8, 15, 23), ncol = 1))
})
