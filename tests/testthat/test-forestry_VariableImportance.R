library(testthat)
test_that("Tests if variable importance works", {
  context('Tests Rf Variable Importance')

  set.seed(56)
  x <- iris[, -1]
  y <- iris[, 1]

  # Test forestry (mimic RF)
  forest <- forestry(x, y, ntree = 1000)

  vi <- getVI(forest)

  ### This is non-deterministic because the seed is not passed.
  expect_equal(
    unlist(vi),
    c(0.218073688508096, 1.21796168072043,
      0.536307185530931, 0.468263916283163),
    tolerance = 0.1)
})
