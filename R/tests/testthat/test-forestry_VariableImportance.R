library(testthat)
test_that("Tests if variable importance works", {
  context('Tests Rf Variable Importance')

  set.seed(56)
  x <- iris[, -1]
  y <- iris[, 1]

  # Test forestry (mimic RF)
  forest <- forestry(x, y, ntree = 1000, nthread = 1, seed = 1)

  vi <- getVI(forest, seed=1)

  ### This is non-deterministic because the seed is not passed.
  expect_equal(
    (vi),
    c(0.141523345265, 0.747180382872,
      0.410477571776, 0.423280918055),
    tolerance = 0.1)
})
