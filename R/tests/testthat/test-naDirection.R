test_that("Tests that setting naDirection sets a default direction for missing values", {
  # Create training data with missing values
  x <- iris[, -1]
  y <- iris[,1]
  x_missing <- x
  x_missing[sample(nrow(x), 50, replace = TRUE), "Species"] <- NA
  x_missing[sample(nrow(x), 50, replace = TRUE), "Sepal.Width"] <- NA

  context("Check predictions with naDirection are deterministic and return default directions")
  library(testthat)
  set.seed(101)
  forest_default <- forestry(
    x,
    y,
    savable=TRUE,
    ntree = 100,
    naDirection=TRUE
  )
  forest_default <- make_savable(forest_default)
  pred_default_1 <- predict(forest_default, newdata = x_missing, aggregation = "average")
  pred_default_2 <- predict(forest_default, newdata = x_missing, aggregation = "average")
  expect_true(all.equal(pred_default_1, pred_default_2))
  expect_true(sum(forest_default@R_forest[[1]]$naDefaultDirections != 0) > 0)

  context("Check naDirection for honest forests, monotonicity, and OOB predictions")
  forest_honest <- forestry(
    x,
    y,
    ntree = 100,
    OOBhonest = TRUE,
    monotonicConstraints = rep(1, ncol(x_missing)),
    naDirection = TRUE
  )
  forest_honest <- make_savable(forest_honest)
  pred_honest_1 <- predict(forest_honest, newdata = x_missing, aggregation = "oob")
  pred_honest_2 <- predict(forest_honest, newdata = x_missing, aggregation = "oob")
  expect_true(all.equal(pred_honest_1, pred_honest_2))
  expect_true(sum(forest_honest@R_forest[[1]]$naDefaultDirections != 0) > 0)
})
