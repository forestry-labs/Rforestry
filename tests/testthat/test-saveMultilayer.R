library(testthat)
test_that("Tests that saving multilayerForestry and loading it works", {
  context("Save and Load Multilayer RF")

  set.seed(238943202)
  x <- iris[,-1]
  y <- iris[, 1]

  #-- Translating C++ to R ------------------------------------------------
  # Check that saving TRUE / FALSE saves and does not save the training data.
  forest <- multilayerForestry(x,
                               y,
                               ntree = 3,
                               nthread = 2,
                               saveable = FALSE)

  # Neither should be filled in yet
  testthat::expect_equal(forest@R_forests, list())
  testthat::expect_equal(forest@R_residuals, list())

  forest <- multilayerForestry(
    x,
    y,
    sample.fraction = 1,
    splitratio = .03,
    ntree = 3,
    nthread = 2,
    saveable = TRUE
  )


  testthat::expect_equal(forest@processed_dta$y[2], 4.9)

  context("Now we try to save and load")
  rf <- multilayerForestry(x = iris[,-1],
                           y = iris[,1],
                           ntree = 2,
                           nrounds = 2,
                           nthread = 2,
                           maxDepth = 3)

  # Get the predictions before saving

  wd <- tempdir()

  preds_before <- predict(rf, newdata = iris[,-1])
  saveForestry(rf, filename = file.path(wd, "forest.Rda"))
  rm(rf)
  rf <- loadForestry(file.path(wd, "forest.Rda"))

  # Get the predictions after loading
  preds_after <- predict(rf, newdata = iris[,-1])
  file.remove(file.path(wd, "forest.Rda"))

  # THey should now be an exact match
  expect_equal(all.equal(preds_after, preds_before, tolerance = 1e-6), TRUE)

})
