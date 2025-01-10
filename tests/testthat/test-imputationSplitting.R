test_that("Tests if imputation splitting works", {
  context('Tests impute')

  # Try Continuous Split ---------------------------------------------------------
  library(Rforestry)

  set.seed(23934)
  x <- as.matrix(runif(100))
  y <- ifelse(x > .4, 4, 2)

  rf <- forestry(x = x,
                 y = y,
                 ntree = 500)

  pred <- predict(rf, x)
  x_missing <- x
  x_missing[c(2,6,12,19,25,34,12,67,90, 97), ] <- NA

  rf_impute <- forestry(x = x_missing,
                        y = y,
                        ntree = 500)

  pred_na <- predict(rf_impute, x)


  # Test saving and loading
  rf_impute <- make_savable(rf_impute)
  save(rf_impute, file = "testForest.Rds")
  rm(rf_impute)
  load("testForest.Rds")
  rf_impute_reloaded <- relinkCPP_prt(rf_impute)

  saved_pred <- predict(rf_impute_reloaded, x)

  expect_equal(saved_pred, pred_na, tolerance = 1e-12)

  # Try Categorical Split ------------------------------------------------------
  set.seed(2349834)

  x <- as.matrix(sample(1:5, size = 100, replace = TRUE))
  y <- sapply(x, function(x) {if (x == 1) {return(4.5)}
                              else if (x == 3) {return(2.8)}
                              else if (x == 4) {return(1.5)}
                              else {return(5.7)} })

  rf <- forestry(x = x,
                 y = y,
                 ntree = 500)

  pred <- predict(rf, x)

  x_missing <- x
  x_missing[c(2,6,34,12,67,90, 97)] <- NA

  rf_impute <- forestry(x = x_missing,
                        y = y,
                        ntree = 500)

  pred_na <- predict(rf_impute, x)

  rf_impute <- make_savable(rf_impute)
  save(rf_impute, file = "testForest.Rds")
  rm(rf_impute)
  load("testForest.Rds")
  rf_impute_reloaded <- relinkCPP_prt(rf_impute)

  saved_pred <- predict(rf_impute_reloaded, x)

  expect_equal(saved_pred, pred_na, tolerance = 1e-12)

  file.remove("testForest.Rds")
})

