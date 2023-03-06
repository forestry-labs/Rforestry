test_that("Tests predict index option", {
  context('Tests getting specific predictions for one observation')

  x <- iris[1:40,-c(1,5)]
  y <- iris[1:40,1]

  # Given a
  test_tree_preds <- function(rf) {
    rf <- make_savable(rf)
    # Check first tree by hand
    insample_idx <- sort(union(rf@R_forest[[1]]$averagingSampleIndex,
                               rf@R_forest[[1]]$splittingSampleIndex))
    outsample_idx <- setdiff(1:nrow(rf@processed_dta$processed_x), insample_idx)

    p_in <- predict(rf, newdata = rf@processed_dta$processed_x[1,], holdOutIdx = insample_idx)
    expect_equal(p_in, NaN)
    p_out <- predict(rf, newdata = rf@processed_dta$processed_x[1,], holdOutIdx = outsample_idx)
    expect_gt(p_out, 1)

    p_all <- predict(rf, newdata = rf@processed_dta$processed_x[1,], holdOutIdx = c(outsample_idx,
                                                                                    insample_idx))
    expect_equal(p_all, NaN)
  }

  test_forest_preds <- function(rf) {
    # First check normal predictions
    pred_all <- predict(rf, newdata = rf@processed_dta$processed_x[1,], weightMatrix = TRUE)
    expect_equal(sum(pred_all$weightMatrix[1,]), 1)

    pred_holdout <- predict(rf, newdata = rf@processed_dta$processed_x[1,], weightMatrix = TRUE, holdOutIdx = c(1:3))
    expect_equal(sum(pred_holdout$weightMatrix[1,1:3]), 0)

    # Now see if a prediction was able to be made
    if (is.nan(pred_holdout$predictions)) {
      expect_equal(sum(pred_holdout$weightMatrix[1,]), 0)
    } else {
      expect_equal(sum(pred_holdout$weightMatrix[1,]), 1)

      # Check that predictions agree with those given by weightMatrix
      pred_holdout_all <- predict(rf, newdata = rf@processed_dta$processed_x, weightMatrix = TRUE, holdOutIdx = c(1:3))
      weightmatrix_preds <- (pred_holdout_all$weightMatrix %*% as.matrix(rf@processed_dta$y))[,1]
      expect_equal(all.equal(weightmatrix_preds, pred_holdout_all$predictions), TRUE)
    }
  }


  context("Test honest forest with holdOutIdx")
  honest_forest <- forestry(x = x,
                 y = y,
                 seed = 131,
                 OOBhonest = TRUE,
                 ntree = 1000)
  test_tree_preds(honest_forest)
  test_forest_preds(honest_forest)


  # Test normal RF
  context("Test normal RF with holdOutIdx")
  forest <- forestry(x = x,
                 y = y,
                 seed = 131,
                 ntree = 1000)
  test_tree_preds(forest)
  test_forest_preds(forest)

  # test splitratio forest
  context("test splitratio RF with holdOutIdx")
  forest_sr <- forestry(x = x,
                     y = y,
                     seed = 131,
                     splitratio = .5,
                     ntree = 1000)
  test_tree_preds(forest_sr)
  test_forest_preds(forest_sr)


  # test groups forest
  context("test groups forest with holdOutIdx")
  forest_groups <- forestry(x = x,
                            y = y,
                            seed = 131,
                            OOBhonest = TRUE,
                            groups = (as.factor(c(1,1,1, rep(2,37)))),
                            ntree = 1000)
  test_tree_preds(forest_groups)
  test_forest_preds(forest_groups)


  # Now test if the results are roughly equal when exact = FALSE
  context("test holdOutIdx when exact = FALSE")
  honest_forest <- forestry(x = x,
                            y = y,
                            seed = 131,
                            OOBhonest = TRUE,
                            ntree = 1000)
  test_tree_preds(honest_forest)
  preds_exact <-  predict(honest_forest,
                          newdata = honest_forest@processed_dta$processed_x,
                          weightMatrix = TRUE,
                          holdOutIdx = c(1:3))
  preds_weight_exact <- (preds_exact$weightMatrix %*% as.matrix(honest_forest@processed_dta$y))[,1]
  preds_inexct <- predict(honest_forest,
                          newdata = honest_forest@processed_dta$processed_x,
                          weightMatrix = TRUE,
                          exact = FALSE,
                          holdOutIdx = c(1:3))
  preds_weight_inexact <- (preds_inexct$weightMatrix %*% as.matrix(honest_forest@processed_dta$y))[,1]

  expect_equal(all.equal(preds_exact, preds_inexct, tolerance = 1e-3), TRUE)
  expect_equal(all.equal(preds_weight_exact, preds_weight_inexact, tolerance = 1e-3), TRUE)

})
