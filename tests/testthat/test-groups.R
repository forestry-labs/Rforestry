test_that("Tests if groups argument works works", {
  context('Tests groups')

  x <- iris[1:40,-c(1,5)]
  y <- iris[1:40,1]

  group_vec = as.factor(1:40)

  set.seed(2332)
  rf <- forestry(
    x = x,
    y = y,
    groups = group_vec,
    ntree = 1,
    seed = 2332
  )

  rf <- make_savable(rf)
  indices <- sort(unique(rf@R_forest[[1]]$averagingSampleIndex))
  groups <- unique(as.integer(group_vec[indices]))
  preds <- predict(rf, aggregation = "oob")

  context("Test several different groups")
  # In the case where each observation is in its own group, this corresponds to
  # standard OOB predictions
  expect_equal(all.equal(sort(which(is.nan(preds))), indices), TRUE)

  group_vec = as.factor(rep(1:8, each = 5))

  set.seed(2332)
  rf <- forestry(
    x = x,
    y = y,
    groups = group_vec,
    sample.fraction = .5,
    replace = TRUE,
    ntree = 1,
    seed = 2332
  )

  rf <- make_savable(rf)
  indices <- sort(unique(rf@R_forest[[1]]$averagingSampleIndex))
  groups <- unique(as.integer(group_vec[indices]))
  preds <- predict(rf, aggregation = "oob")

  # Now we can predict for the first group as it has no obervations selected for the tree.
  expect_equal(all.equal(sort(which(is.nan(preds))), 6:40), TRUE)


  context("Test groups with OOB honesty")
  # The implementation is general- we just loook at the averaging observations
  # for a tree, so it works with OOB honesty too
  set.seed(2332)
  rf <- forestry(
    x = x,
    y = y,
    groups = group_vec,
    sample.fraction = .5,
    replace = TRUE,
    OOBhonest = TRUE,
    ntree = 1,
    seed = 2332
  )

  rf <- make_savable(rf)
  indices <- sort(unique(rf@R_forest[[1]]$averagingSampleIndex))
  groups <- unique(as.integer(group_vec[indices]))
  preds <- predict(rf, aggregation = "oob")

  # The last four groups don't have any observations in the averaging set, so
  # we are allowed to predict on them
  expect_equal(all.equal(sort(which(is.nan(preds))), 1:20), TRUE)

})
