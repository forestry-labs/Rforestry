library(testthat)
test_that("Tests that Monotone splits parameter is working correctly in the case of missing data", {
  # Set seed for reproductivity
  set.seed(24750371)
  x <- data.frame(V1 = runif(100, min = 0, max = 10))
  y <- .2*x[,1] + rnorm(100)

  context('Positive monotone splits with missing data')


  # Take some missing examples
  x_missing <- x
  x_missing[c(2,5,8,23,25,90),1] <- NA

  monotone_forest <- forestry(
    x_missing,
    y,
    ntree = 500,
    nodesizeStrictSpl = 5,
    maxDepth = 10,
    monotonicConstraints = c(1),
    seed = 2
  )
  # Test predictions are monotonic increasing in the first feature
  pred_means <- sapply(c(1:9), function(x) {mean(predict(monotone_forest,
                                                         feature.new = data.frame(V1 = rep(x, 100))))})

  # predictions should be monotonically increasing
  expect_equal(all.equal(order(pred_means), 1:9), TRUE)

  # Now try monotone decreasing in second feature- even though there is no real signal here
  monotone_forest <- forestry(
    x_missing,
    y,
    ntree = 500,
    nodesizeStrictSpl = 3,
    maxDepth = 10,
    monotonicConstraints = c(-1),
    seed = 2
  )
  # Test predictions are monotonic decreasing in the first feature
  pred_means <- sapply(c(1,4,9,15), function(x) {mean(predict(monotone_forest,
                                                             feature.new = data.frame(V1 = rep(x, 100))))})

  skip_if_not_mac()

  # Mean Square Error
  #print(pred_means)

  expect_equal(all.equal(order(pred_means), 4:1), TRUE)
})


test_that("Tests that Monotone splits parameter is working correctly in the case of missing data (sine wave)", {
  set.seed(23423324)

  context('Positive monotone splits with missing data')

  # Sine wave example. Suppose we have a slight trend upwards, but this is
  # complicated by some oscillations, we can then constrain monotonicity to avoid
  # the noise from these complications
  x_1 <- rnorm(150)+5
  x_2 <- rnorm(150)+5
  y <- .15*x_1 + .5*sin(3*x_1)
  x_1_missing <- x_1
  x_2_missing <- x_2

  x_1_missing[c(2,5,8,23,25,90)] <- NA
  x_2_missing[c(24,16,18,19,20,78,76)] <- NA
  data_train <- data.frame(x1 = x_1_missing, x2 = x_2_missing, y = y + rnorm(150, sd = .4))

  monotone_rf <- forestry(x = data_train[, -3],
                          y = data_train$y,
                          monotonicConstraints = c(1,-1),
                          nodesizeStrictSpl = 5,
                          nthread = 1,
                          ntree = 500,
                          seed = 10)

  preds <- predict(monotone_rf, feature.new = data.frame(x1 = c(2, 7,7), x2 = c(4,4,9)))
  # Now should be monotone increasing in first feature
  expect_equal(preds[2] >= preds[1], TRUE)

  # Should be monotone decreasing in second feature
  expect_equal(preds[2] >= preds[3], TRUE)
})
