test_that("Tests splitratio honesty", {

  context("Check that the averaging and splitting sets are disjoint")

  avg_split_disjoint <- function(
    rf,
    tree_id
  ) {
    return(length(intersect(sort(rf@R_forest[[tree_id]]$splittingSampleIndex),
                            sort(rf@R_forest[[tree_id]]$averagingSampleIndex))) == 0)
  }

  avg_split_size <- function(
    rf,
    tree_id,
    splitratio,
    nobs,
    verbose = FALSE
  ) {
    num_spl <- floor(nobs*splitratio)
    num_avg <- nobs - num_spl
    if (verbose) {
      print(num_spl)
      print(length(rf@R_forest[[tree_id]]$splittingSampleIndex))
    }

    return(length(rf@R_forest[[tree_id]]$splittingSampleIndex) == num_spl &&
           length(rf@R_forest[[tree_id]]$averagingSampleIndex) == num_avg)
  }


  splitratio_use = .3
  x = iris[,-1]
  y = iris[,1]

  rf <- forestry(x = x,
                 y = y,
                 ntree = 50,
                 seed = 123,
                 splitratio = splitratio_use)
  rf <- make_savable(rf)

  for (idx in 1:rf@ntree) {
    c1 <- avg_split_disjoint(rf, tree_id = idx)
    c2 <- avg_split_size(rf, tree_id = idx, nobs = nrow(x), splitratio = .3)
    expect_equal(c1,TRUE)
  }

  splitratio_use = .9
  x = data.frame(x1 = cars[,1])
  y = cars[,-1]

  rf2 <- forestry(x = x,
                 y = y,
                 ntree = 50,
                 seed = 123,
                 splitratio = splitratio_use)
  rf2 <- make_savable(rf2)

  for (idx in 1:rf2@ntree) {
    c1 <- avg_split_disjoint(rf2, tree_id = idx)
    c2 <- avg_split_size(rf2, tree_id = idx, nobs = nrow(x), splitratio = splitratio_use)
    expect_equal(c1,TRUE)
  }

  splitratio_use = .632
  x = data.frame(matrix(rnorm(1000), ncol = 10, nrow=100))
  y = rnorm(100)

  rf3 <- forestry(x = x,
                 y = y,
                 ntree = 50,
                 seed = 123,
                 splitratio = splitratio_use)
  rf3 <- make_savable(rf3)

  for (idx in 1:rf3@ntree) {
    c1 <- avg_split_disjoint(rf3, tree_id = idx)
    c2 <- avg_split_size(rf3, tree_id = idx, nobs = nrow(x), splitratio = splitratio_use)
    expect_equal(c1,TRUE)
  }

  context("Attempt to break honesty in zero signal dgp")

  for (iter in 1:10) {
    set.seed(iter+1)
    x <- data.frame(x1 = rnorm(1000))
    y <- rnorm(1000)


    rf5 <- forestry(x = x,
                    y = y,
                    ntree = 100,
                    seed = iter,
                    splitratio = splitratio_use)
    rf5 <- make_savable(rf5)

    p <- predict(rf5, newdata = x)
    expect_gt(cor(p, y), .2)

    p_oob <- predict(rf5, newdata = x, aggregation = "oob")
    expect_lt(cor(p_oob, y), .1)
  }


})
