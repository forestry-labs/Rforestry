test_that("Tests that plots succeed", {
  n <- c(100)
  a <- rnorm(n)
  b <- rnorm(n)
  c <- rnorm(n)
  y <- 4*a + 5.5*b - .78*c
  x <- data.frame(a,b,c)

  context('Plot with linear = FALSE and single unique value at leaf')
  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 10,
    replace = TRUE,
    maxDepth = 100,
    nodesizeStrictSpl = 1,
    nodesizeStrictAvg = 1,
    nthread = 2,
    linear = FALSE
  )
  expect_error(plot(forest), regexp = NA)

  context('Plot with linear = TRUE and single unique value at leaf')
  # Test forestry (mimic RF)
  # forest <- forestry(
  #   x,
  #   y,
  #   ntree = 10,
  #   replace = TRUE,
  #   maxDepth = 100,
  #   nodesizeStrictSpl = 1,
  #   nodesizeStrictAvg = 1,
  #   nthread = 2,
  #   linear = TRUE
  # )

  # expect_error with regexp = NA asserts that there should be no errors. I know wtf.
  # expect_error(plot(forest), regexp = NA)

})
