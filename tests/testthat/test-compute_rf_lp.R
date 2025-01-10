test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  test_idx <- sample(nrow(iris), 11)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train, nthread = 1)

  # Compute the l1 distances in the "Species" dimension
  distances_1 <- compute_lp(object = rf,
                            feature.new = x_test,
                            feature = "Species",
                            p = 1)

  # Compute the l2 distances in the "Petal.Length" dimension
  distances_2 <- compute_lp(object = rf,
                            feature.new = x_test,
                            feature = "Petal.Length",
                            p = 2)

  expect_identical(length(distances_1), nrow(x_test))
  expect_identical(length(distances_2), nrow(x_test))

  #set tolerance
  skip_if_not_mac()

  expect_equal(distances_1,
               c(0.74127647652339, 0.56269154186560, 0.66700207007833, 0.48143305071905,
                 0.42691537245113, 0.79361471149614, 0.69064814060102, 0.60005881782247,
                 0.77731344373143, 0.53970499669885, 0.67328392159715),
               tolerance = 1e-12)
  expect_equal(distances_2,
               c(2.3726809930918, 2.4972611231916, 2.7047479310938, 1.9000801210562,
                 1.6384876050554, 2.4063455932161, 2.1012051982558, 2.4272638737974,
                 3.0785442045313, 2.4121460046764, 2.2978840528426),
               tolerance = 1e-12)
})

