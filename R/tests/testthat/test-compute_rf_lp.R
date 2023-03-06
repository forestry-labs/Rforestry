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
  rf <- forestry(x = x_train, y = y_train, nthread = 1,scale=FALSE)

  # Compute the l1 distances in the "Species" dimension
  distances_1 <- compute_lp(object = rf,
                            newdata = x_test,
                            feature = "Species",
                            p = 1)

  # Compute the l2 distances in the "Petal.Length" dimension
  distances_2 <- compute_lp(object = rf,
                            newdata = x_test,
                            feature = "Petal.Length",
                            p = 2)

  expect_identical(length(distances_1), nrow(x_test))
  expect_identical(length(distances_2), nrow(x_test))

  #set tolerance
  skip_if_not_mac()

  expect_equal(distances_1,
               c(0.00785714285714286, 0.19730952380952380, 0.00000000000000000, 0.01016666666666667, 0.01807380952380952,
                 0.01933333333333333, 0.00686666666666667, 0.02375000000000000, 0.05970952380952381, 0.03026190476190477, 0.06266666666666666),
               tolerance = 1e-10)
  expect_equal(distances_2,
               c(0.464027624082161, 0.479313427569380, 0.199483638581166, 0.359174919735252,
                 0.502185731724906, 0.374985196507176, 0.421947452590307, 0.406795047600484,
                 0.418917363075821, 0.373736732268053, 0.503685111790328),
               tolerance = 1e-10)
})

