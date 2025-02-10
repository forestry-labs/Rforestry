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
               c(0.741505444777, 0.557691541866, 0.667464191290, 0.481433050719,
                 0.428715372451, 0.793614711496, 0.690142079995, 0.598410669674,
                 0.777646777065, 0.542104996699, 0.671783921597),
               tolerance = 1e-6)
  expect_equal(distances_2,
               c(2.37298192278, 2.48866693581, 2.70150942321, 1.90008012106,
                 1.63721780222, 2.40468205396, 2.09935249333, 2.42295512410,
                 3.07584694340, 2.41614965345, 2.29615354894),
               tolerance = 1e-6)
})

