test_that("Tests if imputation for features works", {
  context('Tests imputation for features')
set.seed(4)
#rm(iris)
# Reordering the rows to test an edge case where factor variables are not in
# alphabetical order.
iris <- iris[sample(nrow(iris), ),]
x <- iris[, -1]
y <- iris[,1]

# Testing standard neighborhood imputation without fallback:
x_with_miss <- x
idx_miss_factor <- sample(nrow(x), 50, replace = TRUE)
x_with_miss[idx_miss_factor, "Species"] <- NA
idx_miss_numeric <- sample(nrow(x), 50, replace = TRUE)
x_with_miss[idx_miss_numeric, "Sepal.Width"] <- NA

skip_if_not_mac()

forest <- forestry(x_with_miss, y, ntree = 500, seed = 2, nthread = 1)
imputed_x <- impute_features(forest, x_with_miss, seed = 2)
expect_equal(sum(imputed_x$Species != x$Species), 2)
expect_equal(mean(abs(x$Sepal.Width - imputed_x$Sepal.Width)), 0.074894503323687369734, tolerance = 1e-6)

# Testing mean imputation fallback:
set.seed(1)
x_with_miss <- x
idx_miss_factor <- sample(nrow(x), 140, replace = TRUE)
x_with_miss[idx_miss_factor, "Species"] <- NA
idx_miss_numeric <- sample(nrow(x), 140, replace = TRUE)
x_with_miss[idx_miss_numeric, "Sepal.Width"] <- NA

forest <- forestry(x_with_miss, y, ntree = 2, seed = 2, nthread = 1)
imputed_x <- impute_features(forest, x_with_miss, seed = 2, use_mean_imputation_fallback = TRUE)
expect_equal(sum(imputed_x$Species != x$Species), 27)
expect_equal(mean(abs(x$Sepal.Width - imputed_x$Sepal.Width)), 0.21616277820762141992, tolerance = 1e-6)


})
