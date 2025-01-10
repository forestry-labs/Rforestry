library(mvtnorm)
set.seed(1)

# Constructing a proxy for the edge case.
mean_shift <- 25
series_means <- c(Y = 0.48894351, Problem = 0.17927474, V9 = 0.2309653)
series_sds <- c(Y = 0.2840504, Problem = 1.9852952, V9 = 1.08311552)
series_corr_mat <- matrix(c(
  1, 0.095003012, -0.034304399,
  0.0950030129, 1, 0.401277135,
  -0.0343043992, 0.401277135, 1),
  ncol = 3)
colnames(series_corr_mat) <- c("Y", "Problem", "V9")

synth_data <- as.data.frame(
  rmvnorm(
    100,
    mean = series_means,
    sigma = outer(series_sds, series_sds) * series_corr_mat
    ))

synth_data$Problem[sample(nrow(synth_data), ceiling(0.4* nrow(synth_data)))] <- NA

x <-  subset(synth_data, select = -Y)
y <- synth_data$Y - mean(synth_data$Y)
forest_centered <- forestry(
  x, y,
  ntree = 500,
  mtry = 2, seed = 10)
forest_uncentered <- forestry(
  x, y + mean_shift,
  ntree = 500,
  mtry = 2, seed = 10)

expect_lt(
  mean(abs(
    predict(forest_centered, x, seed = 2) -
      (predict(forest_uncentered, x, seed = 2) - mean_shift)
  )),
  0.01)
