context("Deep splitting")

test_that('Deep splitting obeys variable selection restrictions', {

  set.seed(1)
  n <- 100
  p <- 4
  x <- matrix(rnorm(n * p), ncol = p)
  y <- rnorm(n)
  forest <- forestry(
    x, y, interactionDepth = 3, maxDepth = 20, interactionVariables = 0:1, mtry = 1,
    ntree = 100)
  forest <- make_savable(forest)
  for(tree.id in 1:forest@ntree) {
    # This is an unsightly hack to recover variables used for splitting in tree
    # above and below interactionDepth

    out <- plot(forest, tree.id = tree.id)
    node_matrix <- out$x$nodes
    shallow_variables <- sort(unique(subset(node_matrix, level < 3)$label))
    shallow_variables <- shallow_variables[nchar(shallow_variables) < 5]
    expect_true(all(shallow_variables %in% c("V3", "V4")))

    deep_variables <- sort(unique(subset(node_matrix, level >= 3)$label))
    deep_variables <- deep_variables[nchar(deep_variables) < 5]
    expect_true(length(deep_variables) == p)
  }
})


test_that("Test that deep splitting has finite run time when mtry > number of non zero entries in featureWeights", {
  set.seed(1)
  n <- 100
  p <- 4
  x <- matrix(rnorm(n * p), ncol = p)
  y <- rnorm(n)
  forest <- forestry(
    x, y, interactionDepth = 3, maxDepth = 20, interactionVariables = 0:1, mtry = 3,
    ntree = 100)
  forest <- make_savable(forest)
  for(tree.id in 1:forest@ntree) {
    # This is an unsightly hack to recover variables used for splitting in tree
    # above and below interactionDepth

    out <- plot(forest, tree.id = tree.id)
    node_matrix <- out$x$nodes
    shallow_variables <- sort(unique(subset(node_matrix, level < 3)$label))
    shallow_variables <- shallow_variables[nchar(shallow_variables) < 5]
    expect_true(all(shallow_variables %in% c("V3", "V4")))

    deep_variables <- sort(unique(subset(node_matrix, level >= 3)$label))
    deep_variables <- deep_variables[nchar(deep_variables) < 5]
    expect_true(length(deep_variables) == p)
  }
})

