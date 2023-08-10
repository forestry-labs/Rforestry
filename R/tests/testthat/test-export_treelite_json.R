library(rjson)
test_that("Tests that exporting RF to Treelite JSON works", {
  context("Export RF to Treelite JSON")

  seed <- 238943202
  set.seed(seed)

  test_idx <- 1:11

  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  rf <- forestry(x = x_train, y = y_train, ntree = 1, maxDepth = 1, seed = seed)

  export_treelite_json_result <- export_treelite_json(rf)
  result_json_obj <- fromJSON(json_str = export_treelite_json_result)
  json_data_frame <- as.data.frame(result_json_obj)

  testthat::expect_equivalent(json_data_frame["num_feature"],                    4)
  testthat::expect_equivalent(json_data_frame["task_type"],                      "kBinaryClfRegr")
  testthat::expect_equivalent(json_data_frame["trees.root_id"],                  3)
  testthat::expect_equivalent(json_data_frame["trees.nodes.split_feature_id"],   0)
  testthat::expect_equivalent(json_data_frame["trees.nodes.left_child"],         1)
  testthat::expect_equivalent(json_data_frame["trees.nodes.right_child"],        2)
  testthat::expect_equivalent(json_data_frame["trees.nodes.left_child"],         1)
  testthat::expect_equivalent(json_data_frame["trees.nodes.node_id.1"],          1)
  testthat::expect_equivalent(json_data_frame["trees.nodes.node_id.2"],          2)
})
