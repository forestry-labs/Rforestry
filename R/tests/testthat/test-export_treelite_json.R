library(jsonlite)
test_that("Tests that exporting RF to Treelite JSON works", {
  context("Export RF to Treelite JSON")

  check_in_range <- function(x, lower, upper) {
    all(x >= lower & x <= upper)
  }

  seed <- 238943202
  set.seed(seed)

  test_idx <- 1:11

  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  rf <- forestry(x = x_train, y = y_train, ntree = 1, maxDepth = 1, seed = seed)
  export_json_ret <- export_json(rf)
  json_parsed <- fromJSON(export_json_ret)

  json_parsed$num_feature == 4
  json_parsed$task_type == "kBinaryClfRegr"

  check_in_range(json_parsed$trees$root_id, 1, 3)

  df <- as.data.frame(json_parsed$trees$nodes)
  check_in_range(df["node_id"], 1, 3)

  feature_ids <- df[["split_feature_id"]]
  check_in_range(feature_ids[1], 0, 3) && all(is.na(feature_ids[2:length(feature_ids)]))

  left_child <- df[["left_child"]]
  check_in_range(left_child[1], 1, 3) && all(is.na(left_child[2:length(left_child)]))

  right_child <- df[["right_child"]]
  check_in_range(right_child[1], 1, 3) && all(is.na(right_child[2:length(right_child)]))
})
