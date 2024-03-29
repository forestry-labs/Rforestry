#' @include forestry.R
#' @importFrom stats predict
NULL

# ---plots a tree ----------------------------------------------------------
#' plot
#' @name plot-forestry
#' @title visualize a tree
#' @rdname plot-forestry
#' @description plots a tree in the forest.
#' @param x A forestry x.
#' @param tree.id Specifies the tree number that should be visualized.
#' @param print.meta_dta A flag indicating whether the data for the plot should be printed.
#' @param beta.char.len The length of the beta values in leaf node
#'  representation. This is only required when plotting a forestry object with linear
#'  aggregation functions (linear = TRUE).
#' @param ... additional arguments that are not used.
#' @return A plot of the specified tree in the forest.
#' @import glmnet
#' @examples
#' set.seed(292315)
#' rf <- forestry(x = iris[,-1],
#'                y = iris[, 1],
#'                nthread = 2)
#'
#' plot(x = rf)
#' plot(x = rf, tree.id = 2)
#' plot(x = rf, tree.id = 500)
#'
#'
#' @export
#' @import visNetwork
plot.forestry <- function(x, tree.id = 1, print.meta_dta = FALSE,
                          beta.char.len = 30, ...) {
  if (x@ntree < tree.id | 1 > tree.id) {
    stop("tree.id is too large or too small.")
  }

  forestry_tree <- make_savable(x)

  # Get indicator of whether or not we need to scale
  scale <- forestry_tree@scale

  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val
  node_values <- forestry_tree@R_forest[[tree.id]]$weights
  avg_counts <- forestry_tree@R_forest[[tree.id]]$average_count
  split_counts <- forestry_tree@R_forest[[tree.id]]$split_count
  # get info for the first node ------------------------------------------------
  root_is_leaf <- split_feat[1] < 0
  first_val <- split_val[1]
  node_info <- data.frame(
    node_id = 1,
    is_leaf = root_is_leaf,
    parent = NA,
    left_child = ifelse(root_is_leaf, NA, 2),
    right_child = NA,
    split_feat = ifelse(root_is_leaf, NA, split_feat[1]),
    split_val = ifelse(root_is_leaf, NA, first_val),
    num_splitting = ifelse(root_is_leaf, split_counts[1], 0),
    num_averaging = ifelse(root_is_leaf, avg_counts[1], 0),
    value = 0.0,
    level = 1)
  split_feat <- split_feat[-1]
  split_val <- split_val[-1]
  node_values <- node_values[-1]
  avg_counts <- avg_counts[-1]
  split_counts <- split_counts[-1]
  # loop through split feat to get all the information -------------------------
  while (length(split_feat) != 0) {
    if (!node_info$is_leaf[nrow(node_info)]) {
      # previous node is not leaf => left child of previous node
      parent <- nrow(node_info)
      node_info$left_child[parent] <- nrow(node_info) + 1
    } else {
      # previous node is leaf => right child of last unfilled right
      parent <-
        max(node_info$node_id[!node_info$is_leaf &
                                is.na(node_info$right_child)])
      node_info$right_child[parent] <- nrow(node_info) + 1
    }
    if (split_feat[1] > 0) {
      # it is not a leaf
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = FALSE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = split_feat[1],
          split_val = split_val[1],
          num_splitting = 0,
          num_averaging = 0,
          value = 0.0,
          level = node_info$level[parent] + 1
        )
      )
    } else {
      #split_feat[1] < 0
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = TRUE,
          parent = parent,
          left_child = NA,
          right_child = NA,
          split_feat = NA,
          split_val = NA,
          num_splitting = split_counts[1],
          num_averaging = avg_counts[1],
          value = node_values[1],
          level = node_info$level[parent] + 1
        )
      )
    }

    split_feat <- split_feat[-1]
    split_val <- split_val[-1]
    node_values <- node_values[-1]
    avg_counts <- avg_counts[-1]
    split_counts <- split_counts[-1]
  }

  for (i in nrow(node_info):1) {
    if (node_info$num_splitting[i] != 0) {
      node_info$num_splitting[node_info$parent[i]] <-
        node_info$num_splitting[node_info$parent[i]] + node_info$num_splitting[i]

      node_info$num_averaging[node_info$parent[i]] <-
        node_info$num_averaging[node_info$parent[i]] + node_info$num_averaging[i]
    }
  }


  # Save more information about the splits -------------------------------------
  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  node_info$feat_nm <- NA
  node_info$is_cat_feat <- NA
  node_info$cat_split_value <- NA
  if (sum(!is.na(node_info$split_feat)) != 0) {
    node_info$feat_nm <- feat_names[node_info$split_feat]
    cat_feats <- names(forestry_tree@processed_dta$categoricalFeatureCols_cpp)
    node_info$is_cat_feat <- node_info$feat_nm %in% cat_feats
    node_info$cat_split_value <- as.character(node_info$split_val)

    cat_feat_map <- forestry_tree@categoricalFeatureMapping
    if (length(cat_feat_map) > 0) {
      for (i in 1:length(cat_feat_map)) {
        # i = 1
        nodes_with_this_split <-
          node_info$split_feat == cat_feat_map[[i]]$categoricalFeatureCol &
          (!is.na(node_info$split_feat))

        node_info$cat_split_value[nodes_with_this_split] <-
          as.character(cat_feat_map[[i]]$uniqueFeatureValues[
            node_info$split_val[nodes_with_this_split]])
      }
    }
  }

  # Prepare data for VisNetwork ------------------------------------------------

  nodes <- data.frame(
    id = node_info$node_id,
    shape = ifelse(node_info$is_leaf,
                   "box", "circle"),
    label = ifelse(
      node_info$is_leaf,
      paste0(node_info$num_averaging, " Obs"),
      paste0(feat_names[node_info$split_feat])
    ),
    level = node_info$level
  )

  edges <- data.frame(
    from = node_info$parent,
    to = node_info$node_id,
    smooth = list(
      enabled = TRUE,
      type = "cubicBezier",
      roundness = .5
    )
  )
  edges <- edges[-1, ]


  edges$label =
    ifelse(
      floor(node_info$split_val[edges$from]) == node_info$split_val[edges$from],
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" = ", node_info$cat_split_value[edges$from]),
        paste0(" != ", node_info$cat_split_value[edges$from])
      ),
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" < ", round(node_info$split_val[edges$from], digits = 2)),
        paste0(" >= ", round(node_info$split_val[edges$from], digits = 2))
      )
    )

  edges$width = node_info$num_averaging[edges$to] /
    (node_info$num_averaging[1] / 4)

  # collect data for leaves ----------------------------------------------------
  nodes$label <- as.character(nodes$label)
  nodes$title <- as.character(nodes$label)


  dta_x <- forestry_tree@processed_dta$processed_x
  dta_y <- forestry_tree@processed_dta$y


  if (forestry_tree@linear) {
    plm = this_ds = y_leaf_unique = NULL
    # ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      # leaf_id = 5
      ###
      #this_ds <- dta_x[leaf_idx[[leaf_id]],
      #                 forestry_tree@linFeats + 1]
      encoder <- onehot::onehot(this_ds)
      remat <- predict(encoder, this_ds)
      ###
      #y_leaf <- dta_y[leaf_idx[[leaf_id]]]

      # handle single unique value y_leaf otherwise glmnet fails
      #y_leaf_unique <- unique(y_leaf)
      #plm_pred_names <- c("interc", colnames(remat))

      return_char <- character()
      dev.ratio <- 1

      if(length(y_leaf_unique) == 1) {
        #return_char = paste0(substr(plm_pred_names[1], 1, beta.char.len), " ", round(y_leaf_unique, 2), "<br>")
        dev.ratio <- 1
      } else {
        #plm <- glmnet::glmnet(x = remat,
        #                      y = y_leaf,
        #                      lambda = forestry_tree@overfitPenalty * sd(y_leaf)/nrow(remat),
        #                      alpha	= 0)

        #plm_pred <- predict(plm, type = "coef")



        #for (i in 1:length(plm_pred)) {
        #  return_char <- paste0(return_char,
        #                        substr(plm_pred_names[i], 1, beta.char.len), " ",
        #                        round(plm_pred[i], 2), "<br>")
        #}

        dev.ratio <- plm$dev.ratio
      }


      node_weight = node_info$value[leaf_id]
      if (scale) {
        node_weight = node_weight * forestry_tree@colSd[length(forestry_tree@colSd)] +
          forestry_tree@colMeans[length(forestry_tree@colMeans)]
      }

      nodes$title[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "<br> R2 = ",
                                     dev.ratio,
                                     "<br>========<br>",
                                     return_char)
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "\n R2 = ",
                                     round(dev.ratio, 3),
                                     "\n=======\nm = ",
                                     round(node_weight, 5))
    }
  } else {
    # not ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      node_weight = node_info$value[leaf_id]
      if (scale) {
        node_weight = node_weight * forestry_tree@colSd[length(forestry_tree@colSd)] +
          forestry_tree@colMeans[length(forestry_tree@colMeans)]
      }
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id], "\n=======\nm = ",
                                     round(node_weight,5))
    }
  }

  # defines a colors -----------------------------------------------------------
  split_vals <- node_info$split_feat
  split_vals <- ifelse(is.na(split_vals), 0, split_vals)
  split_vals <- factor(split_vals)
  color_code <- grDevices::terrain.colors(n = length(feat_names) + 1,
                                          alpha = .7)
  names(color_code) <- as.character(0:(length(feat_names)))
  nodes$color <- color_code[split_vals]

  # Plot the actual node -------------------------------------------------------
  (p1 <-
    visNetwork(
      nodes,
      edges,
      width = "100%",
      height = "800px",
      main = paste("Tree", tree.id)
    ) %>%
    visEdges(arrows = "to") %>%
    visHierarchicalLayout() %>% visExport(type = "pdf", name = "ridge_tree"))

  if (print.meta_dta) print(node_info)
  return(p1)
}
