test_that("Tests sampling with groups", {

  # Helper function to check the validity of group out sampling + honesty
  # basically need 1 group out and the others partitioned into averaging + splitting
  check_validity_gout_sampling <- function(
    rf,
    tree_id,
    g_list
  ) {
    # Check that one fold is completely left out
    g_out <- lapply(g_list, function(x) {return(length(intersect(union(
      rf@R_forest[[tree_id]]$splittingSampleIndex,
      rf@R_forest[[tree_id]]$averagingSampleIndex),x))==0) })

    any_g_out <- any(unlist(g_out))

    avg_g <- lapply(g_list, function(x) {return(length(intersect(
      rf@R_forest[[tree_id]]$averagingSampleIndex,x))!=0) })
    spl_g <- lapply(g_list, function(x) {return(length(intersect(
      rf@R_forest[[tree_id]]$splittingSampleIndex,x))!=0) })

    out <- data.frame(r1 = as.numeric(g_out),
                      r2 = as.numeric(avg_g),
                      r3 = as.numeric(spl_g))

    return(all(rowSums(out) <= 1))
  }

  test_fold_left_out <- function(
    rf,
    tree_id,
    g_list,
    foldSize
  ) {
    # Check that one fold is completely left out
    g_out <- lapply(g_list, function(x) {return(length(intersect(union(
      rf@R_forest[[tree_id]]$splittingSampleIndex,
      rf@R_forest[[tree_id]]$averagingSampleIndex),x))==0) })

    # Check that at least foldSize groups are left out
    return(length(which(unlist(g_out))) >= foldSize)
  }

  # Helper function for checking that the averaging and splitting groups
  # are disjoint
  check_avg_spl_groups_disjoint <- function(
    rf,
    tree_id,
    g_list
  ) {

    avg_g <- lapply(g_list, function(x) {return(length(intersect(
      rf@R_forest[[tree_id]]$averagingSampleIndex,x))!=0) })
    spl_g <- lapply(g_list, function(x) {return(length(intersect(
      rf@R_forest[[tree_id]]$splittingSampleIndex,x))!=0) })

    out <- data.frame(r2 = as.numeric(avg_g),
                      r3 = as.numeric(spl_g))
    return(all(rowSums(out) <= 1))
  }


  # Test sampling with groups and honesty
  context("Test sampling with groups and honesty and minTreesPerFold > 0")

  rf <- forestry(x = iris[,-1],
                 y = iris[,1],
                 groups = iris$Species,
                 foldSize = 1,
                 ntree = 30,
                 minTreesPerFold = 10,
                 splitratio = .632,
                 seed = 123123
  )
  rf <- make_savable(rf)


  for (tree_idx in 1:rf@ntree) {
    g_list = list(1:50,51:100,101:150)

    c1 <- check_avg_spl_groups_disjoint(rf = rf,
                                  tree_id = tree_idx,
                                  g_list = g_list)
    c2 <- check_validity_gout_sampling(rf = rf,
                                  tree_id = tree_idx,
                                  g_list = g_list)

    expect_equal(c2, TRUE)
    expect_equal(c1, TRUE)
  }

  rf2 <- forestry(x = iris[,-1],
                 y = iris[,1],
                 groups = as.factor(sapply(1:10, function(x) return(rep(x,15)))),
                 foldSize = 1,
                 ntree = 100,
                 minTreesPerFold = 10,
                 splitratio = .632,
                 seed = 123123
  )
  rf2 <- make_savable(rf2)


  for (tree_idx in 1:rf2@ntree) {
    g_list = lapply(as.list(1:10), function(x){return(which(sapply(1:10, function(x) return(rep(x,15))) == x))})

    c1 <- check_avg_spl_groups_disjoint(rf = rf2,
                                        tree_id = tree_idx,
                                        g_list = g_list)

    c2 <- check_validity_gout_sampling(rf = rf2,
                                       tree_id = tree_idx,
                                       g_list = g_list)
    expect_equal(c2, TRUE)
    expect_equal(c1, TRUE)
  }

  context("Test group sampling with no minTreesPerFold")

  rf3 <- forestry(x = iris[,-1],
                  y = iris[,1],
                  groups = as.factor(sapply(1:10, function(x) return(rep(x,15)))),
                  foldSize = 1,
                  ntree = 50,
                  splitratio = .632,
                  seed = 123123
  )
  rf3 <- make_savable(rf3)

  for (tree_idx in 1:rf3@ntree) {
    g_list = lapply(as.list(1:10), function(x){return(which(sapply(1:10, function(x) return(rep(x,15))) == x))})

    c1 <- check_avg_spl_groups_disjoint(rf = rf3,
                                        tree_id = tree_idx,
                                        g_list = g_list)

    expect_equal(c1, TRUE)
  }

  rf3 <- forestry(x = iris[,-1],
                  y = iris[,1],
                  groups = as.factor(sapply(1:10, function(x) return(rep(x,15)))),
                  foldSize = 3,
                  ntree = 50,
                  splitratio = .632,
                  seed = 123123
  )
  rf3 <- make_savable(rf3)

  for (tree_idx in 1:rf3@ntree) {
    g_list = lapply(as.list(1:10), function(x){return(which(sapply(1:10, function(x) return(rep(x,15))) == x))})

    c1 <- check_avg_spl_groups_disjoint(rf = rf3,
                                        tree_id = tree_idx,
                                        g_list = g_list)

    expect_equal(c1, TRUE)
  }

  context("Test different foldSizes + honesty")

  rf4 <- forestry(x = iris[,-1],
                  y = iris[,1],
                  groups = as.factor(sapply(1:10, function(x) return(rep(x,15)))),
                  foldSize = 2,
                  ntree = 50,
                  minTreesPerFold = 10,
                  splitratio = .632,
                  seed = 123123
  )
  rf4 <- make_savable(rf4)

  for (tree_idx in 1:rf4@ntree) {
    g_list = lapply(as.list(1:10), function(x){return(which(sapply(1:10, function(x) return(rep(x,15))) == x))})

    c1 <- check_avg_spl_groups_disjoint(rf = rf4,
                                        tree_id = tree_idx,
                                        g_list = g_list)

    c2 <- test_fold_left_out(rf = rf4,
                             tree_id = tree_idx,
                             g_list = g_list,
                             foldSize = 2)

    expect_equal(c1, TRUE)
    expect_equal(c2, TRUE)
  }

  rf5 <- forestry(x = iris[,-1],
                  y = iris[,1],
                  groups = as.factor(sapply(1:15, function(x) return(rep(x,10)))),
                  foldSize = 5,
                  ntree = 30,
                  minTreesPerFold = 10,
                  OOBhonest = TRUE,
                  seed = 123123
  )
  rf5 <- make_savable(rf5)

  for (tree_idx in 1:rf5@ntree) {
    g_list = lapply(as.list(1:15), function(x){return(which(sapply(1:15, function(x) return(rep(x,10))) == x))})

    c1 <- check_avg_spl_groups_disjoint(rf = rf5,
                                        tree_id = tree_idx,
                                        g_list = g_list)

    c2 <- test_fold_left_out(rf = rf5,
                             tree_id = tree_idx,
                             g_list = g_list,
                             foldSize = 5)

    expect_equal(c1, TRUE)
    expect_equal(c2, TRUE)
  }

})
