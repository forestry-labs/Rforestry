#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <random>
#include "forestry.h"
#include "utils.h"


extern "C" {
    forestry* train_forest(
        void* data_ptr,
        size_t ntree,
        bool replace,
        size_t sampSize,
        double splitRatio,
        bool OOBhonest,
        bool doubleBootstrap,
        size_t mtry,
        size_t minNodeSizeSpt,
        size_t minNodeSizeAvg,
        size_t minNodeSizeToSplitSpt,
        size_t minNodeSizeToSplitAvg,
        double minSplitGain,
        size_t maxDepth,
        size_t interactionDepth,
        unsigned int seed,
        size_t nthread,
        bool verbose,
        bool splitMiddle,
        size_t maxObs,
        size_t minTreesPerFold,
        size_t foldSize,
        bool hasNas,
        bool naDirection,
        bool linear,
        double overfitPenalty,
        bool doubleTree
    );
    void* get_data (
        double* arr,
        size_t* categorical_vars,
        size_t countCategoricals,
        size_t* linFeat_idx,
        size_t countLinFeats,
        double* feat_weights,
        size_t* feat_weight_vars,
        size_t countFtWeightVars,
        double* deep_feat_weights,
        size_t* deep_feat_weight_vars,
        size_t countDeepFtWeightVars,
        double* observation_weights,
        int* mon_constraints,
        size_t* groupMemberships,
        bool monotoneAvg,
        size_t numRows,
        size_t numColumns,
        unsigned int seed
    );
    forestry* reconstructree(
        void* data_ptr,
        size_t ntree,
        bool replace,
        size_t sampSize,
        double splitRatio,
        bool OOBhonest,
        bool doubleBootstrap,
        size_t mtry,
        size_t minNodeSizeSpt,
        size_t minNodeSizeAvg,
        size_t minNodeSizeToSplitSpt,
        size_t minNodeSizeToSplitAvg,
        double minSplitGain,
        size_t maxDepth,
        size_t interactionDepth,
        unsigned int seed,
        size_t nthread,
        bool verbose,
        bool splitMiddle,
        size_t maxObs,
        size_t minTreesPerFold,
        size_t foldSize,
        bool hasNas,
        bool naDirection,
        bool linear,
        double overfitPenalty,
        bool doubleTree,
        size_t* tree_counts,
        double* thresholds,
        int* features,
        int* na_left_count,
        int* na_right_count,
        int* na_default_directions,
        size_t* split_idx,
        size_t* average_idx,
        double* predict_weights,
        unsigned int* tree_seeds
    );
    void predictOOB_forest(
        forestry* forest,
        void* dataframe_pt,
        double* test_data,
        bool doubleOOB,
        bool exact,
        bool returnWeightMatrix,
        bool verbose,
        std::vector<double>& predictions,
        std::vector<double>& weight_matrix,
        std::vector<size_t> training_idx,
        bool hier_shrinkage,
        double lambda_shrinkage
    );
    void predict_forest(
        forestry* forest,
        void* dataframe_pt,
        double* test_data,
        unsigned int seed,
        size_t nthread,
        bool exact,
        bool returnWeightMatrix,
        bool linear,
        bool use_weights,
        size_t* tree_weights,
        size_t num_test_rows,
        std::vector<double>& predictions,
        std::vector<double>& weight_matrix,
        std::vector<double>& coefs,
        bool hier_shrinkage = false,
        double lambda_shrinkage = 0
    );
    void fill_tree_info(
        forestry* forest,
        int tree_idx,
        std::vector<double>& treeInfo,
        std::vector<int>& split_info,
        std::vector<int>& av_info
    );
    size_t get_node_count(forestry* forest, int tree_idx);
    size_t get_split_node_count(forestry* forest, int tree_idx);
    size_t get_leaf_node_count(forestry* forest, int tree_idx);
    void delete_forestry(forestry* forest, void* dataframe_pt);
}

std::string export_json(forestry* forest, const std::vector<double>& colSds, const std::vector<double>& colMeans);
