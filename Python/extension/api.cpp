#include <vector>
#include <string>
#include <new>
#include <iostream>
#include <random>
#include "forestry.h"
#include "dataFrame.h"
#include "utils.h"
#include "forestryTree.h"


extern "C" {
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
    ) {
        // Create Data: first n_cols - 1 are features, last is outcome
        std::unique_ptr< std::vector< std::vector<double> > > featureData {
                new std::vector< std::vector<double> >(numColumns-1, std::vector<double>(numRows))
        };
    
        for (size_t j = 0; j < numColumns-1; j++) {
            for (size_t i = 0; i<numRows; i++){
                featureData->at(j).at(i) = arr[i*numColumns + j];
            }
        }
        
    
        // Create outcome data
        std::unique_ptr< std::vector<double> > outcomeData {
                new std::vector<double>(numRows)
        };
    
        for (size_t i = 0; i < numRows; i++) {
            outcomeData->at(i) = arr[numColumns*(i+1)-1];
        }
    
        numColumns--;
    
    
        // Categorical features column
        std::unique_ptr< std::vector<size_t> > categoricalFeatureCols (
                new std::vector<size_t> (countCategoricals)
        );
    
        for (size_t i = 0; i < countCategoricals; i++) {
            categoricalFeatureCols->at(i) = categorical_vars[i];
        }
    
    
        // Linear features column
        std::unique_ptr< std::vector<size_t> > linearFeatures (
                new std::vector<size_t> (countLinFeats)
        );
    
        for (size_t i = 0; i < countLinFeats; i++) {
            linearFeatures->at(i) = linFeat_idx[i];
        }
    
    
        // Feature weights for each column
        std::unique_ptr< std::vector<double> > feature_weights (
                new std::vector<double>(numColumns)
        );
    
        for (size_t i = 0; i < numColumns; i++){
            feature_weights->at(i) = feat_weights[i];
        }
    
    
        // Feature indecies based on feature_weights
        std::unique_ptr< std::vector<size_t> > feature_weight_vars (
                new std::vector<size_t> (countFtWeightVars)
        );
    
        for (size_t i = 0; i < countFtWeightVars; i++) {
            feature_weight_vars->at(i) = feat_weight_vars[i];
        }
    
    
        // Deep feature weights for each column
        std::unique_ptr< std::vector<double> > deep_feature_weights (
                new std::vector<double>(numColumns)
        );
    
        for (size_t i = 0; i < numColumns; i++){
            deep_feature_weights->at(i) = deep_feat_weights[i];
        }
    
    
        // Deep feature indices based
        std::unique_ptr< std::vector<size_t> > deep_feature_weight_vars (
                new std::vector<size_t> (countDeepFtWeightVars)
        );
    
        for (size_t i = 0; i < countDeepFtWeightVars; i++) {
            deep_feature_weight_vars->at(i) = deep_feat_weight_vars[i];
        }
    
    
        // Observation weights
        std::unique_ptr< std::vector<double> > obs_weights (
                new std::vector<double>(numRows)
        );
    
        for (size_t i = 0; i < numRows; i++){
            obs_weights->at(i) = observation_weights[i];
        }
    
        std::unique_ptr< std::vector< std::vector<size_t> > > customSplitSample(
                new std::vector< std::vector<size_t> >
        );

        std::unique_ptr< std::vector< std::vector<size_t> > > customAvgSample(
                new std::vector< std::vector<size_t> >
        );

        std::unique_ptr< std::vector< std::vector<size_t> > > customExcludeSample(
                new std::vector< std::vector<size_t> >
        );
    
        // monotone constraints for each column
        std::unique_ptr< std::vector<int> > monotone_constraints (
                new std::vector<int>(numColumns)
        );
    
        for (size_t i = 0; i < numColumns; i++){
            monotone_constraints->at(i) = mon_constraints[i];
        }
    
    
        // group membership for each observation
        std::unique_ptr< std::vector<size_t> > groups (
                new std::vector<size_t>(numRows)
        );
    
        for (size_t i = 0; i < numRows; i++){
            groups->at(i) = groupMemberships[i];
        }
    
    
        DataFrame* test_df = new DataFrame(
            std::move(featureData),
            std::move(outcomeData),
            std::move(categoricalFeatureCols),
            std::move(linearFeatures),
            numRows,
            numColumns,
            std::move(feature_weights),
            std::move(feature_weight_vars),
            std::move(deep_feature_weights),
            std::move(deep_feature_weight_vars),
            std::move(obs_weights),
            std::move(customSplitSample),
            std::move(customAvgSample),
            std::move(customExcludeSample),
            std::move(monotone_constraints),
            std::move(groups),
            monotoneAvg
        );
    
        return test_df;
    }
    
    
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
    ) {
        DataFrame* test_df = reinterpret_cast<DataFrame* >(data_ptr);
        forestry* forest ( new (std::nothrow) forestry(
            test_df,
            ntree,
            replace,
            sampSize,
            splitRatio,
            OOBhonest,
            doubleBootstrap,
            mtry,
            minNodeSizeSpt,
            minNodeSizeAvg,
            minNodeSizeToSplitSpt,
            minNodeSizeToSplitAvg,
            minSplitGain,
            maxDepth,
            interactionDepth,
            seed,
            nthread,
            verbose,
            splitMiddle,
            maxObs,
            minTreesPerFold,
            foldSize,
            hasNas,
            naDirection,
            linear,
            overfitPenalty,
            doubleTree
        ));
    
        if (verbose) {
            std::cout << forest << std::endl;
            forest->getForest()->at(0)->printTree();
        }
        return forest;
    }
    
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
        bool hier_shrinkage,
        double lambda_shrinkage
    ) {   
        DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);
    
        forest->setDataframe(dta_frame);
    
        std::unique_ptr< std::vector<double> > testForestPrediction;
    
        // Create Data
        size_t ncol = dta_frame->getNumColumns();
        std::vector< std::vector<double> >* predi_data {
                new std::vector< std::vector<double> >(ncol, std::vector<double>(num_test_rows))
        };
    
        for (size_t j = 0; j < ncol; j++) {
            for (size_t i = 0; i < num_test_rows; i++){
                predi_data->at(j).at(i) = test_data[i*ncol + j];
            }
        }
    
    
        // Initialize the weightMatrix, terminalNodes, coefficients
        arma::Mat<double> weightMatrix;
        arma::Mat<int> terminalNodes;
        arma::Mat<double> coefficients;
    
        // tree_weights vector
        std::vector<size_t>* weights (
                new std::vector<size_t>(forest->getNtree())
        );
    
        for (size_t i = 0; i < forest->getNtree(); i++){
            weights->at(i) = tree_weights[i];
        }
     
    
        if (returnWeightMatrix) {
    
            weightMatrix.zeros(num_test_rows, dta_frame->getNumRows());
            testForestPrediction = forest->predict(
                predi_data,
                &weightMatrix,
                nullptr,
                nullptr,
                seed,
                nthread,
                exact,
                false,
                nullptr,
                hier_shrinkage,
                lambda_shrinkage
            );
    
            size_t idx = 0;
            for (size_t i = 0; i < num_test_rows; i++){
                for (size_t j = 0; j < dta_frame->getNumRows(); j++){
                    weight_matrix[idx] = weightMatrix(i,j);
                    idx++;
                }
            }
            
        } else if (linear) {
    
            coefficients.zeros(dta_frame->getNumRows(), dta_frame->getLinCols()->size() + 1);
            testForestPrediction = forest->predict(
                predi_data,
                nullptr,
                &coefficients,
                nullptr,
                seed,
                nthread,
                exact,
                false,
                nullptr
            );
    
            size_t idx = 0;
            for (size_t i = 0; i < dta_frame->getNumRows(); i++){
                for (size_t j = 0; j < dta_frame->getLinCols()->size() + 1; j++){
                    coefs[idx] = coefficients(i,j);
                    idx++;
                }
            }
    
        } else {
    
            testForestPrediction = forest->predict(
                predi_data,
                nullptr,
                nullptr,
                nullptr,
                seed,
                nthread,
                exact,
                use_weights,
                weights,
                hier_shrinkage,
                lambda_shrinkage
            );
    
        }
    
        delete(predi_data);
        delete(weights);
    
    
        predictions = *testForestPrediction.get();
    }
    
    
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
    ) {
        if (verbose)
            std::cout << forest << std::endl;
    
        DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);
        forest->setDataframe(dta_frame);
    
        // For now put trainingIdx empty
        std::vector<size_t> training_idx_use = training_idx;
    
        //Create Data
        size_t ncol = dta_frame->getNumColumns();
        std::vector< std::vector<double> >* predi_data {
                new std::vector< std::vector<double> >(ncol, std::vector<double>(dta_frame->getNumRows()))
        };
    
        for (size_t j = 0; j < ncol; j++) {
            for (size_t i = 0; i < dta_frame->getNumRows(); i++){
                predi_data->at(j).at(i) = test_data[i*ncol + j];
            }
        }
    
        // Initialize the weightMatrix
        arma::Mat<double> weightMatrix;
    
        if (returnWeightMatrix) {
            weightMatrix.zeros(dta_frame->getNumRows(), dta_frame->getNumRows());
            std::vector<size_t> treeCounts(dta_frame->getNumRows(), 0);

            predictions = forest->predictOOB(
                predi_data,
                &weightMatrix,
                &treeCounts,
                doubleOOB,
                exact,
                training_idx_use,
                hier_shrinkage,
                lambda_shrinkage
            );
    
            size_t idx = 0;
            for (size_t i = 0; i < dta_frame->getNumRows(); i++){
                for (size_t j = 0; j < dta_frame->getNumRows(); j++){
                    weight_matrix[idx] = weightMatrix(i,j);
                    idx++;
                }
            }
        }
    
        else {
            predictions = forest->predictOOB(
                predi_data,
                nullptr,
                nullptr,
                doubleOOB,
                exact,
                training_idx_use,
                hier_shrinkage,
                lambda_shrinkage
            );
        }
    
        delete(predi_data);
    
    }
    
    void fill_tree_info(
        forestry* forest,
        int tree_idx,
        std::vector<double>& treeInfo,
        std::vector<int>& split_info,
        std::vector<int>& av_info
    ) {
    
        std::unique_ptr<tree_info> info_holder;
    
        info_holder = forest->getForest()->at(tree_idx)->getTreeInfo(forest->getTrainingData());
        int num_nodes = forest->getForest()->at(tree_idx)->getNodeCount();
        for (int i = 0; i < num_nodes; i++) {
            treeInfo[i] = (double)info_holder->var_id.at(i);
        }
    
        for (int i = 0; i < num_nodes; i++) {
            treeInfo[num_nodes + i] = info_holder->values.at(i);
        }
    
        for (int i = 0; i < num_nodes; i++) {
            treeInfo[num_nodes *2 + i] = info_holder->split_val.at(i);
            treeInfo[num_nodes *3 + i] = (double)info_holder->naLeftCount.at(i);
            treeInfo[num_nodes *4 + i] = (double)info_holder->naRightCount.at(i);
            treeInfo[num_nodes *5 + i] = (double)info_holder->naDefaultDirection.at(i);
            treeInfo[num_nodes *6 + i] = (double)info_holder->average_count.at(i);
            treeInfo[num_nodes *7 + i] = (double)info_holder->split_count.at(i);
        }
    
        // Populate splitting samples for the tree
        size_t splitSize = info_holder->splittingSampleIndex.size();
        split_info[0] = splitSize;
        for (size_t i = 0; i < splitSize; i++){
            split_info[i+1] = info_holder->splittingSampleIndex.at(i);
        }
    
        // Populate averaging samples for the tree
        size_t avSize = info_holder->averagingSampleIndex.size();
        av_info[0] = avSize;
        for (size_t i = 0; i < avSize; i++){
            av_info[i+1] = info_holder->averagingSampleIndex.at(i);
        }
    
        treeInfo[num_nodes *8] = info_holder->seed;
    }
    
    
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
    ) {
    
        // Do stuff
        DataFrame* df = reinterpret_cast<DataFrame* >(data_ptr);
        forestry* forest ( new (std::nothrow) forestry(
            df,
            0,
            replace,
            sampSize,
            splitRatio,
            OOBhonest,
            doubleBootstrap,
            mtry,
            minNodeSizeSpt,
            minNodeSizeAvg,
            minNodeSizeToSplitSpt,
            minNodeSizeToSplitAvg,
            minSplitGain,
            maxDepth,
            interactionDepth,
            seed,
            nthread,
            verbose,
            splitMiddle,
            maxObs,
            minTreesPerFold,
            foldSize,
            hasNas,
            naDirection,
            linear,
            overfitPenalty,
            doubleTree
        ));
    
        std::vector<size_t>* categoricalColumns = df->getCatCols();
    
        std::unique_ptr< std::vector<size_t> > categoricalFeatureCols_copy(
          new std::vector<size_t>
        );
        for (size_t i = 0; i < categoricalColumns->size(); i++){
            categoricalFeatureCols_copy->push_back(categoricalColumns->at(i));
        }
        
    
        // Decode the forest data and create appropriate pointers
        std::unique_ptr< std::vector< std::vector<int> > > var_ids(
          new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<int> > > average_counts(
          new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<int> > > split_counts(
          new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<double> > > split_vals(
            new std::vector< std::vector<double> >
        );
        std::unique_ptr< std::vector< std::vector<int> > > naLeftCounts(
            new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<int> > > naRightCounts(
            new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<int> > > naDefaultDirections(
                new std::vector< std::vector<int> >
        );
        std::unique_ptr< std::vector< std::vector<size_t> > > averagingSampleIndex(
            new std::vector< std::vector<size_t> >
        );
        std::unique_ptr< std::vector< std::vector<size_t> > > splittingSampleIndex(
            new std::vector< std::vector<size_t> >
        );
        std::unique_ptr< std::vector< std::vector<size_t> > > excludedSampleIndex(
            new std::vector< std::vector<size_t> >
        );
        std::unique_ptr< std::vector<unsigned int> > treeSeeds(
            new std::vector<unsigned int>
        );
        std::unique_ptr< std::vector< std::vector<double> > > predictWeights(
            new std::vector< std::vector<double> >
        );
    
        // Reserve space for each of the vectors equal to ntree
        var_ids->reserve(ntree);
        average_counts->reserve(ntree);
        split_counts->reserve(ntree);
        split_vals->reserve(ntree);
        averagingSampleIndex->reserve(ntree);
        splittingSampleIndex->reserve(ntree);
        excludedSampleIndex->reserve(ntree);
        naLeftCounts->reserve(ntree);
        naRightCounts->reserve(ntree);
        naDefaultDirections->reserve(ntree);
        treeSeeds->reserve(ntree);
        predictWeights->reserve(ntree);
    
        // Now actually populate the vectors
        size_t ind = 0, ind_s = 0, ind_a = 0;
        for(size_t i = 0; i < ntree; i++){
            // Should be num total nodes
            std::vector<int> cur_var_ids((tree_counts[4*i]), 0);
            std::vector<int> cur_average_counts((tree_counts[4*i]), 0);
            std::vector<int> cur_split_counts((tree_counts[4*i]), 0);
            std::vector<double> cur_split_vals(tree_counts[4*i], 0);
            std::vector<int> curNaLeftCounts(tree_counts[4*i], 0);
            std::vector<int> curNaRightCounts(tree_counts[4*i], 0);
            std::vector<int> curNaDefaultDirections(tree_counts[4*i], 0);
            std::vector<size_t> curSplittingSampleIndex(tree_counts[4*i+1], 0);
            std::vector<size_t> curAveragingSampleIndex(tree_counts[4*i+2], 0);
            std::vector<double> cur_predict_weights(tree_counts[4*i], 0);
    
            for(size_t j = 0; j < tree_counts[4*i]; j++){
                cur_split_vals.at(j) = thresholds[ind];
                curNaLeftCounts.at(j) = na_left_count[ind];
                curNaRightCounts.at(j) = na_right_count[ind];
                curNaDefaultDirections.at(j) = na_default_directions[ind];
                cur_predict_weights.at(j) = predict_weights[ind];
                cur_var_ids.at(j) = features[ind];
                cur_average_counts.at(j) = features[ind];
                cur_split_counts.at(j) = features[ind];
    
                ind++;
            }
    
            for(size_t j = 0; j < tree_counts[4*i+1]; j++){
                curSplittingSampleIndex.at(j) = split_idx[ind_s];
                ind_s++;
            }
    
            for(size_t j = 0; j < tree_counts[4*i+2]; j++){
                curAveragingSampleIndex.at(j) = average_idx[ind_a];
                ind_a++;
            }
    
            var_ids->push_back(cur_var_ids);
            average_counts->push_back(cur_average_counts);
            split_counts->push_back(cur_split_counts);
            split_vals->push_back(cur_split_vals);
            naLeftCounts->push_back(curNaLeftCounts);
            naRightCounts->push_back(curNaRightCounts);
            naDefaultDirections->push_back(curNaDefaultDirections);
            splittingSampleIndex->push_back(curSplittingSampleIndex);
            averagingSampleIndex->push_back(curAveragingSampleIndex);
            excludedSampleIndex->push_back(std::vector<size_t>());
            predictWeights->push_back(cur_predict_weights);
            treeSeeds->push_back(tree_seeds[i]);
        }

        // call reconstructTrees
        forest->reconstructTrees(
            categoricalFeatureCols_copy,
            treeSeeds,
            var_ids,
            average_counts,
            split_counts,
            split_vals,
            naLeftCounts,
            naRightCounts,
            naDefaultDirections,
            averagingSampleIndex,
            splittingSampleIndex,
            excludedSampleIndex,
            predictWeights
        );
    
        return forest;
    
    }

    size_t get_node_count(forestry* forest, int tree_idx) {
        return(forest->getForest()->at(tree_idx)->getNodeCount());
    }
    
    size_t get_split_node_count(forestry* forest, int tree_idx) {
        return(forest->getForest()->at(tree_idx)->getSplitNodeCount());
    }
    
    size_t get_leaf_node_count(forestry* forest, int tree_idx) {
        return(forest->getForest()->at(tree_idx)->getLeafNodeCount());
    }
    
    void delete_forestry(forestry* forest, void* dataframe_pt) {
        delete(reinterpret_cast<DataFrame* >(dataframe_pt));
        delete(forest);
    }   
}

std::string export_json(forestry* forest, bool scale, const std::vector<double>& colSds, const std::vector<double>& colMeans) {
    return exportJson(*forest, scale, colSds, colMeans);
}
