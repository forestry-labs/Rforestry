// [[Rcpp::plugins(cpp11)]]
#include "dataFrame.h"
#include "forestryTree.h"
#include "RFNode.h"
#include "forestry.h"
#include "utils.h"
#include <RcppArmadillo.h>

void freeforestry(
  SEXP ptr
){
  if (NULL == R_ExternalPtrAddr(ptr))
    return;
  forestry* pm = (forestry*)(R_ExternalPtrAddr(ptr));
  delete(pm);
  R_ClearExternalPtr(ptr);
}

// [[Rcpp::export]]
SEXP rcpp_cppDataFrameInterface(
    Rcpp::List x,
    Rcpp::NumericVector y,
    Rcpp::NumericVector catCols,
    Rcpp::NumericVector linCols,
    int numRows,
    int numColumns,
    Rcpp::NumericVector featureWeights,
    Rcpp::NumericVector featureWeightsVariables,
    Rcpp::NumericVector deepFeatureWeights,
    Rcpp::NumericVector deepFeatureWeightsVariables,
    Rcpp::NumericVector observationWeights,
    Rcpp::List customSplitSample,
    Rcpp::List customAvgSample,
    Rcpp::List customExcludeSample,
    Rcpp::NumericVector monotonicConstraints,
    Rcpp::NumericVector groupMemberships,
    bool monotoneAvg
){

  try {
    std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
        new std::vector< std::vector<double> >(
            Rcpp::as< std::vector< std::vector<double> > >(x)
        )
    );

    std::unique_ptr< std::vector<double> > outcomeDataRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(y)
        )
    );

    std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
        new std::vector< size_t>(
            Rcpp::as< std::vector<size_t> >(catCols)
        )
    );

    std::unique_ptr< std::vector<size_t> > linearFeats (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(linCols)
        )
    );

    std::sort(linearFeats->begin(), linearFeats->end());

    std::unique_ptr< std::vector<double> > featureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(featureWeights)
        )
    );

    std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
        )
    );

    std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(deepFeatureWeights)
        )
    );

    std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
        )
    );

    std::unique_ptr< std::vector<double> > observationWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(observationWeights)
        )
    );

    std::unique_ptr<std::vector< std::vector<size_t> > > customSplitSampleRcpp (
        new std::vector< std::vector<size_t> >(
            Rcpp::as< std::vector< std::vector<size_t> > >(customSplitSample)
        )
    );

    std::unique_ptr<std::vector< std::vector<size_t> > > customAvgSampleRcpp (
        new std::vector< std::vector<size_t> >(
            Rcpp::as< std::vector< std::vector<size_t> > >(customAvgSample)
        )
    );

    std::unique_ptr<std::vector< std::vector<size_t> > > customExcludeSampleRcpp (
        new std::vector< std::vector<size_t> >(
            Rcpp::as< std::vector< std::vector<size_t> > >(customExcludeSample)
        )
    );

    std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
        new std::vector<int>(
            Rcpp::as< std::vector<int> >(monotonicConstraints)
        )
    );

    std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(groupMemberships)
        )
    );

    DataFrame* trainingData = new DataFrame(
        std::move(featureDataRcpp),
        std::move(outcomeDataRcpp),
        std::move(categoricalFeatureColsRcpp),
        std::move(linearFeats),
        (size_t) numRows,
        (size_t) numColumns,
        std::move(featureWeightsRcpp),
        std::move(featureWeightsVariablesRcpp),
        std::move(deepFeatureWeightsRcpp),
        std::move(deepFeatureWeightsVariablesRcpp),
        std::move(observationWeightsRcpp),
        std::move(customSplitSampleRcpp),
        std::move(customAvgSampleRcpp),
        std::move(customExcludeSampleRcpp),
        std::move(monotonicConstraintsRcpp),
        std::move(groupMembershipsRcpp),
        (bool) monotoneAvg
    );

    Rcpp::XPtr<DataFrame> ptr(trainingData, true) ;
    return ptr;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return NULL;
}


// [[Rcpp::export]]
SEXP rcpp_cppBuildInterface(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  Rcpp::NumericVector linCols,
  int numRows,
  int numColumns,
  int ntree,
  bool replace,
  int sampsize,
  int mtry,
  double splitratio,
  bool OOBhonest,
  bool doubleBootstrap,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  double minSplitGain,
  int maxDepth,
  int interactionDepth,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  Rcpp::NumericVector featureWeights,
  Rcpp::NumericVector featureWeightsVariables,
  Rcpp::NumericVector deepFeatureWeights,
  Rcpp::NumericVector deepFeatureWeightsVariables,
  Rcpp::NumericVector observationWeights,
  Rcpp::List customSplitSample,
  Rcpp::List customAvgSample,
  Rcpp::List customExcludeSample,
  Rcpp::NumericVector monotonicConstraints,
  Rcpp::NumericVector groupMemberships,
  int minTreesPerFold,
  int foldSize,
  bool monotoneAvg,
  bool hasNas,
  bool naDirection,
  bool linear,
  double overfitPenalty,
  bool doubleTree,
  bool existing_dataframe_flag,
  SEXP existing_dataframe
){

  if (existing_dataframe_flag) {

    try {
      Rcpp::XPtr< DataFrame > trainingData(existing_dataframe) ;

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (double) minSplitGain,
        (size_t) maxDepth,
        (size_t) interactionDepth,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        (size_t) minTreesPerFold,
        (size_t) foldSize,
        hasNas,
        naDirection,
        linear,
        (double) overfitPenalty,
        doubleTree
      );

      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }

  } else {

    try {
      std::unique_ptr< std::vector< std::vector<double> > > featureDataRcpp (
          new std::vector< std::vector<double> >(
              Rcpp::as< std::vector< std::vector<double> > >(x)
          )
      );

      std::unique_ptr< std::vector<double> > outcomeDataRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(y)
          )
      );

      std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(catCols)
          )
      );

      std::unique_ptr< std::vector<size_t> > linearFeats (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(linCols)
          )
      );

      std::sort(linearFeats->begin(), linearFeats->end());

      std::unique_ptr< std::vector<double> > featureWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(featureWeights)
          )
      );

      std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(deepFeatureWeights)
          )
      );


      std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
          )
      );

      std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
          )
      );

      std::unique_ptr< std::vector<double> > observationWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(observationWeights)
          )
      );

      std::unique_ptr<std::vector< std::vector<size_t> > > customSplitSampleRcpp (
          new std::vector< std::vector<size_t> >(
              Rcpp::as< std::vector< std::vector<size_t> > >(customSplitSample)
          )
      );

      std::unique_ptr<std::vector< std::vector<size_t> > > customAvgSampleRcpp (
          new std::vector< std::vector<size_t> >(
              Rcpp::as< std::vector< std::vector<size_t> > >(customAvgSample)
          )
      );

      std::unique_ptr<std::vector< std::vector<size_t> > > customExcludeSampleRcpp (
          new std::vector< std::vector<size_t> >(
              Rcpp::as< std::vector< std::vector<size_t> > >(customExcludeSample)
          )
      );

      std::unique_ptr< std::vector<int> > monotoneConstraintsRcpp (
          new std::vector<int>(
              Rcpp::as< std::vector<int> >(monotonicConstraints)
          )
      );

      std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(groupMemberships)
          )
      );

      DataFrame* trainingData = new DataFrame(
          std::move(featureDataRcpp),
          std::move(outcomeDataRcpp),
          std::move(categoricalFeatureColsRcpp),
          std::move(linearFeats),
          (size_t) numRows,
          (size_t) numColumns,
          std::move(featureWeightsRcpp),
          std::move(featureWeightsVariablesRcpp),
          std::move(deepFeatureWeightsRcpp),
          std::move(deepFeatureWeightsVariablesRcpp),
          std::move(observationWeightsRcpp),
          std::move(customSplitSampleRcpp),
          std::move(customAvgSampleRcpp),
          std::move(customExcludeSampleRcpp),
          std::move(monotoneConstraintsRcpp),
          std::move(groupMembershipsRcpp),
          (bool) monotoneAvg
      );

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (double) minSplitGain,
        (size_t) maxDepth,
        (size_t) interactionDepth,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        (size_t) minTreesPerFold,
        (size_t) foldSize,
        hasNas,
        naDirection,
        linear,
        (double) overfitPenalty,
        doubleTree
      );
      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;

    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  }
  return NULL;
}


// [[Rcpp::export]]
Rcpp::List rcpp_cppPredictInterface(
  SEXP forest,
  Rcpp::List x,
  std::string aggregation,
  int seed,
  int nthread,
  bool exact,
  bool returnWeightMatrix,
  bool use_weights,
  bool use_hold_out_idx,
  Rcpp::NumericVector tree_weights,
  Rcpp::IntegerVector hold_out_idx,
  bool hierShrinkage = false,
  double lambdaShrinkage = 0
){
  try {

    Rcpp::XPtr< forestry > testFullForest(forest) ;

    std::vector< std::vector<double> > featureData =
      Rcpp::as< std::vector< std::vector<double> > >(x);

    std::unique_ptr< std::vector<double> > testForestPrediction;
    // We always initialize the weightMatrix. If the aggregation is weightMatrix
    // then we inialize the empty weight matrix
    arma::Mat<double> weightMatrix;
    arma::Mat<int> terminalNodes;
    arma::Mat<double> coefficients;

    if (returnWeightMatrix) {
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testFullForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix.zeros(nrow, ncol);  // set it all to 0
    }


    // Have to keep track of tree_weights
    std::vector<size_t>* testForestTreeWeights;
    std::vector<size_t> weights;

    // If using predict indices, set weights according to them
    if (use_hold_out_idx) {
      std::vector<size_t> holdOutIdxCpp = Rcpp::as< std::vector<size_t> >(hold_out_idx);

      for (auto &tree : *(testFullForest->getForest())) {
        bool discard_tree = false;
        std::unordered_set<size_t> hold_out_set(holdOutIdxCpp.begin(), holdOutIdxCpp.end());
        for (const auto &averaging_index : *(tree->getAveragingIndex()) ) {
          if (hold_out_set.count(averaging_index)) {
            discard_tree = true;
            break;
          }
        }
        // if Still haven't found any of them, search splitting set
        if (!discard_tree) {
          for (const auto &splitting_index : *(tree->getSplittingIndex()) ) {
            if (hold_out_set.count(splitting_index)) {
              discard_tree = true;
              break;
            }
          }
        }
        if (discard_tree) {
          weights.push_back(0);
        } else {
          weights.push_back(1);
        }
      } // End tree loop
      // Tell forest to use the weights
      use_weights = true;
    } else {
      // If we have weights we want to initialize them.
      weights = Rcpp::as< std::vector<size_t> >(tree_weights);
    }

    // Make ptr to weights
    testForestTreeWeights =
      new std::vector<size_t> (weights);



    size_t threads_to_use;
    if (nthread == 0) {
      threads_to_use = testFullForest->getNthread();
    } else {
      threads_to_use = (size_t) nthread;
    }

    if (aggregation == "coefs") {
      size_t nrow = featureData[0].size();
      // Now we need the number of linear features + 1 for the intercept
      size_t ncol = (*testFullForest).getTrainingData()->getLinObsData(0).size() + 1;
      //Set coefficients to be zero
      coefficients.resize(nrow, ncol);
      coefficients.zeros(nrow, ncol);

      testForestPrediction = (*testFullForest).predict(&featureData,
                                                       NULL,
                                                       &coefficients,
                                                       NULL,
                                                       seed,
                                                       threads_to_use,
                                                       false,
                                                       false,
                                                       NULL);

    } else if (aggregation == "terminalNodes") {
      // In this case, we return both the terminal nodes, and the weightMatrix
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testFullForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix.zeros(nrow, ncol);  // set it all to 0

      // Don't make sparse matrix in C,
      // get indices for each observation/tree combo,
      // then parse the sparse form in R
      ncol = (*testFullForest).getNtree();  // Total nodes across the forest
      nrow = featureData[1].size()+1;   // Total feature.new observations
      // Bottom row is the total node count/tree


      terminalNodes.resize(nrow, ncol);
      terminalNodes.zeros(nrow, ncol);
      // The idea is that, if the weightMatrix is point to NULL it won't be
      // be updated, but otherwise it will be updated:
      testForestPrediction = (*testFullForest).predict(&featureData,
                                                       &weightMatrix,
                                                       NULL,
                                                       &terminalNodes,
                                                       seed,
                                                       threads_to_use,
                                                       exact,
                                                       false,
                                                       NULL,
                                                       hierShrinkage,
                                                       lambdaShrinkage);
    } else {
      // If the weights are zero, we just return NaN's
      if (use_weights &&
          (std::accumulate(testForestTreeWeights->begin(), testForestTreeWeights->end(), 0) == 0)) {

        testForestPrediction = std::unique_ptr< std::vector<double> >(
          new std::vector<double>(featureData[0].size(), std::numeric_limits<double>::quiet_NaN())
        );
      } else {
        testForestPrediction = (*testFullForest).predict(&featureData,
                                returnWeightMatrix ? &weightMatrix : NULL,
                                NULL,
                                NULL,
                                seed,
                                threads_to_use,
                                exact,
                                use_weights,
                                use_weights ? testForestTreeWeights : NULL,
                                hierShrinkage,
                                lambdaShrinkage);
      }
    }

    std::vector<double>* testForestPrediction_ =
      new std::vector<double>(*testForestPrediction.get());

    Rcpp::NumericVector predictions = Rcpp::wrap(*testForestPrediction_);

    delete testForestPrediction_;
    delete testForestTreeWeights;

    return Rcpp::List::create(Rcpp::Named("predictions") = predictions,
                              Rcpp::Named("weightMatrix") = weightMatrix,
                              Rcpp::Named("terminalNodes") = terminalNodes,
                              Rcpp::Named("coef") = coefficients);

    // return output;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
double rcpp_OBBPredictInterface(
    SEXP forest
){

  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    double OOBError = (*testFullForest).getOOBError();
    return OOBError;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}

// [[Rcpp::export]]
Rcpp::List rcpp_OBBPredictionsInterface(
    SEXP forest,
    Rcpp::List x,
    bool existing_df,
    bool doubleOOB,
    bool returnWeightMatrix,
    bool exact,
    bool use_training_idx,
    Rcpp::IntegerVector training_idx,
    bool hierShrinkage,
    double lambdaShrinkage
){
  // Then we predict with the feature.new data
  if (existing_df) {
    std::vector< std::vector<double> > featureData =
      Rcpp::as< std::vector< std::vector<double> > >(x);

    std::vector<size_t> training_idx_cpp;
    if (use_training_idx){
        training_idx_cpp = Rcpp::as< std::vector<size_t> >(training_idx);
    }

    try {
      Rcpp::XPtr< forestry > testFullForest(forest) ;

      arma::Mat<double> weightMatrix;
      std::vector<size_t> treeCounts(1);

      if (returnWeightMatrix) {
        size_t nrow = use_training_idx ? training_idx.size() : (*testFullForest).getNtrain(); // number of features to be predicted
        size_t ncol = (*testFullForest).getNtrain(); // number of train data
        weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
        weightMatrix.zeros(nrow, ncol);// set it all to 0
        treeCounts.resize(nrow);
        std::fill(treeCounts.begin(), treeCounts.end(), 0);

        std::vector<double> OOBpreds = (*testFullForest).predictOOB(&featureData,
                                        &weightMatrix,
                                        &treeCounts,
                                        doubleOOB,
                                        exact,
                                        training_idx_cpp,
                                        hierShrinkage,
                                        lambdaShrinkage);
        Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);

        return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds,
                                  Rcpp::Named("weightMatrix") = weightMatrix,
                                  Rcpp::Named("treeCounts") = treeCounts);
      } else {
        // If we don't need weightMatrix, don't return it
        std::vector<double> OOBpreds = (*testFullForest).predictOOB(&featureData,
                                        NULL,
                                        NULL,
                                        doubleOOB,
                                        exact,
                                        training_idx_cpp,
                                        hierShrinkage,
                                        lambdaShrinkage);
        Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);

        return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds);
      }

    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }

  // Otherwise we predict with just the in sample data
  } else {
    try {
      Rcpp::XPtr< forestry > testFullForest(forest) ;
      std::vector<double> OOBpreds = (*testFullForest).getOOBpreds(doubleOOB);
      Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);
      return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds);
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  }

  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
double rcpp_getObservationSizeInterface(
    SEXP df
){

  try {
    Rcpp::XPtr< DataFrame > trainingData(df) ;
    double nrows = (double) (*trainingData).getNumRows();
    return nrows;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}


// [[Rcpp::export]]
void rcpp_AddTreeInterface(
    SEXP forest,
    int ntree
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    (*testFullForest).addTrees(ntree);
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
}

// [[Rcpp::export]]
Rcpp::String rcpp_ExportJson(
    Rcpp::S4 forestParamR
){
  try {
    SEXP forestPointerR = forestParamR.slot("forest");
    Rcpp::XPtr<forestry> forestPtr(forestPointerR);

    Rcpp::NumericVector colSdsR = forestParamR.slot("colSd");
    std::vector<double> colSds(colSdsR.begin(), colSdsR.end());

    Rcpp::NumericVector colMeansR = forestParamR.slot("colMeans");
    std::vector<double> colMeans(colMeansR.begin(), colMeansR.end());

    return Rcpp::String(exportJson(*forestPtr, colSds, colMeans));

  } catch(const std::exception& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }

  return Rcpp::String();
}

// [[Rcpp::export]]
Rcpp::List rcpp_CppToR_translator(
    SEXP forest
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    std::unique_ptr< std::vector<tree_info> > forest_dta(
      new std::vector<tree_info>
    );
    (*testFullForest).fillinTreeInfo(forest_dta);

    // Return the lis of list. For each tree an element in the first list:
    Rcpp::List list_to_return;

    for(size_t i=0; i!=forest_dta->size(); i++){
      Rcpp::IntegerVector var_id = Rcpp::wrap(((*forest_dta)[i]).var_id);
      Rcpp::IntegerVector average_count = Rcpp::wrap(((*forest_dta)[i]).average_count);
      Rcpp::IntegerVector split_count = Rcpp::wrap(((*forest_dta)[i]).split_count);
      Rcpp::NumericVector split_val = Rcpp::wrap(((*forest_dta)[i]).split_val);
      Rcpp::IntegerVector averagingSampleIndex =
	      Rcpp::wrap(((*forest_dta)[i]).averagingSampleIndex);
      Rcpp::IntegerVector splittingSampleIndex =
	      Rcpp::wrap(((*forest_dta)[i]).splittingSampleIndex);
      Rcpp::IntegerVector excludedSampleIndex =
        Rcpp::wrap(((*forest_dta)[i]).excludedSampleIndex);
      Rcpp::IntegerVector naLeftCounts =
        Rcpp::wrap(((*forest_dta)[i]).naLeftCount);

      Rcpp::IntegerVector naRightCounts =
        Rcpp::wrap(((*forest_dta)[i]).naRightCount);

      Rcpp::IntegerVector naDefaultDirections =
        Rcpp::wrap(((*forest_dta)[i]).naDefaultDirection);

      Rcpp::NumericVector predictWeights =
              Rcpp::wrap(((*forest_dta)[i]).values);


        Rcpp::List list_i =
        Rcpp::List::create(
			   Rcpp::Named("var_id") = var_id,
         Rcpp::Named("average_count") = average_count,
         Rcpp::Named("split_count") = split_count,
			   Rcpp::Named("split_val") = split_val,
			   Rcpp::Named("averagingSampleIndex") = averagingSampleIndex,
			   Rcpp::Named("splittingSampleIndex") = splittingSampleIndex,
			   Rcpp::Named("excludedSampleIndex") = excludedSampleIndex,
			   Rcpp::Named("naLeftCounts") = naLeftCounts,
			   Rcpp::Named("naRightCounts") = naRightCounts,
			   Rcpp::Named("naDefaultDirections") = naDefaultDirections,
			   Rcpp::Named("seed") = (*forest_dta)[i].seed, // Add the seeds to the list we return
               Rcpp::Named("weights") = predictWeights
        );
      list_to_return.push_back(list_i);
    }
    return list_to_return;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::List rcpp_reconstructree(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  Rcpp::NumericVector linCols,
  int numRows,
  int numColumns,
  Rcpp::List R_forest,
  bool replace,
  int sampsize,
  double splitratio,
  bool OOBhonest,
  bool doubleBootstrap,
  int mtry,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  double minSplitGain,
  int maxDepth,
  int interactionDepth,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  int minTreesPerFold,
  Rcpp::NumericVector featureWeights,
  Rcpp::NumericVector featureWeightsVariables,
  Rcpp::NumericVector deepFeatureWeights,
  Rcpp::NumericVector deepFeatureWeightsVariables,
  Rcpp::NumericVector observationWeights,
  Rcpp::List customSplitSample,
  Rcpp::List customAvgSample,
  Rcpp::List customExcludeSample,
  Rcpp::NumericVector monotonicConstraints,
  Rcpp::NumericVector groupMemberships,
  bool monotoneAvg,
  bool hasNas,
  bool naDirection,
  bool linear,
  double overfitPenalty,
  bool doubleTree
){

  // Decode the R_forest data and create appropriate pointers to pointers:
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
      new  std::vector< std::vector<double> >
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
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > splittingSampleIndex(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > excludedSampleIndex(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector<unsigned int> > tree_seeds(
      new std::vector<unsigned int>
  );
  std::unique_ptr< std::vector< std::vector<double> > > predictWeights(
          new  std::vector< std::vector<double> >
  );

  // Reserve space for each of the vectors equal to R_forest.size()
  var_ids->reserve(R_forest.size());
  average_counts->reserve(R_forest.size());
  split_counts->reserve(R_forest.size());
  split_vals->reserve(R_forest.size());
  averagingSampleIndex->reserve(R_forest.size());
  splittingSampleIndex->reserve(R_forest.size());
  excludedSampleIndex->reserve(R_forest.size());
  naLeftCounts->reserve(R_forest.size());
  naRightCounts->reserve(R_forest.size());
  naDefaultDirections->reserve(R_forest.size());
  tree_seeds->reserve(R_forest.size());
  predictWeights->reserve(R_forest.size());


  // Now actually populate the vectors
  for(int i=0; i!=R_forest.size(); i++){
    var_ids->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[0])
      );
    average_counts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[1])
      );
    split_counts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[2])
      );
    split_vals->push_back(
        Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[3])
      );
    averagingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[4])
      );
    splittingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[5])
      );
    excludedSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[6])
    );
    naLeftCounts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[7])
    );
    naRightCounts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[8])
    );
    naDefaultDirections->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[9])
    );
    tree_seeds->push_back(
        Rcpp::as< unsigned int > ((Rcpp::as<Rcpp::List>(R_forest[i]))[10])
    );
    predictWeights->push_back(
            Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[11])
    );
  }

  // Decode catCols and R_forest
  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.

  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp_copy(
      new std::vector<size_t>
  );

  for(size_t i=0; i<(*categoricalFeatureColsRcpp).size(); i++){
    (*categoricalFeatureColsRcpp_copy).push_back(
        (*categoricalFeatureColsRcpp)[i]);
  }

  std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
      new std::vector< std::vector<double> >(
          Rcpp::as< std::vector< std::vector<double> > >(x)
      )
  );

  std::unique_ptr< std::vector<double> > outcomeDataRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(y)
      )
  );

  std::unique_ptr< std::vector<size_t> > linearFeats (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(linCols)
      )
  );

  std::sort(linearFeats->begin(), linearFeats->end());

  std::unique_ptr< std::vector<double> > featureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(featureWeights)
      )
  );

  std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
      )
  );

  std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(deepFeatureWeights)
      )
  );
  std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
      )
  );
  std::unique_ptr< std::vector<double> > observationWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(observationWeights)
      )
  );
  std::unique_ptr<std::vector< std::vector<size_t> > > customSplitSampleRcpp (
      new std::vector< std::vector<size_t> >(
          Rcpp::as< std::vector< std::vector<size_t> > >(customSplitSample)
      )
  );
  std::unique_ptr<std::vector< std::vector<size_t> > > customAvgSampleRcpp (
      new std::vector< std::vector<size_t> >(
          Rcpp::as< std::vector< std::vector<size_t> > >(customAvgSample)
      )
  );
  std::unique_ptr<std::vector< std::vector<size_t> > > customExcludeSampleRcpp (
      new std::vector< std::vector<size_t> >(
          Rcpp::as< std::vector< std::vector<size_t> > >(customExcludeSample)
      )
  );
  std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
      new std::vector<int>(
          Rcpp::as< std::vector<int> >(monotonicConstraints)
      )
  );
  std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(groupMemberships)
      )
  );

  DataFrame* trainingData = new DataFrame(
    std::move(featureDataRcpp),
    std::move(outcomeDataRcpp),
    std::move(categoricalFeatureColsRcpp),
    std::move(linearFeats),
    (size_t) numRows,
    (size_t) numColumns,
    std::move(featureWeightsRcpp),
    std::move(featureWeightsVariablesRcpp),
    std::move(deepFeatureWeightsRcpp),
    std::move(deepFeatureWeightsVariablesRcpp),
    std::move(observationWeightsRcpp),
    std::move(customSplitSampleRcpp),
    std::move(customAvgSampleRcpp),
    std::move(customExcludeSampleRcpp),
    std::move(monotonicConstraintsRcpp),
    std::move(groupMembershipsRcpp),
    (bool) monotoneAvg
  );

  forestry* testFullForest = new forestry(
    (DataFrame*) trainingData,
    (int) 0,
    (bool) replace,
    (int) sampsize,
    (double) splitratio,
    (bool) OOBhonest,
    (bool) doubleBootstrap,
    (int) mtry,
    (int) nodesizeSpl,
    (int) nodesizeAvg,
    (int) nodesizeStrictSpl,
    (int) nodesizeStrictAvg,
    (double) minSplitGain,
    (int) maxDepth,
    (int) interactionDepth,
    (unsigned int) seed,
    (int) nthread,
    (bool) verbose,
    (bool) middleSplit,
    (int) maxObs,
    (size_t) minTreesPerFold,
    1,
    (bool) hasNas,
    (bool) naDirection,
    (bool) linear,
    (double) overfitPenalty,
    doubleTree
  );

  testFullForest->reconstructTrees(categoricalFeatureColsRcpp_copy,
                                   tree_seeds,
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

  Rcpp::XPtr<forestry> ptr(testFullForest, true);
  R_RegisterCFinalizerEx(
    ptr,
    (R_CFinalizer_t) freeforestry,
    (Rboolean) TRUE
  );
  Rcpp::XPtr<DataFrame> df_ptr(trainingData, true) ;
  return Rcpp::List::create(Rcpp::Named("forest_ptr") = ptr,
                            Rcpp::Named("data_frame_ptr") = df_ptr);
}

// [[Rcpp::export]]
std::vector< std::vector<double> > rcpp_cppImputeInterface(
    SEXP forest,
    Rcpp::List x,
    int seed
){
  // There is code duplication with rcpp_cppPredictInterface here. Really the
  // predict member function should be refactored so that the boilerplate
  // happens inside it.
  Rcpp::XPtr< forestry > testFullForest(forest);
  std::vector< std::vector<double> > featureData =
    Rcpp::as< std::vector< std::vector<double> > >(x);

  std::unique_ptr< std::vector<double> > testForestPrediction;
  arma::Mat<double> weightMatrix;

  size_t nrow = featureData[0].size(); // number of features to be predicted
  size_t ncol = (*testFullForest).getNtrain(); // number of train data
  weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
  weightMatrix.zeros(nrow, ncol); // set it all to 0

  testForestPrediction = (*testFullForest).predict(&featureData,
                                                   &weightMatrix,
                                                   NULL,
                                                   NULL,
                                                   seed,
                                                   testFullForest->getNthread(),
                                                   false,
                                                   false,
                                                   NULL);

  std::vector<double>* testForestPrediction_ =
    new std::vector<double>(*testForestPrediction.get());

  Rcpp::NumericVector predictions = Rcpp::wrap(*testForestPrediction_);
  arma::Mat<double> weightMatrixT = weightMatrix;

  // Take tranpose because we want to access by row and armadillo uses column
  // major ordering.
  arma::inplace_trans(weightMatrixT);

  std::vector<std::vector<double>>* imputedX = testFullForest->neighborhoodImpute(
    &featureData,
    &weightMatrixT
  );
  return *imputedX;
}
