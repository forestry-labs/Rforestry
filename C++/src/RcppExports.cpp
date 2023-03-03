// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <armadillo>
#include <RcppThread.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_cppDataFrameInterface
SEXP rcpp_cppDataFrameInterface(Rcpp::List x, Rcpp::NumericVector y, Rcpp::NumericVector catCols, Rcpp::NumericVector linCols, int numRows, int numColumns, Rcpp::NumericVector featureWeights, Rcpp::NumericVector featureWeightsVariables, Rcpp::NumericVector deepFeatureWeights, Rcpp::NumericVector deepFeatureWeightsVariables, Rcpp::NumericVector observationWeights, Rcpp::NumericVector monotonicConstraints, Rcpp::NumericVector groupMemberships, bool monotoneAvg);
RcppExport SEXP _Rforestry_rcpp_cppDataFrameInterface(SEXP xSEXP, SEXP ySEXP, SEXP catColsSEXP, SEXP linColsSEXP, SEXP numRowsSEXP, SEXP numColumnsSEXP, SEXP featureWeightsSEXP, SEXP featureWeightsVariablesSEXP, SEXP deepFeatureWeightsSEXP, SEXP deepFeatureWeightsVariablesSEXP, SEXP observationWeightsSEXP, SEXP monotonicConstraintsSEXP, SEXP groupMembershipsSEXP, SEXP monotoneAvgSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type catCols(catColsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type linCols(linColsSEXP);
    Rcpp::traits::input_parameter< int >::type numRows(numRowsSEXP);
    Rcpp::traits::input_parameter< int >::type numColumns(numColumnsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeights(featureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeightsVariables(featureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeights(deepFeatureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeightsVariables(deepFeatureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type observationWeights(observationWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type monotonicConstraints(monotonicConstraintsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type groupMemberships(groupMembershipsSEXP);
    Rcpp::traits::input_parameter< bool >::type monotoneAvg(monotoneAvgSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppDataFrameInterface(x, y, catCols, linCols, numRows, numColumns, featureWeights, featureWeightsVariables, deepFeatureWeights, deepFeatureWeightsVariables, observationWeights, monotonicConstraints, groupMemberships, monotoneAvg));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_cppBuildInterface
SEXP rcpp_cppBuildInterface(Rcpp::List x, Rcpp::NumericVector y, Rcpp::NumericVector catCols, Rcpp::NumericVector linCols, int numRows, int numColumns, int ntree, bool replace, int sampsize, int mtry, double splitratio, bool OOBhonest, bool doubleBootstrap, int nodesizeSpl, int nodesizeAvg, int nodesizeStrictSpl, int nodesizeStrictAvg, double minSplitGain, int maxDepth, int interactionDepth, int seed, int nthread, bool verbose, bool middleSplit, int maxObs, Rcpp::NumericVector featureWeights, Rcpp::NumericVector featureWeightsVariables, Rcpp::NumericVector deepFeatureWeights, Rcpp::NumericVector deepFeatureWeightsVariables, Rcpp::NumericVector observationWeights, Rcpp::NumericVector monotonicConstraints, Rcpp::NumericVector groupMemberships, int minTreesPerFold, int foldSize, bool monotoneAvg, bool hasNas, bool naDirection, bool linear, double overfitPenalty, bool doubleTree, bool existing_dataframe_flag, SEXP existing_dataframe);
RcppExport SEXP _Rforestry_rcpp_cppBuildInterface(SEXP xSEXP, SEXP ySEXP, SEXP catColsSEXP, SEXP linColsSEXP, SEXP numRowsSEXP, SEXP numColumnsSEXP, SEXP ntreeSEXP, SEXP replaceSEXP, SEXP sampsizeSEXP, SEXP mtrySEXP, SEXP splitratioSEXP, SEXP OOBhonestSEXP, SEXP doubleBootstrapSEXP, SEXP nodesizeSplSEXP, SEXP nodesizeAvgSEXP, SEXP nodesizeStrictSplSEXP, SEXP nodesizeStrictAvgSEXP, SEXP minSplitGainSEXP, SEXP maxDepthSEXP, SEXP interactionDepthSEXP, SEXP seedSEXP, SEXP nthreadSEXP, SEXP verboseSEXP, SEXP middleSplitSEXP, SEXP maxObsSEXP, SEXP featureWeightsSEXP, SEXP featureWeightsVariablesSEXP, SEXP deepFeatureWeightsSEXP, SEXP deepFeatureWeightsVariablesSEXP, SEXP observationWeightsSEXP, SEXP monotonicConstraintsSEXP, SEXP groupMembershipsSEXP, SEXP minTreesPerFoldSEXP, SEXP foldSizeSEXP, SEXP monotoneAvgSEXP, SEXP hasNasSEXP, SEXP naDirectionSEXP, SEXP linearSEXP, SEXP overfitPenaltySEXP, SEXP doubleTreeSEXP, SEXP existing_dataframe_flagSEXP, SEXP existing_dataframeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type catCols(catColsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type linCols(linColsSEXP);
    Rcpp::traits::input_parameter< int >::type numRows(numRowsSEXP);
    Rcpp::traits::input_parameter< int >::type numColumns(numColumnsSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< int >::type sampsize(sampsizeSEXP);
    Rcpp::traits::input_parameter< int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< double >::type splitratio(splitratioSEXP);
    Rcpp::traits::input_parameter< bool >::type OOBhonest(OOBhonestSEXP);
    Rcpp::traits::input_parameter< bool >::type doubleBootstrap(doubleBootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeSpl(nodesizeSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeAvg(nodesizeAvgSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictSpl(nodesizeStrictSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictAvg(nodesizeStrictAvgSEXP);
    Rcpp::traits::input_parameter< double >::type minSplitGain(minSplitGainSEXP);
    Rcpp::traits::input_parameter< int >::type maxDepth(maxDepthSEXP);
    Rcpp::traits::input_parameter< int >::type interactionDepth(interactionDepthSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type middleSplit(middleSplitSEXP);
    Rcpp::traits::input_parameter< int >::type maxObs(maxObsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeights(featureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeightsVariables(featureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeights(deepFeatureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeightsVariables(deepFeatureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type observationWeights(observationWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type monotonicConstraints(monotonicConstraintsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type groupMemberships(groupMembershipsSEXP);
    Rcpp::traits::input_parameter< int >::type minTreesPerFold(minTreesPerFoldSEXP);
    Rcpp::traits::input_parameter< int >::type foldSize(foldSizeSEXP);
    Rcpp::traits::input_parameter< bool >::type monotoneAvg(monotoneAvgSEXP);
    Rcpp::traits::input_parameter< bool >::type hasNas(hasNasSEXP);
    Rcpp::traits::input_parameter< bool >::type naDirection(naDirectionSEXP);
    Rcpp::traits::input_parameter< bool >::type linear(linearSEXP);
    Rcpp::traits::input_parameter< double >::type overfitPenalty(overfitPenaltySEXP);
    Rcpp::traits::input_parameter< bool >::type doubleTree(doubleTreeSEXP);
    Rcpp::traits::input_parameter< bool >::type existing_dataframe_flag(existing_dataframe_flagSEXP);
    Rcpp::traits::input_parameter< SEXP >::type existing_dataframe(existing_dataframeSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppBuildInterface(x, y, catCols, linCols, numRows, numColumns, ntree, replace, sampsize, mtry, splitratio, OOBhonest, doubleBootstrap, nodesizeSpl, nodesizeAvg, nodesizeStrictSpl, nodesizeStrictAvg, minSplitGain, maxDepth, interactionDepth, seed, nthread, verbose, middleSplit, maxObs, featureWeights, featureWeightsVariables, deepFeatureWeights, deepFeatureWeightsVariables, observationWeights, monotonicConstraints, groupMemberships, minTreesPerFold, foldSize, monotoneAvg, hasNas, naDirection, linear, overfitPenalty, doubleTree, existing_dataframe_flag, existing_dataframe));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_cppPredictInterface
Rcpp::List rcpp_cppPredictInterface(SEXP forest, Rcpp::List x, std::string aggregation, int seed, int nthread, bool exact, bool returnWeightMatrix, bool use_weights, bool use_hold_out_idx, Rcpp::NumericVector tree_weights, Rcpp::IntegerVector hold_out_idx);
RcppExport SEXP _Rforestry_rcpp_cppPredictInterface(SEXP forestSEXP, SEXP xSEXP, SEXP aggregationSEXP, SEXP seedSEXP, SEXP nthreadSEXP, SEXP exactSEXP, SEXP returnWeightMatrixSEXP, SEXP use_weightsSEXP, SEXP use_hold_out_idxSEXP, SEXP tree_weightsSEXP, SEXP hold_out_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< std::string >::type aggregation(aggregationSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    Rcpp::traits::input_parameter< bool >::type exact(exactSEXP);
    Rcpp::traits::input_parameter< bool >::type returnWeightMatrix(returnWeightMatrixSEXP);
    Rcpp::traits::input_parameter< bool >::type use_weights(use_weightsSEXP);
    Rcpp::traits::input_parameter< bool >::type use_hold_out_idx(use_hold_out_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type tree_weights(tree_weightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type hold_out_idx(hold_out_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppPredictInterface(forest, x, aggregation, seed, nthread, exact, returnWeightMatrix, use_weights, use_hold_out_idx, tree_weights, hold_out_idx));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_OBBPredictInterface
double rcpp_OBBPredictInterface(SEXP forest);
RcppExport SEXP _Rforestry_rcpp_OBBPredictInterface(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_OBBPredictInterface(forest));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_OBBPredictionsInterface
Rcpp::List rcpp_OBBPredictionsInterface(SEXP forest, Rcpp::List x, bool existing_df, bool doubleOOB, bool returnWeightMatrix, bool exact, bool use_training_idx, Rcpp::IntegerVector training_idx);
RcppExport SEXP _Rforestry_rcpp_OBBPredictionsInterface(SEXP forestSEXP, SEXP xSEXP, SEXP existing_dfSEXP, SEXP doubleOOBSEXP, SEXP returnWeightMatrixSEXP, SEXP exactSEXP, SEXP use_training_idxSEXP, SEXP training_idxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< bool >::type existing_df(existing_dfSEXP);
    Rcpp::traits::input_parameter< bool >::type doubleOOB(doubleOOBSEXP);
    Rcpp::traits::input_parameter< bool >::type returnWeightMatrix(returnWeightMatrixSEXP);
    Rcpp::traits::input_parameter< bool >::type exact(exactSEXP);
    Rcpp::traits::input_parameter< bool >::type use_training_idx(use_training_idxSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type training_idx(training_idxSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_OBBPredictionsInterface(forest, x, existing_df, doubleOOB, returnWeightMatrix, exact, use_training_idx, training_idx));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_getObservationSizeInterface
double rcpp_getObservationSizeInterface(SEXP df);
RcppExport SEXP _Rforestry_rcpp_getObservationSizeInterface(SEXP dfSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type df(dfSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_getObservationSizeInterface(df));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_AddTreeInterface
void rcpp_AddTreeInterface(SEXP forest, int ntree);
RcppExport SEXP _Rforestry_rcpp_AddTreeInterface(SEXP forestSEXP, SEXP ntreeSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    rcpp_AddTreeInterface(forest, ntree);
    return R_NilValue;
END_RCPP
}
// rcpp_CppToR_translator
Rcpp::List rcpp_CppToR_translator(SEXP forest);
RcppExport SEXP _Rforestry_rcpp_CppToR_translator(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_CppToR_translator(forest));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_reconstructree
Rcpp::List rcpp_reconstructree(Rcpp::List x, Rcpp::NumericVector y, Rcpp::NumericVector catCols, Rcpp::NumericVector linCols, int numRows, int numColumns, Rcpp::List R_forest, bool replace, int sampsize, double splitratio, bool OOBhonest, bool doubleBootstrap, int mtry, int nodesizeSpl, int nodesizeAvg, int nodesizeStrictSpl, int nodesizeStrictAvg, double minSplitGain, int maxDepth, int interactionDepth, int seed, int nthread, bool verbose, bool middleSplit, int maxObs, int minTreesPerFold, Rcpp::NumericVector featureWeights, Rcpp::NumericVector featureWeightsVariables, Rcpp::NumericVector deepFeatureWeights, Rcpp::NumericVector deepFeatureWeightsVariables, Rcpp::NumericVector observationWeights, Rcpp::NumericVector monotonicConstraints, Rcpp::NumericVector groupMemberships, bool monotoneAvg, bool hasNas, bool naDirection, bool linear, double overfitPenalty, bool doubleTree);
RcppExport SEXP _Rforestry_rcpp_reconstructree(SEXP xSEXP, SEXP ySEXP, SEXP catColsSEXP, SEXP linColsSEXP, SEXP numRowsSEXP, SEXP numColumnsSEXP, SEXP R_forestSEXP, SEXP replaceSEXP, SEXP sampsizeSEXP, SEXP splitratioSEXP, SEXP OOBhonestSEXP, SEXP doubleBootstrapSEXP, SEXP mtrySEXP, SEXP nodesizeSplSEXP, SEXP nodesizeAvgSEXP, SEXP nodesizeStrictSplSEXP, SEXP nodesizeStrictAvgSEXP, SEXP minSplitGainSEXP, SEXP maxDepthSEXP, SEXP interactionDepthSEXP, SEXP seedSEXP, SEXP nthreadSEXP, SEXP verboseSEXP, SEXP middleSplitSEXP, SEXP maxObsSEXP, SEXP minTreesPerFoldSEXP, SEXP featureWeightsSEXP, SEXP featureWeightsVariablesSEXP, SEXP deepFeatureWeightsSEXP, SEXP deepFeatureWeightsVariablesSEXP, SEXP observationWeightsSEXP, SEXP monotonicConstraintsSEXP, SEXP groupMembershipsSEXP, SEXP monotoneAvgSEXP, SEXP hasNasSEXP, SEXP naDirectionSEXP, SEXP linearSEXP, SEXP overfitPenaltySEXP, SEXP doubleTreeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type catCols(catColsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type linCols(linColsSEXP);
    Rcpp::traits::input_parameter< int >::type numRows(numRowsSEXP);
    Rcpp::traits::input_parameter< int >::type numColumns(numColumnsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type R_forest(R_forestSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< int >::type sampsize(sampsizeSEXP);
    Rcpp::traits::input_parameter< double >::type splitratio(splitratioSEXP);
    Rcpp::traits::input_parameter< bool >::type OOBhonest(OOBhonestSEXP);
    Rcpp::traits::input_parameter< bool >::type doubleBootstrap(doubleBootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeSpl(nodesizeSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeAvg(nodesizeAvgSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictSpl(nodesizeStrictSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictAvg(nodesizeStrictAvgSEXP);
    Rcpp::traits::input_parameter< double >::type minSplitGain(minSplitGainSEXP);
    Rcpp::traits::input_parameter< int >::type maxDepth(maxDepthSEXP);
    Rcpp::traits::input_parameter< int >::type interactionDepth(interactionDepthSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type middleSplit(middleSplitSEXP);
    Rcpp::traits::input_parameter< int >::type maxObs(maxObsSEXP);
    Rcpp::traits::input_parameter< int >::type minTreesPerFold(minTreesPerFoldSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeights(featureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type featureWeightsVariables(featureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeights(deepFeatureWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type deepFeatureWeightsVariables(deepFeatureWeightsVariablesSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type observationWeights(observationWeightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type monotonicConstraints(monotonicConstraintsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type groupMemberships(groupMembershipsSEXP);
    Rcpp::traits::input_parameter< bool >::type monotoneAvg(monotoneAvgSEXP);
    Rcpp::traits::input_parameter< bool >::type hasNas(hasNasSEXP);
    Rcpp::traits::input_parameter< bool >::type naDirection(naDirectionSEXP);
    Rcpp::traits::input_parameter< bool >::type linear(linearSEXP);
    Rcpp::traits::input_parameter< double >::type overfitPenalty(overfitPenaltySEXP);
    Rcpp::traits::input_parameter< bool >::type doubleTree(doubleTreeSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_reconstructree(x, y, catCols, linCols, numRows, numColumns, R_forest, replace, sampsize, splitratio, OOBhonest, doubleBootstrap, mtry, nodesizeSpl, nodesizeAvg, nodesizeStrictSpl, nodesizeStrictAvg, minSplitGain, maxDepth, interactionDepth, seed, nthread, verbose, middleSplit, maxObs, minTreesPerFold, featureWeights, featureWeightsVariables, deepFeatureWeights, deepFeatureWeightsVariables, observationWeights, monotonicConstraints, groupMemberships, monotoneAvg, hasNas, naDirection, linear, overfitPenalty, doubleTree));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_cppImputeInterface
std::vector< std::vector<double> > rcpp_cppImputeInterface(SEXP forest, Rcpp::List x, int seed);
RcppExport SEXP _Rforestry_rcpp_cppImputeInterface(SEXP forestSEXP, SEXP xSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppImputeInterface(forest, x, seed));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_Rforestry_rcpp_cppDataFrameInterface", (DL_FUNC) &_Rforestry_rcpp_cppDataFrameInterface, 14},
    {"_Rforestry_rcpp_cppBuildInterface", (DL_FUNC) &_Rforestry_rcpp_cppBuildInterface, 42},
    {"_Rforestry_rcpp_cppPredictInterface", (DL_FUNC) &_Rforestry_rcpp_cppPredictInterface, 11},
    {"_Rforestry_rcpp_OBBPredictInterface", (DL_FUNC) &_Rforestry_rcpp_OBBPredictInterface, 1},
    {"_Rforestry_rcpp_OBBPredictionsInterface", (DL_FUNC) &_Rforestry_rcpp_OBBPredictionsInterface, 8},
    {"_Rforestry_rcpp_getObservationSizeInterface", (DL_FUNC) &_Rforestry_rcpp_getObservationSizeInterface, 1},
    {"_Rforestry_rcpp_AddTreeInterface", (DL_FUNC) &_Rforestry_rcpp_AddTreeInterface, 2},
    {"_Rforestry_rcpp_CppToR_translator", (DL_FUNC) &_Rforestry_rcpp_CppToR_translator, 1},
    {"_Rforestry_rcpp_reconstructree", (DL_FUNC) &_Rforestry_rcpp_reconstructree, 39},
    {"_Rforestry_rcpp_cppImputeInterface", (DL_FUNC) &_Rforestry_rcpp_cppImputeInterface, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_Rforestry(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}