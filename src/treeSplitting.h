#ifndef FORESTRYCPP_TREESPLIT_H
#define FORESTRYCPP_TREESPLIT_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include "forestryTree.h"
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>

double calculateRSS(
        DataFrame* trainingData,
        std::vector<size_t>* splittingSampleIndex,
        double overfitPenalty,
        std::mt19937_64& random_number_generator
);

void updateBestSplit(
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        double currentSplitLoss,
        double currentSplitValue,
        size_t currentFeature,
        size_t bestSplitTableIndex,
        std::mt19937_64& random_number_generator
);

void updateBestSplitS(
        arma::Mat<double> &bestSplitSL,
        arma::Mat<double> &bestSplitSR,
        const arma::Mat<double> &sTotal,
        DataFrame* trainingData,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitFeature,
        double bestSplitValue
);

void updateBestSplitG(
        arma::Mat<double> &bestSplitGL,
        arma::Mat<double> &bestSplitGR,
        const arma::Mat<double> &gTotal,
        DataFrame* trainingData,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitFeature,
        double bestSplitValue
);

void updateAArmadillo(
        arma::Mat<double>& a_k,
        arma::Mat<double>& new_x,
        bool leftNode
);

void updateSkArmadillo(
        arma::Mat<double>& s_k,
        arma::Mat<double>& next,
        double next_y,
        bool left
);

double computeRSSArmadillo(
        arma::Mat<double>& A_r,
        arma::Mat<double>& A_l,
        arma::Mat<double>& S_r,
        arma::Mat<double>& S_l,
        arma::Mat<double>& G_r,
        arma::Mat<double>& G_l
);

void updateRSSComponents(
        DataFrame* trainingData,
        size_t nextIndex,
        arma::Mat<double>& aLeft,
        arma::Mat<double>& aRight,
        arma::Mat<double>& sLeft,
        arma::Mat<double>& sRight,
        arma::Mat<double>& gLeft,
        arma::Mat<double>& gRight,
        arma::Mat<double>& crossingObservation,
        arma::Mat<double>& obOuter
);

void initializeRSSComponents(
        DataFrame* trainingData,
        size_t index,
        size_t numLinearFeatures,
        double overfitPenalty,
        const arma::Mat<double>& gTotal,
        const arma::Mat<double>& sTotal,
        arma::Mat<double>& aLeft,
        arma::Mat<double>& aRight,
        arma::Mat<double>& sLeft,
        arma::Mat<double>& sRight,
        arma::Mat<double>& gLeft,
        arma::Mat<double>& gRight,
        arma::Mat<double>& crossingObservation
);

double calcMuBarVar(
        // Calculates proxy for MSE of potential split
        double leftSum, size_t leftCount,
        double totalSum, size_t totalCount
);

void findBestSplitRidgeCategorical(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        double overfitPenalty,
        std::shared_ptr< arma::Mat<double> > gtotal,
        std::shared_ptr< arma::Mat<double> > stotal
);

void findBestSplitValueCategorical(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        size_t maxObs
);

void findBestSplitRidge(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        bool splitMiddle,
        size_t maxObs,
        double overfitPenalty,
        std::shared_ptr< arma::Mat<double> > gtotal,
        std::shared_ptr< arma::Mat<double> > stotal
);

void findBestSplitValueNonCategorical(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        bool splitMiddle,
        size_t maxObs,
        bool monotone_splits,
        monotonic_info monotone_details
);

void findBestSplitImpute(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        bool splitMiddle,
        size_t maxObs,
        bool monotone_splits,
        monotonic_info monotone_details
);

void findBestSplitImputeCategorical(
        std::vector<size_t>* averagingSampleIndex,
        std::vector<size_t>* splittingSampleIndex,
        size_t bestSplitTableIndex,
        size_t currentFeature,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        DataFrame* trainingData,
        size_t splitNodeSize,
        size_t averageNodeSize,
        std::mt19937_64& random_number_generator,
        size_t maxObs
);

void determineBestSplit(
        size_t &bestSplitFeature,
        double &bestSplitValue,
        double &bestSplitLoss,
        size_t mtry,
        double* bestSplitLossAll,
        double* bestSplitValueAll,
        size_t* bestSplitFeatureAll,
        size_t* bestSplitCountAll,
        std::mt19937_64& random_number_generator
);

bool acceptMonotoneSplit(
    monotonic_info &monotone_details,
    size_t currentFeature,
    float leftPartitionMean,
    float rightPartitionMean
);

float calculateMonotonicBound(
    float node_mean,
    monotonic_info& monotone_details
);

#endif //FORESTRYCPP_TREESPLIT_H
