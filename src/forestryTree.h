#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>

class forestryTree {

public:
  forestryTree();
  virtual ~forestryTree();

  forestryTree(
    DataFrame* trainingData,
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    double minSplitGain,
    size_t maxDepth,
    size_t interactionDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool hasNas,
    bool linear,
    double overfitPenalty,
    unsigned int seed
  );

  // This tree is only for testing purpose
  void setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    double minSplitGain,
    size_t maxDepth,
    size_t interactionDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    double overfitPenalty
  );

  void predict(
    std::vector<double> &outputPrediction,
    std::vector<int>* terminalNodes,
    std::vector< std::vector<double> >* xNew,
    DataFrame* trainingData,
    arma::Mat<double>* weightMatrix = NULL,
    bool linear = false,
    unsigned int seed = 44
  );

  std::unique_ptr<tree_info> getTreeInfo(
      DataFrame* trainingData
  );

  void reconstruct_tree(
      size_t mtry,
      size_t minNodeSizeSpt,
      size_t minNodeSizeAvg,
      size_t minNodeSizeToSplitSpt,
      size_t minNodeSizeToSplitAvg,
      double minSplitGain,
      size_t maxDepth,
      size_t interactionDepth,
      bool hasNas,
      bool linear,
      double overfitPenalty,
      std::vector<size_t> categoricalFeatureColsRcpp,
      std::vector<int> var_ids,
      std::vector<double> split_vals,
      std::vector<int> naLeftCounts,
      std::vector<int> naRightCounts,
      std::vector<size_t> leafAveidxs,
      std::vector<size_t> leafSplidxs,
      std::vector<size_t> averagingSampleIndex,
      std::vector<size_t> splittingSampleIndex);

  void recursive_reconstruction(
      RFNode* currentNode,
      std::vector<int> * var_ids,
      std::vector<double> * split_vals,
      std::vector<size_t> * leafAveidxs,
      std::vector<size_t> * leafSplidxs,
      std::vector<int> * naLeftCounts,
      std::vector<int> * naRightCounts
  );

  void recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    size_t depth,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    double overfitPenalty,
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal,
    bool monotone_splits,
    monotonic_info monotone_details
  );

  void selectBestFeature(
    size_t& bestSplitFeature,
    double& bestSplitValue,
    double& bestSplitLoss,
    arma::Mat<double> &bestSplitGL,
    arma::Mat<double> &bestSplitGR,
    arma::Mat<double> &bestSplitSL,
    arma::Mat<double> &bestSplitSR,
    std::vector<size_t>* featureList,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    double overfitPenalty,
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal,
    bool monotone_splits,
    monotonic_info &monotone_details
  );

  void initializelinear(
      DataFrame* trainingData,
      arma::Mat<double>& gTotal,
      arma::Mat<double>& sTotal,
      size_t numLinearFeatures,
      std::vector<size_t>* splitIndexes
  );

  void printTree();

  void trainTiming();

  void getOOBindex(
    std::vector<size_t> &outputOOBIndex,
    size_t nRows
  );

  void getOOBPrediction(
    std::vector<double> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData
  );

  void getShuffledOOBPrediction(
      std::vector<double> &outputOOBPrediction,
      std::vector<size_t> &outputOOBCount,
      DataFrame* trainingData,
      size_t shuffleFeature,
      std::mt19937_64& random_number_generator
  );

  size_t getMtry() {
    return _mtry;
  }

  size_t getMinNodeSizeSpt() {
    return _minNodeSizeSpt;
  }

  size_t getMinNodeSizeAvg() {
    return _minNodeSizeAvg;
  }

  size_t getMinNodeSizeToSplitSpt() {
    return _minNodeSizeToSplitSpt;
  }

  size_t getMinNodeSizeToSplitAvg() {
    return _minNodeSizeToSplitAvg;
  }

  double getMinSplitGain() {
    return _minSplitGain;
  }

  size_t getMaxDepth() {
    return _maxDepth;
  }

  size_t getInteractionDepth() {
    return _interactionDepth;
  }

  std::vector<size_t>* getSplittingIndex() {
    return _splittingSampleIndex.get();
  }

  std::vector<size_t>* getAveragingIndex() {
    return _averagingSampleIndex.get();
  }

  RFNode* getRoot() {
    return _root.get();
  }

  double getOverfitPenalty() {
    return _overfitPenalty;
  }

  unsigned int getSeed() {
    return _seed;
  }

  bool gethasNas() {
    return _hasNas;
  }

  void assignNodeId(size_t& node_i) {
    node_i = ++_nodeCount;
  }

  size_t getNodeCount() {
    return _nodeCount;
  }

private:
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  double _minSplitGain;
  size_t _maxDepth;
  size_t _interactionDepth;
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  std::unique_ptr< RFNode > _root;
  bool _hasNas;
  bool _linear;
  double _overfitPenalty;
  unsigned int _seed;
  size_t _nodeCount;
};


#endif //HTECPP_RFTREE_H
