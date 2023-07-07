#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "dataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <armadillo>

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
    std::unique_ptr< std::vector<size_t> > excludedSampleIndex,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool hasNas,
    bool naDirection,
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
    std::unique_ptr< std::vector<size_t> > excludedSampleIndex,
    double overfitPenalty
  );

  void predict(
    std::vector<double> &outputPrediction,
    std::vector<int>* terminalNodes,
    std::vector< std::vector<double> > &outputCoefficients,
    std::vector< std::vector<double> >* xNew,
    DataFrame* trainingData,
    arma::Mat<double>* weightMatrix = NULL,
    bool linear = false,
    bool naDirection = false,
    bool hier_shrinkage = false,
    double lambda_shrinkage = 0,
    unsigned int seed = 44,
    size_t nodesizeStrictAvg = 1,
    std::vector<size_t>* OOBIndex = NULL
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
      bool naDirection,
      bool linear,
      double overfitPenalty,
      unsigned int seed,
      std::vector<size_t> categoricalFeatureColsRcpp,
      std::vector<int> var_ids,
      std::vector<int> average_counts,
      std::vector<double> split_vals,
      std::vector<int> naLeftCounts,
      std::vector<int> naRightCounts,
      std::vector<int> naDefaultDirections,
      std::vector<size_t> averagingSampleIndex,
      std::vector<size_t> splittingSampleIndex,
      std::vector<size_t> excludedSampleIndex,
      std::vector<double> predictWeights,
      std::vector<double> predictWeightsFull);

  void recursive_reconstruction(
      RFNode* currentNode,
      std::vector<int> * var_ids,
      std::vector<int> * average_counts,
      std::vector<double> * split_vals,
      std::vector<int> * naLeftCounts,
      std::vector<int> * naRightCounts,
      std::vector<int> * naDefaultDirections,
      std::vector<double> * weights,
      std::vector<double> * weightsFull
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
    monotonic_info monotone_details,
    bool naDirection
  );

  void selectBestFeature(
      size_t &bestSplitFeature,
      double &bestSplitValue,
      double &bestSplitLoss,
      int &bestSplitNaDir,
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
    std::vector<size_t> &allIndex
  );

  void getDoubleOOBIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> &allIndex
  );

  void getOOBhonestIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> &allIndex
  );

  void getDoubleOOBIndexExcluded(
          std::vector<size_t> &outputOOBIndex,
          std::vector<size_t> &allIndex
  );

  void getOOBIndexExcluded(
          std::vector<size_t> &outputOOBIndex,
          std::vector<size_t> &allIndex
  );

  void getOOGIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> groupMemberships,
      std::vector<size_t> &allIndex,
      bool doubleOOB
  );

  void getOOBPrediction(
    std::vector<double> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData,
    bool OOBhonest,
    bool doubleOOB,
    size_t nodesizeStrictAvg,
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    const std::vector<size_t>& training_idx
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

  std::vector<size_t>* getExcludedIndex() {
    return _excludedSampleIndex.get();
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

  bool getNaDirection() {
    return _naDirection;
  }

  void assignNodeId(size_t& node_i,
                    bool split) {
    node_i = ++_nodeCount;
    if (split) {
        _splitNodeCount++;
    } else {
        _leafNodeCount++;
    }
  }

  size_t getNodeCount() {
    return _nodeCount;
  }

  size_t getSplitNodeCount() {
      return _splitNodeCount;
  }

  size_t getLeafNodeCount() {
      return _leafNodeCount;
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
  std::unique_ptr< std::vector<size_t> > _excludedSampleIndex;
  std::unique_ptr< RFNode > _root;
  bool _hasNas;
  bool _naDirection;
  bool _linear;
  double _overfitPenalty;
  unsigned int _seed;
  size_t _nodeCount;
  size_t _splitNodeCount;
  size_t _leafNodeCount;
};


#endif //HTECPP_RFTREE_H
