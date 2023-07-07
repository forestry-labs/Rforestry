#ifndef HTECPP_RF_H
#define HTECPP_RF_H

#include "forestryTree.h"
#include "dataFrame.h"
#include "forestryTree.h"
#include "utils.h"
#include <armadillo>
#include <iostream>
#include <vector>
#include <string>


class forestry {

public:
  forestry();
  virtual ~forestry();

  forestry(
    DataFrame* trainingData,
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

  std::unique_ptr< std::vector<double> > predict(
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    arma::Mat<double>* coefficients,
    arma::Mat<int>* terminalNodes,
    unsigned int seed,
    size_t nthread,
    bool exact,
    bool use_weights,
    bool hier_shrinkage,
    double lambda_shrinkage,
    std::vector<size_t>* tree_weights
  );

  std::vector<double> predictOOB(
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    std::vector<size_t>* treeCounts,
    bool doubleOOB,
    bool exact,
    std::vector<size_t> &training_idx
  );

  void fillinTreeInfo(
      std::unique_ptr< std::vector< tree_info > > & forest_dta
  );

  void reconstructTrees(
      std::unique_ptr< std::vector<size_t> > & categoricalFeatureColsRcpp,
      std::unique_ptr< std::vector<unsigned int> > & tree_seeds,
      std::unique_ptr< std::vector< std::vector<int> >  > & var_ids,
      std::unique_ptr< std::vector< std::vector<int> >  > & average_counts,
      std::unique_ptr< std::vector< std::vector<double> >  > & split_vals,
      std::unique_ptr< std::vector< std::vector<int> >  > & naLeftCounts,
      std::unique_ptr< std::vector< std::vector<int> >  > & naRightCounts,
      std::unique_ptr< std::vector< std::vector<int> >  > & naDefaultDirections,
      std::unique_ptr< std::vector< std::vector<size_t> >  > & averagingSampleIndex,
      std::unique_ptr< std::vector< std::vector<size_t> >  > & splittingSampleIndex,
      std::unique_ptr< std::vector< std::vector<size_t> >  > & excludedSampleIndex,
      std::unique_ptr< std::vector< std::vector<double> >  > & weights,
      std::unique_ptr< std::vector< std::vector<double> >  > & weightsFull);

  size_t getTotalNodeCount();

  void calculateOOBError(
      bool doubleOOB = false
  );

  double getOOBError() {
    calculateOOBError();
    return _OOBError;
  }

  std::vector<double> getOOBpreds(
      bool doubleOOB = false
  ) {
    calculateOOBError(doubleOOB);
    return _OOBpreds;
  }

  void addTrees(size_t ntree);

  DataFrame* getTrainingData() {
    return _trainingData;
  }

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

  size_t getNtree() {
    return _ntree;
  }

  size_t getNtrain(){
    return (*_trainingData).getNumRows();
  }

  size_t getSampleSize() {
    // This is the sample size used for each tree in the bootstrap not ntrain
    return _sampSize;
  }

  double getSplitRatio() {
    return _splitRatio;
  }

  bool getOOBhonest() {
    return _OOBhonest;
  }

  bool getDoubleBootstrap() {
    return _doubleBootstrap;
  }

  bool isReplacement() {
    return _replace;
  }

  unsigned int getSeed() {
    return _seed;
  }

  std::vector< std::unique_ptr< forestryTree > >* getForest() {
    return _forest.get();
  }

  bool isVerbose() {
    return _verbose;
  }

  size_t getNthread(){
    return _nthread;
  }

  bool getSplitMiddle(){
    return _splitMiddle;
  }

  size_t getMaxObs() {
    return _maxObs;
  }

  size_t getminTreesPerFold() {
    return _minTreesPerFold;
  }

  size_t getFoldSize() {
      return _foldSize;
  }

  bool gethasNas() {
    return _hasNas;
  }

  bool getNaDirection() {
    return _naDirection;
  }

  bool getlinear() {
    return _linear;
  }

  double getOverfitPenalty() {
    return _overfitPenalty;
  }
  std::vector<std::vector<double>>* neighborhoodImpute(
      std::vector< std::vector<double> >* xNew,
      arma::Mat<double>* weightMatrix
  );

  void setDataframe(DataFrame * newDf) {
      _trainingData = std::move(newDf);
  }

private:
  DataFrame* _trainingData;
  size_t _ntree;
  bool _replace;
  size_t _sampSize;
  double _splitRatio;
  bool _OOBhonest;
  bool _doubleBootstrap;
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  double _minSplitGain;
  size_t _maxDepth;
  size_t _interactionDepth;
  std::unique_ptr< std::vector< std::unique_ptr< forestryTree > > > _forest;
  unsigned int _seed;
  bool _verbose;
  size_t _nthread;
  double _OOBError;
  std::vector<double> _OOBpreds;
  bool _splitMiddle;
  size_t _maxObs;
  size_t _minTreesPerFold;
  size_t _foldSize;
  bool _hasNas;
  bool _naDirection;
  bool _linear;
  double _overfitPenalty;
  bool _doubleTree;
};

#endif //HTECPP_RF_H
