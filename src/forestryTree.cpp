#include "forestryTree.h"
#include "utils.h"
#include "treeSplitting.h"
#include <RcppArmadillo.h>
#include <cmath>
#include <set>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
// [[Rcpp::plugins(cpp11)]]

forestryTree::forestryTree():
  _mtry(0),
  _minNodeSizeSpt(0),
  _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0),
  _minNodeSizeToSplitAvg(0),
  _minSplitGain(0),
  _maxDepth(0),
  _interactionDepth(0),
  _averagingSampleIndex(nullptr),
  _splittingSampleIndex(nullptr),
  _root(nullptr) {};

forestryTree::~forestryTree() {};

forestryTree::forestryTree(
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
){
  /**
  * @brief Honest random forest tree constructor
  * @param trainingData    A DataFrame object
  * @param mtry    The total number of features to use for each split
  * @param minNodeSizeSpt    Minimum splitting size of leaf node
  * @param minNodeSizeAvg    Minimum averaging size of leaf node
  * @param minNodeSizeToSplitSpt    Minimum splitting size of a splitting node
  * @param minNodeSizeToSplitAvg    Minimum averaging size of a splitting node
  * @param minSplitGain    Minimum loss reduction to split a node.
  * @param maxDepth    Max depth of a tree
  * @param splittingSampleIndex    A vector with index of splitting samples
  * @param averagingSampleIndex    A vector with index of averaging samples
  * @param random_number_generator    A mt19937 random generator
  * @param splitMiddle    Boolean to indicate if new feature value is
  *    determined at a random position between two feature values
  * @param maxObs    Max number of observations to split on
  */
 /* Sanity Check */
  if (minNodeSizeAvg == 0) {
    throw std::runtime_error("minNodeSizeAvg cannot be set to 0.");
  }
  if (minNodeSizeSpt == 0) {
    throw std::runtime_error("minNodeSizeSpt cannot be set to 0.");
  }
  if (minNodeSizeToSplitSpt == 0) {
    throw std::runtime_error("minNodeSizeToSplitSpt cannot be set to 0.");
  }
  if (minNodeSizeToSplitAvg == 0) {
    throw std::runtime_error("minNodeSizeToSplitAvg cannot be set to 0.");
  }
  if (minNodeSizeToSplitAvg > (*averagingSampleIndex).size()) {
    std::ostringstream ostr;
    ostr << "minNodeSizeToSplitAvg cannot exceed total elements in the "
    "averaging samples: minNodeSizeToSplitAvg=" << minNodeSizeToSplitAvg <<
      ", averagingSampleSize=" << (*averagingSampleIndex).size() << ".";
    throw std::runtime_error(ostr.str());
  }
  if (minNodeSizeToSplitSpt > (*splittingSampleIndex).size()) {
    std::ostringstream ostr;
    ostr << "minNodeSizeToSplitSpt cannot exceed total elements in the "
    "splitting samples: minNodeSizeToSplitSpt=" << minNodeSizeToSplitSpt <<
      ", splittingSampleSize=" << (*splittingSampleIndex).size() << ".";
    throw std::runtime_error(ostr.str());
  }
  if (maxDepth == 0) {
    throw std::runtime_error("maxDepth cannot be set to 0.");
  }
  if (minSplitGain != 0 && !linear) {
    throw std::runtime_error("minSplitGain cannot be set without setting linear to be true.");
  }
  if ((*averagingSampleIndex).size() == 0) {
    throw std::runtime_error("averagingSampleIndex size cannot be set to 0.");
  }
  if ((*splittingSampleIndex).size() == 0) {
    throw std::runtime_error("splittingSampleIndex size cannot be set to 0.");
  }
  if (mtry == 0) {
    throw std::runtime_error("mtry cannot be set to 0.");
  }
  if (mtry > (*trainingData).getNumColumns()) {
    std::ostringstream ostr;
    ostr << "mtry cannot exceed total amount of features: mtry=" << mtry
         << ", totalNumFeatures=" << (*trainingData).getNumColumns() << ".";
    throw std::runtime_error(ostr.str());
  }

  /* Move all pointers to the current object */
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_hasNas = hasNas;
  this->_maxDepth = maxDepth;
  this->_interactionDepth = interactionDepth;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
  std::unique_ptr< RFNode > root ( new RFNode() );
  this->_root = std::move(root);
  /* Node ID's are 1 indexed from left to right */
  this->_nodeCount = 0;
  this->_seed = seed;


  /* If ridge splitting, initialize RSS components to pass to leaves*/

  std::vector<size_t>* splitIndexes = getSplittingIndex();
  size_t numLinearFeatures;
  std::vector<double> firstOb = trainingData->getLinObsData((*splitIndexes)[0]);
  numLinearFeatures = firstOb.size();
  firstOb.push_back(1.0);
  arma::Mat<double> sTotal(firstOb.size(),
                           1);
  sTotal.col(0) = arma::conv_to<arma::Col<double> >::from(firstOb);
  arma::Mat<double> gTotal(numLinearFeatures + 1,
                           numLinearFeatures + 1);
  if (linear) {
    this->initializelinear(trainingData,
                                  gTotal,
                                  sTotal,
                                  numLinearFeatures,
                                  getSplittingIndex());
  }

  /* Make shared pointers to sTotal, gTotal*/
  std::shared_ptr< arma::Mat<double> > g_ptr = std::make_shared< arma::Mat<double> >(gTotal);
  std::shared_ptr< arma::Mat<double> > s_ptr = std::make_shared< arma::Mat<double> >(sTotal);

  /* When not linear splitting, use nullptr for unneeded matrices */
  if (!linear) {
    g_ptr = nullptr;
    s_ptr = nullptr;
  }

  /* We check the monotonicity to see if we need to take this into account when
   * splitting
   */
  bool monotone_splits = !std::all_of(trainingData->getMonotonicConstraints()->begin(),
                                      trainingData->getMonotonicConstraints()->end(),
                                      [](int i) { return i==0; });;
  struct monotonic_info monotonic_details;

  if (monotone_splits) {
    monotonic_details.monotonic_constraints = (*trainingData->getMonotonicConstraints());
    monotonic_details.upper_bound = std::numeric_limits<float>::max();
    monotonic_details.lower_bound = -std::numeric_limits<float>::max();
  }

  /* Recursively grow the tree */
  recursivePartition(
    getRoot(),
    getAveragingIndex(),
    getSplittingIndex(),
    trainingData,
    random_number_generator,
    0,
    splitMiddle,
    maxObs,
    linear,
    overfitPenalty,
    g_ptr,
    s_ptr,
    monotone_splits,
    monotonic_details
  );
}

void forestryTree::setDummyTree(
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
){
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_maxDepth = maxDepth;
  this->_interactionDepth = interactionDepth;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
}

void forestryTree::predict(
    std::vector<double> &outputPrediction,
    std::vector<int>* terminalNodes,
    std::vector< std::vector<double> >* xNew,
    DataFrame* trainingData,
    arma::Mat<double>* weightMatrix,
    bool linear,
    unsigned int seed
){
  // If we are estimating the average in each leaf:
  struct rangeGenerator {
    size_t currentNumber;
    rangeGenerator(size_t startNumber): currentNumber(startNumber) {};
    size_t operator()() {return currentNumber++; }
  };

  std::vector<size_t> updateIndex(outputPrediction.size());
  rangeGenerator _rangeGenerator(0);
  std::generate(updateIndex.begin(), updateIndex.end(), _rangeGenerator);
  (*getRoot()).predict(outputPrediction, terminalNodes, &updateIndex, xNew, trainingData,
   weightMatrix, linear, getOverfitPenalty(), seed);
  //Rcpp::Rcout << "Seed is" << seed << ".\n";
}


std::vector<size_t> sampleFeatures(
    size_t mtry,
    std::mt19937_64& random_number_generator,
    int totalColumns,
    bool numFeaturesOnly,
    std::vector<size_t>* numCols,
    std::vector<double>* weights,
    std::vector<size_t>* sampledFeatures
){
  if(weights->size() == 0)
    return *sampledFeatures;
  else {
    // Sample features without replacement
    std::vector<size_t> featureList;
    if (numFeaturesOnly) {
      while (featureList.size() < mtry) {
        std::discrete_distribution<size_t> discrete_dist(
            weights->begin(), weights->end()
        );

        size_t index = discrete_dist(random_number_generator);

        if (featureList.size() == 0 ||
            std::find(featureList.begin(),
                      featureList.end(),
                      (*numCols)[index]) == featureList.end()
        ) {
          featureList.push_back((*numCols)[index]);
        }
      }

    } else {
      while (featureList.size() < mtry) {
        std::discrete_distribution<size_t> discrete_dist(
            weights->begin(), weights->end()
        );
        size_t randomIndex = discrete_dist(random_number_generator);
        if (
            featureList.size() == 0 ||
              std::find(
                featureList.begin(),
                featureList.end(),
                randomIndex
              ) == featureList.end()
        ) {
          featureList.push_back(randomIndex);
        }
      }
    }
    return featureList;
  }
}


void splitDataIntoTwoParts(
    DataFrame* trainingData,
    std::vector<size_t>* sampleIndex,
    size_t splitFeature,
    double splitValue,
    std::vector<size_t>* leftPartitionIndex,
    std::vector<size_t>* rightPartitionIndex,
    bool categoical,
    bool hasNas,
    size_t &naLeftCount,
    size_t &naRightCount
){

  if (hasNas) {
    std::vector<size_t> naIndices;
    for (
        std::vector<size_t>::iterator it = (*sampleIndex).begin();
        it != (*sampleIndex).end();
        ++it
    ) {
      if (categoical) {
        // categorical, split by (==) or (!=)
        double currentFeatureValue = trainingData->getPoint(*it, splitFeature);
        if (std::isnan(currentFeatureValue)) {
          naIndices.push_back(*it);
        } else if (currentFeatureValue == splitValue) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }

      } else {
        // Non-categorical, split to left (<) and right (>=) according to the
        double currentFeatureValue = trainingData->getPoint(*it, splitFeature);
        if (std::isnan(currentFeatureValue)) {
          naIndices.push_back(*it);
        } else if (currentFeatureValue < splitValue) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }
    }
    // Now we have computed the split indices without NA indices, get two partition means and split NA indices
    double leftMean = trainingData->partitionMean(leftPartitionIndex);
    double rightMean = trainingData->partitionMean(rightPartitionIndex);

    for (const auto& index : naIndices) {
      if (abs(trainingData->getOutcomePoint(index) - leftMean) < abs(trainingData->getOutcomePoint(index) - rightMean)) {
        leftPartitionIndex->push_back(index);
        naLeftCount++;
      } else {
        rightPartitionIndex->push_back(index);
        naRightCount++;
      }
    }
    //Run normal splitting
  } else {

    for (
        std::vector<size_t>::iterator it = (*sampleIndex).begin();
        it != (*sampleIndex).end();
        ++it
    ) {
      if (categoical) {
        // categorical, split by (==) or (!=)
        if ((*trainingData).getPoint(*it, splitFeature) == splitValue) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      } else {
        // Non-categorical, split to left (<) and right (>=) according to the

        if ((*trainingData).getPoint(*it, splitFeature) < splitValue) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }
    }
  }
}

void splitData(
    DataFrame* trainingData,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t splitFeature,
    double splitValue,
    std::vector<size_t>* averagingLeftPartitionIndex,
    std::vector<size_t>* averagingRightPartitionIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    size_t &naLeftCount,
    size_t &naRightCount,
    bool categoical,
    bool hasNas
){
  size_t avgL = 0;
  size_t avgR = 0;
  // averaging data
  splitDataIntoTwoParts(
    trainingData,
    averagingSampleIndex,
    splitFeature,
    splitValue,
    averagingLeftPartitionIndex,
    averagingRightPartitionIndex,
    categoical,
    hasNas,
    avgL,
    avgR
  );


  // splitting data
  splitDataIntoTwoParts(
    trainingData,
    splittingSampleIndex,
    splitFeature,
    splitValue,
    splittingLeftPartitionIndex,
    splittingRightPartitionIndex,
    categoical,
    hasNas,
    naLeftCount,
    naRightCount
  );
}



std::pair<double, double> calculateRSquaredSplit (
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    double overfitPenalty,
    std::mt19937_64& random_number_generator
) {
  // Get residual sum of squares for parent, left child, and right child nodes
  double rssParent, rssLeft, rssRight;
  rssParent = calculateRSS(trainingData,
                           splittingSampleIndex,
                           overfitPenalty,
			   random_number_generator);
  rssLeft = calculateRSS(trainingData,
                         splittingLeftPartitionIndex,
                         overfitPenalty,
			 random_number_generator);
  rssRight = calculateRSS(trainingData,
                          splittingRightPartitionIndex,
                          overfitPenalty,
			  random_number_generator);

  // Calculate total sum of squares
  double outcomeSum = 0;
  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    outcomeSum += trainingData->getOutcomePoint((*splittingSampleIndex)[i]);
  }
  double outcomeMean = outcomeSum/(splittingSampleIndex->size());

  double totalSumSquares = 0;
  double meanDifference;
  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    meanDifference =
      (trainingData->getOutcomePoint((*splittingSampleIndex)[i]) - outcomeMean);
    totalSumSquares += meanDifference * meanDifference;
  }

  // Use TSS and RSS to calculate r^2 values for parent and children
  double rSquaredParent = (1 - (rssParent/totalSumSquares));
  double rSquaredChildren = (1 - ((rssLeft + rssRight)/totalSumSquares));
  return std::make_pair(rSquaredParent, rSquaredChildren);
}

double crossValidatedRSquared (
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    double overfitPenalty,
    size_t numTimesCV,
    std::mt19937_64& random_number_generator
) {
  // Apply 5 times 10-fold cross-validation
  double rSquaredParent, rSquaredChildren;
  double totalRSquaredParent = 0;
  double totalRSquaredChildren = 0;

  for (size_t i = 0; i < numTimesCV; i++) {
    std::tie(rSquaredParent, rSquaredChildren) =
      calculateRSquaredSplit(
        trainingData,
        splittingSampleIndex,
        splittingLeftPartitionIndex,
        splittingRightPartitionIndex,
        overfitPenalty,
	random_number_generator
      );
    totalRSquaredParent += rSquaredParent;
    totalRSquaredChildren += rSquaredChildren;
  }

  return (totalRSquaredChildren/numTimesCV) - (totalRSquaredParent/numTimesCV);
}


void forestryTree::recursivePartition(
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
){
  if ((*averagingSampleIndex).size() < getMinNodeSizeAvg() ||
      (*splittingSampleIndex).size() < getMinNodeSizeSpt() ||
      (depth == getMaxDepth())) {
    // Create two lists on heap and transfer the owernship to the node
    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>(*averagingSampleIndex)
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>(*splittingSampleIndex)
    );
    size_t node_id;
    assignNodeId(node_id);
    (*rootNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_),
        node_id
    );
    return;
  }

  // Sample mtry amounts of features if possible.
  std::vector<size_t> featureList;
  std::vector<double>* featureWeightsUsed;
  std::vector<size_t>* sampledWeightsVariablesUsed;

    if(depth >= getInteractionDepth()) {
      featureWeightsUsed = trainingData->getdeepFeatureWeights();
      sampledWeightsVariablesUsed = trainingData->getdeepFeatureWeightsVariables();
    } else {
      featureWeightsUsed = trainingData->getfeatureWeights();
      sampledWeightsVariablesUsed = trainingData->getfeatureWeightsVariables();
    }

    featureList = sampleFeatures(
      getMtry(),
      random_number_generator,
      ((int) (*trainingData).getNumColumns()),
      false,
      trainingData->getNumCols(),
      featureWeightsUsed,
      sampledWeightsVariablesUsed
    );

  // Select best feature
  size_t bestSplitFeature;
  double bestSplitValue;
  double bestSplitLoss;
  size_t naLeftCount = 0;
  size_t naRightCount = 0;


  /* Arma mat memory is uninitialized now */
  arma::Mat<double> bestSplitGL;
  arma::Mat<double> bestSplitGR;
  arma::Mat<double> bestSplitSL;
  arma::Mat<double> bestSplitSR;


  /* IF LINEAR set size of Arma splitting matrices correctly */
  if (linear) {
    bestSplitGL.set_size(size(*gtotal));
    bestSplitGR.set_size(size(*gtotal));
    bestSplitSL.set_size(size(*stotal));
    bestSplitSR.set_size(size(*stotal));
  }

  selectBestFeature(
    bestSplitFeature,
    bestSplitValue,
    bestSplitLoss,
    bestSplitGL,
    bestSplitGR,
    bestSplitSL,
    bestSplitSR,
    &featureList,
    averagingSampleIndex,
    splittingSampleIndex,
    trainingData,
    random_number_generator,
    splitMiddle,
    maxObs,
    linear,
    overfitPenalty,
    gtotal,
    stotal,
    monotone_splits,
    monotone_details
  );

  // Create a leaf node if the current bestSplitValue is NA
  if (std::isnan(bestSplitValue)) {
    // Create two lists on heap and transfer the owernship to the node
    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>(*averagingSampleIndex)
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>(*splittingSampleIndex)
    );
    size_t node_id;
    assignNodeId(node_id);
    (*rootNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_),
        node_id
    );

  } else {
    // Test if the current feature is categorical
    std::vector<size_t> averagingLeftPartitionIndex;
    std::vector<size_t> averagingRightPartitionIndex;
    std::vector<size_t> splittingLeftPartitionIndex;
    std::vector<size_t> splittingRightPartitionIndex;
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();

    // Create split for both averaging and splitting dataset based on
    // categorical feature or not
    splitData(
      trainingData,
      averagingSampleIndex,
      splittingSampleIndex,
      bestSplitFeature,
      bestSplitValue,
      &averagingLeftPartitionIndex,
      &averagingRightPartitionIndex,
      &splittingLeftPartitionIndex,
      &splittingRightPartitionIndex,
      naLeftCount,
      naRightCount,
      std::find(
        categorialCols.begin(),
        categorialCols.end(),
        bestSplitFeature
      ) != categorialCols.end(),
      gethasNas()
    );

    // Stopping-criteria
    if (getMinSplitGain() > 0) {
      double rSquaredDifference = crossValidatedRSquared(
        trainingData,
        splittingSampleIndex,
        &splittingLeftPartitionIndex,
        &splittingRightPartitionIndex,
        overfitPenalty,
        1,
	      random_number_generator
      );

      if (rSquaredDifference < getMinSplitGain()) {
        std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
            new std::vector<size_t>(*averagingSampleIndex)
        );
        std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
            new std::vector<size_t>(*splittingSampleIndex)
        );
        size_t node_id;
        assignNodeId(node_id);
        (*rootNode).setLeafNode(
            std::move(averagingSampleIndex_),
            std::move(splittingSampleIndex_),
            node_id
        );
        return;
      }
    }

    // Update sample index for both left and right partitions
    // Recursively grow the tree
    std::unique_ptr< RFNode > leftChild ( new RFNode() );
    std::unique_ptr< RFNode > rightChild ( new RFNode() );

    size_t childDepth = depth + 1;

    std::shared_ptr< arma::Mat<double> > g_ptr_r = std::make_shared< arma::Mat<double> >(bestSplitGR);
    std::shared_ptr< arma::Mat<double> > g_ptr_l = std::make_shared< arma::Mat<double> >(bestSplitGL);

    std::shared_ptr< arma::Mat<double> > s_ptr_r = std::make_shared< arma::Mat<double> >(bestSplitSR);
    std::shared_ptr< arma::Mat<double> > s_ptr_l = std::make_shared< arma::Mat<double> >(bestSplitSL);

    if (!linear) {
      g_ptr_r = nullptr;
      g_ptr_l = nullptr;
      s_ptr_r = nullptr;
      s_ptr_l = nullptr;
    }

    // If monotone splitting, we need to pass down the monotone constraints,
    // uncle mean, and right and left child indicators
    struct monotonic_info monotonic_details_left;
    struct monotonic_info monotonic_details_right;

    if (monotone_splits) {
      int monotone_direction = monotone_details.monotonic_constraints[bestSplitFeature];
      monotonic_details_left.monotonic_constraints = (*trainingData->getMonotonicConstraints());
      monotonic_details_right.monotonic_constraints = (*trainingData->getMonotonicConstraints());

      float leftNodeMean = calculateMonotonicBound(trainingData->partitionMean(&splittingLeftPartitionIndex), monotone_details);
      float rightNodeMean = calculateMonotonicBound(trainingData->partitionMean(&splittingRightPartitionIndex), monotone_details);
      float midMean = (leftNodeMean + rightNodeMean )/(2);

      // Pass down the new upper and lower bounds if it is a monotonic split,
      if (monotone_direction == -1) {
        monotonic_details_left.lower_bound = midMean;
        monotonic_details_right.upper_bound = midMean;

        monotonic_details_left.upper_bound = monotone_details.upper_bound;
        monotonic_details_right.lower_bound = monotone_details.lower_bound;
      } else if (monotone_direction == 1) {
        monotonic_details_left.upper_bound = midMean;
        monotonic_details_right.lower_bound = midMean;

        monotonic_details_left.lower_bound =  monotone_details.lower_bound;
        monotonic_details_right.upper_bound = monotone_details.upper_bound;
      } else {
        // otherwise keep the old ones
        monotonic_details_left.upper_bound = monotone_details.upper_bound;
        monotonic_details_left.lower_bound = monotone_details.lower_bound;
        monotonic_details_right.upper_bound = monotone_details.upper_bound;
        monotonic_details_right.lower_bound = monotone_details.lower_bound;
      }
    }

    recursivePartition(
      leftChild.get(),
      &averagingLeftPartitionIndex,
      &splittingLeftPartitionIndex,
      trainingData,
      random_number_generator,
      childDepth,
      splitMiddle,
      maxObs,
      linear,
      overfitPenalty,
      g_ptr_l,
      s_ptr_l,
      monotone_splits,
      monotonic_details_left
    );
    recursivePartition(
      rightChild.get(),
      &averagingRightPartitionIndex,
      &splittingRightPartitionIndex,
      trainingData,
      random_number_generator,
      childDepth,
      splitMiddle,
      maxObs,
      linear,
      overfitPenalty,
      g_ptr_r,
      s_ptr_r,
      monotone_splits,
      monotonic_details_right
    );

    (*rootNode).setSplitNode(
        bestSplitFeature,
        bestSplitValue,
        std::move(leftChild),
        std::move(rightChild),
        naLeftCount,
        naRightCount
    );
  }
}

void forestryTree::initializelinear(
    DataFrame* trainingData,
    arma::Mat<double>& gTotal,
    arma::Mat<double>& sTotal,
    size_t numLinearFeatures,
    std::vector<size_t>* splitIndexes
) {
  gTotal = sTotal * (sTotal.t());
  sTotal = trainingData->getOutcomePoint((*splitIndexes)[0]) * sTotal;

  std::vector<double> temp(numLinearFeatures + 1);
  arma::Mat<double> tempOb(numLinearFeatures + 1, 1);
  /* Sum up sTotal and gTotal once on every observation in splitting set*/
  for (size_t i = 1; i < splitIndexes->size(); i++) {
    temp = trainingData->getLinObsData((*splitIndexes)[i]);
    temp.push_back(1.0);
    tempOb.col(0) = arma::conv_to<arma::Col<double> >::from(temp);
    gTotal = gTotal + (tempOb * (tempOb.t()));
    sTotal = sTotal + trainingData->getOutcomePoint((*splitIndexes)[i])
      * tempOb;
  }
}


void forestryTree::selectBestFeature(
    size_t &bestSplitFeature,
    double &bestSplitValue,
    double &bestSplitLoss,
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
){

  // Get the number of total features
  size_t mtry = (*featureList).size();

  // Initialize the minimum loss for each feature
  double* bestSplitLossAll = new double[mtry];
  double* bestSplitValueAll = new double[mtry];
  size_t* bestSplitFeatureAll = new size_t[mtry];
  size_t* bestSplitCountAll = new size_t[mtry];

  for (size_t i=0; i<mtry; i++) {
    bestSplitLossAll[i] = -std::numeric_limits<double>::infinity();
    bestSplitValueAll[i] = std::numeric_limits<double>::quiet_NaN();
    bestSplitFeatureAll[i] = std::numeric_limits<size_t>::quiet_NaN();
    bestSplitCountAll[i] = 0;
  }

  // Iterate each selected features
  for (size_t i=0; i<mtry; i++) {
    size_t currentFeature = (*featureList)[i];
    // Test if the current feature is in the categorical list
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();
    if (
        std::find(
          categorialCols.begin(),
          categorialCols.end(),
          currentFeature
        ) != categorialCols.end()
    ){
      if (linear) {
        // Ridge split on categorical feature
        findBestSplitRidgeCategorical(
          averagingSampleIndex,
          splittingSampleIndex,
          i,
          currentFeature,
          bestSplitLossAll,
          bestSplitValueAll,
          bestSplitFeatureAll,
          bestSplitCountAll,
          trainingData,
          getMinNodeSizeToSplitSpt(),
          getMinNodeSizeToSplitAvg(),
          random_number_generator,
          overfitPenalty,
          gtotal,
          stotal
        );
      } else if (gethasNas()) {
        // Run imputation split on categorical
        findBestSplitImputeCategorical(
          averagingSampleIndex,
          splittingSampleIndex,
          i,
          currentFeature,
          bestSplitLossAll,
          bestSplitValueAll,
          bestSplitFeatureAll,
          bestSplitCountAll,
          trainingData,
          getMinNodeSizeToSplitSpt(),
          getMinNodeSizeToSplitAvg(),
          random_number_generator,
          maxObs
        );
      } else {
        // Run CART split categorical
        findBestSplitValueCategorical(
          averagingSampleIndex,
          splittingSampleIndex,
          i,
          currentFeature,
          bestSplitLossAll,
          bestSplitValueAll,
          bestSplitFeatureAll,
          bestSplitCountAll,
          trainingData,
          getMinNodeSizeToSplitSpt(),
          getMinNodeSizeToSplitAvg(),
          random_number_generator,
          maxObs
        );
      }
    } else if (linear) {
      findBestSplitRidge(
        averagingSampleIndex,
        splittingSampleIndex,
        i,
        currentFeature,
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        trainingData,
        getMinNodeSizeToSplitSpt(),
        getMinNodeSizeToSplitAvg(),
        random_number_generator,
        splitMiddle,
        maxObs,
        overfitPenalty,
        gtotal,
        stotal
      );
    } else if (gethasNas()) {
      // Run impute split
      findBestSplitImpute(
        averagingSampleIndex,
        splittingSampleIndex,
        i,
        currentFeature,
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        trainingData,
        getMinNodeSizeToSplitSpt(),
        getMinNodeSizeToSplitAvg(),
        random_number_generator,
        splitMiddle,
        maxObs,
        monotone_splits,
        monotone_details
      );
    } else {
      // Run Standard CART split
      findBestSplitValueNonCategorical(
        averagingSampleIndex,
        splittingSampleIndex,
        i,
        currentFeature,
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        trainingData,
        getMinNodeSizeToSplitSpt(),
        getMinNodeSizeToSplitAvg(),
        random_number_generator,
        splitMiddle,
        maxObs,
        monotone_splits,
        monotone_details
      );
    }
  }

  determineBestSplit(
    bestSplitFeature,
    bestSplitValue,
    bestSplitLoss,
    mtry,
    bestSplitLossAll,
    bestSplitValueAll,
    bestSplitFeatureAll,
    bestSplitCountAll,
    random_number_generator
  );

  // If ridge splitting, need to update RSS components to pass down
  if (linear) {
    updateBestSplitG(bestSplitGL,
                     bestSplitGR,
                     (*gtotal),
                     trainingData,
                     splittingSampleIndex,
                     bestSplitFeature,
                     bestSplitValue);

    updateBestSplitS(bestSplitSL,
                     bestSplitSR,
                     (*stotal),
                     trainingData,
                     splittingSampleIndex,
                     bestSplitFeature,
                     bestSplitValue);
  }

  delete[](bestSplitLossAll);
  delete[](bestSplitValueAll);
  delete[](bestSplitFeatureAll);
  delete[](bestSplitCountAll);
}

void forestryTree::printTree(){
  (*getRoot()).printSubtree();
}

void forestryTree::getOOBindex(
    std::vector<size_t> &outputOOBIndex,
    size_t nRows
){

  // Generate union of splitting and averaging dataset
  std::sort(
    (*getSplittingIndex()).begin(),
    (*getSplittingIndex()).end()
  );
  std::sort(
    (*getAveragingIndex()).begin(),
    (*getAveragingIndex()).end()
  );

  std::vector<size_t> allSampledIndex(
      (*getSplittingIndex()).size() + (*getAveragingIndex()).size()
  );

  std::vector<size_t>::iterator it= std::set_union(
    (*getSplittingIndex()).begin(),
    (*getSplittingIndex()).end(),
    (*getAveragingIndex()).begin(),
    (*getAveragingIndex()).end(),
    allSampledIndex.begin()
  );

  allSampledIndex.resize((unsigned long) (it - allSampledIndex.begin()));

  // Generate a vector of all index based on nRows
  struct IncGenerator {
    size_t current_;
    IncGenerator(size_t start): current_(start) {}
    size_t operator()() { return current_++; }
  };
  std::vector<size_t> allIndex(nRows);
  IncGenerator g(0);
  std::generate(allIndex.begin(), allIndex.end(), g);

  // OOB index is the set difference between sampled index and all index
  std::vector<size_t> OOBIndex(nRows);

  it = std::set_difference (
    allIndex.begin(),
    allIndex.end(),
    allSampledIndex.begin(),
    allSampledIndex.end(),
    OOBIndex.begin()
  );
  OOBIndex.resize((unsigned long) (it - OOBIndex.begin()));

  for (
      std::vector<size_t>::iterator it_ = OOBIndex.begin();
      it_ != OOBIndex.end();
      ++it_
  ) {
    outputOOBIndex.push_back(*it_);
  }

}

void forestryTree::getOOBPrediction(
    std::vector<double> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData
){

  std::vector<size_t> OOBIndex;
  getOOBindex(OOBIndex, trainingData->getNumRows());

  for (
      std::vector<size_t>::iterator it=OOBIndex.begin();
      it!=OOBIndex.end();
      ++it
  ) {

    size_t OOBSampleIndex = *it;

    // Predict current oob sample
    std::vector<double> currentTreePrediction(1);
    std::vector<int>* currentTreeTerminalNodes = nullptr;
    std::vector<double> OOBSampleObservation((*trainingData).getNumColumns());
    (*trainingData).getObservationData(OOBSampleObservation, OOBSampleIndex);

    std::vector< std::vector<double> > OOBSampleObservation_;
    for (size_t k=0; k<(*trainingData).getNumColumns(); k++){
      std::vector<double> OOBSampleObservation_iter(1);
      OOBSampleObservation_iter[0] = OOBSampleObservation[k];
      OOBSampleObservation_.push_back(OOBSampleObservation_iter);
    }

    predict(
      currentTreePrediction,
      currentTreeTerminalNodes,
      &OOBSampleObservation_,
      trainingData
    );

    // Update the global OOB vector
    outputOOBPrediction[OOBSampleIndex] += currentTreePrediction[0];
    outputOOBCount[OOBSampleIndex] += 1;
  }
}

void forestryTree::getShuffledOOBPrediction(
    std::vector<double> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData,
    size_t shuffleFeature,
    std::mt19937_64& random_number_generator
){
  // Gives OOB prediction with shuffleFeature premuted randomly
  // For use in determining variable importance

  std::vector<size_t> OOBIndex;
  getOOBindex(OOBIndex, trainingData->getNumRows());

  std::vector<size_t> shuffledOOBIndex = OOBIndex;
  std::shuffle(shuffledOOBIndex.begin(), shuffledOOBIndex.end(), random_number_generator);
  size_t currentIndex = 0;

  for (
      std::vector<size_t>::iterator it=OOBIndex.begin();
      it!=OOBIndex.end();
      ++it
  ) {

    size_t OOBSampleIndex = *it;

    // Predict current oob sample
    std::vector<double> currentTreePrediction(1);
    std::vector<int>* currentTreeTerminalNodes = nullptr;
    std::vector<double> OOBSampleObservation((*trainingData).getNumColumns());
    (*trainingData).getShuffledObservationData(OOBSampleObservation,
                                               OOBSampleIndex,
                                               shuffleFeature,
                                               shuffledOOBIndex[currentIndex]);

    std::vector< std::vector<double> > OOBSampleObservation_;
    for (size_t k=0; k<(*trainingData).getNumColumns(); k++){
      std::vector<double> OOBSampleObservation_iter(1);
      OOBSampleObservation_iter[0] = OOBSampleObservation[k];
      OOBSampleObservation_.push_back(OOBSampleObservation_iter);
    }

    predict(
      currentTreePrediction,
      currentTreeTerminalNodes,
      &OOBSampleObservation_,
      trainingData
    );

    // Update the global OOB vector
    outputOOBPrediction[OOBSampleIndex] += currentTreePrediction[0];
    outputOOBCount[OOBSampleIndex] += 1;
    currentIndex++;
  }
}

// -----------------------------------------------------------------------------
std::unique_ptr<tree_info> forestryTree::getTreeInfo(
    DataFrame* trainingData
){
  std::unique_ptr<tree_info> treeInfo(
    new tree_info
  );
  (*getRoot()).write_node_info(treeInfo, trainingData);

  for (size_t i = 0; i<_averagingSampleIndex->size(); i++) {
    treeInfo->averagingSampleIndex.push_back((*_averagingSampleIndex)[i] + 1);
  }
  for (size_t i = 0; i<_splittingSampleIndex->size(); i++) {
    treeInfo->splittingSampleIndex.push_back((*_splittingSampleIndex)[i] + 1);
  }

  return treeInfo;
}

void forestryTree::reconstruct_tree(
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
    std::vector<size_t> splittingSampleIndex

  ){
  // Setting all the parameters:
  _mtry = mtry;
  _minNodeSizeSpt = minNodeSizeSpt;
  _minNodeSizeAvg = minNodeSizeAvg;
  _minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  _minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  _minSplitGain = minSplitGain;
  _maxDepth = maxDepth;
  _interactionDepth = interactionDepth;
  _hasNas = hasNas;
  _linear = linear;
  _overfitPenalty = overfitPenalty;
  _nodeCount = 0;

  _averagingSampleIndex = std::unique_ptr< std::vector<size_t> > (
    new std::vector<size_t>
  );
  for(size_t i=0; i<averagingSampleIndex.size(); i++){
    (*_averagingSampleIndex).push_back(averagingSampleIndex[i] - 1);
  }
  _splittingSampleIndex = std::unique_ptr< std::vector<size_t> > (
    new std::vector<size_t>
  );
  for(size_t i=0; i<splittingSampleIndex.size(); i++){
    (*_splittingSampleIndex).push_back(splittingSampleIndex[i] - 1);
  }

  std::unique_ptr< RFNode > root ( new RFNode() );
  this->_root = std::move(root);

  recursive_reconstruction(
    _root.get(),
    &var_ids,
    &split_vals,
    &leafAveidxs,
    &leafSplidxs,
    &naLeftCounts,
    &naRightCounts
  );

  return ;
}


void forestryTree::recursive_reconstruction(
  RFNode* currentNode,
  std::vector<int> * var_ids,
  std::vector<double> * split_vals,
  std::vector<size_t> * leafAveidxs,
  std::vector<size_t> * leafSplidxs,
  std::vector<int> * naLeftCounts,
  std::vector<int> * naRightCounts
) {
  int var_id = (*var_ids)[0];
    (*var_ids).erase((*var_ids).begin());
  double  split_val = (*split_vals)[0];
    (*split_vals).erase((*split_vals).begin());

    size_t naLeftCount = (*naLeftCounts)[0];
    (*naLeftCounts).erase((*naLeftCounts).begin());
    size_t naRightCount = (*naRightCounts)[0];
    (*naRightCounts).erase((*naRightCounts).begin());



  if(var_id < 0){
    // This is a terminal node
    int nAve = abs(var_id);
    int nSpl = abs((*var_ids)[0]);
    (*var_ids).erase((*var_ids).begin());

    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>
    );

    for(int i=0; i<nAve; i++){
      averagingSampleIndex_->push_back((*leafAveidxs)[0] - 1);
      (*leafAveidxs).erase((*leafAveidxs).begin());
    }

    for(int i=0; i<nSpl; i++){
      splittingSampleIndex_->push_back((*leafSplidxs)[0] - 1);
      (*leafSplidxs).erase((*leafSplidxs).begin());
    }
    size_t node_id;
    assignNodeId(node_id);
    (*currentNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_),
        node_id
    );
    return;
  } else {
    // This is a normal splitting node
    std::unique_ptr< RFNode > leftChild ( new RFNode() );
    std::unique_ptr< RFNode > rightChild ( new RFNode() );

    recursive_reconstruction(
      leftChild.get(),
      var_ids,
      split_vals,
      leafAveidxs,
      leafSplidxs,
      naLeftCounts,
      naRightCounts
      );

    recursive_reconstruction(
      rightChild.get(),
      var_ids,
      split_vals,
      leafAveidxs,
      leafSplidxs,
      naLeftCounts,
      naRightCounts
    );

    (*currentNode).setSplitNode(
      (size_t) var_id - 1,
      split_val,
      std::move(leftChild),
      std::move(rightChild),
      naLeftCount,
      naRightCount
      );
    return;
  }
}
