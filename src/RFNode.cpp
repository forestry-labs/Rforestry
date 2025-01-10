// [[Rcpp::depends(RcppThread)]]
// [[Rcpp::plugins(cpp11)]]
#include "RFNode.h"
#include <RcppArmadillo.h>
#include <RcppThread.h>
#include <mutex>
#include <thread>
#include "utils.h"

std::mutex mutex_weightMatrix;


RFNode::RFNode():
  _averagingSampleIndex(nullptr), _splittingSampleIndex(nullptr),
  _splitFeature(0), _splitValue(0),
  _leftChild(nullptr), _rightChild(nullptr),
  _naLeftCount(0), _naRightCount(0), _averageCount(0), _splitCount(0) {}

RFNode::~RFNode() {
  //  std::cout << "RFNode() destructor is called." << std::endl;
};

void RFNode::setLeafNode(
  std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
  std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
  size_t nodeId
) {
  if (
      (*averagingSampleIndex).size() == 0 &&
        (*splittingSampleIndex).size() == 0
  ) {
    throw std::runtime_error("Intend to create an empty node.");
  }
  // Give the ownership of the index pointer to the RFNode object
  this->_naLeftCount = 0;
  this->_naRightCount = 0;
  this->_nodeId = nodeId;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_averageCount = (*_averagingSampleIndex).size();
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_splitCount = (*_splittingSampleIndex).size();
}

void RFNode::setSplitNode(
  size_t splitFeature,
  double splitValue,
  std::unique_ptr< RFNode > leftChild,
  std::unique_ptr< RFNode > rightChild,
  size_t naLeftCount,
  size_t naRightCount
) {
  // Split node constructor
  _averageCount = 0;
  _splitCount = 0;
  _splitFeature = splitFeature;
  _splitValue = splitValue;
  // Give the ownership of the child pointer to the RFNode object
  _leftChild = std::move(leftChild);
  _rightChild = std::move(rightChild);
  _naLeftCount = naLeftCount;
  _naRightCount = naRightCount;
}

void RFNode::ridgePredict(
  std::vector<double> &outputPrediction,
  std::vector<size_t>* updateIndex,
  std::vector< std::vector<double> >* xNew,
  DataFrame* trainingData,
  double lambda
) {


  //Observations to do regression with
  std::vector<size_t>* leafObs = getAveragingIndex();

  //Number of linear features in training data
  size_t dimension = (trainingData->getLinObsData((*leafObs)[0])).size();

  arma::Mat<double> x(leafObs->size(),
                     dimension + 1);

  arma::Mat<double> identity(dimension + 1,
                            dimension + 1);
  identity.eye();

  //Don't penalize intercept
  identity(dimension, dimension) = 0.0;

  std::vector<double> outcomePoints;
  std::vector<double> currentObservation;

  //Contruct X and outcome vector
  for (size_t i = 0; i < leafObs->size(); i++) {
    currentObservation = trainingData->getLinObsData((*leafObs)[i]);
    currentObservation.push_back(1.0);

    x.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);

    outcomePoints.push_back(trainingData->getOutcomePoint((*leafObs)[i]));
  }

  arma::Mat<double> y(outcomePoints.size(),
                     1);
  y.col(0) = arma::conv_to<arma::Col<double> >::from(outcomePoints);

  //Compute XtX + lambda * I * Y = C
  arma::Mat<double> coefficients = (x.t() * x +
                                  identity * lambda).i() * x.t() * y;

  //Map xNew into Eigen matrix
  arma::Mat<double> xn(updateIndex->size(),
                      dimension + 1);

  size_t index = 0;
  for (std::vector<size_t>::iterator it = updateIndex->begin();
       it != updateIndex->end();
       ++it) {

    std::vector<double> newObservation;
    for (size_t i = 0; i < dimension; i++) {
      newObservation.push_back((*xNew)[i][*it]);
    }
    newObservation.push_back(1.0);

    xn.row(index) = arma::conv_to<arma::Row<double> >::from(newObservation);
    index++;
  }

  //Multiply xNew * coefficients = result
  arma::Mat<double> predictions = xn * coefficients;

  for (size_t i = 0; i < updateIndex->size(); i++) {
    outputPrediction[(*updateIndex)[i]] = predictions(i, 0);
  }
}

void RFNode::predict(
  std::vector<double> &outputPrediction,
  std::vector<int>* terminalNodes,
  std::vector<size_t>* updateIndex,
  std::vector< std::vector<double> >* xNew,
  DataFrame* trainingData,
  arma::Mat<double>* weightMatrix,
  bool linear,
  double lambda,
  unsigned int seed
) {

  // If the node is a leaf, aggregate all its averaging data samples
  if (is_leaf()) {

      if (linear) {

      //Use ridgePredict (fit linear model on leaf avging obs + evaluate it)
      ridgePredict(outputPrediction,
                   updateIndex,
                   xNew,
                   trainingData,
                   lambda);
      } else {

      // Calculate the mean of current node
      double predictedMean = (*trainingData).partitionMean(getAveragingIndex());

      // Give all updateIndex the mean of the node as prediction values
      for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
      ) {
        outputPrediction[*it] = predictedMean;
      }
    }

    if (weightMatrix){
      // If weightMatrix is not a NULL pointer, then we want to update it,
      // because we have choosen aggregation = "weightmatrix".
      std::vector<size_t> idx_in_leaf =
        (*trainingData).get_all_row_idx(getAveragingIndex());
      // The following will lock the access to weightMatrix
      std::lock_guard<std::mutex> lock(mutex_weightMatrix);
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it ) {
        for (size_t i = 0; i<idx_in_leaf.size(); i++) {
          (*weightMatrix)(*it, idx_in_leaf[i] - 1) =
          (*weightMatrix)(*it, idx_in_leaf[i] - 1) +
          (double) 1.0 / idx_in_leaf.size();
        }
      }
    }

    if (terminalNodes) {
      // If terminalNodes not a NULLPTR, set the terminal node for all X in this
      // leaf to be the leaf node_id
      size_t node_id = getNodeId();
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it
      ) {
        (*terminalNodes)[*it] = node_id;
      }
    }

  } else {

    // Separate prediction tasks to two children
    std::vector<size_t>* leftPartitionIndex = new std::vector<size_t>();
    std::vector<size_t>* rightPartitionIndex = new std::vector<size_t>();

    size_t naLeftCount = getNaLeftCount();
    size_t naRightCount = getNaRightCount();

    std::vector<size_t> naSampling {naLeftCount, naRightCount};
    //std::vector<size_t> naSampling_no_miss {1, 1};
    std::vector<size_t> naSampling_no_miss {
      (*getLeftChild()).getAverageCountAlways(),
      (*getRightChild()).getAverageCountAlways()};

    std::discrete_distribution<size_t> discrete_dist(
        naSampling.begin(), naSampling.end()
    );
    std::discrete_distribution<size_t> discrete_dist_nonmissing(
        naSampling_no_miss.begin(), naSampling_no_miss.end()
    );
    std::mt19937_64 random_number_generator;
    random_number_generator.seed(seed);

    // Test if the splitting feature is categorical
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();
    if (
      std::find(
        categorialCols.begin(),
        categorialCols.end(),
        getSplitFeature()
      ) != categorialCols.end()
    ){

      // If the splitting feature is categorical, split by (==) or (!=)
      for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
      ) {
        double currentValue = (*xNew)[getSplitFeature()][*it];

        if (std::isnan(currentValue)) {
          size_t draw;

          // If we have a missing feature value, if no NAs were observed when
          // splitting send to left/right with probability proportional to
          // number of observations in left/right child node else send
          // right/left with probability in proportion to NA's which went
          // left/right when splitting
          if ((naLeftCount == 0) && (naRightCount == 0)) {
            draw = discrete_dist_nonmissing(random_number_generator);
          } else {
            draw = discrete_dist(random_number_generator);
          }

          if (draw == 0) {
            (*leftPartitionIndex).push_back(*it);
          } else {
            (*rightPartitionIndex).push_back(*it);
          }
        } else if (currentValue == getSplitValue()) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }

    } else {

      // For non-categorical, split to left (<) and right (>=) according to the
      // split value
      for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
      ) {
        double currentValue = (*xNew)[getSplitFeature()][*it];

        if (std::isnan(currentValue)) {
          size_t draw;

          // If we have a missing feature value, if no NAs were observed when
          // splitting send to left/right with probability proportional to
          // number of observations in left/right child node else send
          // right/left with probability in proportion to NA's which went
          // left/right when splitting
          if ((naLeftCount == 0) && (naRightCount == 0)) {
            draw = discrete_dist_nonmissing(random_number_generator);
          } else {
            draw = discrete_dist(random_number_generator);
          }

          if (draw == 0) {
            (*leftPartitionIndex).push_back(*it);
          } else {
            (*rightPartitionIndex).push_back(*it);
          }

        } else if (currentValue < getSplitValue()) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }

    }

    // Recursively get predictions from its children
    if ((*leftPartitionIndex).size() > 0) {
      (*getLeftChild()).predict(
        outputPrediction,
        terminalNodes,
        leftPartitionIndex,
        xNew,
        trainingData,
        weightMatrix,
        linear,
        lambda,
        seed
      );
    }
    if ((*rightPartitionIndex).size() > 0) {
      (*getRightChild()).predict(
        outputPrediction,
        terminalNodes,
        rightPartitionIndex,
        xNew,
        trainingData,
        weightMatrix,
        linear,
        lambda,
        seed
      );
    }

    delete(leftPartitionIndex);
    delete(rightPartitionIndex);

  }

}

bool RFNode::is_leaf() {
  int ave_ct = getAverageCount();
  int spl_ct = getSplitCount();
  if (
      (ave_ct == 0 && spl_ct != 0) ||(ave_ct != 0 && spl_ct == 0)
  ) {
    throw std::runtime_error(
        "Average count or Split count is 0, while the other is not!"
        );
  }
  return !(ave_ct == 0 && spl_ct == 0);
}

size_t RFNode::getAverageCountAlways() {
  if(is_leaf()) {
    return _averageCount;
  }
  else {
    return (*getRightChild()).getAverageCountAlways() +
      (*getLeftChild()).getAverageCountAlways();
  }
}

void RFNode::printSubtree(int indentSpace) {

  // Test if the node is leaf node
  if (is_leaf()) {

    // Print count of samples in the leaf node
    RcppThread::Rcout << std::string((unsigned long) indentSpace, ' ')
              << "Leaf Node: # of split samples = "
              << getSplitCount()
              << ", # of average samples = "
              << getAverageCount()
              << std::endl;
    R_FlushConsole();
    R_ProcessEvents();

  } else {

    // Print split feature and split value
    RcppThread::Rcout << std::string((unsigned long) indentSpace, ' ')
              << "Tree Node: split feature = "
              << getSplitFeature()
              << ", split value = "
              << getSplitValue()
              << std::endl;
    R_FlushConsole();
    R_ProcessEvents();
    // Recursively calling its children
    (*getLeftChild()).printSubtree(indentSpace+2);
    (*getRightChild()).printSubtree(indentSpace+2);

  }
}

// -----------------------------------------------------------------------------

void RFNode::write_node_info(
    std::unique_ptr<tree_info> & treeInfo,
    DataFrame* trainingData
){
  if (is_leaf()) {
    // If it is a leaf: set everything to be 0
    treeInfo->var_id.push_back(-getAveragingIndex()->size());
    treeInfo->var_id.push_back(-getSplittingIndex()->size());
    treeInfo->split_val.push_back(0);
    treeInfo->naLeftCount.push_back(-1);
    treeInfo->naRightCount.push_back(-1);


    std::vector<size_t> idx_in_leaf_Ave = *getAveragingIndex();
    for (size_t i = 0; i<idx_in_leaf_Ave.size(); i++) {
      treeInfo->leafAveidx.push_back(idx_in_leaf_Ave[i] + 1);
    }

    std::vector<size_t> idx_in_leaf_Spl = *getSplittingIndex();
    for (size_t i = 0; i<idx_in_leaf_Spl.size(); i++) {
      treeInfo->leafSplidx.push_back(idx_in_leaf_Spl[i] + 1);
    }


  } else {
    // If it is a usual node: remember split var and split value and recursively
    // call write_node_info on the left and the right child.
    treeInfo->var_id.push_back(getSplitFeature() + 1);
    treeInfo->split_val.push_back(getSplitValue());
    treeInfo->naLeftCount.push_back(getNaLeftCount());
    treeInfo->naRightCount.push_back(getNaRightCount());


    getLeftChild()->write_node_info(treeInfo, trainingData);
    getRightChild()->write_node_info(treeInfo, trainingData);
  }
  return;
}
