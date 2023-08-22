// [[Rcpp::plugins(cpp11)]]
#include "forestry.h"
#include "utils.h"
#include "sampling.h"
#include <random>
#include <algorithm>
#include <thread>
#include <mutex>
#include <armadillo>
#define DOPARELLEL true


forestry::forestry():
  _trainingData(nullptr), _ntree(0), _replace(0), _sampSize(0),
  _splitRatio(0),_OOBhonest(0),_mtry(0), _minNodeSizeSpt(0), _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0), _minNodeSizeToSplitAvg(0), _minSplitGain(0),
  _maxDepth(0), _interactionDepth(0), _forest(nullptr), _seed(0), _verbose(0),
  _nthread(0), _OOBError(0), _splitMiddle(0),_minTreesPerFold(0), _doubleTree(0){};

forestry::~forestry(){};

forestry::forestry(
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
){
  this->_trainingData = trainingData;
  this->_ntree = 0;
  this->_replace = replace;
  this->_sampSize = sampSize;
  this->_splitRatio = splitRatio;
  this->_OOBhonest = OOBhonest;
  this->_doubleBootstrap = doubleBootstrap;
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_maxDepth = maxDepth;
  this->_interactionDepth = interactionDepth;
  this->_seed = seed;
  this->_nthread = nthread;
  this->_verbose = verbose;
  this->_splitMiddle = splitMiddle;
  this->_maxObs = maxObs;
  this->_hasNas = hasNas;
  this->_naDirection = naDirection;
  this->_linear = linear;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;
  this->_naDirection = naDirection;
  this->_minTreesPerFold = minTreesPerFold;
  this->_foldSize = foldSize;

  if (splitRatio <= 0 || splitRatio > 1) {
    throw std::runtime_error("splitRatio should be inside (0, 1]");
  }

  size_t splitSampleSize = (size_t) (getSplitRatio() * sampSize);
  size_t averageSampleSize;
  if (splitRatio == 1 || splitRatio == 0) {
    averageSampleSize = splitSampleSize;
  } else {
    averageSampleSize = sampSize - splitSampleSize;
  }

  if (
    splitSampleSize < minNodeSizeToSplitSpt ||
    averageSampleSize < minNodeSizeToSplitAvg
  ) {
    throw std::runtime_error("splitRatio is too big or too small");
  }

  if (
    overfitPenalty < 0
  ) {
    throw std::runtime_error("overfitPenalty cannot be negative");
  }

  if (
      linear && hasNas
  ) {
    throw std::runtime_error("Imputation for missing values cannot be done for ridge splitting");
  }

  std::unique_ptr< std::vector< std::unique_ptr< forestryTree > > > forest (
    new std::vector< std::unique_ptr< forestryTree > >
  );
  this->_forest = std::move(forest);

  // Create initial trees
  addTrees(ntree);

  // Try sorting the forest by seed, this way we should do predict in the same order
  std::vector< std::unique_ptr< forestryTree > >* curr_forest;
  curr_forest = this->getForest();
  std::sort(curr_forest->begin(), curr_forest->end(), [](const std::unique_ptr< forestryTree >& a,
                                                         const std::unique_ptr< forestryTree >& b) {
    return a.get()->getSeed() > b.get()->getSeed();
  });

  //(*curr_forest)[curr_forest->size()-1]->printTree();
  //(*curr_forest)[0]->printTree();

}

void forestry::addTrees(size_t ntree) {

  const unsigned int newStartingTreeNumber = (unsigned int) getNtree();
  unsigned int newEndingTreeNumber;
  size_t numToGrow, groupToGrow;


  std::vector< std::vector<size_t> > foldMemberships(1);

  // This is called with ntree = 0 only when loading a saved forest.
  // When minTreesPerFold takes precedence over ntree, we need to make sure to
  // train 0 trees when ntree = 0, otherwise this messes up the reconstruction of the forest
  size_t numGroups = (*std::max_element(getTrainingData()->getGroups()->begin(),
                                        getTrainingData()->getGroups()->end()));

  if ((ntree != 0) && (getminTreesPerFold() > 0)) {

    size_t numFolds = ((size_t) std::ceil((double) numGroups / (double) getFoldSize()));

    numToGrow = (unsigned int) getminTreesPerFold() * (numFolds);
    // Want to grow max(ntree, |groups|*minTreePerGroup) total trees
    groupToGrow = numToGrow;
    numToGrow = std::max(numToGrow, ntree);

    std::mt19937_64 group_assign_rng;
    group_assign_rng.seed(getSeed());

    foldMemberships.resize(numFolds);
    for (size_t fold_i = 0; fold_i < numFolds; fold_i++) {
        foldMemberships[fold_i] = std::vector<size_t>(getFoldSize());
    }
    // Assign the groups to different folds. When the foldsize is 1, this is equivalent
    // to the previous minTreesPerFold implementation
    assign_groups_to_folds(
            numGroups,
            getFoldSize(),
            foldMemberships,
            group_assign_rng
    );
  } else {
    numToGrow = ntree;
  }
  newEndingTreeNumber = newStartingTreeNumber + (unsigned int) numToGrow;
  //for (size_t i = 0; i < foldMemberships.size(); i++) {
  //    std::cout << "Fold " << i << std::endl;
  //    print_vector(foldMemberships[i]);
  //}
  //std::cout << newEndingTreeNumber;

  unsigned int nthreadToUse = (unsigned int) getNthread();
  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = (unsigned int) std::thread::hardware_concurrency();
  }
  const unsigned int see = this->getSeed();

  #if DOPARELLEL
  if (isVerbose()) {
    // std::cout << "Training parallel using " << nthreadToUse << " threads"
    //           << std::endl;
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (unsigned int t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const unsigned int iStart, const unsigned int iEnd, const unsigned int t_) {

        // loop over al assigned trees, iStart is the starting tree number
        // and iEnd is the ending tree number

        for (unsigned int i = iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for (unsigned int i=newStartingTreeNumber; i<newEndingTreeNumber; i++) {
  #endif

          const unsigned int myseed = (i+1)*see;

          std::mt19937_64 random_number_generator;
          random_number_generator.seed(myseed);


          // Split sampled indices into averaging and splitting sets
          std::unique_ptr<std::vector<size_t> > splitSampleIndex;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex;
          std::unique_ptr<std::vector<size_t> > excludedSampleIndex;

          std::unique_ptr<std::vector<size_t> > splitSampleIndex2;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex2;

          std::vector<size_t> splitIndicesFill;
          std::vector<size_t> avgIndicesFill;
          std::vector<size_t> excludedIndicesFill;

          std::vector< std::vector<size_t> >* customSplitSample = getTrainingData()->getCustomSplitSample();

          if (customSplitSample->size() == 0) {
            // Generate the splitting and averaging indices for the ith tree
            generate_sample_indices(
              splitIndicesFill,
              avgIndicesFill,
              groupToGrow,
              getminTreesPerFold(),
              i,
              getSampleSize(),
              (ntree != 0) && (getTrainingData()->getGroups()->at(0) != 0) ? numGroups : 0,
              isReplacement(),
              getOOBhonest(),
              getDoubleBootstrap(),
              getSplitRatio(),
              _doubleTree,
              random_number_generator,
              foldMemberships,
              getTrainingData()
            );
          } else {
            splitIndicesFill = customSplitSample->at(i);
            avgIndicesFill = getTrainingData()->getCustomAvgSample()->at(i);
            // Only set excluded indices if provided
            if (getTrainingData()->getCustomExcludeSample()->size() > 0) {
                excludedIndicesFill = getTrainingData()->getCustomExcludeSample()->at(i);
            }
          }

          // Set the smart pointers to use the returned indices
          splitSampleIndex.reset(
                    new std::vector<size_t>(splitIndicesFill)
          );
          averageSampleIndex.reset(
                    new std::vector<size_t>(avgIndicesFill)
          );
          excludedSampleIndex.reset(
            new std::vector<size_t>(excludedIndicesFill)
          );

          // If we are doing doubleTree, swap the indices and make two trees
          if (_doubleTree) {
                splitSampleIndex2.reset(
                        new std::vector<size_t>(splitIndicesFill)
                );
                averageSampleIndex2.reset(
                        new std::vector<size_t>(avgIndicesFill)
                );
          }

          try{

            forestryTree *oneTree(
              new forestryTree(
                getTrainingData(),
                getMtry(),
                getMinNodeSizeSpt(),
                getMinNodeSizeAvg(),
                getMinNodeSizeToSplitSpt(),
                getMinNodeSizeToSplitAvg(),
                getMinSplitGain(),
                getMaxDepth(),
                getInteractionDepth(),
                std::move(splitSampleIndex),
                std::move(averageSampleIndex),
                std::move(excludedSampleIndex),
                random_number_generator,
                getSplitMiddle(),
                getMaxObs(),
                gethasNas(),
                getNaDirection(),
                getlinear(),
                getOverfitPenalty(),
                myseed
              )
            );

            forestryTree *anotherTree;
            if (_doubleTree) {
              anotherTree =
                new forestryTree(
                    getTrainingData(),
                    getMtry(),
                    getMinNodeSizeSpt(),
                    getMinNodeSizeAvg(),
                    getMinNodeSizeToSplitSpt(),
                    getMinNodeSizeToSplitAvg(),
                    getMinSplitGain(),
                    getMaxDepth(),
                    getInteractionDepth(),
                    std::move(averageSampleIndex2),
                    std::move(splitSampleIndex2),
                    std::move(excludedSampleIndex),
                    random_number_generator,
                    getSplitMiddle(),
                    getMaxObs(),
                    gethasNas(),
                    getNaDirection(),
                    getlinear(),
                    getOverfitPenalty(),
                    myseed
                 );
            }

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            if (isVerbose()) {
              // std::cout << "Finish training tree # " << (i + 1) << std::endl;
            }

            (*getForest()).emplace_back(oneTree);
            _ntree = _ntree + 1;
            if (_doubleTree) {
              (*getForest()).emplace_back(anotherTree);
              _ntree = _ntree + 1;
            } else {
              // delete anotherTree;
            }

          } catch (std::runtime_error &err) {
          }

        }
  #if DOPARELLEL
      },
      newStartingTreeNumber + t * numToGrow / nthreadToUse,
      (t + 1) == nthreadToUse ?
        (unsigned int) newEndingTreeNumber :
           newStartingTreeNumber + (t + 1) * numToGrow / nthreadToUse,
           t
    );

    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x){x.join();}
  );
  #endif
}

std::unique_ptr< std::vector<double> > forestry::predict(
  std::vector< std::vector<double> >* xNew,
  arma::Mat<double>* weightMatrix,
  arma::Mat<double>* coefficients,
  arma::Mat<int>* terminalNodes,
  unsigned int seed,
  size_t nthread,
  bool exact,
  bool use_weights,
  std::vector<size_t>* tree_weights,
  bool hierShrinkage,
  double lambdaShrinkage
){

  size_t numObservations = (*xNew)[0].size();
  std::vector<double> prediction(numObservations,0.0);


  // If using weights, we need to initialize this
  double total_weights;
  if (use_weights) {
    total_weights = 0.0;
    for (auto weight_i : *tree_weights) {
      total_weights += (double) weight_i;
    }
  } else {
    total_weights = (double) getNtree();
  }

  // If we want to return the ridge coefficients, initialize a matrix
  if (coefficients) {
    // Create coefficient vector of vectors of zeros
    std::vector< std::vector<float> > coef;
    size_t numObservations = (*xNew)[0].size();
    size_t numCol = (*coefficients).n_cols;
    for (size_t i=0; i<numObservations; i++) {
      std::vector<float> row;
      for (size_t j = 0; j<numCol; j++) {
        row.push_back(0);
      }
      coef.push_back(row);
    }
  }

  // Only needed if exact = TRUE, vector for storing each tree's predictions
  std::vector< std::vector<double> > tree_preds;
  std::vector< std::vector<int> > tree_nodes;
  std::vector<size_t> tree_seeds;
  std::vector<size_t> tree_total_nodes;

  #if DOPARELLEL
  size_t nthreadToUse = nthread;

  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }

  if (isVerbose()) {
    // std::cout << "Prediction parallel using " << nthreadToUse << " threads"
    //           << std::endl;
    if (use_weights) {
      // std::cout << "Weights given by" << std::endl;
      // print_vector(*tree_weights);
    }
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (size_t t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const int iStart, const int iEnd, const int t_) {

        // loop over al assigned trees, iStart is the starting tree number
        // and iEnd is the ending tree number
        for (int i=iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for(int i=0; i<((int) getNtree()); i++ ) {
  #endif
          try {
            std::vector<double> currentTreePrediction(numObservations);
            std::vector<int> currentTreeTerminalNodes(numObservations);
            std::vector< std::vector<double> > currentTreeCoefficients(numObservations);

            //If terminal nodes, pass option to tree predict
            forestryTree *currentTree = (*getForest())[i].get();

            if (use_weights && (tree_weights->at(i) == (size_t) 0)) {
              // If weight for the tree is zero, don't predict with that tree
              std::fill(currentTreePrediction.begin(), currentTreePrediction.end(), 0);
            } else if (coefficients) {
              for (size_t l=0; l<numObservations; l++) {
                currentTreeCoefficients[l] = std::vector<double>(coefficients->n_cols);
              }

              (*currentTree).predict(
                  currentTreePrediction,
                  &currentTreeTerminalNodes,
                  currentTreeCoefficients,
                  xNew,
                  getTrainingData(),
                  weightMatrix,
                  getlinear(),
                  getNaDirection(),
                  seed + i,
                  getMinNodeSizeToSplitAvg(),
                  NULL,
                  hierShrinkage,
                  lambdaShrinkage,
                  std::numeric_limits<double>::infinity()
              );

            } else {
              (*currentTree).predict(
                  currentTreePrediction,
                  &currentTreeTerminalNodes,
                  currentTreeCoefficients,
                  xNew,
                  getTrainingData(),
                  weightMatrix,
                  getlinear(),
                  getNaDirection(),
                  seed + i,
                  getMinNodeSizeToSplitAvg(),
                  NULL,
                  hierShrinkage,
                  lambdaShrinkage,
                  std::numeric_limits<double>::infinity()
              );

            }

            // HERE IF NEED TERMINAL NODES, pass option to tree predict, then
            // lock thread (shouldn't really need to), use i as offset and flip
            // bool of matrix

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            # endif

            // If we need to use the exact seeding order we save the tree
            // predictions and the tree seeds

            // For now store tree seeds even when not running exact,
            // hopefully this solves a valgrind error relating to the sorting
            // based on tree seeds when tree seeds might be uninitialized
            tree_seeds.push_back(currentTree->getSeed());

            if (exact) {
              tree_preds.push_back(currentTreePrediction);
              tree_nodes.push_back(currentTreeTerminalNodes);
              tree_total_nodes.push_back(currentTree->getNodeCount());
            } else {
              if (!use_weights) {
                for (size_t j = 0; j < numObservations; j++) {
                  prediction[j] += currentTreePrediction[j];
                }

                if (coefficients) {
                  for (size_t k = 0; k < numObservations; k++) {
                    for (size_t l = 0; l < coefficients->n_cols; l++) {
                      (*coefficients)(k,l) += currentTreeCoefficients[k][l];
                    }
                  }
                }

                if (terminalNodes) {
                  for (size_t k = 0; k < numObservations; k++) {
                    (*terminalNodes)(k, i) = currentTreeTerminalNodes[k];
                  }
                  (*terminalNodes)(numObservations, i) = (*currentTree).getNodeCount();
                }
              } else {
                if (tree_weights->at(i) != (size_t) 0) {
                  for (size_t j = 0; j < numObservations; j++) {
                    prediction[j] += ((double) tree_weights->at(i)) * currentTreePrediction[j];
                  }

                  if (coefficients) {
                    for (size_t k = 0; k < numObservations; k++) {
                      for (size_t l = 0; l < coefficients->n_cols; l++) {
                        (*coefficients)(k,l) += ((double) tree_weights->at(i)) * currentTreeCoefficients[k][l];
                      }
                    }
                  }
                }
              }
            }

          } catch (std::runtime_error &err) {
            // std::cerr << err.what() << std::endl;
          }
      }
  #if DOPARELLEL
      },
      t * getNtree() / nthreadToUse,
      (t + 1) == nthreadToUse ?
        getNtree() :
        (t + 1) * getNtree() / nthreadToUse,
      t
    );
    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x) { x.join(); }
  );
  #endif

  // If exact, we need to aggregate the predictions by tree seed order.
  if (exact) {
    std::vector<size_t> indices(tree_seeds.size());
    std::iota(indices.begin(), indices.end(), 0);
    //Order the indices by the seeds of the corresponding trees
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) -> bool {
                return tree_seeds[a] > tree_seeds[b];
              });

    size_t weight_index = 0;
    // Now aggregate using the new index ordering
    for (std::vector<size_t>::iterator iter = indices.begin();
        iter != indices.end();
        ++iter)
    {
        size_t cur_index  = *iter;

        double cur_weight = use_weights ? (double) (*tree_weights)[weight_index] : (double) 1.0;
        weight_index++;
        // Aggregate all predictions for current tree
        for (size_t j = 0; j < numObservations; j++) {
          prediction[j] += cur_weight * tree_preds[cur_index][j];
        }

        if (terminalNodes) {
          for (size_t k = 0; k < numObservations; k++) {
            (*terminalNodes)(k, cur_index) = tree_nodes[cur_index][k];
          }
          (*terminalNodes)(numObservations, cur_index) = tree_total_nodes[cur_index];
        }
    }
  }

  for (size_t j=0; j<numObservations; j++){
    prediction[j] /= total_weights;
  }

  std::unique_ptr< std::vector<double> > prediction_ (
    new std::vector<double>(prediction)
  );

  // If we also update the weight matrix, we now have to divide every entry
  // by the number of trees:
  if (weightMatrix) {
    size_t nrow = (*xNew)[0].size();      // number of features to be predicted
    size_t ncol = getNtrain();            // number of train data
    for ( size_t i = 0; i < nrow; i++) {
      for (size_t j = 0; j < ncol; j++) {
        (*weightMatrix)(i,j) = (*weightMatrix)(i,j) / total_weights;
      }
    }
  }

  if (coefficients) {
    for (size_t k = 0; k < numObservations; k++) {
      for (size_t l = 0; l < coefficients->n_cols; l++) {
        (*coefficients)(k,l) /= total_weights;
      }
    }
  }

  return prediction_;
}


std::vector<double> forestry::predictOOB(
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    std::vector<size_t>* treeCounts,
    bool doubleOOB,
    bool exact,
    std::vector<size_t> &training_idx,
    bool hierShrinkage,
    double lambdaShrinkage
) {

  bool use_training_idx = !training_idx.empty();
  size_t numTrainingRows = getTrainingData()->getNumRows();
  size_t numObservations = use_training_idx ? training_idx.size() : numTrainingRows;
  std::vector<double> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

  for (size_t i=0; i<numObservations; i++) {
    outputOOBPrediction[i] = 0;
    outputOOBCount[i] = 0;
  }

  // If we have been giving training indices for the xNew matrix, use these

  // Only needed if exact = TRUE, vector for storing each tree's predictions
  std::vector< std::vector<double> > tree_preds;
  std::vector<size_t> tree_seeds;

    #if DOPARELLEL
      size_t nthreadToUse = getNthread();
      if (nthreadToUse == 0) {
        // Use all threads
        nthreadToUse = std::thread::hardware_concurrency();
      }
      if (isVerbose()) {
        // std::cout << "Calculating OOB parallel using " << nthreadToUse << " threads"
        //                   << std::endl;
      }
      std::vector<std::thread> allThreads(nthreadToUse);
      std::mutex threadLock;

      // For each thread, assign a sequence of tree numbers that the thread
      // is responsible for handling
      for (size_t t = 0; t < nthreadToUse; t++) {
        auto dummyThread = std::bind(
          [&](const int iStart, const int iEnd, const int t_) {
            // loop over all items
            for (int i=iStart; i < iEnd; i++) {
    #else
              // For non-parallel version, just simply iterate all trees serially
              for(int i=0; i<((int) getNtree()); i++ ) {
    #endif
                try {
                  std::vector<double> outputOOBPrediction_iteration(numObservations);
                  std::vector<size_t> outputOOBCount_iteration(numObservations);
                  for (size_t j=0; j<numObservations; j++) {
                    outputOOBPrediction_iteration[j] = 0;
                    outputOOBCount_iteration[j] = 0;
                  }
                  forestryTree *currentTree = (*getForest())[i].get();
                  (*currentTree).getOOBPrediction(
                      outputOOBPrediction_iteration,
                      outputOOBCount_iteration,
                      getTrainingData(),
                      getOOBhonest(),
                      doubleOOB,
                      getMinNodeSizeToSplitAvg(),
                      xNew,
                      weightMatrix,
                      training_idx,
                      hierShrinkage,
                      lambdaShrinkage
                  );
                  #if DOPARELLEL
                  std::lock_guard<std::mutex> lock(threadLock);
                  #endif

                  // based on tree seeds when tree seeds might be uninitialized
                  tree_seeds.push_back(currentTree->getSeed());

                  if (exact) {
                    tree_preds.push_back(outputOOBPrediction_iteration);
                    //std::cout << "Tree predictions "<< std::endl;
                    //print_vector(outputOOBPrediction_iteration);
                    for (size_t j=0; j < numObservations; j++) {
                      outputOOBCount[j] += outputOOBCount_iteration[j];
                    }
                  } else {
                    for (size_t j=0; j < numObservations; j++) {
                      outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
                      outputOOBCount[j] += outputOOBCount_iteration[j];
                    }
                  }

                } catch (std::runtime_error &err) {
                  // std::cerr << err.what() << std::endl;
                }
              }
    #if DOPARELLEL
            },
            t * getNtree() / nthreadToUse,
            (t + 1) == nthreadToUse ?
            getNtree() :
              (t + 1) * getNtree() / nthreadToUse,
              t
        );
        allThreads[t] = std::thread(dummyThread);
          }
          std::for_each(
            allThreads.begin(),
            allThreads.end(),
            [](std::thread& x) { x.join(); }
          );
    #endif

  double OOB_MSE = 0;

  if (exact) {
    std::vector<size_t> indices(tree_seeds.size());
    std::iota(indices.begin(), indices.end(), 0);
    //Order the indices by the seeds of the corresponding trees
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) -> bool {
                return tree_seeds[a] > tree_seeds[b];
              });


    // Now aggregate using the new index ordering
    for (std::vector<size_t>::iterator iter = indices.begin();
         iter != indices.end();
         ++iter)
    {
      size_t cur_index = *iter;

      // Aggregate all predictions for current tree
      for (size_t j = 0; j < numObservations; j++) {
        if (outputOOBCount[j] != 0) {
          outputOOBPrediction[j] += tree_preds[cur_index][j] / outputOOBCount[j];

        } else {
          outputOOBPrediction[j] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
    //Also divide the weightMatrix
    if (weightMatrix) {
      for (size_t j=0; j<numObservations; j++){
        if (outputOOBCount[j] != 0) {
          for (size_t i = 0; i < numTrainingRows; i++) {
            (*weightMatrix)(j,i) = (*weightMatrix)(j,i) / outputOOBCount[j];
          }
            // Set the counts for this tree
            (*treeCounts)[j] = outputOOBCount[j];
        }
      }
    }

  } else {
    for (size_t j=0; j<numObservations; j++){
      double trueValue = getTrainingData()->getOutcomePoint(j);
      if (outputOOBCount[j] != 0) {
        OOB_MSE +=
          pow(trueValue - outputOOBPrediction[j] / outputOOBCount[j], 2);
        outputOOBPrediction[j] = outputOOBPrediction[j] / outputOOBCount[j];
        //Also divide the weightMatrix
        if (weightMatrix) {
          for (size_t i = 0; i < numTrainingRows; i++) {
            (*weightMatrix)(j,i) = (*weightMatrix)(j,i) / outputOOBCount[j];
          }
          (*treeCounts)[j] = outputOOBCount[j];
        }
      } else {
        outputOOBPrediction[j] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  return outputOOBPrediction;
}

void forestry::calculateOOBError(
    bool doubleOOB,
    bool hierShrinkage,
    double lambdaShrinkage
) {

  size_t numObservations = getTrainingData()->getNumRows();

  std::vector<double> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

  std::vector<size_t> training_idx;

  for (size_t i=0; i<numObservations; i++) {
    outputOOBPrediction[i] = 0;
    outputOOBCount[i] = 0;
  }

  #if DOPARELLEL
  size_t nthreadToUse = getNthread();
  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }
  if (isVerbose()) {
    // std::cout << "Calculating OOB parallel using " << nthreadToUse << " threads"
    //           << std::endl;
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (size_t t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const int iStart, const int iEnd, const int t_) {

        // loop over all items
        for (int i=iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for(int i=0; i<((int) getNtree()); i++ ) {
  #endif
          try {
            std::vector<double> outputOOBPrediction_iteration(numObservations);
            std::vector<size_t> outputOOBCount_iteration(numObservations);
            for (size_t j=0; j<numObservations; j++) {
              outputOOBPrediction_iteration[j] = 0;
              outputOOBCount_iteration[j] = 0;
            }
            forestryTree *currentTree = (*getForest())[i].get();
            (*currentTree).getOOBPrediction(
              outputOOBPrediction_iteration,
              outputOOBCount_iteration,
              getTrainingData(),
              getOOBhonest(),
              doubleOOB,
              getMinNodeSizeToSplitAvg(),
              nullptr,
              NULL,
              training_idx,
              hierShrinkage,
              lambdaShrinkage
            );

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            for (size_t j=0; j < numObservations; j++) {
              outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
              outputOOBCount[j] += outputOOBCount_iteration[j];
            }

          } catch (std::runtime_error &err) {
            // std::cerr << err.what() << std::endl;
          }
        }
  #if DOPARELLEL
        },
        t * getNtree() / nthreadToUse,
        (t + 1) == nthreadToUse ?
          getNtree() :
          (t + 1) * getNtree() / nthreadToUse,
        t
    );
    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x) { x.join(); }
  );
  #endif

  double OOB_MSE = 0;
  for (size_t j=0; j<numObservations; j++){
    double trueValue = getTrainingData()->getOutcomePoint(j);
    if (outputOOBCount[j] != 0) {
      OOB_MSE +=
        pow(trueValue - outputOOBPrediction[j] / outputOOBCount[j], 2);
      outputOOBPrediction[j] = outputOOBPrediction[j] / outputOOBCount[j];
    } else {
      outputOOBPrediction[j] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  // Return the MSE and the prediction vector
  this->_OOBError = OOB_MSE /( (double) outputOOBPrediction.size() );
  this->_OOBpreds = outputOOBPrediction;
};


// -----------------------------------------------------------------------------

void forestry::fillinTreeInfo(
    std::unique_ptr< std::vector< tree_info > > & forest_dta
){

  if (isVerbose()) {
    // std::cout << "Starting to translate Forest to R.\n";
  }

  for(int i=0; i<((int) getNtree()); i++ ) {
    // read out each tree and add it to the forest_dta:
    try {
      forestryTree *currentTree = (*getForest())[i].get();
      std::unique_ptr<tree_info> treeInfo_i =
        (*currentTree).getTreeInfo(_trainingData);

      forest_dta->push_back(*treeInfo_i);

    } catch (std::runtime_error &err) {
      // std::cerr << err.what() << std::endl;
    }

    if (isVerbose()) {
      // std::cout << "Done with tree " << i + 1 << " of " << getNtree() << ".\n";
    }

  }

  if (isVerbose()) {
    // std::cout << "Translation done.\n";
  }

  return ;
};


void forestry::reconstructTrees(
    std::unique_ptr< std::vector<size_t> > & categoricalFeatureColsRcpp,
    std::unique_ptr< std::vector<unsigned int> > & tree_seeds,
    std::unique_ptr< std::vector< std::vector<int> >  > & var_ids,
    std::unique_ptr< std::vector< std::vector<int> >  > & average_counts,
    std::unique_ptr< std::vector< std::vector<int> >  > & split_counts,
    std::unique_ptr< std::vector< std::vector<double> >  > & split_vals,
    std::unique_ptr< std::vector< std::vector<int> >  > & naLeftCounts,
    std::unique_ptr< std::vector< std::vector<int> >  > & naRightCounts,
    std::unique_ptr< std::vector< std::vector<int> >  > & naDefaultDirections,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & averagingSampleIndex,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & splittingSampleIndex,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & excludedSampleIndex,
    std::unique_ptr< std::vector< std::vector<double> >  > & weights){

    #if DOPARELLEL
    size_t nthreadToUse = this->getNthread();

    if (nthreadToUse == 0) {
      // Use all threads
      nthreadToUse = std::thread::hardware_concurrency();
    }

    if (isVerbose()) {
      // std::cout << "Reconstructing in parallel using " << nthreadToUse << " threads"
      //                   << std::endl;
    }

    std::vector<std::thread> allThreads(nthreadToUse);
    std::mutex threadLock;

    // For each thread, assign a sequence of tree numbers that the thread
    // is responsible for handling
    for (size_t t = 0; t < nthreadToUse; t++) {
      auto dummyThread = std::bind(
        [&](const int iStart, const int iEnd, const int t_) {

          // loop over al assigned trees, iStart is the starting tree number
          // and iEnd is the ending tree number
          for (int i=iStart; i < iEnd; i++) {
    #else
              // For non-parallel version, just simply iterate all trees serially
    for(int i=0; i<(split_vals->size()); i++ ) {
    #endif

      try{
        forestryTree *oneTree = new forestryTree();

        oneTree->reconstruct_tree(
                getMtry(),
                getMinNodeSizeSpt(),
                getMinNodeSizeAvg(),
                getMinNodeSizeToSplitSpt(),
                getMinNodeSizeToSplitAvg(),
                getMinSplitGain(),
                getMaxDepth(),
                getInteractionDepth(),
                gethasNas(),
                getNaDirection(),
                getlinear(),
                getOverfitPenalty(),
                (*tree_seeds)[i],
                (*categoricalFeatureColsRcpp),
                (*var_ids)[i],
                (*average_counts)[i],
                (*split_counts)[i],
                (*split_vals)[i],
                (*naLeftCounts)[i],
                (*naRightCounts)[i],
                (*naDefaultDirections)[i],
                (*averagingSampleIndex)[i],
                (*splittingSampleIndex)[i],
                (*excludedSampleIndex)[i],
                (*weights)[i]);

#if DOPARELLEL
        std::lock_guard<std::mutex> lock(threadLock);
#endif

        (*getForest()).emplace_back(oneTree);
        _ntree = _ntree + 1;
      } catch (std::runtime_error &err) {
        // std::cerr << err.what() << std::endl;
      }
  }
  #if DOPARELLEL
            },
            t * split_vals->size() / nthreadToUse,
            (t + 1) == nthreadToUse ?
            split_vals->size() :
              (t + 1) * split_vals->size() / nthreadToUse,
              t
        );
        allThreads[t] = std::thread(dummyThread);
          }

          std::for_each(
            allThreads.begin(),
            allThreads.end(),
            [](std::thread& x) { x.join(); }
          );
#endif

  // Try sorting the forest by seed, this way we should do predict in the same order
  std::vector< std::unique_ptr< forestryTree > >* curr_forest;
  curr_forest = this->getForest();
  std::sort(curr_forest->begin(), curr_forest->end(), [](const std::unique_ptr< forestryTree >& a, const std::unique_ptr< forestryTree >& b) {
    return a.get()->getSeed() > b.get()->getSeed();
  });

  return;
}

size_t forestry::getTotalNodeCount() {
  size_t node_count = 0;
  for (size_t i = 0; i < getNtree(); i++) {
    node_count += (*getForest())[i]->getNodeCount();

  }
  return node_count;
}
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
std::vector<std::vector<double>>* forestry::neighborhoodImpute(
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix
) {
  std::vector<size_t>* categoricalCols = getTrainingData()->getCatCols();
  std::vector<size_t>* numericalCols = getTrainingData()->getNumCols();

  for(auto j : *numericalCols) {
    for(size_t i = 0; i < (*xNew)[0].size(); i++) {
      if(std::isnan((*xNew)[j][i])) {
        arma::vec weights = weightMatrix->col(i);
        std::vector<double>* xTrainColj = getTrainingData()->getFeatureData(j);
          double totalWeights = 0;
          double totalProd = 0;
          size_t numRows = getTrainingData()->getNumRows();
          for(size_t k = 0; k < numRows; k++) {
            if(!std::isnan((*xTrainColj)[k])) {
              totalProd = totalProd + (*xTrainColj)[k] * weights(k);
              totalWeights = totalWeights + weights(k);
            }
            (*xNew)[j][i] = totalProd/totalWeights;
          }}}}
  for(auto j : *categoricalCols) {
      for(size_t i = 0; i < (*xNew)[1].size(); i++) {
        if(std::isnan((*xNew)[j][i])) {
          arma::vec weights = weightMatrix->col(i);
          std::vector<double>* xTrainColj = getTrainingData()->getFeatureData(j);
          std::vector<double> categoryContribution;
          categoryContribution.resize(45);
          for(size_t k = 0; k < (*xTrainColj).size(); k++) {
            if(!std::isnan((*xTrainColj)[k])) {
              unsigned int category = round((*xTrainColj)[k]);
              if(category + 1 > categoryContribution.size()) {
                categoryContribution.resize(category + 1);
              }
              categoryContribution[round(category)] += weights(k);
            }}
          // Find position of max weight. In principle this can
          // be done with std::which_max, std::distance. But this
          // is inefficient because we have to iterate over the
          // vector about 1.5 times.
          double runningMax = -std::numeric_limits<double>::infinity();
          size_t maxPosition=0;
          for(size_t l = 0; l < categoryContribution.size(); l++) {
            if(categoryContribution[l] > runningMax) {
              runningMax = categoryContribution[l];
              maxPosition = l;
            }}
          (*xNew)[j][i] = maxPosition;
        }}}
  return xNew;
}
