// [[Rcpp::depends(RcppThread)]]
// [[Rcpp::plugins(cpp11)]]
#include "forestry.h"
#include "utils.h"
#include <RcppThread.h>
#include <random>
#include <thread>
#include <mutex>
#include <RcppArmadillo.h>
#define DOPARELLEL true


forestry::forestry():
  _trainingData(nullptr), _ntree(0), _replace(0), _sampSize(0),
  _splitRatio(0), _mtry(0), _minNodeSizeSpt(0), _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0), _minNodeSizeToSplitAvg(0), _minSplitGain(0),
  _maxDepth(0), _interactionDepth(0), _forest(nullptr), _seed(0), _verbose(0), _nthread(0),
  _OOBError(0), _splitMiddle(0), _doubleTree(0){};

forestry::~forestry(){
//  for (std::vector<forestryTree*>::iterator it = (*_forest).begin();
//       it != (*_forest).end();
//       ++it) {
//    delete(*it);
//  }
//  std::cout << "forestry() destructor is called." << std::endl;
};

forestry::forestry(
  DataFrame* trainingData,
  size_t ntree,
  bool replace,
  size_t sampSize,
  double splitRatio,
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
  bool hasNas,
  bool linear,
  double overfitPenalty,
  bool doubleTree
){
  this->_trainingData = trainingData;
  this->_ntree = 0;
  this->_replace = replace;
  this->_sampSize = sampSize;
  this->_splitRatio = splitRatio;
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
  this->_linear = linear;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;

  if (splitRatio > 1 || splitRatio < 0) {
    throw std::runtime_error("splitRatio shoule be between 0 and 1.");
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
    throw std::runtime_error("splitRatio too big or too small.");
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
  // forestryTree *currentTree;
  // Rcpp::Rcout << "Tree seeds:" << std::endl;
  // R_FlushConsole();
  // R_ProcessEvents();
  // R_CheckUserInterrupt();
  //
  // Rcpp::Rcout << "c(";
  // R_FlushConsole();
  // R_ProcessEvents();
  // R_CheckUserInterrupt();
  //
  // std::vector<unsigned int> seeds;
  // for (auto p = 0; p < (*this->getForest()).size(); p++) {
  //   currentTree = (*this->getForest())[p].get();
  //   seeds.push_back(currentTree->getSeed());
  //   //currentTree->printTree();
  // }
  // std::sort(seeds.begin(), seeds.end());
  // print_vector(seeds);
  //
  // Rcpp::Rcout << std::endl;
  // RcppThread::Rcout << "SORTING NOW";
  // R_FlushConsole();
  // R_ProcessEvents();
  // R_CheckUserInterrupt();

  // Try sorting the forest by seed, this way we should do predict in the same order
  std::vector< std::unique_ptr< forestryTree > >* curr_forest;
  curr_forest = this->getForest();
  std::sort(curr_forest->begin(), curr_forest->end(), [](const std::unique_ptr< forestryTree >& a,
                                                         const std::unique_ptr< forestryTree >& b) {
    return a.get()->getSeed() > b.get()->getSeed();
  });

  // std::unique_ptr< std::vector< std::unique_ptr< forestryTree > > > sorted_forest (
  //     new std::vector< std::unique_ptr< forestryTree > >(*curr_forest)
  // );
  //
  // this->_forest = std::move(sorted_forest);
}

void forestry::addTrees(size_t ntree) {

  const unsigned int newStartingTreeNumber = (unsigned int) getNtree();
  const unsigned int newEndingTreeNumber = newStartingTreeNumber + (unsigned int) ntree;

  unsigned int nthreadToUse = (unsigned int) getNthread();
  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = (unsigned int) std::thread::hardware_concurrency();
  }
  const unsigned int see = this->getSeed();

  size_t splitSampleSize = (size_t) (getSplitRatio() * getSampleSize());
  float percent_complete = 0.0;

  // Rcpp::Rcout << "Training Progress: " << std::endl;
  // R_FlushConsole();
  // R_ProcessEvents();
  //
  // Rcpp::Rcout << "|";
  // for (size_t y = 0; y < 98;y++) {
  //   Rcpp::Rcout << " ";
  //   R_FlushConsole();
  //   R_ProcessEvents();
  // }
  // Rcpp::Rcout << "|" << std::endl;

  // std::vector<unsigned int> seeds;
  // for (unsigned int k = newStartingTreeNumber; k < newEndingTreeNumber; k++) {
  //   seeds.push_back((k+1)*see);
  //   Rcpp::Rcout << (k+1)*see << ", ";
  //   R_FlushConsole();
  //   R_ProcessEvents();
  //   R_CheckUserInterrupt();
  // }
  //print_vector(seeds);

  #if DOPARELLEL
  if (isVerbose()) {
    RcppThread::Rcout << "Training parallel using " << nthreadToUse << " threads"
              << std::endl;
    R_FlushConsole();
    R_ProcessEvents();
    R_CheckUserInterrupt();
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

          const unsigned int myseed = (i+1)*see;   // This line is bad, we are multiplying i, which previously
          // if (myseed > 90 || myseed < 1) {
          //   myseed = 1000;
          // }
          std::mt19937_64 random_number_generator;           // was an int with myseed which is an unsigned int
          random_number_generator.seed(myseed);              // Testing a couple of things, this is deterministic with
                                                            // this->getSeed(), but not with i, or i multuplied.
                                                            // i Must be changing run to run somehow
          // std::uniform_real_distribution<double> unif;

          // Generate a sample index for each tree
          std::vector<size_t> sampleIndex;

          if (isReplacement()) {
            // std::uniform_int_distribution<size_t> unif_dist(
            //   0, (size_t) (*getTrainingData()).getNumRows() - 1
            // );

            // Now we generate a weighted distribution using observationWeights
            std::vector<double>* sampleWeights = (this->getTrainingData()->getobservationWeights());
            std::discrete_distribution<size_t> sample_dist(
              sampleWeights->begin(), sampleWeights->end()
            );

            // Generate index with replacement
            while (sampleIndex.size() < getSampleSize()) {
              size_t randomIndex = sample_dist(random_number_generator);
              sampleIndex.push_back(randomIndex);
            }
          } else {
            // In this case, when we have no replacement, we disregard
            // observationWeights and use a uniform distribution
            std::uniform_int_distribution<size_t> unif_dist(
              0, (size_t) (*getTrainingData()).getNumRows() - 1
            );
            // Generate index without replacement
            while (sampleIndex.size() < getSampleSize()) {
              size_t randomIndex = unif_dist(random_number_generator);

              if (
                sampleIndex.size() == 0 ||
                std::find(
                  sampleIndex.begin(),
                  sampleIndex.end(),
                  randomIndex
                ) == sampleIndex.end()
              ) {
                sampleIndex.push_back(randomIndex);
              }
            }
          }

          std::unique_ptr<std::vector<size_t> > splitSampleIndex;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex;

          std::unique_ptr<std::vector<size_t> > splitSampleIndex2;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex2;

          if (getSplitRatio() == 1 || getSplitRatio() == 0) {

            // Treat it as normal RF
            splitSampleIndex.reset(new std::vector<size_t>(sampleIndex));
            averageSampleIndex.reset(new std::vector<size_t>(sampleIndex));

          } else {

            // Generate sample index based on the split ratio
            std::vector<size_t> splitSampleIndex_;
            std::vector<size_t> averageSampleIndex_;
            for (
              std::vector<size_t>::iterator it = sampleIndex.begin();
              it != sampleIndex.end();
              ++it
            ) {
              if (splitSampleIndex_.size() < splitSampleSize) {
                splitSampleIndex_.push_back(*it);
              } else {
                averageSampleIndex_.push_back(*it);
              }
            }

            splitSampleIndex.reset(
              new std::vector<size_t>(splitSampleIndex_)
            );
            averageSampleIndex.reset(
              new std::vector<size_t>(averageSampleIndex_)
            );

            if (_doubleTree) {
              splitSampleIndex2.reset(
                new std::vector<size_t>(splitSampleIndex_)
              );
              averageSampleIndex2.reset(
                new std::vector<size_t>(averageSampleIndex_)
              );
            }
          }

          try{
            // Rcpp::Rcout << "Tree seed is " << getSeed() << std::endl;
            // R_FlushConsole();
            // R_ProcessEvents();
            // R_CheckUserInterrupt();
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
                random_number_generator,
                getSplitMiddle(),
                getMaxObs(),
                gethasNas(),
                getlinear(),
                getOverfitPenalty(),
                myseed // getSeed()
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
                    random_number_generator,
                    getSplitMiddle(),
                    getMaxObs(),
                    gethasNas(),
                    getlinear(),
                    getOverfitPenalty(),
                    myseed
                 );
            }

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            if (isVerbose()) {
              Rcpp::Rcout << "Finish training tree # " << (i + 1) << std::endl;
              R_FlushConsole();
              R_ProcessEvents();
            }
            // RcppThread::Rcout << (i+1)*see << ",";
            // R_FlushConsole();
            // R_ProcessEvents();
            //oneTree->printTree();


            // R_CheckUserInterrupt();

            (*getForest()).emplace_back(oneTree);
            _ntree = _ntree + 1;
            if (_doubleTree) {
              (*getForest()).emplace_back(anotherTree);
              _ntree = _ntree + 1;
            } else {
              // delete anotherTree;
            }



            // size_t new_dots = (size_t) ((float) _ntree / ntree - percent_complete);
            // percent_complete = (float) _ntree / ntree;
            // for (size_t q = 0; q < 2; q++) {
            //   Rcpp::Rcout << "*";
            //   R_FlushConsole();
            //   R_ProcessEvents();
            //   //RcppThread::checkUserInterrupt();
            // }

            // R_FlushConsole();
            // R_ProcessEvents();
          } catch (std::runtime_error &err) {
            // Rcpp::Rcerr << err.what() << std::endl;
          }

        }
  #if DOPARELLEL
      },
      newStartingTreeNumber + t * ntree / nthreadToUse,
      (t + 1) == nthreadToUse ?
        (unsigned int) newEndingTreeNumber :
           newStartingTreeNumber + (t + 1) * ntree / nthreadToUse,
           t
    );
    // this is a problem, we are apparently casting
    // this to a size_t even though we are iterating through
    // and multiplying it with an unsigned int for the seeds

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
  arma::Mat<int>* terminalNodes,
  unsigned int seed,
  size_t nthread
){
  std::vector<double> prediction;
  size_t numObservations = (*xNew)[0].size();
  for (size_t j=0; j<numObservations; j++) {
    prediction.push_back(0);
  }

  #if DOPARELLEL
  size_t nthreadToUse;
  if (nthread == 1) {
    nthreadToUse = 1;
  } else {
    nthreadToUse = getNthread();
  }

  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }

  if (isVerbose()) {
    RcppThread::Rcout << "Prediction parallel using " << nthreadToUse << " threads"
              << std::endl;
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

            //If terminal nodes, pass option to tree predict
            forestryTree *currentTree = (*getForest())[i].get();
            (*currentTree).predict(
              currentTreePrediction,
              &currentTreeTerminalNodes,
              xNew,
              getTrainingData(),
              weightMatrix,
              getlinear(),
              seed + i
            );


            // HERE IF NEED TERMINAL NODES, pass option to tree predict, then
            // lock thread (shouldn't really need to), use i as offset and flip
            // bool of matrix

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            # endif

            for (size_t j = 0; j < numObservations; j++) {
              prediction[j] += currentTreePrediction[j];
            }

            if (terminalNodes) {
              for (size_t k = 0; k < numObservations; k++) {
                (*terminalNodes)(k, i) = currentTreeTerminalNodes[k];
              }
              (*terminalNodes)(numObservations, i) = (*currentTree).getNodeCount();
            }

          } catch (std::runtime_error &err) {
            Rcpp::Rcerr << err.what() << std::endl;
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

  for (size_t j=0; j<numObservations; j++){
    prediction[j] /= getNtree();
  }

  std::unique_ptr< std::vector<double> > prediction_ (
    new std::vector<double>(prediction)
  );


  // If we also update the weight matrix, we now have to divide every entry
  // by the number of trees:

  if (weightMatrix) {
    size_t nrow = (*xNew)[0].size(); // number of features to be predicted
    size_t ncol = getNtrain(); // number of train data
    for ( size_t i = 0; i < nrow; i++){
      for (size_t j = 0; j < ncol; j++){
        (*weightMatrix)(i,j) = (*weightMatrix)(i,j) / _ntree;
      }
    }
  }

  return prediction_;
}

void forestry::calculateVariableImportance() {
  // For all variables, shuffle + get OOB Error, record in

  size_t numObservations = getTrainingData()->getNumRows();
  std::vector<double> variableImportances;

  std::vector<double> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

  //Loop through all features and populate variableImportances with shuffled OOB
  for (size_t featNum = 0; featNum < getTrainingData()->getNumColumns(); featNum++) {

    // Initialize MSEs/counts
    for (size_t i=0; i<numObservations; i++) {
      outputOOBPrediction[i] = 0;
      outputOOBCount[i] = 0;
    }
    //Use same parallelization scheme as before

    #if DOPARELLEL
    size_t nthreadToUse = getNthread();
    if (nthreadToUse == 0) {
      nthreadToUse = std::thread::hardware_concurrency();
    }
    if (isVerbose()) {
      RcppThread::Rcout << "Calculating OOB parallel using " << nthreadToUse << " threads"
                << std::endl;
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
      unsigned int myseed = getSeed() * (i + 1);
      std::mt19937_64 random_number_generator;
      random_number_generator.seed(myseed);
        try {
          std::vector<double> outputOOBPrediction_iteration(numObservations);
          std::vector<size_t> outputOOBCount_iteration(numObservations);
          for (size_t j=0; j<numObservations; j++) {
            outputOOBPrediction_iteration[j] = 0;
            outputOOBCount_iteration[j] = 0;
          }
          forestryTree *currentTree = (*getForest())[i].get();
          (*currentTree).getShuffledOOBPrediction(
              outputOOBPrediction_iteration,
              outputOOBCount_iteration,
              getTrainingData(),
              featNum,
              random_number_generator
          );
          #if DOPARELLEL
          std::lock_guard<std::mutex> lock(threadLock);
          #endif
          for (size_t j=0; j < numObservations; j++) {
            outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
            outputOOBCount[j] += outputOOBCount_iteration[j];
          }
        } catch (std::runtime_error &err) {
          Rcpp::Rcerr << err.what() << std::endl;
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

      double current_MSE = 0;
      for (size_t j = 0; j < numObservations; j++){
        double trueValue = getTrainingData()->getOutcomePoint(j);
        if (outputOOBCount[j] != 0) {
          current_MSE +=
            pow(trueValue - outputOOBPrediction[j] / outputOOBCount[j], 2);
        }
      }
      variableImportances.push_back(current_MSE);
  }

  std::unique_ptr<std::vector<double> > variableImportances_(
      new std::vector<double>(variableImportances)
  );

  // Populate forest's variable importance with all shuffled MSE's
  this-> _variableImportance = std::move(variableImportances_);
}

void forestry::calculateOOBError() {

  size_t numObservations = getTrainingData()->getNumRows();

  std::vector<double> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

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
    RcppThread::Rcout << "Calculating OOB parallel using " << nthreadToUse << " threads"
              << std::endl;
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
              getTrainingData()
            );

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            for (size_t j=0; j < numObservations; j++) {
              outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
              outputOOBCount[j] += outputOOBCount_iteration[j];
            }

          } catch (std::runtime_error &err) {
            // Rcpp::Rcerr << err.what() << std::endl;
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

  this->_OOBError = OOB_MSE;
  this->_OOBpreds = outputOOBPrediction;
};


// -----------------------------------------------------------------------------

void forestry::fillinTreeInfo(
    std::unique_ptr< std::vector< tree_info > > & forest_dta
){

  if (isVerbose()) {
    RcppThread::Rcout << "Starting to translate Forest to R.\n";
  }

  for(int i=0; i<((int) getNtree()); i++ ) {
    // read out each tree and add it to the forest_dta:
    try {
      forestryTree *currentTree = (*getForest())[i].get();
      std::unique_ptr<tree_info> treeInfo_i =
        (*currentTree).getTreeInfo(_trainingData);

      forest_dta->push_back(*treeInfo_i);

    } catch (std::runtime_error &err) {
      Rcpp::Rcerr << err.what() << std::endl;

    }

    if (isVerbose()) {
      RcppThread::Rcout << "Done with tree " << i + 1 << " of " << getNtree() << ".\n";
    }

  }

  if (isVerbose()) {
    RcppThread::Rcout << "Translation done.\n";
  }

  return ;
};

void forestry::reconstructTrees(
    std::unique_ptr< std::vector<size_t> > & categoricalFeatureColsRcpp,
    std::unique_ptr< std::vector< std::vector<int> >  > & var_ids,
    std::unique_ptr< std::vector< std::vector<double> >  > & split_vals,
    std::unique_ptr< std::vector< std::vector<int> >  > & naLeftCounts,
    std::unique_ptr< std::vector< std::vector<int> >  > & naRightCounts,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & leafAveidxs,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & leafSplidxs,
    std::unique_ptr< std::vector< std::vector<size_t> >  > &
      averagingSampleIndex,
    std::unique_ptr< std::vector< std::vector<size_t> >  > &
      splittingSampleIndex){

    for (size_t i=0; i<split_vals->size(); i++) {
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
                getlinear(),
                getOverfitPenalty(),
                (*categoricalFeatureColsRcpp),
                (*var_ids)[i],
                (*split_vals)[i],
                (*naLeftCounts)[i],
                (*naRightCounts)[i],
                (*leafAveidxs)[i],
                (*leafSplidxs)[i],
                (*averagingSampleIndex)[i],
                (*splittingSampleIndex)[i]);

        (*getForest()).emplace_back(oneTree);
        _ntree = _ntree + 1;
      } catch (std::runtime_error &err) {
        Rcpp::Rcerr << err.what() << std::endl;
      }

  }

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
          size_t maxPosition;
          for(size_t l = 0; l < categoryContribution.size(); l++) {
            if(categoryContribution[l] > runningMax) {
              runningMax = categoryContribution[l];
              maxPosition = l;
            }}
          (*xNew)[j][i] = maxPosition;
        }}}
  return xNew;
  //return weightMatrix;

}
