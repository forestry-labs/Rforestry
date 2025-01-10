#include "multilayerForestry.h"
#include <random>
#include <thread>
#include <mutex>
#include "DataFrame.h"
#include "forestry.h"
#include "utils.h"
#include <RcppArmadillo.h>

multilayerForestry::multilayerForestry():
  _multilayerForests(nullptr), _gammas(0) {}

multilayerForestry::~multilayerForestry() {};

multilayerForestry::multilayerForestry(
  DataFrame* trainingData,
  size_t ntree,
  size_t nrounds,
  double eta,
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
  unsigned int seed,
  size_t nthread,
  bool verbose,
  bool splitMiddle,
  size_t maxObs,
  bool linear,
  double overfitPenalty,
  bool doubleTree
){
  this->_trainingData = trainingData;
  this->_ntree = ntree;
  this->_nrounds= nrounds;
  this->_eta = eta;
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
  this->_seed = seed;
  this->_nthread = nthread;
  this->_verbose = verbose;
  this->_splitMiddle = splitMiddle;
  this->_maxObs = maxObs;
  this->_linear = linear;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;

  addForests(ntree);
}

void multilayerForestry::addForests(size_t ntree) {

  // Create vectors to store gradient boosted forests and gamma values
  std::vector< forestry* > multilayerForests(_nrounds);
  std::vector<double> gammas(_nrounds);

  // Calculate initial prediction
  DataFrame *trainingData = getTrainingData();
  std::vector<double> outcomeData = *(trainingData->getOutcomeData());
  double meanOutcome =
    accumulate(outcomeData.begin(), outcomeData.end(), 0.0) / outcomeData.size();
  this->_meanOutcome = meanOutcome;

  std::vector<double> predictedOutcome(trainingData->getNumRows(), meanOutcome);

  // Apply gradient boosting to predict residuals
  std::vector<double> residuals(trainingData->getNumRows());
  for (size_t o = 0; o < getNrounds(); o++) {
    std::transform(outcomeData.begin(), outcomeData.end(),
                   predictedOutcome.begin(), residuals.begin(), std::minus<double>());

    std::shared_ptr< std::vector< std::vector<double> > > residualFeatureData_(
        new std::vector< std::vector<double> >(*(trainingData->getAllFeatureData()))
    );
    std::unique_ptr< std::vector<double> > residuals_(
        new std::vector<double>(residuals)
    );
    std::unique_ptr< std::vector<size_t> > residualCatCols_(
        new std::vector<size_t>(*(trainingData->getCatCols()))
    );
    std::unique_ptr< std::vector<size_t> > residualLinCols_(
        new std::vector<size_t>(*(trainingData->getLinCols()))
    );
    std::unique_ptr< std::vector<double> > residualfeatureWeights_(
        new std::vector<double>(*(trainingData->getfeatureWeights()))
    );
    std::unique_ptr< std::vector<size_t> > residualfeatureWeightsVariables_(
        new std::vector<size_t>(*(trainingData->getfeatureWeightsVariables()))
    );
    std::unique_ptr< std::vector<double> > residualdeepFeatureWeights_(
        new std::vector<double>(*(trainingData->getdeepFeatureWeights()))
    );
    std::unique_ptr< std::vector<size_t> > residualdeepFeatureWeightsVariables_(
        new std::vector<size_t>(*(trainingData->getdeepFeatureWeightsVariables()))
    );
    std::unique_ptr< std::vector<double> > residualobservationWeights_(
        new std::vector<double>(*(trainingData->getobservationWeights()))
    );
    std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp_(
        new std::vector<int>(*(trainingData->getMonotonicConstraints()))
    );

    DataFrame* residualDataFrame = new DataFrame(
      residualFeatureData_,
      std::move(residuals_),
      std::move(residualCatCols_),
      std::move(residualLinCols_),
      trainingData->getNumRows(),
      trainingData->getNumColumns(),
      std::move(residualfeatureWeights_),
      std::move(residualfeatureWeightsVariables_),
      std::move(residualdeepFeatureWeights_),
      std::move(residualdeepFeatureWeightsVariables_),
      std::move(residualobservationWeights_),
      std::move(monotonicConstraintsRcpp_)
    );

    forestry *residualForest = new forestry(
      residualDataFrame,
      _ntree,
      _replace,
      _sampSize,
      _splitRatio,
      _mtry,
      _minNodeSizeSpt,
      _minNodeSizeAvg,
      _minNodeSizeToSplitSpt,
      _minNodeSizeToSplitAvg,
      _minSplitGain,
      _maxDepth,
      _maxDepth, //set interactionDepth to maxDepth for multilayerforestry
      _seed,   // Is this seed being set each time??
      _nthread,
      _verbose,
      _splitMiddle,
      _maxObs,
      false,  // This is hasNAs being set to false
      _linear,
      _overfitPenalty,
      _doubleTree
    );

    multilayerForests[o] = residualForest;

    // In multilayer forestry, we predict with only a single thread
    // this is because when we predict with different numbers of threads,
    // the nondeterministic order of the thread predictions can introduce
    // nondeterministic predictions, due to the non associative nature of
    // floating point numbers. This is a very small problem for one round of
    // predictions, but when we continue to train forests on the residuals of
    // previous forests, the problem spirals out of control.
    std::unique_ptr< std::vector<double> > predictedResiduals =
      residualForest->predict(getTrainingData()->getAllFeatureData(),
                              NULL,
                              NULL,
                              _seed,
                              1);

    // Calculate and store best gamma value
    // std::vector<double> bestPredictedResiduals(trainingData->getNumRows());
    // double minMeanSquaredError = INFINITY;
    // static inline double computeSquare (double x) { return x*x; }

    // for (double gamma = -1; gamma <= 1; gamma += 0.02) {
    //   std::vector<double> gammaPredictedResiduals(trainingData->getNumRows());
    //   std::vector<double> gammaError(trainingData->getNumRows());
    //
    //   // Find gamma with smallest mean squared error
    //   std::transform(predictedResiduals->begin(), predictedResiduals->end(),
    //                  gammaPredictedResiduals.begin(), std::bind(std::multiplies<double>(), gamma));
    //   std::transform(predictedOutcome->begin(), predictedOutcome->end(),
    //                  gammaPredictedResiduals.begin(), gammaError.begin(), std::plus<double>());
    //   std::transform(outcomeData->begin(), outcomeData->end(),
    //                  gammaError.begin(), gammaError.begin(), std::minus<double>());
    //   std::transform(gammaError.begin(), gammaError.end(), gammaError.begin(), computeSquare);
    //   double gammaMeanSquaredError =
    //     accumulate(gammaError.begin(), gammaError.end(), 0.0)/gammaError.size();
    //   std::cout << gammaMeanSquaredError << std::endl;
    //
    //   if (gammaMeanSquaredError < minMeanSquaredError) {
    //
    //     gammas[i] = (gamma * eta);
    //     minMeanSquaredError = gammaMeanSquaredError;
    //     bestPredictedResiduals = gammaPredictedResiduals;
    //   }
    // }

    // Store the current parameter for learning rate
    gammas[o] = (double) 1 * _eta;

    // Multiply residuals by learning rate
    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind(std::multiplies<double>(), gammas[o], std::placeholders::_1));

    // Update prediction after each round of gradient boosting
    std::transform(predictedOutcome.begin(), predictedOutcome.end(),
                   predictedResiduals->begin(), predictedOutcome.begin(), std::plus<double>());
  }

  // Save vector of forestry objects and gamma values
  std::unique_ptr<std::vector< forestry* > > multilayerForests_(
    new std::vector< forestry* >(multilayerForests)
  );

  this->_multilayerForests = std::move(multilayerForests_);
  this->_gammas = std::move(gammas);
}

std::unique_ptr< std::vector<double> > multilayerForestry::predict(
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    int seed
) {
  std::vector< forestry* > multilayerForests = *getMultilayerForests();
  std::vector<double> gammas = getGammas();

  // For now we let nthread = 1 for multilayer predictions, for reproducibility
  // this can be changed later

  std::unique_ptr< std::vector<double> > initialPrediction =
    multilayerForests[0]->predict(xNew,
                                  weightMatrix,
                                  NULL,
                                  seed,
                                  this->getNthread());

  std::vector<double> prediction(initialPrediction->size(), getMeanOutcome());

  // Use forestry objects and gamma values to make prediction
  for (size_t i = 0; i < getNrounds(); i ++) {


    // for (int j = 0; j < multilayerForests[i]->getForest()->size(); j++) {
    //   print_vector(*multilayerForests[i]->getTrainingData()->getOutcomeData());
    // }

    std::unique_ptr< std::vector<double> > predictedResiduals =
      multilayerForests[i]->predict(xNew,
                                    weightMatrix,
                                    NULL,
                                    seed,
                                    this->getNthread());

    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind(std::multiplies<double>(), gammas[i], std::placeholders::_1));

    std::transform(prediction.begin(), prediction.end(),
                   predictedResiduals->begin(), prediction.begin(), std::plus<double>());
  }

  std::unique_ptr< std::vector<double> > prediction_ (
      new std::vector<double>(prediction)
  );

  return prediction_;
}


void multilayerForestry::reconstructForests(
    std::vector< forestry* >& multilayerForests,
    std::vector<double>& gammas
) {
  for (size_t i = 0; i < multilayerForests.size(); i++) {
    // Add the gamma list to the multilayerForest object
    this->_gammas.push_back(gammas[i]);

    // Add the forest list to the multilayerForest object
    this->_multilayerForests->push_back(multilayerForests[i]);
    _nrounds++;
  }
  return;
}
