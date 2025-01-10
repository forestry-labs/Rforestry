#include "DataFrame.h"

DataFrame::DataFrame():
  _featureData(nullptr), _outcomeData(nullptr), _rowNumbers(nullptr),
  _categoricalFeatureCols(nullptr), _numericalFeatureCols(nullptr),
  _linearFeatureCols(nullptr), _numRows(0), _numColumns(0),
  _featureWeights(nullptr), _featureWeightsVariables(nullptr),  _deepFeatureWeights(nullptr),
  _deepFeatureWeightsVariables(nullptr), _observationWeights(nullptr), _monotonicConstraints(nullptr){}

DataFrame::~DataFrame() {
//  std::cout << "DataFrame() destructor is called." << std::endl;
}

DataFrame::DataFrame(
  std::shared_ptr< std::vector< std::vector<double> > > featureData,
  std::unique_ptr< std::vector<double> > outcomeData,
  std::unique_ptr< std::vector<size_t> > categoricalFeatureCols,
  std::unique_ptr< std::vector<size_t> > linearFeatureCols,
  std::size_t numRows,
  std::size_t numColumns,
  std::unique_ptr<std::vector<double>> featureWeights,
  std::unique_ptr<std::vector<size_t>> featureWeightsVariables,
  std::unique_ptr<std::vector<double>> deepFeatureWeights,
  std::unique_ptr<std::vector<size_t>> deepFeatureWeightsVariables,
  std::unique_ptr< std::vector<double> > observationWeights,
  std::shared_ptr< std::vector<int> > monotonicConstraints
) {
  this->_featureData = std::move(featureData);
  this->_outcomeData = std::move(outcomeData);
  this->_categoricalFeatureCols = std::move(categoricalFeatureCols);
  this->_linearFeatureCols = std::move(linearFeatureCols);
  this->_numRows = numRows;
  this->_numColumns = numColumns;
  this->_featureWeights = std::move(featureWeights);
  this->_featureWeightsVariables = std::move(featureWeightsVariables);
  this->_deepFeatureWeights = std::move(deepFeatureWeights);
  this->_deepFeatureWeightsVariables = std::move(deepFeatureWeightsVariables);
  this->_observationWeights = std::move(observationWeights);
  this->_monotonicConstraints = std::move(monotonicConstraints);

  // define the row numbers to be the numbers from 1 to nrow:
  std::vector<size_t> rowNumberss;
  for(size_t j=0; j<numRows; j++){
    rowNumberss.push_back(j+1);
  }
  std::unique_ptr< std::vector<size_t> > rowNumbers (
      new std::vector<size_t>(rowNumberss));
  this->_rowNumbers = std::move(rowNumbers);

  // Add numericalFeatures
  std::vector<size_t> numericalFeatureColss;
  std::vector<size_t>* catCols = this->getCatCols();

  for (size_t i = 0; i < numColumns; i++) {
    if (std::find(catCols->begin(), catCols->end(), i) == catCols->end()) {
      numericalFeatureColss.push_back(i);
    }
  }

  std::unique_ptr< std::vector<size_t> > numericalFeatureCols (
      new std::vector<size_t>(numericalFeatureColss));
  this->_numericalFeatureCols = std::move(numericalFeatureCols);
}

double DataFrame::getPoint(size_t rowIndex, size_t colIndex) {
  // Check if rowIndex and colIndex are valid
  if (rowIndex < getNumRows() && colIndex < getNumColumns()) {
    return (*getAllFeatureData())[colIndex][rowIndex];
  } else {
    throw std::runtime_error("Invalid rowIndex or colIndex.");
  }
}

double DataFrame::getOutcomePoint(size_t rowIndex) {
  // Check if rowIndex is valid
  if (rowIndex < getNumRows()) {
    return (*getOutcomeData())[rowIndex];
  } else {
    throw std::runtime_error("Invalid rowIndex.");
  }
}

std::vector<double>* DataFrame::getFeatureData(
  size_t colIndex
) {
  if (colIndex < getNumColumns()) {
    return &(*getAllFeatureData())[colIndex];
  } else {
    throw std::runtime_error("Invalid colIndex.");
  }
}

std::vector<double> DataFrame::getLinObsData(
  size_t rowIndex
) {
  if (rowIndex < getNumRows()) {
    std::vector<double> feat;
    for (size_t i = 0; i < getLinCols()->size(); i++) {
      feat.push_back(getPoint(rowIndex, (*getLinCols())[i]));
    }
    return feat;
  } else {
    throw std::runtime_error("Invalid rowIndex");
  }
}

void DataFrame::getObservationData(
  std::vector<double> &rowData,
  size_t rowIndex
) {
  if (rowIndex < getNumRows()) {
    for (size_t i=0; i < getNumColumns(); i++) {
      rowData[i] = (*getAllFeatureData())[i][rowIndex];
    }
  } else {
    throw std::runtime_error("Invalid rowIndex.");
  }
}

void DataFrame::getShuffledObservationData(
  std::vector<double> &rowData,
  size_t rowIndex,
  size_t swapFeature,
  size_t swapIndex
) {
  // Helper that gives rowIndex observation with
  // swapFeature value of swapIndex observation
  if (rowIndex < getNumRows() && swapFeature < getNumColumns()) {
    for (size_t i=0; i < getNumColumns(); i++) {
      rowData[i] = (*getAllFeatureData())[i][rowIndex];
    }
    rowData[swapFeature] = getPoint(swapIndex, swapFeature);
  } else {
    throw std::runtime_error("Invalid row/colIndex.");
  }
}

double DataFrame::partitionMean(
  std::vector<size_t>* sampleIndex
){
  size_t totalSampleSize = (*sampleIndex).size();
  double accummulatedSum = 0;
  for (
    std::vector<size_t>::iterator it = (*sampleIndex).begin();
    it != (*sampleIndex).end();
    ++it
  ) {
    accummulatedSum += getOutcomePoint(*it);
  }
  return accummulatedSum / totalSampleSize;
}

std::vector<size_t> DataFrame::get_all_row_idx(
    std::vector<size_t>* sampleIndex
){
  std::vector<size_t> idx;
  for (
      std::vector<size_t>::iterator it = (*sampleIndex).begin();
      it != (*sampleIndex).end();
      ++it
  ) {
    idx.push_back(get_row_idx(*it));
  }
  return idx;
}

size_t DataFrame::get_row_idx(size_t rowIndex) {
  // Check if rowIndex is valid
  if (rowIndex < getNumRows()) {
    return (*getRowNumbers())[rowIndex];
  } else {
    throw std::runtime_error("rowIndex is too large");
  }
}

void DataFrame::setOutcomeData(std::vector<double> outcomeData) {
  std::unique_ptr<std::vector<double> > outcomeData_(
      new std::vector<double>(outcomeData)
  );
  this->_outcomeData =std::move(outcomeData_);
}
