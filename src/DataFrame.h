#ifndef FORESTRYCPP_DATAFRAME_H
#define FORESTRYCPP_DATAFRAME_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

class DataFrame {

public:
  DataFrame();
  virtual ~DataFrame();

  DataFrame(
    std::shared_ptr< std::vector< std::vector<double> > > featureData,
    std::unique_ptr< std::vector<double> > outcomeData,
    std::unique_ptr< std::vector<size_t> > categoricalFeatureCols,
    std::unique_ptr< std::vector<size_t> > linearCols,
    std::size_t numRows,
    std::size_t numColumns,
    std::unique_ptr< std::vector<double> > featureWeights,
    std::unique_ptr< std::vector<size_t> > featureWeightsVariables,
    std::unique_ptr< std::vector<double> > deepFeatureWeights,
    std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariables,
    std::unique_ptr< std::vector<double> > observationWeights,
    std::shared_ptr< std::vector<int> > monotonicConstraints
  );

  double getPoint(size_t rowIndex, size_t colIndex);

  double getOutcomePoint(size_t rowIndex);

  std::vector<double>* getFeatureData(size_t colIndex);

  std::vector<double> getLinObsData(size_t rowIndex);

  void getObservationData(std::vector<double> &rowData, size_t rowIndex);

  void getShuffledObservationData(std::vector<double> &rowData, size_t rowIndex,
                                  size_t swapFeature, size_t swapIndex);

  double partitionMean(std::vector<size_t>* sampleIndex);

  std::vector< std::vector<double> >* getAllFeatureData() {
    return _featureData.get();
  }

  std::vector<double>* getOutcomeData() {
    return _outcomeData.get();
  }

  size_t getNumColumns() {
    return _numColumns;
  }

  size_t getNumRows() {
    return _numRows;
  }

  std::vector<size_t>* getCatCols() {
    return _categoricalFeatureCols.get();
  }

  std::vector<size_t>* getNumCols() {
    return _numericalFeatureCols.get();
  }

  std::vector<size_t>* getLinCols() {
    return _linearFeatureCols.get();
  }

  std::vector<double>* getfeatureWeights() {
    return _featureWeights.get();
  }

  std::vector<size_t>* getfeatureWeightsVariables() {
    return _featureWeightsVariables.get();
  }

  std::vector<double>* getdeepFeatureWeights() {
    return _deepFeatureWeights.get();
  }

  std::vector<size_t>* getdeepFeatureWeightsVariables() {
    return _deepFeatureWeightsVariables.get();
  }

  std::vector<double>* getobservationWeights() {
    return _observationWeights.get();
  }

  std::vector<int>* getMonotonicConstraints() {
    return _monotonicConstraints.get();
  }

  std::vector<size_t>* getRowNumbers() {
    return _rowNumbers.get();
  }

  std::vector<size_t> get_all_row_idx(std::vector<size_t>* sampleIndex);

  size_t get_row_idx(size_t rowIndex);

  void setOutcomeData(std::vector<double> outcomeData);

private:
  std::shared_ptr< std::vector< std::vector<double> > > _featureData;
  std::unique_ptr< std::vector<double> > _outcomeData;
  std::unique_ptr< std::vector<size_t> > _rowNumbers;
  std::unique_ptr< std::vector<size_t> > _categoricalFeatureCols;
  std::unique_ptr< std::vector<size_t> > _numericalFeatureCols;
  std::unique_ptr< std::vector<size_t> > _linearFeatureCols;
  std::size_t _numRows;
  std::size_t _numColumns;
  std::unique_ptr< std::vector<double> > _featureWeights;
  std::unique_ptr< std::vector<size_t> > _featureWeightsVariables;
  std::unique_ptr< std::vector<double> > _deepFeatureWeights;
  std::unique_ptr< std::vector<size_t> > _deepFeatureWeightsVariables;
  std::unique_ptr< std::vector<double> > _observationWeights;
  std::shared_ptr< std::vector<int> > _monotonicConstraints;
};


#endif //FORESTRYCPP_DATAFRAME_H
