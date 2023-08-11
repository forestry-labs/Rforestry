#ifndef FORESTRYCPP_UTILS_H
#define FORESTRYCPP_UTILS_H

#include "dataFrame.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

class forestry;

void print_vector(
  std::vector<size_t> v
);

void print_vector(
    std::vector<unsigned int> v
);

void print_vector(
    std::vector<double> v
);

int add_vector(
    std::vector<int>* v
);

double square(
    double x
);

struct tree_info {
  std::vector< int > var_id;
  // contains the variable id for a splitting node (-1 indicates leaf node)
  std::vector< int > average_count;
  // contains the number of observations in the averaging set in each node
  std::vector< int > split_count;
  // contains the number of observations in the splitting set in each node
  std::vector< long double > split_val;
  // contains the split values for regular nodes
  std::vector< double > values;
  // contains the weights used for prediction in each node
  // Weights for interior nodes also included
  std::vector< int > num_spl_samples;
  // Counts of splitting samples at each node
  std::vector< int > num_avg_samples;
  // Contains the counts of averaging samples at each node
  std::vector< int > averagingSampleIndex;
  // contains the indices of the average set.
  std::vector< int > splittingSampleIndex;
  // contains the indices of the splitting set.
  std::vector< int > excludedSampleIndex;
  // contains the indices of the excluded set.
  std::vector< int > naLeftCount;
  // Contains the count of NA's which fell to the left for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
  std::vector< int > naRightCount;
  // Contains the count of NA's which fell to the right for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
  std::vector< int > naDefaultDirection;
  // Contains the default direction for all NA values per split node if
  // naDirection == TRUE, -1 indicates left and 1 indicates right
  size_t numLeafNodes;
  size_t numSplitNodes;
  // To help the Python API allocate space for the tree info,
  // store the number of split nodes and number of leaf nodes
  // (this allows us to calculate the size of each of the above vectors)
  unsigned int seed;
  // The seed that the tree was given (this uniquely identifies each tree
  // so that we can tell them apart. Very important for prediction when
  // exact = TRUE as we must aggregate the trees in the right order)
};

typedef std::vector<tree_info> tree_info_vector;

// Contains the information to help with monotonic constraints on splitting
struct monotonic_info {
  // Contains the monotonic constraints on each variable
  // For each continuous variable, we have +1 indicating a positive monotone
  // relationship, -1 indicating a negative monotone relationship, and 0
  // indicates no monotonic relationship
  std::vector<int> monotonic_constraints;

  // These contain the upper and lower bounds on node means for the node
  // currently being split on. These are used to reject potential splits
  // which do not respect the bounds, and therefore enforce global monotonic
  // bounds.
  double upper_bound;
  double lower_bound;

  // This flag indicates whether or not to enforce monotonicity on the averaging
  // set as well as the splitting set
  bool monotoneAvg;

  monotonic_info(){
    monotoneAvg = false;
  };
};

/**
 * Returns Treelite JSON representation of the forest
 * Does not support linear trees
*/
std::string exportJson(forestry& forest, const std::vector<double>& colSds, const std::vector<double>& colMeans);

#endif //FORESTRYCPP_UTILS_H
