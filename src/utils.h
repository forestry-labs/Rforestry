#ifndef FORESTRYCPP_UTILS_H
#define FORESTRYCPP_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>

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


struct tree_info {
  std::vector< int > var_id;
  // contains the variable id for a splitting node and the negative number of
  // observations in a leaf for a leaf node
  std::vector< long double > split_val;
  // contains the split values for regular nodes
  std::vector< int > leafAveidx;
  // contains the indices of observations in a leaf.
  std::vector< int > leafSplidx;
  // contains the indices of observations in a leaf.
  std::vector< int > averagingSampleIndex;
  // contains the indices of the average set.
  std::vector< int > splittingSampleIndex;
  // contains the indices of the splitting set.
  std::vector< int > naLeftCount;
  // Contains the count of NA's which fell to the left for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
  std::vector< int > naRightCount;
  // Contains the count of NA's which fell to the right for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
};

// Contains the information to help with monotonic constraints on splitting
struct monotonic_info {
  // Contains the monotonic constraints on each variable
  // For each continuous variable, we have +1 indicating a positive monotone
  // relationship, -1 indicating a negative monotone relationship, and 0
  // indicates no monotonic relationship
  std::vector<int> monotonic_constraints;

  // These contain the upper and lower bounds on node means for the node
  // currently being split on. These are used to reject potential splits
  // which do not respect the bounds, and therfore enforce global monotonic
  // bounds.
  float upper_bound;
  float lower_bound;
};

#endif //FORESTRYCPP_UTILS_H
