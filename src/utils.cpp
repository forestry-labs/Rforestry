#include "utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <queue>

#include "forestry.h"
#include "forestryTree.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

void print_vector(
    std::vector<size_t> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
}

void print_vector(
    std::vector<unsigned int> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
}


void print_vector(
    std::vector<double> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
}

int add_vector(
    std::vector<int>* v
) {
  int sum=0;
  for (size_t i = 0; i < v->size(); i++) {
    sum += (*v)[i];
  }
  return sum;
}

double square(
  double x
) {
  return (x*x);
}

std::string exportJson(forestry& forest, const std::vector<double>& colSds, const std::vector<double>& colMeans) {
    if (forest.getlinear()) {
      throw std::runtime_error("Linear forest export is not supported");
    }

    assert(colSds.size() == colMeans.size());
    assert(colSds.size() > 0);
    const size_t valueFeatureId = colSds.size()-1; // The last values in colSds and colMeans are for Y
    const size_t numFeature = colSds.size()-1;

    assert(forest.getTrainingData() && forest.getTrainingData()->getCatCols());
    const std::vector<size_t>& categorialCols = *forest.getTrainingData()->getCatCols();

    // To quickly check if the feature is categorical
    const std::unordered_set<size_t> categorialColsSet(categorialCols.begin(), categorialCols.end());

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> w(buffer);

    w.StartObject();
    w.Key("num_feature"); w.Uint(numFeature);
    w.Key("task_type"); w.String("kBinaryClfRegr");
    w.Key("average_tree_output"); w.Bool(true);

    /**
      * kBinaryClfRegr
      * Catch-all task type encoding all tasks that are not multi-class classification, such as binary classification, regression, and learning-to-rank.
      * The kBinaryClfRegr task type implies the following constraints on the task parameters: output_type=float, grove_per_class=false, num_class=1, leaf_vector_size=1
    */
    w.Key("task_param");
    {
      w.StartObject();
      w.Key("output_type"); w.String("float");
      w.Key("grove_per_class"); w.Bool(false);
      w.Key("num_class"); w.Uint(1);
      w.Key("leaf_vector_size"); w.Uint(1);
      w.EndObject();
    }

    w.Key("model_param");
    {
      w.StartObject();
      w.Key("pred_transform"); w.String("identity");
      w.Key("global_bias"); w.Double(0.0);
      w.EndObject();
    }

    w.Key("trees");
    w.StartArray();
    for (size_t i=0; i<forest.getNtree(); i++) {
      forestryTree &tree = *(*forest.getForest())[i];
      RFNode& root = *tree.getRoot();

      w.StartObject();
      w.Key("root_id"); w.Uint(root.getNodeId());
      w.Key("nodes");
      w.StartArray();

      std::queue<RFNode*> nodesToProcess;
      nodesToProcess.push(&root);

      auto unscaleUncenter = [&colSds, &colMeans](double value, size_t featureId) -> double {
        return value * colSds[featureId] + colMeans[featureId];
      };

      while (!nodesToProcess.empty()) {
        RFNode& n = *nodesToProcess.front();
        nodesToProcess.pop();

        w.StartObject();
        w.Key("node_id"); w.Uint(n.getNodeId());

        if (n.is_leaf()) {
          w.Key("leaf_value"); w.Double(unscaleUncenter(n.getPredictWeight(), valueFeatureId));
        } else {
          const size_t splitFeatureId = n.getSplitFeature();
          w.Key("split_feature_id"); w.Uint(splitFeatureId);
          w.Key("default_left"); w.Bool(n.getNaDefaultDirection() == -1); // naDefaultDirection is -1 for left and 1 for right (see RFNode::predict)

          if (categorialColsSet.count(splitFeatureId)) {
            w.Key("split_type"); w.String("categorical");
            w.Key("categories_list");
            w.StartArray();
            w.Uint(static_cast<unsigned>(n.getSplitValue())); // Casting double back to unsigned should be OK if no arithmetic operations were performed with it
            w.EndArray();

            /**
             * When the test criteron ([Feature value] in [categories_list]) is evaluated to be true, the prediction function traverses to the left child node (if categories_list_right_child=false)
            */
            w.Key("categories_list_right_child"); w.Bool(false);
          } else {
            w.Key("split_type"); w.String("numerical");
            w.Key("comparison_op"); w.String("<");
            w.Key("threshold"); w.Double(unscaleUncenter(n.getSplitValue(), splitFeatureId));
          }

          assert(n.getLeftChild());
          w.Key("left_child"); w.Uint(n.getLeftChild()->getNodeId());
          nodesToProcess.push(n.getLeftChild());

          assert(n.getRightChild());
          w.Key("right_child"); w.Uint(n.getRightChild()->getNodeId());
          nodesToProcess.push(n.getRightChild());
        }
        w.EndObject();
      }
      w.EndArray();
      w.EndObject(); // tree object
    }
    w.EndArray(); // trees
    w.EndObject();

    return buffer.GetString();
}
