#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "api.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <typename T>
void copy_vector_to_numpy_array(std::vector<T> &vector, py::array_t<T> &np) {
    py::buffer_info np_info = np.request();
    T *np_ptr = static_cast<T *>(np_info.ptr);

    for (int i = 0; i < np_info.shape[0]; i++)
    {
        np_ptr[i] = vector[i];
    }
}

template <typename T>
std::vector<T> create_vector_from_numpy_array(py::array_t<T> np) {
    py::buffer_info np_info = np.request();
    T *np_ptr = static_cast<T *>(np_info.ptr);

    std::vector<T> vector(np_info.shape[0]);
    for (int i = 0; i < np_info.shape[0]; i++)
    {
        vector[i] = np_ptr[i];
    }
    return vector;
}

py::array_t<double> create_numpy_array(unsigned int size) {
    return py::array(py::buffer_info(
        nullptr,                              /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(double),                       /* Size of one item */
        py::format_descriptor<double>::value, /* Buffer format */
        1,                                    /* How many dimensions? */
        {size},                               /* Number of elements for each dimension */
        {sizeof(double)}                      /* Strides for each dimension */
        ));
}

void *get_data_wrapper(
    py::array_t<double> arr,
    py::array_t<size_t> categorical_vars,
    size_t countCategoricals,
    py::array_t<size_t> linFeat_idx,
    size_t countLinFeats,
    py::array_t<double> feat_weights,
    py::array_t<size_t> feat_weight_vars,
    size_t countFtWeightVars,
    py::array_t<double> deep_feat_weights,
    py::array_t<size_t> deep_feat_weight_vars,
    size_t countDeepFtWeightVars,
    py::array_t<double> observation_weights,
    py::array_t<int> mon_constraints,
    py::array_t<size_t> groupMemberships,
    bool monotoneAvg,
    size_t numRows,
    size_t numColumns,
    unsigned int seed
) {
    return get_data(
        static_cast<double *>(arr.request().ptr),
        static_cast<size_t *>(categorical_vars.request().ptr),
        countCategoricals,
        static_cast<size_t *>(linFeat_idx.request().ptr),
        countLinFeats,
        static_cast<double *>(feat_weights.request().ptr),
        static_cast<size_t *>(feat_weight_vars.request().ptr),
        countFtWeightVars,
        static_cast<double *>(deep_feat_weights.request().ptr),
        static_cast<size_t *>(deep_feat_weight_vars.request().ptr),
        countDeepFtWeightVars,
        static_cast<double *>(observation_weights.request().ptr),
        static_cast<int *>(mon_constraints.request().ptr),
        static_cast<size_t *>(groupMemberships.request().ptr),
        monotoneAvg,
        numRows,
        numColumns,
        seed
    );
}

forestry *reconstructree_wrapper(
    void *data_ptr,
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
    bool doubleTree,
    py::array_t<size_t> tree_counts,
    py::array_t<double> thresholds,
    py::array_t<int> features,
    py::array_t<int> na_left_count,
    py::array_t<int> na_right_count,
    py::array_t<int> na_default_directions,
    py::array_t<size_t> split_idx,
    py::array_t<size_t> average_idx,
    py::array_t<double> predict_weights,
    py::array_t<unsigned int> tree_seeds
) {
    return reconstructree(data_ptr,
        ntree,
        replace,
        sampSize,
        splitRatio,
        OOBhonest,
        doubleBootstrap,
        mtry,
        minNodeSizeSpt,
        minNodeSizeAvg,
        minNodeSizeToSplitSpt,
        minNodeSizeToSplitAvg,
        minSplitGain,
        maxDepth,
        interactionDepth,
        seed,
        nthread,
        verbose,
        splitMiddle,
        maxObs,
        minTreesPerFold,
        foldSize,
        hasNas,
        naDirection,
        linear,
        overfitPenalty,
        doubleTree,
        static_cast<size_t *>(tree_counts.request().ptr),
        static_cast<double *>(thresholds.request().ptr),
        static_cast<int *>(features.request().ptr),
        static_cast<int *>(na_left_count.request().ptr),
        static_cast<int *>(na_right_count.request().ptr),
        static_cast<int *>(na_default_directions.request().ptr),
        static_cast<size_t *>(split_idx.request().ptr),
        static_cast<size_t *>(average_idx.request().ptr),
        static_cast<double *>(predict_weights.request().ptr),
        static_cast<unsigned int *>(tree_seeds.request().ptr)
    );
}

py::tuple predictOOB_forest_wrapper(
    forestry* forest,
    void *dataframe_pt,
    py::array_t<double> test_data,
    bool doubleOOB,
    bool exact,
    bool returnWeightMatrix,
    bool verbose,
    bool use_training_idx,
    unsigned int n_preds,
    unsigned int n_weight_matrix,
    py::array_t<size_t> training_idx,
    bool hier_shrinkage,
    double lambda_shrinkage
) {
    py::array_t<double> predictions = create_numpy_array(n_preds);
    std::vector<double> predictions_vector(n_preds);

    py::array_t<double> weight_matrix = create_numpy_array(n_weight_matrix);
    std::vector<double> weight_matrix_vector(n_weight_matrix);

    std::vector<size_t> training_idx_vector;
    if (use_training_idx) {
       training_idx_vector = create_vector_from_numpy_array(training_idx);
    }

    predictOOB_forest(
        forest,
        dataframe_pt,
        static_cast<double *>(test_data.request().ptr),
        doubleOOB,
        exact,
        returnWeightMatrix,
        verbose,
        predictions_vector,
        weight_matrix_vector,
        training_idx_vector,
        hier_shrinkage,
        lambda_shrinkage
        );

    copy_vector_to_numpy_array(predictions_vector, predictions);
    copy_vector_to_numpy_array(weight_matrix_vector, weight_matrix);

    return py::make_tuple(predictions, weight_matrix);
}

void show_array(double *ptr, size_t size) {
    for (int i = 0; i < size; i++)
    {
        std::cout << *ptr << " ";
        ptr++;
    }
    std::cout << '\n'
              << '\n';
}

void show_array(std::vector<double> array) {
    for (auto i : array)
    {
        std::cout << i << ' ';
    }
    std::cout << '\n'
              << '\n';
}

py::tuple predict_forest_wrapper(
    forestry* forest,
    void *dataframe_pt,
    py::array_t<double> test_data,
    unsigned int seed,
    size_t nthread,
    bool exact,
    bool returnWeightMatrix,
    bool linear,
    bool use_weights,
    py::array_t<size_t> tree_weights,
    size_t num_test_rows,
    unsigned int n_preds,
    unsigned int n_weight_matrix,
    unsigned int n_coefficients,
    bool hier_shrinkage,
    double lambda_shrinkage
) {
    py::array_t<double> predictions = create_numpy_array(n_preds);
    std::vector<double> predictions_vector(n_preds);

    py::array_t<double> weight_matrix = create_numpy_array(n_weight_matrix);
    std::vector<double> weight_matrix_vector(n_weight_matrix);

    py::array_t<double> coefficients = create_numpy_array(n_coefficients);
    std::vector<double> coefficients_vector(n_coefficients);

    predict_forest(
        forest,
        dataframe_pt,
        static_cast<double *>(test_data.request().ptr),
        seed,
        nthread,
        exact,
        returnWeightMatrix,
        linear,
        use_weights,
        static_cast<size_t *>(tree_weights.request().ptr),
        num_test_rows,
        predictions_vector,
        weight_matrix_vector,
        coefficients_vector,
        hier_shrinkage,
        lambda_shrinkage
    );

    copy_vector_to_numpy_array(predictions_vector, predictions);
    copy_vector_to_numpy_array(weight_matrix_vector, weight_matrix);
    copy_vector_to_numpy_array(coefficients_vector, coefficients);

    return py::make_tuple(predictions, weight_matrix, coefficients);
}

void fill_tree_info_wrapper(
    forestry* forest,
    int tree_idx,
    py::array_t<double> treeInfo,
    py::array_t<int> split_info,
    py::array_t<int> av_info
) {
    auto treeInfo_vector = create_vector_from_numpy_array(treeInfo);
    auto split_info_vector = create_vector_from_numpy_array(split_info);
    auto av_info_vector = create_vector_from_numpy_array(av_info);

    fill_tree_info(
        forest,
        tree_idx,
        treeInfo_vector,
        split_info_vector,
        av_info_vector
    );

    copy_vector_to_numpy_array(treeInfo_vector, treeInfo);
    copy_vector_to_numpy_array(split_info_vector, split_info);
    copy_vector_to_numpy_array(av_info_vector, av_info);
}

std::string export_json_wrapper(forestry* forest, bool scale, py::array_t<double> colSdsNp, py::array_t<double> colMeansNp) {
    auto colSds = create_vector_from_numpy_array(colSdsNp);
    auto colMeans = create_vector_from_numpy_array(colMeansNp);

    return export_json(forest, scale, colSds, colMeans);
}

PYBIND11_MODULE(extension, m)
{
    py::class_<forestry>(m, "forestry", py::dynamic_attr())
        .def(py::init([]() {
            throw py::value_error("forestry instances cannot be created from python");
            return nullptr;
        })
    );

    m.doc() = R"pbdoc(
        RForestry Python extension module
        -----------------------

        .. currentmodule:: RForestry.extension

        .. autosummary::
           :toctree: _generate

           vector_get
    )pbdoc";

    m.def("train_forest", &train_forest, py::return_value_policy::reference, R"pbdoc(
        Some help text here
    
        Some other explanation about the train_forest function.
    )pbdoc");
    m.def("get_data", &get_data_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the get_data function.
    )pbdoc");
    m.def("get_tree_node_count", &get_node_count, R"pbdoc(
        Some help text here

        Some other explanation about the getTreeNodeCount function.
    )pbdoc");
    m.def("get_tree_split_count", &get_split_node_count, R"pbdoc(
        Some help text here

        Some other explanation about the getTreeSplitNodeCount function.
    )pbdoc");
    m.def("get_tree_leaf_count", &get_leaf_node_count, R"pbdoc(
        Some help text here

        Some other explanation about the getTreeLeafNodeCount function.
    )pbdoc");
    m.def("reconstruct_tree", &reconstructree_wrapper, py::return_value_policy::reference, R"pbdoc(
        Some help text here

        Some other explanation about the reconstructree function.
    )pbdoc");
    m.def("delete_forestry", &delete_forestry, R"pbdoc(
        Some help text here

        Some other explanation about the delete_forestry function.
    )pbdoc");
    m.def("predict_oob_forest", &predictOOB_forest_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the predictOOB_forest function.
    )pbdoc");
    m.def("predict_forest", &predict_forest_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the predict_forest function.
    )pbdoc");
    m.def("fill_tree_info", &fill_tree_info_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the fill_tree_info function.
    )pbdoc");
    m.def("export_json", &export_json_wrapper, R"pbdoc(
        Export forest to Treelite JSON string
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
