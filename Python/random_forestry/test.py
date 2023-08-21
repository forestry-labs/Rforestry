import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from random_forestry import RandomForest

# Getting the dataset
data = load_iris()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]
cat_col = np.random.choice(["a", "b", "c"], size=len(X.index))
X["CategoricalVar"] = pd.Categorical(cat_col)

# Create a RandomForest object
fr = RandomForest(ntree=100, linear=True, overfit_penalty=0.001, nodesize_strict_spl=10, seed=1)
fr.fit(X.iloc[:, 1:], X.iloc[:, 1], max_depth=5, lin_feats=[0, 1])

fr2 = RandomForest(ntree=100, linear=True, overfit_penalty=0.001, nodesize_strict_spl=10, seed=1)
fr2.fit(X.iloc[:, 1:], X.iloc[:, 1], max_depth=5, lin_feats=[0, 1], double_bootstrap=True)

print("translate the first tree")

fr.translate_tree(0)
# print(fr.saved_forest_[0]["children_left"].size)
fr2.translate_tree(0)
# print(fr2.saved_forest_[0])

print("Making predictions")
preds = fr2.predict(X.iloc[:, 1:], aggregation="doubleOOB", return_weight_matrix=True)

fr2.save_forestry("rforest")
fr_load = RandomForest.load_forestry("rforest")

for k in fr2.get_parameters():
    if getattr(fr2, k) != getattr(fr_load, k):
        print(k)


assert fr2.get_parameters() == fr_load.get_parameters()
assert np.array_equal(fr2.processed_dta.y, fr_load.processed_dta.y)

print(pd.DataFrame(data=fr2.forest_))

preds_after = fr_load.predict(return_weight_matrix=True, aggregation="doubleOOB")
print(preds["weightMatrix"])

print(preds_after["weightMatrix"])
print("\n The two predictions are equal: " + str(np.array_equal(preds["weightMatrix"], preds_after["weightMatrix"])))


# for i in range(150):
#    print(
#        "\n The two predictions are equal: "
#        + str(np.array_equal(preds["weightMatrix"][i], preds_after["weightMatrix"][i]))
#    )
