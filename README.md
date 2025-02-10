[![R-CMD-check](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml/badge.svg)](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml)

## Rforestry: Random Forests, Linear Trees, and Gradient Boosting for Inference and Interpretability

Sören Künzel, Theo Saarinen, Simon Walter, Edward Liu, Allen Tang, Jasjeet Sekhon

## Introduction

Rforestry is a fast implementation of Random Forests, Gradient Boosting,
and Linear Random Forests, with an emphasis on inference and interpretability.

## How to install
1. The GFortran compiler has to be up to date. GFortran Binaries can be found [here](https://gcc.gnu.org/wiki/GFortranBinaries).
2. The [devtools](https://github.com/r-lib/devtools) package has to be installed. You can install it using,  `install.packages("devtools")`.
3. The package contains compiled code, and you must have a development environment to install the development version. You can use `devtools::has_devel()` to check whether you do. If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://apps.apple.com/us/app/xcode/id497799835).
4. The latest development version can then be installed using
`devtools::install_github("forestry-labs/Rforestry")`. For Windows users, you'll need to skip 64-bit compilation `devtools::install_github("forestry-labs/Rforestry", INSTALL_opts = c('--no-multiarch'))` due to an outstanding gcc issue.


## Usage

```R
set.seed(292315)
library(Rforestry)
test_idx <- sample(nrow(iris), 3)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

rf <- forestry(x = x_train, y = y_train)
weights = predict(rf, x_test, aggregation = "weightMatrix")$weightMatrix

weights %*% y_train
predict(rf, x_test)
```

## Ridge Random Forest

A fast implementation of random forests using ridge penalized splitting and ridge regression for predictions.

Example:

  ```R
set.seed(49)
library(Rforestry)

n <- c(100)
a <- rnorm(n)
b <- rnorm(n)
c <- rnorm(n)
y <- 4*a + 5.5*b - .78*c
x <- data.frame(a,b,c)
forest <- forestry(x, y, ridgeRF = TRUE)
predict(forest, x)
```

## Monotonic Constraints

A parameter controlling monotonic constraints for features in forestry.

```R
library(Rforestry)

x <- rnorm(150)+5
y <- .15*x + .5*sin(3*x)
data_train <- data.frame(x1 = x, x2 = rnorm(150)+5, y = y + rnorm(150, sd = .4))

monotone_rf <- forestry(x = data_train %>% select(-y),
                        y = data_train$y,
                        monotonicConstraints = c(-1,-1),
                        nodesizeStrictSpl = 5,
                        nthread = 1,
                        ntree = 25)
predict(monotone_rf, feature.new = data_train %>% select(-y))

```


## OOB Predictions

We can return the predictions for the training dataset using only the trees in
which each observation was out of bag. Note that when there are few trees, or a
high proportion of the observations sampled, there may be some observations
which are not out of bag for any trees.
The predictions for these are returned NaN.


```R
library(Rforestry)

# Train a forest
rf <- forestry(x = iris[,-1],
               y = iris[,1],
               ntree = 500)

# Get the OOB predictions for the training set
oob_preds <- getOOBpreds(rf)

# This should be equal to the OOB error
sum((oob_preds -  iris[,1])^2)
getOOB(rf)
```



