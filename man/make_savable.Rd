% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forestry.R
\name{make_savable}
\alias{make_savable}
\alias{make_savable,forestry-method}
\title{make_savable}
\usage{
make_savable(object)
}
\arguments{
\item{object}{an object of class `forestry`}
}
\value{
A list of lists. Each sublist contains the information to span a
  tree.
}
\description{
When a `foresty` object is saved and then reloaded the Cpp
  pointers for the data set and the Cpp forest have to be reconstructed
}
\note{
`make_savable` does not translate all of the private member variables
  of the C++ forestry object so when the forest is reconstructed with
  `relinkCPP_prt` some attributes are lost. For example, `nthreads` will be
  reset to zero. This makes it impossible to disable threading when
  predicting for forests loaded from disk.
}
\examples{
set.seed(323652639)
x <- iris[, -1]
y <- iris[, 1]
forest <- forestry(x, y, ntree = 3)
y_pred_before <- predict(forest, x)

forest <- make_savable(forest)
saveForestry(forest, file = "forest.Rda")
rm(forest)

forest <- loadForestry("forest.Rda")

y_pred_after <- predict(forest, x)
testthat::expect_equal(y_pred_before, y_pred_after, tolerance = 0.000001)
file.remove("forest.Rda")
}
