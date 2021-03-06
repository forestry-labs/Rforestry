% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forestry.R
\name{testing_data_checker-forestry}
\alias{testing_data_checker-forestry}
\alias{testing_data_checker}
\title{Test data check}
\usage{
testing_data_checker(object, newdata, hasNas)
}
\arguments{
\item{object}{A forestry object.}

\item{newdata}{A data frame of testing predictors.}

\item{hasNas}{TRUE if the there were NAs in the training data FALSE otherwise.}
}
\value{
A feature dataframe if it can be used for new predictions.
}
\description{
Check the testing data to do prediction
}
