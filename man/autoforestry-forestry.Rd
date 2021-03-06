% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forestry.R
\name{autoforestry}
\alias{autoforestry}
\title{autoforestry-forestry}
\usage{
autoforestry(
  x,
  y,
  sampsize = as.integer(nrow(x) * 0.75),
  num_iter = 1024,
  eta = 2,
  verbose = FALSE,
  seed = 24750371,
  nthread = 0
)
}
\arguments{
\item{x}{A data frame of all training predictors.}

\item{y}{A vector of all training responses.}

\item{sampsize}{The size of total samples to draw for the training data.}

\item{num_iter}{Maximum iterations/epochs per configuration. Default is 1024.}

\item{eta}{Downsampling rate. Default value is 2.}

\item{verbose}{if tuning process in verbose mode}

\item{seed}{random seed}

\item{nthread}{Number of threads to train and predict the forest. The default
number is 0 which represents using all cores.}
}
\value{
A `forestry` object
}
\description{
autoforestry-forestry
}
