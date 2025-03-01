% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/forestry.R
\name{getVI}
\alias{getVI}
\title{getVI-forestry}
\usage{
getVI(object, noWarning)
}
\arguments{
\item{object}{A `forestry` object.}

\item{noWarning}{flag to not display warnings}
}
\description{
Calculate increase in OOB for each shuffled feature for forest.
}
\note{
No seed is passed to this function so it is
  not possible in the current implementation to replicate the vector
  permutations used when measuring feature importance.
}
