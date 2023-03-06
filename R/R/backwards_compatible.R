# We are making sure here that RF is backwards compatible

#' @include forestry.R
# -- honestRF ------------------------------------------------------------------
#' @title Honest Random Forest
#' @description This function is deprecated and only exists for backwards
#'   backwards compatibility. The function you want to use is `forestry`.
#' @param ... parameters which are passed directly to `forestry`
#' @return A `forestry` object
#' @export honestRF
honestRF <- function(...) forestry(...)


# -- Random Forest Constructor -------------------------------------------------
#' @title forestry class
#' @name forestry-class
#' @description `honestRF` class only exists for backwards compatibility reasons
setClass(Class = "honestRF",
         contains = 'forestry')
