## Rforestry: Random Forests, Linear Trees, and Gradient Boosting for Inference and Interpretability

Sören Künzel, Theo Saarinen, Simon Walter, Edward Liu, Sam Antonyan, Allen Tang, Jasjeet Sekhon


## Developer Installation

1. Install R, then install RStudio https://posit.co/download/rstudio-desktop/
2. (Windows only) Install [RTools](https://cran.r-project.org/bin/windows/Rtools/)
3. (MacOS only) Install GFortran [gfortran-12.2-universal](https://mac.r-project.org/tools/gfortran-12.2-universal.pkg) from https://mac.r-project.org/tools/

4. Open console or any git UI, clone this git repository **with submodules**:
```bash
git clone --recursive https://github.com/forestry-labs/Rforestry.git
```

Open RStudio:
1. "Create a project" -> "Existing Directory" -> select `Rforestry/R` folder.
2. Install R dependances: open "Console" tab (bottom left corner), execute command:
```r
install.packages(c("devtools","visNetwork","glmnet","onehot","pROC","RcppArmadillo","RcppThread","mvtnorm","jsonlite"))
```
3. Build and test R package: open "Build" tab (top right corner), click "Install", then click "Test".
