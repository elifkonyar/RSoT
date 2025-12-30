# RSoTR: Robust Generalized Scalar-on-Tensor Regression

This repository provides the official R implementation for **Robust Generalized Scalar-on-Tensor Regression**, as presented in:

> Elif Konyar, Mostafa Reisi Gahrooei, Ruizhi Zhang, "Robust Generalized Scalar-on-Tensor Regression", *IISE Transactions*.

## Overview

High-dimensional (HD) data, such as images and profiles, contain significant predictive information but are often prone to outliers that bias standard linear scalar-on-tensor regression models. 

**RSoTR** proposes a robust estimation framework constructed using **maximum $L_q$-likelihood estimation** instead of the classic maximum likelihood estimation. This allows for accurate modeling of multi-dimensional HD input data even when the dataset is contaminated with outliers.

## Prerequisites

The implementation requires **R** and several packages for tensor manipulation and robust modeling. You can install all necessary dependencies using the following command:

```R
install.packages(c("rTensor", "data.table", "MASS", "caret", "epca", "parallel", "tictoc", "robustlm", "ggplot2"))

## Repository Structure

The code is organized by the distribution of the response variable and specific modeling cases:

### 1. Gaussian Response
* **RSOTR_Main_Gaussian.R**: Main execution script for simulations with Gaussian noise.
* **RSOTR_Main_Gaussian_Utils.R**: Core utility functions including tensor operations and $L_q$ estimation logic.

### 2. Poisson Response
* **RSOTR_Main_Poisson.R**: Main execution script for count-data simulations.
* **RSOTR_Main_Poisson_Utils.R**: Specialized functions for handling Poisson likelihood and robust estimation.

### 3. Sparse Regression
* **RSOTR_Main_Sparse.R**: Implementation for cases requiring sparsity constraints.
* **RSOTR_Main_Sparse_Utils.R**: Utility functions for the sparse modeling framework.

### 4. Performance & Evaluation
* **RSOTR_Performance.R**: Centralized functions for calculating metrics (RMSE, $R^2$, WMAPE, Bias) and generating visualizations/heatmaps of the results.

## Getting Started

1. **Download the Scripts**: Ensure all `.R` files (both Main and Utils) are stored in the same directory.
2. **Set Your Paths**: Open the **Main** R script you wish to run (e.g., `RSOTR_Main_Gaussian.R`).
3. **Update Directory**: Update the `source()` and `setwd()` commands at the top of the script to reflect the folder path on your local machine.
4. **Run Simulation**: Execute the script. The code will perform the simulation and automatically export a `.csv` file containing the performance metrics to your working directory.

## Support & Bug Reports

If you find any bugs or errors in the codes, or if you have questions regarding the implementation, please feel free to reach out:

* **Email**: [ekonyar3@gatech.edu](mailto:ekonyar3@gatech.edu)

## Citation

If you use this code or the RSoTR method in your research, please cite our paper:

```bibtex
@article{konyar2024rsotr,
  title={Robust Generalized Scalar-on-Tensor Regression},
  author={Konyar, Elif and Gahrooei, Mostafa Reisi and Zhang, Ruizhi},
  journal={IISE Transactions},
  year={2024},
  publisher={Taylor \& Francis}
}