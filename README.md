# KELMOR

A Python implementation of Kernel Extreme Learning Machine for Ordinal Regression (KELMOR)

## Usage

Download the file `KELMOR.py` and import it into your Python scripts:

```python
from sys import path
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error as mae

# Adding KELMOR.py to the path
path.append(".") # Replace . by the folder that contains KELMOR.py

# Import KELMOR module and class
kelmor_module = __import__("KELMOR") 
KELMOR = kelmor_module.KELMOR

# Load the dataset
X, y = load_iris(return_X_y=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_inds, test_inds = next(sss.split(X, y))
Xtra, ytra = X[train_inds, :], y[train_inds]
Xtst, ytst = X[test_inds, :], y[test_inds]

# Train kelmor on a dataset
kelmor = KELMOR(C=10, kernel="rbf")
kelmor.fit(Xtra, ytra)
# Predicting samples
ypred = kelmor.predict(Xtst)

# Mean absolute error of the predictions
print(mae(ytst, ypred))
# Output: 0.033333333

```

## Parameters

- **C**: float, default=1. Adjusting parameter. Must be strictly positive.
- **method**: string, default="full". Method used for the factorization of the kernel matrix. Currently supported values are:
  - "full": no factorization is applied (KELMOR).
- **S**: int, default=None. The numerical rank expected in the factorization of the kernel matrix. If None, S will be the number of samples in the dataset. Ignored if `method==full`-
- **eps**: float, default=1e-5. Error tolerance in the kernel matrix factorization. Ignored if `method==full`.
- **kernel**: string or callable, default="linear". The kernel to be used. Allowed values are: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed".
- **gamma**: float, default=1/n_features. Kernel coefficient for rbd, poly and sigmoid kernels. Ignored by other kernels.
- **degree**: int, default=3. Degree for poly kernels. Ignored by other kernels.
- **coef0**: float, default=1. Independent term in poly and sigmoid kernels. Ignored by other kernels.
- **kernel_params**: mapping of string to any, default=None. Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.

## References

Shi, Y., Li, P., Yuan, H., Miao, J., & Niu, L. (2019). Fast kernel extreme learning machine for ordinal regression. Knowledge-Based Systems, 177, 44-54.
