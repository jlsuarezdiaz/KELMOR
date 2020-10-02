from sklearn.base import ClassifierMixin, BaseEstimator

import numpy as np
from sklearn import preprocessing
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels


class KELMOR(ClassifierMixin, BaseEstimator):
    """
    Kernel Extreme Learning Machine for Ordinal Regression (KELMOR).

    Parameters
    ----------

    C: float, default=1

        Adjusting parameter. Must be strictly positive.

    method : string, default="full"

        Method used for the factorization of the kernel matrix. Currently supported values are:

        - "full" : no factorization is applied (KELMOR)

    S : int, default=None

        The numerical rank expected in the factorization of the kernel matrix. If None,
        S will be the number of samples of the dataset. Ignored if method=="full".

    eps : float, default=1e-5

        Error tolerance in the kernel matrix factorization. Ignored if method=="full".

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".

    gamma : float, default=1/n_features

        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    degree : int, default=3

        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1

        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, default=None

        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    References
    ----------
        Shi, Y., Li, P., Yuan, H., Miao, J., & Niu, L. (2019). Fast kernel extreme learning machine for ordinal regression. Knowledge-Based Systems, 177, 44-54.

    """

    def __init__(self, C=1, method="full", S=None, eps=1e-5, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        self.C = C
        self.kernel = kernel
        self.method = method
        self.S = S
        self.eps = eps
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def fit(self, X, y):
        """
        Fit the model from the data in X and the labels in y.

        Parameters
        ----------
        X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

        y : array-like, shape (N)
            Labels vector, where N is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        self.X, self.y = X, y
        n, d = X.shape
        self.le_ = preprocessing.LabelEncoder()
        self.le_.fit(y)

        y = self.le_.transform(y)
        classes = np.unique(y)
        nclasses = len(classes)

        self.M = np.array([[(i - j) ** 2 for i in range(nclasses)] for j in range(nclasses)])
        T = self.M[y, :]
        K = self._get_kernel(X)

        if self.method == "full":
            self.beta = np.linalg.inv((1 / self.C) * np.eye(n) + K).dot(T)
        else:
            raise ValueError("Invalid value for argument 'method'.")

        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape (N x d)

            Test samples. N is the number of samples and d the number of features.

        Returns
        -------
        y : array of shape (N)

            Class labels for each data sample.
        """
        K = self._get_kernel(X, self.X)
        coded_preds = K.dot(self.beta)

        predictions = np.argmin(np.linalg.norm(coded_preds[:, None] - self.M, axis=2, ord=1), axis=1)
        predictions = self.le_.inverse_transform(predictions)
        return predictions

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma,
                      'degree': self.degree,
                      'coef0': self.coef0}

        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"
