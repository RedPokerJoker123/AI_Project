import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """
    Plotter for feature-vs-target diagnostics.

    Assumptions (per your convention):
      - X includes an intercept column of 1s as the FIRST column by default.
      - We standardize X feature columns (excluding intercept).
      - For regression plots: we standardize BOTH X and y (as you requested).
      - For classification plots: we standardize X only (y is categorical/labels).

    Public methods:
      - plot_regression(X, y, ...)
      - plot_classification(X, y, ...)
    """

    def __init__(self, assume_intercept_first_col=True, random_state=0):
        self.assume_intercept_first_col = bool(assume_intercept_first_col)
        self.rng = np.random.default_rng(random_state)

    # -----------------------
    # Helpers
    # -----------------------
    def _to_2d_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X

    def _to_1d_y(self, y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.reshape(-1)
        return y

    def _split_intercept(self, X):
        if not self.assume_intercept_first_col:
            return None, X

        if X.shape[1] < 1:
            raise ValueError("X must have at least 1 column.")
        if not np.allclose(X[:, 0], 1.0):
            raise ValueError("Expected intercept column of 1s in X[:,0].")
        return X[:, :1], X[:, 1:]

    def _standardize(self, A):
        A = np.asarray(A, dtype=float)
        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        As = (A - mean) / std
        return As, mean, std

    def _feature_names(self, d, feature_names=None):
        if feature_names is None:
            return [f"x{j+1}" for j in range(d)]
        if len(feature_names) != d:
            raise ValueError(f"feature_names must have length {d}.")
        return list(feature_names)

    # -----------------------
    # Public: Regression
    # -----------------------
    def plot_regression(
        self,
        X,
        y,
        feature_names=None,
        standardize_y=True,
        alpha=0.9,
        s=18,
        show=True,
    ):
        """
        Scatter plots: each standardized feature vs standardized y (default),
        one figure per feature.

        Parameters:
          - standardize_y: if True, standardize y (as you requested).
        """
        X = self._to_2d_X(X)
        y = self._to_1d_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows/samples.")

        _, Xf = self._split_intercept(X)
        Xfs, _, _ = self._standardize(Xf)

        ys = y.astype(float)
        y_label = "y"
        if standardize_y:
            ys = ys.reshape(-1, 1)
            ys, _, _ = self._standardize(ys)
            ys = ys.reshape(-1)
            y_label = "Standardized y"

        n, d = Xfs.shape
        names = self._feature_names(d, feature_names)

        for j in range(d):
            plt.figure()
            plt.scatter(Xfs[:, j], ys, alpha=alpha, s=s)
            plt.xlabel(f"Standardized {names[j]}")
            plt.ylabel(y_label)
            plt.title(f"{names[j]} vs {y_label}")
            plt.grid(True, linewidth=0.3, alpha=0.4)
            if show:
                plt.show()

    # -----------------------
    # Public: Classification
    # -----------------------
    def plot_classification(
        self,
        X,
        y,
        feature_names=None,
        jitter_y=0.08,
        alpha=0.9,
        s=18,
        show=True,
    ):
        """
        Classification-friendly feature plots:
          - Standardize X features
          - y is treated as labels (binary or multiclass)
          - Points are jittered vertically to reduce overlap

        Works for:
          - binary y in {0,1} or {-1,1} (or any 2 labels)
          - multiclass labels (e.g., 0..K-1 or strings)
        """
        X = self._to_2d_X(X)
        y = self._to_1d_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows/samples.")

        _, Xf = self._split_intercept(X)
        Xfs, _, _ = self._standardize(Xf)

        # Encode labels to 0..K-1 (but keep original for tick labels)
        labels, y_enc = np.unique(y, return_inverse=True)
        K = len(labels)

        n, d = Xfs.shape
        names = self._feature_names(d, feature_names)

        for j in range(d):
            plt.figure()

            y_jit = y_enc.astype(float)
            y_jit += self.rng.normal(0.0, jitter_y, size=n)

            plt.scatter(Xfs[:, j], y_jit, alpha=alpha, s=s)

            plt.xlabel(f"Standardized {names[j]}")
            plt.ylabel("Class")
            plt.title(f"{names[j]} vs Class (X standardized)")

            plt.yticks(np.arange(K), [str(lbl) for lbl in labels])
            plt.ylim(-0.6, (K - 1) + 0.6)
            plt.grid(True, linewidth=0.3, alpha=0.4)

            if show:
                plt.show()
