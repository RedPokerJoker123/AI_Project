import numpy as np

class FeatureCorrelation:
    def __init__(self, X, Y):
        self._X = np.asarray(X, dtype=float)
        self._Y = np.asarray(Y, dtype=float).ravel()

        # Ensure X is 2D
        if self._X.ndim == 1:
            self._X = self._X.reshape(-1, 1)

        if self._X.shape[0] != self._Y.size:
            raise ValueError("X and Y must have the same number of samples.")

    def pearson_correlation_coefficient(self):
        """
        Pearson correlation between each feature column in X and Y.

        Returns:
            np.ndarray of shape (n_features,)
        """
        X = self._X
        Y = self._Y

        y_centered = Y - Y.mean()
        y_std = Y.std()
        if y_std == 0.0:
            return np.full(X.shape[1], np.nan, dtype=float)

        pearson = np.empty(X.shape[1], dtype=float)

        for j in range(X.shape[1]):
            xj = X[:, j]
            x_centered = xj - xj.mean()
            x_std = xj.std()

            if x_std == 0.0:
                pearson[j] = np.nan
            else:
                pearson[j] = np.dot(x_centered, y_centered) / (X.shape[0] * x_std * y_std)

        return pearson

    def spearman_rank_correlation(self):
        """
        Spearman rank correlation between each feature column in X and Y.

        Returns:
            np.ndarray of shape (n_features,)
        """
        def rankdata(a):
            """
            rankdata implementation (method='average').
            For ties, assigns the average of the ranks that would have been assigned without ties.
            """
            a = np.asarray(a)
            n = a.size
            sorter = np.argsort(a, kind="mergesort")
            inv = np.empty(n, dtype=int)
            inv[sorter] = np.arange(n)
            a_sorted = a[sorter]

            ranks = np.zeros(n, dtype=float)
            i = 0
            while i < n:
                j = i
                while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
                    j += 1
                rank = 0.5 * (i + j) + 1.0  # average rank, 1-based
                ranks[i:j + 1] = rank
                i = j + 1

            return ranks[inv]

        X = self._X
        Y = self._Y

        Y_rank = rankdata(Y)
        y_centered = Y_rank - Y_rank.mean()
        y_std = Y_rank.std()
        if y_std == 0.0:
            return np.full(X.shape[1], np.nan, dtype=float)

        spearman = np.empty(X.shape[1], dtype=float)

        for j in range(X.shape[1]):
            xj_rank = rankdata(X[:, j])
            x_centered = xj_rank - xj_rank.mean()
            x_std = xj_rank.std()

            if x_std == 0.0:
                spearman[j] = np.nan
            else:
                spearman[j] = np.dot(x_centered, y_centered) / (X.shape[0] * x_std * y_std)

        return spearman

    def result(self):
        return {
            "pearson": self.pearson_correlation_coefficient(),
            "spearman": self.spearman_rank_correlation()
        }

class RegressionEvaluation:
    def __init__(self, X, Y, Y_pred=None):
        self._X = np.asarray(X)
        self._Y = np.asarray(Y).ravel()
        self._Y_pred = None
        if Y_pred is not None:
            self.Y_pred = Y_pred  # uses setter

        if self._X.shape[0] != self._Y.size:
            raise ValueError("X and Y must have the same number of samples.")

    @property
    def Y_pred(self):
        return self._Y_pred

    @Y_pred.setter
    def Y_pred(self, value):
        value = np.asarray(value).ravel()
        if value.size != self._Y.size:
            raise ValueError("Y and Y_pred must have the same length.")
        self._Y_pred = value

    def mse(self):
        return np.mean((self._Y - self._Y_pred) ** 2)

    def rmse(self):
        return float(np.sqrt(self.mse()))

    def mae(self):
        return np.mean(np.abs(self._Y - self._Y_pred))

    def r2(self):
        ss_res = np.sum((self._Y - self._Y_pred) ** 2)
        ss_tot = np.sum((self._Y - self._Y.mean()) ** 2)
        return np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    def adjusted_r2(self):
        X = np.asarray(self._X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]

        # detect intercept column (all ones)
        has_intercept = np.any(np.all(np.isclose(X, 1.0), axis=0))
        p = X.shape[1] - (1 if has_intercept else 0)

        r2_val = self.r2()
        if np.isnan(r2_val) or n <= p + 1:
            return np.nan

        return 1.0 - (1.0 - r2_val) * (n - 1) / (n - p - 1)

    def evaluate(self):
        return {
            "mse": self.mse(),
            "rmse": self.rmse(),
            "mae": self.mae(),
            "r2": self.r2(),
            "adjusted_r2": self.adjusted_r2(),
        }

class ClassificationEvaluation:
    """
    Logistic regression evaluation using:
      - Confusion matrix (TP, FP, FN, TN) at a given threshold
      - Derived rates: accuracy, precision, TPR, TNR, FPR, FNR, FDR, F1
      - ROC curve (FPR/TPR arrays)
      - AUC (trapezoidal rule)

    Inputs:
      X      : (n_samples, n_features)  [kept for signature consistency; not required for metrics]
      Y      : (n_samples,) true labels in {0,1}
      Y_prob : (n_samples,) predicted probabilities P(Y=1|X)
    """

    def __init__(self, X, Y, Y_prob):
        self._X = np.asarray(X)
        self._Y = np.asarray(Y).ravel().astype(int)
        self._Y_prob = np.asarray(Y_prob).ravel().astype(float)

        if self._Y.size != self._Y_prob.size:
            raise ValueError("Y and Y_prob must have the same length.")
        if self._X.shape[0] != self._Y.size:
            raise ValueError("X and Y must have the same number of samples.")
        if not np.all((self._Y == 0) | (self._Y == 1)):
            raise ValueError("Y must contain only 0/1 labels.")

        # Confusion matrix attributes (set when confusion_matrix() is called)
        self.tp = None
        self.fp = None
        self.fn = None
        self.tn = None
        self._threshold_last = None

    def cross_entropy(self):
        # true labels (N,)
        y = np.asarray(self._Y).reshape(-1)

        # probabilities (N,K) or (N,)
        P = np.asarray(self._Y_prob, dtype=float)

        # if P is (N,), convert to (N,2)
        if P.ndim == 1:
            P = np.column_stack([1.0 - P, P])
        
        # if P is (N,1), also convert to (N,2)
        elif P.ndim == 2 and P.shape[1] == 1:
            p1 = P[:, 0]
            P = np.column_stack([1.0 - p1, p1])

        P = np.clip(P, 1e-12, 1.0 - 1e-12)

        # classes order must match columns of P
        classes = getattr(self, "_classes", None)
        if classes is None:
            classes = np.unique(y)
        classes = np.asarray(classes, dtype=object)

        # map each label -> column index
        idx = np.searchsorted(classes, y)

        return float(-np.mean(np.log(P[np.arange(y.size), idx])))

    # -------------------------
    # Confusion matrix (sets tp/fp/fn/tn)
    # -------------------------

    def confusion_matrix(self, threshold=0.5):
        P = np.asarray(self._Y_prob, dtype=float)

        # Accept (N,), (N,1), (N,2)
        if P.ndim == 2:
            if P.shape[1] == 2:
                P = P[:, 1]
            elif P.shape[1] == 1:
                P = P[:, 0]
            else:
                raise ValueError("Only binary probabilities supported: shape must be (N,), (N,1), or (N,2).")

        Y_pred = (P >= threshold).astype(int)

        self.tp = int(np.sum((self._Y == 1) & (Y_pred == 1)))
        self.fp = int(np.sum((self._Y == 0) & (Y_pred == 1)))
        self.fn = int(np.sum((self._Y == 1) & (Y_pred == 0)))
        self.tn = int(np.sum((self._Y == 0) & (Y_pred == 0)))

        self._threshold_last = float(threshold)
        return self.tp, self.fp, self.fn, self.tn

    def _ensure_confusion(self, threshold=0.5):
        """
        Ensure tp/fp/fn/tn are available for the given threshold.
        """
        if (self.tp is None) or (self._threshold_last != float(threshold)):
            self.confusion_matrix(threshold=threshold)

    # -------------------------
    # Scalar metrics (threshold-based)
    # -------------------------

    def accuracy(self, threshold=0.5):
        self._ensure_confusion(threshold)
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self, threshold=0.5):
        self._ensure_confusion(threshold)
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    def f1(self, threshold=0.5):
        """
        F1 score = 2 * (precision * recall) / (precision + recall)
        where recall = TPR.
        """
        prec = self.precision(threshold)
        rec = self.tpr(threshold)
        denom = prec + rec
        return (2.0 * prec * rec / denom) if denom > 0 else 0.0
    
    def tpr(self, threshold=0.5):
        """True Positive Rate (Recall, Sensitivity)."""
        self._ensure_confusion(threshold)
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def tnr(self, threshold=0.5):
        """True Negative Rate (Specificity)."""
        self._ensure_confusion(threshold)
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    def fpr(self, threshold=0.5):
        """False Positive Rate."""
        self._ensure_confusion(threshold)
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    def fnr(self, threshold=0.5):
        """False Negative Rate."""
        self._ensure_confusion(threshold)
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0

    def fdr(self, threshold=0.5):
        """False Discovery Rate = FP / (TP + FP) = 1 - precision."""
        self._ensure_confusion(threshold)
        denom = self.tp + self.fp
        return self.fp / denom if denom > 0 else 0.0

    # -------------------------
    # ROC + AUC (threshold-free)
    # -------------------------

    def roc(self):
        """
        Compute ROC curve points (FPR, TPR) by sweeping thresholds
        over sorted predicted probabilities (descending).

        Returns:
          fpr: np.ndarray
          tpr: np.ndarray
          thresholds: np.ndarray
        """
        Y = self._Y
        P = self._Y_prob
        P = np.asarray(P, dtype=float)
        
        # Accept (N,), (N,1), (N,2)
        if P.ndim == 2:
            if P.shape[1] == 2:
                P = P[:, 1]
            elif P.shape[1] == 1:
                P = P[:, 0]
            else:
                raise ValueError("Only binary probabilities supported: shape must be (N,), (N,1), or (N,2).")


        pos = int(np.sum(Y == 1))
        neg = int(np.sum(Y == 0))
        if pos == 0 or neg == 0:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

        order = np.argsort(P)[::-1]
        Y_sorted = Y[order]
        P_sorted = P[order]

        tp = 0
        fp = 0
        tpr_list = [0.0]
        fpr_list = [0.0]
        thr_list = [np.inf]

        for i in range(Y_sorted.size):
            if Y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / pos)
            fpr_list.append(fp / neg)
            thr_list.append(P_sorted[i])

        return np.asarray(fpr_list), np.asarray(tpr_list), np.asarray(thr_list)

    def auc(self):
        """
        Compute AUC-ROC using trapezoidal rule on the ROC curve.
        Returns np.nan if ROC is undefined.
        """
        fpr, tpr, _ = self.roc()
        if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
            return np.nan
        return float(np.trapezoid(tpr, fpr))

    # -------------------------
    # Optional: one-shot summary
    # -------------------------

    def evaluate(self, threshold=0.5):
        """
        Return all confusion-derived metrics + ROC + AUC in a dict.
        """
        self._ensure_confusion(threshold)
        return {
            "threshold": float(threshold),
            "cross_entropy": self.cross_entropy(),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "accuracy": self.accuracy(threshold),
            "precision": self.precision(threshold),
            "f1": self.f1(threshold),
            "tpr": self.tpr(threshold),
            "tnr": self.tnr(threshold),
            "fpr": self.fpr(threshold),
            "fnr": self.fnr(threshold),
            "fdr": self.fdr(threshold),
            "roc": self.roc(),
            "auc": self.auc(),
        }
