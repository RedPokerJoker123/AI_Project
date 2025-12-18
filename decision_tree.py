import numpy as np
from collections import Counter


# ============================================================
# Internal tree node (used by both classifier + regressor)
# ============================================================
class _TreeNode:
    def __init__(
        self,
        is_leaf=False,
        prediction=None,         # class label (classifier) OR mean y (regression)
        feature_index=None,
        threshold=None,          # for numeric split (binary)
        children=None,           # for categorical split (multiway dict)
        left=None,
        right=None,
        n_samples=0,
        class_counts=None
    ):
        self.is_leaf = bool(is_leaf)
        self.prediction = prediction

        self.feature_index = feature_index
        self.threshold = threshold

        self.children = {} if children is None else children
        self.left = left
        self.right = right

        self.n_samples = int(n_samples)
        self.class_counts = class_counts

# ============================================================
# 1) RegressionDecisionTree  (ID3 / C4.5 / CART)
# ============================================================
class RegressionDecisionTree:
    def __init__(
        self,
        split_type="cart",
        criterion="mse",
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,
        n_bins=10,
        binning="quantile",
        random_state=None
    ):
        self.split_type = str(split_type).lower()
        self.criterion = str(criterion).lower()
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_bins = int(n_bins)
        self.binning = str(binning).lower()
        self.random_state = random_state

        self._root = None
        self._n_features = None
        self._rng = np.random.default_rng(random_state)

        self._y_edges = None
        self._validate_config()

    def _validate_config(self):
        if self.split_type not in ("id3", "c45", "cart"):
            raise ValueError("split_type must be one of: 'id3', 'c45', 'cart'")

        if self.split_type == "cart":
            self.criterion = "mse"
        else:
            if self.split_type == "c45":
                self.criterion = "gain_ratio"
            else:
                if self.criterion not in ("entropy", "gini"):
                    raise ValueError("For split_type='id3', criterion must be 'entropy' or 'gini'")

        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if self.binning not in ("quantile", "uniform"):
            raise ValueError("binning must be 'quantile' or 'uniform'")

    def _mse_impurity(self, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return 0.0
        mu = float(np.mean(y))
        return float(np.mean((y - mu) ** 2))

    def _make_y_bins(self, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.binning == "uniform":
            lo, hi = float(np.min(y)), float(np.max(y))
            edges = np.linspace(lo, hi, self.n_bins + 1)
        else:
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
            if edges.size < 3:
                lo, hi = float(np.min(y)), float(np.max(y))
                edges = np.linspace(lo, hi, self.n_bins + 1)

        self._y_edges = edges
        bins = np.digitize(y, edges[1:-1], right=True)
        return bins.astype(int)

    def _as_classifier(self):
        if self.split_type == "cart":
            return None
        return DecisionTreeClassifier(
            split_type=self.split_type,
            criterion=("entropy" if self.criterion in ("entropy", "gain_ratio") else "gini"),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )

    def _select_feature_indices(self):
        d = self._n_features
        all_idx = np.arange(d)

        if self.max_features is None:
            return all_idx

        if isinstance(self.max_features, (int, np.integer)):
            k = int(self.max_features)
        else:
            k = int(np.ceil(float(self.max_features) * d))

        k = max(1, min(k, d))
        return np.sort(self._rng.choice(all_idx, size=k, replace=False))

    def _best_split_cart_regression(self, X, y):
        n = X.shape[0]
        parent = self._mse_impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thr = None

        feats = self._select_feature_indices()

        for j in feats:
            xj = X[:, j].astype(float)
            order = np.argsort(xj)
            x_sorted = xj[order]
            y_sorted = y[order]

            uniq = np.unique(x_sorted)
            if uniq.size <= 1:
                continue

            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in thresholds:
                left = x_sorted <= thr
                right = ~left

                nl = int(left.sum())
                nr = int(right.sum())

                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                yl = y_sorted[left]
                yr = y_sorted[right]

                after = (nl / n) * self._mse_impurity(yl) + (nr / n) * self._mse_impurity(yr)
                gain = parent - after

                if gain > best_gain:
                    best_gain = float(gain)
                    best_feat = int(j)
                    best_thr = float(thr)

        return best_feat, best_thr, float(best_gain)

    def _build_cart_regression(self, X, y, depth):
        n = X.shape[0]
        if (self.max_depth is not None and depth >= self.max_depth) or (n < self.min_samples_split):
            return _TreeNode(is_leaf=True, prediction=float(np.mean(y)), n_samples=n)

        feat, thr, gain = self._best_split_cart_regression(X, y)
        if feat is None or gain <= 0.0:
            return _TreeNode(is_leaf=True, prediction=float(np.mean(y)), n_samples=n)

        left_mask = X[:, feat].astype(float) <= thr
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _TreeNode(is_leaf=True, prediction=float(np.mean(y)), n_samples=n)

        node = _TreeNode(is_leaf=False, feature_index=feat, threshold=thr, n_samples=n)
        node.left = self._build_cart_regression(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_cart_regression(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self._n_features = X.shape[1]

        if self.split_type == "cart":
            self._root = self._build_cart_regression(X, y, depth=0)
            return self

        y_bins = self._make_y_bins(y)
        clf = self._as_classifier()
        clf.fit(X, y_bins)

        leaf_values = {}

        def route_one(x_row, y_val, node):
            cur = node
            while not cur.is_leaf:
                if cur.threshold is not None:
                    if float(x_row[cur.feature_index]) <= cur.threshold:
                        cur = cur.left
                    else:
                        cur = cur.right
                else:
                    v = x_row[cur.feature_index]
                    nxt = cur.children.get(v, None)
                    if nxt is None:
                        break
                    cur = nxt
            leaf_values.setdefault(id(cur), []).append(float(y_val))

        for i in range(X.shape[0]):
            route_one(X[i], y[i], clf._root)

        def clone(node):
            if node is None:
                return None
            new_node = _TreeNode(
                is_leaf=node.is_leaf,
                prediction=None,
                feature_index=node.feature_index,
                threshold=node.threshold,
                n_samples=node.n_samples,
                class_counts=node.class_counts
            )
            if node.is_leaf:
                ys = leaf_values.get(id(node), None)
                new_node.prediction = float(np.mean(ys)) if ys else float(np.mean(y))
                return new_node

            if node.threshold is not None:
                new_node.left = clone(node.left)
                new_node.right = clone(node.right)
            else:
                for k, child in node.children.items():
                    new_node.children[k] = clone(child)

            return new_node

        self._root = clone(clf._root)
        return self

    def _predict_one(self, x, node):
        cur = node
        while not cur.is_leaf:
            if cur.threshold is not None:
                if float(x[cur.feature_index]) <= cur.threshold:
                    cur = cur.left
                else:
                    cur = cur.right
            else:
                v = x[cur.feature_index]
                nxt = cur.children.get(v, None)
                if nxt is None:
                    if cur.children:
                        best_child = None
                        best_n = -1
                        for c in cur.children.values():
                            if c is not None and c.n_samples > best_n:
                                best_n = c.n_samples
                                best_child = c
                        if best_child is not None:
                            cur = best_child
                            continue
                    break
                cur = nxt
        return float(cur.prediction)

    def predict(self, X):
        if self._root is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        preds = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            preds[i] = self._predict_one(X[i], self._root)
        return preds

# ============================================================
# 2) DecisionTreeClassifier  (ID3 / C4.5 / CART)
# ============================================================
class DecisionTreeClassifier:
    def __init__(
        self,
        split_type="id3",
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None
    ):
        self.split_type = str(split_type).lower()
        self.criterion = str(criterion).lower()
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self._root = None
        self._n_features = None
        self._classes = None
        self._rng = np.random.default_rng(random_state)

        self._validate_config()

    def _validate_config(self):
        if self.split_type not in ("id3", "c45", "cart"):
            raise ValueError("split_type must be one of: 'id3', 'c45', 'cart'")

        if self.split_type == "c45":
            self.criterion = "gain_ratio"

        if self.split_type == "id3":
            if self.criterion not in ("entropy", "gini"):
                raise ValueError("For split_type='id3', criterion must be 'entropy' or 'gini'")

        if self.split_type == "cart":
            if self.criterion not in ("gini", "entropy"):
                raise ValueError("For split_type='cart', criterion must be 'gini' or 'entropy'")

        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be None or >= 0")

        if self.max_features is not None:
            if isinstance(self.max_features, (int, np.integer)):
                if int(self.max_features) <= 0:
                    raise ValueError("max_features (int) must be > 0")
            else:
                mf = float(self.max_features)
                if not (0.0 < mf <= 1.0):
                    raise ValueError("max_features (float) must be in (0,1]")

    def _entropy(self, y):
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        counts = Counter(y.tolist())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()], dtype=float)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _gini(self, y):
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        counts = Counter(y.tolist())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()], dtype=float)
        return float(1.0 - np.sum(probs ** 2))

    def _impurity(self, y):
        if self.criterion in ("entropy", "gain_ratio"):
            return self._entropy(y)
        return self._gini(y)

    def _split_info(self, group_sizes, n_total):
        probs = np.array(group_sizes, dtype=float) / float(n_total)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _gain_categorical(self, y, x_col):
        n = len(y)
        before = self._impurity(y)

        values, inv = np.unique(x_col, return_inverse=True)
        after = 0.0
        sizes = []

        for v in range(len(values)):
            mask = (inv == v)
            yv = y[mask]
            sizes.append(int(yv.size))
            after += (yv.size / n) * self._impurity(yv)

        gain = before - after

        if self.criterion == "gain_ratio":
            si = self._split_info(sizes, n)
            return gain / (si + 1e-12)

        return gain

    def _best_threshold_numeric(self, y, x_col):
        x = np.asarray(x_col, dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        uniq = np.unique(x_sorted)
        if uniq.size <= 1:
            return None, -np.inf

        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        best_thr = None
        best_score = -np.inf

        n = len(y)
        parent_imp = self._impurity(y_sorted)

        for thr in thresholds:
            left_mask = x_sorted <= thr
            right_mask = ~left_mask

            nl = int(left_mask.sum())
            nr = int(right_mask.sum())

            if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                continue

            yl = y_sorted[left_mask]
            yr = y_sorted[right_mask]

            after = (nl / n) * self._impurity(yl) + (nr / n) * self._impurity(yr)
            gain = parent_imp - after

            if self.criterion == "gain_ratio":
                si = self._split_info([nl, nr], n)
                score = gain / (si + 1e-12)
            else:
                score = gain

            if score > best_score:
                best_score = float(score)
                best_thr = float(thr)

        return best_thr, float(best_score)

    def _majority_class(self, y):
        counts = Counter(y.tolist())
        return counts.most_common(1)[0][0]

    def _class_counts(self, y):
        return dict(Counter(y.tolist()))

    def _select_feature_indices(self):
        d = self._n_features
        all_idx = np.arange(d)

        if self.max_features is None:
            return all_idx

        if isinstance(self.max_features, (int, np.integer)):
            k = int(self.max_features)
        else:
            k = int(np.ceil(float(self.max_features) * d))

        k = max(1, min(k, d))
        return np.sort(self._rng.choice(all_idx, size=k, replace=False))

    def _is_numeric_column(self, col):
        col = np.asarray(col)
        if np.issubdtype(col.dtype, np.number):
            return True
        try:
            _ = col.astype(float)
            return True
        except Exception:
            return False

    def _best_split(self, X, y, feature_indices):
        best_feature = None
        best_threshold = None
        best_score = -np.inf
        best_is_numeric = False

        for j in feature_indices:
            col = X[:, j]
            is_num = self._is_numeric_column(col)

            if is_num:
                thr, score = self._best_threshold_numeric(y, col)
                if thr is None:
                    continue
                if score > best_score:
                    best_score = float(score)
                    best_feature = int(j)
                    best_threshold = float(thr)
                    best_is_numeric = True
            else:
                score = self._gain_categorical(y, col)
                if score > best_score:
                    best_score = float(score)
                    best_feature = int(j)
                    best_threshold = None
                    best_is_numeric = False

        return best_feature, best_threshold, best_score, best_is_numeric

    def _build(self, X, y, depth):
        n = X.shape[0]

        if len(set(y.tolist())) == 1:
            return _TreeNode(is_leaf=True, prediction=y[0], n_samples=n, class_counts=self._class_counts(y))

        if (self.max_depth is not None and depth >= self.max_depth) or (n < self.min_samples_split):
            pred = self._majority_class(y)
            return _TreeNode(is_leaf=True, prediction=pred, n_samples=n, class_counts=self._class_counts(y))

        feature_indices = self._select_feature_indices()
        feat, thr, score, is_num = self._best_split(X, y, feature_indices)

        if feat is None or not np.isfinite(score) or score <= 0.0:
            pred = self._majority_class(y)
            return _TreeNode(is_leaf=True, prediction=pred, n_samples=n, class_counts=self._class_counts(y))

        node = _TreeNode(
            is_leaf=False,
            prediction=None,
            feature_index=int(feat),
            threshold=float(thr) if thr is not None else None,
            n_samples=n,
            class_counts=self._class_counts(y)
        )

        if is_num and thr is not None:
            left_mask = X[:, feat].astype(float) <= thr
            right_mask = ~left_mask

            nl = int(left_mask.sum())
            nr = int(right_mask.sum())

            if nl == 0 or nr == 0 or nl == n or nr == n:
                pred = self._majority_class(y)
                return _TreeNode(is_leaf=True, prediction=pred, n_samples=n, class_counts=self._class_counts(y))

            if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                pred = self._majority_class(y)
                return _TreeNode(is_leaf=True, prediction=pred, n_samples=n, class_counts=self._class_counts(y))

            node.left = self._build(X[left_mask], y[left_mask], depth + 1)
            node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        else:
            col = X[:, feat]
            values = np.unique(col)
            for v in values:
                mask = (col == v)
                if mask.sum() == 0:
                    continue
                node.children[v] = self._build(X[mask], y[mask], depth + 1)

        return node

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self._n_features = X.shape[1]
        self._classes = np.unique(y)
        self._root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        while not node.is_leaf:
            f = node.feature_index
            if node.threshold is not None:
                if float(x[f]) <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                v = x[f]
                child = node.children.get(v, None)
                if child is None:
                    if node.class_counts:
                        return max(node.class_counts.items(), key=lambda kv: kv[1])[0]
                    return None
                node = child
        return node.prediction

    def predict(self, X):
        if self._root is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        preds = np.empty(X.shape[0], dtype=object)
        for i in range(X.shape[0]):
            preds[i] = self._predict_one(X[i], self._root)
        return preds
    
    def predict_proba(self, X):
        if self._root is None:
            raise ValueError("Model not fitted yet.")
        X = np.asarray(X)
        classes = np.asarray(self._classes, dtype=object)
        K = classes.size
        if K == 0:
            raise ValueError("No classes learned.")

        proba = np.zeros((X.shape[0], K), dtype=float)

        for i in range(X.shape[0]):
            node = self._root
            x = X[i]

            while not node.is_leaf:
                f = node.feature_index
                if node.threshold is not None:
                    if float(x[f]) <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    v = x[f]
                    child = node.children.get(v, None)
                    if child is None:
                        break
                    node = child

            # use leaf counts if available; otherwise fallback to predicted class
            counts = getattr(node, "class_counts", None)
            if isinstance(counts, dict) and len(counts) > 0:
                total = float(sum(counts.values()))
                if total <= 0:
                    proba[i, :] = 1.0 / K
                else:
                    for j, c in enumerate(classes):
                        proba[i, j] = float(counts.get(c, 0)) / total
            else:
                # fallback: one-hot on the predicted label
                pred = node.prediction
                j = np.where(classes == pred)[0]
                if j.size == 0:
                    proba[i, :] = 1.0 / K
                else:
                    proba[i, :] = 0.0
                    proba[i, int(j[0])] = 1.0

        return proba

class SVMClassifier:
    """
    Unified SVM Classifier:
      - kernel = "linear"  -> primal hinge-loss SVM
      - kernel = "rbf"     -> dual RBF-kernel SVM (SMO)

    Binary classification only.
    """

    def __init__(
        self,
        kernel="linear",       # "linear" | "rbf"
        C=1.0,
        gamma=1.0,             # used only for rbf
        learning_rate=0.01,    # linear only
        epochs=1,           # linear only
        tol=1e-3,              # rbf only
        max_passes=5,          # rbf only
        random_state=None
    ):
        self.kernel = str(kernel).lower()
        self.C = float(C)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.max_passes = int(max_passes)
        self.random_state = random_state

        # learned state
        self._w = None
        self._b = 0.0
        self.X_train_ = None
        self.y_train_ = None
        self.alphas_ = None
        self.K_ = None

        if self.kernel not in ("linear", "rbf"):
            raise ValueError("kernel must be 'linear' or 'rbf'")

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _rbf_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)

        Xn = np.sum(X ** 2, axis=1)[:, None]
        Yn = np.sum(Y ** 2, axis=1)[None, :]
        dist = Xn + Yn - 2.0 * X @ Y.T
        return np.exp(-self.gamma * np.maximum(dist, 0.0))

    def _encode_labels(self, y):
        classes = np.unique(y)

        if classes.size != 2:
            # could be 1 (all same class) or >2 (multiclass)
            raise ValueError("SVMClassifier supports binary classification only.")

        # now safe: classes[0], classes[1] exist
        if set(classes) == {0, 1}:
            return np.where(y == 1, 1.0, -1.0), classes

        return np.where(y == classes[1], 1.0, -1.0), classes

    # --------------------------------------------------
    # Linear SVM (primal)
    # --------------------------------------------------
    def _fit_linear(self, X, y_signed):
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self._w = rng.normal(0.0, 0.01, size=d)
        self._b = 0.0

        for _ in range(self.epochs):
            margins = y_signed * (X @ self._w + self._b)
            mis = margins < 1.0

            dw = self._w - self.C * np.sum(y_signed[mis, None] * X[mis], axis=0)
            db = -self.C * np.sum(y_signed[mis])

            self._w -= self.learning_rate * dw
            self._b -= self.learning_rate * db

    # --------------------------------------------------
    # RBF SVM (dual, SMO)
    # --------------------------------------------------
    def _fit_rbf(self, X, y_signed):
        n = X.shape[0]
        self.X_train_ = X
        self.y_train_ = y_signed
        self.alphas_ = np.zeros(n)
        self.b_ = 0.0
        self.K_ = self._rbf_kernel(X)

        rng = np.random.default_rng(self.random_state)
        passes = 0

        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                f_i = np.sum(self.alphas_ * y_signed * self.K_[:, i]) + self.b_
                E_i = f_i - y_signed[i]

                if ((y_signed[i] * E_i < -self.tol and self.alphas_[i] < self.C) or
                    (y_signed[i] * E_i > self.tol and self.alphas_[i] > 0)):

                    j = i
                    while j == i:
                        j = rng.integers(0, n)

                    f_j = np.sum(self.alphas_ * y_signed * self.K_[:, j]) + self.b_
                    E_j = f_j - y_signed[j]

                    ai, aj = self.alphas_[i], self.alphas_[j]

                    if y_signed[i] != y_signed[j]:
                        L = max(0.0, aj - ai)
                        H = min(self.C, self.C + aj - ai)
                    else:
                        L = max(0.0, ai + aj - self.C)
                        H = min(self.C, ai + aj)

                    if L == H:
                        continue

                    eta = 2.0 * self.K_[i, j] - self.K_[i, i] - self.K_[j, j]
                    if eta >= 0:
                        continue

                    self.alphas_[j] -= y_signed[j] * (E_i - E_j) / eta
                    self.alphas_[j] = np.clip(self.alphas_[j], L, H)

                    if abs(self.alphas_[j] - aj) < 1e-5:
                        continue

                    self.alphas_[i] += y_signed[i] * y_signed[j] * (aj - self.alphas_[j])

                    b1 = self.b_ - E_i \
                        - y_signed[i] * (self.alphas_[i] - ai) * self.K_[i, i] \
                        - y_signed[j] * (self.alphas_[j] - aj) * self.K_[i, j]

                    b2 = self.b_ - E_j \
                        - y_signed[i] * (self.alphas_[i] - ai) * self.K_[i, j] \
                        - y_signed[j] * (self.alphas_[j] - aj) * self.K_[j, j]

                    if 0 < self.alphas_[i] < self.C:
                        self.b_ = b1
                    elif 0 < self.alphas_[j] < self.C:
                        self.b_ = b2
                    else:
                        self.b_ = 0.5 * (b1 + b2)

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).ravel()

        y_signed, self._classes = self._encode_labels(y)

        if self.kernel == "linear":
            self._fit_linear(X, y_signed)
        else:
            self._fit_rbf(X, y_signed)

        return self

    def predict(self, X):
        X = np.asarray(X, float)

        if self.kernel == "linear":
            scores = X @ self._w + self._b
        else:
            K = self._rbf_kernel(X, self.X_train_)
            scores = K @ (self.alphas_ * self.y_train_) + self.b_

        return np.where(scores >= 0, self._classes[1], self._classes[0])
    
    def decision_function(self, X):
        X = np.asarray(X, float)
        if self.kernel == "linear":
            return X @ self._w + self._b
        else:
            K = self._rbf_kernel(X, self.X_train_)
            return K @ (self.alphas_ * self.y_train_) + self.b_

    def predict_proba(self, X):
        X = np.asarray(X, float)

        if self.kernel == "linear":
            scores = X @ self._w + self._b
        else:
            K = self._rbf_kernel(X, self.X_train_)
            scores = K @ (self.alphas_ * self.y_train_) + self.b_

        # stable sigmoid
        scores = np.clip(scores, -500.0, 500.0)
        p_pos = 1.0 / (1.0 + np.exp(-scores))

        # IMPORTANT: return (n,2) in the SAME order as self._classes
        # convention here: p_pos corresponds to self._classes[1]
        proba = np.column_stack([1.0 - p_pos, p_pos])
        return proba

# ============================================================
# 3) EnsembleLearning  (Bagging + Feature Bagging + OOB + AdaBoost)
# ============================================================
class EnsembleLearning:
    def __init__(
        self,
        base_estimator,
        method="bagging",
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        oob_score=False,
        random_state=None,
        feature_selection=None,     # None | "subspace" | "mask"
        mask_keep_prob=0.7
    ):
        self.base_estimator = base_estimator
        self.method = str(method).lower()
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.max_features = max_features
        self.oob_score = bool(oob_score)
        self.random_state = random_state

        self.feature_selection = None if feature_selection is None else str(feature_selection).lower()
        self.mask_keep_prob = float(mask_keep_prob)

        self._rng = np.random.default_rng(random_state)

        self._estimators = []
        self._feature_indices = []
        self._oob_score = None

        self._estimator_weights = []
        self._classes = None

        self._validate_config()

    def _validate_config(self):
        if self.method not in ("bagging", "adaboost"):
            raise ValueError("method must be 'bagging' or 'adaboost'")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        if self.feature_selection not in ("none", "subspace", "mask"):
            raise ValueError("feature_selection must be 'none', 'subspace', or 'mask'")
        if not (0.0 < self.mask_keep_prob <= 1.0):
            raise ValueError("mask_keep_prob must be in (0,1]")

    def _bootstrap_indices(self, n):
        if isinstance(self.max_samples, (int, np.integer)):
            size = int(self.max_samples)
        else:
            ms = float(self.max_samples)
            if not (0.0 < ms <= 1.0):
                raise ValueError("max_samples (float) must be in (0,1]")
            size = int(np.ceil(ms * n))
        size = max(1, min(size, n))
        return self._rng.integers(0, n, size=size)

    def _subspace_indices(self, d):
        if isinstance(self.max_features, (int, np.integer)):
            k = int(self.max_features)
        else:
            mf = float(self.max_features)
            if not (0.0 < mf <= 1.0):
                raise ValueError("max_features (float) must be in (0,1]")
            k = int(np.ceil(mf * d))
        k = max(1, min(k, d))
        return np.sort(self._rng.choice(np.arange(d), size=k, replace=False))

    def _mask_indices(self, d):
        keep = self._rng.random(d) < self.mask_keep_prob
        if not np.any(keep):
            keep[self._rng.integers(0, d)] = True
        return np.where(keep)[0]

    def _choose_feature_indices_for_estimator(self, d):
        if self.feature_selection is None or self.feature_selection == "subspace":
            return self._subspace_indices(d)
        return self._mask_indices(d)

    def _majority_vote(self, labels):
        counts = Counter(labels.tolist())
        return counts.most_common(1)[0][0]

    def _clone_estimator(self, estimator):
        estimator_type = type(estimator)

        if estimator_type is DecisionTreeClassifier:
            return DecisionTreeClassifier(
                split_type=estimator.split_type,
                criterion=estimator.criterion,
                max_depth=estimator.max_depth,
                min_samples_split=estimator.min_samples_split,
                min_samples_leaf=estimator.min_samples_leaf,
                max_features=estimator.max_features,
                random_state=int(self._rng.integers(0, 10**9))
            )

        if estimator_type is RegressionDecisionTree:
            return RegressionDecisionTree(
                split_type=estimator.split_type,
                criterion=estimator.criterion,
                max_depth=estimator.max_depth,
                min_samples_split=estimator.min_samples_split,
                min_samples_leaf=estimator.min_samples_leaf,
                max_features=estimator.max_features,
                n_bins=estimator.n_bins,
                binning=estimator.binning,
                random_state=int(self._rng.integers(0, 10**9))
            )

        if estimator_type is SVMClassifier:
            return SVMClassifier(
                kernel=estimator.kernel,
                C=estimator.C,
                gamma=estimator.gamma,
                learning_rate=estimator.learning_rate,
                epochs=estimator.epochs,
                tol=estimator.tol,
                max_passes=estimator.max_passes,
                random_state=int(self._rng.integers(0, 10**9))
            )

        raise ValueError("Unsupported base_estimator type.")

    def _fit_bagging(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        self._classes = np.unique(y)

        self._estimators = []
        self._feature_indices = []
        self._oob_score = None

        oob_predictions = [[] for _ in range(n)] if self.oob_score else None

        for _ in range(self.n_estimators):
            feat_idx = self._choose_feature_indices_for_estimator(d)
            
            # resample until both classes exist (binary task)
            for _try in range(50):
                sample_idx = self._bootstrap_indices(n)
                y_train = y[sample_idx]
                if np.unique(y_train).size == 2:
                    break
            else:
                # fallback: force at least one from each class
                c0, c1 = self._classes[0], self._classes[1]
                i0 = self._rng.choice(np.where(y == c0)[0])
                i1 = self._rng.choice(np.where(y == c1)[0])
                sample_idx = np.concatenate([sample_idx[:-2], [i0, i1]])
                y_train = y[sample_idx]

            X_train = X[sample_idx][:, feat_idx]
            y_train = y[sample_idx]

            estimator = self._clone_estimator(self.base_estimator)
            estimator.fit(X_train, y_train)

            self._estimators.append(estimator)
            self._feature_indices.append(feat_idx)

            if self.oob_score:
                in_bag = set(sample_idx.tolist())
                oob_idx = [i for i in range(n) if i not in in_bag]
                if oob_idx:
                    X_oob = X[oob_idx][:, feat_idx]
                    preds = estimator.predict(X_oob)
                    for i, p in zip(oob_idx, preds):
                        oob_predictions[i].append(p)

        if self.oob_score:
            correct = 0
            count = 0
            for i in range(n):
                if oob_predictions[i]:
                    y_hat = self._majority_vote(np.array(oob_predictions[i], dtype=object))
                    if y_hat == y[i]:
                        correct += 1
                    count += 1
            self._oob_score = (correct / count) if count > 0 else None

        return self

    def _predict_bagging(self, X):
        X = np.asarray(X)
        n = X.shape[0]

        all_preds = []
        for estimator, feat_idx in zip(self._estimators, self._feature_indices):
            all_preds.append(estimator.predict(X[:, feat_idx]))

        all_preds = np.array(all_preds, dtype=object)

        preds = np.empty(n, dtype=object)
        for i in range(n):
            preds[i] = self._majority_vote(all_preds[:, i])
        return preds

    def _fit_adaboost(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("AdaBoost here supports only binary classification.")

        self._classes = classes
        y_signed = np.where(y == classes[1], 1.0, -1.0)

        n, d = X.shape
        weights = np.full(n, 1.0 / n, dtype=float)

        self._estimators = []
        self._estimator_weights = []
        self._feature_indices = []
        self._oob_score = None

        for _ in range(self.n_estimators):
            feat_idx = self._choose_feature_indices_for_estimator(d)
            X_sub = X[:, feat_idx]

            # resample until both classes exist (needed for binary-only learners like SVM)
            for _try in range(50):
                idx = self._rng.choice(np.arange(n), size=n, replace=True, p=weights)
                y_train = y[idx]
                if np.unique(y_train).size == 2:
                    break
            else:
                # fallback: force at least one from each class
                c0, c1 = classes[0], classes[1]
                i0 = self._rng.choice(np.where(y == c0)[0], p=weights[np.where(y == c0)[0]] /
                                    np.sum(weights[np.where(y == c0)[0]]))
                i1 = self._rng.choice(np.where(y == c1)[0], p=weights[np.where(y == c1)[0]] /
                                    np.sum(weights[np.where(y == c1)[0]]))
                idx[-2:] = [i0, i1]
                y_train = y[idx]

            X_train = X_sub[idx]
            y_train = y[idx]

            estimator = self._clone_estimator(self.base_estimator)
            estimator.fit(X_train, y_train)

            pred = estimator.predict(X_sub)
            pred_signed = np.where(pred == classes[1], 1.0, -1.0)

            mis = (pred_signed != y_signed)
            err = float(np.sum(weights[mis]))

            if err <= 1e-12:
                alpha = 1.0
                self._estimators.append(estimator)
                self._estimator_weights.append(float(alpha))
                self._feature_indices.append(feat_idx)
                break

            err = min(max(err, 1e-12), 1.0 - 1e-12)
            alpha = 0.5 * np.log((1.0 - err) / err)

            weights *= np.exp(alpha * mis.astype(float))
            weights /= np.sum(weights)

            self._estimators.append(estimator)
            self._estimator_weights.append(float(alpha))
            self._feature_indices.append(feat_idx)

        return self

    def _predict_adaboost(self, X):
        if not self._estimators:
            raise ValueError("AdaBoost not fitted.")
        X = np.asarray(X)

        scores = np.zeros(X.shape[0], dtype=float)

        for estimator, alpha, feat_idx in zip(self._estimators, self._estimator_weights, self._feature_indices):
            X_sub = X[:, feat_idx]
            pred = estimator.predict(X_sub)
            pred_signed = np.where(pred == self._classes[1], 1.0, -1.0)
            scores += float(alpha) * pred_signed

        return np.where(scores > 0.0, self._classes[1], self._classes[0])

    def fit(self, X, y):
        if self.method == "bagging":
            return self._fit_bagging(X, y)
        return self._fit_adaboost(X, y)

    def predict(self, X):
        if self.method == "bagging":    
            return self._predict_bagging(X)
        return self._predict_adaboost(X)
    
    def predict_proba(self, X):
        if not self._estimators:
            raise ValueError("EnsembleLearning not fitted.")
        X = np.asarray(X)
        n = X.shape[0]

        # infer / store class order once (keeps columns consistent)
        if self._classes is None:
            # try to recover from first estimator, else from training labels used in adaboost
            if hasattr(self._estimators[0], "_classes") and self._estimators[0]._classes is not None:
                self._classes = np.asarray(self._estimators[0]._classes, dtype=object)
            else:
                self._classes = np.unique(self._estimators[0].predict(X[:, self._feature_indices[0]]))
        classes = np.asarray(self._classes, dtype=object)
        if classes.size < 2:
            raise ValueError("predict_proba currently supports binary classification only.")

        # helper: map each estimator's proba columns to [classes[0], classes[1]]
        def align_proba(p, est, global_classes):
            p = np.asarray(p, float)
            if p.ndim == 1:  # binary shorthand -> expand to 2D
                p = np.column_stack([1.0 - p, p])

            est_classes = np.asarray(getattr(est, "_classes", global_classes), dtype=object)
            out = np.zeros((p.shape[0], len(global_classes)), dtype=float)
            for k, c in enumerate(global_classes):
                j = np.where(est_classes == c)[0]
                out[:, k] = p[:, int(j[0])] if j.size else 0.0
            return out

        if self.method == "bagging":
            P = np.zeros((n, 2), dtype=float)
            for est, feat_idx in zip(self._estimators, self._feature_indices):
                p = align_proba(est.predict_proba(X[:, feat_idx]), est, self._classes)
                P += p
            P /= float(len(self._estimators))
            return P

        # adaboost: weighted vote, then logistic squashing for probabilities
        scores = np.zeros(n, dtype=float)
        for est, alpha, feat_idx in zip(self._estimators, self._estimator_weights, self._feature_indices):
            pred = est.predict(X[:, feat_idx])
            pred_signed = np.where(pred == classes[1], 1.0, -1.0)
            scores += float(alpha) * pred_signed

        # convert margin -> P(y=classes[1]) using sigmoid(2*F(x)) (common AdaBoost link)
        p1 = 1.0 / (1.0 + np.exp(-2.0 * scores))
        P = np.column_stack([1.0 - p1, p1])
        return P
