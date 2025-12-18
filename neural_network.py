"""
mlp_from_scratch_full_dropin.py

3 classes (NumPy only, from scratch, backprop):
    1) MLPRegressor
    2) MLPClassifier (binary)
    3) MultiTaskMLP (regression + binary classification)

Options you asked for:
    - Activation for ALL hidden layers:
        {"linear","tanh","sigmoid","relu","leaky_relu","special_relu"}
    - GD optimizers:
        {"vanilla","momentum","nesterov","adagrad","rmsprop","adadelta","adam"}
    - Batch modes:
        {"batch","stochastic","minibatch"}
    - Regularization:
        penalty ∈ {"none","ridge","lasso","elasticnet"}
        alpha, l1_ratio

Notes:
    - Backpropagation is used for all models.
    - Regularization is applied to WEIGHTS only (not biases).
    - Uses He init for ReLU-like activations, Xavier for tanh/sigmoid/linear.
"""

import numpy as np

EPOCHS = 100

# ============================================================
# Utilities
# ============================================================

def _safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _batch_iterator(X, y_list, mode="batch", batch_size=32, rng=None, shuffle=True):
    mode = str(mode).lower()
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)

    if mode == "batch":
        yield X, y_list
        return

    if mode == "stochastic":
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for i in idx:
            Xi = X[i:i+1]
            yi = [y[i:i+1] for y in y_list]
            yield Xi, yi
        return

    if mode == "minibatch":
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be > 0 for minibatch mode")

        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        for start in range(0, n, bs):
            sel = idx[start:start+bs]
            Xb = X[sel]
            yb = [y[sel] for y in y_list]
            yield Xb, yb
        return

    raise ValueError("mode must be one of: batch, stochastic, minibatch")

def _kmeans_centers(X, n_clusters, n_iters=25, random_state=0):
    X = np.asarray(X, float)
    n, d = X.shape
    rng = np.random.default_rng(int(random_state))
    if n_clusters > n:
        raise ValueError("n_centers cannot exceed number of samples.")

    centers = X[rng.choice(n, size=n_clusters, replace=False)].copy()

    for _ in range(int(n_iters)):
        # assign
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (n,k)
        labels = np.argmin(dists, axis=1)
        # update
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
    return centers


def _rbf_design(X, centers, gamma):
    """
    Phi[i,k] = exp(-gamma * ||x_i - c_k||^2)
    """
    X = np.asarray(X, float)
    C = np.asarray(centers, float)
    Xn = np.sum(X**2, axis=1)[:, None]      # (n,1)
    Cn = np.sum(C**2, axis=1)[None, :]      # (1,k)
    sq = Xn + Cn - 2.0 * (X @ C.T)
    sq = np.maximum(sq, 0.0)
    return np.exp(-float(gamma) * sq)       # (n,k)

# ============================================================
# Regularization
# ============================================================

class Regularizer:
    """
    penalty: "none", "ridge", "lasso", "elasticnet"
    alpha: overall regularization strength
    l1_ratio: used only for elasticnet
    """
    def __init__(self, penalty="none", alpha=0.0, l1_ratio=0.5):
        self.penalty = str(penalty).lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)

        if self.penalty not in ("none", "ridge", "lasso", "elasticnet"):
            raise ValueError("penalty must be one of: none, ridge, lasso, elasticnet")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0,1]")

    def loss(self, weights, n_scale=1.0):
        if self.penalty == "none" or self.alpha == 0.0:
            return 0.0

        a = self.alpha / max(1.0, float(n_scale))

        if self.penalty == "ridge":
            return 0.5 * a * sum(np.sum(W * W) for W in weights)

        if self.penalty == "lasso":
            return a * sum(np.sum(np.abs(W)) for W in weights)

        # elasticnet
        l1 = a * self.l1_ratio
        l2 = a * (1.0 - self.l1_ratio)
        return (l1 * sum(np.sum(np.abs(W)) for W in weights)) + (0.5 * l2 * sum(np.sum(W * W) for W in weights))

    def grad(self, W, n_scale=1.0):
        if self.penalty == "none" or self.alpha == 0.0:
            return np.zeros_like(W)

        a = self.alpha / max(1.0, float(n_scale))

        if self.penalty == "ridge":
            return a * W

        if self.penalty == "lasso":
            return a * np.sign(W)

        # elasticnet
        l1 = a * self.l1_ratio
        l2 = a * (1.0 - self.l1_ratio)
        return (l2 * W) + (l1 * np.sign(W))


# ============================================================
# Optimizers (GD options)
# ============================================================

class Optimizer:
    """
    method ∈ {"vanilla","momentum","nesterov","adagrad","rmsprop","adadelta","adam"}
    """
    def __init__(
        self,
        method="adam",
        lr=0.01,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        rho=0.95
    ):
        self.method = str(method).lower()
        self.lr = float(lr)

        self.momentum = float(momentum)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.rho = float(rho)

        if self.method not in ("vanilla","momentum","nesterov","adagrad","rmsprop","adadelta","adam"):
            raise ValueError("Unknown optimizer method")

        self.state = {}
        self.t = 0

    def _get_state(self, param):
        pid = id(param)
        if pid not in self.state:
            self.state[pid] = {}
            if self.method in ("momentum","nesterov"):
                self.state[pid]["v"] = np.zeros_like(param)
            if self.method == "adagrad":
                self.state[pid]["G"] = np.zeros_like(param)
            if self.method == "rmsprop":
                self.state[pid]["Eg2"] = np.zeros_like(param)
            if self.method == "adadelta":
                self.state[pid]["Eg2"] = np.zeros_like(param)
                self.state[pid]["Edx2"] = np.zeros_like(param)
            if self.method == "adam":
                self.state[pid]["m"] = np.zeros_like(param)
                self.state[pid]["v"] = np.zeros_like(param)
        return self.state[pid]

    def step(self, param, grad):
        g = grad
        st = self._get_state(param)

        if self.method == "vanilla":
            param -= self.lr * g
            return

        if self.method == "momentum":
            v = st["v"]
            v[:] = self.momentum * v - self.lr * g
            param += v
            return

        if self.method == "nesterov":
            v = st["v"]
            v_prev = v.copy()
            v[:] = self.momentum * v - self.lr * g
            param += (-self.momentum * v_prev + (1.0 + self.momentum) * v)
            return

        if self.method == "adagrad":
            G = st["G"]
            G[:] = G + g * g
            param -= (self.lr / (np.sqrt(G) + self.eps)) * g
            return

        if self.method == "rmsprop":
            Eg2 = st["Eg2"]
            Eg2[:] = self.rho * Eg2 + (1.0 - self.rho) * (g * g)
            param -= (self.lr / (np.sqrt(Eg2) + self.eps)) * g
            return

        if self.method == "adadelta":
            Eg2 = st["Eg2"]
            Edx2 = st["Edx2"]
            Eg2[:] = self.rho * Eg2 + (1.0 - self.rho) * (g * g)
            rms_dx = np.sqrt(Edx2 + self.eps)
            rms_g  = np.sqrt(Eg2 + self.eps)
            dx = -(rms_dx / rms_g) * g
            param += dx
            Edx2[:] = self.rho * Edx2 + (1.0 - self.rho) * (dx * dx)
            return

        # adam
        self.t += 1
        m = st["m"]
        v = st["v"]
        m[:] = self.beta1 * m + (1.0 - self.beta1) * g
        v[:] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

        mhat = m / (1.0 - self.beta1 ** self.t)
        vhat = v / (1.0 - self.beta2 ** self.t)

        param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return


# ============================================================
# Activation functions (hidden layers)
# ============================================================

class Activation:
    """
    Supports:
      "linear", "tanh", "sigmoid", "relu", "leaky_relu", "special_relu"
    special_relu(x) = max(0, min(1, x))  (clipped ReLU)
    """
    def __init__(self, name="relu", leaky_slope=0.01):
        self.name = str(name).lower()
        self.leaky_slope = float(leaky_slope)

        if self.name not in ("linear","tanh","sigmoid","relu","leaky_relu","special_relu"):
            raise ValueError("activation must be one of: linear, tanh, sigmoid, relu, leaky_relu, special_relu")

    def forward(self, z):
        if self.name == "linear":
            return z
        if self.name == "tanh":
            return np.tanh(z)
        if self.name == "sigmoid":
            return _safe_sigmoid(z)
        if self.name == "relu":
            return np.maximum(0, z)
        if self.name == "leaky_relu":
            return np.where(z > 0, z, self.leaky_slope * z)
        # special_relu
        return np.clip(z, 0.0, 1.0)

    def backward(self, z, grad_a):
        """
        Given z (pre-activation) and upstream gradient wrt activation a,
        return gradient wrt z.
        """
        if self.name == "linear":
            return grad_a
        if self.name == "tanh":
            a = np.tanh(z)
            return grad_a * (1.0 - a * a)
        if self.name == "sigmoid":
            a = _safe_sigmoid(z)
            return grad_a * a * (1.0 - a)
        if self.name == "relu":
            return grad_a * (z > 0)
        if self.name == "leaky_relu":
            return grad_a * np.where(z > 0, 1.0, self.leaky_slope)
        # special_relu: derivative 1 in (0,1), 0 outside, undefined at 0/1 -> use 0
        return grad_a * ((z > 0.0) & (z < 1.0)).astype(float)


# ============================================================
# Base MLP trunk (shared by reg/clf/multitask)
# ============================================================

class _BaseMLP:
    """
    Shared trunk:
      - configurable hidden layer count
      - either constant width via (hidden_layers, hidden_size)
        OR per-layer widths via hidden_sizes=[...]
      - configurable activation for ALL hidden layers (single activation)
      - backprop, optimizer, batch modes, regularization
    """
    def __init__(
        self,
        input_dim,
        hidden_layers=1,
        hidden_size=16,
        hidden_sizes=None,          # <--- NEW

        activation="relu",
        leaky_slope=0.01,

        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        rho=0.95,

        batch_mode="minibatch",
        batch_size=32,
        shuffle=True,

        penalty="none",
        alpha=0.0,
        l1_ratio=0.5,

        n_iters=EPOCHS,
        max_grad_value=None,
        random_state=0,
        verbose=False
    ):
        self.input_dim = int(input_dim)

        # ---- NEW: allow per-layer widths ----
        if hidden_sizes is not None:
            hs = list(hidden_sizes)
            if len(hs) == 0:
                self.hidden_sizes = []
            else:
                if any(int(x) <= 0 for x in hs):
                    raise ValueError("All hidden_sizes must be > 0")
                self.hidden_sizes = [int(x) for x in hs]
            self.hidden_layers = len(self.hidden_sizes)
            self.hidden_size = self.hidden_sizes[-1] if self.hidden_layers > 0 else 0
        else:
            self.hidden_layers = int(hidden_layers)
            self.hidden_size = int(hidden_size)
            if self.hidden_layers < 0:
                raise ValueError("hidden_layers must be >= 0")
            if self.hidden_layers > 0 and self.hidden_size <= 0:
                raise ValueError("hidden_size must be > 0")
            self.hidden_sizes = [self.hidden_size] * self.hidden_layers
        # ------------------------------------

        self.activation_name = activation
        self.leaky_slope = leaky_slope

        self.optimizer_name = optimizer
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.rho = float(rho)

        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        self.penalty = penalty
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)

        self.n_iters = int(n_iters)
        self.max_grad_value = max_grad_value
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.batch_mode not in ("batch","stochastic","minibatch"):
            raise ValueError("batch_mode must be one of: batch, stochastic, minibatch")

        self._rng = np.random.default_rng(self.random_state)
        self._act = Activation(self.activation_name, leaky_slope=self.leaky_slope)

        self._regularizer = Regularizer(penalty=self.penalty, alpha=self.alpha, l1_ratio=self.l1_ratio)

        self._opt = Optimizer(
            method=self.optimizer_name,
            lr=self.learning_rate,
            momentum=self.momentum,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            rho=self.rho
        )

        self.W = []
        self.b = []
        self._init_trunk()

    def _init_trunk(self):
        self.W = []
        self.b = []

        in_dim = self.input_dim
        relu_like = self.activation_name.lower() in ("relu", "leaky_relu", "special_relu")

        for out_dim in self.hidden_sizes:
            scale = np.sqrt(2.0 / in_dim) if relu_like else np.sqrt(1.0 / in_dim)
            Wl = self._rng.normal(0.0, scale, size=(in_dim, out_dim))
            bl = np.zeros((1, out_dim))
            self.W.append(Wl)
            self.b.append(bl)
            in_dim = out_dim

    def _clip_grad(self, g):
        if self.max_grad_value is None:
            return g
        np.clip(g, -self.max_grad_value, self.max_grad_value, out=g)
        return g

    def _forward_trunk(self, X):
        a = X
        activations = [a]
        pre = []

        for l in range(self.hidden_layers):
            z = a @ self.W[l] + self.b[l]
            a = self._act.forward(z)
            pre.append(z)
            activations.append(a)

        h = activations[-1]
        return h, (activations, pre)

    def _backward_trunk(self, dh, cache, batch_scale):
        activations, pre = cache
        dW = [None] * self.hidden_layers
        db = [None] * self.hidden_layers

        da = dh
        for l in reversed(range(self.hidden_layers)):
            z = pre[l]
            a_prev = activations[l]
            dz = self._act.backward(z, da)

            dW_l = a_prev.T @ dz
            db_l = np.sum(dz, axis=0, keepdims=True)

            dW_l += self._regularizer.grad(self.W[l], n_scale=batch_scale)

            dW[l] = self._clip_grad(dW_l)
            db[l] = self._clip_grad(db_l)

            da = dz @ self.W[l].T

        return dW, db

    def _apply_updates_trunk(self, dW, db):
        for l in range(self.hidden_layers):
            self._opt.step(self.W[l], dW[l])
            self._opt.step(self.b[l], db[l])

# ============================================================
# 1) MLPRegressor
# ============================================================

class MLPRegressor(_BaseMLP):
    """
    Regression head: linear output (n,1)
    Loss: MSE
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # head params
        in_dim = self.hidden_sizes[-1] if self.hidden_layers > 0 else self.input_dim
        # init
        relu_like = self.activation_name.lower() in ("relu", "leaky_relu", "special_relu")
        scale = np.sqrt(2.0 / in_dim) if relu_like else np.sqrt(1.0 / in_dim)
        self.W_out = self._rng.normal(0.0, scale, size=(in_dim, 1))
        self.b_out = np.zeros((1, 1))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")

        for it in range(1, self.n_iters + 1):
            total_loss = 0.0
            nbatches = 0

            for Xb, (yb,) in _batch_iterator(
                X, [y],
                mode=self.batch_mode,
                batch_size=self.batch_size,
                rng=self._rng,
                shuffle=self.shuffle
            ):
                nb = Xb.shape[0]
                nbatches += 1

                # forward
                h, cache = self._forward_trunk(Xb)
                yhat = h @ self.W_out + self.b_out

                # loss
                mse = np.mean((yhat - yb) ** 2)
                reg_loss = self._regularizer.loss(self.W + [self.W_out], n_scale=nb)
                loss = mse + reg_loss
                total_loss += loss

                # backward head
                d_yhat = 2.0 * (yhat - yb) / nb  # (n,1)
                dW_out = h.T @ d_yhat + self._regularizer.grad(self.W_out, n_scale=nb)
                db_out = np.sum(d_yhat, axis=0, keepdims=True)

                dW_out = self._clip_grad(dW_out)
                db_out = self._clip_grad(db_out)

                # backward into trunk
                dh = d_yhat @ self.W_out.T
                dW, db = self._backward_trunk(dh, cache, batch_scale=nb)

                # update trunk
                self._apply_updates_trunk(dW, db)

                # update head
                self._opt.step(self.W_out, dW_out)
                self._opt.step(self.b_out, db_out)

            if self.verbose and (it % max(1, self.n_iters // 10) == 0):
                print(f"[MLPRegressor] iter={it}/{self.n_iters} loss={total_loss/max(1,nbatches):.6f}")

        return self

    def predict(self, X):
        X = np.asarray(X, float)
        h, _ = self._forward_trunk(X)
        yhat = h @ self.W_out + self.b_out
        return yhat  # (n,1)


# ============================================================
# 2) MLPClassifier (binary)
# ============================================================

class MLPClassifier(_BaseMLP):
    """
    Binary classification head: linear -> sigmoid
    Loss: BCE
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        in_dim = self.hidden_sizes[-1] if self.hidden_layers > 0 else self.input_dim
        relu_like = self.activation_name.lower() in ("relu", "leaky_relu", "special_relu")
        scale = np.sqrt(2.0 / in_dim) if relu_like else np.sqrt(1.0 / in_dim)
        self.W_out = self._rng.normal(0.0, scale, size=(in_dim, 1))
        self.b_out = np.zeros((1, 1))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")

        for it in range(1, self.n_iters + 1):
            total_loss = 0.0
            nbatches = 0

            for Xb, (yb,) in _batch_iterator(
                X, [y],
                mode=self.batch_mode,
                batch_size=self.batch_size,
                rng=self._rng,
                shuffle=self.shuffle
            ):
                nb = Xb.shape[0]
                nbatches += 1

                h, cache = self._forward_trunk(Xb)
                logits = h @ self.W_out + self.b_out
                probs = _safe_sigmoid(logits)

                eps = 1e-8
                bce = -np.mean(yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps))
                reg_loss = self._regularizer.loss(self.W + [self.W_out], n_scale=nb)
                loss = bce + reg_loss
                total_loss += loss

                # BCE + sigmoid => dL/dlogit = (p - y)/nb
                d_logit = (probs - yb) / nb

                dW_out = h.T @ d_logit + self._regularizer.grad(self.W_out, n_scale=nb)
                db_out = np.sum(d_logit, axis=0, keepdims=True)

                dW_out = self._clip_grad(dW_out)
                db_out = self._clip_grad(db_out)

                dh = d_logit @ self.W_out.T
                dW, db = self._backward_trunk(dh, cache, batch_scale=nb)

                self._apply_updates_trunk(dW, db)
                self._opt.step(self.W_out, dW_out)
                self._opt.step(self.b_out, db_out)

            if self.verbose and (it % max(1, self.n_iters // 10) == 0):
                print(f"[MLPClassifier] iter={it}/{self.n_iters} loss={total_loss/max(1,nbatches):.6f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        h, _ = self._forward_trunk(X)
        logits = h @ self.W_out + self.b_out
        probs = _safe_sigmoid(logits)
        return probs  # (n,1)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        labels = (probs >= float(threshold)).astype(int)
        return probs, labels

# ============================================================
# RBF "MLP" (1 hidden RBF layer) - GD-trained head
#   - centers fixed by kmeans
#   - head trained via your Optimizer + Regularizer
# ============================================================

class RBFMLPRegressor:
    """
    RBF hidden layer + linear output.
    Trains only output weights/bias with your optimizer variants.
    """
    def __init__(
        self,
        n_centers=20,
        gamma=None,                 # if None -> heuristic from centers
        center_iters=25,

        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        rho=0.95,

        batch_mode="minibatch",
        batch_size=32,
        shuffle=True,

        penalty="none",
        alpha=0.0,
        l1_ratio=0.5,

        n_iters=EPOCHS,
        max_grad_value=None,
        random_state=0,
        verbose=False
    ):
        self.n_centers = int(n_centers)
        self.gamma = gamma
        self.center_iters = int(center_iters)

        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        self.n_iters = int(n_iters)
        self.max_grad_value = max_grad_value
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self._rng = np.random.default_rng(self.random_state)

        self._opt = Optimizer(
            method=str(optimizer).lower(),
            lr=float(learning_rate),
            momentum=float(momentum),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            rho=float(rho)
        )
        self._reg = Regularizer(penalty=penalty, alpha=float(alpha), l1_ratio=float(l1_ratio))

        # learned
        self.centers_ = None
        self.gamma_ = None
        self.W_out = None   # (k,1)
        self.b_out = None   # (1,1)

    def _clip(self, g):
        if self.max_grad_value is None:
            return g
        np.clip(g, -self.max_grad_value, self.max_grad_value, out=g)
        return g

    def _init_head(self):
        k = self.n_centers
        scale = np.sqrt(1.0 / max(1, k))
        self.W_out = self._rng.normal(0.0, scale, size=(k, 1))
        self.b_out = np.zeros((1, 1))

    def _auto_gamma(self):
        # heuristic: gamma = 1/(2 * median(center_dist^2))
        C = self.centers_
        d2 = np.sum((C[:, None, :] - C[None, :, :])**2, axis=2)
        tri = d2[np.triu_indices(C.shape[0], k=1)]
        med = np.median(tri) if tri.size else 1.0
        med = 1.0 if med <= 0 else float(med)
        return 1.0 / (2.0 * med)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")

        # 1) centers
        self.centers_ = _kmeans_centers(X, self.n_centers, n_iters=self.center_iters, random_state=self.random_state)

        # 2) gamma
        self.gamma_ = float(self.gamma) if self.gamma is not None else float(self._auto_gamma())

        # 3) init head
        self._init_head()

        # precompute Phi for speed
        Phi_full = _rbf_design(X, self.centers_, self.gamma_)  # (n,k)

        for it in range(1, self.n_iters + 1):
            total = 0.0
            nbatches = 0

            for idx_X, (idx_y,) in _batch_iterator(
                np.arange(n), [y],
                mode=self.batch_mode,
                batch_size=self.batch_size,
                rng=self._rng,
                shuffle=self.shuffle
            ):
                # idx_X here is indices array; adapt:
                sel = idx_X.reshape(-1)
                Phi = Phi_full[sel]
                yb = idx_y

                nb = Phi.shape[0]
                nbatches += 1

                yhat = Phi @ self.W_out + self.b_out
                mse = np.mean((yhat - yb) ** 2)
                reg_loss = self._reg.loss([self.W_out], n_scale=nb)
                loss = mse + reg_loss
                total += loss

                d_yhat = 2.0 * (yhat - yb) / nb
                dW = Phi.T @ d_yhat + self._reg.grad(self.W_out, n_scale=nb)
                db = np.sum(d_yhat, axis=0, keepdims=True)

                dW = self._clip(dW)
                db = self._clip(db)

                self._opt.step(self.W_out, dW)
                self._opt.step(self.b_out, db)

            if self.verbose and (it % max(1, self.n_iters // 10) == 0):
                print(f"[RBFMLPRegressor] iter={it}/{self.n_iters} loss={total/max(1,nbatches):.6f}")

        return self

    def predict(self, X):
        X = np.asarray(X, float)
        Phi = _rbf_design(X, self.centers_, self.gamma_)
        return Phi @ self.W_out + self.b_out  # (n,1)


class RBFMLPClassifier:
    """
    RBF hidden layer + sigmoid output (binary).
    Trains only output weights/bias with your optimizer variants.
    """
    def __init__(
        self,
        n_centers=20,
        gamma=None,
        center_iters=25,

        optimizer="adam",
        learning_rate=0.01,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        rho=0.95,

        batch_mode="minibatch",
        batch_size=32,
        shuffle=True,

        penalty="none",
        alpha=0.0,
        l1_ratio=0.5,

        n_iters=EPOCHS,
        max_grad_value=None,
        random_state=0,
        verbose=False
    ):
        self.n_centers = int(n_centers)
        self.gamma = gamma
        self.center_iters = int(center_iters)

        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        self.n_iters = int(n_iters)
        self.max_grad_value = max_grad_value
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self._rng = np.random.default_rng(self.random_state)

        self._opt = Optimizer(
            method=str(optimizer).lower(),
            lr=float(learning_rate),
            momentum=float(momentum),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            rho=float(rho)
        )
        self._reg = Regularizer(penalty=penalty, alpha=float(alpha), l1_ratio=float(l1_ratio))

        self.centers_ = None
        self.gamma_ = None
        self.W_out = None
        self.b_out = None

    def _clip(self, g):
        if self.max_grad_value is None:
            return g
        np.clip(g, -self.max_grad_value, self.max_grad_value, out=g)
        return g

    def _init_head(self):
        k = self.n_centers
        scale = np.sqrt(1.0 / max(1, k))
        self.W_out = self._rng.normal(0.0, scale, size=(k, 1))
        self.b_out = np.zeros((1, 1))

    def _auto_gamma(self):
        C = self.centers_
        d2 = np.sum((C[:, None, :] - C[None, :, :])**2, axis=2)
        tri = d2[np.triu_indices(C.shape[0], k=1)]
        med = np.median(tri) if tri.size else 1.0
        med = 1.0 if med <= 0 else float(med)
        return 1.0 / (2.0 * med)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")

        self.centers_ = _kmeans_centers(X, self.n_centers, n_iters=self.center_iters, random_state=self.random_state)
        self.gamma_ = float(self.gamma) if self.gamma is not None else float(self._auto_gamma())
        self._init_head()

        Phi_full = _rbf_design(X, self.centers_, self.gamma_)  # (n,k)

        for it in range(1, self.n_iters + 1):
            total = 0.0
            nbatches = 0

            for idx_X, (yb,) in _batch_iterator(
                np.arange(n), [y],
                mode=self.batch_mode,
                batch_size=self.batch_size,
                rng=self._rng,
                shuffle=self.shuffle
            ):
                sel = idx_X.reshape(-1)
                Phi = Phi_full[sel]
                nb = Phi.shape[0]
                nbatches += 1

                logits = Phi @ self.W_out + self.b_out
                probs = _safe_sigmoid(logits)

                eps = 1e-8
                bce = -np.mean(yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps))
                reg_loss = self._reg.loss([self.W_out], n_scale=nb)
                loss = bce + reg_loss
                total += loss

                # BCE+sigmoid => dL/dlogit = (p - y)/nb
                dlog = (probs - yb) / nb
                dW = Phi.T @ dlog + self._reg.grad(self.W_out, n_scale=nb)
                db = np.sum(dlog, axis=0, keepdims=True)

                dW = self._clip(dW)
                db = self._clip(db)

                self._opt.step(self.W_out, dW)
                self._opt.step(self.b_out, db)

            if self.verbose and (it % max(1, self.n_iters // 10) == 0):
                print(f"[RBFMLPClassifier] iter={it}/{self.n_iters} loss={total/max(1,nbatches):.6f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Phi = _rbf_design(X, self.centers_, self.gamma_)
        logits = Phi @ self.W_out + self.b_out
        return _safe_sigmoid(logits)  # (n,1)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        yhat = (p >= float(threshold)).astype(int)
        return p, yhat
