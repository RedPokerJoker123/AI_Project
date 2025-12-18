import numpy as np
from itertools import combinations_with_replacement

EPOCHS = 100

class LinearRegression:
    """
    Full-feature Linear Regression (from scratch) with:

    Methods:
      - method="closed_form"
      - method="gd"
      - method="rbf"          (Kernel Ridge Regression with RBF kernel)

    Gradient Descent optimizers:
        "vanilla", "momentum", "nesterov",
        "adagrad", "rmsprop", "adadelta", "adam"

    Batch modes:
        "batch", "stochastic", "minibatch"

    Regularization:
        penalty ∈ {"none", "ridge", "lasso", "elasticnet"}
        alpha: overall regularization strength
        l1_ratio: only for elasticnet

    RBF options (method="rbf"):
        gamma: RBF kernel gamma
        alpha: kernel ridge regularization strength (same alpha param is used)

    Assumptions:
      - X passed to __init__ ALREADY includes an intercept column of 1s (FIRST column).
      - Training is done in standardized space (X features + Y), then theta is unscaled
        to ORIGINAL space so you can call predict() on raw X.

    Storage:
      - Learned parameters for linear methods stored ONLY in self._theta (ORIGINAL space).
      - For RBF method, parameters stored in:
            self._alpha_vec_ (scaled-space dual weights),
            self._X_train_scaled_ (scaled X including intercept),
        and self._theta stays None (by design).
    """

    # -------------------------
    # Init
    # -------------------------
    def __init__(
        self,
        X,
        Y,
        method="gd",              # "gd", "closed_form", "rbf"
        optimizer="vanilla",      # gd only
        batch_mode="batch",       # "batch", "stochastic", "minibatch"
        batch_size=32,            # minibatch only
        learning_rate=0.01,       # gd only
        epochs=EPOCHS,                 # gd epochs
        penalty="none",           # "none", "ridge", "lasso", "elasticnet"
        alpha=1.0,                # reg strength (linear GD/closed form), ALSO used as KRR strength for RBF
        l1_ratio=0.5,
        reg_lambda=None,          # closed-form ridge strength; defaults to alpha if penalty=="ridge"
        use_pinv=True,            # closed-form choice when no ridge
        beta_m=0.9,               # momentum/adam beta1
        beta_v=0.999,             # rmsprop/adam beta2
        rho=0.95,                 # adadelta decay
        eps=1e-8,
        random_state=0,
        shuffle=True,
        verbose=False,

        # RBF
        gamma=1.0
    ):
        # Data
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float).reshape(-1, 1)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.X.shape[1] < 1:
            raise ValueError("X must have at least 1 column (the intercept).")
        if not np.allclose(self.X[:, 0], 1.0):
            raise ValueError("X must include an intercept column of 1s as the FIRST column.")

        # Config
        self.method = str(method).lower()
        self.optimizer = str(optimizer).lower()
        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)

        self.penalty = "none" if penalty is None else str(penalty).lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)

        self.reg_lambda = reg_lambda
        self.use_pinv = bool(use_pinv)

        self.beta_m = float(beta_m)
        self.beta_v = float(beta_v)
        self.rho = float(rho)
        self.eps = float(eps)

        self.random_state = random_state
        self.shuffle = bool(shuffle)
        self.verbose = bool(verbose)

        # RBF
        self.gamma = float(gamma)

        # Learned params
        self._theta = None  # for linear methods only

        # RBF learned params (scaled space)
        self._alpha_vec_ = None
        self._X_train_scaled_ = None

        # Training history (scaled space)
        self.loss_history_ = []

        # Scaling stats
        self.X_mean_ = None  # (d,) excluding bias
        self.X_std_ = None   # (d,) excluding bias
        self.Y_mean_ = None  # scalar
        self.Y_std_ = None   # scalar

        self._validate_config()

    # -------------------------
    # Validation
    # -------------------------
    def _validate_config(self):
        if self.method not in ("gd", "closed_form", "rbf"):
            raise ValueError("method must be 'gd', 'closed_form', or 'rbf'.")

        opt_ok = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
        if self.method == "gd" and self.optimizer not in opt_ok:
            raise ValueError(f"optimizer must be one of {opt_ok}")

        mode_ok = ("batch", "stochastic", "minibatch")
        if self.method == "gd" and self.batch_mode not in mode_ok:
            raise ValueError(f"batch_mode must be one of {mode_ok}")

        pen_ok = ("none", "ridge", "lasso", "elasticnet")
        if self.penalty not in pen_ok:
            raise ValueError(f"penalty must be one of {pen_ok}")

        if self.penalty == "elasticnet" and not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1] for elasticnet.")

        if self.method == "gd" and self.epochs <= 0:
            raise ValueError("epochs must be > 0 for GD.")

        if self.method == "gd" and self.batch_mode == "minibatch" and self.batch_size <= 0:
            raise ValueError("batch_size must be > 0 for minibatch.")

        if self.method == "rbf" and self.gamma <= 0.0:
            raise ValueError("gamma must be > 0 for RBF method.")

        if self.method == "rbf" and self.alpha <= 0.0:
            raise ValueError("For RBF kernel ridge, alpha must be > 0 (regularization strength).")

    # =========================================================
    # Scaling helpers
    # =========================================================
    def _standardize_X(self, X_with_bias):
        """
        Standardize X excluding intercept column 0.
        Returns: (X_scaled, mean(d,), std(d,))
        """
        X_with_bias = np.asarray(X_with_bias, dtype=float)
        if not np.allclose(X_with_bias[:, 0], 1.0):
            raise ValueError("X must have intercept column of 1s in column 0.")

        X_feat = X_with_bias[:, 1:]
        mean = X_feat.mean(axis=0)
        std = X_feat.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)

        X_scaled = X_with_bias.copy()
        X_scaled[:, 1:] = (X_feat - mean) / std
        return X_scaled, mean, std

    def _standardize_Y(self, Y):
        Y = np.asarray(Y, dtype=float).reshape(-1, 1)
        mean = float(Y.mean())
        std = float(Y.std())
        if std == 0.0:
            std = 1.0
        return (Y - mean) / std, mean, std

    def _unscale_theta(self, theta_scaled, X_mean, X_std, Y_mean, Y_std):
        theta_scaled = np.asarray(theta_scaled, dtype=float).reshape(-1)
        X_mean = np.asarray(X_mean, dtype=float).reshape(-1)
        X_std = np.asarray(X_std, dtype=float).reshape(-1)

        d = X_mean.size
        if theta_scaled.size != d + 1:
            raise ValueError("theta_scaled must have length d+1 (bias + weights).")

        beta = np.zeros_like(theta_scaled)
        beta[1:] = (Y_std / X_std) * theta_scaled[1:]
        beta[0] = Y_mean + (Y_std * theta_scaled[0]) - np.sum(beta[1:] * X_mean)

        return beta.reshape(-1, 1)

    # =========================================================
    # Core utilities
    # =========================================================
    def _mse_loss_half(self, X, Y, theta):
        n = X.shape[0]
        diff = (X @ theta) - Y
        return float((diff.T @ diff) / (2.0 * n))

    def _soft_threshold(self, w, lam):
        return np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)

    def _iterate_batches(self, X, Y, batch_size, shuffle=True, seed=None):
        n = X.shape[0]
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]
            yield X[b], Y[b]

    # =========================================================
    # RBF Kernel (INSTANCE METHOD, not static)
    # =========================================================
    def rbf_kernel(self, X, Y=None):
        """
        RBF kernel K(x,y)=exp(-gamma||x-y||^2)
        IMPORTANT: ignores intercept column (col 0) in distance computation.
        """
        X = np.asarray(X, float)
        Xf = X[:, 1:]
        if Y is None:
            Yf = Xf
        else:
            Y = np.asarray(Y, float)
            Yf = Y[:, 1:]

        X_norm = np.sum(Xf ** 2, axis=1)[:, None]
        Y_norm = np.sum(Yf ** 2, axis=1)[None, :]
        sq = X_norm + Y_norm - 2.0 * (Xf @ Yf.T)
        sq = np.maximum(sq, 0.0)
        return np.exp(-self.gamma * sq)

    # =========================================================
    # Closed-form (linear)
    # =========================================================
    def _fit_closed_form(self, Xs, Ys):
        n_features_b = Xs.shape[1]

        lam = 0.0
        if self.reg_lambda is not None:
            lam = float(self.reg_lambda)
        elif self.penalty == "ridge":
            lam = float(self.alpha)

        if lam > 0.0:
            I = np.eye(n_features_b)
            I[0, 0] = 0.0
            A = Xs.T @ Xs + (Xs.shape[0] * lam) * I
            b = Xs.T @ Ys
            theta = np.linalg.solve(A, b)
        else:
            if self.use_pinv:
                theta = np.linalg.pinv(Xs) @ Ys
            else:
                theta = np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ Ys)

        return theta

    # =========================================================
    # GD (linear)
    # =========================================================
    def _compute_gradient(self, Xb, Yb, theta):
        n = Xb.shape[0]
        diff = (Xb @ theta) - Yb
        grad = (Xb.T @ diff) / n

        if self.penalty in ("ridge", "elasticnet") and self.alpha > 0.0:
            lam2 = self.alpha if self.penalty == "ridge" else self.alpha * (1.0 - self.l1_ratio)
            reg = np.zeros_like(theta)
            reg[1:, :] = lam2 * theta[1:, :]
            grad = grad + reg

        return grad

    def _apply_prox_l1(self, theta):
        if self.penalty in ("lasso", "elasticnet") and self.alpha > 0.0:
            lam1 = self.alpha if self.penalty == "lasso" else self.alpha * self.l1_ratio
            theta[1:, :] = self._soft_threshold(theta[1:, :], self.learning_rate * lam1)
        return theta

    def _fit_gd(self, Xs, Ys):
        n_samples, n_features_b = Xs.shape

        # batch size
        if self.batch_mode == "batch":
            bs = n_samples
        elif self.batch_mode == "stochastic":
            bs = 1
        else:
            bs = min(self.batch_size, n_samples)

        rng = np.random.default_rng(self.random_state)
        theta = rng.normal(0.0, 0.01, size=(n_features_b, 1))

        # optimizer state
        v_mom = np.zeros_like(theta)   # velocity for momentum / nesterov
        G = np.zeros_like(theta)       # adagrad accumulator
        v2 = np.zeros_like(theta)      # rmsprop/adam second moment or adadelta E[Δθ^2]
        Eg = np.zeros_like(theta)      # adadelta E[g^2]
        m_adam = np.zeros_like(theta)  # adam first moment
        v_adam = np.zeros_like(theta)  # adam second moment
        t = 0

        self.loss_history_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            seed = None if (not self.shuffle or self.random_state is None) else (self.random_state + epoch)

            for Xb, Yb in self._iterate_batches(Xs, Ys, batch_size=bs, shuffle=self.shuffle, seed=seed):

                if self.optimizer == "vanilla":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    theta = theta - self.learning_rate * grad

                elif self.optimizer == "momentum":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    v_mom = self.beta_m * v_mom + grad
                    theta = theta - self.learning_rate * v_mom

                elif self.optimizer == "nesterov":
                    # lookahead point
                    theta_look = theta - self.learning_rate * self.beta_m * v_mom
                    grad = self._compute_gradient(Xb, Yb, theta_look)
                    v_mom = self.beta_m * v_mom + grad
                    theta = theta - self.learning_rate * v_mom

                elif self.optimizer == "adagrad":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    G = G + grad * grad
                    theta = theta - self.learning_rate * grad / (np.sqrt(G) + self.eps)

                elif self.optimizer == "rmsprop":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    v2 = self.beta_v * v2 + (1.0 - self.beta_v) * (grad * grad)
                    theta = theta - self.learning_rate * grad / (np.sqrt(v2) + self.eps)

                elif self.optimizer == "adadelta":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    Eg = self.rho * Eg + (1.0 - self.rho) * (grad * grad)
                    # RMS(Δθ) stored in v2, RMS(g) stored in Eg
                    update = - (np.sqrt(v2 + self.eps) / np.sqrt(Eg + self.eps)) * grad
                    v2 = self.rho * v2 + (1.0 - self.rho) * (update * update)
                    theta = theta + update  # NOTE: adadelta already sets magnitude; no lr needed (optional)

                elif self.optimizer == "adam":
                    grad = self._compute_gradient(Xb, Yb, theta)
                    t += 1
                    m_adam = self.beta_m * m_adam + (1.0 - self.beta_m) * grad
                    v_adam = self.beta_v * v_adam + (1.0 - self.beta_v) * (grad * grad)

                    m_hat = m_adam / (1.0 - self.beta_m ** t)
                    v_hat = v_adam / (1.0 - self.beta_v ** t)
                    theta = theta - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")

                theta = self._apply_prox_l1(theta)
                epoch_loss += self._mse_loss_half(Xb, Yb, theta)

            # average over number of batches
            num_batches = int(np.ceil(n_samples / bs))
            epoch_loss /= max(1, num_batches)
            self.loss_history_.append(epoch_loss)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"[GD] epoch {epoch}/{self.epochs} - loss(scaled): {epoch_loss:.6f}")

        return theta

    # =========================================================
    # RBF Kernel Ridge (scaled space)
    # =========================================================
    def _fit_rbf(self, Xs, Ys):
        """
        Kernel ridge regression in scaled space:
            alpha_vec = (K + alpha*I)^(-1) Ys
        """
        K = self.rbf_kernel(Xs, None)
        n = K.shape[0]
        A = K + self.alpha * np.eye(n)
        self._alpha_vec_ = np.linalg.solve(A, Ys)
        self._X_train_scaled_ = Xs.copy()
        self.loss_history_ = [float(np.mean((K @ self._alpha_vec_ - Ys) ** 2))]

    # =========================================================
    # Public API
    # =========================================================
    def fit(self):
        """
        Fit according to method. Stores:
          - linear: self._theta (original space)
          - rbf:    self._alpha_vec_ + self._X_train_scaled_ (scaled space)
        """
        self._validate_config()

        Xs, X_mean, X_std = self._standardize_X(self.X)
        Ys, Y_mean, Y_std = self._standardize_Y(self.Y)

        self.X_mean_ = X_mean
        self.X_std_ = X_std
        self.Y_mean_ = Y_mean
        self.Y_std_ = Y_std

        # clear
        self._theta = None
        self._alpha_vec_ = None
        self._X_train_scaled_ = None

        if self.method == "closed_form":
            theta_scaled = self._fit_closed_form(Xs, Ys)
            self.loss_history_ = [self._mse_loss_half(Xs, Ys, theta_scaled)]
            self._theta = self._unscale_theta(theta_scaled, X_mean, X_std, Y_mean, Y_std)

        elif self.method == "gd":
            theta_scaled = self._fit_gd(Xs, Ys)
            self._theta = self._unscale_theta(theta_scaled, X_mean, X_std, Y_mean, Y_std)

        else:  # "rbf"
            self._fit_rbf(Xs, Ys)

        return self

    def predict(self, X_new):
        """
        Predict on raw X_new (must include intercept col).
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(f"X_new must have {self.X.shape[1]} columns (including bias).")
        if not np.allclose(X_new[:, 0], 1.0):
            raise ValueError("X_new must include intercept column of 1s as FIRST column.")

        if self.method in ("gd", "closed_form"):
            if self._theta is None:
                raise ValueError("Model not fitted yet.")
            return (X_new @ self._theta).ravel()

        # RBF:
        if self._alpha_vec_ is None or self._X_train_scaled_ is None:
            raise ValueError("RBF model not fitted yet.")
        Xs_new = X_new.copy()
        Xs_new[:, 1:] = (Xs_new[:, 1:] - self.X_mean_) / self.X_std_

        Kt = self.rbf_kernel(Xs_new, self._X_train_scaled_)
        y_scaled = Kt @ self._alpha_vec_
        y = y_scaled.reshape(-1) * self.Y_std_ + self.Y_mean_
        return y

class LogisticRegression:
    """
    Full-feature Logistic Regression (from scratch) with:

    Methods:
        - method="newton" : Newton-Raphson / IRLS (full-batch), supports penalty ∈ {"none","ridge"}
        - method="gd"     : Gradient-based training with optimizers:
            "vanilla", "momentum", "nesterov",
            "adagrad", "rmsprop", "adadelta", "adam"
          supports penalty ∈ {"none","ridge","lasso","elasticnet"}
        - method="rbf"    : RBF Kernel Ridge "classification head" (least-squares on ±1, then sigmoid)

    Assumptions:
      - X passed to __init__ ALREADY includes an intercept column of 1s (FIRST column).
      - Standardization is applied to feature columns 1..d (intercept stays 1).
      - Learned parameters are stored ONLY in self._w for linear methods.
      - For RBF method, store:
            self._alpha_vec_ (dual weights in scaled space),
            self._X_train_scaled_ (support points in scaled space)
        and self._w stays None (by design).
    """

    def __init__(
        self,
        X,
        y,
        method="gd",              # "gd", "newton", "rbf"
        penalty="none",           # "none","ridge","lasso","elasticnet"
        alpha=1.0,
        l1_ratio=0.5,

        # GD options
        optimizer="vanilla",
        learning_rate=0.01,
        epochs=EPOCHS,
        batch_mode="batch",
        batch_size=32,

        # Newton options
        newton_iters=25,
        newton_tol=1e-6,

        # Optimizer hyperparams
        beta_m=0.9,
        beta_v=0.999,
        rho=0.95,
        eps=1e-8,

        random_state=0,
        shuffle=True,
        verbose=False,

        # RBF
        gamma=1.0
    ):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int).reshape(-1, 1)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features_with_bias).")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if self.X.shape[1] < 1:
            raise ValueError("X must have at least 1 column (the intercept).")
        if not np.allclose(self.X[:, 0], 1.0):
            raise ValueError("X must include an intercept column of 1s as the FIRST column.")
        if not np.all((self.y == 0) | (self.y == 1)):
            raise ValueError("y must contain only 0/1 labels.")

        self.method = str(method).lower()
        self.penalty = "none" if penalty is None else str(penalty).lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)

        self.optimizer = str(optimizer).lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)

        self.newton_iters = int(newton_iters)
        self.newton_tol = float(newton_tol)

        self.beta_m = float(beta_m)
        self.beta_v = float(beta_v)
        self.rho = float(rho)
        self.eps = float(eps)

        self.random_state = random_state
        self.shuffle = bool(shuffle)
        self.verbose = bool(verbose)

        # RBF
        self.gamma = float(gamma)

        # Learned weights (ONLY) for linear logistic in ORIGINAL scale
        self._w = None

        # RBF learned params (scaled space)
        self._alpha_vec_ = None
        self._X_train_scaled_ = None

        # scale stats (for standardizing cols 1:)
        self.x_mean_ = None
        self.x_std_ = None

        # history (scaled space)
        self.nll_history_ = []

        self._validate_config()

    def _validate_config(self):
        if self.method not in ("gd", "newton", "rbf"):
            raise ValueError("method must be 'gd', 'newton', or 'rbf'.")

        pen_ok = ("none", "ridge", "lasso", "elasticnet")
        if self.penalty not in pen_ok:
            raise ValueError(f"penalty must be one of {pen_ok}")

        if self.penalty == "elasticnet" and not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0,1] for elasticnet.")

        if self.method == "newton" and self.penalty not in ("none", "ridge"):
            raise ValueError("Newton/IRLS supports only penalty='none' or 'ridge'.")

        if self.method == "gd":
            opt_ok = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
            if self.optimizer not in opt_ok:
                raise ValueError(f"optimizer must be one of {opt_ok}")

            mode_ok = ("batch", "stochastic", "minibatch")
            if self.batch_mode not in mode_ok:
                raise ValueError(f"batch_mode must be one of {mode_ok}")

            if self.epochs <= 0:
                raise ValueError("epochs must be > 0 for GD.")
            if self.batch_mode == "minibatch" and self.batch_size <= 0:
                raise ValueError("batch_size must be > 0 for minibatch.")

        if self.method == "newton":
            if self.newton_iters <= 0:
                raise ValueError("newton_iters must be > 0.")
            if self.newton_tol <= 0:
                raise ValueError("newton_tol must be > 0.")

        if self.method == "rbf" and self.gamma <= 0.0:
            raise ValueError("gamma must be > 0 for RBF method.")
        if self.method == "rbf" and self.alpha <= 0.0:
            raise ValueError("For RBF kernel ridge classification, alpha must be > 0.")

    # =========================================================
    # Standardize X (keep intercept = 1, standardize cols 1:)
    # =========================================================
    def _standardize_X_with_bias(self, Xb):
        Xb = np.asarray(Xb, dtype=float)
        if not np.allclose(Xb[:, 0], 1.0):
            raise ValueError("X must include intercept column of 1s as the FIRST column.")

        X_feat = Xb[:, 1:]
        mean = X_feat.mean(axis=0)
        std = X_feat.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)

        Xs = Xb.copy()
        Xs[:, 1:] = (X_feat - mean) / std
        return Xs, mean, std

    def _sigmoid(self, z):
        z = np.asarray(z, dtype=float)
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _unscale_logistic_weights(self, w_scaled, x_mean, x_std):
        """
        Scaled model: z = a0 + sum a_j * ((X_j - mu_j)/sigma_j)
        Original:     z = w0 + sum w_j * X_j
        """
        w_scaled = np.asarray(w_scaled, dtype=float).reshape(-1, 1)
        if w_scaled.shape[0] <= 1:
            return w_scaled.copy()

        a0 = float(w_scaled[0, 0])
        a = w_scaled[1:, 0]

        mu = np.asarray(x_mean, dtype=float).reshape(-1)
        sigma = np.asarray(x_std, dtype=float).reshape(-1)
        sigma = np.where(sigma == 0.0, 1.0, sigma)

        w_nonbias = a / sigma
        correction = np.sum(a * mu / sigma)
        w0 = a0 - correction

        w_unscaled = np.empty_like(w_scaled)
        w_unscaled[0, 0] = w0
        w_unscaled[1:, 0] = w_nonbias
        return w_unscaled

    # =========================================================
    # Regularization mapping
    # =========================================================
    def _lambdas(self):
        if self.penalty == "none":
            return 0.0, 0.0
        if self.penalty == "ridge":
            return 0.0, self.alpha
        if self.penalty == "lasso":
            return self.alpha, 0.0
        return self.alpha * self.l1_ratio, self.alpha * (1.0 - self.l1_ratio)

    # =========================================================
    # Loss / grad / Hessian (scaled space)
    # =========================================================
    def _nll(self, Xb, y, w, l2_lambda=0.0, l1_lambda=0.0):
        y = np.asarray(y).reshape(-1, 1)
        z = Xb @ w
        p = self._sigmoid(z)

        eps = 1e-12
        n = Xb.shape[0]

        # mean cross-entropy
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) / n

        if l2_lambda > 0.0:
            nll += 0.5 * l2_lambda * float(np.sum(w[1:] ** 2))
        if l1_lambda > 0.0:
            nll += l1_lambda * float(np.sum(np.abs(w[1:])))

        return float(nll)

    def _grad(self, Xb, y, w, l2_lambda=0.0):
        y = np.asarray(y).reshape(-1, 1)
        p = self._sigmoid(Xb @ w)
        n = Xb.shape[0]

        # mean gradient of CE
        g = (Xb.T @ (p - y)) / n

        if l2_lambda > 0.0:
            reg = np.zeros_like(w)
            reg[1:, :] = l2_lambda * w[1:, :]
            g = g + reg

        return g

    def _hessian(self, Xb, w, l2_lambda=0.0):
        p = self._sigmoid(Xb @ w)
        n = Xb.shape[0]

        r = (p * (1 - p)).ravel()
        Xw = Xb * r[:, None]
        H = (Xb.T @ Xw) / n

        if l2_lambda > 0.0:
            reg = np.eye(Xb.shape[1])
            reg[0, 0] = 0.0
            H = H + l2_lambda * reg

        return H

    # =========================================================
    # Prox for L1
    # =========================================================
    def _soft_threshold_vec(self, v, lam):
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

    def _apply_prox_l1(self, w, l1_lambda):
        if l1_lambda > 0.0:
            w[1:, 0] = self._soft_threshold_vec(w[1:, 0], self.learning_rate * l1_lambda)
        return w

    # =========================================================
    # Batch iterator
    # =========================================================
    def _iterate_batches(self, Xb, y, batch_size, shuffle=True, seed=None):
        n = Xb.shape[0]
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]
            yield Xb[b], y[b]

    # =========================================================
    # RBF Kernel (INSTANCE METHOD, not static)
    # =========================================================
    def rbf_kernel(self, X, Y=None):
        """
        RBF kernel K(x,y)=exp(-gamma||x-y||^2)
        Ignores intercept col in distance.
        """
        X = np.asarray(X, float)
        Xf = X[:, 1:]
        if Y is None:
            Yf = Xf
        else:
            Y = np.asarray(Y, float)
            Yf = Y[:, 1:]

        X_norm = np.sum(Xf ** 2, axis=1)[:, None]
        Y_norm = np.sum(Yf ** 2, axis=1)[None, :]
        sq = X_norm + Y_norm - 2.0 * (Xf @ Yf.T)
        sq = np.maximum(sq, 0.0)
        return np.exp(-self.gamma * sq)

    # =========================================================
    # Fit: Newton / IRLS (scaled space)
    # =========================================================
    def _fit_newton_scaled(self, Xb_s, y, l2_lambda):
        n_features_b = Xb_s.shape[1]
        w = np.zeros((n_features_b, 1))
        self.nll_history_ = []

        for it in range(1, self.newton_iters + 1):
            nll = self._nll(Xb_s, y, w, l2_lambda=l2_lambda, l1_lambda=0.0)
            self.nll_history_.append(nll)

            g = self._grad(Xb_s, y, w, l2_lambda=l2_lambda)
            H = self._hessian(Xb_s, w, l2_lambda=l2_lambda)

            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H) @ g

            w_new = w - step
            step_norm = float(np.linalg.norm(step))

            if self.verbose:
                print(f"[NEWTON] iter {it}/{self.newton_iters} - NLL(scaled): {nll:.6f}, step_norm: {step_norm:.3e}")

            w = w_new
            if step_norm < self.newton_tol:
                if self.verbose:
                    print(f"[NEWTON] converged at iter {it}")
                break

        return w

    # =========================================================
    # Fit: GD (scaled space)
    # =========================================================
    def _fit_gd_scaled(self, Xb_s, y, l1_lambda, l2_lambda):
        n_samples, n_features_b = Xb_s.shape
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(0.0, 0.01, size=(n_features_b, 1))

        # batch size
        if self.batch_mode == "batch":
            bs = n_samples
        elif self.batch_mode == "stochastic":
            bs = 1
        else:
            bs = min(self.batch_size, n_samples)

        # optimizer state
        v_mom = np.zeros_like(w)      # momentum velocity
        G = np.zeros_like(w)          # adagrad accumulator
        v2 = np.zeros_like(w)         # rmsprop/adam second moment or adadelta E[Δw^2]
        Eg = np.zeros_like(w)         # adadelta E[g^2]
        m_adam = np.zeros_like(w)     # adam first moment
        v_adam = np.zeros_like(w)     # adam second moment
        t = 0

        self.nll_history_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_nll = 0.0
            seed = None if (not self.shuffle or self.random_state is None) else (self.random_state + epoch)

            # IMPORTANT: iterate generator directly (no list())
            for Xb, yb in self._iterate_batches(Xb_s, y, batch_size=bs, shuffle=self.shuffle, seed=seed):

                if self.optimizer == "vanilla":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    w = w - self.learning_rate * g

                elif self.optimizer == "momentum":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    v_mom = self.beta_m * v_mom + g
                    w = w - self.learning_rate * v_mom

                elif self.optimizer == "nesterov":
                    # lookahead with velocity (same style as your LinearRegression)
                    w_look = w - self.learning_rate * self.beta_m * v_mom
                    g = self._grad(Xb, yb, w_look, l2_lambda=l2_lambda)
                    v_mom = self.beta_m * v_mom + g
                    w = w - self.learning_rate * v_mom

                elif self.optimizer == "adagrad":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    G = G + g * g
                    w = w - self.learning_rate * g / (np.sqrt(G) + self.eps)

                elif self.optimizer == "rmsprop":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    v2 = self.beta_v * v2 + (1.0 - self.beta_v) * (g * g)
                    w = w - self.learning_rate * g / (np.sqrt(v2) + self.eps)

                elif self.optimizer == "adadelta":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    Eg = self.rho * Eg + (1.0 - self.rho) * (g * g)
                    update = - (np.sqrt(v2 + self.eps) / np.sqrt(Eg + self.eps)) * g
                    v2 = self.rho * v2 + (1.0 - self.rho) * (update * update)
                    w = w + update  # NOTE: no learning_rate multiplier (consistent with your LR)

                elif self.optimizer == "adam":
                    g = self._grad(Xb, yb, w, l2_lambda=l2_lambda)
                    t += 1
                    m_adam = self.beta_m * m_adam + (1.0 - self.beta_m) * g
                    v_adam = self.beta_v * v_adam + (1.0 - self.beta_v) * (g * g)

                    m_hat = m_adam / (1.0 - self.beta_m ** t)
                    v_hat = v_adam / (1.0 - self.beta_v ** t)
                    w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")

                # prox for L1 (if any)
                w = self._apply_prox_l1(w, l1_lambda)

                # tracking (data-fit nll + reg, as your _nll defines)
                epoch_nll += self._nll(Xb, yb, w, l2_lambda=l2_lambda, l1_lambda=l1_lambda)

            # average over batches
            num_batches = int(np.ceil(n_samples / bs))
            epoch_nll /= max(1, num_batches)
            self.nll_history_.append(epoch_nll)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"[GD-{self.optimizer}] epoch {epoch}/{self.epochs} - NLL(scaled): {epoch_nll:.6f}")

        return w

    # =========================================================
    # Fit: RBF Kernel Ridge classification head (scaled space)
    # =========================================================
    def _fit_rbf(self, Xs, y):
        """
        Least-squares on signed labels (+1/-1) with kernel ridge:
            alpha_vec = (K + alpha*I)^(-1) y_signed
        Predict: sigmoid(K_test @ alpha_vec)
        """
        y = np.asarray(y, int).reshape(-1, 1)
        y_signed = 2.0 * y - 1.0

        K = self.rbf_kernel(Xs, None)
        n = K.shape[0]
        A = K + self.alpha * np.eye(n)
        self._alpha_vec_ = np.linalg.solve(A, y_signed)
        self._X_train_scaled_ = Xs.copy()
        self.nll_history_ = []

    # =========================================================
    # Public API
    # =========================================================
    def fit(self):
        """
        Fits and stores:
          - linear methods: self._w (ORIGINAL X scale)
          - rbf:           self._alpha_vec_, self._X_train_scaled_ (SCALED space)
        """
        self._validate_config()

        Xs, x_mean, x_std = self._standardize_X_with_bias(self.X)
        self.x_mean_ = x_mean
        self.x_std_ = x_std

        l1_lambda, l2_lambda = self._lambdas()

        # clear
        self._w = None
        self._alpha_vec_ = None
        self._X_train_scaled_ = None

        if self.method == "rbf":
            self._fit_rbf(Xs, self.y)
            return self

        if self.method == "newton":
            w_scaled = self._fit_newton_scaled(Xs, self.y, l2_lambda=l2_lambda)
        else:
            w_scaled = self._fit_gd_scaled(Xs, self.y, l1_lambda=l1_lambda, l2_lambda=l2_lambda)

        # Unscale to ORIGINAL feature space
        self._w = self._unscale_logistic_weights(w_scaled, self.x_mean_, self.x_std_)
        return self

    def predict_proba(self, X_new):
        """
        Returns probabilities P(y=1|x).
        X_new must include intercept column of 1s in col 0.
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(f"X_new must have {self.X.shape[1]} columns (including bias).")
        if not np.allclose(X_new[:, 0], 1.0):
            raise ValueError("X_new must include intercept column of 1s as FIRST column.")

        if self.method == "rbf":
            if self._alpha_vec_ is None or self._X_train_scaled_ is None:
                raise ValueError("RBF model not fitted yet.")

            Xs_new = X_new.copy()
            Xs_new[:, 1:] = (Xs_new[:, 1:] - self.x_mean_) / self.x_std_

            Kt = self.rbf_kernel(Xs_new, self._X_train_scaled_)
            score = (Kt @ self._alpha_vec_).ravel()
            return self._sigmoid(score)

        # linear logistic:
        if self._w is None:
            raise ValueError("Model not fitted yet.")
        return self._sigmoid(X_new @ self._w).ravel()

    def predict(self, X_new, threshold=0.5):
        p = self.predict_proba(X_new)
        return (p >= threshold).astype(int)

class PolynomialRegression:
    """
    Full-feature Polynomial Regression (from scratch) with:

    Polynomial feature expansion (with interactions)
    Internal standardization of (Phi, y), then unscales final weights
    Methods:
        - "closed_form" : OLS (no penalty) + Ridge (L2)
        - "cd"          : Lasso / ElasticNet via coordinate descent
        - "gd"          : Gradient descent optimizers (L2 in objective)
        - "rbf"         : RBF Kernel Ridge Regression (Gaussian kernel)

    Optimizers (GD):
        "vanilla", "momentum", "nesterov",
        "adagrad", "rmsprop", "adadelta", "adam"

    Batch modes (GD):
        "batch", "stochastic", "minibatch"

    Penalties:
        penalty ∈ {"none", "ridge", "lasso", "elasticnet"}
        alpha: overall reg strength
        l1_ratio: only for elasticnet

    RBF options (method="rbf"):
        gamma: RBF kernel gamma (must be > 0)
        alpha: kernel ridge regularization strength (must be > 0)

    Assumption (per your instruction):
      - X passed to __init__ ALREADY includes an intercept column of 1s as FIRST column.
      - Polynomial features are built from REAL features only (excluding intercept col 0).
      - Predict expects X_new also includes intercept col 0.

    Notes:
      - For non-RBF methods, learned weights stored ONLY in self._w (original polynomial basis).
      - For RBF method, learned dual params stored in:
            self._alpha_vec_ (scaled-space dual weights),
            self._Z_train_scaled_ (scaled expanded features, with intercept col),
        and self._w stays None.
    """

    # -------------------------
    # Init
    # -------------------------
    def __init__(
        self,
        X,
        y,
        degree=2,
        method="closed_form",     # "closed_form", "cd", "gd", "rbf"
        penalty="none",           # "none", "ridge", "lasso", "elasticnet" (rbf effectively uses ridge via alpha)
        alpha=1.0,
        l1_ratio=0.5,

        # GD options
        optimizer="vanilla",
        batch_mode="batch",
        batch_size=32,
        learning_rate=0.01,
        epochs=EPOCHS,                 # (init attribute)

        # Closed-form options
        use_pinv=True,

        # Optimizer hyperparams
        beta_m=0.9,
        beta_v=0.999,
        rho=0.95,
        eps=1e-8,

        # CD options
        cd_max_iters=5000,
        cd_tol=1e-4,

        random_state=0,
        shuffle=True,
        verbose=False,

        # RBF options
        gamma=1.0
    ):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1, 1)

        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if degree < 1:
            raise ValueError("degree must be >= 1")

        # IMPORTANT assumption: X has intercept in col 0
        if self.X.shape[1] < 1:
            raise ValueError("X must have at least 1 column (the intercept).")
        if not np.allclose(self.X[:, 0], 1.0):
            raise ValueError("X must include an intercept column of 1s as the FIRST column.")

        # config
        self.degree = int(degree)
        self.method = str(method).lower()
        self.penalty = "none" if penalty is None else str(penalty).lower()
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)

        # gd
        self.optimizer = str(optimizer).lower()
        self.batch_mode = str(batch_mode).lower()
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate) * pow(10,-(3 * degree))
        
        self.epochs = int(epochs)

        # closed-form
        self.use_pinv = bool(use_pinv)

        # optimizer hyperparams
        self.beta_m = float(beta_m)
        self.beta_v = float(beta_v)
        self.rho = float(rho)
        self.eps = float(eps)

        # cd
        self.cd_max_iters = int(cd_max_iters)
        self.cd_tol = float(cd_tol)

        self.random_state = random_state
        self.shuffle = bool(shuffle)
        self.verbose = bool(verbose)

        # rbf
        self.gamma = float(gamma)

        # learned state (polynomial basis path)
        self.feature_powers_ = None     # list of tuples describing polynomial terms (on real features only)
        self.phi_mean_ = None           # (1, m)
        self.phi_std_ = None            # (1, m)
        self.y_mean_ = None             # scalar
        self.y_std_ = None              # scalar

        self._w = None                  # (m+1,1) in ORIGINAL polynomial basis (non-rbf methods)
        self.loss_history_ = []         # GD loss in standardized space

        # learned state (rbf path)
        self._alpha_vec_ = None         # (n,1) dual weights in scaled y space
        self._Z_train_scaled_ = None    # (n, 1+m) scaled expanded features with intercept
        self._y_mean_rbf_ = None        # scalar (mean of y for rbf scaling)
        self._y_std_rbf_ = None         # scalar (std  of y for rbf scaling)

        self._validate_config()

    # -------------------------
    # Validation
    # -------------------------
    def _validate_config(self):
        if self.method not in ("closed_form", "cd", "gd", "rbf"):
            raise ValueError("method must be one of: 'closed_form', 'cd', 'gd', 'rbf'")

        pen_ok = ("none", "ridge", "lasso", "elasticnet")
        if self.penalty not in pen_ok:
            raise ValueError(f"penalty must be one of {pen_ok}")

        if self.penalty == "elasticnet" and not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0,1] for elasticnet")

        if self.method == "closed_form" and self.penalty not in ("none", "ridge"):
            raise ValueError("closed_form supports only penalty='none' or 'ridge'")

        if self.method == "cd" and self.penalty not in ("lasso", "elasticnet"):
            raise ValueError("cd method is intended for penalty='lasso' or 'elasticnet'")

        opt_ok = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
        mode_ok = ("batch", "stochastic", "minibatch")
        if self.method == "gd":
            if self.optimizer not in opt_ok:
                raise ValueError(f"optimizer must be one of {opt_ok}")
            if self.batch_mode not in mode_ok:
                raise ValueError(f"batch_mode must be one of {mode_ok}")
            if self.epochs <= 0:
                raise ValueError("epochs must be > 0 for GD")
            if self.batch_mode == "minibatch" and self.batch_size <= 0:
                raise ValueError("batch_size must be > 0 for minibatch")

        if self.method == "rbf":
            if self.gamma <= 0.0:
                raise ValueError("gamma must be > 0 for RBF.")
            if self.alpha <= 0.0:
                raise ValueError("For RBF kernel ridge, alpha must be > 0 (regularization strength).")

    # =========================================================
    # Polynomial feature helpers
    # =========================================================
    def _make_polynomial_features(self, X):
        """
        Build polynomial design matrix Phi (NO bias column).
        IMPORTANT: expects X includes intercept in col 0, but we EXCLUDE it from polynomial construction.
        Returns (Phi, feature_powers) where powers refer to indices in the REAL feature block (excluding bias).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not np.allclose(X[:, 0], 1.0):
            raise ValueError("X must include intercept column of 1s as the FIRST column.")

        Xf = X[:, 1:]  # exclude intercept
        n_samples, n_features = Xf.shape

        cols = []
        feature_powers = []

        for d in range(1, self.degree + 1):
            for comb in combinations_with_replacement(range(n_features), d):
                cols.append(np.prod(Xf[:, comb], axis=1))
                feature_powers.append(comb)

        Phi = np.column_stack(cols) if cols else np.empty((n_samples, 0))
        return Phi, feature_powers

    def _make_polynomial_from_powers(self, X, feature_powers):
        """
        Build Phi (NO bias) using previously stored feature_powers.
        Expects X includes intercept in col 0, but we exclude it.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not np.allclose(X[:, 0], 1.0):
            raise ValueError("X must include intercept column of 1s as the FIRST column.")

        Xf = X[:, 1:]
        cols = []
        for comb in feature_powers:
            cols.append(np.prod(Xf[:, comb], axis=1))
        return np.column_stack(cols) if cols else np.empty((X.shape[0], 0))

    def _add_intercept(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        n = Phi.shape[0]
        return np.hstack([np.ones((n, 1)), Phi])

    # =========================================================
    # Standardization (Phi, y) and unscale weights
    # =========================================================
    def _standardize_design_and_target(self, Phi, y):
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if Phi.size > 0:
            phi_mean = Phi.mean(axis=0, keepdims=True)
            phi_std = Phi.std(axis=0, keepdims=True)
            phi_std = np.where(phi_std == 0.0, 1.0, phi_std)
            Phi_s = (Phi - phi_mean) / phi_std
        else:
            phi_mean = np.zeros((1, Phi.shape[1]))
            phi_std = np.ones((1, Phi.shape[1]))
            Phi_s = Phi

        y_mean = float(y.mean())
        y_std = float(y.std())
        if y_std == 0.0:
            y_std = 1.0
        y_s = (y - y_mean) / y_std

        return Phi_s, y_s, phi_mean, phi_std, y_mean, y_std

    def _unscale_poly_weights(self, w_scaled, phi_mean, phi_std, y_mean, y_std):
        w_scaled = np.asarray(w_scaled, dtype=float).reshape(-1, 1)
        if w_scaled.shape[0] <= 1:
            return w_scaled.copy()

        a0 = float(w_scaled[0, 0])
        a = w_scaled[1:, 0]

        phi_mean_vec = phi_mean.reshape(-1)
        phi_std_vec = phi_std.reshape(-1)
        phi_std_vec = np.where(phi_std_vec == 0.0, 1.0, phi_std_vec)

        w_nonbias = y_std * a / phi_std_vec
        correction = np.sum(y_std * a * phi_mean_vec / phi_std_vec)
        w0 = y_mean + y_std * a0 - correction

        w_unscaled = np.empty_like(w_scaled)
        w_unscaled[0, 0] = w0
        w_unscaled[1:, 0] = w_nonbias
        return w_unscaled

    # =========================================================
    # RBF kernel (INSTANCE METHOD; not static)
    # =========================================================
    def rbf_kernel(self, Z, Y=None):
        """
        RBF kernel K(z,y) = exp(-gamma ||z - y||^2)

        IMPORTANT:
          - Z and Y are expected to be expanded design matrices WITH intercept in col 0 (1s).
          - We IGNORE intercept column in distance computations.
        """
        Z = np.asarray(Z, float)
        Zf = Z[:, 1:]
        if Y is None:
            Yf = Zf
        else:
            Y = np.asarray(Y, float)
            Yf = Y[:, 1:]

        Z_norm = np.sum(Zf ** 2, axis=1)[:, None]
        Y_norm = np.sum(Yf ** 2, axis=1)[None, :]
        sq = Z_norm + Y_norm - 2.0 * (Zf @ Yf.T)
        sq = np.maximum(sq, 0.0)
        return np.exp(-self.gamma * sq)

    # =========================================================
    # Loss + gradient (for GD)
    # =========================================================
    def _mse_loss(self, Xb, y, w, l2_lambda=0.0):
        Xb = np.asarray(Xb)
        y = np.asarray(y).reshape(-1, 1)
        w = np.asarray(w)
        n = Xb.shape[0]

        r = Xb @ w - y
        loss = 0.5 * np.mean(r ** 2)

        if l2_lambda > 0.0:
            w_reg = w[1:, :]
            loss += 0.5 * l2_lambda * float(np.sum(w_reg ** 2))

        return float(loss)

    def _mse_grad(self, Xb, y, w, l2_lambda=0.0):
        Xb = np.asarray(Xb)
        y = np.asarray(y).reshape(-1, 1)
        w = np.asarray(w)
        n = Xb.shape[0]

        r = Xb @ w - y
        grad = (Xb.T @ r) / n

        if l2_lambda > 0.0:
            reg = np.vstack([np.zeros((1, 1)), w[1:, :]])
            grad += l2_lambda * reg

        return grad

    # =========================================================
    # Batch iterator (GD)
    # =========================================================
    def _iterate_batches(self, Xb, y, batch_size, shuffle=True, seed=None):
        Xb = np.asarray(Xb)
        y = np.asarray(y).reshape(-1, 1)
        n = Xb.shape[0]

        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]
            yield Xb[b], y[b]

    # =========================================================
    # Closed-form fit (OLS / Ridge)
    # =========================================================
    def _fit_closed_form(self, Xb_s, y_s, l2_lambda):
        n_features_b = Xb_s.shape[1]
        reg = np.eye(n_features_b)
        reg[0, 0] = 0.0

        A = Xb_s.T @ Xb_s + l2_lambda * reg
        b = Xb_s.T @ y_s

        try:
            a = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            a = np.linalg.pinv(A) @ b if self.use_pinv else np.linalg.pinv(A) @ b

        return a

    # =========================================================
    # Coordinate descent (Lasso / ElasticNet)
    # =========================================================
    def _soft_threshold_scalar(self, rho, lam):
        if rho > lam:
            return rho - lam
        if rho < -lam:
            return rho + lam
        return 0.0

    def _fit_coordinate_descent(self, Phi_s, y_s):
        n, m = Phi_s.shape
        if m == 0:
            w0 = float(y_s.mean())
            return np.array([[w0]])

        w0 = float(y_s.mean())
        w = np.zeros((m, 1))

        col_norms = np.sum(Phi_s ** 2, axis=0) / n

        l1 = self.alpha * (1.0 if self.penalty == "lasso" else self.l1_ratio)
        l2 = 0.0 if self.penalty == "lasso" else self.alpha * (1.0 - self.l1_ratio)

        for it in range(1, self.cd_max_iters + 1):
            w_old = w.copy()
            y_pred = w0 + Phi_s @ w

            for j in range(m):
                pj = Phi_s[:, j].reshape(-1, 1)
                r_j = y_s - (y_pred - pj * w[j])
                rho_j = float((pj.T @ r_j) / n)
                z_j = float(col_norms[j])

                w_j_new = self._soft_threshold_scalar(rho_j, l1) / (z_j + l2)
                y_pred += pj * (w_j_new - w[j])
                w[j] = w_j_new

            residual = y_s - (w0 + Phi_s @ w)
            w0 = float(w0 + residual.mean())

            max_delta = float(np.max(np.abs(w - w_old)))
            if self.verbose and it % 500 == 0:
                print(f"[CD] iter {it} - max |Δw| = {max_delta:.4e}")
            if max_delta < self.cd_tol:
                if self.verbose:
                    print(f"[CD] converged at iter {it}, max |Δw| = {max_delta:.4e}")
                break

        return np.vstack([np.array([[w0]]), w])

    # =========================================================
    # GD fit (L2 only)
    # =========================================================
    def _fit_gd(self, Xb_s, y_s, l2_lambda):
        n, d1 = Xb_s.shape
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(0.0, 0.01, size=(d1, 1))

        # batch size
        if self.batch_mode == "batch":
            bs = n
        elif self.batch_mode == "stochastic":
            bs = 1
        else:
            bs = min(self.batch_size, n)

        # optimizer state
        v_mom = np.zeros_like(w)
        G = np.zeros_like(w)
        v2 = np.zeros_like(w)
        Eg = np.zeros_like(w)
        m_adam = np.zeros_like(w)
        v_adam = np.zeros_like(w)
        t = 0

        self.loss_history_ = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            seed = None if self.random_state is None else (self.random_state + epoch)

            for Xb, yb in self._iterate_batches(Xb_s, y_s, bs, shuffle=self.shuffle, seed=seed):

                if self.optimizer == "vanilla":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    w = w - self.learning_rate * grad

                elif self.optimizer == "momentum":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    v_mom = self.beta_m * v_mom + grad
                    w = w - self.learning_rate * v_mom

                elif self.optimizer == "nesterov":
                    w_look = w - self.learning_rate * self.beta_m * v_mom
                    grad = self._mse_grad(Xb, yb, w_look, l2_lambda=l2_lambda)
                    v_mom = self.beta_m * v_mom + grad
                    w = w - self.learning_rate * v_mom

                elif self.optimizer == "adagrad":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    G = G + grad * grad
                    w = w - self.learning_rate * grad / (np.sqrt(G) + self.eps)

                elif self.optimizer == "rmsprop":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    v2 = self.beta_v * v2 + (1.0 - self.beta_v) * (grad * grad)
                    w = w - self.learning_rate * grad / (np.sqrt(v2) + self.eps)

                elif self.optimizer == "adadelta":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    Eg = self.rho * Eg + (1.0 - self.rho) * (grad * grad)
                    update = - (np.sqrt(v2 + self.eps) / np.sqrt(Eg + self.eps)) * grad
                    v2 = self.rho * v2 + (1.0 - self.rho) * (update * update)
                    w = w + update  # no learning_rate multiplier

                elif self.optimizer == "adam":
                    grad = self._mse_grad(Xb, yb, w, l2_lambda=l2_lambda)
                    t += 1
                    m_adam = self.beta_m * m_adam + (1.0 - self.beta_m) * grad
                    v_adam = self.beta_v * v_adam + (1.0 - self.beta_v) * (grad * grad)
                    m_hat = m_adam / (1.0 - self.beta_m ** t)
                    v_hat = v_adam / (1.0 - self.beta_v ** t)
                    w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")

                epoch_loss += self._mse_loss(Xb, yb, w, l2_lambda=l2_lambda)

            num_batches = int(np.ceil(n / bs))
            epoch_loss /= max(1, num_batches)
            self.loss_history_.append(epoch_loss)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"[GD-{self.optimizer}] epoch {epoch}/{self.epochs} - loss(scaled): {epoch_loss:.6f}")

        return w

    # =========================================================
    # RBF Kernel Ridge fit (on expanded polynomial features)
    # =========================================================
    def _fit_rbf(self, Phi, y):
        """
        Fit RBF KRR on polynomial-expanded features.

        Steps:
          1) Standardize Phi (but NOT y with same phi stats)
          2) Standardize y separately (store mean/std)
          3) Build Z_s = [1, Phi_s]
          4) Solve alpha_vec = (K + alpha I)^-1 y_s
        """
        # standardize Phi
        Phi = np.asarray(Phi, float)
        y = np.asarray(y, float).reshape(-1, 1)

        if Phi.size > 0:
            phi_mean = Phi.mean(axis=0, keepdims=True)
            phi_std = Phi.std(axis=0, keepdims=True)
            phi_std = np.where(phi_std == 0.0, 1.0, phi_std)
            Phi_s = (Phi - phi_mean) / phi_std
        else:
            phi_mean = np.zeros((1, Phi.shape[1]))
            phi_std = np.ones((1, Phi.shape[1]))
            Phi_s = Phi

        # standardize y
        y_mean = float(y.mean())
        y_std = float(y.std())
        if y_std == 0.0:
            y_std = 1.0
        y_s = (y - y_mean) / y_std

        # store stats for prediction
        self.phi_mean_ = phi_mean
        self.phi_std_ = phi_std
        self._y_mean_rbf_ = y_mean
        self._y_std_rbf_ = y_std

        Z_s = self._add_intercept(Phi_s)  # (n, 1+m)

        K = self.rbf_kernel(Z_s, None)
        n = K.shape[0]
        A = K + self.alpha * np.eye(n)

        self._alpha_vec_ = np.linalg.solve(A, y_s)
        self._Z_train_scaled_ = Z_s.copy()

        # simple training loss record (MSE in y_s space)
        y_hat_s = K @ self._alpha_vec_
        self.loss_history_ = [float(np.mean((y_hat_s - y_s) ** 2))]

    # =========================================================
    # Public fit/predict
    # =========================================================
    def fit(self):
        """
        Fit according to method + penalty.
        Stores:
          - polynomial param methods: self._w (ORIGINAL polynomial basis), self.feature_powers_
          - rbf method: self._alpha_vec_, self._Z_train_scaled_, and stats
        """
        self._validate_config()

        Phi, feature_powers = self._make_polynomial_features(self.X)
        self.feature_powers_ = feature_powers

        # clear all learned params
        self._w = None
        self._alpha_vec_ = None
        self._Z_train_scaled_ = None
        self._y_mean_rbf_ = None
        self._y_std_rbf_ = None
        self.loss_history_ = []

        # RBF path
        if self.method == "rbf":
            self._fit_rbf(Phi, self.y)
            return self

        # Standardize for parametric methods
        Phi_s, y_s, phi_mean, phi_std, y_mean, y_std = self._standardize_design_and_target(Phi, self.y)
        self.phi_mean_ = phi_mean
        self.phi_std_ = phi_std
        self.y_mean_ = y_mean
        self.y_std_ = y_std

        Xb_s = self._add_intercept(Phi_s)

        # map penalty->l2_lambda when relevant
        if self.penalty == "ridge":
            l2_lambda = self.alpha
        elif self.penalty == "elasticnet":
            l2_lambda = self.alpha * (1.0 - self.l1_ratio)
        else:
            l2_lambda = 0.0

        if self.method == "closed_form":
            a = self._fit_closed_form(Xb_s, y_s, l2_lambda=l2_lambda)

        elif self.method == "cd":
            a = self._fit_coordinate_descent(Phi_s, y_s)

        else:  # "gd"
            if self.penalty in ("lasso", "elasticnet"):
                raise ValueError("GD supports only penalty='none' or 'ridge' (use method='cd' for L1).")
            a = self._fit_gd(Xb_s, y_s, l2_lambda=l2_lambda)

        # unscale weights to original polynomial basis
        self._w = self._unscale_poly_weights(a, phi_mean, phi_std, y_mean, y_std)

        # if not gd, keep a loss_history_ convenience
        if self.method != "gd":
            Xb = self._add_intercept(Phi)
            self.loss_history_ = [self._mse_loss(Xb, self.y, self._w, l2_lambda=0.0)]

        return self

    def predict(self, X_new):
        """
        Predict y in ORIGINAL scale for new raw X_new.
        Assumes X_new includes intercept col 0.
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(f"X_new must have {self.X.shape[1]} columns (including bias).")
        if not np.allclose(X_new[:, 0], 1.0):
            raise ValueError("X_new must include intercept column of 1s as the FIRST column.")

        # RBF prediction
        if self.method == "rbf":
            if self._alpha_vec_ is None or self._Z_train_scaled_ is None:
                raise ValueError("RBF model not fitted yet.")

            Phi_new = self._make_polynomial_from_powers(X_new, self.feature_powers_)

            # standardize Phi_new using training phi_mean_/phi_std_
            if Phi_new.size > 0:
                Phi_s_new = (Phi_new - self.phi_mean_) / self.phi_std_
            else:
                Phi_s_new = Phi_new

            Z_s_new = self._add_intercept(Phi_s_new)
            Kt = self.rbf_kernel(Z_s_new, self._Z_train_scaled_)

            y_hat_s = (Kt @ self._alpha_vec_).reshape(-1)
            y_hat = y_hat_s * self._y_std_rbf_ + self._y_mean_rbf_
            return y_hat

        # Parametric (polynomial weights) prediction
        if self._w is None or self.feature_powers_ is None:
            raise ValueError("Model not fitted yet.")

        Phi_new = self._make_polynomial_from_powers(X_new, self.feature_powers_)
        Xb_new = self._add_intercept(Phi_new)
        return (Xb_new @ self._w).ravel()
