import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, ExpSineSquared, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

def iso_week_to_monday_datetime(week_str: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_str) + "-1", format="%G-W%V-%u", errors="coerce")

def datetime_to_decimal_year(dt: pd.Series) -> np.ndarray:
    year = dt.dt.year.to_numpy()
    start = pd.to_datetime(year.astype(str) + "-01-01")
    end = pd.to_datetime((year + 1).astype(str) + "-01-01")
    frac = (dt - start) / (end - start)
    return year + frac.to_numpy(dtype=float)

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def fit_co2_trend_plus_gpr(csv_path: str, train_frac: float = 0.7, random_state: int = 0):
    df = pd.read_csv(csv_path)
    df["date"] = df["week"].astype(str).apply(iso_week_to_monday_datetime)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    t = datetime_to_decimal_year(df["date"])
    X = t.reshape(-1, 1)
    y = df["value"].to_numpy(float)

    n = len(df)
    n_train = int(np.floor(train_frac * n))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # 1) Fit trend (linear) on training
    trend = LinearRegression()
    trend.fit(X_train, y_train)

    y_train_trend = trend.predict(X_train)
    y_test_trend  = trend.predict(X_test)

    # Residuals for GP
    r_train = y_train - y_train_trend

    # 2) GP on residuals (stationary is OK now)
    # One smooth term + seasonal term + noise
    kernel = (
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(0.1, 50.0)) +
        C(1.0, (1e-3, 1e3)) * ExpSineSquared(
            length_scale=1.0,
            periodicity=1.0,
            length_scale_bounds=(1e-2, 10.0),
            periodicity_bounds=(0.9, 1.1),
        ) +
        WhiteKernel(noise_level=0.2, noise_level_bounds=(1e-6, 10.0))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=random_state,
    )
    gpr.fit(X_train, r_train)

    # Predict residuals + combine with trend
    rhat_all, rstd_all = gpr.predict(X, return_std=True)
    yhat_all = trend.predict(X) + rhat_all

    yhat_train = yhat_all[:n_train]
    yhat_test  = yhat_all[n_train:]

    # Metrics
    def metrics(y_true, y_pred, p_features=1):
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ar2 = adjusted_r2(r2, len(y_true), p_features)
        evs = explained_variance_score(y_true, y_pred)
        return mae, rmse, r2, ar2, evs

    tr = metrics(y_train, yhat_train)
    te = metrics(y_test, yhat_test)

    print("---- Trend (linear) ----")
    print(f"y = {trend.coef_[0]:.6f} * t + {trend.intercept_:.6f}")
    print("\n---- GP kernel (residuals) ----")
    print(gpr.kernel_)
    print("\n---- Train: MAE RMSE R2 AdjR2 EVS ----")
    print(tr)
    print("\n---- Test:  MAE RMSE R2 AdjR2 EVS ----")
    print(te)

    # Plot
    dates = df["date"].to_numpy()
    split_date = dates[n_train]

    plt.figure()
    plt.plot(dates[:n_train], y_train, label="Train (actual)")
    plt.plot(dates[n_train:], y_test, label="Test (actual)")
    plt.plot(dates, yhat_all, label="Trend + GP mean")
    plt.fill_between(dates, yhat_all - 1.96 * rstd_all, yhat_all + 1.96 * rstd_all, alpha=0.2, label="95% CI (residual)")
    plt.axvline(split_date, linestyle="--", label="Train/Test split")
    plt.xlabel("Date (week start)")
    plt.ylabel("CO₂ (ppm)")
    plt.title("CO₂: Linear Trend + Seasonal GP on Residuals")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example:
fit_co2_trend_plus_gpr("/co2_data.csv")

