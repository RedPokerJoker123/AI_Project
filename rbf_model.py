import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel as C
)

# ==========================
# EDIT THIS
# ==========================
CSV_PATH = "./project_files/co2_data.csv"   # columns: year_week,value

# weekly series -> annual cycle ~ 1 year
PERIOD_YEARS = 1.0


# --------------------------
# ISO week -> date (Monday)
# --------------------------
def parse_iso_year_week_to_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    parts = s.str.split("-W", expand=True)
    if parts.shape[1] != 2:
        raise ValueError("Expected 'YYYY-Www' in year_week column.")
    year = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
    week = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    if year.isna().any() or week.isna().any():
        bad = s[year.isna() | week.isna()].head(5).tolist()
        raise ValueError(f"Bad year_week entries. Examples: {bad}")

    iso_str = year.astype(str) + "-W" + week.astype(str).str.zfill(2) + "-1"  # Monday
    dt = pd.to_datetime(iso_str, format="%G-W%V-%u", utc=True, errors="coerce")
    if dt.isna().any():
        bad = s[dt.isna()].head(5).tolist()
        raise ValueError(f"Could not parse some ISO weeks. Examples: {bad}")
    return dt


def to_fractional_year(dt_utc: pd.Series) -> np.ndarray:
    """
    Convert timestamps to continuous 'year + fraction' (1D feature for GPR).
    """
    dt = dt_utc.dt.tz_convert("UTC")
    year = dt.dt.year.to_numpy().astype(float)
    day_of_year = dt.dt.dayofyear.to_numpy().astype(float)

    is_leap = ((dt.dt.year % 4 == 0) & ((dt.dt.year % 100 != 0) | (dt.dt.year % 400 == 0))).to_numpy()
    days_in_year = np.where(is_leap, 366.0, 365.0)

    frac = (day_of_year - 1.0) / days_in_year
    return year + frac


# --------------------------
# Metrics
# --------------------------
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if not np.isfinite(r2) or n <= (p + 1):
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


# --------------------------
# Robust outlier down-weighting (NO data modification)
# --------------------------
def build_alpha_outlier_weights(y_train_c: np.ndarray,
                                base_std: float = 1.0,
                                outlier_std: float = 10.0,
                                z_thresh: float = 6.0,
                                window: int = 25):
    """
    Returns:
      alpha: per-point observation variance array for GPR (shape n_train,)
      z: robust z-scores used for flagging

    We do NOT change y. We just tell GPR: "these points are very noisy".
    """
    y = np.asarray(y_train_c, float).reshape(-1)

    s = pd.Series(y)
    roll = s.rolling(window=window, center=True, min_periods=max(10, window // 2))
    med = roll.median().to_numpy()
    mad = roll.apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True).to_numpy()

    # fallback for edges / NaNs
    mad_ok = mad[np.isfinite(mad) & (mad > 1e-12)]
    mad_fallback = float(np.median(mad_ok)) if mad_ok.size else 1.0
    mad = np.where(np.isfinite(mad) & (mad > 1e-12), mad, mad_fallback)

    # robust z (MAD -> sigma via 1.4826)
    z = np.abs(y - med) / (1.4826 * mad + 1e-12)
    z = np.where(np.isfinite(z), z, 0.0)

    alpha = np.full_like(y, (base_std ** 2), dtype=float)
    alpha[z > z_thresh] = (outlier_std ** 2)

    return alpha, z


# --------------------------
# Main
# --------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    if "year_week" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Expected columns ['year_week','value']. Found: {list(df.columns)}")

    df = df[["year_week", "value"]].dropna().copy()
    df["date"] = parse_iso_year_week_to_date(df["year_week"])
    df = df.sort_values("date").reset_index(drop=True)

    y = df["value"].astype(float).to_numpy()
    X = to_fractional_year(df["date"]).reshape(-1, 1)  # 1D time feature in years

    n = len(df)
    split = int(np.floor(0.7 * n))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # mean-center for stability
    y_mean = float(np.mean(y_train))
    y_train_c = y_train - y_mean

    # --- per-point noise weights (downweight training outliers)
    alpha, z = build_alpha_outlier_weights(
        y_train_c,
        base_std=1.0,      # baseline noise std in ppm
        outlier_std=10.0,  # outlier treated as much noisier
        z_thresh=6.0,
        window=25
    )
    print(f"[OUTLIERS] flagged {int(np.sum(z > 6.0))} / {len(z)} training points (down-weighted, not removed)")

    # --- Kernel similar to Mauna Loa example: trend + seasonality + irregularities + noise
    kernel = (
        40.0**2 * RBF(length_scale=40.0) +
        2.0**2 * RBF(length_scale=10.0) * ExpSineSquared(length_scale=1.0, periodicity=PERIOD_YEARS, periodicity_bounds="fixed") +
        0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.5) +
        0.1**2 * RBF(length_scale=0.1) +
        WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        alpha=alpha,                 # <-- key improvement
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=3,
        random_state=0
    )

    gpr.fit(X_train, y_train_c)

    y_pred_c, y_std = gpr.predict(X_test, return_std=True)
    y_pred = y_pred_c + y_mean

    test_rmse = rmse(y_test, y_pred)
    test_mae = mae(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_adj = adjusted_r2(test_r2, n=len(y_test), p=1)

    print("[KERNEL] learned:", gpr.kernel_)
    print(f"[TEST] n={n} train={split} test={n-split} | RMSE={test_rmse:.4f} MAE={test_mae:.4f} R2={test_r2:.4f} AdjR2={test_adj:.4f}")

    # Save predictions
    out = df.iloc[split:].copy()
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out["pred_std"] = y_std
    out["lower_2std"] = y_pred - 2.0 * y_std
    out["upper_2std"] = y_pred + 2.0 * y_std
    out["error"] = out["y_pred"] - out["y_true"]
    out.to_csv("gpr_predictions.csv", index=False)
    print("[SAVE] gpr_predictions.csv")

    # Also save which training points were flagged (for your own inspection)
    flagged = pd.DataFrame({
        "date": df["date"].iloc[:split].to_numpy(),
        "y_train": y_train,
        "y_train_centered": y_train_c,
        "robust_z": z,
        "alpha_used": alpha
    })
    flagged.to_csv("gpr_training_outliers.csv", index=False)
    print("[SAVE] gpr_training_outliers.csv")

    # Plot
    plt.figure()
    plt.plot(df["date"].iloc[:split], y_train, label="train (true)")
    plt.plot(df["date"].iloc[split:], y_test, label="test (true)")
    plt.plot(df["date"].iloc[split:], y_pred, label="test (pred)")
    plt.fill_between(
        df["date"].iloc[split:],
        y_pred - 2.0 * y_std,
        y_pred + 2.0 * y_std,
        alpha=0.2,
        label="±2 std"
    )
    plt.title(f"GPR CO2 (outlier down-weighted) | R2={test_r2:.3f} AdjR2={test_adj:.3f}")
    plt.xlabel("date")
    plt.ylabel("CO2 (ppm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gpr_forecast.png", dpi=160)
    print("[SAVE] gpr_forecast.png")


if __name__ == "__main__":
    main()
