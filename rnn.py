import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler

from scikeras.wrappers import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Helpers
# -----------------------------
def iso_week_to_monday_datetime(week_str: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_str) + "-1", format="%G-W%V-%u", errors="coerce")

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def make_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X, dtype=np.float32)[:, :, None]  # (N, T, 1)
    y = np.array(y, dtype=np.float32)
    return X, y

def metrics(y_true, y_pred, p_features=1):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ar2 = adjusted_r2(r2, len(y_true), p_features)
    evs = explained_variance_score(y_true, y_pred)
    denom = np.clip(np.abs(y_true), 1e-12, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {
        "n": int(len(y_true)),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Adj_R2": float(ar2) if np.isfinite(ar2) else np.nan,
        "ExplainedVariance": float(evs),
        "MAPE_%": float(mape),
    }


# -----------------------------
# Model builder (control layers/neurons)
# -----------------------------
def build_rnn_model(
    lookback: int,
    hidden_layers: int = 2,
    neurons_per_layer=64,     # int or list[int]
    cell_type: str = "LSTM",  # "LSTM" or "GRU"
    dropout: float = 0.1,
    lr: float = 1e-3
):
    if isinstance(neurons_per_layer, int):
        units = [neurons_per_layer] * hidden_layers
    else:
        units = list(neurons_per_layer)
        if len(units) != hidden_layers:
            raise ValueError("neurons_per_layer must be an int or a list of length hidden_layers")

    inputs = keras.Input(shape=(lookback, 1))

    x = inputs
    RNNLayer = layers.LSTM if cell_type.upper() == "LSTM" else layers.GRU

    for i in range(hidden_layers):
        return_sequences = (i < hidden_layers - 1)
        x = RNNLayer(units[i], return_sequences=return_sequences)(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    return model


# -----------------------------
# Train/test pipeline (70% train, 30% test) -> returns metrics row
# -----------------------------
def run_rnn_baseline_sklearn(
    csv_path: str,
    lookback: int = 26,
    hidden_layers: int = 2,
    neurons_per_layer=64,
    cell_type: str = "LSTM",
    epochs: int = 60,
    batch_size: int = 64,
    random_state: int = 0,
    do_plot: bool = False,
):
    df = pd.read_csv(csv_path)
    df.columns = ["year_week", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = df["year_week"].astype(str).apply(iso_week_to_monday_datetime)
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

    values = df["value"].to_numpy(dtype=np.float32)

    X, y = make_sequences(values, lookback=lookback)

    N = X.shape[0]
    n_train = int(np.floor(0.7 * N))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # scale y on train
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # scale X on train
    x_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 1)
    x_scaler.fit(X_train_flat)
    X_train_s = x_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    reg = KerasRegressor(
        model=build_rnn_model,
        model__lookback=lookback,
        model__hidden_layers=hidden_layers,
        model__neurons_per_layer=neurons_per_layer,
        model__cell_type=cell_type,
        model__dropout=0.1,
        model__lr=1e-3,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        random_state=random_state,
    )

    reg.fit(X_train_s, y_train_s)

    # predict and unscale
    yhat_train_s = reg.predict(X_train_s)
    yhat_test_s = reg.predict(X_test_s)

    yhat_train = y_scaler.inverse_transform(np.asarray(yhat_train_s).reshape(-1, 1)).ravel()
    yhat_test = y_scaler.inverse_transform(np.asarray(yhat_test_s).reshape(-1, 1)).ravel()

    train_m = metrics(y_train, yhat_train, p_features=1)
    test_m = metrics(y_test, yhat_test, p_features=1)

    if do_plot:
        target_dates = df["date"].to_numpy()[lookback:]
        split_date = target_dates[n_train]
        yhat_all = np.concatenate([yhat_train, yhat_test])
        y_all = np.concatenate([y_train, y_test])

        plt.figure()
        plt.plot(target_dates[:n_train], y_all[:n_train], label="Train (actual)")
        plt.plot(target_dates[n_train:], y_all[n_train:], label="Test (actual)")
        plt.plot(target_dates, yhat_all, label="RNN prediction")
        plt.axvline(split_date, linestyle="--", label="Train/Test split")
        plt.xlabel("Date (week start)")
        plt.ylabel("CO₂ (ppm)")
        plt.title(f"{cell_type} | lookback={lookback} | layers={hidden_layers} | units={neurons_per_layer}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    row = {
        "cell_type": cell_type,
        "lookback": lookback,
        "layers": hidden_layers,
        "units": neurons_per_layer,
        "epochs": epochs,
        "batch_size": batch_size,

        "train_n": train_m["n"],
        "train_MAE": train_m["MAE"],
        "train_RMSE": train_m["RMSE"],
        "train_R2": train_m["R2"],
        "train_Adj_R2": train_m["Adj_R2"],
        "train_ExplainedVariance": train_m["ExplainedVariance"],
        "train_MAPE_%": train_m["MAPE_%"],

        "test_n": test_m["n"],
        "test_MAE": test_m["MAE"],
        "test_RMSE": test_m["RMSE"],
        "test_R2": test_m["R2"],
        "test_Adj_R2": test_m["Adj_R2"],
        "test_ExplainedVariance": test_m["ExplainedVariance"],
        "test_MAPE_%": test_m["MAPE_%"],
    }
    return row


def run_grid_and_report(
    csv_path: str,
    lookback: int = 26,
    layers_list=(1, 2, 3, 5, 10),
    units_list=(1, 2, 3, 5, 10, 20, 30),
    cell_type="LSTM",
    epochs=100,
    batch_size=64,
    save_csv_path="rnn_grid_results.csv",
):
    rows = []
    failures = []

    for layer in layers_list:
        for neuron in units_list:
            try:
                row = run_rnn_baseline_sklearn(
                    csv_path=csv_path,
                    lookback=lookback,
                    hidden_layers=layer,
                    neurons_per_layer=neuron,
                    cell_type=cell_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    do_plot=False,
                )
                rows.append(row)
                print(f"OK  layers={layer:>2} units={neuron:>2} | test_RMSE={row['test_RMSE']:.6f} test_R2={row['test_R2']:.6f}")
            except Exception as e:
                failures.append({"layers": layer, "units": neuron, "error": str(e)})
                print(f"FAIL layers={layer:>2} units={neuron:>2} | {e}")

    results_df = pd.DataFrame(rows)

    # Pretty print formatting
    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", 2000)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    # Best by lowest test_RMSE (then test_MAE)
    results_sorted = results_df.sort_values(["test_RMSE", "test_MAE"], ascending=[True, True])

    print("\n==================== TOP 20 (best test_RMSE) ====================")
    cols = [
        "layers","units","lookback",
        "test_RMSE","test_MAE","test_R2","test_Adj_R2","test_ExplainedVariance","test_MAPE_%",
        "train_RMSE","train_R2","train_Adj_R2"
    ]
    if len(results_sorted) > 0:
        print(results_sorted[cols].head(20).to_string(index=False))
    else:
        print("No successful runs. Check failures CSV.")

    if save_csv_path and len(results_df) > 0:
        results_df.to_csv(save_csv_path, index=False)
        print(f"\nSaved full grid results to: {save_csv_path}")

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = "rnn_grid_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"Saved failures to: {fail_path} (count={len(failures)})")

    return results_df

results = run_grid_and_report(
    csv_path="co2_data.csv",
    lookback=26,
    layers_list=(1, 2, 3, 5, 10, 20, 30),
    units_list=(1, 2, 3, 5, 10, 20, 30),
    cell_type="LSTM",
    epochs=100,
    batch_size=64,
    save_csv_path="rnn_grid_results.csv",
)

if len(results) > 0:
    best = results.sort_values(["test_RMSE", "test_MAE"], ascending=[True, True]).iloc[0]
    print("\nBest config:", dict(best[["layers","units","test_RMSE","test_R2","test_Adj_R2"]]))

    _ = run_rnn_baseline_sklearn(
        csv_path="/co2_data.csv",
        lookback=int(best["lookback"]),
        hidden_layers=int(best["layers"]),
        neurons_per_layer=int(best["units"]),
        cell_type=str(best["cell_type"]),
        epochs=int(best["epochs"]),
        batch_size=int(best["batch_size"]),
        do_plot=True,
    )

"""
2025-12-18 05:55:59.405476: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units= 1 | test_RMSE=25.441243 test_R2=-4.921831
2025-12-18 05:56:06.433260: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units= 2 | test_RMSE=20.370551 test_R2=-2.796513
2025-12-18 05:56:13.583297: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units= 3 | test_RMSE=19.671158 test_R2=-2.540293
2025-12-18 05:56:20.839837: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units= 5 | test_RMSE=15.126824 test_R2=-1.093509
2025-12-18 05:56:28.415209: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units=10 | test_RMSE=8.606044 test_R2=0.322380
2025-12-18 05:56:36.732786: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units=20 | test_RMSE=4.617244 test_R2=0.804950
2025-12-18 05:56:46.013277: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 1 units=30 | test_RMSE=3.886520 test_R2=0.861802
2025-12-18 05:56:59.117152: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units= 1 | test_RMSE=22.342929 test_R2=-3.567300
2025-12-18 05:57:12.659503: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units= 2 | test_RMSE=17.150233 test_R2=-1.691035
2025-12-18 05:57:26.528073: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units= 3 | test_RMSE=18.032081 test_R2=-1.974891
2025-12-18 05:57:40.615260: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units= 5 | test_RMSE=16.066523 test_R2=-1.361691
2025-12-18 05:57:55.766435: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units=10 | test_RMSE=7.871084 test_R2=0.433176
2025-12-18 05:58:12.753873: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units=20 | test_RMSE=6.556699 test_R2=0.606677
2025-12-18 05:58:32.969372: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 2 units=30 | test_RMSE=7.215413 test_R2=0.523677
2025-12-18 05:58:52.642314: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units= 1 | test_RMSE=22.148679 test_R2=-3.488229
2025-12-18 05:59:15.025945: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units= 2 | test_RMSE=19.977069 test_R2=-2.651261
2025-12-18 05:59:35.932754: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units= 3 | test_RMSE=18.517655 test_R2=-2.137265
2025-12-18 05:59:58.375611: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units= 5 | test_RMSE=14.944464 test_R2=-1.043337
2025-12-18 06:00:20.418530: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units=10 | test_RMSE=11.611748 test_R2=-0.233600
2025-12-18 06:00:46.308186: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units=20 | test_RMSE=10.016217 test_R2=0.082118
2025-12-18 06:01:18.344402: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 3 units=30 | test_RMSE=6.676985 test_R2=0.592113
2025-12-18 06:01:49.100921: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units= 1 | test_RMSE=25.982055 test_R2=-5.176271
2025-12-18 06:02:21.722231: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units= 2 | test_RMSE=22.260005 test_R2=-3.533461
2025-12-18 06:02:54.422740: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units= 3 | test_RMSE=20.059641 test_R2=-2.681507
2025-12-18 06:03:28.058199: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units= 5 | test_RMSE=18.995906 test_R2=-2.301409
2025-12-18 06:04:07.645594: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units=10 | test_RMSE=13.124318 test_R2=-0.575915
2025-12-18 06:04:52.210243: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units=20 | test_RMSE=14.353131 test_R2=-0.884832
2025-12-18 06:05:46.679667: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers= 5 units=30 | test_RMSE=12.936173 test_R2=-0.531056
2025-12-18 06:06:48.361510: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units= 1 | test_RMSE=26.454869 test_R2=-5.403104
2025-12-18 06:07:56.042941: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units= 2 | test_RMSE=24.985897 test_R2=-4.711750
2025-12-18 06:09:01.910177: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units= 3 | test_RMSE=24.082350 test_R2=-4.306120
2025-12-18 06:10:12.496359: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units= 5 | test_RMSE=23.649329 test_R2=-4.117018
2025-12-18 06:11:27.996049: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units=10 | test_RMSE=18.085851 test_R2=-1.992659
2025-12-18 06:13:00.445039: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units=20 | test_RMSE=17.589889 test_R2=-1.830776
2025-12-18 06:14:53.726945: E tensorflow/core/framework/node_def_util.cc:680] NodeDef mentions attribute use_unbounded_threadpool which is not in the op definition: Op<name=MapDataset; signature=input_dataset:variant, other_arguments: -> handle:variant; attr=f:func; attr=Targuments:list(type),min=0; attr=output_types:list(type),min=1; attr=output_shapes:list(shape),min=1; attr=use_inter_op_parallelism:bool,default=true; attr=preserve_cardinality:bool,default=false; attr=force_synchronous:bool,default=false; attr=metadata:string,default=""> This may be expected if your graph generating binary is newer  than this binary. Unknown attributes will be ignored. NodeDef: {{node ParallelMapDatasetV2/_14}}
OK  layers=10 units=30 | test_RMSE=17.454229 test_R2=-1.787281
"""