import json
from collections import defaultdict

def model_group(run_id: str) -> str:
    # Decide "family" from run_id text (tune if your naming differs)
    if "NeuralNetwork" in run_id or "MLP" in run_id:
        return "neural_network"
    if "DecisionTreeClassifier" in run_id or "RegressionDecisionTree" in run_id:
        return "decision_tree"
    if "PolynomialRegression" in run_id or "LinearRegression" in run_id:
        return "regression"
    return "other"

def is_regression_metric(params: dict) -> bool:
    return params.get("adjusted_r2") is not None

def is_classification_metric(params: dict) -> bool:
    return params.get("auc") is not None

def keep_best_per_run(entries, key, higher_is_better=True):
    """
    entries: list of dicts containing run_id and metric key
    keeps only one record per run_id (the best metric)
    """
    best = {}
    for e in entries:
        rid = e["run_id"]
        val = e[key]
        if val is None:
            continue
        if rid not in best:
            best[rid] = e
        else:
            if higher_is_better:
                if val > best[rid][key]:
                    best[rid] = e
            else:
                if val < best[rid][key]:
                    best[rid] = e
    return list(best.values())

def top_k(entries, key, k=5, higher_is_better=True):
    entries = [e for e in entries if e.get(key) is not None]
    return sorted(entries, key=lambda x: x[key], reverse=higher_is_better)[:k]

def build_leaderboards(jsonl_path: str, k=5):
    regression_entries = []
    classification_entries = []

    # Read all lines
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            run_id = obj.get("run_id", "")
            params = obj.get("params", {}) or {}
            artifact = obj.get("artifact_name")

            grp = model_group(run_id)

            # Regression metric-bearing records
            if is_regression_metric(params):
                regression_entries.append({
                    "run_id": run_id,
                    "group": grp,
                    "artifact": artifact,
                    "adjusted_r2": float(params["adjusted_r2"]),
                    "r2": params.get("r2"),
                    "mse": params.get("mse"),
                    "rmse": params.get("rmse"),
                    "mae": params.get("mae"),
                    "timestamp_unix": obj.get("timestamp_unix"),
                })

            # Classification metric-bearing records
            if is_classification_metric(params):
                classification_entries.append({
                    "run_id": run_id,
                    "group": grp,
                    "artifact": artifact,
                    "auc": float(params["auc"]),
                    "accuracy": params.get("accuracy"),
                    "f1": params.get("f1"),
                    "precision": params.get("precision"),
                    "recall": params.get("tpr"),   # your file uses tpr
                    "cross_entropy": params.get("cross_entropy"),
                    "timestamp_unix": obj.get("timestamp_unix"),
                })

    # Deduplicate per run_id (keep best metric per run)
    regression_unique = keep_best_per_run(regression_entries, "adjusted_r2", higher_is_better=True)
    classification_unique = keep_best_per_run(classification_entries, "auc", higher_is_better=True)

    # Overall top-k
    top_regression = top_k(regression_unique, "adjusted_r2", k=k, higher_is_better=True)
    top_classification = top_k(classification_unique, "auc", k=k, higher_is_better=True)

    # Family-specific top-k
    top_regression_family = top_k(
        [e for e in regression_unique if e["group"] == "regression"],
        "adjusted_r2", k=k, higher_is_better=True
    )

    # Decision trees: do both regression + classification
    top_tree_reg = top_k(
        [e for e in regression_unique if "RegressionDecisionTree" in e["run_id"]],
        "adjusted_r2", k=k, higher_is_better=True
    )
    top_tree_clf = top_k(
        [e for e in classification_unique if "DecisionTreeClassifier" in e["run_id"]],
        "auc", k=k, higher_is_better=True
    )

    top_neural = top_k(
        [e for e in classification_unique if e["group"] == "neural_network"],
        "auc", k=k, higher_is_better=True
    )

    return {
        "top_regression_overall": top_regression,
        "top_classification_overall": top_classification,
        "top_regression_family": top_regression_family,
        "top_decision_tree_regression": top_tree_reg,
        "top_decision_tree_classification": top_tree_clf,
        "top_neural_network": top_neural,
    }

def pretty_print(title, rows, metric_key):
    print(f"\n=== {title} ===")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r['run_id']} | {metric_key}={r[metric_key]} | artifact={r['artifact']}")

# ---- run it ----
leaderboards = build_leaderboards("data/runs.jsonl", k=10)

pretty_print("TOP 5 REGRESSION (Adjusted R2)", leaderboards["top_regression_overall"], "adjusted_r2")
pretty_print("TOP 5 CLASSIFICATION (AUC)", leaderboards["top_classification_overall"], "auc")

pretty_print("TOP 5 REGRESSION FAMILY (Linear + Polynomial)", leaderboards["top_regression_family"], "adjusted_r2")

pretty_print("TOP 5 DECISION TREE (Regression)", leaderboards["top_decision_tree_regression"], "adjusted_r2")
pretty_print("TOP 5 DECISION TREE (Classification)", leaderboards["top_decision_tree_classification"], "auc")

pretty_print("TOP 5 NEURAL NETWORK (Classification)", leaderboards["top_neural_network"], "auc")
