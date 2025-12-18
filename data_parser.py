import os
import json
import time
import uuid
import hashlib
from typing import Any, Dict, Optional, Sequence, Union, Callable, List
import numpy as np
import csv
import copy
import types

def parse_data(input_path, data_fields, target_fields, string_fields):
    X = []
    Y = []

    f = open(input_path)
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        x_row = []
        for i in range(len(row)):
            if i in data_fields:
                continue

            if i not in string_fields:
                row[i] = float(row[i])
            
            if i in target_fields:
                Y.append(row[i])
            else:
                x_row.append(row[i])
        
        X.append(x_row)
    
    f.close()

    split = int(len(X) * 0.7)
    X_train = X[:split]
    Y_train = Y[:split]

    X_test = X[split:]
    Y_test = Y[split:]

    return X_train, Y_train, X_test, Y_test

def prepare_X(X):
    X_copy = copy.deepcopy(X)

    for i in range(len(X_copy)):
        X_copy[i].insert(0, 1)
    
    return X_copy

def _jsonable(x: Any) -> Any:
    """Convert common numpy/python objects into JSON-serializable types."""
    if x.__class__.__name__ == "_TreeNode":
        return str(x)
    
    if isinstance(x, np.random.Generator):
        return _jsonable(x.bit_generator.state)
    if isinstance(x, types.GeneratorType):
        return [_jsonable(v) for v in list(x)]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

def _make_run_id(params: Dict[str, Any], prefix: str = "run") -> str:
    """Deterministic-ish run id: timestamp + hash(params) + short uuid."""
    params_str = json.dumps(_jsonable(params), sort_keys=True)
    h = hashlib.md5(params_str.encode("utf-8")).hexdigest()[:10]
    u = uuid.uuid4().hex[:6]
    ts = int(time.time())
    return f"{prefix}_{ts}_{h}_{u}"

def save_npz_single(
    out_dir: str,
    values: Union[Sequence[float], np.ndarray],
    run_id: str,
    key: str = "data"
) -> str:
    """
    Saves ONLY the provided list/array into a compressed .npz file.

    The .npz will contain exactly ONE array with name == `key`.
    Returns the npz file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    arr = np.asarray(values)
    npz_path = os.path.join(out_dir, f"{run_id}.npz")

    # Exactly one array stored:
    np.savez_compressed(npz_path, **{key: arr})
    return npz_path

def append_run_metadata(
    out_dir: str,
    run_id: str,
    npz_path: str,
    params: Dict[str, Any],
    artifact_name: str,
    key_in_npz: str = "data",
    extra: Optional[Dict[str, Any]] = None,
    jsonl_name: str = "runs.jsonl",
) -> str:
    """
    Appends one JSON line with metadata to out_dir/jsonl_name.

    - params: all hyperparams / settings you want to record
    - artifact_name: what the stored list represents ("theta", "y_pred", etc.)
    - extra: optional extra fields (metrics, notes, etc.)
    Returns the JSONL path.
    """
    os.makedirs(out_dir, exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp_unix": int(time.time()),
        "artifact_name": str(artifact_name),
        "npz_path": str(npz_path),
        "npz_key": str(key_in_npz),
        "params": _jsonable(params),
    }

    if extra:
        record["extra"] = _jsonable(extra)

    jsonl_path = os.path.join(out_dir, jsonl_name)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return jsonl_path

def save_run_artifact(
    out_dir: str,
    values: Union[Sequence[float], np.ndarray],
    params: Dict[str, Any],
    artifact_name: str,
    key_in_npz: str = "data",
    extra: Optional[Dict[str, Any]] = None,
    run_id_prefix: str = "run",
) -> Dict[str, str]:
    """
    One-call convenience:
      1) create run_id
      2) save ONLY `values` to .npz
      3) append metadata row to JSONL

    Returns paths/ids you might want to print/log.
    """
    run_id = _make_run_id(params, prefix=run_id_prefix)
    npz_path = save_npz_single(out_dir, values, run_id=run_id, key=key_in_npz)
    jsonl_path = append_run_metadata(
        out_dir=out_dir,
        run_id=run_id,
        npz_path=npz_path,
        params=params,
        artifact_name=artifact_name,
        key_in_npz=key_in_npz,
        extra=extra,
    )
    return {"run_id": run_id, "npz_path": npz_path, "jsonl_path": jsonl_path}

# ============================================================
# Save/Load ONLY the tree structure rooted at `root` (self._root)
# Stores the ENTIRE tree (splits, thresholds, children, leaves, counts)
# No pickle; allow_pickle=False friendly.
# ============================================================

def _tree_to_dict(root: Any) -> Dict[str, Any]:
    """Serialize _TreeNode graph -> pure python dict (JSONable)."""
    def rec(node):
        if node is None:
            return None

        # children is a dict for categorical splits. Keys may be non-JSON types.
        # Store as list of (key, child) pairs.
        children_items = []
        if getattr(node, "children", None):
            for k, child in node.children.items():
                children_items.append([_jsonable(k), rec(child)])

        return {
            "is_leaf": bool(node.is_leaf),
            "prediction": _jsonable(node.prediction),
            "feature_index": _jsonable(node.feature_index),
            "threshold": _jsonable(node.threshold),
            "n_samples": int(getattr(node, "n_samples", 0)),
            "class_counts": _jsonable(getattr(node, "class_counts", None)),
            "left": rec(getattr(node, "left", None)),
            "right": rec(getattr(node, "right", None)),
            "children_items": children_items,  # list of [key, child_dict]
        }

    if root is None:
        raise ValueError("root is None (tree not fitted?)")
    return {"artifact_type": "tree_root", "root": rec(root)}

def _dict_to_tree(tree_payload: Dict[str, Any]) -> Any:
    """Deserialize dict -> _TreeNode graph."""
    if tree_payload.get("artifact_type") != "tree_root":
        raise ValueError("Not a tree_root payload.")

    def rec(d):
        if d is None:
            return None

        node = _TreeNode(
            is_leaf=bool(d.get("is_leaf", False)),
            prediction=d.get("prediction", None),
            feature_index=d.get("feature_index", None),
            threshold=d.get("threshold", None),
            left=None,
            right=None,
            children={},
            n_samples=int(d.get("n_samples", 0)),
            class_counts=d.get("class_counts", None),
        )

        node.left = rec(d.get("left", None))
        node.right = rec(d.get("right", None))

        for pair in d.get("children_items", []) or []:
            # pair = [key, child_dict]
            k = pair[0]
            child = rec(pair[1])
            node.children[k] = child

        return node

    return rec(tree_payload["root"])

def save_run_tree(
    out_dir: str,
    root: Any,                               # a single root OR list/tuple of roots
    params: Dict[str, Any],
    artifact_name: str = "tree_root",
    key_in_npz: str = "data",
    extra: Optional[Dict[str, Any]] = None,
    run_id_prefix: str = "run_Tree",
) -> Dict[str, str]:
    """
    Serializes and stores:
      - a single ENTIRE tree rooted at `root`, OR
      - a forest (list/tuple of roots)

    Saves to NPZ as ONE unicode JSON string array (no pickle).
    Also appends metadata in runs.jsonl (same style as save_run_artifact).
    """
    # --- NEW: allow list/tuple of roots (forest) ---
    if isinstance(root, (list, tuple)):
        payload = {
            "artifact_type": "tree_forest",
            "n_trees": len(root),
            "trees": [_tree_to_dict(r) for r in root],
        }
        if artifact_name == "tree_root":
            artifact_name = "tree_forest"
    else:
        payload = _tree_to_dict(root)

    json_str = json.dumps(_jsonable(payload), sort_keys=True)

    # store as a single-element unicode array (works with allow_pickle=False)
    arr = np.array([json_str], dtype=np.str_)

    if extra is None:
        extra = {}
    extra = dict(extra)
    extra["stored_as"] = "tree_json_unicode"
    extra["artifact_type"] = payload.get("artifact_type", "tree_root")

    run_id = _make_run_id(params, prefix=run_id_prefix)
    npz_path = save_npz_single(out_dir, arr, run_id=run_id, key=key_in_npz)
    jsonl_path = append_run_metadata(
        out_dir=out_dir,
        run_id=run_id,
        npz_path=npz_path,
        params=params,
        artifact_name=artifact_name,
        key_in_npz=key_in_npz,
        extra=extra,
    )
    return {"run_id": run_id, "npz_path": npz_path, "jsonl_path": jsonl_path}

def load_run_tree(
    npz_path: str,
    key: str = "data",
) -> Any:
    """
    Loads the stored tree payload from NPZ and returns:
      - a reconstructed _TreeNode root (single tree), OR
      - a list of _TreeNode roots (forest)
    """
    arr = load_npz_artifact(npz_path, key=key)  # allow_pickle=False inside
    if arr.size != 1:
        raise ValueError("Expected a single JSON string element in NPZ.")

    payload = json.loads(str(arr.reshape(-1)[0]))

    # forest case (ensemble)
    if isinstance(payload, dict) and payload.get("artifact_type") == "tree_forest":
        trees = payload.get("trees", [])
        return [_dict_to_tree(t) for t in trees]

    # single-tree case (backward compatible)
    return _dict_to_tree(payload)

def read_tree_run(
    out_dir: str,
    run_id: str,
    jsonl_name: str = "runs.jsonl",
) -> Dict[str, Any]:
    """
    Find metadata record by run_id and return:
      - meta
      - root: reconstructed _TreeNode root
    """
    records = read_metadata_jsonl(
        out_dir,
        jsonl_name=jsonl_name,
        predicate=lambda r: r.get("run_id") == run_id
    )
    if not records:
        raise ValueError(f"run_id '{run_id}' not found in {os.path.join(out_dir, jsonl_name)}")

    meta = records[0]
    root = load_run_tree(meta["npz_path"], key=meta.get("npz_key", "data"))
    return {"meta": meta, "root": root}

# ============================================================
# Save/Load Neural Network state (MLP / RBFMLP / CNN / RNN)
# Stores as ONE unicode JSON string array (no pickle).
# Requires: model instance already created for loading (same architecture).
# ============================================================

def _nn_to_dict(model: Any) -> Dict[str, Any]:
    cls = model.__class__.__name__

    payload = {
        "artifact_type": "nn_state",
        "class_name": cls,
        "state": {},
    }

    st = payload["state"]

    # ---------- MLP family ----------
    if cls in ("MLPRegressor", "MLPClassifier"):
        st["W"] = [w for w in getattr(model, "W", [])]
        st["b"] = [b for b in getattr(model, "b", [])]
        st["W_out"] = getattr(model, "W_out", None)
        st["b_out"] = getattr(model, "b_out", None)

    elif cls == "MultiTaskMLP":
        st["W"] = [w for w in getattr(model, "W", [])]
        st["b"] = [b for b in getattr(model, "b", [])]
        st["W_reg"] = getattr(model, "W_reg", None)
        st["b_reg"] = getattr(model, "b_reg", None)
        st["W_clf"] = getattr(model, "W_clf", None)
        st["b_clf"] = getattr(model, "b_clf", None)

    # ---------- RBF MLP family ----------
    elif cls in ("RBFMLPRegressor", "RBFMLPClassifier"):
        st["centers_"] = getattr(model, "centers_", None)
        st["gamma_"] = getattr(model, "gamma_", None)
        st["W_out"] = getattr(model, "W_out", None)
        st["b_out"] = getattr(model, "b_out", None)

    # ---------- CNN family ----------
    elif cls in ("CNNRegressor", "CNNClassifier"):
        # conv stack
        convs = getattr(model, "convs", [])
        st["convs"] = [{"W": c.W, "b": c.b} for c in convs]
        st["W_out"] = getattr(model, "W_out", None)
        st["b_out"] = getattr(model, "b_out", None)

    elif cls == "MultiTaskCNN":
        convs = getattr(model, "convs", [])
        st["convs"] = [{"W": c.W, "b": c.b} for c in convs]
        st["W_reg"] = getattr(model, "W_reg", None)
        st["b_reg"] = getattr(model, "b_reg", None)
        st["W_clf"] = getattr(model, "W_clf", None)
        st["b_clf"] = getattr(model, "b_clf", None)

    # ---------- RNN family ----------
    elif cls in ("RNNRegressor", "RNNClassifier"):
        rnn = getattr(model, "rnn", None)
        if rnn is None:
            raise ValueError("Model has no .rnn to save.")
        st["rnn"] = {"Wx": rnn.Wx, "Wh": rnn.Wh, "b": rnn.b}
        st["W_out"] = getattr(model, "W_out", None)
        st["b_out"] = getattr(model, "b_out", None)

    elif cls == "MultiTaskRNN":
        rnn = getattr(model, "rnn", None)
        if rnn is None:
            raise ValueError("Model has no .rnn to save.")
        st["rnn"] = {"Wx": rnn.Wx, "Wh": rnn.Wh, "b": rnn.b}
        st["W_reg"] = getattr(model, "W_reg", None)
        st["b_reg"] = getattr(model, "b_reg", None)
        st["W_clf"] = getattr(model, "W_clf", None)
        st["b_clf"] = getattr(model, "b_clf", None)

    else:
        raise ValueError(f"Unsupported NN class for saving: {cls}")

    # Make JSON-friendly (numpy -> lists, etc.)
    return _jsonable(payload)

def _dict_to_nn(model: Any, payload: Dict[str, Any]) -> Any:
    if payload.get("artifact_type") != "nn_state":
        raise ValueError("Not an nn_state payload.")
    cls_saved = payload.get("class_name")
    cls_now = model.__class__.__name__
    if cls_saved != cls_now:
        raise ValueError(f"Class mismatch: saved={cls_saved} current={cls_now}")

    st = payload["state"]

    # helper: list->np.array
    def arr(x):
        return None if x is None else np.asarray(x, dtype=float)

    if cls_now in ("MLPRegressor", "MLPClassifier"):
        model.W = [arr(w) for w in st.get("W", [])]
        model.b = [arr(b) for b in st.get("b", [])]
        model.W_out = arr(st.get("W_out"))
        model.b_out = arr(st.get("b_out"))

    elif cls_now == "MultiTaskMLP":
        model.W = [arr(w) for w in st.get("W", [])]
        model.b = [arr(b) for b in st.get("b", [])]
        model.W_reg = arr(st.get("W_reg"))
        model.b_reg = arr(st.get("b_reg"))
        model.W_clf = arr(st.get("W_clf"))
        model.b_clf = arr(st.get("b_clf"))

    elif cls_now in ("RBFMLPRegressor", "RBFMLPClassifier"):
        model.centers_ = arr(st.get("centers_"))
        g = st.get("gamma_")
        model.gamma_ = None if g is None else float(g)
        model.W_out = arr(st.get("W_out"))
        model.b_out = arr(st.get("b_out"))

    elif cls_now in ("CNNRegressor", "CNNClassifier", "MultiTaskCNN"):
        convs_saved = st.get("convs", [])
        if len(convs_saved) != len(getattr(model, "convs", [])):
            raise ValueError("Conv layer count mismatch (saved vs current model).")

        for c_obj, c_sd in zip(model.convs, convs_saved):
            c_obj.W = arr(c_sd["W"])
            c_obj.b = arr(c_sd["b"])

        if cls_now in ("CNNRegressor", "CNNClassifier"):
            model.W_out = arr(st.get("W_out"))
            model.b_out = arr(st.get("b_out"))
        else:
            model.W_reg = arr(st.get("W_reg"))
            model.b_reg = arr(st.get("b_reg"))
            model.W_clf = arr(st.get("W_clf"))
            model.b_clf = arr(st.get("b_clf"))

    elif cls_now in ("RNNRegressor", "RNNClassifier", "MultiTaskRNN"):
        rnn_sd = st.get("rnn", None)
        if rnn_sd is None:
            raise ValueError("Saved state missing rnn.")
        model.rnn.Wx = arr(rnn_sd["Wx"])
        model.rnn.Wh = arr(rnn_sd["Wh"])
        model.rnn.b  = arr(rnn_sd["b"])

        if cls_now in ("RNNRegressor", "RNNClassifier"):
            model.W_out = arr(st.get("W_out"))
            model.b_out = arr(st.get("b_out"))
        else:
            model.W_reg = arr(st.get("W_reg"))
            model.b_reg = arr(st.get("b_reg"))
            model.W_clf = arr(st.get("W_clf"))
            model.b_clf = arr(st.get("b_clf"))

    return model

def save_run_nn(
    out_dir: str,
    model: Any,
    params: Dict[str, Any],
    artifact_name: str = "nn_state",
    key_in_npz: str = "data",
    extra: Optional[Dict[str, Any]] = None,
    run_id_prefix: str = "run_NN",
) -> Dict[str, str]:
    payload = _nn_to_dict(model)
    json_str = json.dumps(payload, sort_keys=True)

    arr = np.array([json_str], dtype=np.str_)

    if extra is None:
        extra = {}
    extra = dict(extra)
    extra["stored_as"] = "nn_json_unicode"
    extra["class_name"] = model.__class__.__name__

    run_id = _make_run_id(params, prefix=run_id_prefix)
    npz_path = save_npz_single(out_dir, arr, run_id=run_id, key=key_in_npz)
    jsonl_path = append_run_metadata(
        out_dir=out_dir,
        run_id=run_id,
        npz_path=npz_path,
        params=params,
        artifact_name=artifact_name,
        key_in_npz=key_in_npz,
        extra=extra,
    )
    return {"run_id": run_id, "npz_path": npz_path, "jsonl_path": jsonl_path}

def load_run_nn(
    model: Any,
    npz_path: str,
    key: str = "data",
) -> Any:
    arr = load_npz_artifact(npz_path, key=key)
    if arr.size != 1:
        raise ValueError("Expected a single JSON string element in NPZ.")
    payload = json.loads(str(arr.reshape(-1)[0]))
    return _dict_to_nn(model, payload)

def read_metadata_jsonl(
    out_dir: str,
    jsonl_name: str = "runs.jsonl",
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Read all metadata records from out_dir/jsonl_name.

    predicate(record) -> True keeps the record (optional filter).
    Returns: list of dict records.
    """
    path = os.path.join(out_dir, jsonl_name)
    records: List[Dict[str, Any]] = []

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # skip corrupted line rather than crashing
                continue
            if predicate is None or predicate(rec):
                records.append(rec)

    return records

def load_npz_artifact(
    npz_path: str,
    key: str = "data",
) -> np.ndarray:
    """
    Load the single stored array from an .npz file by key.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        if key not in z:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available: {list(z.keys())}")
        return z[key]

def read_run(
    out_dir: str,
    run_id: str,
    jsonl_name: str = "runs.jsonl",
) -> Dict[str, Any]:
    """
    Find the first record with matching run_id, then load its stored artifact.
    Returns a dict with:
      - 'meta': metadata record
      - 'data': np.ndarray loaded from the referenced npz file
    """
    records = read_metadata_jsonl(
        out_dir,
        jsonl_name=jsonl_name,
        predicate=lambda r: r.get("run_id") == run_id
    )
    if not records:
        raise ValueError(f"run_id '{run_id}' not found in {os.path.join(out_dir, jsonl_name)}")

    meta = records[0]
    npz_path = meta["npz_path"]
    key = meta.get("npz_key", "data")

    data = load_npz_artifact(npz_path, key=key)
    return {"meta": meta, "data": data}
