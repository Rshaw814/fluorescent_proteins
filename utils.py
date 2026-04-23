import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import TARGET_NAMES, CLASSIFICATION_TASKS, CLASSIFICATION_DIMS, ERROR_BINS
from helper_func.biophysical_properties import compute_biophysical_properties

# ---------------------------------------------------------------------
# Class vocab (IDs are fixed by order). We use the CSV short codes.
# ---------------------------------------------------------------------
CLASS_VOCAB = {
    "oligomerization": ["m", "d", "td", "o", "wd"],
    "switch_type":     ["b", "pa", "ps", "pc", "o"],
    "maturation":      ["fast", "medium", "slow"],   # binned from minutes
    "lifetime":        ["short", "medium", "long"],  # binned from ns
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _encode_with_vocab_from_strings(series: pd.Series, vocab: list[str]) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    cls2id = {c: i for i, c in enumerate(vocab)}
    return s.map(cls2id).fillna(-1).astype(np.int64).to_numpy()

def _bin_maturation_minutes_to_labels(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    out[x < 30] = "fast"
    out[(x >= 30) & (x <= 60)] = "medium"
    out[x > 60] = "slow"
    return out

def _bin_lifetime_ns_to_labels(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    out[x < 2.0] = "short"
    out[(x >= 2.0) & (x <= 3.0)] = "medium"
    out[x > 3.0] = "long"
    return out

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def load_data(csv_file, error_label_targets=None, y_pred_cols=None):
    print(f"📂 Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)

    # Optionally add derived biophysical features
    df = add_biophysical_targets(df)

    # --- embeddings -> X ---
    df["emb"] = df["embeddings"].apply(ast.literal_eval)
    X = np.vstack(df["emb"].values)

    # --- continuous targets ---
    y = df[TARGET_NAMES].values
    mask = ~np.isnan(y)
    print(f"✅ Loaded {len(df)} samples with {mask.sum()} non-NaN targets across {len(TARGET_NAMES)} tasks")

    # --- classification targets (exact header names) ---
    class_targets = {}

    # 1) oligomerization (short codes: m/d/td/o)
    if "oligomerization" in df.columns:
        oligo_arr = _encode_with_vocab_from_strings(df["oligomerization"], CLASS_VOCAB["oligomerization"])
        class_targets["oligomerization"] = oligo_arr
        print(f"• oligomerization: {(oligo_arr >= 0).sum()} labeled / {len(oligo_arr)} total")

    # 2) switch type (short codes: b/pa/ps/pc)
    if "switch type" in df.columns:
        switch_arr = _encode_with_vocab_from_strings(df["switch type"], CLASS_VOCAB["switch_type"])
        class_targets["switch_type"] = switch_arr
        print(f"• switch_type: {(switch_arr >= 0).sum()} labeled / {len(switch_arr)} total")

    # 3) maturation (min) -> bins -> ids
    if "maturation (min)" in df.columns:
        mat_labels = _bin_maturation_minutes_to_labels(df["maturation (min)"])
        mat_arr = _encode_with_vocab_from_strings(mat_labels.fillna(""), CLASS_VOCAB["maturation"])
        class_targets["maturation"] = mat_arr
        print(f"• maturation: {(mat_arr >= 0).sum()} labeled / {len(mat_arr)} total")

    # 4) lifetime (ns) -> bins -> ids
    if "lifetime (ns)" in df.columns:
        life_labels = _bin_lifetime_ns_to_labels(df["lifetime (ns)"])
        life_arr = _encode_with_vocab_from_strings(life_labels.fillna(""), CLASS_VOCAB["lifetime"])
        class_targets["lifetime"] = life_arr
        print(f"• lifetime: {(life_arr >= 0).sum()} labeled / {len(life_arr)} total")

    # (Optional) multi-class error labels support could go here; return None by default
    error_class_targets = None

    return X, y, mask, class_targets, error_class_targets

def make_loader(X, y, mask, batch_size=64, class_targets=None, error_class_targets=None):
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    mask_tensor = torch.from_numpy(mask).bool()

    tensors = [X_tensor, y_tensor, mask_tensor]

    if class_targets is not None:
        # append in config order, but only tasks that exist
        for task_name in [t for t in CLASSIFICATION_TASKS if t in class_targets]:
            arr = class_targets[task_name]
            assert len(arr) == len(X), f"class target '{task_name}' has wrong length"
            tensors.append(torch.from_numpy(arr).long())

    if error_class_targets is not None:
        assert len(error_class_targets) == len(X), "error_class_targets has wrong length"
        tensors.append(torch.from_numpy(error_class_targets).long())

    for t in tensors:
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == X.shape[0]

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_biophysical_targets(df):
    metrics = df['sequence'].apply(compute_biophysical_properties)
    metric_df = pd.DataFrame(metrics.tolist())
    return pd.concat([df, metric_df], axis=1)

# ---------------------------------------------------------------------
# Optional utilities you were already using elsewhere
# ---------------------------------------------------------------------
def bin_errors(errors, edges):
    return np.digitize(errors, edges, right=False).astype(np.int64)

def build_label_matrix(y_true, y_pred):
    labels = np.full_like(y_true, fill_value=-1, dtype=np.int64)
    for idx, task in enumerate(TARGET_NAMES):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        m = ~np.isnan(true)
        abs_err = np.abs(pred[m] - true[m])
        labels[m, idx] = bin_errors(abs_err, ERROR_BINS[task])
    return labels

def batched_features(model, feature_fn, X_np, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).float().to(device)
            feats = feature_fn(xb).detach().cpu().numpy()
            out.append(feats)
    return np.concatenate(out, axis=0)