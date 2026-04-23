import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import torch
from models import mlpCalibrator
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import os
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def _to_numpy(x):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def fit_linear_calibrators_from_layer(model, layer_name, X_tensor, y, mask, target_names, eval_data=None):
    """
    Fit linear calibrators using features extracted from a specific layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y = y.copy()
    mask = mask.copy()

    if eval_data is not None:
        x_eval_tensor, y_eval, mask_eval = eval_data
    else:
        x_eval_tensor, y_eval, mask_eval = X_tensor, y, mask

    calibrators = []
    preds = np.full_like(y_eval, np.nan)

    for i, target in enumerate(target_names):
        print(f"🔍 Fitting linear calibrator for target: {target}")
        target_mask = mask[:, i] & ~np.isnan(y[:, i])
        if target_mask.sum() < 5:
            calibrators.append(None)
            continue

        # === Extract task-specific hidden features ===
        feats = extract_features_from_layer(model, f"{layer_name}.{target}.1", X_tensor)
        feats_eval = extract_features_from_layer(model, f"{layer_name}.{target}.1", x_eval_tensor)

        X_train = feats[target_mask]
        y_train = y[target_mask, i]

        # === Train and optionally evaluate ===
        linreg = LinearRegression().fit(X_train, y_train)
        calibrators.append(linreg)

        val_mask = mask_eval[:, i] & ~np.isnan(y_eval[:, i])
        if val_mask.sum() > 0:
            X_val = feats_eval[val_mask]
            y_val = y_eval[val_mask, i]
            y_pred = linreg.predict(X_val)
            preds[val_mask, i] = y_pred

            r2_val = r2_score(y_val, y_pred)
            print(f"✅ R² (validation): {r2_val:.3f}")

    return calibrators, preds
    
import torch
import numpy as np

def extract_layer_features(model, layer_name, X_np, device, batch_size=512):
    """
    model: torch.nn.Module (already loaded with weights)
    layer_name: str, e.g. 'encoder.layers.11' or 'mlp.2'
    X_np: (N, D) numpy features (PLM embeddings in your case)
    returns: (N, H) numpy activations captured from the named layer
    """
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found. Available: {list(modules.keys())[:10]} ...")

    feats = []
    buf = []

    def hook_fn(_, __, output):
        # output: Tensor [B, H,...]; flatten batch feat dimension
        buf.append(output.detach().cpu())

    handle = modules[layer_name].register_forward_hook(hook_fn)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).float().to(device)
            _ = model(xb)  # forward just to trigger hook
            if len(buf) == 0:
                raise RuntimeError(f"No output captured from layer {layer_name}")
            feats.append(buf[0])
            buf.clear()
    handle.remove()

    feats = torch.cat(feats, dim=0)
    # If layer outputs >2D (e.g., [B, H, L]), pool or flatten; here we flatten last dims
    if feats.dim() > 2:
        feats = feats.flatten(start_dim=1)
    return feats.numpy()

def fit_isotonic_calibrators(score_preds, true_vals, mask):
    """
    Calibrate predictions using isotonic regression per target.
    Only uses values where mask == 1.
    """
    calibrators = []
    calibrated_preds = np.full_like(score_preds, np.nan)
    r2s, rhos = [], []
    score_preds = _to_numpy(score_preds)
    true_vals   = _to_numpy(true_vals)
    mask        = np.asarray(mask).astype(bool)

    for i in range(score_preds.shape[1]):
        x_i = score_preds[:, i]
        y_i = true_vals[:, i]
        m_i = mask[:, i].astype(bool)

        # Skip if too few unique values in masked subset
        if m_i.sum() < 2 or len(np.unique(x_i[m_i])) < 2 or len(np.unique(y_i[m_i])) < 2:
            calibrators.append(None)
            r2s.append(np.nan)
            rhos.append(np.nan)
            continue

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(x_i[m_i], y_i[m_i])
        calibrated_preds[:, i] = iso.predict(x_i)
        calibrators.append(iso)

        r2s.append(r2_score(y_i[m_i], calibrated_preds[m_i, i]))
        rho, _ = spearmanr(y_i[m_i], calibrated_preds[m_i, i])
        rhos.append(rho)

    return calibrators, calibrated_preds, r2s, rhos
    
    
def fit_isotonic_calibrators_multihead(score_preds, true_vals, mask, agg="median", calibration_data=None):
    """
    Calibrate multi-head predictions using isotonic regression per target and per head.

    Parameters:
    - score_preds: (N, T, H) predictions to be calibrated
    - true_vals: (N, T) true values corresponding to score_preds
    - mask: (N, T) binary mask of valid predictions
    - agg: "mean" or "median" to aggregate across heads
    - calibration_data: tuple (train_preds, train_true, train_mask), all np.ndarrays
                        If provided, fit calibrators on this instead.

    Returns:
    - calibrators: list of list of IsotonicRegression models (or None)
    - preds_mean: calibrated and aggregated predictions (N, T)
    - preds_all: all calibrated headwise predictions (N, T, H)
    - r2s: R² scores per target
    - rhos: Spearman ρ per target
    """
    n_samples, n_targets, n_heads = score_preds.shape
    preds_all = np.full((n_samples, n_targets, n_heads), np.nan)
    calibrators = []
    score_preds = _to_numpy(score_preds)     # [N, T, H]
    true_vals   = _to_numpy(true_vals)       # [N, T]
    mask        = np.asarray(mask).astype(bool)

    if calibration_data is not None:
        cal_preds, cal_true, cal_mask = calibration_data
        cal_preds = _to_numpy(cal_preds)
        cal_true  = _to_numpy(cal_true)
        cal_mask  = np.asarray(cal_mask).astype(bool)
    else:
        cal_preds, cal_true, cal_mask = score_preds, true_vals, mask

    for t in range(n_targets):
        target_cals = []

        for h in range(n_heads):
            preds_fit = cal_preds[:, t, h]
            true_fit = cal_true[:, t]
            m_fit = cal_mask[:, t].astype(bool)

            # Further restrict to samples where preds_fit is not NaN
            valid_idx = m_fit & ~np.isnan(preds_fit) & ~np.isnan(true_fit)

        if np.sum(valid_idx) < 5:
            preds_to_calibrate = score_preds[:, t, h]
            valid_pred_idx = ~np.isnan(preds_to_calibrate)
            preds_all[valid_pred_idx, t, h] = preds_to_calibrate[valid_pred_idx]
            target_cals.append(None)
            continue

            model = IsotonicRegression(out_of_bounds='clip')
            try:
                model.fit(preds_fit[valid_idx], true_fit[valid_idx])

                # Predict only where input to predict is not NaN
                preds_to_calibrate = score_preds[:, t, h]
                valid_pred_idx = ~np.isnan(preds_to_calibrate)
                preds_all[valid_pred_idx, t, h] = model.predict(preds_to_calibrate[valid_pred_idx])

                target_cals.append(model)
            except Exception as e:
                print(f"❌ Failed calibration for target {t}, head {h}: {e}")
                target_cals.append(None)

        calibrators.append(target_cals)

    # Aggregate calibrated predictions across heads
    if agg == "mean":
        preds_mean = np.nanmean(preds_all, axis=2)
    elif agg == "median":
        preds_mean = np.nanmedian(preds_all, axis=2)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    # Compute R² and Spearman ρ for each target
    r2s, rhos = [], []
    for t in range(n_targets):
        m = mask[:, t].astype(bool)
        true_vals_t = true_vals[m, t]
        preds_vals_t = preds_mean[m, t]

        nan_mask = ~np.isnan(true_vals_t) & ~np.isnan(preds_vals_t)

        if np.sum(nan_mask) >= 2:
            r2s.append(r2_score(true_vals_t[nan_mask], preds_vals_t[nan_mask]))
            rhos.append(spearmanr(true_vals_t[nan_mask], preds_vals_t[nan_mask])[0])
        else:
            r2s.append(np.nan)
            rhos.append(np.nan)

    return calibrators, preds_mean, preds_all, r2s, rhos
    
    
def compute_stoke_shift_metrics(true_vals, pred_vals, mask=None):
    true_ex = true_vals[:, 0]
    true_em = true_vals[:, 1]
    pred_ex = pred_vals[:, 0]
    pred_em = pred_vals[:, 1]

    if mask is not None:
        valid = (mask[:, 0] == 1) & (mask[:, 1] == 1)
    else:
        valid = ~np.isnan(true_ex) & ~np.isnan(true_em)

    # Additional check: exclude rows where em or ex is zero in true values
    valid &= (true_em != 0) & (true_ex != 0)

    if valid.sum() < 2:
        return np.nan, np.nan

    true_shift = true_em[valid] - true_ex[valid]
    pred_shift = pred_em[valid] - pred_ex[valid]

    rho, _ = spearmanr(true_shift, pred_shift)
    r2 = r2_score(true_shift, pred_shift)
    return rho, r2

def apply_calibrators(calibrators, raw_preds):
    """
    Args:
        calibrators: list of calibrators (torch.nn.Module, sklearn, callable, or None)
        raw_preds: numpy array of shape (N, T, H) where:
                   N = number of samples,
                   T = number of targets,
                   H = number of output heads (e.g. 10)
    Returns:
        Calibrated predictions of shape (N, T)
    """
    # 🔍 Step 1: Validate input shape
    if raw_preds.ndim != 3:
        raise ValueError(f"Expected raw_preds to have shape (N, T, H), got {raw_preds.shape}")

    N, T, H = raw_preds.shape
    if T != len(calibrators):
        print(f"⚠️ Warning: Number of calibrators ({len(calibrators)}) does not match number of targets ({T})")

    # 🔎 Step 2: Take median across output heads
    medians = np.median(raw_preds, axis=2)  # shape (N, T)

    # 🧪 Step 3: Apply per-target calibration
    calibrated = []

    for i in range(T):
        x_raw = medians[:, i].reshape(-1, 1)  # (N, 1)
        cal = calibrators[i] if i < len(calibrators) else None

        if cal is None:
            print(f"⚠️ No calibrator for target {i}, using raw predictions.")
            calibrated.append(x_raw.flatten())
        elif hasattr(cal, "predict"):
            calibrated.append(cal.predict(x_raw))
        elif isinstance(cal, torch.nn.Module):
            with torch.no_grad():
                input_tensor = torch.tensor(x_raw, dtype=torch.float32)
                input_tensor = input_tensor.to(next(cal.parameters()).device)
                cal.eval()
                preds = cal(input_tensor).cpu().numpy().flatten()
                calibrated.append(preds)
        elif callable(cal):
            calibrated.append(cal(x_raw.flatten()))
        else:
            raise ValueError(f"❌ Unsupported calibrator type at index {i}: {type(cal)}")

    return np.stack(calibrated, axis=1)  # shape (N, T)
    



def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d
    
def fit_sigmoid_calibrator(x_raw, y_true):
    """
    Fits a 4-parameter logistic (sigmoid): y = a/(1+exp(-b*(x-c))) + d
    Returns (cal_fn, params)
    """
    import numpy as np
    x = _to_numpy(x_raw).reshape(-1)
    y = _to_numpy(y_true).reshape(-1)

    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 5:
        def cal_fn(z):
            z = _to_numpy(z)
            out = np.full_like(z, np.nan, dtype=float)
            out[~np.isnan(z)] = z[~np.isnan(z)]
            return out
        return cal_fn, None

    x = x[valid]; y = y[valid]

    # initial params
    a0 = np.nanmax(y) - np.nanmin(y)
    b0 = 1.0
    c0 = np.nanmedian(x)
    d0 = np.nanmin(y)
    p0 = np.array([a0, b0, c0, d0], dtype=float)

    from scipy.optimize import least_squares

    def model(p, x):
        a, b, c, d = p
        return a / (1.0 + np.exp(-b * (x - c))) + d

    def resid(p):
        return model(p, x) - y

    try:
        res = least_squares(resid, p0, max_nfev=5000)
        p = res.x
    except Exception:
        p = p0

    def cal_fn(z):
        z = _to_numpy(z)
        return model(p, z)

    return cal_fn, p

def fit_sigmoid_calibrators(x_raw_all, y_true_all, mask_all):
    """
    Fits a sigmoid calibration function for each output.

    Parameters:
    - x_raw_all: (N, T) raw predictions
    - y_true_all: (N, T) true values
    - mask_all: (N, T) mask indicating valid points

    Returns:
    - calibrator_fns: list of callable calibrator functions
    - calibrated_preds: (N, T) calibrated predictions
    """
    
    x_raw_all  = _to_numpy(x_raw_all)   # [N, T]
    y_true_all = _to_numpy(y_true_all)  # [N, T]
    mask_all   = np.asarray(mask_all).astype(bool)
    T = x_raw_all.shape[1]
    calibrator_fns = []
    calibrated_preds = np.zeros_like(x_raw_all)

    for t in range(T):
        x_raw = x_raw_all[:, t]
        y_true = y_true_all[:, t]
        mask = mask_all[:, t]

        valid = mask & ~np.isnan(x_raw) & ~np.isnan(y_true)

        if valid.sum() < 5:
            # Not enough data, fallback to identity
            calibrator_fns.append(lambda x: x)
            calibrated_preds[:, t] = x_raw
            continue

        cal_fn, _ = fit_sigmoid_calibrator(x_raw[valid], y_true[valid])
        calibrator_fns.append(cal_fn)
        calibrated_preds[:, t] = cal_fn(x_raw)

    return calibrator_fns, calibrated_preds
    
def five_pl(x, a, b, c, d, g):
    return d + (a - d) / ((1 + np.exp(-np.clip(b * (x - c), -60, 60)))**g)

def fit_fivepl_calibrator(x_raw, y_true):
    x = _to_numpy(x_raw).reshape(-1)
    y = _to_numpy(y_true).reshape(-1)

    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 5:
        # Fallback: identity
        return (lambda z: _to_numpy(z)), [1, 1, 0, 0, 1]

    x = x[valid]; y = y[valid]

    # Reasonable initial guesses
    a0 = np.nanmax(y)
    d0 = np.nanmin(y)
    b0 = 1.0
    c0 = np.nanmedian(x)
    g0 = 1.0
    p0 = [a0, b0, c0, d0, g0]

    try:
        popt, _ = curve_fit(five_pl, x, y, p0=p0, maxfev=20000)
    except Exception:
        # Fallback to identity if curve fitting fails
        return (lambda z: _to_numpy(z)), p0

    def calibrator_fn(z):
        z = _to_numpy(z).reshape(-1)
        return five_pl(z, *popt)

    return calibrator_fn, popt
    
def fit_fivepl_calibrators(x_raw_all, y_true_all, mask_all):
    T = x_raw_all.shape[1]
    calibrator_fns = []
    calibrated_preds = np.zeros_like(x_raw_all)

    for t in range(T):
        x_raw = x_raw_all[:, t]
        y_true = y_true_all[:, t]
        mask = mask_all[:, t]

        valid = mask & ~np.isnan(x_raw) & ~np.isnan(y_true)

        if valid.sum() < 5:
            calibrator_fns.append(lambda x: x)
            calibrated_preds[:, t] = x_raw
            continue

        cal_fn, _ = fit_fivepl_calibrator(x_raw[valid], y_true[valid])
        calibrator_fns.append(cal_fn)
        calibrated_preds[:, t] = cal_fn(x_raw)

    return calibrator_fns, calibrated_preds
    
def fit_mlp_calibrators(x_train, y_train, mask_train, hidden_dim=32, epochs=200, lr=1e-3, eval_data=None):
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrators = []

    if eval_data is not None:
        x_eval, y_eval, mask_eval = eval_data
        preds = np.full_like(y_eval, np.nan)
    else:
        preds = np.full_like(y_train, np.nan)
        x_eval, y_eval, mask_eval = x_train, y_train, mask_train

    num_targets = y_train.shape[1]
    input_dim = x_train.shape[1]

    for i in range(num_targets):
        train_mask = mask_train[:, i] & ~np.isnan(y_train[:, i])
        if train_mask.sum() < 10:
            calibrators.append(None)
            continue

        model = mlpCalibrator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_tr = torch.tensor(x_train[train_mask], dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_train[train_mask, i:i+1], dtype=torch.float32).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_tr)
            loss = criterion(output, y_tr)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                with torch.no_grad():
                    model.eval()

                    # === Train R² ===
                    train_preds = model(x_tr).cpu().numpy().flatten()
                    r2_train = r2_score(y_tr.cpu().numpy().flatten(), train_preds)

                    # === Val R² ===
                    r2_val = np.nan
                    if eval_data is not None:
                        val_valid = mask_eval[:, i] & ~np.isnan(y_eval[:, i])
                        if val_valid.sum() > 0:
                            x_val = torch.tensor(x_eval[val_valid], dtype=torch.float32).to(device)
                            y_val = y_eval[val_valid, i]
                            val_preds = model(x_val).cpu().numpy().flatten()
                            r2_val = r2_score(y_val, val_preds)

                    print(f"Target {i:2d} | Epoch {epoch+1:05d} | Loss: {loss.item():.4f} | "
                          f"Train R²: {r2_train:.3f} | Val R²: {r2_val:.3f}")

        # === Final Predictions ===
        model.eval()
        with torch.no_grad():
            val_mask = mask_eval[:, i] & ~np.isnan(y_eval[:, i])
            if val_mask.sum() > 0:
                x_te = torch.tensor(x_eval[val_mask], dtype=torch.float32).to(device)
                y_pred = model(x_te).cpu().numpy().flatten()
                preds[val_mask, i] = y_pred

        calibrators.append(model)

    return calibrators, preds
    
def _build_features_from_heads(x_heads, include_stats=True):
    """
    x_heads: (N, H) np array (all heads for one target)
    returns: (N, D) features
    """
    feats = [x_heads]  # raw heads
    if include_stats:
        mean = np.nanmean(x_heads, axis=1, keepdims=True)
        std  = np.nanstd(x_heads,  axis=1, keepdims=True)
        med  = np.nanmedian(x_heads, axis=1, keepdims=True)
        mn   = np.nanmin(x_heads, axis=1, keepdims=True)
        mx   = np.nanmax(x_heads, axis=1, keepdims=True)
        feats += [mean, std, med, mn, mx]
    # concat along feature dim
    return np.concatenate(feats, axis=1)
    
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _nan_safe_mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.ndim == 1:
            m &= np.isfinite(a)
        else:
            m &= np.all(np.isfinite(a), axis=1)
    return m

def fit_mlp_calibrators_multihead(
    x_train_full, y_train, mask_train,
    x_val_full,   y_val,   mask_val,
    target_names=None,
    hidden_dim=64, epochs=2000, lr=1e-3,
    weight_decay=1e-4, patience=100,
    include_stats=True, verbose=True,
    checkpoint_dir="checkpoints/calibrators",
    fold_id=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = y_train.shape[1]
    H = x_train_full.shape[2]
    if target_names is None:
        target_names = [f"t{i}" for i in range(T)]

    _ensure_dir(checkpoint_dir)

    preds_val = np.full_like(y_val, np.nan, dtype=float)
    mlp_models, scalers, logs = [], [], {name: {"train_loss": [], "val_loss": [], 
                                                "train_r2": [], "val_r2": []} 
                                         for name in target_names}

    for t in range(T):
        name = target_names[t]
        mtr = (mask_train[:, t] & np.isfinite(y_train[:, t]))
        mva = (mask_val[:, t]   & np.isfinite(y_val[:, t]))
        if mtr.sum() < 8 or mva.sum() < 3:
            if verbose:
                print(f"⚠️  Skipping {name} — not enough valid samples")
            mlp_models.append(None); scalers.append(None)
            continue

        Xtr = x_train_full[mtr, t, :]
        Xva = x_val_full[mva,   t, :]
        ytr = y_train[mtr, t]
        yva = y_val[mva,   t]

        if include_stats:
            Xtr = np.concatenate([Xtr, Xtr.mean(axis=1, keepdims=True), Xtr.std(axis=1, keepdims=True)], axis=1)
            Xva = np.concatenate([Xva, Xva.mean(axis=1, keepdims=True), Xva.std(axis=1, keepdims=True)], axis=1)

        mtr2 = _nan_safe_mask(Xtr, ytr)
        mva2 = _nan_safe_mask(Xva, yva)
        Xtr, ytr = Xtr[mtr2], ytr[mtr2]
        Xva, yva = Xva[mva2], yva[mva2]

        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xva = scaler.transform(Xva)
        scalers.append(scaler)

        in_dim = Xtr.shape[1]
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        ).to(device)

        opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(1)
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
        yva_t = torch.tensor(yva, dtype=torch.float32, device=device).unsqueeze(1)

        best_vloss = np.inf
        best_state = None
        wait = 0
        tag = f"{name}" if fold_id is None else f"{name}_fold{fold_id}"
        ckpt_path = os.path.join(checkpoint_dir, f"calib_{tag}.pt")

        for e in range(1, epochs+1):
            mlp.train()
            opt.zero_grad()
            pred_tr = mlp(Xtr_t)
            tr_loss = loss_fn(pred_tr, ytr_t)
            tr_loss.backward()
            opt.step()

            mlp.eval()
            with torch.no_grad():
                pred_va = mlp(Xva_t)
                va_loss = loss_fn(pred_va, yva_t).item()

                # Compute R²
                tr_r2 = r2_score(ytr_t.cpu().numpy(), pred_tr.cpu().numpy())
                va_r2 = r2_score(yva_t.cpu().numpy(), pred_va.cpu().numpy())

            logs[name]["train_loss"].append(tr_loss.item())
            logs[name]["val_loss"].append(va_loss)
            logs[name]["train_r2"].append(tr_r2)
            logs[name]["val_r2"].append(va_r2)

            if verbose and (e == 1 or e % 1000 == 0):
                print(f"[{name:>18s}] epoch {e:4d} | "
                      f"train loss {tr_loss.item():.4f} | val loss {va_loss:.4f} | "
                      f"train R² {tr_r2:.4f} | val R² {va_r2:.4f}")

            if va_loss < best_vloss - 1e-7:
                best_vloss = va_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
                torch.save(best_state, ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  ↳ Early stop [{name}] at epoch {e} "
                              f"(best val={best_vloss:.4f}). Saved: {ckpt_path}")
                    break

        if best_state is not None:
            mlp.load_state_dict(best_state)
        mlp_models.append(mlp)

        mlp.eval()
        with torch.no_grad():
            vpred = mlp(Xva_t).squeeze(1).cpu().numpy()
        full_idx = np.where(mask_val[:, t] & np.isfinite(y_val[:, t]))[0][mva2]
        preds_val[full_idx, t] = vpred

    return mlp_models, preds_val, scalers, logs