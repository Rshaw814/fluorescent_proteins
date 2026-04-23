import argparse
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from config import TARGET_NAMES, NUM_OUTPUTS, SPECTRAL_TARGETS, NUM_SPECTRAL, ERROR_BINS, NUM_CLASSES, CLASSIFICATION_TASKS, CLASSIFICATION_DIMS
from utils import load_data, make_loader, count_params, batched_features
from models import MultiTaskRegressorSelfConditioned, MultiClassErrorClassifier, mlpCalibrator
from models import MultiTaskRegressorDual, MultiTaskRegressorMultiHead
from training import train_epoch, train_rank_error_epoch, masked_r2_score, plackett_luce_loss, train_epoch_hrl, train_epoch_multihead,compute_bracketing_penalty_pairwise
from calibration import fit_isotonic_calibrators, compute_stoke_shift_metrics, fit_isotonic_calibrators_multihead, _build_features_from_heads, fit_mlp_calibrators_multihead
from evaluation import (
    compute_rank_errors,
    plot_all_heads,
    plot_rank_vs_rank_colored_by_value,
    evaluate_heads
)
from models import MultiTaskRegressorMultiHead_wClass
from helper_func.preprocessing import EmbeddingNormalizer

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from calibration import fit_sigmoid_calibrators, fit_fivepl_calibrators, fit_mlp_calibrators, fit_linear_calibrators_from_layer, extract_layer_features


from sklearn.linear_model import LinearRegression

from fp_or_not import FPBinaryClassifier
from foldability.train_plddt import PLDDTPredictor

from training import train_epoch_multihead_with_classification, eval_epoch_multihead_with_classification, eval_classification_accuracy
from models import MultiTaskRegressorMultiHead_wClass

def _as_raw_preds_np(reg_out, target_names):
    """
    Accepts either:
      - dict: {task_name -> tensor [B, H] or [B]}
      - tensor: [B, T, H] or [B, T]
    Returns a numpy array [B, T, H].
    """
    if isinstance(reg_out, dict):
        parts = []
        for t in target_names:
            x = to_numpy_safe(reg_out[t])
            if x.ndim == 1:
                x = x[:, None]   # [B] -> [B, 1]
            parts.append(x)      # each [B, H]
        return np.stack(parts, axis=1)  # [B, T, H]
    else:
        a = to_numpy_safe(reg_out)
        if a.ndim == 2:
            a = a[:, :, None]   # [B, T] -> [B, T, 1]
        return a   

def fit_linear_calibrators_from_layer(
    model,
    layer_name,  # unused, for tracking
    X_train_tensor,
    y_train,
    mask_train,
    X_val_tensor,
    y_val,
    mask_val,
    target_names
):
    """
    Fit one LinearRegression per target using training features from the task-specific head,
    then apply to validation features for evaluation.
    """
    model.eval()
    with torch.no_grad():
        _, hiddens_train = model(X_train_tensor, return_hidden=True)
        _, hiddens_val = model(X_val_tensor, return_hidden=True)

    calibrators = {}
    preds_val = np.full(y_val.shape, np.nan)
    r2_scores = []
    rho_scores = []

    for i, target in enumerate(target_names):
        # Extract features and targets
        X_train_feat = hiddens_train[target].cpu().numpy()
        X_val_feat = hiddens_val[target].cpu().numpy()
        y_train_target = y_train[:, i]
        y_val_target = y_val[:, i]
        mask_train_target = mask_train[:, i] & ~np.isnan(y_train_target)
        mask_val_target = mask_val[:, i] & ~np.isnan(y_val_target)

        if np.sum(mask_train_target) < 5 or np.sum(mask_val_target) < 5:
            print(f"⚠️ Skipping {target} — insufficient valid samples.")
            r2_scores.append(np.nan)
            rho_scores.append(np.nan)
            continue

        # Fit calibrator
        reg = LinearRegression().fit(X_train_feat[mask_train_target], y_train_target[mask_train_target])
        calibrators[target] = reg

        # Predict on validation set
        y_val_pred = reg.predict(X_val_feat[mask_val_target])
        preds_val[mask_val_target, i] = y_val_pred

        # Evaluate
        r2 = r2_score(y_val_target[mask_val_target], y_val_pred)
        rho, _ = spearmanr(y_val_target[mask_val_target], y_val_pred)
        r2_scores.append(r2)
        rho_scores.append(rho)

        print(f"{target:20}: R² = {r2:.3f} | ρ = {rho:.3f}")

    return calibrators, preds_val, r2_scores, rho_scores

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
    
def mean_abs_deviation(y_true, y_pred, mask):
    deviations = []
    for i in range(y_true.shape[1]):
        valid = mask[:, i] & ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        if np.sum(valid) >= 1:
            deviation = np.abs(y_true[valid, i] - y_pred[valid, i]).mean()
            deviations.append(deviation)
        else:
            deviations.append(np.nan)
    return np.array(deviations)

def driver(args):
    print("\n🚀 Starting driver script")
    X, y, mask, class_targets_all, error_class_targets = load_data(args.csv)
    
    # Build TASK_ORDER once
    TASK_ORDER = [t for t in CLASSIFICATION_TASKS if t in class_targets_all]
    
    # Sanity: labels within range
    for t in TASK_ORDER:
        arr = class_targets_all[t]
        valid = arr >= 0
        if valid.any():
            max_lab = int(arr[valid].max())
            assert max_lab < CLASSIFICATION_DIMS[t], (
                f"Label {max_lab} for '{t}' >= n_classes={CLASSIFICATION_DIMS[t]}"
            )
            # Optional: per-class counts
            print(t, {i: int((arr==i).sum()) for i in range(CLASSIFICATION_DIMS[t])})
    for t in CLASSIFICATION_TASKS:
        if t not in class_targets_all:
            print(f"[label coverage] {t}: 0 / {len(y)} (task not found in class_targets)")
        else:
            n_lab = int((class_targets_all[t] >= 0).sum())
            print(f"[label coverage] {t}: {n_lab} / {len(class_targets_all[t])}")
    all_raw_preds = []
    all_preds_all_heads = []
    all_val_records = []
    all_sigmoid_preds = []
    all_mlp_preds = []
    all_val_masks = []  # <— add this

    # ---- device first ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Keep a raw copy for aux models (they were trained on raw PLM embeddings)
    X_raw = X.copy()
    
    # ---- Load aux models (exact hidden_dims must match training) ----
    fluor_model = FPBinaryClassifier(input_dim=X_raw.shape[1], hidden_dims=[256, 128, 32], dropout=0.2)
    fluor_model.load_state_dict(torch.load("best_fp_classifier.pt", map_location="cpu"))
    fluor_model.to(device).eval()
    for p in fluor_model.parameters():
        p.requires_grad = False
    
    plddt_model = PLDDTPredictor(input_dim=X_raw.shape[1], hidden_dims=[1024, 768, 528, 128, 32])
    plddt_model.load_state_dict(torch.load("foldability/plddt_predictor.pt", map_location="cpu"))
    plddt_model.to(device).eval()
    for p in plddt_model.parameters():
        p.requires_grad = False
    
    # ---- Feature extraction on RAW embeddings ----
    plddt_feats = batched_features(plddt_model, plddt_model.forward_features, X_raw, device, batch_size=512)  # (N, Hp)
    fp_feats    = batched_features(fluor_model, fluor_model.forward_features, X_raw, device, batch_size=512)  # (N, Hf)
    
    # (Optional) also append scalar outputs
    # with torch.no_grad():
    #     plddt_scalar = plddt_model(torch.from_numpy(X_raw).float().to(device)).cpu().numpy()[:, None]
    #     fp_logit     = fluor_model(torch.from_numpy(X_raw).float().to(device)).cpu().numpy()[:, None]
    
    # ---- Concatenate: [PLM_raw] + [FP_hidden] + [pLDDT_hidden] ----
    X_concat = np.concatenate([X_raw, fp_feats, plddt_feats], axis=1)
    print(fp_feats.shape, plddt_feats.shape)
    # If you also want the scalars:
    # X_concat = np.concatenate([X_raw, fp_feats, plddt_feats, fp_logit, plddt_scalar], axis=1)
    
    # ---- Now normalize the concatenated features for the spectral model ----
    if args.load_normalizer:
        with open("normalized_emb.pkl", "rb") as f:
            normalizer = pickle.load(f)
        X = normalizer.transform(X_concat)
    else:
        normalizer = EmbeddingNormalizer()
        X = normalizer.fit_transform(X_concat)
        with open("normalized_emb.pkl", "wb") as f:
            pickle.dump(normalizer, f)
    
    input_dim = X.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_val_inputs, all_val_targets, all_preds, all_true = [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n🔁 Fold {fold + 1}/5")
        X_train, y_train, mask_train = X[train_idx], y[train_idx], mask[train_idx]
        X_val, y_val, mask_val = X[val_idx], y[val_idx], mask[val_idx]
        class_targets_train = {k: v[train_idx] for k, v in class_targets_all.items()}
        class_targets_val   = {k: v[val_idx]   for k, v in class_targets_all.items()}
        all_val_inputs.append(X_val)
    
        model = MultiTaskRegressorMultiHead_wClass(
            input_dim=input_dim,
            shared_dims=[256, 96],
            head_hidden_dim=64,
            dropout=0.01,
            n_outputs_per_task=5,
            target_names=TARGET_NAMES
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
        # Masks for ranking TRAINING (drop NaNs, don't throw out real zeros)
        mask_train_rank = (mask_train == 1) & ~np.isnan(y_train)
        mask_val_rank   = (mask_val   == 1) & ~np.isnan(y_val)
        
        train_loader = make_loader(
            X=X_train, y=y_train, mask=mask_train_rank,
            class_targets={k: v[train_idx] for k, v in class_targets_all.items()},
            batch_size=args.batch_size
        )
        val_loader = make_loader(
            X=X_val, y=y_val, mask=mask_val_rank,
            class_targets={k: v[val_idx] for k, v in class_targets_all.items()},
            batch_size=args.batch_size
        )
        
        # Masks for CALIBRATION/METRICS (separate from training masks; drop NaNs only)
        mask_train_cal = (mask_train == 1) & ~np.isnan(y_train)
        mask_val_cal   = (mask_val   == 1) & ~np.isnan(y_val)
    
        for epoch in range(args.epochs):
            train_loss = train_epoch_multihead_with_classification(
                model, train_loader, optimizer, device, args.margin
            )
            val_rho, class_accs = eval_epoch_multihead_with_classification(
                model, val_loader, device
            )
            val_class_accs = eval_classification_accuracy(model, val_loader, device)
            # nice compact print:
            acc_str = ", ".join([f"{t}={a:.3f}" for t, a in zip(CLASSIFICATION_TASKS, val_class_accs)])
            """
            print(
                f"Epoch {epoch+1:03d} | Train Loss={train_loss:.4f} "
                f"| Val ρ={val_rho:.3f} "
                f"| Class Acc={[round(a, 3) for a in class_accs]}"
            )
            """
            print(f"Epoch {epoch+1:03d} | Train Loss={train_loss:.4f} | Val ρ={val_rho:.3f} | Val Class Acc: [{acc_str}]")

        model.eval()
        with torch.no_grad():
            train_tensor = torch.from_numpy(X_train).float().to(device)
            _, train_hiddens = model(train_tensor, return_hidden=True)
        
            train_hidden_concat = [train_hiddens[task].cpu() for task in TARGET_NAMES]
            train_hidden_features = torch.stack(train_hidden_concat, dim=1).cpu().numpy()
            train_hidden_flat = train_hidden_features.reshape(train_hidden_features.shape[0], -1)
        """
        with torch.no_grad():
            val_tensor = torch.from_numpy(X_val).float().to(device)
            raw_preds = model(val_tensor).cpu().numpy()
        """  
        def _stack_regression_dict(reg_dict, target_names):
            # reg_dict: {task -> [B, H] tensor}; returns [B, T, H] numpy
            return np.stack([to_numpy_safe(reg_dict[t]) for t in target_names], axis=1)
            
        def _as_raw_preds_np(reg_out, target_names):
            """
            Accepts either:
              - dict: {task_name -> tensor [B, H] or [B, 1]}
              - tensor: [B, T, H] or [B, T]
            Returns a numpy array [B, T, H].
            """
            a = to_numpy_safe(reg_out)
            if isinstance(reg_out, dict):
                parts = []
                for t in target_names:
                    x = to_numpy_safe(reg_out[t])
                    if x.ndim == 1:
                        x = x[:, None]           # [B] -> [B, 1]
                    parts.append(x)               # each [B, H]
                return np.stack(parts, axis=1)    # [B, T, H]
            else:
                # tensor case
                if a.ndim == 2:
                    a = a[:, :, None]            # [B, T] -> [B, T, 1]
                # if already [B, T, H], just return
                return a
        
        model.eval()
        with torch.no_grad():
            val_tensor = torch.from_numpy(X_val).float().to(device)
            out = model(val_tensor)
        
        # out can be (reg_dict, class_dict) or just reg_tensor
        if isinstance(out, tuple) and len(out) == 2:
            reg_out_val, class_out_val = out
        else:
            reg_out_val, class_out_val = out, None
        
        raw_preds_np = _as_raw_preds_np(reg_out_val, TARGET_NAMES)  # [Nv, T, H]
        x_raw_all    = raw_preds_np.mean(axis=-1)                    # [Nv, T]
        y_val_np     = to_numpy_safe(y_val)
        mask_val_np  = to_numpy_safe(mask_val_cal).astype(bool)
        all_val_masks.append(mask_val_np)
        
        # Keep these names for any downstream code that expects them
        X_tensor     = torch.from_numpy(X_val).float().to(device)
        y_np_array   = y_val_np
        mask_np_array= mask_val_np
        X_eval_tensor= X_tensor
        y_eval       = y_val_np
        mask_eval    = mask_val_np

        # --- Aggregate heads to features ---
        # Get TRAIN raw preds (for fitting the calibrator)
        with torch.no_grad():
            train_tensor_for_cal = torch.from_numpy(X_train).float().to(device)
            out_tr = model(train_tensor_for_cal)
        
        if isinstance(out_tr, tuple) and len(out_tr) == 2:
            reg_out_train, _ = out_tr
        else:
            reg_out_train = out_tr
        
        train_raw_preds = _as_raw_preds_np(reg_out_train, TARGET_NAMES)  # [Nt, T, H]
        x_train_feats   = train_raw_preds.mean(axis=-1)                   # [Nt, T]
        x_val_feats   = x_raw_all                      # [Nv, T] (already mean across heads)
        
        # --- Fit per-target MLP calibrator on TRAIN, evaluate on VAL ---
        mlp_calibrators, mlp_preds, mlp_scalers, mlp_logs = fit_mlp_calibrators_multihead(
            x_train_full=train_raw_preds, y_train=y_train, mask_train=mask_train_cal,
            x_val_full=raw_preds_np,      y_val=y_val_np,   mask_val=mask_val_np,
            target_names=TARGET_NAMES,
            hidden_dim=64, epochs=500, lr=1e-4,
            weight_decay=1e-4, patience=1000,
            include_stats=True, verbose=True,
            checkpoint_dir=f"checkpoints/calibrators/fold_{fold}",  # per-fold folder
            fold_id=fold
        )
        
        # (optional) print per-target metrics for this fold
        print("\n📈 MLP Calibration Results (fold):")
        for i, name in enumerate(TARGET_NAMES):
            m = mask_val_np[:, i] & ~np.isnan(mlp_preds[:, i]) & ~np.isnan(y_val_np[:, i])
            if np.sum(m) < 3 or len(np.unique(mlp_preds[m, i])) < 2 or len(np.unique(y_val_np[m, i])) < 2:
                print(f"{name:20}: R² = NaN | ρ = NaN")
            else:
                r2_i = r2_score(y_val_np[m, i], mlp_preds[m, i])
                rho_i = spearmanr(y_val_np[m, i], mlp_preds[m, i])[0]
                print(f"{name:20}: R² = {r2_i:.3f} | ρ = {rho_i:.3f}")
        
    
        # ✅ Immediately append raw_preds to preserve order
        all_raw_preds.append(raw_preds_np)
        print(f"✅ Fold {fold+1}: raw_preds.shape = {raw_preds_np.shape}")
        
        mask_train_cal = (mask[train_idx] == 1) & ~np.isnan(y_train)
        mask_val_cal = (mask[val_idx] == 1) & ~np.isnan(y_val)
    
        train_tensor = torch.from_numpy(X_train).float().to(device)
        out_tr2 = model(train_tensor)
        if isinstance(out_tr2, tuple) and len(out_tr2) == 2:
            reg_out_train2, _ = out_tr2
        else:
            reg_out_train2 = out_tr2
        train_raw_preds = _as_raw_preds_np(reg_out_train2, TARGET_NAMES)  # [Nt, T, H]
        
        ranking_mask = np.zeros_like(mask_train, dtype=bool)

        for t in range(mask_train.shape[1]):
            valid = mask_train[:, t] & ~np.isnan(y_train[:, t])
            if valid.sum() < 5:
                continue
        
            preds_t = train_raw_preds[valid, t].mean(axis=-1)
            true_t = y_train[valid, t]
            rho, _ = spearmanr(preds_t, true_t)
        
            if rho >= 0.6:
                ranking_mask[valid, t] = True
        # ✅ Single calibration call
        calibrators, preds, preds_all_heads, r2s, rhos = fit_isotonic_calibrators_multihead(
            raw_preds_np, y_val_np, mask_val_np,
            calibration_data=(to_numpy_safe(train_raw_preds), y_train, mask_train_cal)
        )
        all_preds.append(preds)
        all_preds_all_heads.append(preds_all_heads)
        all_val_targets.append(y_val)
        all_true.append(y_val)
        
       
        
        sigmoid_calibrators, sigmoid_preds = fit_sigmoid_calibrators(
            x_raw_all, y_val_np, mask_val_np
        )
        
        sigmoid_r2 = masked_r2_score(y_val, sigmoid_preds, mask_val)
        sigmoid_rho = [
            spearmanr(sigmoid_preds[:, i][mask_val[:, i]], y_val[:, i][mask_val_np[:, i]])[0]
            if np.sum(mask_val[:, i]) > 0 else np.nan
            for i in range(len(TARGET_NAMES))
        ]
        all_sigmoid_preds.append(sigmoid_preds)
        
        # 5-Parameter Logistic Calibration
        
        fivepl_calibrators, fivepl_preds = fit_fivepl_calibrators(
            x_raw_all, y_val_np, mask_val_np
        )
        
        fivepl_r2 = masked_r2_score(y_val, fivepl_preds, mask_val)
        fivepl_rho = [
            spearmanr(fivepl_preds[:, i][mask_val[:, i]], y_val[:, i][mask_val[:, i]])[0]
            if np.sum(mask_val[:, i]) > 0 else np.nan
            for i in range(len(TARGET_NAMES))
        ]
        
        #mlp_calibrators, mlp_preds = fit_mlp_calibrators(raw_preds.mean(axis=-1), y_val, mask_val)
        
        mlp_r2 = masked_r2_score(y_val, mlp_preds, mask_val)
        mlp_rho = [
            spearmanr(mlp_preds[:, i][mask_val[:, i]], y_val[:, i][mask_val[:, i]])[0]
            if np.sum(mask_val[:, i]) > 0 else np.nan
            for i in range(len(TARGET_NAMES))
        ]
        all_mlp_preds.append(mlp_preds)
        
        
        for i in range(len(y_val)):
            row = {
                "fold": fold,
                "true_values": y_val[i].tolist(),
                "mlp_pred": mlp_preds[i].tolist(),
                "sigmoid_pred": sigmoid_preds[i].tolist(),
                "raw_pred": raw_preds_np[i].mean(axis=-1).tolist()
            }
            all_val_records.append(row)
        """
        # === Plot sample predictions ===
        ex_idx = TARGET_NAMES.index("ex_max")
        em_idx = TARGET_NAMES.index("em_max")
    
        print("\n🔍 Sample calibrated predictions vs. true values (ex_max, em_max, stoke shift):")
        for i in range(min(30, len(y_val))):
            pred_ex = preds[i, ex_idx]
            pred_em = preds[i, em_idx]
            true_ex = y_val[i, ex_idx]
            true_em = y_val[i, em_idx]
            pred_shift = pred_em - pred_ex
            true_shift = true_em - true_ex
            print(f"  Row {i+1:02d} | ex: {pred_ex:.2f} / {true_ex:.2f} | em: {pred_em:.2f} / {true_em:.2f} | shift: {pred_shift:.2f} / {true_shift:.2f}")
        """              
        mad = mean_abs_deviation(y_val, mlp_preds, mask_val)
        print("\n📉 Mean Absolute Deviation (nm):")
        for i, name in enumerate(TARGET_NAMES):
            print(f"  {name:20}: {mad[i]:.2f} nm")
 
        """
        all_preds.append(preds)
        all_val_targets.append(y_val)
        all_true.append(y_val)
        """
    # === After CV loop ===
    # Make sure you appended mask_val_np each fold: all_val_masks.append(mask_val_np)
    mask_all         = np.concatenate(all_val_masks).astype(bool)
    
    X_oof_inputs     = np.concatenate(all_val_inputs)
    y_true_all       = np.concatenate(all_true)
    y_pred_all       = np.concatenate(all_preds)
    preds_all_heads  = np.concatenate(all_preds_all_heads, axis=0)
    sigmoid_pred_all = np.concatenate(all_sigmoid_preds) if len(all_sigmoid_preds) else None
    mlp_pred_all     = np.concatenate(all_mlp_preds)     if len(all_mlp_preds)     else None
    raw_pred_all     = np.concatenate(all_raw_preds)
    
    # Sanity check
    assert mask_all.shape[0] == y_true_all.shape[0] == y_pred_all.shape[0], \
        f"Shape mismatch: mask={mask_all.shape}, y={y_true_all.shape}, pred={y_pred_all.shape}"
    
    # Metrics that use the mask
    headwise_r2_vec = masked_r2_score(y_true_all, y_pred_all, mask_all, return_per_target=True)
    spectral_indices = [TARGET_NAMES.index(n) for n in SPECTRAL_TARGETS]
    final_r2 = np.nanmean(headwise_r2_vec[spectral_indices])

    error_targets = compute_rank_errors(y_true_all, y_pred_all)
    if args.train_error_model:
        print("\n🔍 Training error predictor on residual rank errors")
        kf_error = KFold(n_splits=5, shuffle=True, random_state=42)
        error_preds_val, error_true_val = [], []
    
        for fold, (train_idx, val_idx) in enumerate(kf_error.split(X_oof_inputs)):
            print(f"\n🔁 Error Predictor Fold {fold + 1}/5")
            X_train, X_val = X_oof_inputs[train_idx], X_oof_inputs[val_idx]
            y_train, y_val = error_targets[train_idx], error_targets[val_idx]
    
            error_model = MultiTaskErrorPredictor(input_dim=input_dim).to(device)
            optimizer = torch.optim.Adam(error_model.parameters(), lr=args.lr)
            dummy_mask = np.ones_like(y_train)
            train_loader = make_loader(X_train, y_train, dummy_mask, batch_size=args.batch_size)
    
            for epoch in range(args.epochs):
                train_loss = train_rank_error_epoch(error_model, train_loader, optimizer, args.margin, device)
                print(f"  📉 Error Epoch {epoch+1:03d}: Loss = {train_loss:.4f}")
                
    
            error_model.eval()
            with torch.no_grad():
                val_tensor = torch.from_numpy(X_val).float().to(device)
                val_preds = error_model(val_tensor).cpu().numpy()
                error_preds_val.append(val_preds)
                error_true_val.append(y_val)
    
        raw_predicted_errors = np.concatenate(error_preds_val)
        true_errors = np.concatenate(error_true_val)
        dummy_mask = ~np.isnan(true_errors)
        error_calibrators, calibrated_predicted_errors, r2_err, rho_err = fit_isotonic_calibrators(raw_predicted_errors, true_errors, dummy_mask)
    
        print("\n📊 Error Predictor R² and Spearman ρ per target:")
        for i, name in enumerate(TARGET_NAMES):
            print(f"  {name:15}: R² = {r2_err[i]:.3f} | ρ = {rho_err[i]:.3f}")
    mlp_pred_all = np.concatenate(all_mlp_preds)

    plot_all_heads(y_true_all, y_pred_all, TARGET_NAMES, output_path=args.plot_path)
    if sigmoid_pred_all is not None:
        plot_all_heads(y_true_all, sigmoid_pred_all, TARGET_NAMES, output_path="json/all_heads_prediction_vs_actual_sigmoid.png")
    if mlp_pred_all is not None:
        plot_all_heads(y_true_all, mlp_pred_all, TARGET_NAMES, output_path="json/all_heads_prediction_vs_actual_mlp.png")

    #plot_all_heads(y_true_all,raw_pred_all, TARGET_NAMES, output_path="plot_raw.png")
    raw_pred_all = np.concatenate(all_raw_preds)
    print(f"✅ Final concatenated raw_pred_all.shape = {raw_pred_all.shape}")
    print(f"✅ y_true_all.shape = {y_true_all.shape}")

    plot_all_heads(y_true_all, raw_pred_all.mean(axis=-1), TARGET_NAMES, output_path="plot_raw.png")
    evaluate_heads(y_true_all, preds_all_heads, TARGET_NAMES, output_dir = "graphs")

    if args.train_error_model:
        plot_rank_vs_rank_colored_by_value(y_true_all, y_pred_all, TARGET_NAMES, calibrated_predicted_errors)

    param_count = count_params(model)
    
    headwise_r2_vec = masked_r2_score(y_true_all, y_pred_all, mask_all, return_per_target=True)
    spectral_indices = [TARGET_NAMES.index(n) for n in SPECTRAL_TARGETS]
    final_r2 = np.nanmean(headwise_r2_vec[spectral_indices])


    #headwise_r2 = masked_r2_score(y_true_all, y_pred_all)
    #final_r2 = np.nanmean(headwise_r2)
    
    
    #final_r2 = r2_score(y_true_all, y_pred_all, multioutput='uniform_average')
    
    print("\n🏁 Final Evaluation Across All Folds")
    print(f"🔢 Model parameter count: {param_count}")
    print(f"📊 Final Mean Calibrated R²: {final_r2:.4f}")
    
    if args.final_train:
        print("\n🏁 Running Final Training on Full Dataset")

        model = MultiTaskRegressorMultiHead_wClass(
            input_dim=input_dim,
            shared_dims=[256, 96],
            head_hidden_dim=64,
            dropout=0.01,
            n_outputs_per_task=5,
            target_names=TARGET_NAMES
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        full_loader = make_loader(X, y, mask, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            #train_loss = train_epoch_multihead(model, full_loader, full_loader, optimizer, args.margin, device)
            train_loss, _ = train_epoch_multihead_with_classification(model, full_loader, full_loader, optimizer, args.margin, device, 100)
            print(f"  📉 FinalTrain Epoch {epoch+1:03d}: Loss = {train_loss:.4f}")
    
        # === Final inference ===
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(X).float().to(device)
            raw_preds = model(input_tensor).cpu().numpy()

        for i, name in enumerate(SPECTRAL_TARGETS):
            col = raw_preds[:, i]
            print(f"{name:15} → min: {col.min():.2f} | max: {col.max():.2f} | mean: {col.mean():.2f} | std: {col.std():.2f}")
        train_raw_preds = raw_preds  # Use raw predictions from full model
        ranking_mask = (mask == 1) & ~np.isnan(y)
        
        calibrators, preds = fit_linear_calibrators_from_layer(
            model=model,
            layer_name="head_hidden",
            X_tensor=X_tensor,           # torch.tensor of shape (N, embed_dim)
            y=y_np_array,                # shape (N, T)
            mask=mask_np_array,          # shape (N, T), bool
            target_names=TARGET_NAMES,
            eval_data=(X_eval_tensor, y_eval, mask_eval)  # Optional
        )
        r2_final = masked_r2_score(y, calibrated_preds, mask_val)
        rho_final = [
            spearmanr(calibrated_preds[:, i][mask[:, i]], y[:, i][mask[:, i]])[0]
            if np.sum(mask[:, i]) > 0 else np.nan
            for i in range(len(TARGET_NAMES))
        ]
        print("\n✅ Final Calibrated R² and Spearman ρ (full train):")
        for name, r2i, rhoi in zip(SPECTRAL_TARGETS, r2_final, rho_final):
            print(f"  {name:15}: R² = {r2i:.3f} | ρ = {rhoi:.3f}")

        # === Save final models ===
        torch.save(model.state_dict(), "saved_models/final_regressor.pt")
        with open("saved_models/mlp_calibrators.pkl", "wb") as f:
            pickle.dump(mlp_calibrators, f)
        np.save("final_predictions.npy", calibrated_preds)
        np.save("final_targets.npy", y)
        print("💾 Saved final model, calibrators, and predictions.")
        
        plot_all_heads(y, calibrated_preds, TARGET_NAMES, output_path="json/final_mlp_prediction_vs_actual.png")
        
        if args.train_error_model:
        

        
    # === Compute final rank errors ===
            error_targets = compute_rank_errors(y, calibrated_preds)
    
            print("\n🔍 Training Final Error Predictor")
            error_model = MultiTaskErrorPredictor(input_dim=input_dim).to(device)
            optimizer = torch.optim.Adam(error_model.parameters(), lr=args.lr)
    
            dummy_mask = np.ones_like(error_targets)
            error_loader = make_loader(X, error_targets, mask=dummy_mask, batch_size=args.batch_size)
    
            for epoch in range(args.epochs):
                train_loss = train_rank_error_epoch(error_model, error_loader, optimizer, args.margin, device)
                print(f"  📉 FinalError Epoch {epoch+1:03d}: Loss = {train_loss:.4f}")

            error_model.eval()
            with torch.no_grad():
                error_preds = error_model(torch.from_numpy(X).float().to(device)).cpu().numpy()
            
            final_mask = np.ones_like(error_targets)
    
            error_calibrators, calibrated_error_preds, r2_err_final, rho_err_final = fit_isotonic_calibrators(
                error_preds, error_targets, final_mask
            )

            print("\n📊 Final Error Calibrated R² and Spearman ρ:")
            for i, name in enumerate(TARGET_NAMES):
                print(f"  {name:15}: R² = {r2_err_final[i]:.3f} | ρ = {rho_err_final[i]:.3f}")
            # === Save error model and results ===
            torch.save(error_model.state_dict(), "saved_models/final_error_predictor.pt")
            with open("saved_models/error_calibrators.pkl", "wb") as f:
                pickle.dump(error_calibrators, f)
            np.save("final_error_preds.npy", calibrated_error_preds)
            np.save("final_rank_errors.npy", error_targets)
            print("💾 Saved final error model, calibrators, and rank errors.")
            
    print("\n📈 Final Task-Specific Validation Scores (Calibrated):")
    print("────────────────────────────────────────────────────────")
    
    for i, name in enumerate(TARGET_NAMES):
        y_true_i = y_true_all[:, i]
        y_pred_i = mlp_pred_all[:, i]
    
        valid = ~np.isnan(y_true_i) & ~np.isnan(y_pred_i)
        if np.sum(valid) == 0:
            print(f"{name:<20s} R²: NaN       ρ: NaN")
            continue
    
        r2 = r2_score(y_true_i[valid], y_pred_i[valid])
        rho, _ = spearmanr(y_true_i[valid], y_pred_i[valid])
    
        print(f"{name:<20s} R²: {r2:.3f}     ρ: {rho:.3f}")

    if args.output_json:
        run_id = Path(args.output_json).stem.split("_")[-1]
        np.save(f"variance_results/preds_{run_id}.npy", y_pred_all)
        np.save(f"variance_results/true_{run_id}.npy", y_true_all)
        results = {
            "param_count": param_count,
            "final_r2": np.nanmean(mlp_r2),
            "headwise_r2": mlp_r2
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f)
        print(f"💾 Saved results to {args.output_json}")
        
    if sigmoid_pred_all is not None:
        sigmoid_r2_vec = masked_r2_score(y_true_all, sigmoid_pred_all, mask_all, return_per_target=True)
        sigmoid_rho_vec = []
        for i in range(len(TARGET_NAMES)):
            m = mask_all[:, i] & np.isfinite(sigmoid_pred_all[:, i]) & np.isfinite(y_true_all[:, i])
            if m.sum() >= 2:
                sigmoid_rho_vec.append(spearmanr(y_true_all[m, i], sigmoid_pred_all[m, i]).correlation)
            else:
                sigmoid_rho_vec.append(np.nan)
    
        print("\n📈 Sigmoid Calibration (Platt-like) Results:")
        for i, name in enumerate(TARGET_NAMES):
            r2i = sigmoid_r2_vec[i]
            rhoi = sigmoid_rho_vec[i]
            print(f"{name:20}: R² = {np.nan_to_num(r2i, nan=np.nan):.3f} | ρ = {np.nan_to_num(rhoi, nan=np.nan):.3f}")
    sigmoid_rho = [
        spearmanr(sigmoid_pred_all[:, i][~np.isnan(y_true_all[:, i])], y_true_all[:, i][~np.isnan(y_true_all[:, i])])[0]
        if np.sum(~np.isnan(y_true_all[:, i])) > 0 else np.nan
        for i in range(len(TARGET_NAMES))
    ]
        
 
        
    # --- Five-Parameter Logistic (5-PL) summary (after CV) ---
    if 'fivepl_pred_all' in locals() and fivepl_pred_all is not None:
        fivepl_r2_vec = masked_r2_score(y_true_all, fivepl_pred_all, mask_all, return_per_target=True)
    
        fivepl_rho_vec = []
        for i in range(len(TARGET_NAMES)):
            m = mask_all[:, i] & np.isfinite(fivepl_pred_all[:, i]) & np.isfinite(y_true_all[:, i])
            if m.sum() >= 2 and np.unique(y_true_all[m, i]).size > 1:
                fivepl_rho_vec.append(spearmanr(y_true_all[m, i], fivepl_pred_all[m, i]).correlation)
            else:
                fivepl_rho_vec.append(np.nan)
    
        print("\n📈 Five-Parameter Logistic (5-PL) Calibration Results:")
        for i, name in enumerate(TARGET_NAMES):
            r2i  = fivepl_r2_vec[i]
            rhoi = fivepl_rho_vec[i]
            print(f"{name:20}: R² = {np.nan_to_num(r2i,  nan=np.nan):.3f} | ρ = {np.nan_to_num(rhoi, nan=np.nan):.3f}")
        
    # --- MLP calibration summary (after CV) ---
    if 'mlp_pred_all' in locals() and mlp_pred_all is not None:
        mlp_r2_vec = masked_r2_score(y_true_all, mlp_pred_all, mask_all, return_per_target=True)
    
        mlp_rho_vec = []
        for i in range(len(TARGET_NAMES)):
            m = mask_all[:, i] & np.isfinite(mlp_pred_all[:, i]) & np.isfinite(y_true_all[:, i])
            if m.sum() >= 2 and np.unique(y_true_all[m, i]).size > 1:
                mlp_rho_vec.append(spearmanr(y_true_all[m, i], mlp_pred_all[m, i]).correlation)
            else:
                mlp_rho_vec.append(np.nan)
    
        print("\n📈 MLP Calibration Results:")
        for i, name in enumerate(TARGET_NAMES):
            r2i  = mlp_r2_vec[i]
            rhoi = mlp_rho_vec[i]
            print(f"{name:20}: R² = {np.nan_to_num(r2i,  nan=np.nan):.3f} | ρ = {np.nan_to_num(rhoi, nan=np.nan):.3f}")
            

        
        
    val_df = pd.DataFrame(all_val_records)
    Path("saved_models").mkdir(exist_ok=True)
    val_df.to_csv("saved_models/validation_predictions.csv", index=False)
    print("💾 Saved validation predictions to 'saved_models/validation_predictions.csv'")
    

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", default="embeddings/max_embeddings_plus.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--output_json", type=str, help="Save results to JSON")
    parser.add_argument("--plot_path", type=str, default="json/all_heads_prediction_vs_actual.png")
    parser.add_argument("--load_normalizer", action="store_true", help="Load saved normalizer instead of fitting a new one")
    parser.add_argument("--final_train", action="store_true", help="Train final model on full dataset")
    parser.add_argument("--train_error_model", action="store_true", help="Enable error model training")
    args = parser.parse_args()
    driver(args)