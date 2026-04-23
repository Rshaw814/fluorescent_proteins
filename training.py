import torch.nn as nn
import torch
from scipy.stats import spearmanr
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from config import TARGET_NAMES, NUM_OUTPUTS, CLASSIFICATION_NAMES, ERROR_BINS, SPECTRAL_TARGETS, NUM_CLASSES, CLASSIFICATION_TASKS, CLASSIFICATION_LOSS_WEIGHTS
from calibration import fit_isotonic_calibrators
from collections import defaultdict
from utils import bin_errors


def train_epoch_range(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rho = []

    for batch in loader:
        X, y, mask = batch[:3]
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        lower_preds, upper_preds, mu_preds, log_widths = model(X)

        loss = 0.0
        for i, name in enumerate(TARGET_NAMES):
            y_true = y[:, i]
            valid = mask[:, i].bool()
            if valid.sum() == 0:
                continue

            l = lower_preds[name][valid]
            u = upper_preds[name][valid]
            y_tgt = y_true[valid]

            loss += range_loss(l, u, y_tgt)

            mid = (l + u) / 2
            rho = spearmanr(mid.detach().cpu(), y_tgt.cpu()).correlation
            total_rho.append(rho)

        if len(total_rho) > 0:
            loss /= len(total_rho)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    mean_rho = np.mean(total_rho) if total_rho else 0.0
    return total_loss / len(loader), mean_rho

def range_loss(mu, log_width, y_true, alpha=1.0, beta=0.1, gamma=1.0):
    width = F.softplus(log_width) + 1e-3
    lower = mu - width / 2
    upper = mu + width / 2

    in_range = ((y_true >= lower) & (y_true <= upper)).float()
    coverage_loss = 1.0 - in_range.mean()

    width_penalty = width.mean()
    mse_loss = F.mse_loss(mu, y_true)

    return alpha * coverage_loss + beta * width_penalty + gamma * mse_loss

def nll_gaussian(mu, logvar, y_true):
    precision = torch.exp(-logvar)
    return 0.5 * precision * (y_true - mu) ** 2 + 0.5 * logvar

def train_epoch_uncertainty(model, loader, optimizer, device, lambda_constraints=10.0):
    model.train()
    total_loss = 0.0
    total_rho = []

    # General min/max ranges for each target (domain knowledge)
    target_ranges = {
        "ex_max": (350, 700),
        "em_max": (400, 800),
        "ext_coeff": (10000, 200000),  # units: M^-1 cm^-1
        "qy": (0.0, 1.0),
        "brightness": (0.0, 200.0),
        "stroke_shift": (0.0, 300.0),
        "aromaticity": (0.0, 0.3),
        "instability_index": (20.0, 80.0),
    }

    for batch in loader:
        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
        X, y, mask = batch[:3]

        optimizer.zero_grad()
        mu_preds, logvar_preds = model(X)

        loss = 0.0
        constraint_loss = 0.0

        for i, name in enumerate(TARGET_NAMES):
            y_true = y[:, i]
            valid = mask[:, i].bool()
            if valid.sum() == 0:
                continue

            mu = mu_preds[name][valid]
            logvar = logvar_preds[name][valid]
            y_tgt = y_true[valid]

            loss += nll_gaussian(mu, logvar, y_tgt).mean()

            # Spearman ρ for monitoring
            rho = spearmanr(mu.detach().cpu(), y_tgt.cpu()).correlation
            total_rho.append(rho)

            # === Constraint 3: range violation penalty ===
            if name in target_ranges:
                min_val, max_val = target_ranges[name]
                mu_full = mu_preds[name]
                valid_mask = mask[:, i].bool()
                below_min = F.relu(min_val - mu_full[valid_mask])
                above_max = F.relu(mu_full[valid_mask] - max_val)
                constraint_loss += (below_min + above_max).mean()

        # === Constraint 1: em_max > ex_max ===
        if "em_max" in mu_preds and "ex_max" in mu_preds:
            em = mu_preds["em_max"]
            ex = mu_preds["ex_max"]
            valid = mask[:, TARGET_NAMES.index("em_max")] * mask[:, TARGET_NAMES.index("ex_max")]
            violation = F.relu(ex - em)  # Penalize em < ex
            constraint_loss += violation[valid.bool()].mean()

        # === Constraint 2: brightness ≈ qy * ext_coeff ===
        if all(k in mu_preds for k in ["brightness", "qy", "ext_coeff"]):
            bright = mu_preds["brightness"]
            qy = mu_preds["qy"]
            ext = mu_preds["ext_coeff"]
            valid = (
                mask[:, TARGET_NAMES.index("brightness")] *
                mask[:, TARGET_NAMES.index("qy")] *
                mask[:, TARGET_NAMES.index("ext_coeff")]
            )
            pred_brightness = qy * ext / 1000  # brightness unit scaled down
            deviation = torch.abs(bright - pred_brightness)
            constraint_loss += deviation[valid.bool()].mean()

        # === Total loss (main + constraint) ===
        if len(total_rho) > 0:
            loss = loss / len(total_rho)
        total = loss + lambda_constraints * constraint_loss

        total.backward()
        optimizer.step()
        total_loss += total.item()

    mean_rho = np.mean(total_rho) if total_rho else 0.0
    return total_loss / len(loader), mean_rho

def train_multiclass_epoch(model, loader, optimizer, device, alpha=0.5):
    model.train()
    running = 0.0

    # --- define offsets for slicing logits ---
    offsets = [0]
    for t in SPECTRAL_TARGETS:
        offsets.append(offsets[-1] + NUM_CLASSES[t])

    for X, y_labels, _ in loader:  # y_labels: int64 with NaN→-1
        X = X.to(device)
        y = y_labels.to(device)

        optimizer.zero_grad()
        logits = model(X)  # [B, ΣC]

        loss_sum, n_tasks = 0.0, 0
        for i, task in enumerate(SPECTRAL_TARGETS):
            lo, hi = offsets[i], offsets[i+1]
            logit_slice = logits[:, lo:hi]  # [B, C_t]
            labels_t = y[:, i]              # [B]

            valid = labels_t != -1
            if valid.sum() == 0:
                continue

            labels_valid = labels_t[valid]
            logits_valid = logit_slice[valid]

            log_probs = F.log_softmax(logits_valid, dim=1)              # [B, C]
            nll_loss = F.nll_loss(log_probs, labels_valid, reduction='none')  # [B]

            pred_bins = log_probs.argmax(dim=1)                         # [B]
            distance = (pred_bins - labels_valid).abs().float()        # [B]
            penalty = 1.0 + alpha * distance                           # [B]

            penalized_loss = penalty * nll_loss                        # [B]
            loss = penalized_loss.mean()                               # scalar

            loss_sum += loss
            n_tasks += 1

        if n_tasks > 0:
            (loss_sum / n_tasks).backward()
            optimizer.step()
            running += (loss_sum / n_tasks).item()

    return running / len(loader)
    
def build_label_matrix(y_true, y_pred):
    """
    y_true, y_pred: numpy arrays shape [N, T] (absolute units, nm for ex/em)
    returns       : int64 label matrix shape [N, NUM_OUTPUTS]
                    with -1 where y_true is NaN
    """
    labels = np.full_like(y_true, fill_value=-1, dtype=np.int64)
    for idx, task in enumerate(SPECTRAL_TARGETS):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        mask = ~np.isnan(true)

        abs_err = np.abs(pred[mask] - true[mask])
        labels[mask, idx] = bin_errors(abs_err, ERROR_BINS[task])
    

    return labels

def pairwise_margin_loss(pred, true, margin=1.0):
    """
    pred, true: 1D tensors of shape [N]
    """
    diff_true = true[:, None] - true[None, :]
    diff_pred = pred[:, None] - pred[None, :]
    signs = torch.sign(diff_true)
    mask = signs != 0

    loss = torch.relu(-signs[mask] * diff_pred[mask] + margin)
    return loss.mean()
    
def pearson_corr(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()

    return torch.sum(pred * target) / (torch.norm(pred) * torch.norm(target) + 1e-8)

def train_epoch(model, loader, val_loader, optimizer, margin, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch, mb in loader:
        X = X_batch.to(device)
        y = y_batch.to(device)
        mb = mb.to(device)
        optimizer.zero_grad()
        scores = model(X)

        loss_sum = 0.0
        count = 0
        mse_loss = 0.0
        mse_count = 0

        for i in range(NUM_OUTPUTS):
            valid = mb[:, i].bool() & ~torch.isnan(y[:, i])  # Mask valid + non-NaN
            if valid.sum() < 2:
                continue

            #pred_i = scores[valid, i]
            pred_i = scores[:, i, :][valid].mean(dim=-1)
            true_i = y[valid, i]

            # Add pairwise margin loss
            margin_loss = pairwise_margin_loss(pred_i, true_i, margin)
            loss_sum += margin_loss
            count += 1

            # Add value-based MSE loss
            #mse_loss += F.mse_loss(pred_i, true_i)
            #mse_count += 1

        # Optional additional stoke shift margin ranking loss
        valid_stoke_margin = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke_margin.sum() >= 2:
            ex = scores[valid_stoke_margin, 0]
            em = scores[valid_stoke_margin, 1]
            ex_true = y[valid_stoke_margin, 0]
            em_true = y[valid_stoke_margin, 1]
            pred_stoke = em - ex
            true_stoke = em_true - ex_true

            stoke_corr_loss = pearson_corr(pred_stoke, true_stoke)
            loss_sum += (1-stoke_corr_loss)  #pairwise_margin_loss(pred_stoke, true_stoke, margin)
            count += 1

        # Constraint losses (still masked)
        brightness_pred = scores[:, 4]
        qy_pred = scores[:, 3]
        ext_pred = scores[:, 2]
        brightness_calc = qy_pred * ext_pred
        brightness_constraint_loss = F.mse_loss(brightness_pred, brightness_calc)

        # Stoke shift constraint: penalize em <= ex, only for valid rows
        valid_stoke_constraint = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke_constraint.sum() >= 2:
            ex_pred = scores[valid_stoke_constraint, 0]
            em_pred = scores[valid_stoke_constraint, 1]
            stoke_shift_violation = torch.relu(ex_pred - em_pred)  # Only penalize if em < ex
            stoke_shift_constraint_loss = stoke_shift_violation.mean()
        else:
            stoke_shift_constraint_loss = torch.tensor(0.0, device=device)
        
        ex_std = torch.std(scores[:, 0])
        em_std = torch.std(scores[:, 1])
        
        std_spread_penalty = -(ex_std + em_std)
            
        std_loss = -torch.std(scores, dim=0).mean()

        # === Final loss calculation ===
        if count > 0:
            alpha_mse = 0.0
            alpha1 = 0.0
            alpha2 = 0.1
            alpha_std = 0
            beta = 0.0
            loss = (loss_sum / count)
 

            if mse_count > 0:
                loss += alpha_mse * (mse_loss / mse_count)

            # Uncomment if you want constraint penalties
            loss += alpha1 * brightness_constraint_loss + alpha2 * stoke_shift_constraint_loss +beta + alpha_std * std_spread_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # === Validation (Spearman ρ with NaN masking)
    val_rho_mean = 0.0
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            all_preds, all_targets = [], []
            for X_batch, y_batch, _ in val_loader:
                X = X_batch.to(device)
                y = y_batch.to(device)
                scores = model(X).cpu().numpy()
                targets = y.cpu().numpy()
                all_preds.append(scores)
                all_targets.append(targets)

            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            val_rhos = []
            for i in range(NUM_OUTPUTS):
                mask = ~np.isnan(all_targets[:, i])
                if mask.sum() >= 2:
                    rho, _ = spearmanr(all_targets[mask, i], all_preds[mask, i])
                    val_rhos.append(rho)
                else:
                    val_rhos.append(np.nan)
            val_rho_mean = np.nanmean(val_rhos)
    """            
        # === Optional: Stoke shift validation metric
    valid_stoke = ~np.isnan(all_targets[:, 0]) & ~np.isnan(all_targets[:, 1])
    if valid_stoke.sum() >= 2:
        ex_pred = all_preds[valid_stoke, 0]
        em_pred = all_preds[valid_stoke, 1]
        ex_true = all_targets[valid_stoke, 0]
        em_true = all_targets[valid_stoke, 1]
        pred_stoke = em_pred - ex_pred
        true_stoke = em_true - ex_true
        stoke_rho, _ = spearmanr(true_stoke, pred_stoke)
        print(f"🔍 Validation Stokes shift ρ = {stoke_rho:.3f}")
        stoke_r2 = r2_score(true_stoke, pred_stoke)
        print(f"🔍 Validation Stokes shift R² = {stoke_r2:.3f}")
    """

    return total_loss / len(loader), val_rho_mean
    
def train_epoch_dual(model, loader, val_loader, optimizer, margin, device, classification_loss_fns):
    model.train()
    total_loss = 0.0
    classification_names = list(classification_loss_fns.keys())

    for X_batch, y_batch, mask_batch, class_batch in loader:
        X = X_batch.to(device)
        y = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        class_batch = class_batch.to(device)

        optimizer.zero_grad()
        reg_out, class_out = model(X)

        # === Regression/Ranking loss ===
        reg_loss_sum = 0.0
        reg_count = 0
        for i, task in enumerate(TARGET_NAMES):
            valid = mask_batch[:, i].bool() & ~torch.isnan(y[:, i])
            if valid.sum() < 2:
                continue
            pred_i = reg_out[task][valid]
            true_i = y[valid, i]
            loss = pairwise_margin_loss(pred_i, true_i, margin)
            reg_loss_sum += loss
            reg_count += 1

        # === Classification loss ===
        class_loss_sum = 0.0
        class_count = 0
        for i, name in enumerate(classification_names):
            pred_logits = class_out[name]
            class_labels = class_batch[:, i]
            valid = (class_labels >= 0) & (class_labels < pred_logits.shape[1])
            if valid.sum() < 1:
                continue
            pred_valid = pred_logits[valid]
            label_valid = class_labels[valid].long()
            loss_fn = classification_loss_fns[name]
            class_loss_sum += loss_fn(pred_valid, label_valid)
            class_count += 1

        # === Combine and backprop ===
        alpha_class = 1.0
        alpha_rank = 1.0
        total = 0.0
        if reg_count > 0:
            total += alpha_rank * (reg_loss_sum / reg_count)
        if class_count > 0:
            total += alpha_class * (class_loss_sum / class_count)

        total.backward()
        optimizer.step()
        total_loss += total.item()

    # === Final training: no validation
    if val_loader is None:
        return total_loss / len(loader), None, None

    # === Validation ===
    model.eval()
    with torch.no_grad():
        all_preds = {task: [] for task in TARGET_NAMES}
        all_true = {task: [] for task in TARGET_NAMES}
        class_pred_logits = {name: [] for name in classification_names}
        class_true_labels = {name: [] for name in classification_names}

        for X_batch, y_batch, mb, class_batch in val_loader:
            X = X_batch.to(device)
            y = y_batch.to(device)
            mb = mb.to(device)
            class_batch = class_batch.to(device)

            reg_out, class_out = model(X)

            for i, task in enumerate(TARGET_NAMES):
                mask = mb[:, i].bool() & ~torch.isnan(y[:, i])
                if mask.sum() >= 1:
                    all_preds[task].append(reg_out[task][mask].cpu().numpy())
                    all_true[task].append(y[mask, i].cpu().numpy())

            for i, name in enumerate(classification_names):
                logits = class_out[name]
                labels = class_batch[:, i]
                valid = (labels != -1) & ~torch.isnan(labels)
                if valid.sum() < 1:
                    continue
                class_pred_logits[name].append(logits[valid].cpu())
                class_true_labels[name].append(labels[valid].long().cpu())

        # === Spearman ρ
        val_rhos = []
        for task in TARGET_NAMES:
            pred = np.concatenate(all_preds[task]) if all_preds[task] else []
            true = np.concatenate(all_true[task]) if all_true[task] else []
            if len(true) >= 2:
                rho, _ = spearmanr(true, pred)
                val_rhos.append(rho)
            else:
                val_rhos.append(np.nan)
        val_rho_mean = np.nanmean(val_rhos)

        # === Classification Accuracy
        accs = []
        for name in CLASSIFICATION_NAMES:
            if class_pred_logits[name] and class_true_labels[name]:
                logits_cat = torch.cat(class_pred_logits[name], dim=0)
                labels_cat = torch.cat(class_true_labels[name], dim=0)
                preds = torch.argmax(logits_cat, dim=1)
                acc = (preds == labels_cat).sum().item() / len(labels_cat)
                accs.append((name, acc))
                print(f"{name:20}: {acc * 100:.2f}%")

                # Per-class accuracy
                num_classes = CLASSIFICATION_NAMES[name]
                from collections import defaultdict
                class_correct = defaultdict(int)
                class_total = defaultdict(int)

                for true, pred in zip(labels_cat, preds):
                    class_total[int(true)] += 1
                    if true == pred:
                        class_correct[int(true)] += 1

                print("   📉 Hardest classes (lowest accuracy):")
                class_accs = {
                    cls: (class_correct[cls] / class_total[cls]) if class_total[cls] > 0 else np.nan
                    for cls in range(num_classes)
                }
                hardest = sorted(class_accs.items(), key=lambda x: (x[1] if not np.isnan(x[1]) else 1.0))
                for cls, acc_cls in hardest[:3]:
                    acc_str = f"{acc_cls * 100:.2f}%" if not np.isnan(acc_cls) else "N/A"
                    print(f"     ➤ Label {cls}: {acc_str} accuracy")

    return total_loss / len(loader), val_rho_mean, accs

def train_rank_error_epoch(model, loader, optimizer, margin, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch, _ in loader:
        X = X_batch.to(device)
        y = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss_sum = 0.0
        count = 0

        for i in range(NUM_OUTPUTS):
            pred = preds[:, i]
            target = y[:, i]
            valid = ~torch.isnan(target)
            if valid.sum() < 2:
                continue
            pred_valid = pred[valid]
            target_valid = target[valid]
            margin_loss = pairwise_margin_loss(pred_valid, target_valid, margin)
            loss_sum += margin_loss
            count += 1

        brightness_pred = preds[:, 4]
        qy_pred = preds[:, 3]
        ext_pred = preds[:, 2]
        brightness_calc = qy_pred * ext_pred
        brightness_constraint_loss = F.mse_loss(brightness_pred, brightness_calc)



        if count > 0:
            alpha1 = 0.0
            alpha2 = 10
            loss = (loss_sum / count) + alpha1 * brightness_constraint_loss + alpha2  #stoke_shift_constraint_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)

def masked_ranking_loss(preds, targets, mask, margin=0.5):
    loss = 0.0
    count = 0
    for i in range(preds.size(1)):
        valid = mask[:, i].bool()
        if valid.sum() < 2:
            continue
        pred = preds[valid, i]
        true = targets[valid, i]
        margin_loss = pairwise_margin_loss(pred, true, margin)
        loss += margin_loss
        count += 1
    return loss / max(count, 1)

def masked_r2_score(y_true, y_pred, mask, return_per_target=False):
    """
    y_true, y_pred: shape (N, T)
    mask:          shape (N, T) boolean (True = valid)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask   = np.asarray(mask).astype(bool)

    T = y_true.shape[1]
    r2s = np.full(T, np.nan, dtype=float)

    for i in range(T):
        m = mask[:, i] & np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if m.sum() >= 2 and np.unique(y_true[m, i]).size > 1:
            r2s[i] = r2_score(y_true[m, i], y_pred[m, i])

    if return_per_target:
        return r2s
    return np.nanmean(r2s)
def calibration_loss(preds, targets, target_idx, λ=1.0):
    mse = F.mse_loss(preds, targets)

    ex = preds[:, target_idx["ex_max"]]
    em = preds[:, target_idx["em_max"]]
    qy = preds[:, target_idx["qy"]]
    ext = preds[:, target_idx["ext_coeff"]]
    bright = preds[:, target_idx["brightness"]]

    penalty_em_ex = F.relu(ex - em).mean()
    penalty_bright = ((bright * 1000 - qy * ext) ** 2).mean()

    return mse + λ * (penalty_em_ex + penalty_bright)
    
    
def train_calibrator(calibrator, raw_preds, y_true, mask, target_names, lr=1e-3, epochs=100, λ=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrator.to(device)

    target_idx = {name: i for i, name in enumerate(target_names)}
    spectral_idx = {"ex_max": target_idx["ex_max"], "em_max": target_idx["em_max"]}

    correct_mask = filter_correctly_ranked(raw_preds, y_true, spectral_idx)
    x_train = torch.tensor(raw_preds[correct_mask], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_true[correct_mask], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(calibrator.parameters(), lr=lr)

    for epoch in range(epochs):
        calibrator.train()
        optimizer.zero_grad()
        preds = calibrator(x_train)
        loss = calibration_loss(preds, y_train, target_idx, λ=λ)
        loss.backward()
        optimizer.step()

        print(f"  📉 Calibrator Epoch {epoch+1:03d} | Loss = {loss.item():.4f}")
        
def filter_correctly_ranked(preds, y_true, spectral_idx):
    ex_idx = spectral_idx["ex_max"]
    em_idx = spectral_idx["em_max"]
    return (preds[:, ex_idx] < preds[:, em_idx]) & (y_true[:, ex_idx] < y_true[:, em_idx])
    
def plackett_luce_loss(scores, ranking):
    loss = 0.0
    n = scores.size(0)
    for i in range(n):
        denom = torch.logsumexp(scores[ranking[i:]], dim=0)
        loss += denom - scores[ranking[i]]
    return loss/n
    
def train_epoch_hrl(model, loader, val_loader, optimizer, margin, device, target_names):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch, mb in loader:
        X = X_batch.to(device)
        y = y_batch.to(device)
        mb = mb.to(device)
        optimizer.zero_grad()
        scores_full, scores_mean, scores_std = model(X, return_stats=True)

        loss_sum = 0.0
        count = 0
        mse_loss = 0.0
        mse_count = 0

        # scores_full: [B, T, H]
        for i in range(NUM_OUTPUTS):
            valid = mb[:, i].bool() & ~torch.isnan(y[:, i])
            if valid.sum() < 2:
                continue
        
            pred_heads = scores_full[valid, i, :]  # shape: [N_valid, H]
            true_vals = y[valid, i]                # shape: [N_valid]
        
            task_loss = independent_head_loss(pred_heads, true_vals)
            loss_sum += task_loss
            count += 1

            # Add value-based MSE loss
            #mse_loss += F.mse_loss(pred_i, true_i)
            #mse_count += 1

        # Optional additional stoke shift margin ranking loss
        valid_stoke_margin = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke_margin.sum() >= 2:
            ex = scores_mean[valid_stoke_margin, 0]
            em = scores_mean[valid_stoke_margin, 1]
            ex_true = y[valid_stoke_margin, 0]
            em_true = y[valid_stoke_margin, 1]
            pred_stoke = em - ex
            true_stoke = em_true - ex_true

            stoke_corr_loss = pearson_corr(pred_stoke, true_stoke)
            loss_sum += (1-stoke_corr_loss)  #pairwise_margin_loss(pred_stoke, true_stoke, margin)
            count += 1

        # Constraint losses (still masked)
        brightness_pred = scores_mean[:, 4]
        qy_pred = scores_mean[:, 3]
        ext_pred = scores_mean[:, 2]
        brightness_calc = qy_pred * ext_pred
        brightness_constraint_loss = F.mse_loss(brightness_pred, brightness_calc)

        # Stoke shift constraint: penalize em <= ex, only for valid rows
        valid_stoke_constraint = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke_constraint.sum() >= 2:
            ex_pred = scores_mean[valid_stoke_constraint, 0]
            em_pred = scores_mean[valid_stoke_constraint, 1]
            stoke_shift_violation = torch.relu(ex_pred - em_pred)  # Only penalize if em < ex
            stoke_shift_constraint_loss = stoke_shift_violation.mean()
        else:
            stoke_shift_constraint_loss = torch.tensor(0.0, device=device)
        
        ex_std = torch.std(scores_mean[:, 0])
        em_std = torch.std(scores_mean[:, 1])
        
        std_spread_penalty = -(ex_std + em_std)
            
        std_loss = -torch.std(scores_mean, dim=0).mean()

        # === Final loss calculation ===
        if count > 0:
            alpha_mse = 0.0
            alpha1 = 0.0
            alpha2 = 0.1
            alpha_std = 0
            beta = 0.0
            loss = (loss_sum / count)
 

            if mse_count > 0:
                loss += alpha_mse * (mse_loss / mse_count)

            # Uncomment if you want constraint penalties
            loss += alpha1 * brightness_constraint_loss + alpha2 * stoke_shift_constraint_loss +beta + alpha_std * std_spread_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # === Validation (Spearman ρ with NaN masking)
    val_rho_mean = 0.0
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            all_preds_mean, all_preds_full, all_uncert, all_targets = [], [], [], []
            for X_batch, y_batch, _ in val_loader:
                X = X_batch.to(device)
                y = y_batch.to(device)
                scores_full, scores_mean, scores_std = model(X, return_stats=True)
                all_preds_mean.append(scores_mean.cpu().numpy())    # [N, T]
                all_preds_full.append(scores_full.cpu().numpy())    # [N, T, H]
                all_uncert.append(scores_std.cpu().numpy())         # [N, T]
                all_targets.append(y.cpu().numpy())                 # [N, T]
    
            all_preds_mean = np.concatenate(all_preds_mean)
            all_preds_full = np.concatenate(all_preds_full)
            all_uncert = np.concatenate(all_uncert)
            all_targets = np.concatenate(all_targets)
    
            # === Apply isotonic calibration ===
            mask_val = ~np.isnan(all_targets)
            calibrators, calibrated_preds, r2s, rhos = fit_isotonic_calibrators(all_preds_mean, all_targets, mask_val)
    
            val_rhos = []
            df_out = pd.DataFrame()
            print("\n📊 Per-Target Best-Head Validation Metrics:")
            for i, name in enumerate(TARGET_NAMES):
                y_true = all_targets[:, i]
                preds_i = all_preds_full[:, i, :]  # [N, H]
    
                # Best head per sample = argmin absolute error
                errors = np.abs(preds_i - y_true[:, None])
                best_head_idx = np.argmin(errors, axis=1)
                best_preds = preds_i[np.arange(len(y_true)), best_head_idx]
    
                # Save all head predictions
                for h in range(preds_i.shape[1]):
                    df_out[f"{name}_pred_head{h}"] = preds_i[:, h]
    
                df_out[f"{name}_pred_best"] = best_preds
                df_out[f"{name}_pred_mean"] = all_preds_mean[:, i]
                df_out[f"{name}_pred_calib"] = calibrated_preds[:, i]
                df_out[f"{name}_true"] = y_true
                df_out[f"{name}_uncert"] = all_uncert[:, i]
    
                mask = ~np.isnan(y_true)
                if mask.sum() >= 2:
                    rho, _ = spearmanr(y_true[mask], calibrated_preds[mask, i])
                    r2 = r2_score(y_true[mask], calibrated_preds[mask, i])
                    print(f"  {name:15} | Best-Head R² = {r2:.3f} | ρ = {rho:.3f}")
                    val_rhos.append(rho)
                else:
                    val_rhos.append(np.nan)
    
            df_out.to_csv("val_predictions_full.csv", index=False)
            val_rho_mean = np.nanmean(val_rhos)
            
    return total_loss / len(loader), val_rho_mean
    
def build_tournament_graph(rank_matrix):
    """
    rank_matrix: shape (n_samples, n_heads)
    """
    n = rank_matrix.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            votes_i = np.sum(rank_matrix[i] < rank_matrix[j])
            votes_j = np.sum(rank_matrix[i] > rank_matrix[j])
            if votes_i > rank_matrix.shape[1] // 2:
                G.add_edge(i, j)
            elif votes_j > rank_matrix.shape[1] // 2:
                G.add_edge(j, i)

    try:
        return list(nx.topological_sort(G))  # Returns a list of indices sorted by consensus
    except nx.NetworkXUnfeasible:
        return list(np.argsort(np.mean(rank_matrix, axis=1)))  # fallback to identity if graph is cyclic
        
        
def tournament_loss(pred_matrix, true_values, margin=0.1):
    """
    pred_matrix: Tensor (N, H) — predictions from H heads
    true_values: Tensor (N,) — ground truth values
    """
    N, H = pred_matrix.shape
    if N < 2:
        return torch.tensor(0.0, device=pred_matrix.device)

    # Get average prediction per sample across heads

    # Rank each sample under each head
    rank_matrix = torch.argsort(pred_matrix, dim=0, descending=True).cpu().numpy()

    # Build consensus tournament
    consensus_order = build_tournament_graph(rank_matrix.T)  # transpose to shape (samples, heads)

    # Convert consensus order to torch ranking indices
    ranking = torch.zeros(N, dtype=torch.long)
    for i, idx in enumerate(consensus_order):
        ranking[idx] = i

    # Compute Plackett-Luce loss
    avg_pred_torch = pred_matrix.mean(dim=1)
    return plackett_luce_loss(avg_pred_torch, ranking.to(avg_pred_torch.device))
    
    
def independent_head_loss(pred_matrix, true_values):
    """
    pred_matrix: Tensor (N, H) — predictions from H heads
    true_values: Tensor (N,) — ground truth values
    Returns the average Plackett-Luce loss across heads.
    """
    N, H = pred_matrix.shape
    if N < 2:
        return torch.tensor(0.0, device=pred_matrix.device)

    loss = 0.0
    # Precompute ranking ONCE
    ranking = torch.argsort(true_values, descending=True)

    for h in range(H):
        preds = pred_matrix[:, h]
        loss += plackett_luce_loss(preds, ranking)

    return loss / H
    
    
def train_epoch_multihead(model, loader, val_loader, optimizer, margin, device, lambda_bracket=100):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch, mb in loader:
        X = X_batch.to(device)
        y = y_batch.to(device)
        mb = mb.to(device)
        optimizer.zero_grad()

        scores_full, scores_mean, scores_std = model(X, return_stats=True)  # [B, T, H]
        loss_sum = 0.0
        count = 0

        for t in range(NUM_OUTPUTS):
            valid = mb[:, t].bool() & ~torch.isnan(y[:, t])
            if valid.sum() < 2:
                continue

            pred_heads = scores_full[valid, t, :]  # [N_valid, H]
            true_vals = y[valid, t]                # [N_valid]

            # === Pairwise Ranking Loss per head
            task_loss = 0.0
            for h in range(pred_heads.shape[1]):
                task_loss += pairwise_margin_loss(pred_heads[:, h], true_vals, margin)
            task_loss /= pred_heads.shape[1]

            # === Bracketing penalty (optional)
            bracket_penalty = compute_bracketing_penalty_pairwise(pred_heads, true_vals)
            task_loss += lambda_bracket * bracket_penalty

            loss_sum += task_loss
            count += 1

        # === Optional: Stokes shift loss on mean predictions
        valid_stoke = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke.sum() >= 2:
            ex = scores_mean[valid_stoke, 0]
            em = scores_mean[valid_stoke, 1]
            ex_true = y[valid_stoke, 0]
            em_true = y[valid_stoke, 1]

            pred_stoke = em - ex
            true_stoke = em_true - ex_true

            stoke_corr_loss = pearson_corr(pred_stoke, true_stoke)
            loss_sum += (1 - stoke_corr_loss)
            count += 1

        # === Constraints on mean predictions
        brightness_pred = scores_mean[:, 4]
        qy_pred = scores_mean[:, 3]
        ext_pred = scores_mean[:, 2]
        brightness_calc = qy_pred * ext_pred
        brightness_constraint_loss = F.mse_loss(brightness_pred, brightness_calc)

        valid_stoke_constraint = mb[:, 0].bool() & mb[:, 1].bool() & ~torch.isnan(y[:, 0]) & ~torch.isnan(y[:, 1])
        if valid_stoke_constraint.sum() >= 2:
            ex_pred = scores_mean[valid_stoke_constraint, 0]
            em_pred = scores_mean[valid_stoke_constraint, 1]
            stoke_shift_violation = torch.relu(ex_pred - em_pred)
            stoke_shift_constraint_loss = stoke_shift_violation.mean()
        else:
            stoke_shift_constraint_loss = torch.tensor(0.0, device=device)

        ex_std = torch.std(scores_mean[:, 0])
        em_std = torch.std(scores_mean[:, 1])
        std_spread_penalty = -(ex_std + em_std)

        if count > 0:
            alpha1, alpha2, alpha_std = 0.0, 0.1, 0.0
            beta = 0.0
            loss = (loss_sum / count)
            loss += alpha1 * brightness_constraint_loss + alpha2 * stoke_shift_constraint_loss + beta + alpha_std * std_spread_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # === Validation (mean prediction only)
    val_rho_mean = 0.0
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            all_preds, all_targets = [], []
            for X_batch, y_batch, _ in val_loader:
                X = X_batch.to(device)
                y = y_batch.to(device)
                scores_full, scores_mean, _ = model(X, return_stats=True)
                all_preds.append(scores_mean.cpu().numpy())
                all_targets.append(y.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            val_rhos = []
            for i in range(NUM_OUTPUTS):
                mask = ~np.isnan(all_targets[:, i])
                if mask.sum() >= 2:
                    rho, _ = spearmanr(all_targets[mask, i], all_preds[mask, i])
                    val_rhos.append(rho)
                else:
                    val_rhos.append(np.nan)
            val_rho_mean = np.nanmean(val_rhos)

    return total_loss / len(loader), val_rho_mean
    
    
def compute_bracketing_penalty_pairwise(preds: torch.Tensor, targets: torch.Tensor):
    """
    Penalizes if all heads are wrong in pairwise comparisons.

    preds: [N, H] (predictions for N samples, H heads)
    targets: [N] (true values for N samples)
    """
    device = preds.device
    N, H = preds.shape
    if N < 2:
        return torch.tensor(0.0, device=device)

    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    t_i = targets[idx_i]
    t_j = targets[idx_j]
    p_i = preds[idx_i]  # shape [num_pairs, H]
    p_j = preds[idx_j]

    true_dir = torch.sign(t_i - t_j).unsqueeze(1)  # shape [num_pairs, 1]
    pred_diff = p_i - p_j  # shape [num_pairs, H]
    correct = true_dir * pred_diff  # shape [num_pairs, H]

    all_wrong = (correct < 0).all(dim=1).float()
    return all_wrong.mean()
    
def train_epoch_multihead_with_classification(
    model, loader, optimizer, device, classification_tasks, classification_weights=None, margin=1.0
):
    model.train()
    total_loss = 0.0

    for X, y_reg, y_class in loader:
        X = X.to(device)
        y_reg = {k: v.to(device) for k, v in y_reg.items()}
        y_class = {k: v.to(device) for k, v in y_class.items()}

        optimizer.zero_grad()

        reg_outputs, class_outputs = model(X)  # reg: dict[name → (B, H)], class: dict[name → (B, C)]

        # === Regression loss (ranking) ===
        rank_loss = 0.0
        for name, preds in reg_outputs.items():
            # pairwise margin ranking loss over heads
            B, H = preds.shape
            loss = 0.0
            for i in range(H):
                for j in range(H):
                    if i == j:
                        continue
                    loss += F.margin_ranking_loss(
                        preds[:, i], preds[:, j], 
                        torch.sign((y_reg[name] > y_reg[name]).float() - 0.5) * 2,
                        margin=margin
                    )
            rank_loss += loss / (H * (H - 1))

        # === Classification loss ===
        class_loss = 0.0
        for name, logits in class_outputs.items():
            labels = y_class[name]  # shape (B,)
            valid = labels != -1
            if valid.sum() == 0:
                continue
            weight = classification_weights.get(name) if classification_weights else None
            loss = F.cross_entropy(logits[valid], labels[valid], weight=weight)
            class_loss += loss

        loss = rank_loss + class_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
    
def ce_masked_loss(logits, targets, weight=None):
    mask = (targets >= 0)
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], targets[mask], weight=weight, reduction="mean")

def train_epoch_multihead_with_classification(model, loader, optimizer, device, margin=1.0,
                                              class_weights=None):
    """
    Expects each batch like: X, y_reg, mask, <one tensor per classification task in order of CLASSIFICATION_TASKS>
    Model forward returns: (scores, class_outputs) where
        scores: [B, T, H]
        class_outputs: dict(task -> [B, C])
    """
    model.train()
    total_loss = 0.0
    class_acc = {k: [] for k in CLASSIFICATION_TASKS}

    for batch in loader:
        X, y_reg, mask, *rest = batch
        X = X.to(device); y_reg = y_reg.to(device); mask = mask.to(device)
        y_clf = {}
        for i, t in enumerate(CLASSIFICATION_TASKS):
            if i < len(rest):
                y_clf[t] = rest[i].to(device).long()

        optimizer.zero_grad()
        (scores, class_outputs) = model(X)         # scores: [B,T,H]

        # === ranking/regression loss (pairwise margin on mean over heads) ===
        B, T, H = scores.shape
        rank_loss = 0.0
        count = 0
        mean_scores = scores.mean(dim=2)          # [B, T]
        for t_idx in range(T):
            valid = mask[:, t_idx].bool() & ~torch.isnan(y_reg[:, t_idx])
            if valid.sum() < 2:
                continue
            pred = mean_scores[valid, t_idx]
            true = y_reg[valid, t_idx]
            rank_loss += pairwise_margin_loss(pred, true, margin)
            count += 1
        if count > 0:
            rank_loss = rank_loss / count
        else:
            rank_loss = scores.new_zeros(())

        # === constraints (same as your original; using mean over heads) ===
        brightness_pred = mean_scores[:, 4]
        qy_pred = mean_scores[:, 3]
        ext_pred = mean_scores[:, 2]
        brightness_constraint_loss = F.mse_loss(brightness_pred, qy_pred * ext_pred)

        stoke_mask = (mask[:, 0] & mask[:, 1] &
                      ~torch.isnan(y_reg[:, 0]) & ~torch.isnan(y_reg[:, 1]))
        if stoke_mask.sum() >= 2:
            ex_pred = mean_scores[stoke_mask, 0]
            em_pred = mean_scores[stoke_mask, 1]
            stoke_shift_violation = torch.relu(ex_pred - em_pred)
            stoke_shift_constraint_loss = stoke_shift_violation.mean()
        else:
            stoke_shift_constraint_loss = mean_scores.new_zeros(())

        # === classification losses ===
        cls_loss = scores.new_zeros(())
        for t in CLASSIFICATION_TASKS:
            if t not in class_outputs or t not in y_clf:
                continue
            logits = class_outputs[t]
            labels = y_clf[t]
            cw = (class_weights or {}).get(t)
            cls_loss = cls_loss + CLASSIFICATION_LOSS_WEIGHTS.get(t, 1.0) * ce_masked_loss(logits, labels, cw)

            # accuracy (masked)
            with torch.no_grad():
                mask_lab = (labels >= 0)
                if mask_lab.sum() > 0:
                    acc = (logits.argmax(dim=1)[mask_lab] == labels[mask_lab]).float().mean().item()
                    class_acc[t].append(acc)

        # === total loss ===
        alpha1 = 0.0
        alpha2 = 0.1
        loss = rank_loss + alpha1 * brightness_constraint_loss + alpha2 * stoke_shift_constraint_loss + cls_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Quick train-time printout
    print("\n📊 Classification (train) accuracy:")
    for t in CLASSIFICATION_TASKS:
        if class_acc[t]:
            print(f" - {t}: {np.nanmean(class_acc[t]):.3f}")

    return total_loss / len(loader)
    
def eval_epoch_multihead_with_classification(model, val_loader, device):
    model.eval()

    reg_preds_all, reg_trues_all = [], []
    cls_buf = {t: {"preds": [], "targets": []} for t in CLASSIFICATION_TASKS}

    with torch.no_grad():
        for batch in val_loader:
            # flexible unpack
            X, y, mb, *rest = batch
            X = X.to(device); y = y.to(device)

            out = model(X)
            if isinstance(out, tuple):
                scores = out[0]                    # (scores, class_logits) OR scores
                class_logits = out[1] if isinstance(out[0], torch.Tensor) and len(out) > 1 else \
                               (out[1] if isinstance(out[0], torch.Tensor) else out[0][1])
                if isinstance(scores, tuple):      # ((scores, class_logits), hiddens) case
                    scores, class_logits = scores
            else:
                scores, class_logits = out, {}

            # regression: average across heads
            mean_scores = scores.mean(dim=-1).cpu().numpy()   # [B, T]
            reg_preds_all.append(mean_scores)
            reg_trues_all.append(y.cpu().numpy())

            # classification: collect masked preds per task
            for i, t in enumerate(CLASSIFICATION_TASKS):
                if i >= len(rest):
                    continue
                y_cls = rest[i].to(device).long()             # [B], -1 = missing
                if t not in class_logits:
                    continue
                logits = class_logits[t]                      # [B, C]
                preds = logits.argmax(dim=1)                  # [B]
                mask = (y_cls >= 0)
                if mask.any():
                    cls_buf[t]["preds"].append(preds[mask].cpu())
                    cls_buf[t]["targets"].append(y_cls[mask].cpu())

    # stack regression
    reg_preds_all = np.concatenate(reg_preds_all, axis=0)
    reg_trues_all = np.concatenate(reg_trues_all, axis=0)

    # Spearman ρ per target, mean across targets
    rhos = []
    for ti in range(reg_preds_all.shape[1]):
        m = ~np.isnan(reg_trues_all[:, ti])
        if m.sum() >= 2:
            rhos.append(spearmanr(reg_trues_all[m, ti], reg_preds_all[m, ti]).correlation)
        else:
            rhos.append(np.nan)
    val_rho_mean = float(np.nanmean(rhos))

    # classification accuracies in CLASSIFICATION_TASKS order
    class_accs = []
    for t in CLASSIFICATION_TASKS:
        if len(cls_buf[t]["preds"]) == 0:
            class_accs.append(np.nan)
            continue
        preds = torch.cat(cls_buf[t]["preds"]).numpy()
        targs = torch.cat(cls_buf[t]["targets"]).numpy()
        class_accs.append(accuracy_score(targs, preds))

    return val_rho_mean, class_accs
    
def eval_classification_accuracy(model, val_loader, device):
    import numpy as np
    model.eval()
    # We'll accumulate per-task accuracies and average at the end
    acc_lists = {t: [] for t in CLASSIFICATION_TASKS}

    with torch.no_grad():
        for batch in val_loader:
            # Expect: (X, y_reg, mask, class_targets..., [error_targets?])
            X = batch[0].to(device)
            # Number of extra tensors beyond X, y_reg, mask
            n_extra = len(batch) - 3
            class_tensors = []
            if n_extra > 0:
                # take the first n_class tensors (we added them in sorted order in make_loader)
                n_class = min(n_extra, len(CLASSIFICATION_TASKS))
                class_tensors = batch[3:3 + n_class]

            reg_out, class_out = model(X)  # dicts: task -> logits

            # Ensure a consistent order
            cls_names_sorted = sorted(class_out.keys())
            for j, task in enumerate(cls_names_sorted):
                if j >= len(class_tensors):
                    continue
                labels = class_tensors[j].to(device)            # [B]
                logits = class_out[task]                        # [B, C]
                valid = labels != -1
                if valid.sum() == 0:
                    continue
                preds = torch.argmax(logits[valid], dim=1)
                acc = (preds == labels[valid]).float().mean().item()
                acc_lists[task].append(acc)

    # average per task
    accs = [float(np.mean(acc_lists[t])) if acc_lists[t] else float("nan")
            for t in CLASSIFICATION_TASKS]
    return accs