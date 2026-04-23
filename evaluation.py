import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from pathlib import Path
from config import TARGET_NAMES, NUM_OUTPUTS




def compute_rank_errors(y_true, y_pred):
    rank_errors = []
    for i in range(y_true.shape[1]):
        true_ranks = np.argsort(np.argsort(y_true[:, i]))
        pred_ranks = np.argsort(np.argsort(y_pred[:, i]))
        rank_errors.append(np.abs(true_ranks - pred_ranks))
    return np.stack(rank_errors, axis=1)



def plot_all_heads(y_true, y_pred, target_names, mask=None, output_path="all_heads_prediction_vs_actual.png"):
    # collapse heads if needed
    if y_pred.ndim == 3:
        y_pred = np.nanmean(y_pred, axis=2)
    if y_true.ndim == 3:
        y_true = np.nanmean(y_true, axis=2)

    # ensure 2D
    if y_pred.ndim == 1: y_pred = y_pred[:, None]
    if y_true.ndim == 1: y_true = y_true[:, None]

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Mismatch in number of samples: y_true={y_true.shape}, y_pred={y_pred.shape}")

    num_targets = min(y_true.shape[1], y_pred.shape[1], len(target_names))
    cols, rows = 3, (num_targets + 2) // 3
    plt.figure(figsize=(cols * 5, rows * 4))

    for i in range(num_targets):
        plt.subplot(rows, cols, i + 1)

        # base validity: both finite
        base_valid = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        # include provided mask if any
        if mask is not None:
            base_valid &= mask[:, i].astype(bool)

        if base_valid.sum() < 2:
            plt.title(f"{target_names[i]} (No valid data)")
            plt.xticks([]); plt.yticks([])
            continue

        x = y_true[base_valid, i]
        y = y_pred[base_valid, i]

        plt.scatter(x, y, alpha=0.5, s=10)

        # safe limits
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        y_min, y_max = np.nanmin(y), np.nanmax(y)

        if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
            x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
            y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            lo = min(x_min, y_min)
            hi = max(x_max, y_max)
            plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)
        else:
            plt.title(f"{target_names[i]} (No finite bounds)")
            plt.xticks([]); plt.yticks([])

        plt.xlabel("True Value")
        plt.ylabel("Mean Prediction")
        plt.title(target_names[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Prediction vs Actual (Mean Across Heads)", fontsize=16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"📈 Saved plot to {output_path}")
    
def plot_rank_vs_rank_colored_by_value(y_true, y_pred, target_names, predicted_rank_errors, output_path="rank_vs_rank.png"):
    num_targets = len(target_names)
    cols, rows = 3, (num_targets + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    print(f"🔍 Rank error range for each target:")
    for i, name in enumerate(target_names):
        errs = predicted_rank_errors[:, i]
        print(f"  {name:15}: min={errs.min():.2f}, max={errs.max():.2f}, mean={errs.mean():.2f}")

    for i, name in enumerate(target_names):
        ax = axes[i]

        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        true_ranks = np.argsort(np.argsort(true_vals))
        pred_ranks = np.argsort(np.argsort(pred_vals))

        conf = 1 / (predicted_rank_errors[:, i] + 1e-6)
        log_conf = np.log(conf + 1e-6)

        scaled = (log_conf - log_conf.min()) / (log_conf.max() - log_conf.min())
        sizes = 10 + 90 * (1 - scaled)

        cmap = "turbo" if name in ["ex_max", "em_max"] else "viridis"
        norm = Normalize(vmin=true_vals.min(), vmax=true_vals.max())
        sm = ScalarMappable(norm=norm, cmap=cmap)
        colors = sm.to_rgba(true_vals)

        sc = ax.scatter(true_ranks, pred_ranks, c=colors, s=sizes, alpha=0.6, edgecolors='none')
        ax.plot([0, len(true_vals)], [0, len(true_vals)], 'k--', linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("True Rank")
        ax.set_ylabel("Predicted Rank")
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("True Value")

    for j in range(len(target_names), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Predicted Rank vs True Rank (Colored by Value, Size = Confidence)", fontsize=18)
    fig.savefig(output_path)
    print(f"📈 Saved rank-vs-rank plot to {output_path}")


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    rho = [spearmanr(y_true[:, i], y_pred[:, i]).correlation for i in range(NUM_OUTPUTS)]
    return r2, rho
    
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import os

def evaluate_heads(y_true, y_pred_raw, target_names, calibrated_pred=None, output_dir="graphs"):
    """
    Evaluate accuracy vs head disagreement. If calibrated_pred is provided, use it for accuracy metrics.
    y_pred_raw: (N, T, H)
    calibrated_pred: (N, T) or None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    import os

    if y_pred_raw.ndim != 3:
        raise ValueError("Expected y_pred_raw to have shape (N, T, H)")

    os.makedirs(output_dir, exist_ok=True)

    # Count how many finite heads per (n, t)
    mask_pred_finite = np.isfinite(y_pred_raw)              # (N, T, H)
    heads_count = mask_pred_finite.sum(axis=2)              # (N, T)

    # Aggregate across heads; where there are no finite heads, leave NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_pred = np.nanmean(y_pred_raw, axis=2)          # (N, T)
        std_pred  = np.nanstd (y_pred_raw, axis=2, ddof=0)  # (N, T)

    mean_pred[heads_count == 0] = np.nan
    # std for a single head is 0; if no heads, leave NaN
    std_pred[heads_count == 0] = np.nan
    std_pred[(heads_count == 1) & np.isnan(std_pred)] = 0.0

    final_pred = calibrated_pred if calibrated_pred is not None else mean_pred
    abs_error  = np.abs(final_pred - y_true)

    print("🧠 Per-Target Head Agreement vs Accuracy")
    print("Target             R²     ρ      MAE    std     ρ(err,std)")

    for i, name in enumerate(target_names):
        # Valid only where both true and final_pred are finite
        valid = np.isfinite(y_true[:, i]) & np.isfinite(final_pred[:, i])
        if valid.sum() < 2:
            # Not enough points to compute stable metrics
            continue

        true_vals = y_true[valid, i]
        pred_vals = final_pred[valid, i]
        err_vals  = abs_error[valid, i]
        std_vals  = std_pred[valid, i]

        # Drop any lingering NaNs from std (can happen if no heads)
        m2 = np.isfinite(true_vals) & np.isfinite(pred_vals) & np.isfinite(std_vals) & np.isfinite(err_vals)
        if m2.sum() < 2:
            continue

        true_vals = true_vals[m2]
        pred_vals = pred_vals[m2]
        err_vals  = err_vals[m2]
        std_vals  = std_vals[m2]

        r2  = r2_score(true_vals, pred_vals)
        rho = spearmanr(true_vals, pred_vals).correlation
        mae = float(np.mean(err_vals))
        std_mean = float(np.mean(std_vals))
        rho_err_std = spearmanr(err_vals, std_vals).correlation

        print(f"{name:18} {r2:6.3f}  {rho:5.3f}  {mae:6.1f}  {std_mean:5.2f}   {rho_err_std:6.3f}")

        # === Error vs Std Plot ===
        plt.figure(figsize=(5, 4))
        plt.scatter(std_vals, err_vals, alpha=0.6)
        plt.xlabel("Std Across Heads")
        plt.ylabel("Abs Error")
        plt.title(f"{name} — Error vs Std")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/disagreement_vs_error_{name}.png")
        plt.close()

        # === Std Histogram ===
        plt.figure(figsize=(5, 4))
        plt.hist(std_vals[~np.isnan(std_vals)], bins=30, alpha=0.7)
        plt.xlabel("Std Across Heads")
        plt.ylabel("Count")
        plt.title(f"{name} — Std Histogram")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/head_std_hist_{name}.png")
        plt.close()

    coverages = coverage_within_heads(y_true, y_pred_raw)
    print("True value coverage within head prediction ranges")
    for i, name in enumerate(target_names):
        print(f"  {name:20}: {coverages[i]:.1f}% covered")

        
def coverage_within_heads(y_true, y_pred_heads):
    """
    Compute the % of true values that fall within the range of head predictions for each target.
    """
    n_samples, n_targets, n_heads = y_pred_heads.shape
    coverage = []

    for t in range(n_targets):
        preds_t = y_pred_heads[:, t, :]
        true_t = y_true[:, t]

        # Only consider non-NaN targets
        mask = ~np.isnan(true_t)

        min_preds = preds_t[mask].min(axis=1)
        max_preds = preds_t[mask].max(axis=1)
        truth = true_t[mask]

        in_range = (truth >= min_preds) & (truth <= max_preds)
        percent_covered = 100 * np.mean(in_range)
        coverage.append(percent_covered)

    return coverage
    
def evaluate_heads_single(y_true, y_pred, target_names, output_dir="graphs_single"):
    os.makedirs(output_dir, exist_ok=True)
    abs_error = np.abs(y_pred - y_true)
    
    print("Target             R²     ρ      MAE")
    for i, name in enumerate(target_names):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 2:
            continue

        true_vals = y_true[mask, i]
        pred_vals = y_pred[mask, i]
        err_vals = abs_error[mask, i]

        r2 = r2_score(true_vals, pred_vals)
        rho = spearmanr(true_vals, pred_vals).correlation
        mae = np.mean(err_vals)

        print(f"{name:18} {r2:6.3f}  {rho:5.3f}  {mae:6.1f}")