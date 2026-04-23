# models.py
import torch.nn as nn
import torch
from config import TARGET_NAMES, NUM_CLASSES, SPECTRAL_TARGETS
import numpy as np
from config import CLASSIFICATION_TASKS, CLASSIFICATION_DIMS, CLASSIFICATION_LOSS_WEIGHTS



class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, loc_output_dim, ptm_output_dim):
        super().__init__()

        # Shared Encoder: Add more layers with residuals
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads: Deeper heads if needed
        self.loc_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, loc_output_dim)
        )

        self.ptm_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, ptm_output_dim)
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        return self.loc_head(shared), self.ptm_head(shared)

class MultiTaskRegressorRange(nn.Module):
    def __init__(self, input_dim, shared_dims, head_hidden_dim, regression_targets, dropout=0.0):
        super().__init__()
        self.regression_targets = regression_targets

        layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        self.mu_heads = nn.ModuleDict()
        self.log_width_heads = nn.ModuleDict()

        for name in regression_targets:
            self.mu_heads[name] = nn.Sequential(
                nn.Linear(prev_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1)
            )
            self.log_width_heads[name] = nn.Sequential(
                nn.Linear(prev_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1)
            )

    def forward(self, x):
        h = self.shared(x)
        lowers, uppers, mus, log_widths = {}, {}, {}, {}

        for name in self.regression_targets:
            mu = self.mu_heads[name](h).squeeze(1)
            log_w = self.log_width_heads[name](h).squeeze(1)
            width = F.softplus(log_w) + 1e-3  # ensure positive width

            lower = mu - width / 2
            upper = mu + width / 2

            mus[name] = mu
            log_widths[name] = log_w
            lowers[name] = lower
            uppers[name] = upper

        return lowers, uppers, mus, log_widths
        


class MultiTaskRegressorUncertainty(nn.Module):
    def __init__(self, input_dim, shared_dims, head_hidden_dims, regression_targets, dropout=0.0):
        super().__init__()
        self.regression_targets = regression_targets

        # Shared trunk
        layers, prev = [], input_dim
        for dim in shared_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        self.shared = nn.Sequential(*layers)

        # Deeper heads for mu and logvar
        self.mu_heads = nn.ModuleDict()
        self.logvar_heads = nn.ModuleDict()

        for name in regression_targets:
            self.mu_heads[name] = self._make_head(prev, head_hidden_dims, dropout)
            self.logvar_heads[name] = self._make_head(prev, head_hidden_dims, dropout)

    def _make_head(self, input_dim, hidden_dims, dropout):
        """Builds a deeper head network."""
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.shared(x)
        mu_outputs = {}
        logvar_outputs = {}
        for name in self.regression_targets:
            mu_outputs[name] = self.mu_heads[name](h).squeeze(1)        # shape [B]
            logvar_outputs[name] = self.logvar_heads[name](h).squeeze(1)
        return mu_outputs, logvar_outputs

class MultiTaskRegressorRankClass(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[256, 96], head_hidden_dim=64,
                 regression_targets=None, classification_dims=None, dropout=0.0):
        """
        regression_targets: list of task names like ['ex_max', 'em_max', ...]
        classification_dims: dict like {'Ex max': 7, 'Brightness': 4, ...}
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        # Regression/ranking heads
        self.regression_heads = nn.ModuleDict()
        if regression_targets is not None:
            for task in regression_targets:
                self.regression_heads[task] = nn.Sequential(
                    nn.Linear(shared_dims[-1], head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, 1)
                )

        # Classification heads
        self.classification_heads = nn.ModuleDict()
        if classification_dims is not None:
            for name, num_classes in classification_dims.items():
                self.classification_heads[name] = nn.Sequential(
                    nn.Linear(shared_dims[-1], head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, num_classes)
                )

    def forward(self, x):
        shared_out = self.shared(x)

        reg_outputs = {
            name: self.regression_heads[name](shared_out).squeeze(-1)
            for name in self.regression_heads
        }

        class_outputs = {
            name: self.classification_heads[name](shared_out)
            for name in self.classification_heads
        }

        return reg_outputs, class_outputs
        
class MultiTaskRegressorDual(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[256, 96], head_hidden_dim=64):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            self.dropout=0.01
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        self.heads = nn.ModuleDict()
        for task in TARGET_NAMES:
        
            self.heads[task] = nn.Sequential(
                nn.Linear(shared_dims[-1], head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1)
                )

    def forward(self, x):
        shared_out = self.shared(x)
        return torch.cat([self.heads[task](shared_out) for task in TARGET_NAMES], dim=1)

        
class MultiClassErrorClassifier(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[256, 96],
                 head_hidden_dim=64):
        super().__init__()

        # === shared trunk ===
        layers, prev = [], input_dim
        for dim in shared_dims:
            layers += [nn.Linear(prev, dim),
                       nn.BatchNorm1d(dim),
                       nn.ReLU(),
                       nn.Dropout(0.3)
                       ]
            prev = dim
        self.shared = nn.Sequential(*layers)

        # === task-specific heads ===
        self.heads = nn.ModuleDict()
        for task in SPECTRAL_TARGETS:
            n_out = NUM_CLASSES[task]               # 4 for ex & em
            self.heads[task] = nn.Sequential(
                nn.Linear(shared_dims[-1], head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, n_out)   # logits
            )

    def forward(self, x):              # x: [B, input_dim]
        h = self.shared(x)
        # Return concatenated logits [B, Σ n_classes] or dict
        logits = [self.heads[t](h) for t in SPECTRAL_TARGETS]  # list of [B, C_t]
        return torch.cat(logits, dim=1)                    # convenient flat
        
        
class MultiTaskRegressorSelfConditioned(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[1024, 768, 512], head_hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dims = shared_dims

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.shared = nn.Sequential(*layers)

        # Task-specific heads
        self.heads = nn.ModuleDict()
        self.head_input_dim = shared_dims[-1]  # Initial input size for first head
        for i, task in enumerate(TARGET_NAMES):
            input_dim = self.head_input_dim + i  # add i prediction scalars
            self.heads[task] = nn.Sequential(
                nn.Linear(input_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1)
            )

    def forward(self, x):
        shared_out = self.shared(x)
        preds = []
        out = []

        for i, task in enumerate(TARGET_NAMES):
            if preds:
                # Concatenate shared representation with all previous predictions
                conditioning = torch.cat(preds, dim=1)
                head_input = torch.cat([shared_out, conditioning], dim=1)
            else:
                head_input = shared_out

            pred = self.heads[task](head_input)  # shape: (batch, 1)
            preds.append(pred)
            out.append(pred)

        return torch.cat(out, dim=1)  # shape: (batch, num_tasks)
        
        
class SingleHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MultiHeadCalibrator(nn.Module):
    def __init__(self, input_dim, target_names):
        super().__init__()
        self.target_names = target_names
        self.heads = nn.ModuleDict({
            name: SingleHead(input_dim) for name in target_names
        })

    def forward(self, x):
        outputs = []
        for name in self.target_names:
            head = self.heads[name]
            outputs.append(head(x).unsqueeze(-1))
        return torch.cat(outputs, dim=-1)  # shape: (B, num_targets)
        
        
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiTaskRegressorMultiHead(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[256, 96], head_hidden_dim=64, 
                 dropout=0.01, n_outputs_per_task=5, target_names=None, mask_fraction=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.n_outputs_per_task = n_outputs_per_task
        self.target_names = target_names
        self.num_tasks = len(target_names)
        self.mask_fraction = mask_fraction
        self.input_dim = input_dim

        # === Shared layers ===
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, dim))
            shared_layers.append(nn.BatchNorm1d(dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)

        # === Per-task heads ===
        self.head_hidden = nn.ModuleDict()
        self.head_outputs = nn.ModuleDict()

        for task in target_names:
            self.head_hidden[task] = nn.Sequential(
                nn.Linear(shared_dims[-1], head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            )
            self.head_outputs[task] = nn.ModuleList([
                nn.Linear(head_hidden_dim, 1) for _ in range(n_outputs_per_task)
            ])

        self._init_weights()

    def _init_weights(self):
        for task in self.target_names:
            for head in self.head_outputs[task]:
                nn.init.kaiming_uniform_(head.weight, nonlinearity='relu')
                nn.init.zeros_(head.bias)

    def forward(self, x, return_hidden=False, return_stats=False):
        B = x.size(0)
        device = x.device
        outputs = []
        hiddens = {}

        for task in self.target_names:
            task_preds = []

            for _ in range(self.n_outputs_per_task):
                # === Feature masking ===
                x_aug = x.clone()
                if self.training:  # <-- Only apply masking during training
                    n_mask = int(self.mask_fraction * self.input_dim)
                    mask_idx = torch.randperm(self.input_dim)[:n_mask]
                    x_aug[:, mask_idx] = 0.0

                # Shared + task-specific layers
                shared_out = self.shared(x_aug)
                hidden = self.head_hidden[task](shared_out)
                out = self.head_outputs[task][_](hidden)  # [B, 1]
                task_preds.append(out)

            task_out = torch.cat(task_preds, dim=1)  # [B, H]
            outputs.append(task_out.unsqueeze(1))    # [B, 1, H]

            if return_hidden:
                hiddens[task] = hidden

        output_tensor = torch.cat(outputs, dim=1)  # [B, T, H]

        if return_stats:
            mean_preds = output_tensor.mean(dim=2)  # [B, T]
            std_preds = output_tensor.std(dim=2)    # [B, T]
            if return_hidden:
                return output_tensor, mean_preds, std_preds, hiddens
            else:
                return output_tensor, mean_preds, std_preds

        if return_hidden:
            return output_tensor, hiddens
        else:
            return output_tensor
            
class SimpleMLPWithError(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = dim

        # Output 2 values: [prediction, predicted_error]
        layers.append(nn.Linear(prev_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)  # shape: (batch, 2)
        pred = out[:, 0]  # predicted value
        err = torch.abs(out[:, 1])  # enforce positivity
        return pred, err
        
        
def train_multiclass_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0

    for X, y_labels, _ in loader:          # y_labels: int64 with NaN→-1
        X  = X.to(device)
        y  = y_labels.to(device)

        optimizer.zero_grad()
        logits = model(X)                  # [B, ΣC]

        loss_sum, n_tasks = 0.0, 0
        # loop through tasks, slice logits, compute CE only where labels != -1
        for i, task in enumerate(TARGET_NAMES):
            lo, hi = offsets[i], offsets[i+1]
            logit_slice = logits[:, lo:hi]              # [B, C_t]
            labels_t    = y[:, i]                      # [B]

            valid = labels_t != -1                     # -1 marks missing
            if valid.sum() == 0:
                continue

            loss = F.cross_entropy(logit_slice[valid], labels_t[valid])
            loss_sum += loss
            n_tasks  += 1

        if n_tasks > 0:
            (loss_sum / n_tasks).backward()
            optimizer.step()
            running += (loss_sum / n_tasks).item()

    return running / len(loader)
    
    
class mlpCalibrator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),   # <- new
            nn.Linear(hidden_dim, 1)
        )
        # (optional) a little init helps
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # expects shape (N, input_dim)
        return self.net(x)
        
class MultiTaskRegressorMultiHead_wClass(nn.Module):
    def __init__(self, input_dim=1152, shared_dims=[256, 96], head_hidden_dim=64, 
                 dropout=0.01, n_outputs_per_task=5, target_names=None, mask_fraction=0.1):
        super().__init__()
        self.dropout_rate = dropout
        self.n_outputs_per_task = n_outputs_per_task
        self.target_names = target_names
        self.num_tasks = len(target_names)
        self.mask_fraction = mask_fraction
        self.input_dim = input_dim

        # === Shared layers ===
        shared_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, dim))
            shared_layers.append(nn.BatchNorm1d(dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)

        # === Per-task regression heads (multi-head outputs) ===
        self.head_hidden = nn.ModuleDict()
        self.head_outputs = nn.ModuleDict()
        for task in target_names:
            self.head_hidden[task] = nn.Sequential(
                nn.Linear(shared_dims[-1], head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            )
            self.head_outputs[task] = nn.ModuleList([
                nn.Linear(head_hidden_dim, 1) for _ in range(n_outputs_per_task)
            ])

        # === NEW: classification heads (logits) ===
        self.classification_heads = nn.ModuleDict({
            cls_task: nn.Linear(shared_dims[-1], CLASSIFICATION_DIMS[cls_task])
            for cls_task in CLASSIFICATION_TASKS
        })

        self._init_weights()

    def _init_weights(self):
        for task in self.target_names:
            for head in self.head_outputs[task]:
                nn.init.kaiming_uniform_(head.weight, nonlinearity='relu')
                nn.init.zeros_(head.bias)

    def forward(self, x, return_hidden=False, return_stats=False):
        B = x.size(0)
        outputs = []
        hiddens = {}

        # We’ll compute the shared features **once** (no feature masking here);
        # if you really want per-head feature masking, keep your old loop and
        # compute shared inside it—then also compute class logits from the
        # unmasked shared features (shown here).
        shared_out = self.shared(x)

        # Regression multi-heads
        for task in self.target_names:
            hidden = self.head_hidden[task](shared_out)
            task_preds = [head(hidden) for head in self.head_outputs[task]]  # list of [B,1]
            task_out = torch.cat(task_preds, dim=1)                           # [B, H]
            outputs.append(task_out.unsqueeze(1))                              # [B,1,H]
            if return_hidden:
                hiddens[task] = hidden

        output_tensor = torch.cat(outputs, dim=1)  # [B, T, H]

        # Classification logits (one pass)
        class_outputs = {t: self.classification_heads[t](shared_out)
                         for t in self.classification_heads.keys()}  # [B, C_t]

        if return_stats:
            mean_preds = output_tensor.mean(dim=2)  # [B, T]
            std_preds  = output_tensor.std(dim=2)   # [B, T]
            if return_hidden:
                return (output_tensor, class_outputs), mean_preds, std_preds, hiddens
            else:
                return (output_tensor, class_outputs), mean_preds, std_preds

        if return_hidden:
            return (output_tensor, class_outputs), hiddens
        else:
            return (output_tensor, class_outputs)


def classification_accuracy(preds, targets):
    with torch.no_grad():
        preds = torch.argmax(preds, dim=1)
        mask = targets != -1
        if mask.sum() == 0:
            return float('nan')
        correct = (preds[mask] == targets[mask]).sum().item()
        return correct / mask.sum().item()


def train_epoch_multihead_with_classification(
    model, loader, optimizer, device, margin=1.0
):
    model.train()
    total_loss = 0.0
    class_acc = {k: [] for k in CLASSIFICATION_TASKS}
    class_loss_total = {k: 0.0 for k in CLASSIFICATION_TASKS}
    class_counts = {k: 0 for k in CLASSIFICATION_TASKS}

    for batch in loader:
        if len(batch) == 5:
            X, y_reg, mask, y_class, _ = batch
        elif len(batch) == 4:
            X, y_reg, mask, y_class = batch

        X = X.to(device)
        y_reg = y_reg.to(device)
        mask = mask.to(device)
        y_class = y_class.to(device)

        optimizer.zero_grad()
        reg_outputs, class_outputs = model(X)

        # === Ranking loss ===
        rank_loss = 0.0
        for i, task in enumerate(model.target_names):
            preds = reg_outputs[task]  # (B, H)
            B, H = preds.shape
            y = y_reg[:, i]
            task_mask = mask[:, i]
            if task_mask.sum() == 0:
                continue

            loss = 0.0
            for h1 in range(H):
                for h2 in range(H):
                    if h1 == h2:
                        continue
                    diff = torch.sign(y - y).detach()
                    margin_labels = (diff + 1e-6).sign()  # avoid 0 grad
                    loss += F.margin_ranking_loss(
                        preds[:, h1], preds[:, h2], margin_labels, margin=margin, reduction="mean"
                    )
            rank_loss += loss / (H * (H - 1))

        # === Classification loss ===
        class_loss = 0.0
        for j, cls_task in enumerate(CLASSIFICATION_TASKS):
            logits = class_outputs[cls_task]
            labels = y_class[:, j]
            valid = labels != -1
            if valid.sum() == 0:
                continue

            loss = F.cross_entropy(logits[valid], labels[valid])
            class_loss_total[cls_task] += loss.item() * valid.sum().item()
            class_counts[cls_task] += valid.sum().item()
            acc = classification_accuracy(logits[valid], labels[valid])
            class_acc[cls_task].append(acc)

            class_loss += CLASSIFICATION_LOSS_WEIGHTS.get(cls_task, 1.0) * loss

        loss = rank_loss + class_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("\n📊 Classification Accuracy (Train):")
    for k in CLASSIFICATION_TASKS:
        mean_acc = np.nanmean(class_acc[k]) if class_acc[k] else float("nan")
        mean_loss = class_loss_total[k] / class_counts[k] if class_counts[k] else float("nan")
        print(f" - {k}: Acc={mean_acc:.3f}, Loss={mean_loss:.3f}")

    return total_loss / len(loader)
