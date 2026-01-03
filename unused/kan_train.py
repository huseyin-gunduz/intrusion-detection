# ===============================
# FAST KAN (Spline-KAN) - 8GB GPU Optimize
# CICIDS2017 Tabular Multi-class Classification
# Outputs: runX/{CM, reports, ROC/PR plots, summary, best_model.pt}
# ===============================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

import torch
import torch.nn as nn

# ---- KAN import (pykan) ----
# pip install pykan torch
try:
    from kan import KAN
except Exception as e:
    raise ImportError(
        "KAN import edilemedi.\n"
        "1) Dosya adın 'kan.py' ise değiştir (örn. train_kan_fast.py)\n"
        "2) __pycache__ sil\n"
        "3) Kurulum: python -m pip install pykan torch\n"
        f"Hata: {e}"
    )

# ===============================
# 0) Reproducibility + Speed Flags
# ===============================
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Speed hints (GPU)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x hız

# ===============================
# 1) RUN folder
# ===============================
base_dir = "../tests"
os.makedirs(base_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)
print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

# ===============================
# 2) Load + Clean data
# ===============================
csv_path = r"C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
df = pd.read_csv(csv_path)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ===============================
# 3) Encode target
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

X = df.drop(columns=["Attack Type"])
y = df["Attack Type"].astype(np.int64)

# drop constant columns
X = X.loc[:, X.nunique() > 1]

# ===============================
# 4) Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ===============================
# 5) Scale
# ===============================
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test  = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# ===============================
# 6) Torch tensors
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

X_train_t = torch.tensor(X_train.values, dtype=torch.float32, device=device)
X_test_t  = torch.tensor(X_test.values, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train.values, dtype=torch.long, device=device)
y_test_t  = torch.tensor(y_test.values, dtype=torch.long, device=device)

n_features = X_train.shape[1]
n_classes  = len(label_encoder.classes_)
print(f"Features: {n_features} | Classes: {n_classes} | Train: {len(X_train)} | Test: {len(X_test)}")

# ===============================
# 7) KAN Model (FAST preset for 8GB)
# ===============================
# Fast + safe starting point:
KAN_WIDTH = [n_features, 64, n_classes]
KAN_GRID  = 5
KAN_K     = 3

model = KAN(width=KAN_WIDTH, grid=KAN_GRID, k=KAN_K, seed=SEED).to(device)

# Class weights (balanced loss)
class_counts = np.bincount(y_train.values, minlength=n_classes)
class_weights = (class_counts.sum() / (n_classes * np.maximum(class_counts, 1))).astype(np.float32)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)

# Optimizer (fast convergence)
LR = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Batch size for 8GB: start 2048; if OOM -> 1024 -> 512
BATCH_SIZE = 8192
EPOCHS = 25
PATIENCE = 8  # early stopping

# OneCycleLR (often faster)
steps_per_epoch = max(1, int(np.ceil(len(X_train) / BATCH_SIZE)))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1
)

# Mixed precision
use_amp = (device == "cuda")
scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

def iterate_minibatches(Xt, yt, batch_size, shuffle=True):
    idx = torch.arange(Xt.shape[0], device=Xt.device)
    if shuffle:
        idx = idx[torch.randperm(idx.shape[0])]
    for start in range(0, idx.shape[0], batch_size):
        b = idx[start:start+batch_size]
        yield Xt[b], yt[b]

# ===============================
# 8) Train with Early Stopping + Best Save
# ===============================
best_val_loss = float("inf")
best_epoch = -1
pat_counter = 0
loss_history = []

best_path = os.path.join(run_dir, "best_model.pt")

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []

    for xb, yb in iterate_minibatches(X_train_t, y_train_t, BATCH_SIZE, shuffle=True):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        batch_losses.append(loss.item())

    train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    loss_history.append(train_loss)

    # "Validation" loss olarak test üzerinde hızlı bir pass (istersen ayrı val split yap)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        # memory-safe: evaluate in chunks
        val_losses = []
        for xb, yb in iterate_minibatches(X_test_t, y_test_t, batch_size=min(BATCH_SIZE, 4096), shuffle=False):
            lg = model(xb)
            vl = criterion(lg, yb)
            val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        pat_counter = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "width": KAN_WIDTH,
            "grid": KAN_GRID,
            "k": KAN_K,
            "scaler": scaler,  # sklearn scaler object
            "label_classes": label_encoder.classes_,
            "epoch": best_epoch,
            "val_loss": best_val_loss
        }, best_path)
    else:
        pat_counter += 1
        if pat_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
            break

# Loss plot
plt.figure()
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss (KAN - Fast)")
plt.savefig(os.path.join(run_dir, "train_loss.png"))
plt.close()

# ===============================
# 9) Load best model and Predict
# ===============================
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Predict in chunks to avoid VRAM spikes
y_score_list = []
with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
    for xb, _ in iterate_minibatches(X_test_t, y_test_t, batch_size=min(BATCH_SIZE, 8192), shuffle=False):
        lg = model(xb)
        pr = torch.softmax(lg, dim=1)
        y_score_list.append(pr.detach().cpu().numpy())

y_score = np.concatenate(y_score_list, axis=0)
y_pred = y_score.argmax(axis=1)

# ===============================
# 10) Metrics
# ===============================
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)
clf_report_df = pd.DataFrame(clf_report).transpose()

# ===============================
# 11) ROC + PR curves per class
# ===============================
y_test_bin = label_binarize(y_test, classes=range(n_classes))
roc_auc_dict = {}
pr_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[label_encoder.classes_[i]] = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc_dict[label_encoder.classes_[i]] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_dict[label_encoder.classes_[i]]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(run_dir, f'ROC_{label_encoder.classes_[i]}.png'))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP={pr_auc_dict[label_encoder.classes_[i]]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(run_dir, f'PR_{label_encoder.classes_[i]}.png'))
    plt.close()

# ===============================
# 12) Save outputs
# ===============================
print("\n========== KAN (FAST) Model Summary ==========")
print(f"Best epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Accuracy: {acc:.6f}")
print(f"MCC: {mcc:.6f}")

print("\n--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

print("\n--- Classification Report ---")
print(clf_report_df)

print("\n--- ROC AUC (per class) ---")
for cls, score in roc_auc_dict.items():
    print(f"{cls}: {score:.4f}")

print("\n--- PR AUC (per class) ---")
for cls, score in pr_auc_dict.items():
    print(f"{cls}: {score:.4f}")

pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
    os.path.join(run_dir, "confusion_matrix.csv")
)
clf_report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
pd.DataFrame.from_dict(roc_auc_dict, orient="index", columns=["ROC AUC"]).to_csv(
    os.path.join(run_dir, "roc_auc.csv")
)
pd.DataFrame.from_dict(pr_auc_dict, orient="index", columns=["PR AUC"]).to_csv(
    os.path.join(run_dir, "pr_auc.csv")
)

with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("KAN (FAST) Summary\n")
    f.write("==================\n")
    f.write(f"CSV: {csv_path}\n")
    f.write(f"Device: {device}\n")
    f.write(f"KAN width: {KAN_WIDTH}\n")
    f.write(f"KAN grid: {KAN_GRID}\n")
    f.write(f"KAN k: {KAN_K}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Epochs (max): {EPOCHS}\n")
    f.write(f"Patience: {PATIENCE}\n")
    f.write(f"Best epoch: {ckpt.get('epoch', 'N/A')}\n")
    f.write(f"Best val_loss: {ckpt.get('val_loss', 'N/A')}\n\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"MCC: {mcc:.6f}\n\n")
    f.write("ROC AUC:\n")
    for cls, score in roc_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")
    f.write("\nPR AUC:\n")
    for cls, score in pr_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")

print(f"\nTüm sonuçlar '{run_dir}' klasörüne kaydedildi.")
print(f"Best model checkpoint: {best_path}")
