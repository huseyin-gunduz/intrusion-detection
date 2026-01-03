import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===============================
# 0. TEST KLASÖRÜ VE RUN NUMARASI
# ===============================
base_dir = "tests/transformer_lite"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)

print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")


# ===============================
# 1. VERİYİ YÜKLE
# ===============================
csv_path = (r"C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
            r"/versions/6/cicids2017_cleaned.csv")
df = pd.read_csv(csv_path)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


# ===============================
# 2. HEDEF DEĞİŞKENİ ENCODE ET
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

X = df.drop(columns=["Attack Type"])
y = df["Attack Type"].astype(np.int64)

# Sabit kolonları çıkar
X = X.loc[:, X.nunique() > 1]


# ===============================
# 3. TRAIN / TEST AYIR
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Küçük bir validation (early stopping için)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.01,        # SVM scriptindeki validation_fraction=0.01 mantığı
    random_state=42,
    stratify=y_train
)


# ===============================
# 4. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# label map kaydet (isteğe bağlı ama faydalı)
with open(os.path.join(run_dir, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({int(i): str(c) for i, c in enumerate(label_encoder.classes_)}, f, ensure_ascii=False, indent=2)


# ===============================
# 5. TRANSFORMER-LITE MODEL
# ===============================
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))  # int64

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerLite(nn.Module):
    """
    - Her feature bir token
    - Scalar -> d_model embedding
    - Küçük TransformerEncoder
    - Mean pool -> classification
    """
    def __init__(self, n_features, n_classes, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.scalar_embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (B, F)
        b, f = x.shape
        x = x.view(b, f, 1)                # (B, F, 1)
        tok = self.scalar_embed(x)         # (B, F, d_model)
        tok = tok + self.pos_emb[:, :f, :] # pos
        tok = self.encoder(tok)            # (B, F, d_model)
        pooled = tok.mean(dim=1)           # (B, d_model)
        return self.head(pooled)


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)


# ===============================
# 6. TRAIN (Early stopping)
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
n_features = X_tr.shape[1]
n_classes = len(label_encoder.classes_)

# 8GB GPU için batch: 1024/2048 öneri
BATCH = 2048
EPOCHS = 20
LR = 1e-3
WD = 1e-4
PATIENCE = 3
USE_AMP = True

train_loader = DataLoader(TabDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(TabDataset(X_val, y_val), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(TabDataset(X_test, y_test), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

model = TransformerLite(
    n_features=n_features,
    n_classes=n_classes,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_ff=128,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scaler_amp = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.startswith("cuda")))

print(f"Device: {device}")
print(f"Features: {n_features} | Classes: {n_classes} | Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("checkpoint directory created:", run_dir)

best_val = float("inf")
best_state = None
bad = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    tr_loss_sum = 0.0
    tr_n = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.startswith("cuda"))):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        tr_loss_sum += loss.item() * xb.size(0)
        tr_n += xb.size(0)

    train_loss = tr_loss_sum / max(tr_n, 1)

    # val
    model.eval()
    val_loss_sum = 0.0
    val_n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device.startswith("cuda"))):
                logits = model(xb)
                loss = criterion(logits, yb)
            val_loss_sum += loss.item() * xb.size(0)
            val_n += xb.size(0)

    val_loss = val_loss_sum / max(val_n, 1)
    print(f"Epoch {ep:02d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stopping. Best val_loss={best_val:.5f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# model save
torch.save(
    {"state_dict": model.state_dict(), "n_features": n_features, "n_classes": n_classes},
    os.path.join(run_dir, "model.pt")
)


# ===============================
# 7. TAHMİN + SCORE (ROC/PR için)
# ===============================
model.eval()
all_logits = []
all_y = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.numpy())

logits = np.concatenate(all_logits, axis=0)
y_true = np.concatenate(all_y, axis=0)

y_score = softmax_np(logits)
y_pred = y_score.argmax(axis=1)


# ===============================
# 8. METRİKLER
# ===============================
acc = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

clf_report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)
clf_report_df = pd.DataFrame(clf_report).transpose()


# ===============================
# 9. ROC ve PR (Sınıf bazlı)
# ===============================
y_true_bin = label_binarize(y_true, classes=range(n_classes))

roc_auc_dict = {}
pr_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc_dict[label_encoder.classes_[i]] = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    pr_auc_dict[label_encoder.classes_[i]] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # ROC grafiği
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_dict[label_encoder.classes_[i]]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(run_dir, f'ROC_{label_encoder.classes_[i]}.png'))
    plt.close()

    # PR grafiği
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP={pr_auc_dict[label_encoder.classes_[i]]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(run_dir, f'PR_{label_encoder.classes_[i]}.png'))
    plt.close()


# ===============================
# 10. MICRO / MACRO ROC AUC
# ===============================
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

all_fpr = np.unique(np.concatenate([
    roc_curve(y_true_bin[:, i], y_score[:, i])[0]
    for i in range(n_classes)
]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= n_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

roc_macro_micro_df = pd.DataFrame({
    "Type": ["Micro-average", "Macro-average"],
    "ROC_AUC": [roc_auc_micro, roc_auc_macro]
})


# ===============================
# 11. NORMALIZED CONFUSION MATRIX + GÖRSEL
# ===============================
cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

cm_norm_df = pd.DataFrame(
    cm_norm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

plt.figure(figsize=(9, 7))
plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.colorbar()
ticks = np.arange(len(label_encoder.classes_))
plt.xticks(ticks, label_encoder.classes_, rotation=45, ha="right")
plt.yticks(ticks, label_encoder.classes_)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix_normalized.png"))
plt.close()


# ===============================
# 12. ÖZET TABLOLAR (CSV)
# ===============================
summary_rows = []
summary_rows.append(["Accuracy", acc])
summary_rows.append(["MCC", mcc])

for avg in ["macro avg", "weighted avg"]:
    summary_rows.append([f"{avg} Precision", clf_report_df.loc[avg, "precision"]])
    summary_rows.append([f"{avg} Recall", clf_report_df.loc[avg, "recall"]])
    summary_rows.append([f"{avg} F1-score", clf_report_df.loc[avg, "f1-score"]])

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

roc_pr_df = pd.DataFrame({
    "Class": label_encoder.classes_,
    "ROC_AUC": [roc_auc_dict[c] for c in label_encoder.classes_],
    "PR_AUC": [pr_auc_dict[c] for c in label_encoder.classes_],
})


# ===============================
# 13. ÇIKTILARI KAYDET
# ===============================
print("\n========== Transformer-Lite Detailed ==========")
print(f"Accuracy: {acc:.6f}")
print(f"MCC: {mcc:.6f}")
print(f"ROC AUC (Micro): {roc_auc_micro:.4f}")
print(f"ROC AUC (Macro): {roc_auc_macro:.4f}")

print("\n--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

print("\n--- Classification Report ---")
print(clf_report_df)

print("\n--- ROC AUC (her sınıf) ---")
for cls, score in roc_auc_dict.items():
    print(f"{cls}: {score:.4f}")

print("\n--- PR AUC (her sınıf) ---")
for cls, score in pr_auc_dict.items():
    print(f"{cls}: {score:.4f}")

pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
    os.path.join(run_dir, "confusion_matrix.csv")
)
cm_norm_df.to_csv(os.path.join(run_dir, "confusion_matrix_normalized.csv"))

clf_report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
summary_df.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)

pd.DataFrame.from_dict(roc_auc_dict, orient='index', columns=["ROC AUC"]).to_csv(
    os.path.join(run_dir, "roc_auc.csv")
)
pd.DataFrame.from_dict(pr_auc_dict, orient='index', columns=["PR AUC"]).to_csv(
    os.path.join(run_dir, "pr_auc.csv")
)

roc_pr_df.to_csv(os.path.join(run_dir, "roc_pr_per_class.csv"), index=False)
roc_macro_micro_df.to_csv(os.path.join(run_dir, "roc_macro_micro.csv"), index=False)

with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("Transformer-Lite (TabTransformer-lite) Detailed Summary\n")
    f.write("===============================================\n\n")
    f.write(f"CSV: {csv_path}\n")
    f.write(f"Train size: {len(X_tr)}\n")
    f.write(f"Val size: {len(X_val)}\n")
    f.write(f"Test size: {len(X_test)}\n")
    f.write(f"Features: {n_features}\n")
    f.write(f"Classes: {n_classes}\n\n")

    f.write("Model Params\n")
    f.write("------------\n")
    f.write("d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1\n")
    f.write(f"epochs={EPOCHS}, batch={BATCH}, lr={LR}, wd={WD}, patience={PATIENCE}\n\n")

    f.write("Global Metrics\n")
    f.write("--------------\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"MCC: {mcc:.6f}\n")
    f.write(f"ROC AUC (Micro): {roc_auc_micro:.4f}\n")
    f.write(f"ROC AUC (Macro): {roc_auc_macro:.4f}\n\n")

    f.write("Macro / Weighted Averages\n")
    f.write("-------------------------\n")
    for avg in ["macro avg", "weighted avg"]:
        f.write(f"{avg} Precision: {clf_report_df.loc[avg, 'precision']:.4f}\n")
        f.write(f"{avg} Recall: {clf_report_df.loc[avg, 'recall']:.4f}\n")
        f.write(f"{avg} F1-score: {clf_report_df.loc[avg, 'f1-score']:.4f}\n\n")

    f.write("ROC AUC per Class\n")
    f.write("-----------------\n")
    for cls, score in roc_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")

    f.write("\nPR AUC per Class\n")
    f.write("----------------\n")
    for cls, score in pr_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")

print(f"\nTüm sonuçlar '{run_dir}' klasörüne kaydedildi.")
print("Oluşan dosyalar:")
print("- model.pt / label_map.json")
print("- confusion_matrix.csv")
print("- confusion_matrix_normalized.csv / confusion_matrix_normalized.png")
print("- classification_report.csv")
print("- summary_metrics.csv")
print("- roc_auc.csv / pr_auc.csv")
print("- roc_pr_per_class.csv")
print("- roc_macro_micro.csv")
print("- ROC_*.png / PR_*.png")
print("- summary.txt")
