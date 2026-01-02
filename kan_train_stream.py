# ===============================
# KAN - 20M satır için STREAM TRAIN (CSV chunks)
# - RAM'e komple yüklemez
# - Online standardization (running mean/var)
# - fp16 (cuda) + AdamW + OneCycleLR
# - Optional: holdout test sample for ROC/PR
# ===============================

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

import torch
import torch.nn as nn

try:
    from kan import KAN
except Exception as e:
    raise ImportError(
        "KAN import edilemedi.\n"
        "1) Dosya adın 'kan.py' ise değiştir\n"
        "2) __pycache__ sil\n"
        "3) Kurulum: python -m pip install pykan torch\n"
        f"Hata: {e}"
    )

# ===============================
# 0) Config
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

CSV_PATH = r"C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
TARGET_COL = "Attack Type"

# 20M satırda CSV tek dosya ise chunk şart
CHUNK_ROWS = 200_000          # 100k-500k arası ideal (RAM'e göre)
BATCH_SIZE = 1024             # 8GB için güvenli
EPOCHS = 5                    # büyük veri => az epoch + çok step
STEPS_PER_EPOCH = 20_000      # epoch başına kaç batch işleyeceğiz (hız kontrol)
LR = 1e-3

# Holdout test sample (ROC/PR için küçük örneklem)
TEST_SAMPLE_N = 200_000       # 0 yaparsan eval atlar

# KAN hızlı preset
KAN_HIDDEN = 32               # 32 hızlı, 64 daha güçlü ama yavaş
KAN_GRID = 4                  # 4 hızlı, 5 daha güçlü
KAN_K = 3

# ===============================
# 1) Run folder
# ===============================
base_dir = "tests"
os.makedirs(base_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)
print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
print("Device:", device)

# ===============================
# 2) First pass: get columns + fit LabelEncoder (stream)
#    (Attack Type string -> int)
# ===============================
# CSV'yi ilk chunk ile kolonları al
first = pd.read_csv(CSV_PATH, nrows=5)
all_cols = list(first.columns)
if TARGET_COL not in all_cols:
    raise ValueError(f"Target column '{TARGET_COL}' bulunamadı. Kolonlar: {all_cols}")

feature_cols = [c for c in all_cols if c != TARGET_COL]

# LabelEncoder için etiketleri stream ederek topla (memory-safe)
# Not: Çok sınıf sayısı küçükse hızlı olur.
label_encoder = LabelEncoder()
labels_seen = []

print("LabelEncoder için etiketler taranıyor...")
for chunk in pd.read_csv(CSV_PATH, usecols=[TARGET_COL], chunksize=CHUNK_ROWS):
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    labels_seen.append(chunk[TARGET_COL].astype(str).values)

labels_seen = np.concatenate(labels_seen, axis=0)
label_encoder.fit(labels_seen)
classes = label_encoder.classes_
n_classes = len(classes)
print("Classes:", n_classes)

# ===============================
# 3) Online StandardScaler (running mean/var) - Welford
# ===============================
class RunningStandardizer:
    def __init__(self, n_features, eps=1e-6):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.eps = eps

    def update(self, X):
        # X: (m, d) float64
        for x in X:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        var = self.M2 / max(self.n - 1, 1)
        std = np.sqrt(var + self.eps)
        return self.mean.astype(np.float32), std.astype(np.float32)

# 20M için tüm veriyle scaler fit yapmak mümkün ama zaman alır.
# Pratik: İlk ~1-2M satırla fit.
SCALER_FIT_ROWS = 1_000_000

print(f"Scaler fit (online) ~{SCALER_FIT_ROWS} satır...")
rs = RunningStandardizer(n_features=len(feature_cols))
seen = 0

for chunk in pd.read_csv(CSV_PATH, usecols=feature_cols + [TARGET_COL], chunksize=CHUNK_ROWS):
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    # sadece featureları al
    Xc = chunk[feature_cols].astype(np.float32).values
    # sabit kolonları burada atlamak zor; CICIDS'te zaten sabit az.
    rs.update(Xc.astype(np.float64))
    seen += len(Xc)
    if seen >= SCALER_FIT_ROWS:
        break

mean, std = rs.finalize()
np.save(os.path.join(run_dir, "scaler_mean.npy"), mean)
np.save(os.path.join(run_dir, "scaler_std.npy"), std)
print("Scaler hazır. mean/std kaydedildi.")

# ===============================
# 4) Build KAN
# ===============================
n_features = len(feature_cols)

model = KAN(width=[n_features, KAN_HIDDEN, n_classes], grid=KAN_GRID, k=KAN_K, seed=SEED).to(device)

# Class weights (stream ile yaklaşık hesap)
# İlk 2M satırdan class frekansı yeterli
FREQ_ROWS = 2_000_000
counts = np.zeros(n_classes, dtype=np.int64)
seen = 0
print(f"Class frekansı (~{FREQ_ROWS} satır) hesaplanıyor...")

for chunk in pd.read_csv(CSV_PATH, usecols=[TARGET_COL], chunksize=CHUNK_ROWS):
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    y_str = chunk[TARGET_COL].astype(str).values
    y_enc = label_encoder.transform(y_str)
    binc = np.bincount(y_enc, minlength=n_classes)
    counts += binc
    seen += len(y_enc)
    if seen >= FREQ_ROWS:
        break

class_weights = (counts.sum() / (n_classes * np.maximum(counts, 1))).astype(np.float32)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# OneCycleLR: total_steps = EPOCHS * STEPS_PER_EPOCH
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start=0.1
)

scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

# ===============================
# 5) Helper: batch generator from chunks
# ===============================
def standardize_np(X):
    return (X - mean) / std

def chunk_to_batches(chunk_df):
    # clean
    chunk_df = chunk_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(chunk_df) == 0:
        return

    # X, y
    Xnp = chunk_df[feature_cols].astype(np.float32).values
    Xnp = standardize_np(Xnp)

    y_str = chunk_df[TARGET_COL].astype(str).values
    ynp = label_encoder.transform(y_str).astype(np.int64)

    # shuffle within chunk for better SGD
    idx = np.random.permutation(len(Xnp))
    Xnp = Xnp[idx]
    ynp = ynp[idx]

    # yield mini-batches
    for start in range(0, len(Xnp), BATCH_SIZE):
        xb = Xnp[start:start+BATCH_SIZE]
        yb = ynp[start:start+BATCH_SIZE]
        if len(xb) < 2:
            continue
        yield xb, yb

# ===============================
# 6) Optional: collect a fixed TEST sample for evaluation
# ===============================
X_test_sample = None
y_test_sample = None

if TEST_SAMPLE_N > 0:
    print(f"Eval için test örneklemi toplanıyor (n={TEST_SAMPLE_N})...")
    X_buf = []
    y_buf = []
    total = 0
    for chunk in pd.read_csv(CSV_PATH, usecols=feature_cols + [TARGET_COL], chunksize=CHUNK_ROWS):
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        # random sample from chunk
        if len(chunk) == 0:
            continue
        take = min(TEST_SAMPLE_N - total, max(1000, int(0.05 * len(chunk))))
        samp = chunk.sample(n=min(take, len(chunk)), random_state=SEED)
        Xnp = standardize_np(samp[feature_cols].astype(np.float32).values)
        y_str = samp[TARGET_COL].astype(str).values
        ynp = label_encoder.transform(y_str).astype(np.int64)

        X_buf.append(Xnp)
        y_buf.append(ynp)
        total += len(ynp)
        if total >= TEST_SAMPLE_N:
            break

    X_test_sample = np.concatenate(X_buf, axis=0)[:TEST_SAMPLE_N]
    y_test_sample = np.concatenate(y_buf, axis=0)[:TEST_SAMPLE_N]
    print("Eval sample hazır:", X_test_sample.shape, y_test_sample.shape)

# ===============================
# 7) Train (streaming)
# ===============================
print("\nTRAIN başladı (stream)...")

global_step = 0
epoch_logs = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    steps = 0

    # Her epoch'ta CSV'yi baştan stream ediyoruz (20M için normal)
    for chunk in pd.read_csv(CSV_PATH, usecols=feature_cols + [TARGET_COL], chunksize=CHUNK_ROWS):
        for xb_np, yb_np in chunk_to_batches(chunk):
            xb = torch.tensor(xb_np, dtype=torch.float32, device=device)
            yb = torch.tensor(yb_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            running_loss += loss.item()
            steps += 1
            global_step += 1

            if steps >= STEPS_PER_EPOCH:
                break
        if steps >= STEPS_PER_EPOCH:
            break

    train_loss = running_loss / max(steps, 1)
    print(f"Epoch {epoch}/{EPOCHS} | steps={steps} | train_loss={train_loss:.6f}")

    # Quick eval on sample
    if X_test_sample is not None:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            # batch predict
            y_score_list = []
            for start in range(0, len(X_test_sample), 8192):
                xb = torch.tensor(X_test_sample[start:start+8192], dtype=torch.float32, device=device)
                lg = model(xb)
                pr = torch.softmax(lg, dim=1).detach().cpu().numpy()
                y_score_list.append(pr)
            y_score = np.concatenate(y_score_list, axis=0)
            y_pred = y_score.argmax(axis=1)

        acc = accuracy_score(y_test_sample, y_pred)
        mcc = matthews_corrcoef(y_test_sample, y_pred)
        print(f"  Eval(sample) acc={acc:.4f} | mcc={mcc:.4f}")
        epoch_logs.append((epoch, train_loss, acc, mcc))
    else:
        epoch_logs.append((epoch, train_loss, None, None))

# Save train log
log_df = pd.DataFrame(epoch_logs, columns=["epoch", "train_loss", "eval_acc_sample", "eval_mcc_sample"])
log_df.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

plt.figure()
plt.plot(log_df["epoch"], log_df["train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss (KAN Stream)")
plt.savefig(os.path.join(run_dir, "train_loss.png"))
plt.close()

# Save model
best_path = os.path.join(run_dir, "model_final.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "width": [n_features, KAN_HIDDEN, n_classes],
    "grid": KAN_GRID,
    "k": KAN_K,
    "label_classes": classes,
    "feature_cols": feature_cols,
    "mean": mean,
    "std": std
}, best_path)

print("\nModel kaydedildi:", best_path)

# ===============================
# 8) Full evaluation (OPTIONAL)
# ===============================
# 20M için full test 4M vs çok uzun. Sample ile rapor yeterli.
# İstersen burada full test stream evaluation da yazılır.
if X_test_sample is not None:
    y_true = y_test_sample
    y_test_bin = label_binarize(y_true, classes=range(n_classes))

    roc_auc_dict = {}
    pr_auc_dict = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_dict[classes[i]] = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc_dict[classes[i]] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_dict[classes[i]]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {classes[i]} (sample)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(run_dir, f'ROC_{classes[i]}_sample.png'))
        plt.close()

        plt.figure()
        plt.plot(recall, precision, label=f'PR (AP={pr_auc_dict[classes[i]]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {classes[i]} (sample)')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(run_dir, f'PR_{classes[i]}_sample.png'))
        plt.close()

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(run_dir, "confusion_matrix_sample.csv"))

    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    rep_df.to_csv(os.path.join(run_dir, "classification_report_sample.csv"))

    pd.DataFrame.from_dict(roc_auc_dict, orient="index", columns=["ROC AUC"]).to_csv(os.path.join(run_dir, "roc_auc_sample.csv"))
    pd.DataFrame.from_dict(pr_auc_dict, orient="index", columns=["PR AUC"]).to_csv(os.path.join(run_dir, "pr_auc_sample.csv"))

print("\nBitti. Çıktılar:", run_dir)
