import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import SGDClassifier

# ===============================
# 0) RUN KLASÖRÜ (seninki gibi)
# ===============================
base_dir = "tests/unsw_nb15_svm"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)

print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

# ===============================
# 1) VERİYİ YÜKLE (PARQUET)
# ===============================
train_path = r"C:\Users\Admin\.cache\kagglehub\datasets\dhoogla\unswnb15\versions\5\UNSW_NB15_training-set.parquet"
test_path = r"C:\Users\Admin\.cache\kagglehub\datasets\dhoogla\unswnb15\versions\5\UNSW_NB15_testing-set.parquet"

train_df = pd.read_parquet(train_path)  # pyarrow/fastparquet gerekir
test_df = pd.read_parquet(test_path)

for df in (train_df, test_df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

# ===============================
# 2) HEDEF SEÇ
# ===============================
TARGET = "attack_cat"  # multiclass
# TARGET = "label"      # binary istersen

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError(
        f"TARGET '{TARGET}' kolonunu bulamadım.\n"
        f"Train kolon örnek: {train_df.columns.tolist()[:40]}"
    )

y_train_raw = train_df[TARGET]
y_test_raw = test_df[TARGET]

X_train = train_df.drop(columns=[TARGET])
X_test = test_df.drop(columns=[TARGET])

# Eğer diğer hedef kolonu feature olarak kalmışsa çıkar (UNSW'de sık)
for col in ["label", "attack_cat"]:
    if col != TARGET and col in X_train.columns:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

# ===============================
# 3) KATEGORİK KOLONLARI ENCODE ET
# ===============================
cat_cols = [c for c in ["proto", "service", "state"] if c in X_train.columns]

for col in cat_cols:
    le = LabelEncoder()
    full = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(full)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ===============================
# 4) LABEL ENCODER (HEDEF)
# ===============================
y_le = LabelEncoder()
y_train = y_le.fit_transform(y_train_raw.astype(str))
y_test = y_le.transform(y_test_raw.astype(str))

class_names = y_le.classes_
n_classes = len(class_names)

# ===============================
# 5) SABİT KOLONLARI AT + HİZALA
# ===============================
X_train = X_train.loc[:, X_train.nunique(dropna=False) > 1]
X_test = X_test.reindex(columns=X_train.columns)

# Numeric'e zorla (parquet bazen object/bool gelebiliyor)
X_train = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

# ===============================
# 6) SCALE
# ===============================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# ===============================
# 7) MODEL (ÖNEMLİ DEĞİŞİKLİK): log_loss -> predict_proba var
# ===============================
model = SGDClassifier(
    loss="log_loss",  # <-- hinge yerine bunu kullan
    penalty="l2",
    alpha=1e-4,
    max_iter=30,
    tol=1e-3,
    n_jobs=-1,
    class_weight="balanced",
    early_stopping=True,
    validation_fraction=0.01,
    n_iter_no_change=3,
    random_state=42
)
model.fit(X_train_s, y_train)

# ===============================
# 8) TAHMİN + PROB
# ===============================
y_pred = model.predict(X_test_s)
y_score = model.predict_proba(X_test_s)  # (n, C) gerçek olasılık

# ===============================
# 9) METRİKLER
# ===============================
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)
clf_report_df = pd.DataFrame(clf_report).transpose()

# ===============================
# 10) ROC ve PR (sınıf bazlı) + PNG
# ===============================
y_test_bin = label_binarize(y_test, classes=range(n_classes))

roc_auc_dict = {}
pr_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[class_names[i]] = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc_dict[class_names[i]] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # ROC grafiği
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_dict[class_names[i]]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {class_names[i]}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(run_dir, f'ROC_{class_names[i]}.png'))
    plt.close()

    # PR grafiği
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AP={pr_auc_dict[class_names[i]]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {class_names[i]}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(run_dir, f'PR_{class_names[i]}.png'))
    plt.close()

# ===============================
# 11) MICRO / MACRO ROC AUC (seninki gibi)
# ===============================
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

all_fpr = np.unique(np.concatenate([
    roc_curve(y_test_bin[:, i], y_score[:, i])[0]
    for i in range(n_classes)
]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= n_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

# ===============================
# 12) NORMALIZED CONFUSION MATRIX + PNG/CSV
# ===============================
cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

cm_norm_df = pd.DataFrame(
    cm_norm,
    index=class_names,
    columns=class_names
)

plt.figure(figsize=(9, 7))
plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Normalized Confusion Matrix")
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix_normalized.png"))
plt.close()

# ===============================
# 13) ÖZET TABLOLAR (CSV)
# ===============================
summary_rows = []
summary_rows.append(["Accuracy", acc])
summary_rows.append(["MCC", mcc])
summary_rows.append(["ROC AUC (Micro)", roc_auc_micro])
summary_rows.append(["ROC AUC (Macro)", roc_auc_macro])

for avg in ["macro avg", "weighted avg"]:
    summary_rows.append([f"{avg} Precision", clf_report_df.loc[avg, "precision"]])
    summary_rows.append([f"{avg} Recall", clf_report_df.loc[avg, "recall"]])
    summary_rows.append([f"{avg} F1-score", clf_report_df.loc[avg, "f1-score"]])

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

roc_pr_df = pd.DataFrame({
    "Class": class_names,
    "ROC_AUC": [roc_auc_dict[c] for c in class_names],
    "PR_AUC": [pr_auc_dict[c] for c in class_names],
})

roc_macro_micro_df = pd.DataFrame({
    "Type": ["Micro-average", "Macro-average"],
    "ROC_AUC": [roc_auc_micro, roc_auc_macro]
})

# ===============================
# 14) DOSYALARA KAYDET (tam liste)
# ===============================
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    os.path.join(run_dir, "confusion_matrix.csv")
)
cm_norm_df.to_csv(os.path.join(run_dir, "confusion_matrix_normalized.csv"))

clf_report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
summary_df.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)

pd.DataFrame.from_dict(roc_auc_dict, orient="index", columns=["ROC AUC"]).to_csv(
    os.path.join(run_dir, "roc_auc.csv")
)
pd.DataFrame.from_dict(pr_auc_dict, orient="index", columns=["PR AUC"]).to_csv(
    os.path.join(run_dir, "pr_auc.csv")
)

roc_pr_df.to_csv(os.path.join(run_dir, "roc_pr_per_class.csv"), index=False)
roc_macro_micro_df.to_csv(os.path.join(run_dir, "roc_macro_micro.csv"), index=False)

# Detaylı summary.txt (seninki gibi)
with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("UNSW-NB15 - SGDClassifier (log_loss) Detailed Summary\n")
    f.write("====================================================\n\n")
    f.write(f"Train: {train_path}\n")
    f.write(f"Test : {test_path}\n")
    f.write(f"Target: {TARGET}\n\n")
    f.write(f"Train size: {len(X_train_s)}\n")
    f.write(f"Test size: {len(X_test_s)}\n")
    f.write(f"Features: {X_train_s.shape[1]}\n")
    f.write(f"Classes: {n_classes}\n\n")

    f.write("Model Params\n")
    f.write("------------\n")
    f.write("loss=log_loss, penalty=l2, alpha=1e-4, max_iter=30, tol=1e-3\n")
    f.write("class_weight=balanced, early_stopping=True, validation_fraction=0.01, n_iter_no_change=3\n\n")

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

# ===============================
# 15) CONSOLE ÖZETİ (seninki gibi)
# ===============================
print("\n========== SGD (log_loss) Detailed ==========")
print(f"Accuracy: {acc:.6f}")
print(f"MCC: {mcc:.6f}")
print(f"ROC AUC (Micro): {roc_auc_micro:.4f}")
print(f"ROC AUC (Macro): {roc_auc_macro:.4f}")

print("\n--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

print("\n--- Classification Report ---")
print(clf_report_df)

print("\n--- ROC AUC (her sınıf) ---")
for cls, score in roc_auc_dict.items():
    print(f"{cls}: {score:.4f}")

print("\n--- PR AUC (her sınıf) ---")
for cls, score in pr_auc_dict.items():
    print(f"{cls}: {score:.4f}")

print(f"\nTüm sonuçlar '{run_dir}' klasörüne kaydedildi.")
print("Oluşan dosyalar:")
print("- confusion_matrix.csv")
print("- confusion_matrix_normalized.csv / confusion_matrix_normalized.png")
print("- classification_report.csv")
print("- summary_metrics.csv")
print("- roc_auc.csv / pr_auc.csv")
print("- roc_pr_per_class.csv")
print("- roc_macro_micro.csv")
print("- ROC_*.png / PR_*.png")
print("- summary.txt")
