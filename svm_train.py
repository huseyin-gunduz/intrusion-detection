import os
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
from sklearn.linear_model import SGDClassifier

# ===============================
# 0. TEST KLASÖRÜ VE RUN NUMARASI
# ===============================
base_dir = "tests/svm"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)

print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

# ===============================
# 1. VERİYİ YÜKLE
# ===============================
csv_path = r"C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv"
df = pd.read_csv(csv_path)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ===============================
# 2. HEDEF DEĞİŞKENİ ENCODE ET
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

X = df.drop(columns=["Attack Type"])
y = df["Attack Type"].astype(np.int32)

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

# ===============================
# 4. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# ===============================
# 5. LINEAR SVM (SGD) - CPU
# ===============================
model = SGDClassifier(
    loss="hinge",             # Linear SVM
    penalty="l2",
    alpha=1e-4,
    max_iter=20,
    tol=1e-3,
    n_jobs=-1,
    class_weight="balanced",
    early_stopping=True,
    validation_fraction=0.01,
    n_iter_no_change=3,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. TAHMİN + SCORE (ROC/PR için)
# ===============================
y_pred = model.predict(X_test)

scores = model.decision_function(X_test)
# Binary ise (n,), multiclass ise (n, C)
if scores.ndim == 1:
    scores = np.vstack([-scores, scores]).T

# Pseudo-probability: softmax(decision_function)
scores = scores - scores.max(axis=1, keepdims=True)
y_score = np.exp(scores)
y_score = y_score / (y_score.sum(axis=1, keepdims=True) + 1e-12)

# ===============================
# 7. METRİKLER
# ===============================
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

clf_report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)
clf_report_df = pd.DataFrame(clf_report).transpose()

# ===============================
# 8. ROC ve PR (Sınıf bazlı)
# ===============================
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

roc_auc_dict = {}
pr_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[label_encoder.classes_[i]] = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc_dict[label_encoder.classes_[i]] = average_precision_score(y_test_bin[:, i], y_score[:, i])

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
# 9. MICRO / MACRO ROC AUC
# ===============================
# Micro-average
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average
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

roc_macro_micro_df = pd.DataFrame({
    "Type": ["Micro-average", "Macro-average"],
    "ROC_AUC": [roc_auc_micro, roc_auc_macro]
})

# ===============================
# 10. NORMALIZED CONFUSION MATRIX + GÖRSEL
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
# 11. ÖZET TABLOLAR (CSV)
# ===============================
# 11.1 Global/Macro/Weighted summary
summary_rows = []
summary_rows.append(["Accuracy", acc])
summary_rows.append(["MCC", mcc])

for avg in ["macro avg", "weighted avg"]:
    summary_rows.append([f"{avg} Precision", clf_report_df.loc[avg, "precision"]])
    summary_rows.append([f"{avg} Recall", clf_report_df.loc[avg, "recall"]])
    summary_rows.append([f"{avg} F1-score", clf_report_df.loc[avg, "f1-score"]])

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

# 11.2 ROC/PR per class tek tablo
roc_pr_df = pd.DataFrame({
    "Class": label_encoder.classes_,
    "ROC_AUC": [roc_auc_dict[c] for c in label_encoder.classes_],
    "PR_AUC": [pr_auc_dict[c] for c in label_encoder.classes_],
})

# ===============================
# 12. ÇIKTILARI KAYDET
# ===============================
# Console çıktısı
print("\n========== Linear SVM (SGD) Detailed ==========")
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

# Dosyalar
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

# Detaylı summary.txt
with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("Linear SVM (SGDClassifier - hinge) Detailed Summary\n")
    f.write("===============================================\n\n")
    f.write(f"CSV: {csv_path}\n")
    f.write(f"Train size: {len(X_train)}\n")
    f.write(f"Test size: {len(X_test)}\n")
    f.write(f"Features: {X_train.shape[1]}\n")
    f.write(f"Classes: {n_classes}\n\n")

    f.write("Model Params\n")
    f.write("------------\n")
    f.write(f"loss=hinge, penalty=l2, alpha=1e-4, max_iter=20, tol=1e-3\n")
    f.write(f"class_weight=balanced, early_stopping=True, validation_fraction=0.01, n_iter_no_change=3\n\n")

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
print("- confusion_matrix.csv")
print("- confusion_matrix_normalized.csv / confusion_matrix_normalized.png")
print("- classification_report.csv")
print("- summary_metrics.csv")
print("- roc_auc.csv / pr_auc.csv")
print("- roc_pr_per_class.csv")
print("- roc_macro_micro.csv")
print("- ROC_*.png / PR_*.png")
print("- summary.txt")
