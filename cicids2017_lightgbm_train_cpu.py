import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, matthews_corrcoef,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ===============================
# 0. TEST KLASÖRÜ VE RUN NUMARASI
# ===============================
base_dir = r"tests\cicids2017\lightgbm"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)

print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

# ===============================
# 1. VERİYİ YÜKLE
# ===============================
df = pd.read_csv(
    'C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv'
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ===============================
# 2. ÇOK KÜÇÜK SINIFLARI KALDIR
# ===============================
# Her sınıfta en az 50 örnek olmalı
# class_counts = df['Attack Type'].value_counts()
# valid_classes = class_counts[class_counts >= 50].index
# df = df[df['Attack Type'].isin(valid_classes)]

# ===============================
# 3. HEDEF DEĞİŞKENİ ENCODE ET
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

X = df.drop(columns=["Attack Type"])
y = df["Attack Type"]

# Sabit kolonları çıkar
X = X.loc[:, X.nunique() > 1]

# ===============================
# 4. TRAIN / TEST AYIR
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train).astype(np.float32)
# X_test = scaler.transform(X_test).astype(np.float32)

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# ===============================
# 6. LightGBM MODELİ (CPU)
# ===============================
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(label_encoder.classes_),
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    device='cpu',  # GPU yerine CPU
    min_child_samples=1,  # split hatasını önlemek için
    min_split_gain=0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

# ===============================
# 7. DEĞERLENDİRME METRİKLERİ
# ===============================
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
clf_report_df = pd.DataFrame(clf_report).transpose()

# ===============================
# 8. ROC ve Precision-Recall
# ===============================
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
roc_auc_dict = {}
pr_auc_dict = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[label_encoder.classes_[i]] = roc_auc

    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    pr_auc = average_precision_score(y_test_bin[:, i], y_score[:, i])
    pr_auc_dict[label_encoder.classes_[i]] = pr_auc

    # ROC grafiğini kaydet
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(run_dir, f'ROC_{label_encoder.classes_[i]}.png'))
    plt.close()

    # PR grafiğini kaydet
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {label_encoder.classes_[i]}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(run_dir, f'PR_{label_encoder.classes_[i]}.png'))
    plt.close()

# ===============================
# 9. Feature Importance
# ===============================
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# ===============================
# 10. Sonuçları kaydet
# ===============================
pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
    os.path.join(run_dir, "confusion_matrix.csv"))
clf_report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
feature_importance.head(10).to_csv(os.path.join(run_dir, "top10_features.csv"))
pd.DataFrame.from_dict(roc_auc_dict, orient='index', columns=["ROC AUC"]).to_csv(os.path.join(run_dir, "roc_auc.csv"))
pd.DataFrame.from_dict(pr_auc_dict, orient='index', columns=["PR AUC"]).to_csv(os.path.join(run_dir, "pr_auc.csv"))

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"Matthews Correlation Coefficient: {mcc:.6f}\n")
    f.write("\nTop 10 Feature Importance:\n")
    f.write(feature_importance.head(10).to_string())
    f.write("\n\nROC AUC:\n")
    for cls, score in roc_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")
    f.write("\nPR AUC:\n")
    for cls, score in pr_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")

print(f"Tüm sonuçlar '{run_dir}' klasörüne kaydedildi.")
