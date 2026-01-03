import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import label_binarize

# ===============================
# 0. TEST KLASÖRÜ VE RUN NUMARASI
# ===============================
base_dir = r"tests\cicids2017\xgboost"
os.makedirs(base_dir, exist_ok=True)

# Mevcut run klasörlerini say
existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
run_number = len(existing_runs) + 1
run_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(run_dir, exist_ok=True)

print(f"Çıktılar '{run_dir}' klasörüne kaydedilecek.\n")

# ===============================
# 1. VERİYİ YÜKLE
# ===============================
df = pd.read_csv(
    r'C:\Users\Admin\.cache\kagglehub\datasets\ericanacletoribeiro\cicids2017-cleaned-and-preprocessed\versions\6\cicids2017_cleaned.csv'
)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ===============================
# 2. HEDEF DEĞİŞKENİ ENCODE ET
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

X = df.drop(columns=["Attack Type"])
y = df["Attack Type"]

# ===============================
# 3. TRAIN / TEST AYIR
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. XGBOOST MODELİ
# ===============================
model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

# ===============================
# 6. DEĞERLENDİRME METRİKLERİ
# ===============================
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
clf_report_df = pd.DataFrame(clf_report).transpose()

# ===============================
# 7. EK METRİKLER: ROC AUC
# ===============================
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
roc_auc_dict = {}
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[label_encoder.classes_[i]] = roc_auc

# ===============================
# 8. SONUÇLARI EKRANA YAZDIR
# ===============================
print("\n========== MODEL ÖZETİ ==========")
print(f"Accuracy: {acc:.6f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.6f}")

print("\n--- Confusion Matrix ---")
print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

print("\n--- Classification Report ---")
print(clf_report_df)

print("\n--- ROC AUC (her sınıf için) ---")
for cls, score in roc_auc_dict.items():
    print(f"{cls}: {score:.4f}")

print("\n--- En Önemli 10 Özellik ---")
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance.head(10))

# ===============================
# 9. SONUÇLARI DOSYALARA KAYDET
# ===============================
pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
    os.path.join(run_dir, "confusion_matrix.csv"))
clf_report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
feature_importance.head(10).to_csv(os.path.join(run_dir, "top10_features.csv"))
pd.DataFrame.from_dict(roc_auc_dict, orient='index', columns=["ROC AUC"]).to_csv(os.path.join(run_dir, "roc_auc.csv"))

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"Matthews Correlation Coefficient: {mcc:.6f}\n")
    f.write("\nTop 10 Feature Importance:\n")
    f.write(feature_importance.head(10).to_string())
    f.write("\n\nROC AUC:\n")
    for cls, score in roc_auc_dict.items():
        f.write(f"{cls}: {score:.4f}\n")

print(f"\nTüm sonuçlar '{run_dir}' klasörüne kaydedildi.")
