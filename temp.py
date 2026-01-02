# ===============================
# 1. KÜTÜPHANELER
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# 2. VERİYİ YÜKLE
# ===============================
df = pd.read_csv("C:/Users/Admin/.cache/kagglehub/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/versions/6/cicids2017_cleaned.csv")  # DOSYA ADINI DEĞİŞTİR

print("Veri boyutu:", df.shape)
print(df.head())

# ===============================
# 3. VERİ TEMİZLEME
# ===============================
# Sonsuz ve boş değerleri temizle
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Temizlenmiş veri boyutu:", df.shape)

# ===============================
# 4. HEDEF DEĞİŞKENİ ENCODE ET
# ===============================
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

print("Saldırı sınıfları:")
print(label_encoder.classes_)

# ===============================
# 5. ÖZELLİKLER VE ETİKETLER
# ===============================
X = df.drop(columns=["Attack Type"])
y = df["Attack Type"]

# ===============================
# 6. TRAIN / TEST AYIR
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 7. ÖLÇEKLENDİRME
# ===============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 8. RANDOM FOREST MODELİ
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# 9. TAHMİN
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 10. DEĞERLENDİRME
# ===============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 11. FEATURE IMPORTANCE
# ===============================
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nEn önemli 10 özellik:")
print(feature_importance.head(10))
