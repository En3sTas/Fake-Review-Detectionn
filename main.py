import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Kütüphaneleri
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Görselleştirme Ayarları
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)

# -------------------------------
# 1. MATERIALS & DATA LOADING
# -------------------------------
print("--- 1. LOADING DATA ---")
try:
    df = pd.read_csv("fake reviews dataset.csv", encoding="utf-8")
    print(f"Dataset loaded successfully. Total samples: {len(df)}")
except FileNotFoundError:
    print("ERROR: 'fake reviews dataset.csv' not found. Please check the file path.")
    exit()

# -------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------
print("\n--- 2. DATA VISUALIZATION (Checking Imbalance) ---")
# Sınıf dağılımını kontrol ediyoruz (Week 2 & 7: Dengesiz veri kontrolü)
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Class Distribution (Fake vs Real)')
plt.show()

# -------------------------------
# 3. PRE-PROCESSING (Basic Cleaning)
# -------------------------------
print("\n--- 3. PRE-PROCESSING ---")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama kaldır
    text = re.sub(r'\d+', '', text)      # Sayıları kaldır
    return text

# Sadece metin temizliği yapıyoruz, Vektörleştirme (TF-IDF) Pipeline içinde olacak.
df['text_cleaned'] = df['text_'].apply(clean_text)

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(df['label'])
classes = list(le.classes_)
print(f"Classes encoded: {classes}")

# -------------------------------
# 4. DATA SPLITTING (Prevention of Leakage)
# -------------------------------
print("\n--- 4. DATA SPLITTING (Week 3 Strategy) ---")

X = df['text_cleaned']  # Ham metin

# Week 3: Stratified Split ile sınıf oranlarını koruyoruz.
# Test setini tamamen ayırıyoruz ("completely held out" [cite: 1026])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

# -------------------------------
# 5. PIPELINE & HYPERPARAMETER TUNING (Week 3 & 4)
# -------------------------------
print("\n--- 5. MODELING WITH PIPELINE & GRID SEARCH ---")

# Pipeline: Önce TF-IDF yap, sonra Lojistik Regresyon uygula.
# Bu yapı sayesinde TF-IDF sadece Train setine "fit" edilir, Test setine sızmaz.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), # Stop words temizliği
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')) # Dengesiz veri önlemi
])

# Grid Search Parametreleri
# Modelin en iyi ayarlarını deneyerek bulacağız.
param_grid = {
    'tfidf__max_features': [3000, 5000],      # Kelime haznesi boyutu
    'tfidf__ngram_range': [(1, 1), (1, 2)],   # Sadece kelimeler mi, kelime grupları mı?
    'clf__C': [0.1, 1, 10]                    # Regularization gücü
}

# Cross-Validation (Week 3 )
# StratifiedKFold kullanarak katmanlarda dengeyi sağlıyoruz.
cv = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1_macro', # Week 7[cite: 424]: Accuracy yerine F1 tercih ediliyor.
    n_jobs=-1,
    verbose=1
)

print("Training started... (This might take a moment)")
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters Found: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

# -------------------------------
# 6. FINAL EVALUATION (Week 7 Metrics)
# -------------------------------
print("\n--- 6. FINAL TEST RESULTS (Week 7) ---")

# En iyi modeli al
best_model = grid_search.best_estimator_

# Test seti üzerinde tahmin (Sadece transform uygulanır, fit yapılmaz)
y_pred = best_model.predict(X_test)

# Detaylı Rapor
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# Confusion Matrix [cite: 385]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n" + "="*40)
print("      LIVE TEST MODE        ")
print("="*40)

while True:
    user_input = input("\nEnter a review (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    # 1. Temizle
    cleaned_input = clean_text(user_input)
    
    # 2. Tahmin Et
    prediction_idx = best_model.predict([cleaned_input])[0]
    
    # 3. Orijinal Etiketi Bul (Örn: 'CG' veya 'OR')
    raw_label = le.inverse_transform([prediction_idx])[0]
    
    # 4. İnsan Tarafından Okunabilir Hale Getir
    # Not: Veri setinde 'CG' = Fake, 'OR' = Real kabul ediyoruz.
    if raw_label == 'CG': 
        result_text = "FAKE (Yapay Zeka)"
    else:
        result_text = "REAL (Gerçek İnsan)"
    
    # Olasılık değeri (Güven oranı)
    proba = best_model.predict_proba([cleaned_input]).max()
    
    print(f"Prediction: {result_text}")
    print(f"Confidence: {proba*100:.2f}%")