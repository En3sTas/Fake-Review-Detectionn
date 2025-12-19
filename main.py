import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os # Dosya kaydetmek için

# Scikit-learn Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Imbalanced-learn (Template 4.2: Data Augmentation for Text)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Rapor için görsellerin kaydedileceği klasörü oluştur
if not os.path.exists('project_results'):
    os.makedirs('project_results')

# ==========================================
# 1. HELPER FUNCTIONS (Template 4.2: Pre-processing)
# ==========================================
# ==========================================
# 1. HELPER FUNCTIONS (GÜNCELLENMİŞ HALİ)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    
    # [DÜZELTME BURADA] '' yerine ' ' (boşluk) koyduk
    # Böylece "it.I" -> "it I" olur, "iti" oluşmaz.
    text = re.sub(r'[^\w\s]', ' ', text)  
    
    text = re.sub(r'\d+', '', text)
    
    # Fazladan oluşan boşlukları (  ) teke indirip temizleyelim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

def calculate_sentiment_scores(x):
    return x.apply(get_sentiment).values.reshape(-1, 1)

def plot_feature_importance(model, feature_names, model_name):
    """
    Plots and SAVES the feature importance graph.
    """
    importances = []
    
    if hasattr(model, 'coef_'): # Logistic Regression
        importances = model.coef_[0]
    elif hasattr(model, 'feature_importances_'): # Random Forest
        importances = model.feature_importances_
    
    if len(importances) == len(feature_names):
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp['AbsImportance'] = df_imp['Importance'].abs()
        df_top = df_imp.sort_values(by='AbsImportance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=df_top, palette='viridis')
        plt.title(f'Top 15 Most Important Features - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # SAVE FIGURE (Template 6 Requirement)
        filename = f"project_results/feature_importance_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"[INFO] Feature Importance Plot saved to {filename}")
        plt.show()
    else:
        print("Feature mismatch, skipping plot.")

# ==========================================
# 2. DATA LOADING & DESCRIPTION (Template 4.1)
# ==========================================
print("--- 1. DATASET DESCRIPTION (For Template Section 4.1) ---")
try:
    df = pd.read_csv("fake reviews dataset.csv", encoding="utf-8")
    print(f"Total Number of Samples: {len(df)}")
    
    # Class Distribution (Rapora yazman gereken sayılar)
    print("\nClass Distribution:")
    dist = df['label'].value_counts()
    print(dist)
    
    # Görselleştir ve Kaydet
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Class Distribution (Original)')
    plt.savefig('project_results/class_distribution.png')
    plt.show()
    
except FileNotFoundError:
    print("ERROR: 'fake reviews dataset.csv' not found.")
    exit()

# ==========================================
# 3. PRE-PROCESSING (Template 4.2)
# ==========================================
print("\n--- 2. PRE-PROCESSING (For Template Section 4.2) ---")
print("Steps applied: Lowercasing, Regex Cleaning, Label Encoding.")
df['text_cleaned'] = df['text_'].apply(clean_text)

le = LabelEncoder()
y = le.fit_transform(df['label'])
classes = list(le.classes_)
print(f"Classes Encoded: {classes}")

# ==========================================
# 4. DATA SPLITTING (Template 5.1)
# ==========================================
print("\n--- 3. DATA SPLITTING (For Template Section 5.1) ---")
print("Splitting Strategy: 80% Train, 20% Test (Stratified)")
X = df['text_cleaned'] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# ==========================================
# 5. MODELING & COMPARISON (Template 5.2)
# ==========================================
print("\n" + "="*50)
print("--- 4. MACHINE LEARNING MODELS (For Template Section 5.2) ---")
print("Comparison: Logistic Regression vs Random Forest")
print("Validation: 5-Fold Cross Validation (Template Compliance)")
print("Augmentation: SMOTE applied for text data balance")
print("="*50)

combined_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('sentiment', FunctionTransformer(calculate_sentiment_scores, validate=False))
])

model_configs = [
    {
        'name': 'Logistic Regression',
        'pipeline': ImbPipeline([
            ('features', combined_features),
            ('smote', SMOTE(random_state=42)), # DATA AUGMENTATION
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        'params': {'clf__C': [0.1, 1, 10]}
    },
    {
        'name': 'Random Forest',
        'pipeline': ImbPipeline([
            ('features', combined_features),
            ('smote', SMOTE(random_state=42)), # DATA AUGMENTATION
            ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
        ]),
        'params': {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, 20]}
    }
]

best_global_score = 0
best_global_model = None
best_global_name = ""
results_summary = []

for config in model_configs:
    print(f"\n⏳ TRAINING: {config['name']} with 5-Fold CV...")
    
    # CV=5 YAPILDI (Template isteği)
    rs = RandomizedSearchCV(
        config['pipeline'], param_distributions=config['params'],
        n_iter=3, cv=5, scoring='f1_macro', n_jobs=1, random_state=42, verbose=1
    )
    rs.fit(X_train, y_train)
    
    # Evaluation (Template 6)
    y_pred = rs.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ {config['name']} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion Matrix Plot & Save
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {config["name"]}')
    plt.savefig(f'project_results/confusion_matrix_{config["name"].replace(" ", "_")}.png')
    plt.show()
    
    results_summary.append({'Model': config['name'], 'Accuracy': acc})
    
    if acc > best_global_score:
        best_global_score = acc
        best_global_model = rs.best_estimator_
        best_global_name = config['name']

# ==========================================
# 6. FINAL RESULTS SUMMARY (Template 6)
# ==========================================
print("\n" + "*"*40)
print(f"🏆 WINNER MODEL: {best_global_name} (Score: {best_global_score:.4f})")
print("*"*40)

# Accuracy Comparison Plot & Save
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=pd.DataFrame(results_summary), palette='viridis')
plt.title('Model Comparison Results')
plt.ylim(0.5, 1.0)
plt.savefig('project_results/model_comparison.png')
plt.show()

# Feature Importance Plot & Save
print(f"\n--- GENERATING FEATURE IMPORTANCE FOR REPORT ({best_global_name}) ---")
try:
    step_clf = best_global_model.named_steps['clf']
    features_step = best_global_model.named_steps['features']
    tfidf_transformer = features_step.transformer_list[0][1]
    
    feature_names = list(tfidf_transformer.get_feature_names_out())
    feature_names.append("Sentiment_Score")
    
    plot_feature_importance(step_clf, feature_names, best_global_name)

except Exception as e:
    print(f"Could not plot features: {e}")

# ==========================================
# 7. LIVE TEST MODE
# ==========================================
print("\n" + "="*40)
print(f"      LIVE TEST MODE ({best_global_name})        ")
print("="*40)

while True:
    user_input = input("\nEnter a review (or 'q' to quit): ")
    if user_input.lower() == 'q': break
    
    input_series = pd.Series([clean_text(user_input)])
    prediction_idx = best_global_model.predict(input_series)[0]
    raw_label = le.inverse_transform([prediction_idx])[0]
    proba = best_global_model.predict_proba(input_series).max()
    
    label = "FAKE (Computer Generated)" if raw_label == 'CG' else "REAL (Original)"
    print(f"Prediction: {label} (Confidence: {proba*100:.2f}%)")